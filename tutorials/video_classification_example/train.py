# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import argparse
import itertools
import logging
import numpy as np
import os

import pytorch_lightning
from pytorch_lightning.loggers import WandbLogger
import wandb
import pytorchvideo.data
import pytorchvideo.models.resnet
import pytorchvideo.models.x3d
import pytorchvideo.models.slowfast
import pytorchvideo.models.stem
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorchvideo.transforms import (ApplyTransformToKey, Normalize, RandomShortSideScale, RemoveKey, ShortSideScale,
                                     UniformTemporalSubsample, )
from slurm import copy_and_run_with_config
from torch.utils.data import DistributedSampler, SequentialSampler, BatchSampler, Sampler, RandomSampler
from torchaudio.transforms import MelSpectrogram, Resample
from torchvision.transforms import (CenterCrop, Compose, Lambda, RandomCrop,  # RandomHorizontalFlip,
)

from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sn

"""
This video classification example demonstrates how PyTorchVideo models, datasets and
transforms can be used with PyTorch Lightning module. Specifically it shows how a
simple pipeline to train a Resnet on the Kinetics video dataset can be built.

Don't worry if you don't have PyTorch Lightning experience. We'll provide an explanation
of how the PyTorch Lightning module works to accompany the example.

The code can be separated into three main components:
1. VideoClassificationLightningModule (pytorch_lightning.LightningModule), this defines:
    - how the model is constructed,
    - the inner train or validation loop (i.e. computing loss/metrics from a minibatch)
    - optimizer configuration

2. KineticsDataModule (pytorch_lightning.LightningDataModule), this defines:
    - how to fetch/prepare the dataset
    - the train and val dataloaders for the associated dataset

3. pytorch_lightning.Trainer, this is a concrete PyTorch Lightning class that provides
  the training pipeline configuration and a fit(<lightning_module>, <data_module>)
  function to start the training/validation loop.

All three components are combined in the train() function. We'll explain the rest of the
details inline.
"""


class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, args):
        """
        This LightningModule implementation constructs a PyTorchVideo ResNet,
        defines the train and val loss to be trained with (cross_entropy), and
        configures the optimizer.
        """
        self.args = args
        super().__init__()
        self.train_accuracy = pytorch_lightning.metrics.Accuracy()
        self.val_accuracy = pytorch_lightning.metrics.Accuracy()
        #self.save_hyperparameters() #not here, do it for wandb, look at the end of this script more or less.. wandb save hyper

        #############
        # PTV Model #
        #############

        # Here we construct the PyTorchVideo model. For this example we're using a
        # ResNet that works with Kinetics (e.g. 400 num_classes). For your application,
        # this could be changed to any other PyTorchVideo model (e.g. for SlowFast use
        # create_slowfast).
        if self.args.arch == "video_resnet":
            self.batch_key = "video"
            if self.args.selectnet == 'RESNET3D':
                self.model = pytorchvideo.models.resnet.create_resnet(input_channel=3, model_num_class=7)
            elif self.args.selectnet == 'X3D':
                self.model = pytorchvideo.models.x3d.create_x3d(model_num_class=7,
                                                                input_clip_length=self.args.clip_duration,
                                                                input_crop_size=224)
            elif self.args.selectnet == 'SLOWFAST':
                self.model = pytorchvideo.models.slowfast.create_slowfast(model_num_class=7)
            else:
                exit(-33)

        elif self.args.arch == "audio_resnet":
            return (-2)
            self.model = pytorchvideo.models.resnet.create_acoustic_resnet(input_channel=1, model_num_class=400, )
            self.batch_key = "audio"
        else:
            raise Exception("{self.args.arch} not supported")

    def on_train_epoch_start(self):
        """
        For distributed training we need to set the datasets video sampler epoch so
        that shuffling is done correctly
        """
        epoch = self.trainer.current_epoch
        if self.trainer.use_ddp:
            self.trainer.datamodule.train_dataset.dataset.video_sampler.set_epoch(epoch)

    def forward(self, x):
        """
        Forward defines the prediction/inference actions.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        This function is called in the inner loop of the training epoch. It must
        return a loss that is used for loss.backwards() internally. The self.log(...)
        function can be used to log any training metrics.

        PyTorchVideo batches are dictionaries containing each modality or metadata of
        the batch collated video clips. Kinetics contains the following notable keys:
           {
               'video': <video_tensor>,
               'audio': <audio_tensor>,
               'label': <action_label>,
           }

        - "video" is a Tensor of shape (batch, channels, time, height, Width)
        - "audio" is a Tensor of shape (batch, channels, time, 1, frequency)
        - "label" is a Tensor of shape (batch, 1)

        The PyTorchVideo models and transforms expect the same input shapes and
        dictionary structure making this function just a matter of unwrapping the dict and
        feeding it through the model/loss.
        """
        x = batch[self.batch_key]
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, batch["video_label"][0])
        acc = self.train_accuracy(F.softmax(y_hat, dim=-1), batch["video_label"][0])
        self.log("train_loss", loss)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)


        return loss

    def on_validation_start(self):
        self.val_confusionmatrix = []

    def on_validation_end(self) -> None:
        print('-------------------------------------------')
        print('Printing sum of confusion matrices **VAL.**')
        sum = np.full_like(self.val_confusionmatrix[0], 0)
        for i in range(len(self.val_confusionmatrix)):
            sum = sum + self.val_confusionmatrix[i]
        print('\n',sum)
        print('-------------------------------------------')

        # fig = plt.figure()
        # plt.imshow(sum, cmap='greys')
        fig = plt.figure(figsize=(10, 7))
        sn.set(font_scale=1.6)
        sn.heatmap(sum, annot=True, cmap='Greys', linewidths=.01, linecolor='Black', square=True, cbar=False)
        #plt.show()
        #lightning logger - self.logger.experiment.add_figure('epoch_confmat_val', fig, global_step=self.global_step)
        self.logger.experiment.log({"epoch_confmat_val": [wandb.Image(fig, caption="epoch_confmat_val")]})
        plt.close(fig)

    def validation_step(self, batch, batch_idx):
        """
        This function is called in the inner loop of the evaluation cycle. For this
        simple example it's mostly the same as the training loop but with a different
        metric name.
        """
        if batch_idx == 0:
            print('******************>>>>>>>>>>>>>>> labels, first batch: ', batch["video_label"][0])

        x = batch[self.batch_key]
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, batch["video_label"][0])
        acc = self.val_accuracy(F.softmax(y_hat, dim=-1), batch["video_label"][0])
        self.log("val_loss", loss)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # https://torchmetrics.readthedocs.io/en/latest/references/functional.html
        # confmat = ConfusionMatrix(num_classes=7, normalize='none').cuda()
        # print(confmat(F.softmax(y_hat, dim=-1), batch["video_label"][0]))
        # https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#logging-from-a-lightningmodule
        confmat = ConfusionMatrix(num_classes=7, normalize='none').cuda()
        self.val_confusionmatrix.append(confmat(F.softmax(y_hat, dim=-1), batch["video_label"][0]).cpu().numpy())
        #self.log("confusionmatrix", data, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def on_test_start(self):
        print('gigi')
        self.test_confusionmatrix = []

    def on_test_epoch_end(self):
        print('-------------------------------------------')
        print('Printing sum of confusion matrices **TEST**')
        sum = np.full_like(self.test_confusionmatrix[0], 0)
        for i in range(len(self.test_confusionmatrix)):
            sum = sum + self.test_confusionmatrix[i]
        print(sum)
        print('-------------------------------------------')

        fig = plt.figure(figsize=(10, 7))
        sn.set(font_scale=1.6)
        sn.heatmap(sum, annot=True, cmap='Greys', linewidths=.01, linecolor='Black', square=True, cbar=False)
        #plt.show()
        #lightning logger self.logger.experiment.add_figure('epoch_confmat_test', fig, global_step=self.global_step)
        self.logger.experiment.log({"epoch_confmat_test": [wandb.Image(fig, caption="epoch_confmat_test")]})
        plt.close(fig)

    def test_step(self, batch, batch_idx):
        """
        This function is called in the inner loop of the evaluation cycle. For this
        simple example it's mostly the same as the training loop but with a different
        metric name.
        """
        x = batch[self.batch_key]
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, batch["video_label"][0])
        acc = self.val_accuracy(F.softmax(y_hat, dim=-1), batch["video_label"][0])
        self.log("test_loss", loss)
        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        confmat = ConfusionMatrix(num_classes=7, normalize='none').cuda()
        self.test_confusionmatrix.append(confmat(F.softmax(y_hat, dim=-1), batch["video_label"][0]).cpu().numpy())
        #self.log("confusion_matrix", self.confusionmatrix, prog_bar=False, on_epoch=True)
        return loss




    def configure_optimizers(self):
        """
        We use the SGD optimizer with per step cosine annealing scheduler.
        """
        # ORIGINAL: optimizer = torch.optim.SGD(self.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)

        # AUGUSTO: optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_epochs, last_epoch=-1)
        return [optimizer], [scheduler]


class KineticsDataModule(pytorch_lightning.LightningDataModule):
    """
    This LightningDataModule implementation constructs a PyTorchVideo Kinetics dataset for both
    the train and val partitions. It defines each partition's augmentation and
    preprocessing transforms and configures the PyTorch DataLoaders.
    """

    def __init__(self, args):
        self.args = args
        super().__init__()

    def _make_transforms(self, mode: str):
        """
        ##################
        # PTV Transforms #
        ##################

        # Each PyTorchVideo dataset has a "transform" arg. This arg takes a
        # Callable[[Dict], Any], and is used on the output Dict of the dataset to
        # define any application specific processing or augmentation. Transforms can
        # either be implemented by the user application or reused from any library
        # that's domain specific to the modality. E.g. for video we recommend using
        # TorchVision, for audio we recommend TorchAudio.
        #
        # To improve interoperation between domain transform libraries, PyTorchVideo
        # provides a dictionary transform API that provides:
        #   - ApplyTransformToKey(key, transform) - applies a transform to specific modality
        #   - RemoveKey(key) - remove a specific modality from the clip
        #
        # In the case that the recommended libraries don't provide transforms that
        # are common enough for PyTorchVideo use cases, PyTorchVideo will provide them in
        # the same structure as the recommended library. E.g. TorchVision didn't
        # have a RandomShortSideScale video transform so it's been added to PyTorchVideo.
        """
        if self.args.data_type == "video":
            transform = [self._video_transform(mode), RemoveKey("audio"), ]
        elif self.args.data_type == "audio":
            transform = [self._audio_transform(), RemoveKey("video"), ]
        else:
            raise Exception(f"{self.args.data_type} not supported")

        return Compose(transform)

    def _video_transform(self, mode: str):
        """
        This function contains example transforms using both PyTorchVideo and TorchVision
        in the same Callable. For 'train' mode, we use augmentations (prepended with
        'Random'), for 'val' mode we use the respective determinstic function.
        """
        args = self.args
        return ApplyTransformToKey(key="video", transform=Compose(
            [UniformTemporalSubsample(args.video_num_subsampled), Normalize(args.video_means, args.video_stds), ] + (
                [  # AUGUSTO: OPTION 2
                    # ShortSideScale(args.video_min_short_side_scale),
                    # CenterCrop(args.video_crop_size),

                    # AUGUSTO: OPTION 1
                    RandomShortSideScale(min_size=args.video_min_short_side_scale,
                        max_size=args.video_max_short_side_scale, ), RandomCrop(args.video_crop_size),

                    # BUT ALWAYS EXCLUDE
                    # RandomHorizontalFlip(p=args.video_horizontal_flip_p),
                ] if mode == "train" else [ShortSideScale(args.video_min_short_side_scale),
                    CenterCrop(args.video_crop_size), ])), )

    def _audio_transform(self):
        """
        This function contains example transforms using both PyTorchVideo and TorchAudio
        in the same Callable.
        """
        args = self.args
        n_fft = int(float(args.audio_resampled_rate) / 1000 * args.audio_mel_window_size)
        hop_length = int(float(args.audio_resampled_rate) / 1000 * args.audio_mel_step_size)
        eps = 1e-10
        return ApplyTransformToKey(key="audio", transform=Compose(
            [Resample(orig_freq=args.audio_raw_sample_rate, new_freq=args.audio_resampled_rate, ),
                MelSpectrogram(sample_rate=args.audio_resampled_rate, n_fft=n_fft, hop_length=hop_length,
                    n_mels=args.audio_num_mels, center=False, ), Lambda(lambda x: x.clamp(min=eps)), Lambda(torch.log),
                UniformTemporalSubsample(args.audio_mel_num_subsample), Lambda(lambda x: x.transpose(1, 0)),
                # (F, T) -> (T, F)
                Lambda(lambda x: x.view(1, x.size(0), 1, x.size(1))),  # (T, F) -> (1, T, 1, F)
                Normalize((args.audio_logmel_mean,), (args.audio_logmel_std,)), ]), )

    def train_dataloader(self):
        """
        Defines the train DataLoader that the PyTorch Lightning Trainer trains/tests with.
        """
        sampler = DistributedSampler if self.trainer.use_ddp else RandomSampler
        train_transform = self._make_transforms(mode="train")

        if self.args.whichdataset== 'KITTI-360_3D-MASKED':
            self.train_dataset = pytorchvideo.data.Charades(data_path='/media/14TBDISK/ballardini/pytorchvideotest/KITTI-360_3D-MASKED/train/annotations_train.txt',
                                                            clip_sampler=pytorchvideo.data.make_clip_sampler("random",
                                                                                                self.args.clip_duration),
                                                            video_sampler=sampler,
                                                            transform=train_transform,
                                                            video_path_prefix='/media/14TBDISK/ballardini/pytorchvideotest/KITTI-360_3D-MASKED/train',
                                                            frames_per_clip=None
                                                            )
        elif self.args.whichdataset== 'KITTI-360':
            self.train_dataset = pytorchvideo.data.Charades(data_path='/media/14TBDISK/ballardini/pytorchvideotest/KITTI-360/train/annotations_train.txt',
                                                            clip_sampler=pytorchvideo.data.make_clip_sampler("random",
                                                                                                self.args.clip_duration),
                                                            video_sampler=sampler,
                                                            transform=train_transform,
                                                            video_path_prefix='/media/14TBDISK/ballardini/pytorchvideotest/KITTI-360/train',
                                                            frames_per_clip=None
                                                            )
        elif self.args.whichdataset== 'alcala26':
            self.train_dataset = pytorchvideo.data.Charades(data_path='/media/14TBDISK/ballardini/pytorchvideotest/alcala26/train/annotations_train.txt',
                                                            clip_sampler=pytorchvideo.data.make_clip_sampler("random",
                                                                                                self.args.clip_duration),
                                                            video_sampler=sampler,
                                                            transform=train_transform,
                                                            video_path_prefix='/media/14TBDISK/ballardini/pytorchvideotest/alcala26/train',
                                                            frames_per_clip=None
                                                            )
        elif self.args.whichdataset== 'alcala26-15frame':
            self.train_dataset = pytorchvideo.data.Charades(
                data_path='/media/14TBDISK/ballardini/pytorchvideotest/alcala26-15frame/train/annotations_train.txt',
                clip_sampler=pytorchvideo.data.make_clip_sampler("random", self.args.clip_duration), video_sampler=sampler,
                #clip_sampler=pytorchvideo.data.make_clip_sampler("constant_clips_per_video", self.args.clip_duration, 1), video_sampler=sampler,
                transform=train_transform,
                video_path_prefix='/media/14TBDISK/ballardini/pytorchvideotest/alcala26-15frame/train',
                frames_per_clip=None)

        # self.train_dataset = LimitDataset(
        #     pytorchvideo.data.Kinetics(
        #         data_path=os.path.join(self.args.data_path, "train.csv"),
        #         clip_sampler=pytorchvideo.data.make_clip_sampler(
        #             "random", self.args.clip_duration
        #         ),
        #         video_path_prefix=self.args.video_path_prefix,
        #         transform=train_transform,
        #         video_sampler=sampler,
        #     )
        # )
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.args.batch_size,
            num_workers=self.args.workers, pin_memory=True)

    def val_dataloader(self):
        """
        Defines the train DataLoader that the PyTorch Lightning Trainer trains/tests with.
        """
        sampler = DistributedSampler if self.trainer.use_ddp else RandomSampler
        #sampler = DistributedSampler if self.trainer.use_ddp else SequentialSampler
        val_transform = self._make_transforms(mode="val")

        if self.args.whichdataset== 'KITTI-360_3D-MASKED':
            self.val_dataset = pytorchvideo.data.Charades(data_path='/media/14TBDISK/ballardini/pytorchvideotest/KITTI-360_3D-MASKED/validation/annotations_validation.txt',
                                                          clip_sampler=pytorchvideo.data.make_clip_sampler("random",
                                                                                              self.args.clip_duration),
                                                          video_sampler=sampler,
                                                          transform=val_transform,
                                                          video_path_prefix='/media/14TBDISK/ballardini/pytorchvideotest/KITTI-360_3D-MASKED/validation',
                                                          frames_per_clip=None
                                                          )
        elif self.args.whichdataset== 'KITTI-360':
            self.val_dataset = pytorchvideo.data.Charades(data_path='/media/14TBDISK/ballardini/pytorchvideotest/KITTI-360/validation/annotations_validation.txt',
                                                          clip_sampler=pytorchvideo.data.make_clip_sampler("random",
                                                                                              self.args.clip_duration),
                                                          video_sampler=sampler,
                                                          transform=val_transform,
                                                          video_path_prefix='/media/14TBDISK/ballardini/pytorchvideotest/KITTI-360/validation',
                                                          frames_per_clip=None
                                                          )
        elif self.args.whichdataset== 'alcala26':
            self.val_dataset = pytorchvideo.data.Charades(data_path='/media/14TBDISK/ballardini/pytorchvideotest/alcala26/validation/annotations_validation.txt',
                                                          clip_sampler=pytorchvideo.data.make_clip_sampler("random",
                                                                                              self.args.clip_duration),
                                                          video_sampler=sampler,
                                                          transform=val_transform,
                                                          video_path_prefix='/media/14TBDISK/ballardini/pytorchvideotest/alcala26/validation',
                                                          frames_per_clip=None
                                                          )
        elif self.args.whichdataset== 'alcala26-15frame':
            self.val_dataset = pytorchvideo.data.Charades(
                data_path='/media/14TBDISK/ballardini/pytorchvideotest/alcala26-15frame/validation/annotations_validation.txt',
                #clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.args.clip_duration), video_sampler=sampler,
                #clip_sampler=pytorchvideo.data.make_clip_sampler("constant_clips_per_video", self.args.clip_duration, 1),video_sampler=sampler,
                clip_sampler=pytorchvideo.data.make_clip_sampler("random",self.args.clip_duration),video_sampler=sampler,
                transform=val_transform,
                video_path_prefix='/media/14TBDISK/ballardini/pytorchvideotest/alcala26-15frame/validation',
                frames_per_clip=None)

        # AUGUSTO:
        # ???          self.val_dataset = pytorchvideo.data.Charades(data_path='/media/14TBDISK/ballardini/pytorchvideotest/alcala26-15frame/validation/annotations_validation.txt',
        #                                                  video_path_prefix = '/media/14TBDISK/ballardini/pytorchvideotest/alcala26-15frame/validation',

        # self.val_dataset = LimitDataset(
        #     pytorchvideo.data.Kinetics(
        #         data_path=os.path.join(self.args.data_path, "val.csv"),
        #         clip_sampler=pytorchvideo.data.make_clip_sampler(
        #             "uniform", self.args.clip_duration
        #         ),
        #         video_path_prefix=self.args.video_path_prefix,
        #         transform=val_transform,
        #         video_sampler=sampler,
        #     )
        # )
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.args.batch_size,
            num_workers=self.args.workers)

    # AUGUSTO: I THINK I NEVER USED THIS
    def test_dataloader(self):

        #sampler = DistributedSampler if self.trainer.use_ddp else RandomSampler
        sampler = DistributedSampler if self.trainer.use_ddp else SequentialSampler
        val_transform = self._make_transforms(mode="val")

        if self.args.whichdataset== 'KITTI-360_3D-MASKED':
            self.val_dataset = pytorchvideo.data.Charades(
                data_path='/media/14TBDISK/ballardini/pytorchvideotest/KITTI-360_3D-MASKED/test/annotations_test.txt',
                clip_sampler=pytorchvideo.data.make_clip_sampler("random", self.args.clip_duration),
                video_sampler=sampler,
                transform=val_transform,
                video_path_prefix='/media/14TBDISK/ballardini/pytorchvideotest/KITTI-360_3D-MASKED/test',
                frames_per_clip=None)

        elif self.args.whichdataset== 'KITTI-360':
            self.val_dataset = pytorchvideo.data.Charades(
                data_path='/media/14TBDISK/ballardini/pytorchvideotest/KITTI-360/test/annotations_test.txt',
                clip_sampler=pytorchvideo.data.make_clip_sampler("random", self.args.clip_duration),
                video_sampler=sampler,
                transform=val_transform,
                video_path_prefix='/media/14TBDISK/ballardini/pytorchvideotest/KITTI-360/test',
                frames_per_clip=None)

        elif self.args.whichdataset== 'alcala26':
            self.val_dataset = pytorchvideo.data.Charades(
                data_path='/media/14TBDISK/ballardini/pytorchvideotest/alcala26/test/annotations_test.txt',
                clip_sampler=pytorchvideo.data.make_clip_sampler("random", self.args.clip_duration),
                video_sampler=sampler,
                transform=val_transform,
                video_path_prefix='/media/14TBDISK/ballardini/pytorchvideotest/alcala26/test',
                frames_per_clip=None)

        elif self.args.whichdataset== 'alcala26-15frame':
            self.val_dataset = pytorchvideo.data.Charades(
                data_path='/media/14TBDISK/ballardini/pytorchvideotest/alcala26-15frame/test/annotations_test.txt',
                clip_sampler=pytorchvideo.data.make_clip_sampler("random", self.args.clip_duration),
                video_sampler=sampler,
                transform=val_transform,
                video_path_prefix='/media/14TBDISK/ballardini/pytorchvideotest/alcala26-15frame/test',
                frames_per_clip=None)

            # self.test_dataset = pytorchvideo.data.Charades(
            #     data_path='/media/14TBDISK/ballardini/pytorchvideotest/alcala26-15frame/test/annotations_test.txt',
            #     clip_sampler=pytorchvideo.data.make_clip_sampler("random", self.args.clip_duration),
            #     video_sampler=sampler, transform=val_transform,
            #     video_path_prefix='/media/14TBDISK/ballardini/pytorchvideotest/alcala26-15frame/test',
            #     frames_per_clip=None)



        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.args.batch_size,
            num_workers=self.args.workers)


class LimitDataset(torch.utils.data.Dataset):
    """
    To ensure a constant number of samples are retrieved from the dataset we use this
    LimitDataset wrapper. This is necessary because several of the underlying videos
    may be corrupted while fetching or decoding, however, we always want the same
    number of steps per epoch.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(itertools.repeat(iter(dataset), 2))

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos()


def main():
    """
    To train the ResNet with the Kinetics dataset we construct the two modules above,
    and pass them to the fit function of a pytorch_lightning.Trainer.

    This example can be run either locally (with default parameters) or on a Slurm
    cluster. To run on a Slurm cluster provide the --on_cluster argument.
    """
    setup_logger()

    pytorch_lightning.trainer.seed_everything()
    parser = argparse.ArgumentParser()

    #  Cluster parameters.
    parser.add_argument("--on_cluster", action="store_true")
    parser.add_argument("--job_name", default="ptv_video_classification", type=str)
    parser.add_argument("--working_directory", default=".", type=str)
    parser.add_argument("--partition", default="dev", type=str)

    # Model parameters.
    #parser.add_argument("--lr", "--learning-rate", default=0.1, type=float)  # AUGUSTO: ORIGINAL
    #parser.add_argument("--lr", "--learning-rate", default=0.01, type=float) #AUGUSTO: TEST 1
    #parser.add_argument("--lr", "--learning-rate", default=0.0001, type=float) #AUGUSTO: TEST 2
    parser.add_argument("--lr", "--learning-rate", default=0.00001, type=float) #AUGUSTO: TEST 2
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--arch", default="video_resnet", choices=["video_resnet", "audio_resnet"], type=str, )

    # Data parameters.
    parser.add_argument("--data_path", default=None, type=str, required=False)
    parser.add_argument("--video_path_prefix", default="", type=str)
    parser.add_argument("--workers", default=8, type=int)  # AUGUSTO: ORIGINALLY WAS 16 -> 8 -> 1
    parser.add_argument("--batch_size", default=8, type=int)  # AUGUSTO: ORIGINALLY WAS 16

    # this should not be effective anymore as i changed the code to use all the sequence length. was originally used to
    # select the starting position inside a long sequence, to create a short clip.
    parser.add_argument("--clip_duration", default=26, type=float)
    parser.add_argument("--data_type", default="video", choices=["video", "audio"], type=str)
    parser.add_argument("--video_num_subsampled", default=15, type=int)  # AUGUSTO: ORIGINALLY WAS 6
    parser.add_argument("--video_means", default=(0.45, 0.45, 0.45), type=tuple)
    parser.add_argument("--video_stds", default=(0.225, 0.225, 0.225), type=tuple)
    parser.add_argument("--video_crop_size", default=224, type=int)
    parser.add_argument("--video_min_short_side_scale", default=256, type=int)
    parser.add_argument("--video_max_short_side_scale", default=320, type=int)
    parser.add_argument("--video_horizontal_flip_p", default=0.0, type=float)

    parser.add_argument("--whichdataset", type=str, default=None,
                        choices=['KITTI-360_3D-MASKED', 'KITTI-360', 'alcala26', 'alcala26-15frame'])

    parser.add_argument("--selectnet", type=str, default=None,
                        choices=['RESNET3D', 'X3D', 'SLOWFAST'])

    parser.add_argument("--wandb", action="store_true")

    # parser.add_argument("--audio_raw_sample_rate", default=44100, type=int)
    # parser.add_argument("--audio_resampled_rate", default=16000, type=int)
    # parser.add_argument("--audio_mel_window_size", default=32, type=int)
    # parser.add_argument("--audio_mel_step_size", default=16, type=int)
    # parser.add_argument("--audio_num_mels", default=80, type=int)
    # parser.add_argument("--audio_mel_num_subsample", default=128, type=int)
    # parser.add_argument("--audio_logmel_mean", default=-7.03, type=float)
    # parser.add_argument("--audio_logmel_std", default=4.66, type=float)

    wandb_logger = WandbLogger()

    checkpoint_callback = ModelCheckpoint(monitor='val_acc_epoch',
                                          mode='max',
                                          save_top_k=1,
                                          dirpath='/media/14TBDISK/ballardini/pytorchvideotest/checkpoints',
                                          verbose=True,
                                          save_last=False,
                                          filename=wandb_logger.experiment.name+'_{epoch}-{val_acc_epoch:.2f}')

    early_stopping = EarlyStopping(monitor='val_acc_epoch', patience=200, verbose=True, mode='max')

    # Trainer parameters.
    parser = pytorch_lightning.Trainer.add_argparse_args(parser)
    parser.set_defaults(min_epochs=200,
                        max_epochs=1000,  # AUGUSTO NUMBER OF EPOCHS
                        callbacks=[LearningRateMonitor(), checkpoint_callback, early_stopping],
                        replace_sampler_ddp=False,
                        reload_dataloaders_every_epoch=False,
                        check_val_every_n_epoch=1,
                        whichdataset='alcala26-15frame',
                        selectnet='RESNET3D')

    # Build trainer, ResNet lightning-module and Kinetics data-module.
    args = parser.parse_args()

    # if args.on_cluster:
    #     copy_and_run_with_config(train, args, args.working_directory, job_name=args.job_name, time="72:00:00",
    #         partition=args.partition, gpus_per_node=args.gpus, ntasks_per_node=args.gpus, cpus_per_task=10, mem="470GB",
    #         nodes=args.num_nodes, constraint="volta32gb", )
    # else:  # local
    #     train(args)

    wandb_logger.log_hyperparams(args)
    trainer = pytorch_lightning.Trainer.from_argparse_args(args)
    trainer.logger = wandb_logger
    classification_module = VideoClassificationLightningModule(args)
    data_module = KineticsDataModule(args)
    trainer.fit(classification_module, data_module)
    trainer.test()


def train(args):
    trainer = pytorch_lightning.Trainer.from_argparse_args(args)
    wandb_logger = WandbLogger()
    wandb_logger.log_hyperparams(args)
    trainer.logger = wandb_logger
    classification_module = VideoClassificationLightningModule(args)
    data_module = KineticsDataModule(args)
    trainer.fit(classification_module, data_module)
    trainer.test()


def setup_logger():
    ch = logging.StreamHandler()
    formatter = logging.Formatter("\n%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch.setFormatter(formatter)
    logger = logging.getLogger("pytorchvideo")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)


if __name__ == "__main__":
    main()
