import os
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader
from typing import Tuple, List, Optional, Callable, NewType
from .volumentations import *
from data import NiftiDataset

# Type hint
Transform =  NewType('Transform', Optional[Callable[[np.ndarray], torch.FloatTensor]])


class DataModule(LightningDataModule):

    """ A Lightning Trainer uses a model and a datamodule. Here is defined a datamodule.
        It's basically a wrapper around dataloaders.
    """  
    
    def __init__(self, input_root: str, target_resolution: Tuple[int] = (1.5, 1.5, 8),
                 target_shape: Tuple[int]= None, class_indexes: List[int] = [1, 2, 3, 4],
                 patch_size: Tuple[int] = (128, 128, 26), train_batch_size: int=64,
                 val_batch_size: int=64, num_workers: int=4) -> None:
        """ Instanciate a Datamodule able to return three Pytorch DataLoaders (train/val/test).

        Args:
            input_root (str): Path to the folder containing the images and masks.
            target_resolution (Tuple[int]): The resolution (x, y, z) after resampling, that is
                                            one voxel sizes in mm3. 
            target_shape (Tuple[int]): Shape of one image after resampling. Will reshape by
                                       resampling. If None, resampling will affect the resolution
                                       only and not the shape. Defaults to None.
            classes (List[int]): A vanilla mask has 4 classes annotated as follows:
                                 * Background...: 0
                                 * Liver........: 1
                                 * Right kidney.: 2
                                 * Left  kidney.: 3
                                 * Spleen.......: 4
                                 This parameter gives class indexes to keep for the classification
                                 task. E.g: [1, 2] for segment liver and right kidney only.
                                 Default to [1, 2, 3, 4] (ie all classes).
            patch_size: Tuple[int]: Some transforms will sample 3D patches from a 3D array.
                                    This specifies those patches size.
            train_batch_size (int, optional): Training batch size. Defaults to 64.
            val_batch_size (int, optional): Validation batch size. Defaults to 64.
            num_workers (int, optional): How many subprocesses to use for data loading.
                                         Defaults to 4.
        """
        super().__init__()
        self.input_root        = input_root
        self.target_resolution = target_resolution
        self.target_shape      = target_shape
        self.class_indexes     = class_indexes
        self.train_batch_size  = train_batch_size
        self.val_batch_size    = val_batch_size
        self.num_workers       = num_workers
        self.train_transform, self.test_transform = None, None
        #self.train_transform, self.test_transform = self.init_transforms(patch_size)

    def init_transforms(self, patch_size):
        train_transform = Compose([
                Normalize(range_norm=True, always_apply=True),
                RandomCrop(patch_size, always_apply=True)
            ], p=0.8)
        test_transform = Compose([
                Normalize(range_norm=True, always_apply=True),
                RandomCrop(patch_size, always_apply=True)
            ], p=0.8) 
        return train_transform, test_transform

    def setup(self, stage: str=None) -> None:
        """ Basically nothing more than train/val split.

        Args:
            stage (str, optional): 'fit' or 'test'.
                                   Init two splitted dataset or one full. Defaults to None.
        """
        total_length = len(os.listdir(self.input_root))//2
        train_length, val_length = int(0.8*total_length), int(0.2*total_length)
        if train_length + val_length != total_length: # round error
            val_length += 1
        if stage == 'fit' or stage is None:
            full_set = NiftiDataset(self.input_root, self.target_resolution, self.target_shape,
                                    self.class_indexes, transform=self.train_transform)
            self.train_set, self.val_set = random_split(full_set, [train_length, val_length])
        if stage == 'test' or stage is None:
            self.test_set = NiftiDataset(self.input_root, self.target_resolution,
                                         self.target_shape, self.class_indexes, 
                                         transform=self.test_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, num_workers=self.num_workers,
                          batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, num_workers=self.num_workers, 
                          batch_size=self.val_batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, num_workers=self.num_workers,
                          batch_size=self.val_batch_size, shuffle=False)

    @classmethod
    def from_config(cls, config):
        """ From a DataModule config object (see config.py) instanciate a
            Datamodule object.
        """
        return cls(config.input_root, config.target_resolution, config.target_shape,
                   config.class_indexes,  config.train_batch_size, 
                   config.val_batch_size, config.num_workers)