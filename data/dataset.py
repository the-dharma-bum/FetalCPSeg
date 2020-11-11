import os
import numpy as np
import nibabel as nib
from nibabel.affines import rescale_affine
from nilearn.image import resample_img
import torch
from torch.utils.data.dataset import Dataset
from einops import rearrange
from typing import Tuple, List, Optional, Callable, NewType


# Type hint
Transform = NewType('Transform', Optional[Callable[[np.ndarray], np.ndarray]])


class NiftiDataset(Dataset):

    """ A Pytorch Dataset to load nifti couples (image, mask). """

    def __init__(self, rootdir: str, target_resolution: Tuple[int] = (1.5, 1.5, 8),
                 target_shape: Tuple[int]= None, class_indexes: List[int] = [1, 2, 3, 4],
                 transform: Transform=None) -> None:
        """ Instanciate a dataset able to apply 3d resampling and transforms to 
            couple (image, mask).

        Args:
            rootdir (str): Path to the folder containing the images and masks.
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
            transform (Transform, optional): Transformations to apply on couple (image, mask).
                                             Based on volumentations. Defaults to None.
                                             See https://github.com/ashawkey/volumentations.
        """
        super().__init__()
        self.rootdir           = rootdir
        self.image_list        = sorted(filter(lambda x: 'src'  in x, os.listdir(self.rootdir)))
        self.mask_list         = sorted(filter(lambda x: 'mask' in x, os.listdir(self.rootdir)))
        self.target_resolution = target_resolution
        self.target_shape      = np.array(target_shape)
        self.class_indexes     = class_indexes
        self.transform         = transform

    def resample(self, image: nib.nifti1.Nifti1Image) -> nib.nifti1.Nifti1Image:
        """ Resample a nifti image, ie changes its affine matrix and its shape
            so that it matches self.target_resolution and self.target_shape
            while preserving the image orientation and origin.

        Args:
            image (nib.nifti1.Nifti1Image): A 3D Nifti1 image.

        Returns:
            [nib.nifti1.Nifti1Image]: A resampled 3D Nifti1 image of resolution 
                                      self.target_resolution and of shape self.target_shape.
        """
        if not self.target_shape:
            self.target_shape = image.shape
        target_affine = rescale_affine(image.affine, image.shape, 
                                       self.target_resolution, new_shape=self.target_shape)
        return resample_img(image, target_affine=target_affine,
                            target_shape=self.target_shape, interpolation='nearest')

    def select_classes(self, mask: np.ndarray) -> None:
        """ A vanilla mask has 4 classes annotated as follows:
            * Background...: 0
            * Liver........: 1
            * Right kidney.: 2
            * Left  kidney.: 3
            * Spleen.......: 4

        Note that this function doesn't return anything but instead acts on the mask directly.
        That is because numpy's ndarrays are mutable objects and create a copy would be time and
        space consuming while being unnecessary.

        Args:
            mask (np.ndarray): A 3D array acting of values in {0, 1, 2, 3, 4} acting as a
                               segmentation mask. 
        """
        all_classes = [1,2,3,4]
        classes_to_delete = list(filter(lambda x: x not in self.class_indexes, all_classes))
        for class_index in classes_to_delete:
            mask[mask == class_index] = 0

    def apply_transform(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray]:
        """ Apply a composition of 3D transformations to both image and mask arrays.
            See https://github.com/ashawkey/volumentations and datamodule.py.

        Args:
            image (np.ndarray): A 3D numpy array of shape (W, H, D).
            mask  (np.ndarray): A 3D numpy array of shape (W, H, D).

        Returns:
            Tuple[np.ndarray]: Two 3D numpy arrays of same shape, which could be different from 
                               input shapes, for instance if padding and/or cropping is applied.
        """
        data = {'image': image, 'mask': mask}
        transformed_data = self.transform(**data)
        return transformed_data['image'], transformed_data['mask']

    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """ Load, apply transforms and return a couple (image, mask).

        Args:
            index (int): The dataset index (one index for one couple (image, mask)).

        Returns:
            Tuple[np.ndarray, np.ndarray]: An image and a mask.
        """
        image = nib.load(os.path.join(self.rootdir, self.image_list[index]))
        mask  = nib.load(os.path.join(self.rootdir,  self.mask_list[index]))
        image, mask = self.resample(image), self.resample(mask)
        image_array, mask_array = image.get_fdata(), mask.get_fdata()
        self.select_classes(mask_array)
        if self.transform is not None:
            image_array, mask_array = self.apply_transform(image_array, mask_array)
        image_tensor = torch.from_numpy(image_array)
        mask_tensor  = torch.from_numpy(mask_array)
        # Permute from (width, height, depth) to (depth, widht, height) and add channel dim.
        image_tensor = rearrange(image_tensor, 'w h (d n) -> n d w h', n=1)
        mask_tensor  = rearrange( mask_tensor, 'w h (d n) -> n d w h', n=1)
        return image_tensor, mask_tensor

    def __len__(self) -> int:
        return len(self.image_list)