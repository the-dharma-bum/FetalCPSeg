""" This file handles all the hyperparameters of the pipeline.
Those hyperparameters are divided in two dataclasses:
    - DataModule
    - Train
They will be used to instanciate two objects of same names.
Those two objects (DataModule and Model) will be used by a Trainer object
to run the training pipeline.
"""

from torch import nn
from dataclasses import dataclass
from typing import Tuple, Optional


# +------------------------------------------------------------------------------------------+ #
# |                                                                                          | #
# |                                         DATAMODULE                                       | #
# |                                                                                          | #
# +------------------------------------------------------------------------------------------+ #

@dataclass
class DataModule:
    
    """ Preprocessing and data loading config used to instanciate a DataModule object.
    
    Args:

        input_root (str): The path of the folder containing the images and masks.
        target_resolution (Tuple[int]): The resolution (x, y, z) after resampling, that is
                                        one voxel sizes in mm3. 
        target_shape (Tuple[int], optional): Shape of one image after resampling. Will reshape
                                             by resampling. If None, resampling will affect the
                                             resolution only and not the shape.
                                             Defaults to None.
        class_indexes (Tuple[int]): A vanilla mask has 4 classes annotated as follows:
                                    * Background...: 0
                                    * Liver........: 1
                                    * Right kidney.: 2
                                    * Left  kidney.: 3
                                    * Spleen.......: 4
                                   This parameter gives class indexes to keep for the
                                   classification task.
                                   E.g: (1, 2) to segment liver and right kidney only.
        patch_size (Tuple[int], optional) : If not None, random patches will be cropped out from
                                            image.
        train_batch_size (int): Batch size of the training dataloader.
        val_batch_size (int): Batch size of the validation dataloader.
        num_workers (int): Num of threads for the 3 dataloaders (train, val, test).
    """

    input_root: str = "/homes/l17vedre/Bureau/Sanssauvegarde/patnum_data/train/"
    target_resolution:      Tuple[int] = (1.5*2, 1.5*2, 8)
    target_shape: Optional[Tuple[int]] = (128, 128, 26)
    class_indexes:          Tuple[int] = (1, )
    patch_size:   Optional[Tuple[int]] = None   
    train_batch_size:              int = 2
    val_batch_size:                int = 2
    num_workers:                   int = 4




# +------------------------------------------------------------------------------------------+ #
# |                                                                                          | #
# |                                          TRAIN                                           | #
# |                                                                                          | #
# +------------------------------------------------------------------------------------------+ #

@dataclass
class Train:
    
    """ Config related to training, ie optimizer and learning rate scheduler.
    
    Args:

        in_channels (int): Number of channels for input images and masks.
        supervision (bool): If True, the loss will be a weighted sum of losses computed at 
                            several depths in the network.
        attention (bool): Controls the use of Attention Module in the network.
        depth: (int): How many Residual Blocks should the encoder and decoder have ?
        activation (nn.Module): Which non linear layer to use ? Can be any Pytorch activation
                                function.
        se (bool): Use Squeeze and Excite or not. If True, add a Squeeze and excite layer with a
                   reduction of 1 at the end of every encoder and decoder Residual Blocks.
        dropout (float): If > 0, add a Dropout Layer of specifed rate at the end of every
                         encoder and decoder Residual Blocks. 
        lr (float): Initial learning rate.
        weight_decay (float): L2 penalty of model's weights. 
    """

    in_channels:       int = 1
    supervision:      bool = True
    attention:        bool = True
    depth:             int = 4
    activation:  nn.Module = nn.CELU
    se:               bool = True
    dropout:         float = 0.7
    optimizer:         str = 'adam'
    scheduler:         str = 'rop' 
    lr:              float = 1e-3
    weight_decay:    float = 5e-4




# +------------------------------------------------------------------------------------------+ #
# |                                                                                          | #
# |                                          CONFIG                                          | #
# |                                                                                          | #
# +------------------------------------------------------------------------------------------+ #

@dataclass
class Config:

    """ A simple  wrapper around the subconfigurations classes. """

    datamodule: DataModule
    train: Train