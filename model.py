""" Base Model Class: A Lighning Module
    This class implements all the logic code and will be the one to be fit by a Trainer.
"""

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import pytorch_lightning as pl
from typing import Tuple, Dict
from network import MixAttNet


class LightningModel(pl.LightningModule):
    
    """ LightningModule handling everything training related.
    
    Some attributes and methods used aren't explicitely defined here but comes from the
    LightningModule class. Please refer to the Lightning documentation for further details.
    Note that Lighning Callbacks handles tensorboard logging, early stopping, and auto checkpoints
    for this class. Those are gave to a Trainer object. See init_trainer() in main.py.
    """

    def __init__(self, **kwargs) -> None:
        """ Instanciate a Lightning Model. 
        The call to the Lightning method save_hyperparameters() make every hp accessible through
        self.hparams. e.g: self.hparams.lr. It also sends them to TensorBoard.
        See the from_config class method to see them all.
        """
        super().__init__()
        self.save_hyperparameters()
        self.net  = MixAttNet(self.hparams.in_channels, self.hparams.attention)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def configure_optimizers(self) -> Dict:
        """ Instanciate an optimizer and a learning rate scheduler to be used during training.
        Returns:
            Dict: Dict containing the optimizer(s) and learning rate scheduler(s) to be used by
                  a Trainer object using this model. 
                  The 'monitor' key may be used by some schedulers (e.g: ReduceLROnPlateau).                        
        """
        optimizer = Adam(self.net.parameters(),
                         lr           = self.hparams.lr,
                         weight_decay = self.hparams.weight_decay)
        scheduler = MultiStepLR(optimizer,
                                milestones = self.hparams.milestones,
                                gamma      = self.hparams.gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def get_pos_weight(self, targets: torch.Tensor) -> torch.Tensor:
        """ Calculate the positive label ratio in the input batch data
        Will be used to weight the loss accordingly (adress the class imbalance issue).

        Args:
            targets (torch.Tensor): The ground truth segmentation masks.
                                    Shape: (batch_size, channels, depth, width, height)

        Returns:
            torch.Tensor: (Num positive voxels)/(Num all voxels). Shape (1).
        """
        targets = targets.cpu()
        if np.where(targets == 1)[0].shape[0] == 0:
            weight = 1
        else:
            if self.hparams.patch_size is not None:
                image_size = self.hparams.patch_size
            else:
                image_size = self.hparams.target_shape
            num_voxels = self.hparams.batch_size * image_size[0] * image_size[1] * image_size[2]
            weight = num_voxels / np.where(targets == 1)[0].shape[0]
        weight_tensor = torch.FloatTensor([weight])
        weight_tensor = weight_tensor.to(self.device)
        return weight_tensor

    def compute_loss(self, outputs: Tuple[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """ Compute the loss over a whole batch. 
            If deep supervision, the loss will be a weighted sum of losses computed at several
            depths in the network.

        Args:
            outputs (Tuple[torch.Tensor]): Outputs of the network taken at several depths.
                                           Tuple of length 9. Each of those 9 elements is of shape:
                                           (batch_size, channels, depth, width, height).
                                           See the forward method in network/MixAttNet.py.
                                    Shape: (batch_size, channels, depth, width, height)
            targets (torch.Tensor): The ground truth segmentation masks.
                                    Shape: (batch_size, channels, depth, width, height)
        
        Returns:
            torch.Tensor: Batch loss. Shape: (1).
            
        """ 
        loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=self.get_pos_weight(targets))
        loss   = 0.
        coeffs = [1.] + 2*[0.8, 0.7, 0.6, 0.5]
        if self.net.training and self.hparams.deep_supervision:
            for coeff, output in zip(coeffs, outputs):
                loss += coeff * loss_function(output, targets)
        else:
            loss = loss_function(outputs[0], targets)
        return loss

    @staticmethod
    def dice_score(outputs, targets, ratio=0.5):
        outputs = outputs.flatten()
        targets = targets.flatten()
        outputs[outputs > ratio] = np.float32(1)
        outputs[outputs < ratio] = np.float32(0)    
        return float(2 * (targets * outputs).sum())/float(targets.sum() + outputs.sum())     

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        """ Perform the classic training step (infere + compute loss) on a batch.
        Note that the backward pass is handled under the hood by Pytorch Lightning.
        Args:
            batch (torch.Tensor): Tuple of two tensor. 
                                  One given to the network to be segmented. The other being the
                                  ground truth segmentation mask.
                                  Shapes: (batch_size, channels, depth, width, height)
            batch_idx ([type]): Dataset index of the batch. In range (dataset length)/(batch size).
        Returns:
            Dict: Scalars computed in this function. Note that this dict is accesible from 'hooks'
                  methods from Lightning, e.g on_epoch_start, on_epoch_end, etc...
        """
        inputs, targets = batch        
        outputs = self.net(inputs)
        loss = self.compute_loss(outputs, targets)
        outputs_array = outputs[0].cpu().detach().numpy()
        targets_array = targets.cpu().detach().numpy()
        dice = self.dice_score(outputs_array, targets_array)
        self.log('Dice Score/Train', dice)
        self.log('Loss/Train', loss)
        return {'loss': loss, 'dice': dice}
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        """ Perform the classic training step (infere + compute loss) on a batch.
        Args:
            batch (torch.Tensor): Tuple of two tensor. 
                                  One given to the network to be segmented. The other being the
                                  ground truth segmentation mask.
                                  Shapes: (batch_size, channels, depth, width, height)
            batch_idx (int): Dataset index of the batch. In range (dataset length)/(batch size).
        Returns:
            Dict: Scalars computed in this function. Note that this dict is accesible from 'hooks'
                  methods from Lightning, e.g on_epoch_start, on_epoch_end, etc...
        """
        inputs, targets = batch
        outputs = self.net(inputs)
        loss = self.compute_loss(outputs, targets)
        outputs_array = outputs[0].cpu().detach().numpy()
        targets_array = targets.cpu().detach().numpy()
        dice = self.dice_score(outputs_array, targets_array)
        self.log('Loss/Validation', loss)
        self.log('Dice Score/Validation', dice, prog_bar=True)
        return {'val_loss': loss, 'val_dice': dice}

    def test_step(self, batch: torch.Tensor, batch_idx) ->  torch.Tensor:
        """ Not implemented. """

    @classmethod
    def from_config(cls, config):
        return cls(
            in_channels       = config.train.in_channels,
            attention         = config.train.attention,
            deep_supervision  = config.train.deep_supervision,
            lr                = config.train.lr,
            weight_decay      = config.train.weight_decay,
            milestones        = config.train.milestones,
            gamma             = config.train.gamma,
            verbose           = config.train.verbose,
            target_resolution = config.datamodule.target_resolution,
            target_shape      = config.datamodule.target_shape,
            class_indexes     = config.datamodule.class_indexes,
            batch_size        = config.datamodule.train_batch_size,
            patch_size        = config.datamodule.patch_size,
        )