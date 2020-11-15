""" Some operations on tensors to be used in the Mixed Attention Network. s"""

import torch
from torch.nn import functional as F


def up_sample3d(x: torch.Tensor, t: torch.Tensor, mode: str="trilinear") -> torch.Tensor:
    """ 3D Up sampling of x alike t size. 

    Args:
        x (torch.Tensor): The input tensor to be upsampled of shape (B, C, D, H, W).
        t (torch.Tensor): Tensor of shape (B, C, D', H', W') with D' >= D, H' >= H, W' >= W.
        mode (str, optional): Algorithm used for upsampling. Can be one of:
                              'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area'.
                              Defaults to "trilinear".

    Returns:
        torch.Tensor: The upsampled tensor, of shape (B, C, D', H', W').
    """
    return F.interpolate(x, t.size()[2:], mode=mode, align_corners=False)


def up_sample_and_concat3d(x: torch.Tensor, y: torch.Tensor, mode="trilinear") -> torch.Tensor:
    """ Upsample y alike x and concat the result with x. 

    Args:
        x (torch.Tensor): The input tensor to be concatenaded of shape (B, C, D', H', W') with
                           D' >= D, H' >= H, W' >= W.
        y (torch.Tensor): The input tensor to be upsampled of shape (B, C, D, H, W).
        mode (str, optional): [description]. Defaults to "trilinear".

    Returns:
        torch.Tensor: First tensor and upsampled second tensor concatenated on the channels dim.
                      Shape (B, C, D', H', W').
    """
    return torch.cat((x, up_sample3d(y, x, mode=mode)), dim=1)  