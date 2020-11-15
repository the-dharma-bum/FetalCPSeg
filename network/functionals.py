import torch
from torch.nn import functional as F


def up_sample3d(x, t, mode="trilinear"):
    """ 3D Up sampling of x alike t size. """
    return F.interpolate(x, t.size()[2:], mode=mode, align_corners=False)


def up_sample_and_concat3d(x, y, mode="trilinear"):
    """ Upsample y alike x and concat the result with x. """
    return torch.cat((x, up_sample3d(y, x)), dim=1)  