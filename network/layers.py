import torch
from torch import nn
from typing import List


# +---------------------------------------------------------------------------------------------+ #
# |                                                                                             | #
# |                                         3D CONVOLUTION                                      | #
# |                                                                                             | #
# +---------------------------------------------------------------------------------------------+ #

class ConvBlock3D(nn.Module):

    """ A simple sequence of Conv3d -> BatchNorm3d """

    def __init__(self, in_chan: int, out_chan: int,
                 kernel_size: int=3, padding: int=1, stride: int=1, bias: bool=False)  -> None:
        super().__init__()
        self.conv3d = nn.Conv3d(in_chan, out_chan,
                                kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        self.bn3d   = nn.BatchNorm3d(out_chan)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn3d(self.conv3d(x))




# +---------------------------------------------------------------------------------------------+ #
# |                                                                                             | #
# |                                        SQUEEZE AND EXCITE                                   | #
# |                                                                                             | #
# +---------------------------------------------------------------------------------------------+ #


class SEBlock3D(nn.Module):
    
    """ 3D Squeeze-and-Excitation (SE) block as described in:
        Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, in_channels, activation, reduction_ratio=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio, bias=True)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels, bias=True)
        self.relu = activation()

    def forward(self, input_tensor):
        batch_size, num_channels, D, H, W = input_tensor.size()
        squeeze_tensor = self.avg_pool(input_tensor)
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = torch.sigmoid(self.fc2(fc_out_1))
        return torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1))



# +---------------------------------------------------------------------------------------------+ #
# |                                                                                             | #
# |                                             RESIDUAL                                        | #
# |                                                                                             | #
# +---------------------------------------------------------------------------------------------+ #

class ResBlock(nn.Module):
    
    """ A 3D Residual Block. 

        Normal path..: Conv3d -> BatchNorm3d -> NonLinear -> Conv3d (-> SE) -> BatchNorm3d
        Shortcut.....: DownSampling (ie = Conv3d with kernel size = 1) -> BatchNorm3d
        Output.......: NonLinear(Normal Path + Shortcut)  
    """

    def __init__(self, in_chan: int, out_chan: int, activation: nn.Module, se: bool,
                 dropout: float, stride: int=1):
        super().__init__()
        self.conv1       = ConvBlock3D(in_chan, out_chan, stride=stride)
        self.conv2       = ConvBlock3D(out_chan, out_chan, bias=True)
        self.non_linear  = activation()
        self.down_sample = ConvBlock3D(in_chan, out_chan, kernel_size=1, padding=0, stride=stride)
        self.use_se, self.dropout_rate = se, dropout
        if self.use_se:
            self.se = SEBlock3D(out_chan, activation)
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out      = self.conv2(self.non_linear(self.conv1(x)))
        if self.use_se: 
            out = self.se(out)
        if self.dropout_rate > 0:
            out = self.dropout(out)
        shortcut = self.down_sample(x)
        return self.non_linear(out + shortcut)




# +---------------------------------------------------------------------------------------------+ #
# |                                                                                             | #
# |                                           FAST MIX BLOCK                                    | #
# |                                                                                             | #
# +---------------------------------------------------------------------------------------------+ #

class FastMixBlock(nn.Module):
    
    def __init__(self, in_chan: int, out_chan: int, activation: nn.Module) -> None:
        super().__init__()
        kernel_size = [1, 3, 5, 7]
        self.num_groups = len(kernel_size)
        self.split_in_channels  = self.split_channels(in_chan, self.num_groups)
        self.split_out_channels = self.split_channels(out_chan, self.num_groups)
        self.non_linear         = activation()
        self.grouped_conv       = nn.ModuleList()
        for i in range(self.num_groups):
            self.grouped_conv.append(
                ConvBlock3D(self.split_in_channels[i], self.split_out_channels[i],
                            kernel_size[i], padding=(kernel_size[i]-1)//2, stride=1, bias=True))

    @staticmethod
    def split_channels(channels: int, num_groups: int) -> List[int]:
        split_channels = [channels//num_groups for _ in range(num_groups)]
        split_channels[0] += channels - sum(split_channels)
        return split_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_groups == 1:
            return self.grouped_conv[0](x)
        x_split = torch.split(x, self.split_in_channels, dim=1)
        x = [self.non_linear(conv(t)) for conv, t in zip(self.grouped_conv, x_split)]
        return torch.cat(x, dim=1)




# +---------------------------------------------------------------------------------------------+ #
# |                                                                                             | #
# |                                      ATTENTION (with MD Conv)                               | #
# |                                                                                             | #
# +---------------------------------------------------------------------------------------------+ #

class Attention(nn.Module):
    
    def __init__(self, in_chan: int, out_chan: int, activation: nn.Module) -> None:
        super().__init__()
        self.mix1  = FastMixBlock(in_chan, out_chan, activation)
        self.mix2  = FastMixBlock(out_chan, out_chan, activation)
        self.conv1 = nn.Conv3d(out_chan, out_chan, kernel_size=1)
        self.conv2 = nn.Conv3d(out_chan, out_chan, kernel_size=1)
        self.norm1 = nn.BatchNorm3d(out_chan)
        self.norm2 = nn.BatchNorm3d(out_chan)
        self.non_linear = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        mix1     = self.conv1(self.mix1(x))
        mix2     = self.mix2(mix1)
        att_map  = torch.sigmoid(self.conv2(mix2))
        out      = self.norm1(x*att_map) + self.norm2(shortcut)
        return self.non_linear(out), att_map