import torch
from torch import nn
from torch.nn import functional as F
from typing import List


# +---------------------------------------------------------------------------------------------+ #
# |                                                                                             | #
# |                                             BASE BLOCKS                                     | #
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


class ResBlock(nn.Module):
    
    """ A 3D Residual Block. 

        Normal path..: Conv3d -> BatchNorm3d -> NonLinear -> Conv3d -> BatchNorm3d
        Shortcut.....: DownSampling (ie = Conv3d with kernel size = 1) -> BatchNorm3d
        Output.......: NonLinear(Normal Path + Shortcut)  
    """

    def __init__(self, in_chan: int, out_chan: int, stride: int=1):
        super().__init__()
        self.conv1       = ConvBlock3D(in_chan, out_chan, stride=stride)
        self.conv2       = ConvBlock3D(out_chan, out_chan, bias=True)
        self.non_linear  = nn.PReLU()
        self.down_sample = ConvBlock3D(in_chan, out_chan, kernel_size=1, padding=0, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out      = self.conv2(self.non_linear(self.conv1(x)))
        shortcut = self.down_sample(x)
        return self.non_linear(out + shortcut)


def up_sample3d(x, t, mode="trilinear"):
    """ 3D Up Sampling. """
    return F.interpolate(x, t.size()[2:], mode=mode, align_corners=False)




# +---------------------------------------------------------------------------------------------+ #
# |                                                                                             | #
# |                                           FAST MIX BLOCK                                    | #
# |                                                                                             | #
# +---------------------------------------------------------------------------------------------+ #

class FastMixBlock(nn.Module):
    
    def __init__(self, in_chan: int, out_chan: int) -> None:
        super().__init__()
        kernel_size = [1, 3, 5, 7]
        self.num_groups = len(kernel_size)
        self.split_in_channels  = self.split_channels(in_chan, self.num_groups)
        self.split_out_channels = self.split_channels(out_chan, self.num_groups)
        self.non_linear         = nn.PReLU()
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
# |                                           FAST MIX BLOCK                                    | #
# |                                                                                             | #
# +---------------------------------------------------------------------------------------------+ #

class Attention(nn.Module):
    
    def __init__(self, in_chan: int, out_chan: int) -> None:
        super().__init__()
        self.mix1  = FastMixBlock(in_chan, out_chan)
        self.mix2  = FastMixBlock(out_chan, out_chan)
        self.conv1 = nn.Conv3d(out_chan, out_chan, kernel_size=1)
        self.conv2 = nn.Conv3d(out_chan, out_chan, kernel_size=1)
        self.norm1 = nn.BatchNorm3d(out_chan)
        self.norm2 = nn.BatchNorm3d(out_chan)
        self.non_linear = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        mix1     = self.conv1(self.mix1(x))
        mix2     = self.mix2(mix1)
        att_map  = torch.sigmoid(self.conv2(mix2))
        out      = self.norm1(x*att_map) + self.norm2(shortcut)
        return self.non_linear(out), att_map




# +---------------------------------------------------------------------------------------------+ #
# |                                                                                             | #
# |                                             MAIN NETWORK                                    | #
# |                                                                                             | #
# +---------------------------------------------------------------------------------------------+ #

class MixAttNet(nn.Module):
    
    def __init__(self, in_channels: int=1, attention: bool=True, num_blocks: int=4):
        super().__init__()
        self.attention  = attention
        self.num_blocks = num_blocks
        dim = [16] + [2 ** (4+i) for i in range(num_blocks)] + [2 ** (4+num_blocks-1)]
        strides = [1] + num_blocks * [2]
        self.init_block = ConvBlock3D(in_channels, 16)
        self.encoders   = nn.ModuleList(
            [ResBlock(dim[i], dim[i+1], stride=strides[i]) for i in range(num_blocks+1)])
        self.decoders   = nn.ModuleList([ResBlock(2*dim[i+1],  dim[i]) for i in range(num_blocks)])
        self.down       = nn.ModuleList(
            [ConvBlock3D(dim[i] , 16, kernel_size=1) for i in range(num_blocks)])
        self.down_out   = nn.ModuleList(num_blocks * [nn.Conv3d(16, 1, kernel_size=1)])
        if self.attention:
            self.mix    = nn.ModuleList(num_blocks * [Attention(16, 16)])
        self.mix_out    = nn.ModuleList(num_blocks * [nn.Conv3d(16, 1, kernel_size=1)])
        self.last_block = ConvBlock3D(num_blocks * 16, 64, bias=True)
        self.final_conv = nn.Conv3d(64, 1, kernel_size=1)
        self.non_linear = nn.PReLU()

    def forward(self, x):
        x = self.non_linear(self.init_block(x))
        encoders_outputs = [self.encoders[0](x)]
        for i in range(1, self.num_blocks+1):
            encoders_outputs.append(self.encoders[i](encoders_outputs[i-1]))
        decoders_outputs = [self.decoders[-1](
            torch.cat((encoders_outputs[-2], 
                       up_sample3d(encoders_outputs[-1], encoders_outputs[-2])), dim=1))]
        for i in reversed(range(self.num_blocks-1)):
            upsampling_output   = up_sample3d(decoders_outputs[-1], encoders_outputs[i])
            concatenated_output = torch.cat((encoders_outputs[i], upsampling_output), dim=1)
            decoders_outputs.append(self.decoders[i](concatenated_output))
        decoders_outputs.reverse()
        down_outputs  = []
        for i in range(self.num_blocks):
            down_outputs.append(up_sample3d(self.down[i](decoders_outputs[i]), x))
        down_outputs_for_supervision = []
        for i in range(self.num_blocks):
            down_outputs_for_supervision.append(self.down_out[i](down_outputs[i]))
        mix_outputs = []
        if self.attention:
            attention_maps = []
            for i in range(self.num_blocks):
                mix_output, attention_map = self.mix[i](down_outputs[i])
                mix_outputs.append(mix_output)
                attention_maps.append(attention_map)
        else:
            for i in range(self.num_blocks):
                mix_outputs.append(down_outputs[i])
        mix_outputs_for_supervision = [self.mix_out[i](mix_outputs[i]) for i in range(self.num_blocks)]
        out = self.non_linear(self.last_block(torch.cat(mix_outputs, dim=1)))
        out = self.final_conv(out)
        outputs_for_supersion = mix_outputs_for_supervision + down_outputs_for_supervision
        return out, *outputs_for_supersion



