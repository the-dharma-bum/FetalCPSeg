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
    
    def __init__(self, in_channels: int=1, attention: bool=True):
        super().__init__()
        self.attention  = attention
        self.init_block = ConvBlock3D(in_channels, 16)
        num_filters = [16,16,32,64,128,128]
        strides     = [1, 2, 2, 2, 2]
        self.encoders = nn.ModuleList()
        for i in range(len(num_filters)-1):
            self.encoders.append(ResBlock(num_filters[i], num_filters[i+1], stride=strides[i]))
        self.decoders = nn.ModuleList()
        for i in range(len(num_filters)-2):
            self.decoders.append(ResBlock(2*num_filters[i+1],  num_filters[i]))
        self.down = nn.ModuleList()
        for i in range(len(num_filters)-2):
            self.down.append(ConvBlock3D(num_filters[i] , 16, kernel_size=1))
        self.down_out = nn.ModuleList()
        for i in range(len(num_filters)-2):
            self.down_out.append(nn.Conv3d(16, 1, kernel_size=1))
        if self.attention:
            self.mix = nn.ModuleList()
            for i in range(len(num_filters)-2):
                self.mix.append(Attention(16, 16))
        self.mix_out = nn.ModuleList()
        for i in range(len(num_filters)-2):
            self.mix_out.append(nn.Conv3d(16, 1, kernel_size=1))
        self.last_block = ConvBlock3D(16*4, 64, bias=True)
        self.final_conv = nn.Conv3d(64, 1, kernel_size=1)
        self.non_linear = nn.PReLU()

    def forward(self, x):
        x = self.non_linear(self.init_block(x))
        encoder1 = self.encoders[0](x)
        encoder2 = self.encoders[1](encoder1)
        encoder3 = self.encoders[2](encoder2)
        encoder4 = self.encoders[3](encoder3)
        encoder5 = self.encoders[4](encoder4)
        decoder4 = self.decoders[3](torch.cat((encoder4, up_sample3d(encoder5, encoder4)), dim=1))
        decoder3 = self.decoders[2](torch.cat((encoder3, up_sample3d(decoder4, encoder3)), dim=1))
        decoder2 = self.decoders[1](torch.cat((encoder2, up_sample3d(decoder3, encoder2)), dim=1))
        decoder1 = self.decoders[0](torch.cat((encoder1, up_sample3d(decoder2, encoder1)), dim=1))
        down1 = up_sample3d(self.down[0](decoder1), x)
        down2 = up_sample3d(self.down[1](decoder2), x)
        down3 = up_sample3d(self.down[2](decoder3), x)
        down4 = up_sample3d(self.down[3](decoder4), x)
        down_out1 = self.down_out[0](down1)
        down_out2 = self.down_out[1](down2)
        down_out3 = self.down_out[2](down3)
        down_out4 = self.down_out[3](down4)
        print('test')
        if self.attention:
            mix1, att1 = self.mix[0](down1)
            mix2, att2 = self.mix[1](down2)
            mix3, att3 = self.mix[2](down3)
            mix4, att4 = self.mix[3](down4)
        else:
            mix1, mix2, mix3, mix4 = down1, down2, down3, down4
        mix_out1 = self.mix_out[0](mix1)
        mix_out2 = self.mix_out[1](mix2)
        mix_out3 = self.mix_out[2](mix3)
        mix_out4 = self.mix_out[3](mix4)
        out = self.non_linear(self.last_block(torch.cat((mix1, mix2, mix3, mix4), dim=1)))
        out = self.final_conv(out)
        return out, mix_out1, mix_out2, mix_out3, mix_out4, down_out1, down_out2, down_out3, down_out4

