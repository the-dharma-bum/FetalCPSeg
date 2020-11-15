import torch
from torch import nn
import functionals as F
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
# |                                   SQUEEZE AND EXCITE BLOCK                                  | #
# |                                                                                             | #
# +---------------------------------------------------------------------------------------------+ #


class SELayer(nn.Module):
    
    def __init__(self, channel: int, reduction: int=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc       = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# +---------------------------------------------------------------------------------------------+ #
# |                                                                                             | #
# |                                   SQUEEZE AND EXCITE BLOCK                                  | #
# |                                                                                             | #
# +---------------------------------------------------------------------------------------------+ #

class ResBlock(nn.Module):
    
    """ A 3D Residual Block. 

        Normal path..: Conv3d -> BatchNorm3d -> NonLinear -> Conv3d -> BatchNorm3d
        Shortcut.....: DownSampling (ie = Conv3d with kernel size = 1) -> BatchNorm3d
        Output.......: NonLinear(Normal Path + Shortcut)  
    """

    def __init__(self, in_chan: int, out_chan: int, activation: nn.Module, stride: int=1):
        super().__init__()
        self.conv1       = ConvBlock3D(in_chan, out_chan, stride=stride)
        self.conv2       = ConvBlock3D(out_chan, out_chan, bias=True)
        self.non_linear  = activation()
        self.down_sample = ConvBlock3D(in_chan, out_chan, kernel_size=1, padding=0, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out      = self.conv2(self.non_linear(self.conv1(x)))
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




# +---------------------------------------------------------------------------------------------+ #
# |                                                                                             | #
# |                                             MAIN NETWORK                                    | #
# |                                                                                             | #
# +---------------------------------------------------------------------------------------------+ #

class MixAttNet(nn.Module):
    
    def __init__(self, in_channels: int=1, attention: bool=True, supervision: bool=True,
                 depth: int=4, activation: nn.Module = nn.PReLU) -> None:
        super().__init__()
        dim = [16] + [2 ** (4+i) for i in range(depth)] + [2 ** (4+depth-1)]
        self.use_attention = attention
        self.supervision   = supervision
        self.depth         = depth
        self.init_block    = ConvBlock3D(in_channels, 16)
        self.encoders      = self._init_encoders(depth, dim, activation)
        self.decoders      = self._init_decoders(depth, dim, activation)
        self.down          = self._init_down_modules(depth, dim) 
        if self.use_attention:
            self.attention = nn.ModuleList(depth * [Attention(16, 16, activation)])
        if self.supervision:
            self.down_out  = nn.ModuleList(depth * [nn.Conv3d(16, 1, kernel_size=1)])
            self.mix_out   = nn.ModuleList(depth * [nn.Conv3d(16, 1, kernel_size=1)])
        self.last_block    = ConvBlock3D(depth * 16, 64, bias=True)
        self.final_conv    = nn.Conv3d(64, 1, kernel_size=1)
        self.non_linear    = activation()

    @staticmethod
    def _init_encoders(depth, dim, activation):
        strides = [1] + depth * [2]
        encoders = nn.ModuleList()
        for i in range(depth+1):
            encoders.append(ResBlock(dim[i], dim[i+1], activation, stride=strides[i]))
        return encoders
    
    @staticmethod
    def _init_decoders(depth, dim, activation):
        return nn.ModuleList([ResBlock(2*dim[i+1], dim[i], activation) for i in range(depth)])

    @staticmethod
    def _init_down_modules(depth, dim):
        return nn.ModuleList([ConvBlock3D(dim[i] , 16, kernel_size=1) for i in range(depth)])

    def through_encoders(self, x):
        encoders_outputs = [self.encoders[0](x)]
        for i in range(self.depth):
            encoders_outputs.append(self.encoders[i+1](encoders_outputs[i]))
        return encoders_outputs

    def through_decoders(self, encoders_outputs):
        decoders_outputs = [self.decoders[-1](
            F.up_sample_and_concat3d(encoders_outputs[-2], encoders_outputs[-1]))]
        for i in reversed(range(self.depth-1)):
            decoders_outputs.append(self.decoders[i](
                F.up_sample_and_concat3d(encoders_outputs[i], decoders_outputs[-1])))
        decoders_outputs.reverse()
        return decoders_outputs

    def through_down_modules(self, x, decoders_outputs):
        down_outputs= []
        for i in range(self.depth):
            down_outputs.append(F.up_sample3d(self.down[i](decoders_outputs[i]), x))
        return down_outputs

    def through_attention_modules(self, down_outputs):
        attention_outputs, attention_maps = [], []
        for i in range(self.depth):
            mix_output, attention_map = self.attention[i](down_outputs[i])
            attention_outputs.append(mix_output)
            attention_maps.append(attention_map)
        return attention_outputs, attention_maps

    def supervise(self, down_outputs, mix_outputs):
        supervised_down_outputs = [self.down_out[i](down_outputs[i]) for i in range(self.depth)]
        supervised_mix_outputs  = [self.mix_out[i](mix_outputs[i])   for i in range(self.depth)]
        supervised_outputs = supervised_mix_outputs + supervised_down_outputs
        return supervised_outputs

    def through_last_block(self, mix_outputs):
        return self.final_conv(self.non_linear(self.last_block(torch.cat(mix_outputs, dim=1))))

    def forward(self, x):
        x = self.non_linear(self.init_block(x))
        encoders_outputs = self.through_encoders(x)
        decoders_outputs = self.through_decoders(encoders_outputs)
        down_outputs     = self.through_down_modules(x, decoders_outputs)
        mix_outputs = down_outputs
        if self.attention:
            mix_outputs, attention_maps = self.through_attention_modules(down_outputs)
        out = self.through_last_block(mix_outputs)
        if self.supervision:
            supervised_outputs = self.supervise(down_outputs, mix_outputs)
            return out, *supervised_outputs
        return out



