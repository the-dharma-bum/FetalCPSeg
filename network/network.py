""" Here is defined  the main Mixed Attention Network class. """

import torch
from torch import nn
from torch.tensor import Tensor
from network.layers import ConvBlock3D, ResBlock, Attention 
from network import functionals as F
from typing import List, Tuple, Union


# +---------------------------------------------------------------------------------------------+ #
# |                                                                                             | #
# |                                             MAIN NETWORK                                    | #
# |                                                                                             | #
# +---------------------------------------------------------------------------------------------+ #

class MixAttNet(nn.Module):

    """ Mixed Attention Network.
        Adapted from https://github.com/wulalago/FetalCPSeg/blob/master/Network/MixAttNet.py 
    """
    
    def __init__(self, in_channels: int=1, attention: bool=True, supervision: bool=True,
                 depth: int=4, activation: nn.Module = nn.PReLU, se: bool=True, dropout: float=0.3):
        """ Init the network to be trained.

        Args:
            in_channels (int, optional): Number of input 3d images channels. Defaults to 1.
            attention (bool, optional): If True, will refine the feature maps with a stagewise
                                        attention mechanism. Defaults to True.
            supervision (bool, optional): If True, will perfom 3d convolution at each stage
                                          before and after attention. The outputs will be used
                                          to compute a weighted loss. Defaults to True.
            depth (int, optional): How many stages to use for the encoder and decoder. That is,
                                   how many Residual Block. Defaults to 4.
            activation (nn.Module, optional): Any pytorch activation function. Defaults to nn.PReLU.
            se (bool, optional): Use Squeeze and Excite or not. If True, add a Squeeze and excite
                                 layer with a reduction of 1 at the end of every encoder and decoder
                                 Residual Blocks. Defaults to True.
            dropout (float, optional): If > 0, add a Dropout Layer of specifed rate at the end of 
                                       every encoder and decoder Residual Blocks. Defaults to 0.3.
        """
        super().__init__()
        dim = [16] + [2 ** (4+i) for i in range(depth)] + [2 ** (4+depth-1)]
        self.use_attention = attention
        self.supervision   = supervision
        self.depth         = depth
        self.init_block    = ConvBlock3D(in_channels, 16)
        self.encoders      = self._init_encoders(depth, dim, activation, se, dropout)
        self.decoders      = self._init_decoders(depth, dim, activation, se, dropout)
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
    def _init_encoders(depth: int, dim: int,
                       activation: nn.Module, se: bool, dropout: float) -> nn.ModuleList:
        strides = [1] + depth * [2]
        encoders = nn.ModuleList()
        for i in range(depth+1):
            encoders.append(ResBlock(dim[i], dim[i+1], activation, se, dropout, stride=strides[i]))
        return encoders
    
    @staticmethod
    def _init_decoders(depth: int, dim: int, 
                       activation: nn.Module, se: bool, dropout: float) -> nn.ModuleList:
        return nn.ModuleList([ResBlock(2*dim[i+1], dim[i], activation, se, dropout) for i in range(depth)])

    @staticmethod
    def _init_down_modules(depth: int, dim: int) -> nn.ModuleList:
        return nn.ModuleList([ConvBlock3D(dim[i] , 16, kernel_size=1) for i in range(depth)])

    def through_encoders(self, x: torch.Tensor) -> List[torch.Tensor]:
        encoders_outputs = [self.encoders[0](x)]
        for i in range(self.depth):
            encoders_outputs.append(self.encoders[i+1](encoders_outputs[i]))
        return encoders_outputs

    def through_decoders(self, encoders_outputs: List[torch.Tensor]) -> List[torch.Tensor]:
        decoders_outputs = [self.decoders[-1](
            F.up_sample_and_concat3d(encoders_outputs[-2], encoders_outputs[-1]))]
        for i in reversed(range(self.depth-1)):
            decoders_outputs.append(self.decoders[i](
                F.up_sample_and_concat3d(encoders_outputs[i], decoders_outputs[-1])))
        decoders_outputs.reverse()
        return decoders_outputs

    def through_down_modules(self, x: torch.Tensor, 
                             decoders_outputs: List[torch.Tensor]) -> List[torch.Tensor]:
        down_outputs= []
        for i in range(self.depth):
            down_outputs.append(F.up_sample3d(self.down[i](decoders_outputs[i]), x))
        return down_outputs

    def through_attention_modules(self, down_outputs: List[torch.Tensor]) -> Tuple[List[torch.Tensor]]:
        attention_outputs, attention_maps = [], []
        for i in range(self.depth):
            mix_output, attention_map = self.attention[i](down_outputs[i])
            attention_outputs.append(mix_output)
            attention_maps.append(attention_map)
        return attention_outputs, attention_maps

    def supervise(self, 
                  down_outputs: List[torch.Tensor], mix_outputs: List[torch.Tensor]) -> List[torch.Tensor]:
        supervised_down_outputs = [self.down_out[i](down_outputs[i]) for i in range(self.depth)]
        supervised_mix_outputs  = [self.mix_out[i](mix_outputs[i])   for i in range(self.depth)]
        supervised_outputs = supervised_mix_outputs + supervised_down_outputs
        return supervised_outputs

    def through_last_block(self, mix_outputs; List[torch.Tensor]) -> torch.Tensor:
        return self.final_conv(self.non_linear(self.last_block(torch.cat(mix_outputs, dim=1))))

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
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
            return (out, *supervised_outputs)
        return out



