import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mlp import MLP


class ConvBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 stride=1,
                 padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # return self.lrelu(x)
        return torch.relu(x)


class FeatureExtractor(nn.Module):
    """Feature Extractor Phi"""

    def __init__(self, in_channels=3):
        super(FeatureExtractor, self).__init__()
        self.net = nn.Sequential(
            # Input 3 x 128 x 128                                                 Outputs
            ConvBlock(in_channels, 32, kernel_size=(7, 7), stride=1,
                      padding=3),  # 1 32 x 128 x 128
            ConvBlock(32, 32, kernel_size=(3, 3), stride=1),  # 2 32 x 128 x 128
            ConvBlock(32, 64, kernel_size=(3, 3), stride=2),  # 3 64 x 64 x 64
            ConvBlock(64, 64, kernel_size=(3, 3), stride=1),  # 4 64 x 64 x 64
            ConvBlock(64, 128, kernel_size=(3, 3), stride=2),  # 5 128 x 32 x 32
            ConvBlock(128, 128, kernel_size=(3, 3),
                      stride=1),  # 6 128 x 32 x 32
        )

    def forward(self, x):
        """
        Args
        ====
        x: (N, C, H, W) tensor.
        Returns
        =======
        y: (N, C, H, K) tensor.
        """
        return self.net(x)


class ConvMLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims=[], **kwargs):
        super().__init__()

        in_channels = input_dim[0]
        # self.base = FeatureExtractor(in_channels)
        self.base = nn.Sequential(
            # Input 3 x 128 x 128                                                 Outputs
            ConvBlock(in_channels, 32, kernel_size=(7, 7), stride=1,
                      padding=3),  # 1 32 x 128 x 128
            ConvBlock(32, 32, kernel_size=(3, 3), stride=1),  # 2 32 x 128 x 128
            ConvBlock(32, 64, kernel_size=(3, 3), stride=2),  # 3 64 x 64 x 64
        )

        # self.pooling = nn.AdaptiveAvgPool2d(1)
        self.pooling = nn.AdaptiveMaxPool2d(1)
        self.flatten = nn.Flatten()

        mlp_input_dim = 64
        self.head = MLP(mlp_input_dim, output_dim, hidden_dims=hidden_dims)

    def forward(self, x):
        out = self.base(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.head(out)
        return out


def get_conv_output_dim(input_dim, kernel, stride=1, padding=0, dilation=1):
    """Formula to calculate output size after a convolution layer.
    
    Reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    """
    temp = input_dim + 2 * padding - dilation * (kernel - 1) - 1
    output_dim = math.floor(temp / stride + 1)
    return output_dim


def get_padding_size(input_dim, kernel, stride):
    """For zero padding.
    
    Reference: https://github.com/ray-project/ray/blob/5289690d1c0ff0086d30a3e315689a66f29134da/rllib/models/torch/misc.py#L21
    """
    out_dim = math.ceil(float(input_dim) / float(stride))
    pad = int((out_dim - 1) * stride + kernel - input_dim)
    pad_half = pad // 2
    return pad_half


def get_conv_stack_output_dim(inptut_shape, conv_specs):
    """Calculates final output size from stack of conv layers.
    
    Args:
        inptut_shape (list): [channel, input_dim]
        conv_specs (list): [[channel, kernel, stride], [], ...]

    Returns:
        list: output shape
    """
    output_shape = inptut_shape
    for c, k, s in conv_specs:
        in_c, in_dim = output_shape
        # pad = get_padding_size(in_dim, k, s)
        out_dim = get_conv_output_dim(in_dim, k, s)
        output_shape = [c, out_dim]
    return output_shape
