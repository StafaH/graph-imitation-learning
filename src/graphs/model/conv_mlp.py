import numpy as np

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
        self.base = FeatureExtractor(in_channels)

        self.pooling = nn.AdaptiveAvgPool2d(1)
        # self.pooling = nn.AdaptiveMaxPool2d(1)
        # self.flatten = nn.Flatten()

        mlp_input_dim = 128
        self.head = MLP(mlp_input_dim, output_dim, hidden_dims=hidden_dims)

    def forward(self, x):
        out = self.base(x)
        out = self.pooling(out)
        # out = self.flatten(out)
        out = self.head(out)
        return out
