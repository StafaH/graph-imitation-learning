import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name):
    return getattr(F, name) if name else lambda x: x


def init_(module):
    # could have different gains with different activations
    nn.init.orthogonal_(module.weight.data, gain=1)
    nn.init.constant_(module.bias.data, 0)
    return module