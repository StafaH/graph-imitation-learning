import torch
from torch import nn


class GraphModel(nn.Module):

    def __init__(self):
        super(GraphModel, self).__init__()

    def forward(self, source_images, target_images):
        return 1
