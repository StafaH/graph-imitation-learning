import os
import numpy as np

import torch
from torch_geometric.data import InMemoryDataset


def ProcessStateToGraphData(file):
    data = np.load(file, allow_pickle=True)
    return data

