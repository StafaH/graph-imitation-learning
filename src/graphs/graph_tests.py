#!/usr/bin/env python

"""grpah_tests.py: Testing installation, models, environment etc.

    Arguments:
        Required:

        Optional:

    Usage: grpah_tests.py [-h] -d DATA -l LOG -k KEYPOINTS [-c CHANNELS] [-e EPOCHS] [-n NAME] [-b BATCH]
    Usage:
    Example:
"""


# Imports
import sys
import os
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

'''
These test cases are borrowed from the Pytorch Geometric introduction found at the
following link: https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
'''


class Net(torch.nn.Module):

    def __init__(self, num_node_features, num_classes):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def main():

    print('Testing Basic Graph...')
    nodes = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    graph_data = Data(x=nodes, edge_index=edge_index)
    print(f'Generated graph data: {graph_data}')

    print('Testing graphs to GPU...')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    graph_data.to(device)
    print(f'Graph data was sent to {device}')

    print('Testing graph datasets...')
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    print(f'Enzyme dataset of size {len(dataset)} was loaded')
    print(f'Example graph: {dataset[0]}')

    print('Testing train loop...')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = Net(dataset.num_node_features, dataset.num_classes).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    print('Accuracy: {:.4f}'.format(acc))
    print('Done!')


if __name__ == '__main__':
    main()
