import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F


class SimpleGCNModel(nn.Module):

    def __init__(self, num_node_features, num_output_channels):
        super(SimpleGCNModel, self).__init__()
        hidden_layers = 64
        self.conv1 = GCNConv(num_node_features, hidden_layers)
        self.conv2 = GCNConv(hidden_layers, hidden_layers)
        self.conv3 = GCNConv(hidden_layers, hidden_layers)
        self.lin = nn.Linear(hidden_layers, num_output_channels)

    def forward(self, x, edge_index, batch):

        # Node embedding
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)

        # Node aggregation
        x = global_mean_pool(x, batch)

        x = self.lin(x)

        return x
