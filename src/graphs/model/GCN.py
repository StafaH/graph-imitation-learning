import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

from model.model_utils import get_activation, init_

# -----------------------------------------------------------------------------------
#                   Graph Nets
# -----------------------------------------------------------------------------------


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


class GCNModel(nn.Module):
    """Just GCN with some customizable things."""

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=[],
                 act="relu",
                 output_act=None,
                 **kwargs):
        super(GCNModel, self).__init__()
        dims = [input_dim] + hidden_dims + [output_dim]

        self.gconvs = nn.ModuleList(
            [GCNConv(dims[i], dims[i + 1]) for i in range(len(dims) - 2)])
        self.output_fc = nn.Linear(dims[-2], dims[-1])

        self.act = get_activation(act)
        self.output_act = get_activation(output_act)

    def forward(self, x, edge_index, batch):
        out = x
        # Node embedding
        for gconv in self.gconvs:
            out = self.act(gconv(out, edge_index))

        # Node aggregation
        out = global_mean_pool(out, batch)

        # final output
        out = self.output_act(self.output_fc(out))
        return out
