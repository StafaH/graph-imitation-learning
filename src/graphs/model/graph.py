import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
import torch.nn.functional as F

from model.model_utils import get_activation, init_
sys.path.append("..")
from transporter.model.transporter import FeatureExtractor, KeyNet, RefineNet, Transporter, gaussian_map
from transporter.model.model_utils import spatial_softmax

# -----------------------------------------------------------------------------------
#                   Graph Networks
# -----------------------------------------------------------------------------------

class GATModel(nn.Module):
    """Graph Attention Nework"""

    def __init__(self,
                 input_dim,
                 output_dim,
                 graph_hidden_dims=[],
                 mlp_hidden_dims=[],
                 act="relu",
                 output_act=None,
                 **kwargs):
        super(GATModel, self).__init__()
        
        graph_dims = [input_dim] + graph_hidden_dims
        mlp_dims = mlp_hidden_dims + [output_dim]

        self.graph_conv = nn.ModuleList(
            [GATConv(graph_dims[i], graph_dims[i + 1]) for i in range(len(graph_dims) - 1)])
        
        self.linear = nn.ModuleList(
            [nn.Linear(mlp_dims[i], mlp_dims[i+1]) for i in range(len(mlp_dims) - 1)])

        self.act = get_activation(act)
        self.output_act = get_activation(output_act)

    def forward(self, x, edge_index, batch):
        out = x
        # Node embedding
        for graph_conv in self.graph_conv:
            out = graph_conv(out, edge_index)

        # Node aggregation
        out = global_mean_pool(out, batch)

        # MLP
        for lin in self.linear:
            out = self.act(lin(out))

        return out

class GCNModel(nn.Module):
    """Graph Convolutional Nework"""

    def __init__(self,
                 input_dim,
                 output_dim,
                 graph_hidden_dims=[],
                 mlp_hidden_dims=[],
                 act="relu",
                 output_act=None,
                 **kwargs):
        super(GCNModel, self).__init__()
        
        graph_dims = [input_dim] + graph_hidden_dims
        mlp_dims = mlp_hidden_dims + [output_dim]

        self.graph_conv = nn.ModuleList(
            [GCNConv(graph_dims[i], graph_dims[i + 1], improved=True) for i in range(len(graph_dims) - 1)])
        
        self.linear = nn.ModuleList(
            [nn.Linear(mlp_dims[i], mlp_dims[i+1]) for i in range(len(mlp_dims) - 2)])

        self.final_layer = nn.Linear(mlp_dims[-2], mlp_dims[-1])

        self.act = get_activation(act)
        self.output_act = get_activation(output_act)

    def forward(self, x, edge_index, batch):
        out = x
        # Node embedding
        for graph_conv in self.graph_conv:
            out = self.act(graph_conv(out, edge_index))

        # Dropout
        out = F.dropout(out)

        # Node aggregation
        out = global_max_pool(out, batch)

        # MLP
        for lin in self.linear:
            out = self.act(lin(out))
        
        # Final Layer
        out = self.final_layer(out)

        return out

class GCNTransporterModel(nn.Module):
    """Graph Convolutional Nework with Transporter Keypoint + Feature embedding"""

    def __init__(self, gcn_model, transporter_model):
        super(GCNTransporterModel, self).__init__()
        
        self.gcn_model = gcn_model
        self.transporter_model = transporter_model

    def forward(self, x, edge_index, batch):
        out = x
        


        return out