import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_utils import get_activation, init_

# -----------------------------------------------------------------------------------
#                   Vanilla MLP
# -----------------------------------------------------------------------------------


class MLP(nn.Module):
    """ MLP network (can be used as value or policy)
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=[],
                 act="relu",
                 output_act=None,
                 init_weights=False,
                 use_dropout=False,
                 **kwargs):
        """ multi-layer perception / fully-connected network

        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            hidden_dims (list): hidden layer dimensions
            act (str): hidden layer activation
            output_act (str): output layer activation
        """
        super(MLP, self).__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        init_func = init_ if init_weights else lambda x: x

        self.fcs = nn.ModuleList([
            init_func(nn.Linear(dims[i], dims[i + 1]))
            for i in range(len(dims) - 1)
        ])
        self.act = get_activation(act)
        self.output_act = get_activation(output_act)

        self.use_dropout = use_dropout

    def forward(self, x):
        out = x
        for fc in self.fcs[:-1]:
            out = self.act(fc(out))

        if self.use_dropout:
            out = F.dropout(out)

        out = self.output_act(self.fcs[-1](out))
        return out
