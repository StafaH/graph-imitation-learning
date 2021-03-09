import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------------
#                   funcs   
# -----------------------------------------------------------------------------------

def get_activation(name):
    return getattr(F, name) if name else lambda x: x


def init_(module):
    # could have different gains with different activations 
    nn.init.orthogonal_(module.weight.data, gain=1)
    nn.init.constant_(module.bias.data, 0)
    return module


# -----------------------------------------------------------------------------------
#                   Vanilla MLP   
# -----------------------------------------------------------------------------------

class MLP(nn.Module):
    """ MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[], act="relu", output_act=None, init_weights=False, **kwargs):
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
            init_func(nn.Linear(dims[i], dims[i+1]))
            for i in range(len(dims)-1)
        ])
        self.act = get_activation(act)
        self.output_act = get_activation(output_act)
                

    def forward(self, x):
        out = x 
        for fc in self.fcs[:-1]:
            out = self.act(fc(out))
        out = self.output_act(self.fcs[-1](out))
        return out


