#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 15:59:05 2022

@author: yo1
"""

from typing import Callable, Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch.nn.utils import spectral_norm as sn
from torch_geometric.nn import MessagePassing

class mlp(nn.Module):
    def __init__(self, 
                 in_, 
                 out_,
                 hidden_, 
                 activation = nn.CELU(),
                 ):
        super().__init__()
        n_layers = len(hidden_)-1
        layers = []
        layers.append(nn.Linear(in_, hidden_[0]))
        layers.append(activation)
        for i in range(n_layers):
            layers.append(nn.Linear(hidden_[i], hidden_[i+1]))
            layers.append(activation)
        layers.append(nn.Linear(hidden_[-1], out_))
                
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class mlp_sn(nn.Module):
    def __init__(self, 
                 in_, 
                 out_,
                 hidden_, 
                 activation = nn.CELU(),
                 ):
        super().__init__()
        n_layers = len(hidden_)-1
        layers = []
        layers.append(sn(nn.Linear(in_, hidden_[0])))
        layers.append(activation)
        for i in range(n_layers):
            layers.append(sn(nn.Linear(hidden_[i], hidden_[i+1])))
            layers.append(activation)
        layers.append(sn(nn.Linear(hidden_[-1], out_)))
                
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class GNN(MessagePassing):
    def __init__(self, in_channels_nodes, 
                 in_channels_edges, 
                 out_channels, 
                 spectral_norm = False):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        
        if spectral_norm:
            self.edge_linear = mlp_sn(2*in_channels_nodes+in_channels_edges, 
                                   out_channels,
                                   2*[2*in_channels_nodes+in_channels_edges])         
            self.node_linear= mlp_sn(out_channels, 
                                     out_channels, 
                                     2*[out_channels])
            self.node_linear2 = mlp_sn(in_channels_nodes, 
                                       out_channels, 
                                       2*[out_channels])
        else:
            self.edge_linear = mlp(2*in_channels_nodes+in_channels_edges, 
                                   out_channels,
                                   2*[2*in_channels_nodes+in_channels_edges])         
            self.node_linear = mlp(out_channels, 
                                  out_channels, 
                                  2*[out_channels])
            self.node_linear2 = mlp(in_channels_nodes, 
                                  out_channels, 
                                  2*[out_channels])
        
        
    def forward(self, edge_index, x, edge_attr = None, norm = None):
        x_new = self.propagate(edge_index, x=x, edge_attr = edge_attr, 
                               norm = norm)
        x_new = self.node_linear(x_new)
        return self.node_linear2(x) + x_new, self.edge_attr

    def message(self, x_i, x_j, edge_attr=None, norm = None):
        # x_i and x_j have shape (E, in_channel_nodes)
        # edge_attr has shape (E, in_channel_edges)

        if edge_attr is not None:
            edge_attr = torch.cat((x_i, x_j, edge_attr), dim=1)
        else:
            edge_attr = torch.cat((x_i, x_j), dim=1)
        
        if norm is not None:
            edge_attr = edge_attr * norm.unsqueeze(1)
        edge_attr = self.edge_linear(edge_attr)
        self.edge_attr = edge_attr
        return edge_attr

class GNN2(MessagePassing):
    def __init__(self, in_channels_nodes, 
                 in_channels_edges, 
                 out_channels, 
                 spectral_norm = False):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        
        if spectral_norm:
            self.edge_linear = mlp_sn(2*in_channels_nodes+in_channels_edges, 
                                   out_channels,
                                   2*[2*in_channels_nodes+in_channels_edges])         
            self.node_linear= mlp_sn(out_channels + in_channels_nodes, 
                                     out_channels, 
                                     2*[out_channels + in_channels_nodes])
            
        else:
            self.edge_linear = mlp(2*in_channels_nodes+in_channels_edges, 
                                   out_channels,
                                   2*[2*in_channels_nodes+in_channels_edges])         
            
            self.node_linear = mlp(out_channels + in_channels_nodes, 
                                  out_channels, 
                                  2*[out_channels + in_channels_nodes])
        
        
    def forward(self, edge_index, x, edge_attr = None, norm = None):
        x_new = self.propagate(edge_index, x=x, edge_attr = edge_attr, 
                               norm = norm)

        x_new = torch.cat((x, x_new), dim = 1)
        x_new = self.node_linear(x_new)
        return x_new, self.edge_attr

    def message(self, x_i, x_j, edge_attr=None, norm = None):
        # x_i and x_j have shape (E, in_channel_nodes)
        # edge_attr has shape (E, in_channel_edges)

        if edge_attr is not None:
            edge_attr = torch.cat((x_i, x_j, edge_attr), dim=1)
        else:
            edge_attr = torch.cat((x_i, x_j), dim=1)
        
        if norm is not None:
            edge_attr = edge_attr * norm.unsqueeze(1)
        edge_attr = self.edge_linear(edge_attr)
        self.edge_attr = edge_attr
        return edge_attr

class GNN3(MessagePassing):
    def __init__(self, in_channels_nodes, 
                 in_channels_edges, 
                 out_channels, 
                 spectral_norm = False):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        
        if spectral_norm:
            self.edge_linear = mlp_sn(in_channels_nodes+in_channels_edges, 
                                   out_channels,
                                   2*[in_channels_nodes+in_channels_edges])         
            self.node_linear= mlp_sn(in_channels_nodes, 
                                     out_channels, 
                                     2*[out_channels + in_channels_nodes])
            
        else:
            self.edge_linear = mlp(in_channels_nodes+in_channels_edges, 
                                   out_channels,
                                   2*[in_channels_nodes+in_channels_edges])         
            
            self.node_linear = mlp(in_channels_nodes, 
                                  out_channels, 
                                  2*[in_channels_nodes])
        
        
    def forward(self, edge_index, x, edge_attr = None, norm = None):
        x_new = self.propagate(edge_index, x=x, edge_attr = edge_attr, 
                               norm = norm)
        x = self.node_linear(x)
        x_new = 0.5*(x_new - x)
        return x_new, self.edge_attr

    def message(self, x_i, x_j, edge_attr=None, norm = None):
        # x_i and x_j have shape (E, in_channel_nodes)
        # edge_attr has shape (E, in_channel_edges)

        if edge_attr is not None:
            edge_attr = torch.cat(( .5*(x_i - x_j), edge_attr), dim=1)
        else:
            edge_attr = .5*(x_i, x_j)
        
        if norm is not None:
            edge_attr = edge_attr * norm.unsqueeze(1)
        edge_attr = self.edge_linear(edge_attr)
        self.edge_attr = edge_attr
        return edge_attr



class GINEConv(MessagePassing):
    '''My own modified version of the GINEConv:
      - I concat the edge and node features instead of summing. If not, edge
      and node features should have the same size. 
    '''
    r"""The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)

    that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
    the aggregation procedure.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GINEConv, self).__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # Node and edge feature dimensionalites need to match.
        if isinstance(edge_index, Tensor):
            assert edge_attr is not None
            assert x[0].size(-1) == edge_attr.size(-1)
        elif isinstance(edge_index, SparseTensor):
            assert x[0].size(-1) == edge_index.size(-1)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return F.relu(self.nn(out))

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return torch.cat((x_j, edge_attr), dim=-1)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)