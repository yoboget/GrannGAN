# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:47:06 2020

@author: yboge
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as sn

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


class gnn_nodes(nn.Module): 
    def __init__(self,
                 n_nodefeat_in,
                 n_nodefeat_out,
                 n_feat_out,
                 ):
        super().__init__()
        self.edge_linear = nn.Linear(2*n_nodefeat_in, n_feat_out)
        self.node_linear= nn.Linear(n_feat_out, n_feat_out)
        self.node_linear2= mlp(n_nodefeat_in, n_feat_out, 2*[n_feat_out])
            
    def forward(self, nodes, skeleton):
        edges = nodes2edges(nodes)
        edges = self.edge_linear(edges)
        edges = edges * skeleton
        nodes_new = torch.sum(edges, axis= 1)
        nodes_new = self.node_linear(nodes_new)
        nodes = self.node_linear2(nodes)+nodes_new
        return nodes, edges

class gnn(nn.Module): 
    def __init__(self,
                 n_nodefeat_in,
                 n_edgefeat_in,
                 n_feat_out,
                 ):
        super().__init__()  
        #self.edge_linear = nn.Linear(2*n_nodefeat_in+n_edgefeat_in, n_feat_out)         
        #self.node_linear= nn.Linear(n_feat_out, n_feat_out)
        self.edge_linear = mlp(2*n_nodefeat_in+n_edgefeat_in, 
                               n_feat_out,
                               2*[2*n_nodefeat_in+n_edgefeat_in])         
        self.node_linear= mlp(n_feat_out, n_feat_out, 2*[n_feat_out])
        self.node_linear2= mlp(n_nodefeat_in, n_feat_out, 2*[n_feat_out])
         
    def forward(self, nodes, edges, skeleton):
        edges_nodes = nodes2edges(nodes)
        edges = torch.cat([edges, edges_nodes], dim=3)
        edges = self.edge_linear(edges)
        edges = edges * skeleton
        nodes_new = torch.sum(edges, axis= 1)
        nodes_new = self.node_linear(nodes_new)
        nodes = self.node_linear2(nodes)+nodes_new
        return nodes, edges

class gnn_no_edge(nn.Module): 
    def __init__(self,
                 n_nodefeat_in,
                 n_edgefeat_in,
                 n_feat_out,
                 ):
        super().__init__()  
        #self.edge_linear = nn.Linear(2*n_nodefeat_in+n_edgefeat_in, n_feat_out)         
        #self.node_linear= nn.Linear(n_feat_out, n_feat_out)
        self.edge_linear = mlp(2*n_nodefeat_in+n_edgefeat_in, 
                               n_feat_out,
                               2*[2*n_nodefeat_in+n_edgefeat_in])         
        self.node_linear= mlp(n_feat_out, n_feat_out, 2*[n_feat_out])
        self.node_linear2= mlp(n_nodefeat_in, n_feat_out, 2*[n_feat_out])
         
    def forward(self, nodes, skeleton):
        edges = nodes2edges(nodes)
        edges = self.edge_linear(edges)
        edges = edges * skeleton
        nodes_new = torch.sum(edges, axis= 1)
        nodes_new = self.node_linear(nodes_new)
        nodes = self.node_linear2(nodes)+nodes_new
        return nodes, edges

class gnn_no_edge_sn(nn.Module): 
    def __init__(self,
                 n_nodefeat_in,
                 n_edgefeat_in,
                 n_feat_out,
                 ):
        super().__init__()  
        #self.edge_linear = nn.Linear(2*n_nodefeat_in+n_edgefeat_in, n_feat_out)         
        #self.node_linear= nn.Linear(n_feat_out, n_feat_out)
        self.edge_linear = mlp_sn(2*n_nodefeat_in+n_edgefeat_in, 
                               n_feat_out,
                               2*[2*n_nodefeat_in+n_edgefeat_in])         
        self.node_linear= mlp_sn(n_feat_out, n_feat_out, 2*[n_feat_out])
        self.node_linear2= mlp_sn(n_nodefeat_in, n_feat_out, 2*[n_feat_out])
         
    def forward(self, nodes, skeleton):
        edges = nodes2edges(nodes)
        edges = self.edge_linear(edges)
        edges = edges * skeleton
        nodes_new = torch.sum(edges, axis= 1)
        nodes_new = self.node_linear(nodes_new)
        nodes = self.node_linear2(nodes)+nodes_new
        return nodes, edges

class gnn_sn(nn.Module): 
    def __init__(self,
                 n_nodefeat_in,
                 n_edgefeat_in,
                 n_feat_out,
                 ):
        super().__init__()  
        #self.edge_linear = sn(nn.Linear(2*n_nodefeat_in+n_edgefeat_in, 
        #                                n_feat_out))       
        #self.node_linear= sn(nn.Linear(n_feat_out, n_feat_out))
        self.edge_linear = mlp_sn(2*n_nodefeat_in+n_edgefeat_in, 
                               n_feat_out,
                               2*[2*n_nodefeat_in+n_edgefeat_in])         
        self.node_linear= mlp_sn(n_feat_out, n_feat_out, 2*[n_feat_out]) 
        self.node_linear2= mlp_sn(n_nodefeat_in, n_feat_out, 2*[n_feat_out]) 
    def forward(self, nodes, edges, skeleton):
        edges_nodes = nodes2edges(nodes)
        edges = torch.cat([edges, edges_nodes], dim=3)
        edges = self.edge_linear(edges)
        edges = edges * skeleton
        nodes_new = torch.sum(edges, axis= 1)
        nodes_new = self.node_linear(nodes_new)
        nodes = self.node_linear2(nodes)+nodes_new
        return nodes, edges

def nodes2edges(nodes):
    nodes_ = nodes.unsqueeze(3).permute(0, 3, 1, 2)
    nodes_ = nodes_.repeat(1, nodes.shape[1], 1, 1)
    nodesT = nodes_.transpose(1, 2)
    return torch.cat([nodes_, nodesT], dim=3)

class transformer_layer(nn.Module): 
    def __init__(self,
                 embed_dim,
                 out_dim,
                 n_nodes,
                 n_multi_head
                 ):
        super().__init__()
        
        d_model = embed_dim
        self.multi_head = nn.MultiheadAttention(d_model, 
                                                n_multi_head, 
                                                vdim=embed_dim,
                                                kdim=embed_dim,
                                                batch_first=True, 
                                                )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.mlp = mlp(2*embed_dim, out_dim, [2*embed_dim])
        
    def forward(self, node_embed):
        node_sublayer = self.multi_head(node_embed, node_embed, node_embed)
        node_embed = self.layer_norm1(node_embed + node_sublayer[0])
        node_mean = node_embed.mean(1, keepdims=True)
        node_mean = node_mean.repeat(1, node_embed.shape[1], 1)
        nodes_concat = torch.cat((node_embed,node_mean), axis=2)
        node_sublayer =  self.mlp(nodes_concat)
        node_embed =self.layer_norm2(node_embed + node_sublayer) 
        return node_embed



def symetric_matrix_product(tensor):
    message = "The 2 last dimentsions of the tensor should have the same shape"
    assert tensor.shape[-2] == tensor.shape[-1], message
    tensorT = tensor.transpose(-2, -1)
    return tensor*tensorT

def symetric_matrix_mean(tensor):
    message = "The 2 last dimentsions of the tensor should have the same shape"
    assert tensor.shape[-2] == tensor.shape[-1], message
    tensorT = tensor.transpose(-2, -1)
    return (tensor+tensorT)/2

def get_inverse_degree_vector(adjacency, leak = 0):
    degree = adjacency.sum([1, 2]).type(torch.long)
    idx_vector_is_0 = (degree == 0)*1.
    degree_inv = degree + idx_vector_is_0
    degree_inv = 1/degree_inv
    degree_inv = degree_inv - idx_vector_is_0*(1-leak)
    return degree_inv