#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 22:14:03 2022

@author: yo1
"""
import torch
import torch.nn as nn
import nets.nn2 as nn_
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import GINEConv, GINConv
pdist = nn.PairwiseDistance(p=2)


class MMD_RBF(nn.Module):
    def __init__(self, node_size, edge_size):
        super().__init__()       
        MLP = nn_.mlp(35, 35, [64], activation = nn.ReLU())
        self.node_emb = nn.Linear(node_size, 35)
        self.edge_emb = nn.Linear(edge_size, 35)
        self.GINE = GINEConv(MLP)
        self.GIN = GINConv(MLP)
        
    def __call__(self, sample_real, sample_gen):
        'Samples is a tuple containing x, edge, index_edge_attr, batch_idx)'
        emb_real = self.get_embeddings(*sample_real)
        emb_gen = self.get_embeddings(*sample_gen)
        return self.mmd_rbf(emb_real, emb_gen)
        
    def RBFkernel(self, x1, x2, sigma):
        'sigma fixed but wrong'
        d = torch.cdist(x1, x2)
        d = -d/(2*sigma*sigma)
        return d.exp()
    
    def mmd(self, X, Y, kernel = RBFkernel, sigma = 1):
        term1 = kernel(X, X, sigma).sum()/X.shape[-1]**2
        term2 = kernel(Y, Y, sigma).sum()/Y.shape[-1]**2
        term3 = 2*kernel(X, Y, sigma).sum()/(X.shape[-1]*Y.shape[-1])
        return term1 + term2 - term3 
    
    def mmd_rbf(self, sample_real, sample_gen):
        return self.mmd(sample_real, sample_gen, kernel = self.RBFkernel, 
                        sigma = 1)
    
    
    def get_embeddings(self, x, edge_index, batch, edge_attr):
        x = self.node_emb(x.float())
        if edge_attr is not None:
            edge_attr = self.edge_emb(edge_attr.float())
            node_feat1 = self.GINE(x = x, edge_index = edge_index,
                              edge_attr = edge_attr)
            node_feat2 = self.GINE(x = node_feat1, edge_index = edge_index,
                              edge_attr = edge_attr)
        else:
            node_feat1 = self.GIN(x = x, edge_index = edge_index)
            node_feat2 = self.GIN(x = node_feat1, edge_index = edge_index)
        node_feat1 = to_dense_batch(node_feat1, batch)[0].sum(1)
        node_feat2 = to_dense_batch(node_feat2, batch)[0].sum(1)
        return torch.cat((node_feat1, node_feat2), dim=-1)
    

    