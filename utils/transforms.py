#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 14:21:46 2022

@author: yoann
"""
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data
from mol_utils.cycles import get_cycles


class MyFingerprintTransform(BaseTransform):
    r"""Normalizes node features to the interval :math:`(-1, 1)`.
    """
    def __init__(self):
        super().__init__()      
    def __call__(self, data):
        range_ = 225-30 #225 and 30 are resp. the max and min for both node attr. 
        normalized = ((data.x-30)/range_)*2 - 1
        data.x = normalized
        data.edge_attr[:, 1] = data.edge_attr[:, 1]/3.14
        cycles  = torch.zeros(data.x.shape[0], 5)
        if data.edge_index.shape[1]>0:
            adj = to_dense_adj(edge_index = data.edge_index, 
                               max_num_nodes=data.x.shape[0] )
            
            cycles = get_cycles(adj)
            data = Data(data.x, 
                        edge_index=data.edge_index, 
                        edge_attr=data.edge_attr, 
                        cycles = cycles.squeeze())
            '''
            data = Data(data.x, 
                        edge_index=data.edge_index, 
                        edge_attr=data.edge_attr)
            '''
        else: 
            
            data = Data(data.x, 
                        edge_index=data.edge_index, 
                        edge_attr=data.edge_attr, 
                        cycles = torch.zeros(data.x.shape[0], 5))
            '''
            data = Data(data.x, 
                        edge_index=data.edge_index, 
                        edge_attr=data.edge_attr)
            '''
        assert data.x.shape[0]== data.cycles.shape[0], f'{data.x.shape} ,\
            {data.cycles.shape} {data.edge_index.shape}'
        return data