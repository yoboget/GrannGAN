#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 12:40:53 2022

@author: yo1
"""

import torch
from mol_utils.dataset import MyMolDataset
from torch_geometric.loader import DataLoader
from mol_utils.rdkitfuncs import MolConstructor
from torch_geometric.utils import to_dense_adj, to_dense_batch
from rdkit.Chem.Draw import MolToImage
import matplotlib.pyplot as plt

data_path = '/home/yoann/datasets/'
dataset = 'zinc'

if dataset == 'qm9':
    data_path = data_path + 'qm9/'
elif dataset == 'zinc':
    data_path = data_path + 'zinc/'

dataset = MyMolDataset(dataset, data_path, kekulize=False, countH=False,
                 formal_charge=False, exclude_arom=False, exclude_charged=True)
meta = dataset.meta
dataset = DataLoader(dataset, batch_size=10, 
                     shuffle=True, drop_last=False, pin_memory=False)
dataset = next(iter(dataset))

adj = to_dense_adj(edge_index = dataset.edge_index, 
                                   batch = dataset.batch, 
                                   edge_attr = dataset.edge_attr)
adj = adj.permute(0,3,1,2)           
no_bond = adj.sum(1, keepdim = True) *(-1) + 1
adj = torch.cat((adj, no_bond), dim=1)

ann = to_dense_batch(dataset.x, dataset.batch)[0] 

dataset = MolConstructor(ann.type(torch.long),
                          adj.type(torch.long), meta)

def plot_mols(dataset, n_mols = 1, random = True):
    dataset.create_rdkit()
    for i in range(10):
        image = MolToImage(dataset.mols[i], size=(300, 300), kekulize=False, 
                               wedgeBonds=True, fitImage=False, 
                               options=None, canvas=None)
        image.save(f'./img/zinc_{i}.png')
    
plot_mols(dataset)