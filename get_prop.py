#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:27:52 2022

@author: yoann
"""


import torch
from mol_utils.dataset import MyMolDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_adj, to_dense_batch
import numpy as np


dataset_version = 'cheat'
dataset = 'zinc'
#dataset = 'qm9'
max_mols =48

data_path = '/home/yoann/Downloads/icml18-jtnn-master/molvae/'
data_path = '/home/yoann/datasets/zinc/'
#data_path = '/home/yoann/datasets/qm9/'
#data_path = '/home/yoann/Documents/mol_models_others/GraphAF/'
N_SAMPLE = 10000

if dataset_version == 'cheat':
    dataset = MyMolDataset(dataset, data_path, kekulize=False, countH=False,
             formal_charge=False, exclude_arom=False, exclude_charged=True)
elif dataset_version == 'arom':
    dataset = MyMolDataset(dataset,data_path, kekulize=False, countH=False,
             formal_charge=False, exclude_arom=False, exclude_charged=True)
elif dataset_version == 'full':
    dataset = MyMolDataset(dataset, data_path, kekulize=False, countH=True,
             formal_charge=True, exclude_arom=False, exclude_charged=True)
if N_SAMPLE is None:
    N_SAMPLE = len(dataset)
print(N_SAMPLE)   
print(dataset.proc_dir)
loader = DataLoader(dataset, batch_size=N_SAMPLE, 
                          shuffle=True, drop_last=True, pin_memory=True)
batch = next(iter(loader))
meta = dataset.meta
print(meta) 
adj = to_dense_adj(edge_index = batch.edge_index, 
                                   batch = batch.batch, 
                                   edge_attr = batch.edge_attr)
adj = adj.permute(0,3,1,2)           
no_bond = adj.sum(1, keepdim = True) *(-1) + 1
adj = torch.cat((adj, no_bond), dim=1)

ann = to_dense_batch(batch.x, batch.batch, 
                            max_num_nodes = max_mols)[0]
no_atom = ann.sum(2, keepdim = True) *(-1) + 1
n_g = ann.sum([1, 2]).numpy()
bins_ = int(n_g.max() - n_g.min())
print(n_g)
count, bins, bar = plt.hist(n_g, bins = max_mols-1, range= (1, max_mols), density=True)
plt.title('# atom by molecules: ZINC')
plt.savefig('./img/distribution_nodes_data.png', format='png')
plt.show()
ann = torch.cat((ann, no_atom), dim=2)
prop_edge = []
print(bins, count)
'''
np.savetxt("bar_zinc_gaf.csv", 
           count,
           delimiter =", ", 
           fmt ='% s')
'''
for edge in range(adj.shape[1]):
    prop_edge.append((adj[:, edge].sum()/adj.sum()).item())
    
prop_node = []  
n_atom = max_mols * N_SAMPLE


for atom in range(ann.shape[2]):        
    prop_node.append((ann[:, :, atom].sum()/n_atom).item() )
   
print(prop_edge, np.asarray(prop_node[:-1])/np.asarray(prop_node[:-1]).sum())
