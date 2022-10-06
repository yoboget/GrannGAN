#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:03:10 2022

@author: yoann
"""


import torch
import torch.nn as nn
import torch.optim as optim
from mol_utils.dataset import MyMolDataset
from utils.transforms import MyFingerprintTransform
from nets.generators import Gen_edge4, Gen_node3
from nets.discriminators import Disc_edge5, Disc_node3
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import os
from trainer import Trainer
from utils.func import JSD_distance

data_path = '/home/yoann/datasets/Fingerprint/'
dataset = TUDataset(data_path, 'Fingerprint',
                        pre_transform= MyFingerprintTransform(),
                        use_node_attr=True,
                        use_edge_attr=True)

torch.manual_seed(42)
perm = torch.randperm(len(dataset))
train_idx = perm[500:]
test_idx = perm[:500]
train_loader = DataLoader(dataset[train_idx], batch_size=500, 
                          shuffle=True, drop_last=True, pin_memory=True)

batch1 = next(iter(train_loader))
batch2 = next(iter(train_loader))

delta_real1 = batch1.x[batch1.edge_index[0]]-batch1.x[batch1.edge_index[1]]  
dist_real1 = (delta_real1[:,0]**2 + delta_real1[:, 1]**2).sqrt()

delta_real2 = batch2.x[batch2.edge_index[0]]-batch2.x[batch2.edge_index[1]]  
dist_real2 = (delta_real2[:,0]**2 + delta_real2[:, 1]**2).sqrt()

jsd_d, _, _ = JSD_distance(dist_real1, dist_real2, batch1.edge_index)
jsd_x1, x_real, x_g = JSD_distance(batch1.x[:,0], batch2.x[:,0], batch1.edge_index)
jsd_x2, y_real, y_g = JSD_distance(batch1.x[:,1], batch2.x[:,1], batch1.edge_index)
jsd_e1, x_real, x_g = JSD_distance(batch1.edge_attr[:,0], batch2.edge_attr[:,0], batch1.edge_index)
jsd_e2, y_real, y_g = JSD_distance(batch1.edge_attr[:,1], batch2.edge_attr[:,1], batch1.edge_index)
print(jsd_d, jsd_x1, jsd_x2, jsd_e1, jsd_e2)