#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 16:59:49 2022

@author: yoann
"""




import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset
from utils.transforms import MyFingerprintTransform
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
import wandb
import os
import glob
from utils.func import JSD_distance, normalize_adj, get_symetric_sparse_adj
from nets.generators import Gen_edge4, Gen_node3
import torch.nn as nn
import numpy as np

data_path = '/home/yoann/datasets/'
dataset = 'Fingerprint'

cuda = torch.cuda.is_available()
if cuda:
    device = torch.device("cuda")
    #torch.set_default_tensor_type(torch.cuda.FloatTensor) 
else:
    device = torch.device("cpu")

root = os.getcwd()

edge_generator_name = 'edge_generator_ep189000.pt'
edge_run_path = 'yobo/Fingerprint_arom_adja/258fbll7'

node_generator_name = 'node_generator_ep538000.pt'
node_run_path = 'yobo/Fingerprint_arom_nodes/1yc9hl33'


api = wandb.Api()
node_run = api.run(node_run_path)
args_node = node_run.config
id_ = node_run.id
root = os.getcwd()
folder = glob.glob(f'{root}/wandb_saved/*{id_}')[0] 
node_generator_folder = f'{folder}/files/'


api = wandb.Api()
edge_run = api.run(edge_run_path)
args_edge = edge_run.config
id_ = edge_run.id
root = os.getcwd()
folder = glob.glob(f'{root}/wandb_saved/*{id_}')[0] 
edge_generator_folder = f'{folder}/files/'

dataset = TUDataset(data_path, dataset,
                    pre_transform= MyFingerprintTransform(),
                    use_node_attr=True,
                    use_edge_attr=True)

torch.manual_seed(43)
perm = torch.randperm(len(dataset))
train_idx = perm[500:]
test_idx = perm[:500]
dataloader = DataLoader(dataset[test_idx], batch_size=500, 
                     shuffle=True, drop_last=False, pin_memory=False)

batch = next(iter(dataloader))
batch = batch.to(device)

print(args_edge['noise_dim'])
sizes = {'n_nodes': batch[0].x.shape[0],
         'node_types': batch[0].x.shape[-1],
         'edge_types': batch.edge_attr.shape[-1],
         'batch': 500,
         'noise_edge': args_edge['noise_dim'],
         'noise_node': args_node['noise_dim'],
         'noise': args_node['noise_dim'],
         'cycles': 5}



generator = Gen_node3(
                    [sizes['noise_node']+sizes['cycles']] +\
                    args_node['n_layers_g']*[args_node['layer_size']]+ [sizes['node_types']],
                    [0] + (1+args_node['n_layers_g'])*[args_node['layer_size']],
                    activation=nn.CELU()).to(device)


generator.load_state_dict(torch.load(os.path.join(node_generator_folder, 
                                                       node_generator_name)))





z = torch.randn(batch.x.shape[0], 
                    32).to(device)
edge_index = batch.edge_index
norm = normalize_adj(batch.edge_index, batch.x)

z = torch.cat((z, batch.cycles), dim = -1)
x_gen  = generator(edge_index, z, norm = norm)
x_gen = torch.tanh(x_gen)
mean_true  = batch.x.mean()
mean_gen = x_gen.mean()

delta_real = batch.x[edge_index[0]]-batch.x[edge_index[1]]  
dist_real = (delta_real[:,0]**2 + delta_real[:, 1]**2).sqrt()

delta_gen = x_gen[edge_index[0]]-x_gen[edge_index[1]]  
dist_gen = (delta_gen[:,0]**2 + delta_gen[:, 1]**2).sqrt()

jsd_dist, dist_real, dist_gen = JSD_distance(dist_real, dist_gen, edge_index, 0, 2.828427)
jsd_x, x_real, x_g = JSD_distance(batch.x[:,0], x_gen[:,0], edge_index, -1, 1)
jsd_y, y_real, y_g = JSD_distance(batch.x[:,1], x_gen[:,1], edge_index, -1, 1)

np.savetxt("./csv/Fingerprint_true_x1.csv", 
           (x_real/x_real.sum()).detach().cpu().numpy(),
           delimiter =", ", 
           fmt ='% s')

np.savetxt("./csv/Fingerprint_true_x2.csv", 
           (y_real/y_real.sum()).detach().cpu().numpy(),
           delimiter =", ", 
           fmt ='% s')

np.savetxt("./csv/Fingerprint_gen_x1.csv", 
           (x_g/x_g.sum()).detach().cpu().numpy(),
           delimiter =", ", 
           fmt ='% s')

np.savetxt("./csv/Fingerprint_gen_x2.csv", 
           (y_g/y_g.sum()).detach().cpu().numpy(),
           delimiter =", ", 
           fmt ='% s')

np.savetxt("./csv/Fingerprint_true_dist.csv", 
           (dist_real/dist_real.sum()).detach().cpu().numpy(),
           delimiter =", ", 
           fmt ='% s')

np.savetxt("./csv/Fingerprint_gen_dist.csv", 
           (dist_gen/dist_gen.sum()).detach().cpu().numpy(),
           delimiter =", ", 
           fmt ='% s')


generator = Gen_edge4(
                     [sizes['node_types']+sizes['cycles']] + args_edge['n_layers_g']*[args_edge['layer_size']] + [sizes['edge_types']], 
                          [sizes['noise_edge']] + args_edge['n_layers_g']*[args_edge['layer_size']] + [sizes['edge_types']],
                          activation=nn.CELU()).to(device)

generator.load_state_dict(torch.load(os.path.join(edge_generator_folder, 
                                                       edge_generator_name)))


z = torch.randn(batch.edge_index.shape[-1], 
                        args_edge['noise_dim']).to(device)       
z = get_symetric_sparse_adj(batch.edge_index, z)

norm = normalize_adj(batch.edge_index, batch.x)

x = torch.cat((batch.x, batch.cycles), dim = -1)

edge_attr_gen  = generator(batch.edge_index, x,
                           z, norm = norm)
edge_attr_gen = torch.tanh(edge_attr_gen)

jsd1, wg1, wr1 = JSD_distance(edge_attr_gen[:,0], batch.edge_attr[:,0], 
             batch.edge_index, -1, 1)
jsd2, wg2, wr2 = JSD_distance(edge_attr_gen[:,1], batch.edge_attr[:,1], 
             batch.edge_index, -1, 1)
wr1 = wr1.detach().cpu().numpy()
wr1 = wr1/wr1.sum()
wr2 = wr2.detach().cpu().numpy()
wr2 = wr2/wr2.sum()

wg1 = wg1.detach().cpu().numpy()
wg1 = wg1/wg1.sum()
wg2 = wg2.detach().cpu().numpy()
wg2 = wg2/wg2.sum()

plt.bar(np.arange(-1, 1, 0.01), wr1, width = 0.01)
plt.bar(np.arange(-1, 1, 0.01), wg1, width = 0.01)
#plt.bar(np.arange(-1,1,200), wg1.detach().cpu().numpy())
plt.show()
plt.bar(np.arange(-1, 1, 0.01), wr2, width = 0.01)
plt.bar(np.arange(-1, 1, 0.01), wg2, width = 0.01)
#plt.bar(np.arange(-1,1,200), wg1.detach().cpu().numpy())
plt.show()



np.savetxt("./csv/Fingerprint_true_w1.csv", 
           wr1,
           delimiter =", ", 
           fmt ='% s')

np.savetxt("./csv/Fingerprint_true_w2.csv", 
           wr2,
           delimiter =", ", 
           fmt ='% s')

np.savetxt("./csv/Fingerprint_gen_w1.csv", 
           wg1,
           delimiter =", ", 
           fmt ='% s')

np.savetxt("./csv/Fingerprint_gen_w2.csv", 
           wg2,
           delimiter =", ", 
           fmt ='% s')

plt.bar(np.arange(0, 3, 0.015), (dist_real/dist_real.sum()).detach().cpu().numpy(), width = 0.01)
plt.bar(np.arange(0, 3, 0.015), (dist_gen/dist_gen.sum()).detach().cpu().numpy(), width = 0.01)
#plt.bar(np.arange(-1,1,200), wg1.detach().cpu().numpy())
plt.show()
print(jsd_x, jsd_y, jsd_dist, jsd1, jsd2)