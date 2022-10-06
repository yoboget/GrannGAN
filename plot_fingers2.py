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
from utils.func import JSD_distance, normalize_adj
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

edge_generator_name = 'edge_generator_ep126000.pt'
edge_run_path = 'yobo/Fingerprint_arom_adja/258fbll7'

node_generator_name = 'node_generator_ep122000.pt'
node_run_path = 'yobo/Fingerprint_arom_nodes/2bzlnup2'


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

torch.manual_seed(42)
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




batch = next(iter(dataset)).to(device)
z = torch.randn(batch.x.shape[0], 
                    32).to(device)
edge_index = batch.edge_index
norm = normalize_adj(batch.edge_index, batch.x)
print(z.shape, batch.x.shape, batch.cycles.shape)
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
           (y_g/y_g.sum()).detach().cpu().numpy(),
           delimiter =", ", 
           fmt ='% s')


Gs = []
list_dict_node = []
list_dict_edge = []
k=0
n=5
m=5

for i, batch in enumerate(dataloader):
    '''
    g_gen = []
    dict_inst = []
    g_gen.append(batch)
    dict_inst.append(batch.x[0].item(), batch.x[1].item())
    for _ in range(m):
        z = torch.randn(batch.x.shape[0], 
                        sizes['noise']).to(device)
        
        norm = normalize_adj(batch.edge_index, batch.x)
        x_gen  = generator(batch.edge_index, z, norm = norm)
        x_gen = torch.tanh(x_gen)
        g_gen.append(Data(x_gen, batch.edge_index))
        dict_inst.append(x_gen[0].item(), x_gen[1].item())
    Gs.append(g_gen)
    list_dict.append(dict_inst)
    '''
    'REPRENDRE LA FIN DEPUIS ICI'
    
    
    Gs.append(to_networkx(batch,
            to_undirected = True, remove_self_loops = True))
    #print(Gs[-1].nodes())
    dict_node={}
    for j, d in enumerate(batch.x):
        dict_node[j] =(d[0].item(), d[1].item())
    list_dict_node.append(dict_node)
    dict_edge={}
    print(batch.edge_attr)
    for j, d in enumerate(batch.edge_attr):
        dict_edge[j] =(d[0].item()*0.5+0.5, 0, d[1].item()*0.5+0.5),
    list_dict_edge.append(dict_edge)

fig, ax = plt.subplots(n, m, figsize=(20, 20), dpi=100)
for l, G in enumerate(Gs[:500]):
    if k > n*m-1:
        break
    if len(G.edges)>3:
        nx.draw_networkx(G, pos = list_dict_node[l], 
                         with_labels=False, ax=ax[k//m, k%m], 
                         node_size = 10, 
                         edge_color = np.asarray(list(list_dict_edge[l].values())),
                         width = 8, node_color= 'black')
        ax[k//m, k%m].set_xlim(-1, 1)
        ax[k//m, k%m].set_ylim(-1, 1)
        k+=1

#fig.suptitle('Fingerprint graphs', size = 50)    
#plt.title('Fingerprint: real and generated graphs')
plt.show()

        

