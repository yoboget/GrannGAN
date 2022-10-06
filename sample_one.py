#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:51:02 2022

@author: yoann
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from mol_utils.dataset import MyMolDataset 
import os
import glob
from nets.generators import Gen_edge4, Gen_node3
from torch_geometric.utils import to_dense_adj, to_dense_batch
from utils.func import get_symetric_sparse_adj, normalize_adj, discretize
from mol_utils.rdkitfuncs import MolConstructor
from mol_utils.molmetrics import getmetrics
import wandb

def get_mol_metrics(annotation, adjacency, annotation_real,
                    adjacency_real, meta, get_novel):
    graphset = MolConstructor(annotation.type(torch.long),
                              adjacency.type(torch.long), meta)
    graphset_db = MolConstructor(annotation_real.type(torch.long),
                              adjacency_real.type(torch.long), meta)
    graphset.create_rdkit()
    graphset_db.create_rdkit()
    if get_novel:
        metrics = getmetrics(graphset.mols, graphset.n_errs, graphset_db.mols)
    else:
        metrics = getmetrics(graphset.mols, graphset.n_errs)
    return metrics

get_novel = False
mol_metric = True
prop = True
root = '/home/users/b/boget3/'

'''
ZINC
'''
# 500000 = 57%
edge_generator_name = 'edge_generator_ep130000.pt'
edge_run_path = 'yobo/zinc_arom_adja/2xzuvivl'

node_generator_name = 'node_generator_ep400000.pt'
node_run_path = 'yobo/zinc_arom_nodes/2wtc31mn'


'''
QM9

edge_generator_name = 'edge_generator_ep260000.pt'
edge_run_path = 'yobo/qm9_arom_adja/28ibmyid'

node_generator_name = 'node_generator_ep660000.pt'
node_run_path = 'yobo/qm9_arom_nodes/35f857na'
'''



api = wandb.Api()
node_run = api.run(node_run_path)
args_node = node_run.config
id_ = node_run.id
root = os.getcwd()
folder = glob.glob(f'{root}/wandb/*{id_}')[0] 
node_generator_folder = f'{folder}/files/'


api = wandb.Api()
edge_run = api.run(edge_run_path)
args_edge = edge_run.config
id_ = edge_run.id
root = os.getcwd()
folder = glob.glob(f'{root}/wandb/*{id_}')[0] 
edge_generator_folder = f'{folder}/files/'


cuda = torch.cuda.is_available()
if cuda:
    device = torch.device("cuda")
    #torch.set_default_tensor_type(torch.cuda.FloatTensor) 
else:
    device = torch.device("cpu")

path = f'{root}/data/{args_node["dataset"]}/'
dataset = MyMolDataset(args_node['dataset'], path, kekulize=False, countH=False,
             formal_charge=False, exclude_arom=False, exclude_charged=True)
meta = dataset.meta

torch.manual_seed(43)
perm = torch.randperm(len(dataset))
idx = perm[:1]
loader = DataLoader(dataset[idx], batch_size=1, 
                          shuffle=False, drop_last=False, pin_memory=True)


batch = next(iter(loader))
sizes = {'n_nodes': batch[0].x.shape[0],
     'node_types': batch[0].x.shape[-1],
     'edge_types': batch.edge_attr.shape[-1],
     'batch': 1,
     'noise_edge': args_edge['noise_dim'],
     'noise_node': args_node['noise_dim']}

annotation = to_dense_batch(batch.x, batch.batch, 
                                        max_num_nodes = sizes['n_nodes'])[0]  

sizes['noise'] = args_node['noise_dim']
        
if args_node['cycles']:
    sizes['cycles'] = 5

node_generator = Gen_node3(
                    [sizes['noise_node']+sizes['cycles']] +\
                    args_node['n_layers_g']*[args_node['layer_size']]+ [sizes['node_types']],
                    [0] + (1+args_node['n_layers_g'])*[args_node['layer_size']],
                    activation=nn.CELU()).to(device)


#node_generator = wandb.restore('generator_annot_ep500.pt', run_path="yobo/debug/29vrbwyk")

node_generator.load_state_dict(torch.load(os.path.join(node_generator_folder, 
                                                       node_generator_name)))
#gen_node.load_state_dict(torch.load(os.path.join(path_to_models, node_model),
#                                    map_location=torch.device('cpu')))
    

edge_generator = Gen_edge4(
                     [sizes['node_types']+sizes['cycles']] + args_edge['n_layers_g']*[args_edge['layer_size']] + [sizes['edge_types']], 
                          [sizes['noise_edge']] + args_edge['n_layers_g']*[args_edge['layer_size']] + [sizes['edge_types']],
                          activation=nn.CELU()).to(device)

edge_generator.load_state_dict(torch.load(os.path.join(edge_generator_folder, 
                                                       edge_generator_name)))

n_samples = 32
adjacency_generated = torch.zeros(1025, 5, 38, 38).to(device)
annotation_generated = torch.zeros(1025, 38, 10).to(device)

annotation_generated_ = to_dense_batch(batch.x, batch.batch, 
                                        max_num_nodes = sizes['n_nodes'])[0]   
adjacency_generated_ = to_dense_adj(edge_index = batch.edge_index, 
                                   batch = batch.batch, 
                                   edge_attr = batch.edge_attr)
adjacency_generated_ = adjacency_generated_.permute(0,3,1,2)           
no_bond = adjacency_generated_.sum(1, keepdim = True) *(-1) + 1
idx=annotation.sum(2)==0
annotation_generated_[idx]=0
no_atom = annotation_generated_.sum(2, keepdim = True) *(-1) + 1
annotation_generated_ = torch.cat((annotation_generated_, no_atom), dim=2)

adjacency_generated[0] = torch.cat((adjacency_generated_, no_bond), dim=1)
annotation_generated[0] = annotation_generated_



for batch in loader: 
    batch.to(device)
    for n in range(n_samples):    
    
        '''
        GENERATE NODE TYPES
        '''   
        
        z = torch.randn(batch.x.shape[0], 
                                sizes['noise_node']).to(device)      
        if args_node['normalize']:
            norm = normalize_adj(batch.edge_index, batch.x)
        else: norm = None
        if args_node['cycles']:
            z = torch.cat((z, batch.cycles), dim = -1)
        
        x_gen  = node_generator(batch.edge_index, z, norm = norm)
        if args_node['discretizing_method']:                                        
            x_gen = discretize(x_gen, method = args_node['discretizing_method'])
        
        for m in range(n_samples):
            z = torch.randn(batch.edge_index.shape[-1], 
                        args_edge['noise_dim']).to(device)       
            z = get_symetric_sparse_adj(batch.edge_index, z)
            if args_edge['normalize']:
                norm = normalize_adj(batch.edge_index, batch.x)
            else: norm = None
            if args_edge['cycles']:
                x_gen_ = torch.cat((x_gen, batch.cycles), dim = -1)
            else: x_gen_ = x_gen
            
            edge_attr_gen  = edge_generator(batch.edge_index, x_gen_,
                                       z, norm = norm)

            if args_edge['discretizing_method'] == 'gumbel-softmax':
                edge_attr_gen = F.gumbel_softmax(edge_attr_gen, hard=True)
            else:        
                argmax = torch.argmax(edge_attr_gen, axis=-1)
                edge_attr_gen = torch.eye(edge_attr_gen.shape[0])[argmax]
                edge_attr_gen = edge_attr_gen.to(device)
            
            edge_attr_gen = get_symetric_sparse_adj(batch.edge_index, 
                                                edge_attr_gen, 
                                                method = 'keep_upper_tri')
            
            annotation_generated_ = to_dense_batch(x_gen, batch.batch, 
                                        max_num_nodes = sizes['n_nodes'])[0]   
            adjacency_generated_ = to_dense_adj(edge_index = batch.edge_index, 
                                               batch = batch.batch, 
                                               edge_attr = edge_attr_gen)
            adjacency_generated_ = adjacency_generated_.permute(0,3,1,2)           
            no_bond = adjacency_generated_.sum(1, keepdim = True) *(-1) + 1
            idx=annotation.sum(2)==0
            annotation_generated_[idx]=0
            no_atom = annotation_generated_.sum(2, keepdim = True) *(-1) + 1
            annotation_generated_ = torch.cat((annotation_generated_, no_atom), dim=2)

            adjacency_generated[n_samples*n+m+1] = torch.cat((adjacency_generated_, no_bond), dim=1)
            annotation_generated[n_samples*n+m+1] = annotation_generated_
            
adjacency_real = to_dense_adj(edge_index = batch.edge_index, 
                                   batch = batch.batch, 
                                   edge_attr = batch.edge_attr)
adjacency_real = adjacency_real.permute(0,3,1,2)           
no_bond = adjacency_real.sum(1, keepdim = True) *(-1) + 1
adjacency_real = torch.cat((adjacency_real, no_bond), dim=1)

annotation_real = to_dense_batch(batch.x, batch.batch, 
                            max_num_nodes = sizes['n_nodes'])[0]   

idx=annotation_real.sum(2)==0
annotation_real[idx]=0
no_atom = annotation_real.sum(2, keepdim = True) *(-1) + 1
annotation_real = torch.cat((annotation_real, no_atom), dim=2)



if mol_metric:
    stats_total, valid, unique, novel = get_mol_metrics(
                                            annotation_generated, 
                                            adjacency_generated,
                                            annotation_real,
                                            adjacency_real,
                                            meta, 
                                            get_novel)
    print(valid, unique, novel) 
if prop:
    prop_edge_gen = []
    prop_edge_real = []
    for edge in range(adjacency_real.shape[1]):
        prop_edge_gen.append( adjacency_generated[:, edge].sum()/adjacency_generated[:, :-1].sum())
        prop_edge_real.append(adjacency_real[:, edge].sum()/adjacency_real[:, :-1].sum())
    prop_node_gen = []
    prop_node_real = []   
    n_atom = sizes['n_nodes'] * 1025
    n_atom2 = sizes['n_nodes'] * 1025
    print(len(dataset))
    for atom in range(sizes['node_types']):        
        prop_node_gen.append(annotation_generated[:, :, atom].sum()/n_atom)
        prop_node_real.append(annotation_real[:, :, atom].sum()/n_atom2)
    print(prop_edge_real, prop_edge_gen, prop_node_real, prop_node_gen)

