#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 09:35:09 2022

@author: yo1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from mol_utils.dataset import MyMolDataset 

from nets.generators import Gen_edge, Gen_node
from torch_geometric.utils import to_dense_adj, to_dense_batch
from utils.func import get_symetric_sparse_adj, normalize_adj, discretize
from mol_utils.rdkitfuncs import MolConstructor
from mol_utils.molmetrics import getmetrics
import wandb

def get_mol_metrics(annotation, adjacency, meta):
    graphset = MolConstructor(annotation.type(torch.long),
                              adjacency.type(torch.long), meta)
    graphset.create_rdkit()
    metrics = getmetrics(graphset.mols, graphset.n_errs)
    return metrics


cuda = torch.cuda.is_available()
if cuda:
    device = torch.device("cuda")
    #torch.set_default_tensor_type(torch.cuda.FloatTensor) 
else:
    device = torch.device("cpu")

api = wandb.Api()
run = api.run('yobo/debug/29vrbwyk')
args_node = run.config


ARGS_EDGE = {'discretizing_method': 'gumbel-softmax',
             'tau': 1,
             'noise_dim': 32,
             'n_layers_g': 4,
             'layer_size': 64, 
             'normalize': True,
             'cycles': True,
             'device': device}

ARGS_NODE = {'discretizing_method': 'gumbel-softmax',
             'tau': 1,
             'noise_dim': 32,
             'n_layers_g': 4,
             'layer_size': 16, 
             'normalize': True, 
             'cycles': True,
             'device': device }

DATASET = 'qm9'
DATASET_VERSION = 'cheat'
#ROOT = '/home/users/b/boget3/data/' 
ROOT = '/home/yoann/datasets/'
GEN_FOLDER = 'generators/'

path_to_model_folder = ROOT +  DATASET + '/' + DATASET_VERSION + '/' +GEN_FOLDER + '/' 
edgemodel_filename = ''
nodemodel_filename = ''

edgemodel_path = path_to_model_folder+ '/' + edgemodel_filename
nodemodel_path = path_to_model_folder+ '/' + nodemodel_filename
N_SAMPLE = 5000
path = ROOT + DATASET + '/'
print(path)


if DATASET_VERSION == 'cheat':
    dataset = MyMolDataset(DATASET, path, kekulize=True, countH=False,
             formal_charge=False, exclude_arom=False, exclude_charged=True)
elif DATASET_VERSION == 'arom':
    dataset = MyMolDataset(DATASET,ROOT + '/' + DATASET + '/', kekulize=False, countH=False,
             formal_charge=False, exclude_arom=False, exclude_charged=True)
elif DATASET_VERSION == 'full':
    dataset = MyMolDataset(DATASET, ROOT + '/' + DATASET + '/', kekulize=False, countH=True,
             formal_charge=True, exclude_arom=False, exclude_charged=True)
else: raise NotImplementedError('Dataset version not implemented yet')



meta = dataset.meta
data_loader = DataLoader(dataset, batch_size= N_SAMPLE, 
                         shuffle=True, drop_last=True, pin_memory=True)
batch = next(iter(data_loader)).to(device)
print(ARGS_EDGE['noise_dim'])
sizes = {'n_nodes': batch[0].x.shape[0],
         'node_types': batch[0].x.shape[-1],
         'edge_types': batch.edge_attr.shape[-1],
         'batch': N_SAMPLE,
         'noise_edge': ARGS_EDGE['noise_dim'],
         'noise_node': ARGS_NODE['noise_dim']}


sizes['noise'] = ARGS_NODE['noise_dim']
        
if ARGS_NODE['cycles']:
    sizes['cycles'] = 5

node_generator = Gen_node(
                    [sizes['noise']+sizes['cycles']] +\
                    ARGS_NODE['n_layers_g']*[ARGS_NODE['layer_size']]+ [sizes['node_types']],
                    [0] + (1+ARGS_NODE['n_layers_g'])*[ARGS_NODE['layer_size']],
                    activation=nn.CELU()).to(device)

path = ROOT + DATASET + '/' + GEN_FOLDER
best_model = wandb.restore('generator_annot_ep500.pt', run_path="yobo/debug/29vrbwyk")

#node_generator.load_state_dict(torch.load(os.path.join(path, nodemodel_filename)))
#gen_node.load_state_dict(torch.load(os.path.join(path_to_models, node_model),
#                                    map_location=torch.device('cpu')))
    

edge_generator = Gen_edge(
                     [sizes['node_types']+sizes['cycles']] + ARGS_EDGE['n_layers_g']*[ARGS_EDGE['layer_size']] + [sizes['edge_types']], 
                          [sizes['noise_edge']] + ARGS_EDGE['n_layers_g']*[ARGS_EDGE['layer_size']] + [sizes['edge_types']],
                          activation=nn.CELU()).to(device)

#edge_generator.load_state_dict(torch.load(os.path.join(path, edgemodel_filename)))



'''
GENERATE NODE TYPES
'''   
z = torch.randn(batch.x.shape[0], 
                        sizes['noise_node']).to(device)      
if ARGS_NODE['normalize']:
    norm = normalize_adj(batch.edge_index, batch.x)
else: norm = None
if ARGS_NODE['cycles']:
    z = torch.cat((z, batch.cycles), dim = -1)

x_gen  = node_generator(batch.edge_index, z, norm = norm)
if ARGS_NODE['discretizing_method']:                                        
    x_gen = discretize(x_gen, method = ARGS_NODE['discretizing_method'])




    
'''
GENERATE EDGE TYPES
'''                
import torch
z = torch.randn(batch.edge_index.shape[-1], 
                        sizes['noise']).to(device)       
z = get_symetric_sparse_adj(batch.edge_index, z)
if ARGS_EDGE['normalize']:
    norm = normalize_adj(batch.edge_index, batch.x)
else: norm = None
if ARGS_EDGE['cycles']:
    x = torch.cat((x_gen, batch.cycles), dim = -1)
edge_attr_gen  = edge_generator(batch.edge_index, x,
                           z, norm = norm)



if ARGS_EDGE['discretizing_method'] == 'gumbel-softmax':
    edge_attr_gen = F.gumbel_softmax(edge_attr_gen, hard=True)
else:        
    argmax = torch.argmax(edge_attr_gen, axis=-1)
    edge_attr_gen = torch.eye(edge_attr_gen.shape[0])[argmax]
    edge_attr_gen = edge_attr_gen.to(device)

edge_attr_gen = get_symetric_sparse_adj(batch.edge_index, 
                                    edge_attr_gen, 
                                    method = 'keep_upper_tri')

annotation = to_dense_batch(batch.x, batch.batch, 
                            max_num_nodes = sizes['n_nodes'])[0]   
adjacency_generated = to_dense_adj(edge_index = batch.edge_index, 
                                   batch = batch.batch, 
                                   edge_attr = edge_attr_gen)
adjacency_real = to_dense_adj(edge_index = batch.edge_index, 
                                   batch = batch.batch, 
                                   edge_attr = batch.edge_attr)
adjacency_generated = adjacency_generated.permute(0,3,1,2)
adjacency_real = adjacency_real.permute(0,3,1,2)





stats_total, valid, unique, novel = get_mol_metrics(
                                            annotation, 
                                            adjacency_generated, 
                                            meta)


print(valid, unique, novel)        
