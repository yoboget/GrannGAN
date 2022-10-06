# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:23:53 2020
Main call

@author: Yoann Boget
"""

import args_parse
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

args = args_parse.parse_args()
if args.debug:
    args.batch_size = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.device = device
print(f'Run on {device}')

####################
# Data
####################
#data_path = utils.func.set_path(args)
#args.data_path = data_path
if args.dataset == 'qm9':
    args.data_path = args.data_path + 'qm9/'
elif args.dataset == 'zinc':
    args.data_path = args.data_path + 'zinc/'

if args.dataset != 'Fingerprint':
    if args.dataset_version == 'cheat':
        kekulize = True
        countH = False
        formal_charge = False
    elif args.dataset_version == 'arom':
        kekulize = False
        countH = False
        formal_charge = False
    elif args.dataset_version == 'full':
        kekulize = False
        countH = True
        formal_charge = True
    else: raise NotImplementedError('Dataset version not implemented yet')


    dataset = MyMolDataset(args.dataset, args.data_path, kekulize=kekulize, countH=countH,
                 formal_charge=formal_charge, exclude_arom=False, exclude_charged=True)
    args.meta = dataset.meta

else:
    if args.cycles:
        transform = 'FUNCTION ADD CYCLE'
    else: transform = None
    args.discretizing_method = None
    dataset = TUDataset(args.data_path, args.dataset,
                        pre_transform= MyFingerprintTransform(),
                        use_node_attr=True,
                        use_edge_attr=True)

torch.manual_seed(42)
perm = torch.randperm(len(dataset))
train_idx = perm[500:]
test_idx = perm[:500]
print(dataset[:50])
train_loader = DataLoader(dataset[train_idx], batch_size=args.batch_size,
                          shuffle=True, drop_last=True, pin_memory=True)

test_loader = DataLoader(dataset[test_idx], batch_size=500)

batch = next(iter(train_loader))
sizes = {'n_nodes': batch[0].x.shape[0],
         'node_types': batch[0].x.shape[-1],
         'edge_types': batch.edge_attr.shape[-1],
         'batch': args.batch_size,
         'noise': args.noise_dim,
         'batch_test': 500}
if args.cycles:
    sizes['cycles'] = 5
else: sizes['cycles'] = 0

args.sizes = sizes 
print(f'Maximum of {sizes["n_nodes"]} atoms by molecule. {sizes["node_types"]}\
      types of atoms. {sizes["edge_types"]} types of bonds. Cycles info as node features: {args.cycles}')


if args.train_step == 'adja':
    generator = Gen_edge4(
                          [sizes['node_types']+sizes['cycles']] +\
                          args.n_layers_g*[args.layer_size] +\
                          [sizes['edge_types']],
                          [sizes['noise']] + args.n_layers_g*[args.layer_size] + [sizes['edge_types']],
                          activation=args.activation
                          ).to(device)
    

    discriminator = Disc_edge5([sizes['node_types']+sizes['cycles']] + args.n_layers_d*[args.layer_size],
                                     [sizes['edge_types']] + args.n_layers_d*[args.layer_size], 
                                     [64, 64],
                                     activation=args.activation,
                                     end = args.discriminator_end).to(device)

elif args.train_step == 'nodes':
    generator = Gen_node3( 
                          [sizes['noise']+sizes['cycles']] +\
                          args.n_layers_g*[args.layer_size]+ [sizes['node_types']],
                          [0] + (1+args.n_layers_g)*[args.layer_size],
                          activation=args.activation).to(device)
                             
    discriminator = Disc_node3(
                              [sizes['node_types']+sizes['cycles']] +\
                              args.n_layers_d*[args.layer_size], 
                              [0] + args.n_layers_d*[args.layer_size], 
                              [64, 64],
                              activation=args.activation,
                                         end = args.discriminator_end
                                         ).to(device)

else:
    raise Exception('training step not defined')
   
if args.path_model_to_load is not None:
    generator.load_state_dict(torch.load(os.path.join(args.path_model_to_load, f'generator_ep{args.num_model_to_load}.pt')))
    discriminator.load_state_dict(torch.load(os.path.join(args.path_model_to_load, f'discriminator_ep{args.num_model_to_load}.pt')))

opt_generator=optim.Adam(generator.parameters(), 
                       lr=args.learning_rate_g, betas=args.betas_g)

opt_discriminator=optim.Adam(discriminator.parameters(),
                       lr=args.learning_rate_d, betas=args.betas_d)

lr_scheduler_g = optim.lr_scheduler.ExponentialLR(optimizer=opt_generator, 
                                                gamma=args.learning_rate_decay)
lr_scheduler_d = optim.lr_scheduler.ExponentialLR(optimizer=opt_discriminator, 
                                                gamma=args.learning_rate_decay)

schedulers = (lr_scheduler_g, lr_scheduler_d)


models = [generator, discriminator]
optimizors = [opt_generator, opt_discriminator]

trainer = Trainer(train_loader, test_loader, models, 
                  optimizors, schedulers, args)

trainer.fit()
    


