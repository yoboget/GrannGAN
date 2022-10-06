#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 12:17:23 2021

@author: Yoann Boget
"""
import wandb
from wandb.plot import histogram
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch
import glob
from mol_utils.rdkitfuncs import MolConstructor 
from mol_utils.molmetrics import getmetrics
from nets.generators import Gen_edge
from utils.func import get_symetric_sparse_adj, normalize_adj, discretize, mse_angles, JSD_distance

import os




class Eval():
    def __init__(self, args, generator, discriminator):
        

        wandb.init(project= f'{args.dataset}_{args.dataset_version}_{args.train_step}', 
                   entity='yobo', 
                   config = args)
        
        #wandb.init(project= f'{args.dataset}_{args.dataset_version}_MolGAN_{args.train_step}', 
        #           entity='yobo', 
        #           config = args)
        wandb.config.generator = generator
        wandb.config.discriminator = discriminator
        self.wandb = wandb
        
        self.history_keys = ['loss']
        self.history = {key: [] for key in self.history_keys}
        self.history_temp = {key: [] for key in self.history_keys}
        self.valid_max = 0
        
        
    def init_history_batch(self):
        self.history_batch = {key: [] for key in self.history_keys}
    
    def add_epoch_loss(self, n_iter):
        for key in self.history_keys:
            if len(self.history_temp[key]) != 0:
                temp_mean = sum(self.history_temp[key])/len(self.history_temp[key])
                self.history[key] = temp_mean
                wandb.log({key: self.history[key], 'iterations': n_iter})
        self.history_temp = {key: [] for key in self.history_keys}
            
    def step_edges(self, generator, 
                        batch, 
                        sizes, 
                        args, 
                        n_iter, device):
        with torch.no_grad():
            print('hello world')
    
    def step_nodes_attr(self, generator, 
                    batch, 
                    sizes,                         
                    args, 
                    n_iter):
        with torch.no_grad():
            generator.eval()           
            z = torch.randn(batch.x.shape[0], 
                    sizes['noise']).to(args.device)      
            if args.normalize:
                norm = normalize_adj(batch.edge_index, batch.x)
            else: norm = None
            if args.cycles:
                z = torch.cat((z, batch.cycles), dim = -1)
            edge_index = batch.edge_index
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
            
            wandb.log({'JSD_point_distance': jsd_dist})
           
                       
            wandb.log({'JSD_point_distance': jsd_dist, 'True_mean': mean_true,
                       'Gen_mean': mean_gen,'JSD x': jsd_x, 'JSD y': jsd_y,
                       'n_iter':n_iter})  

    
    def step_nodes(self, 
                   generator, 
                   batch, 
                   sizes,                       
                   args, 
                   n_iter):
        with torch.no_grad():          
            generator.eval()    
            
            z = torch.randn(batch.x.shape[0], 
                        sizes['noise']).to(args.device)      
            if args.normalize:
                norm = normalize_adj(batch.edge_index, batch.x)
            else: norm = None
            if args.cycles:
                z = torch.cat((z, batch.cycles), dim = -1)
        
            x_gen  = generator(batch.edge_index, z, norm = norm)
            if args.discretizing_method:                                        
                x_gen = discretize(x_gen, method = args.discretizing_method)

            annotation = to_dense_batch(x_gen, batch.batch, 
                                        max_num_nodes = sizes['n_nodes'])[0]           
            no_atom = annotation.sum(2, keepdim = True) *(-1) + 1
            annotation = torch.cat((annotation, no_atom), dim=2)
            
            adjacency_generated = to_dense_adj(edge_index = batch.edge_index, 
                                               batch = batch.batch, 
                                               edge_attr = batch.edge_attr)
            
            adjacency_generated = adjacency_generated.permute(0,3,1,2)
           
            
            no_bond = adjacency_generated.sum(1, keepdim = True) *(-1) + 1
            adjacency_generated = torch.cat((adjacency_generated, no_bond), dim=1)
            
            stats_total, valid, unique, novel = self.get_mol_metrics(
                                                        annotation, 
                                                        adjacency_generated, 
                                                        args.meta)
            wandb.log({'valid': valid,'unique':unique,'n_iter':n_iter})
            
            '''
            adjacency_generated = utils.func.generate_adjacency(
                                                        generator,
                                                        annotation,
                                                         scaffold, 
                                                         SIZES, 
                                                         ARGS)

            stats_total, valid, unique, novel = utils.func.get_mol_metrics(
                                                            annotation, 
                                                            adjacency_generated, 
                                                            args.data_path)
            
                wandb.log({'valid': valid,'unique':unique,'epoch':epoch+1})
                self.log_edge_type(adjacency_generated, 
                                   adjacency_real, 
                               sizes, epoch)
            '''
            annotation = to_dense_batch(x_gen, batch.batch, 
                                        max_num_nodes = sizes['n_nodes'])[0]
            annotation_real = to_dense_batch(batch.x, batch.batch, 
                                        max_num_nodes = sizes['n_nodes'])[0]
            self.log_annot(annotation, annotation_real, sizes, n_iter)
            
           
            if (n_iter) % 10000 == 0:
                torch.save(generator.state_dict(), 
                           os.path.join(self.wandb.run.dir, 
                                        f'node_generator_ep{n_iter}.pt'))
            
            
    
    def step_edge_attr(self, generator, 
                        batch, 
                        sizes,                         
                        args, 
                        n_iter):
        with torch.no_grad():
            generator.eval()           
            z = torch.randn(batch.edge_index.shape[-1], 
                        sizes['noise']).to(args.device)       
            z = get_symetric_sparse_adj(batch.edge_index, z)
            if args.normalize:
                norm = normalize_adj(batch.edge_index, batch.x)
            else: norm = None
            if args.cycles:
                x = torch.cat((batch.x, batch.cycles), dim = -1)
            else: x = batch.x
            edge_attr_gen  = generator(batch.edge_index, x,
                                       z, norm = norm)
            edge_attr_gen = torch.tanh(edge_attr_gen)
            mse = mse_angles(batch, edge_attr_gen)
            mse_data = mse_angles(batch, batch.edge_attr)
            mse_0 = ((edge_attr_gen[:,0] - batch.edge_attr[:,0])**2).mean()
            mse_1 = ((edge_attr_gen[:,1] - batch.edge_attr[:,1])**2).mean()
            print(edge_attr_gen[:,0].shape, batch.edge_attr[:,0])
            jsd1, a, b = JSD_distance(edge_attr_gen[:,0], batch.edge_attr[:,0], 
                         batch.edge_index, -1, 1)
            jsd2, a, b = JSD_distance(edge_attr_gen[:,1], batch.edge_attr[:,1], 
                         batch.edge_index, -1, 1)
            #print('jsd1')
            #print(jsd1)
            #print(a)
            wandb.log({'mse angle': mse, 
                       'mse_data': mse_data, 
                       'mse_0' : mse_0, 
                       'mse_1': mse_1, 
                       'jsd_1': jsd1.item(),
                       'jsd_2': jsd2.item(),
                       'n_iter':n_iter})
            
    def step_edge_types(self, generator, 
                        batch, 
                        sizes,                         
                        args, 
                        n_iter):
        with torch.no_grad():
            generator.eval()           
            z = torch.randn(batch.edge_index.shape[-1], 
                        sizes['noise']).to(args.device)       
            z = get_symetric_sparse_adj(batch.edge_index, z)
            if args.normalize:
                norm = normalize_adj(batch.edge_index, batch.x)
            else: norm = None
            if args.cycles:
                x = torch.cat((batch.x, batch.cycles), dim = -1)
            edge_attr_gen  = generator(batch.edge_index, x,
                                       z, norm = norm)
            

            
            if args.discretizing_method == 'gumbel-softmax':
                edge_attr_gen = F.gumbel_softmax(edge_attr_gen, hard=True)
            else:        
                argmax = torch.argmax(edge_attr_gen, axis=-1)
                edge_attr_gen = torch.eye(edge_attr_gen.shape[0])[argmax]
                edge_attr_gen = edge_attr_gen.to(args.device)
            
            edge_attr_gen = get_symetric_sparse_adj(batch.edge_index, 
                                                edge_attr_gen, 
                                                method = 'keep_upper_tri')
            
            annotation = to_dense_batch(batch.x, batch.batch, 
                                        max_num_nodes = sizes['n_nodes'])[0] 
            no_atom = annotation.sum(2, keepdim = True) *(-1) + 1
            annotation = torch.cat((annotation, no_atom), dim=2)
            
            adjacency_generated = to_dense_adj(edge_index = batch.edge_index, 
                                               batch = batch.batch, 
                                               edge_attr = edge_attr_gen)
            adjacency_real = to_dense_adj(edge_index = batch.edge_index, 
                                               batch = batch.batch, 
                                               edge_attr = batch.edge_attr)
            adjacency_generated = adjacency_generated.permute(0,3,1,2)
            adjacency_real = adjacency_real.permute(0,3,1,2)
            
            no_bond = adjacency_real.sum(1, keepdim = True) *(-1) + 1
            adjacency_real = torch.cat((adjacency_real, no_bond), dim=1)
            adjacency_generated = torch.cat((adjacency_generated, no_bond), dim=1)
            stats_total, valid, unique, novel = self.get_mol_metrics(
                                                        annotation, 
                                                        adjacency_generated, 
                                                        args.meta)
            
            wandb.log({'valid': valid,'unique':unique,'n_iter':n_iter})
            self.log_edge_type(adjacency_generated, 
                               adjacency_real, 
                               sizes, n_iter)
            

            if (n_iter) % 10000 == 0:
                torch.save(generator.state_dict(), 
                           os.path.join(self.wandb.run.dir, 
                                        f'edge_generator_ep{n_iter}.pt'))
                

    def log_edge_type(self, adjacency, adjacency_real, sizes, n_iter):
        dict_ = {}
        bond_dict={}
        for edge in range(adjacency_real.shape[1]):
            prop_gen = adjacency[:, edge].sum()/adjacency[:, :-1].sum()
            prop_real = adjacency_real[:, edge].sum()/adjacency_real[:, :-1].sum()
            bond_dict = {f'Prop bond {edge} gen': prop_gen, 
                       f'Prop bond {edge} real': prop_real}
            dict_ = {**dict_, **bond_dict}
        self.wandb.log({**dict_, 'epoch':n_iter})
    
    def log_annot(self, annotation_generated, annotation_real, sizes, n_iter):
        dict_ = {}
        atom_dict={}
        n_atom = sizes['n_nodes'] * sizes['batch_test']
        for atom in range(sizes['node_types']):        
            prop_gen = annotation_generated[:, :, atom].sum()/n_atom
            prop_real = annotation_real[:, :, atom].sum()/n_atom
            atom_dict = {f'Prop atom {atom} gen': prop_gen, 
                       f'Prop atom {atom} real': prop_real}
            dict_ = {**dict_, **atom_dict}

        self.wandb.log({**dict_, 'epoch':n_iter})
    
    def get_mol_metrics(self, annotation, adjacency, meta):
        graphset = MolConstructor(annotation.type(torch.long),
                                  adjacency.type(torch.long), meta)
        graphset.create_rdkit()
        metrics = getmetrics(graphset.mols, graphset.n_errs)
        return metrics
          