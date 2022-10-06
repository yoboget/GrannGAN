#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 09:52:30 2021

@author: yo1
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import networkx as nx
from torch_sparse import SparseTensor
from torch_geometric.utils import degree


def mse_angles(batch, edge_attr):
    delta = batch.x[batch.edge_index[0]]-batch.x[batch.edge_index[1]]     
    true_angle = torch.acos(delta[:, 0]/ delta[:, 1])
    true_angle[(torch.isnan(true_angle))] = torch.acos(delta[:, 1]/delta[:, 0])[(torch.isnan(true_angle))]
    mse = ((true_angle - edge_attr[:,1])**2).mean()
    print(true_angle.shape, edge_attr[:, 1].shape)
    print(torch.cat(((true_angle - edge_attr[:,1]).unsqueeze(1), 
                    true_angle.unsqueeze(1), edge_attr[:,1].unsqueeze(1)), 
                    dim =1))
    return mse

def JSD_distance(x1, x2, edge_index, min_, max_):

    hist1 = torch.histc(x1, bins=200, min = min_, max= max_)
    pmf1 = hist1/hist1.sum()
    hist2 = torch.histc(x2, bins=200, min = min_, max= max_)
    pmf2 = hist2/hist2.sum()
    m = .5*(pmf1+pmf2)
    KL1 = (pmf1 * torch.log2( pmf1/m))
    KL2 = (pmf2 * torch.log2( pmf2/m))
    JSD = .5*(KL1.nansum() + KL2.nansum())
    return JSD, hist1, hist2

def normalize_adj(edge_index, x):
    row, col = edge_index
    deg_inv_sqrt = degree(col, x.size(0)).float()
    deg_inv_sqrt[deg_inv_sqrt.nonzero()] = deg_inv_sqrt[deg_inv_sqrt.nonzero()]**(-0.5)
    return deg_inv_sqrt[row] * deg_inv_sqrt[col]
    

def discretize(edge_attr, method = 'gumbel-softmax'):
    if method == 'gumbel-softmax':
        return F.gumbel_softmax(edge_attr, hard=True)
    elif method == 'softmax-st':
        device = edge_attr.device 
        edge_attr_gen = edge_attr.softmax(-1)
        idx = torch.argmax(edge_attr, dim=-1)
        hard = torch.eye(edge_attr.shape[-1])[idx].to(device)
        return edge_attr_gen - edge_attr.detach() + hard.detach()
    else: raise NotImplementedError('Discretizing method not implemented')

def discretize_adjacency(adjacency, 
                         args,
                         skeleton = None):
    method = args.discretizing_method
    device = args.device
    skeleton = skeleton.unsqueeze(1)
    adjacency = adjacency.permute(0, 3, 1, 2)
    if method == 'gumbel-softmax':
        adjacency_generated = gumbel_for_adjacency(adjacency, 
                                                   device,
                                                   dim = 1,
                                                   tau = args.tau)
        
    elif method == 'st':
        adjacency_generated = straight_through(adjacency, 
                                                    dim = 1)
    elif method == 'softmax-st':
        adjacency_generated = softmax_st(adjacency, 
                                                    dim = 1)  
        #print(adjacency_generated[0])
    elif method == 'sigmoid':
        adjacency_generated = adjacency.sigmoid()

    elif method == 'sigmoid-st':
        adjacency_generated = adjacency.sigmoid()
        hard = (adjacency_generated>0.5)*1.
        adjacency_generated = adjacency_generated - adjacency_generated.detach() \
                                + hard
    elif method == 'round':
        adjacency_generated = adjacency
        hard = (adjacency_generated>0.5)*1.
        adjacency_generated = adjacency_generated - adjacency_generated.detach() \
                                + hard
    elif method == 'do not':
        adjacency_generated = adjacency
        
    else:
        raise Exception('Discretization method not implemented')
    if skeleton is  not None:
        adjacency_generated = adjacency_generated*skeleton
    return adjacency_generated
    
def gumbel_for_adjacency(logit, device, dim = 1, tau = 1, hard = True):
    noise = -torch.empty_like(logit).to(device)
    gumbel_noise = noise.exponential_().log() # ~Gumbel(0,1)
    gumbel_noise_symmetric = torch.zeros(logit.shape).to(device) +\
                        torch.triu(gumbel_noise, 1) +\
                        torch.triu(gumbel_noise, 1).transpose(-2, -1)
                        
    gumbel_noise_symmetric = gumbel_noise_symmetric.to(device)
    y_soft = (logit + gumbel_noise_symmetric) / tau 
    y_soft = y_soft.softmax(dim)

    if hard:
        idx = torch.argmax(y_soft, dim=dim)
        y_hard = torch.eye(logit.shape[dim])[idx].permute(0, 3, 1, 2).to(device)
        y_hard = y_soft - y_soft.detach() + y_hard.detach()
        return y_hard
    else: 
        return y_soft

def softmax_st(logit, dim = 1, tau = 1, hard = True):
    zero_diag = torch.ones(logit.shape[-2:]) - torch.eye(logit.shape[-1])
    logit = logit.softmax(dim)
    device = logit.device
    logit = logit * zero_diag.to(device)
    device = logit.device
    idx = torch.argmax(logit, dim=dim)
    y_hard = torch.eye(logit.shape[dim])[idx].permute(0, 3, 1, 2).to(device)
    y_hard = logit - logit.detach() + y_hard.detach()
    return y_hard*zero_diag.to(device)

def straight_through(logit, dim = 1, tau = 1, hard = True):
    device = logit.device
    zero_diag = torch.ones(logit.shape[-2:]).to(device) - torch.eye(logit.shape[-1]).to(device)
    logit = logit * zero_diag
    idx = torch.argmax(logit, dim=dim)
    y_hard = torch.eye(logit.shape[dim])[idx].permute(0, 3, 1, 2).to(device)
    y_hard = logit - logit.detach() + y_hard.detach()
    return y_hard*zero_diag

def discretize_annotation(annotation, method = 'sigmoid-st'):
    if method == 'sigmoid-st':
        annotation_generated = annotation.sigmoid()
        hard = (annotation_generated>0.5)*1.
        #hard = idx = torch.argmax(annotation_generated, dim=2)
        #hard = torch.eye(annotation_generated.shape[2])[idx].permute(0, 3, 1, 2)
        annotation_generated = annotation_generated - annotation_generated.detach() \
                           + hard
    elif method == 'softmax-st':         
        annotation_generated = annotation.softmax(2)
        idx = torch.argmax(annotation_generated, dim=2)
        hard = torch.eye(annotation_generated.shape[2])[idx]
        annotation_generated = annotation_generated - annotation_generated.detach() \
                           + hard
                           
    elif method == 'gumbel-softmax':
        annotation_generated = F.gumbel_softmax(annotation_generated, 
                                                tau = 1, 
                                                hard=True, dim=2)
        
    elif method == 'gumbel-sigmoid':  
        annotation_generated = gumbel_sigmoid(annotation_generated, 
                                                hard=True)
    elif method == 'sigmoid':
        annotation_generated = annotation.sigmoid()
    
    elif method == 'softmax':
        annotation_generated = annotation.softmax(2)
    
    return annotation_generated
        
def gumbel_sigmoid(logit, tau = 1, hard = True):
        noise = -torch.empty_like(logit, memory_format=torch.legacy_contiguous_format)
        gumbel_noise = noise.exponential_().log() # ~Gumbel(0,1)
        y_soft = logit + gumbel_noise*tau
        y_soft = y_soft.sigmoid()

        if hard:
            y_hard = (y_soft>0.5)*1.
            y_hard = y_soft - y_soft.detach() + y_hard.detach()
            
            return y_hard
        else: 
            return y_soft
        
def discretize_scaffold(scaffold, args):
    method = args.discretizing_method
    sizes = args.sizes
    device = args.device
    
    if method == 'gumbel-sigmoid':  
        scaffold_generated = gumbel_sigmoid_edges(scaffold, sizes, device)
    elif method == 'sigmoid-st':
        soft = scaffold.sigmoid()
        scaffold_generated = soft - soft.detach() + soft.round().detach()
    elif method == 'sigmoid':
        scaffold_generated = scaffold.sigmoid()
    else:
        raise Exception('Method not implemented')
 
    zero_diag = torch.eye(sizes['mols']).to(device)*(-1) + 1
    scaffold_generated = scaffold_generated*zero_diag
    
    return scaffold_generated
    
def gumbel_sigmoid_edges(logit, sizes, device, tau = 1, hard = True):
    noise = -torch.empty_like(logit, memory_format=torch.legacy_contiguous_format)
    gumbel_noise = noise.exponential_().log() # ~Gumbel(0,1)
    dim = logit.shape[1]
    gumbel_noise = torch.zeros(sizes['batch'], 
                                        dim,
                                        sizes['mols'],
                                        sizes['mols']).to(device) +\
                            torch.triu(gumbel_noise, 1) +\
                            torch.triu(gumbel_noise, 1).transpose(-2, -1)
    
    y_soft = logit + gumbel_noise*tau
    y_soft = y_soft.sigmoid()

    if hard:
        y_hard = (y_soft>0.5)*1.
        y_hard = y_soft - y_soft.detach() + y_hard.detach()
        
        return y_hard
    else: 
        return y_soft
    
def get_adjacency_one_hot(adjacency, sizes, scaffold = None):
    device = adjacency.device
    argmax = torch.argmax(adjacency, axis=1)
    adja_one_hot = torch.eye(sizes['edge_types'])[argmax].to(device)
    adja_one_hot = adja_one_hot.permute(0,3,2,1)
    adja_one_hot = adja_one_hot*scaffold
    adja_one_hot = torch.cat([adja_one_hot,(scaffold*(-1))+1], axis=1)
    return adja_one_hot

def generate_adjacency(generator, annotation, scaffold, sizes, args):
    z = torch.randn(sizes['batch'],
                            sizes['noise'],
                            sizes['mols'], 
                            sizes['mols']).to(args.device)
    z = z*scaffold
    if args.norm_scaffold:
        scaffold_normalized = normalize_adjacency(scaffold) 
    else: scaffold_normalized = scaffold

 
    adjacency_generated  = generator(annotation, z, 
                                     scaffold_normalized) 
    if args.discretizing_method == 'gumbel-softmax':
        adjacency_generated = gumbel_for_adjacency(adjacency_generated,
                                                   args.device)
        adjacency_generated = adjacency_generated*scaffold
        adjacency_generated = torch.cat([adjacency_generated,(scaffold*(-1))+1], axis=1)
    else:
        adjacency_generated = get_adjacency_one_hot(
                                                adjacency_generated, 
                                                sizes,
                                                scaffold)
    return adjacency_generated

def generate_annotation(generator, batch, sizes, args):
    adjacency = batch[:,:4].sum(1, keepdim=True)
    device = adjacency.device
    z = torch.randn(sizes['batch'], 
                            sizes['mols'], 
                            sizes['noise']).to(args.device)
    if args.cycles:
            cycles = batch[:, 5:]
            adjacency = torch.cat([adjacency, cycles], dim = 1)
    annotation_generated = generator(z, adjacency)
    argmax = torch.argmax(annotation_generated, axis=2)
    annotation = torch.eye(sizes['node_types'])[argmax].to(device)
    return annotation


def normalize_adjacency(adjacency):
    D = adjacency.sum(-1)
    D = torch.diag_embed(D)
    D_sqrtInv = torch.where(D == 0, D, 1/D).sqrt()
    return D_sqrtInv@adjacency@D_sqrtInv

def get_n_order_adjacency(n, adjacency):
    degrees = torch.diag_embed(adjacency.sum(2))
    multi_adjacency = torch.cat([degrees,
                                 adjacency], dim=1)
    adja2 = adjacency@adjacency-degrees
    multi_adjacency = torch.cat([multi_adjacency,
                                 adja2], dim=1)
    degree_minus_one = torch.diag_embed(adjacency.sum(2)-1)
    adja0 = adjacency
    adja1 = adja2
    for i in range(n-3):
        adja2 = adjacency@adja2
        adja2 = adja2 - degree_minus_one@adja0      
        adja0 = adja1
        adja1 = adja2
        multi_adjacency = torch.cat([multi_adjacency,
                                 adja2], dim=1)
    return multi_adjacency


def adj_to_graph(adj, node_flags=None):
    G = nx.from_numpy_matrix(adj)
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))
    if G.number_of_nodes() < 1:
        G.add_node(1)
    return G

def set_path(args):
    if args.run_on == 'heg':
        if args.dataset == 'qm9':
            data_folder = Path('/home/yoann/datasets/qm9/')
        elif args.dataset == 'zinc':
            data_folder = Path('/home/yoann/datasets/ZINC/')
        else:
            raise Exception('Dataset not supported yet')
    elif args.run_on == 'asus':
        if args.dataset == 'qm9':
            data_folder = Path('/home/yo1/datasets/qm9/')
        elif args.dataset == 'zinc':
            data_folder = Path('/home/y01/datasets/ZINC/')
        else:
            raise Exception('Dataset not supported yet')
    elif args.run_on == 'hpc':
        data_folder = Path('/home/users/b/boget3/data/')
    else:
        raise Exception('Computer not declared!')
        
    if args.dataset == 'qm9':
        if args.dataset_version == 'cheat':
            data_file = 'qm9_graphs_excludeCharged_kekul_noH_noFormalCharge_132040.pt'
        elif args.dataset_version == 'arom':
            data_file = 'qm9_graphs_arom_noH_noFormalCharge_133885.pt'
        elif args.dataset_version == 'full':
            data_file = 'qm9_graphs_arom_H_133885.pt'
    elif args.dataset == 'zinc':
        if args.dataset_version == 'cheat':
            data_file = 'ZINC_graphs_excludeCharged_kekul_noH_noFormalCharge_168130.pt'
        elif args.dataset_version == 'arom':
            data_file = 'ZINC_graphs_arom_noH_noFormalCharge_168130.pt'
        elif args.dataset_version == 'full':
            data_file = 'ZINC_graphs_arom_H_168130.pt'
    if args.cycles:
        data_file = Path(f'{data_file[:-3]}_with_cycle.pt')
    
    data_file = Path(data_file)    
    data_path = data_folder / data_file 
    return data_path    



def get_cycles(adj, device):
    'ng_k is the number of non-simple closed path of length k' 
    if adj.shape[0]!=1:
        adj = adj.squeeze()
    adj = adj.type(torch.FloatTensor).to(device)
    degree  = adj.sum(-2)
    degree_matrix = degree.diag_embed(dim1=-2, dim2=-1)
    degree_minus_one = (degree-1).diag_embed(dim1=-2, dim2=-1)
    adj_2 = adj@adj
    adj_3 = adj@adj_2
    cycles_3 = torch.diagonal(adj_3, dim1=-2, dim2 = -1) /2
    cycles_3_matrix = cycles_3.diag_embed(dim1=-2, dim2=-1)

    '------------------------------------'
    'closed path of length 4'
    '------------------------------------'
    adj_4 = adj@adj_3
    
    degree2 = adj_2.sum(1)-degree
    ng_4 = degree**2 + degree2
    
    cycles_4 = torch.diagonal(adj_4, dim1=-2, dim2 = -1)
    cycles_4 = (cycles_4 - ng_4)/2

    '------------------------------------'
    'closed path of length 5'
    '------------------------------------'
    adj_5 = adj@adj_4
    'Only inside the triangle'
    a = 2*5*cycles_3
    'From the triangle, add external back and forth'
    'adja*adja2 = nodes in a triangles'

    b = 2*( ((adj*adj_2)@adj - 2*cycles_3_matrix - (adj*adj_2)).sum(-1)\
      + 2 *cycles_3*(degree-2))
    'From outside the triangle, in, round and back'
    c = 2*((adj@cycles_3_matrix) - cycles_3_matrix*2).sum(-1)
    ng5 = a + b + c
    cycles_5 = torch.diagonal(adj_5, dim1=-2, dim2 = -1)
    cycles_5 = (cycles_5 - ng5)/2
    for i in range(adj.shape[0]):
        if cycles_5[i].sum()% 5 != 0:
            print(cycles_5[i].sum(),cycles_5[i].sum() % 5)
            print(cycles_5[i])
            assert cycles_5[i].sum() % 5 == 0, 'Ca merde'
    
    
    '------------------------------------'
    'closed path of length 6'
    '------------------------------------'
    
    adj_6 = adj@adj_5
    
    'First degree'
    deg1 = degree**3
    
    'second degree'
    'degree 2: 2nd back and forth, plus back and forth with neighbors'
    deg2 = (degree2*degree)*2
    
    'degree 2: forth - 2 back and forth - back'

    deg2b = (adj@((degree-1)**2).unsqueeze(-1)).squeeze()
    
    'degree 3 back and forth (omit degree3 in a square)'
    degree3 = adj_3 - (adj * adj_3 + 2*cycles_3_matrix)
    degree3 = degree3.sum(-1)
       
    'triangles'
    '2 loops triangles'
    triangles = 4 * cycles_3**2
    
   
    '1 loop in one triangle, the other in another (some are counted twice)'
    multi_triangle2 = (adj@cycles_3_matrix*adj*adj_2-adj*adj_2).sum(-1)*2

    triangles = triangles + multi_triangle2
    
    'Squares'
    adj_2_without_loops = adj_2*((torch.eye(adj.shape[-1]).to(device)-1)*-1)
    squares_opposite = choose(adj_2_without_loops, 2, device)   
    in_squares = squares_opposite.sum(-1)
    #is_adj_2_in_square = (adj_2_without_loops > 1)*1.
    #squares_adj = adj@is_adj_2_in_square * adj
    squares_adj = (adj*adj_3) - adj@degree_matrix - degree_minus_one@adj

    square_paths = 2*6*in_squares    
    in_square_matrix = in_squares.diag_embed(dim1=-2, dim2=-1)
    
    all_in_squares = squares_opposite + in_square_matrix + squares_adj
    
    all_in_squares = squares_opposite + in_square_matrix + squares_adj
    
    
    square_path_out = (all_in_squares@adj-2*all_in_squares).sum(-1)*2
    is_insquare = (in_squares>0).type(torch.FloatTensor).to(device)
    adj2in_squares = (((is_insquare)*degree-2) * in_squares )*2

    
    'out path square'
    out_path_square = ((adj@in_square_matrix-squares_adj).sum(-1))*2

    'Miss or double counted'
    
    'adjust degree3'
    adjust = (squares_adj).sum(-2)


    'In triangles and square'
    squares_opposite_connected = squares_opposite*adj*adj_2
    counted_twice1 = 4*squares_opposite_connected.sum(-1)
    is_adj_in_square = ((adj*adj_2)>0).type(torch.FloatTensor).to(device)
    counted_twice2 = (adj@(squares_opposite_connected/2)* is_adj_in_square).sum(-1) 
    counted_twice = counted_twice1 + counted_twice2
    
    ng6 = deg1+ deg2 + deg2b + degree3 + triangles +\
          square_paths + square_path_out+out_path_square \
              + adj2in_squares + adjust - counted_twice
    
    
    cycles_6 = torch.diagonal(adj_6, dim1=-2, dim2 = -1)
    cycles_6 = (cycles_6 - ng6)/2
    return torch.cat((degree.unsqueeze(-1), 
                      cycles_3.unsqueeze(-1),
                      cycles_4.unsqueeze(-1), 
                      cycles_5.unsqueeze(-1),
                      cycles_6.unsqueeze(-1)), dim=-1)
def choose(n, k, device):
    k = torch.Tensor([k]).to(device)
    log_coef = ((n + 1).lgamma() - (k + 1).lgamma() - ((n - k) + 1).lgamma())
    return log_coef.exp()
        

def get_symetric_sparse_adj(edge_index, value,
                               method = 'transposed_avg'):
    if method == 'transposed_avg':
        value_sparse = SparseTensor(row = edge_index[0], 
                           col = edge_index[1], 
                           value = value)
        _, _, value_t = value_sparse.t().coo()
        return (value + value_t)/2
    elif method == 'keep_upper_tri':
        row = edge_index[0, edge_index[0]<edge_index[1]]
        col = edge_index[1, edge_index[0]<edge_index[1]]
        value= value[edge_index[0]<edge_index[1]]       
        _, _, out = SparseTensor(row = torch.cat((row, col), dim=0), 
                                 col = torch.cat((col, row), dim=0),
                    value = torch.cat((value, value), dim=0)).coo()
        return out
    else: raise NotImplementedError('method for symetrizing adjacency not implemented')
    
        