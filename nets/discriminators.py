#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 09:29:18 2021

@author: Yoann Boget
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as sn
import nets.nn2 as nn_
    
class Disc_edge(nn.Module):
    def __init__(self,
                 nodes_sizes, 
                 edges_sizes, 
                 lin_sizes,
                 activation=nn.ReLU(),
                 end = 'edge_mean'):
        super().__init__()
        self.end  = end
        layers=[]       
        for layer in range(len(nodes_sizes)-1):
            layers.append(nn_.GNN(nodes_sizes[layer], 
                                        edges_sizes[layer], 
                                        edges_sizes[layer+1], 
                                        spectral_norm = True))

        self.layers = nn.Sequential(*layers)
        
        self.linear = nn_.mlp_sn(edges_sizes[layer+1], 
                                 1, [edges_sizes[layer+1], 
                                     edges_sizes[layer+1] ])
        
        
    def forward(self, edge_index, x,  edge_attr, norm = None):
        x = x.float()
        x, edge_attr = self.layers[0](edge_index,
                                      x, 
                                      edge_attr = edge_attr,
                                      norm=norm) 
        
        for layer in self.layers[1:-1]:
            nodes, edge_attr = layer(edge_index, x, 
                                     edge_attr = edge_attr)
        nodes, edges = self.layers[-1](edge_index, x, 
                                       edge_attr = edge_attr)
        if self.end == 'node':
            x = nodes
        elif self.end == 'edge':
            x = edges
        elif self.end == 'node_mean':
            x = nodes.mean(1)
        elif self.end == 'edge_mean':
            x = edges.mean([1,2])
        else:
            raise Exception('Discriminator end not implemented')
        x = self.linear(x)
        return x.squeeze()

class Disc_node(nn.Module):
    def __init__(self,
                 nodes_sizes, 
                 edges_sizes, 
                 lin_sizes,
                 activation=nn.ReLU(),
                 end = 'edge_mean'):
        super().__init__()
        self.end  = end
        layers=[]       
        for layer in range(len(nodes_sizes)-1):
            layers.append(nn_.GNN(nodes_sizes[layer], 
                                        edges_sizes[layer], 
                                        edges_sizes[layer+1], 
                                        spectral_norm = True))

        self.layers = nn.Sequential(*layers)
        
        self.linear = nn_.mlp_sn(edges_sizes[layer+1], 
                                 1, [edges_sizes[layer+1], 
                                     edges_sizes[layer+1] ])
        
        
    def forward(self, edge_index, x, norm = None):
        x = x.float()
        x, edge_attr = self.layers[0](edge_index,
                                      x, 
                                      norm=norm) 
        
        for layer in self.layers[1:-1]:
            nodes, edge_attr = layer(edge_index, x, edge_attr)
        nodes, edges = self.layers[-1](edge_index, x, edge_attr)
        if self.end == 'node':
            x = nodes
        elif self.end == 'edge':
            x = edges
        elif self.end == 'node_mean':
            x = nodes.mean(1)
        elif self.end == 'edge_mean':
            x = edges.mean([1,2])
        else:
            raise Exception('Discriminator end not implemented')
        x = self.linear(x)
        return x
    
class Disc_node_old(nn.Module):
    def __init__(self, 
                 nodes_sizes, 
                 edges_sizes, 
                 lin_sizes,
                 activation=nn.ReLU(), 
                 end = 'edge_mean'
                 ):
        super().__init__()   
        self.end = end
        layers = []
        layers.append( nn_.gnn_no_edge_sn(nodes_sizes[0], 
                                  edges_sizes[0], 
                                  nodes_sizes[1]))
        for layer in range(1, len(nodes_sizes)-1):
            layers.append(nn_.gnn_sn(nodes_sizes[layer], 
                                  edges_sizes[layer], 
                                  nodes_sizes[layer+1]))
            #layers.append(non_linearity)      
        #layers.pop()
        self.layers = nn.Sequential(*layers)
        
        self.linear = nn_.mlp_sn(edges_sizes[layer+1], 
                                 1, [edges_sizes[layer+1], edges_sizes[layer+1] ])
              
    def forward(self, nodes, scaffold):
        scaffold = scaffold.permute(0,2,3,1)
        x, edges = self.layers[0](nodes, scaffold)
        x_new = torch.zeros(x.shape)       
        edges_new = torch.zeros(edges.shape)
        for layer in self.layers[1:-1]:
            x_new, edges_new = layer(x, edges, scaffold)
            x = x + x_new
            edges = edges + edges_new
        x, edges = self.layers[-1](x, edges, scaffold)
        if self.end == 'node':
            x = x
        elif self.end == 'edge':
            x = edges
        elif self.end == 'node_mean':
            x = x.mean(1)
        elif self.end == 'edge_mean':
            x = edges.mean([1,2])
        else:
            raise Exception('Discriminator end not implemented')
        x = self.linear(x)
        if self.end == 'edge':
            x = x*scaffold
        return x.squeeze()



class Disc_edge2(nn.Module):
    def __init__(self,
                 nodes_sizes, 
                 edges_sizes, 
                 lin_sizes,
                 activation=nn.ReLU(),
                 end = 'edge_mean'):
        super().__init__()
        self.end  = end
        layers=[]       
        for layer in range(len(nodes_sizes)-1):
            layers.append(nn_.GNN2(nodes_sizes[layer], 
                                        edges_sizes[layer], 
                                        edges_sizes[layer+1], 
                                        spectral_norm = True))

        self.layers = nn.Sequential(*layers)
        
        self.linear = nn_.mlp_sn(edges_sizes[layer+1], 
                                 1, [edges_sizes[layer+1], 
                                     edges_sizes[layer+1] ])
        
        
    def forward(self, edge_index, x,  edge_attr, norm = None):
        x = x.float()
        x, edge_attr = self.layers[0](edge_index,
                                      x, 
                                      edge_attr = edge_attr,
                                      norm=norm)
        
        
        for layer in self.layers[1:-1]:
            x_, edge_attr_ = layer(edge_index, x, 
                                     edge_attr = edge_attr)
            x = .5*(x + x_)
            edge_attr = .5*(edge_attr+edge_attr_)
        nodes, edges = self.layers[-1](edge_index, x, 
                                       edge_attr = edge_attr)
        if self.end == 'node':
            x = nodes
        elif self.end == 'edge':
            x = edges
        elif self.end == 'node_mean':
            x = nodes.mean(1)
        elif self.end == 'edge_mean':
            x = edges.mean([1,2])
        else:
            raise Exception('Discriminator end not implemented')
        x = self.linear(x)
        return x.squeeze()

class Disc_edge3(nn.Module):
    def __init__(self,
                 nodes_sizes, 
                 edges_sizes, 
                 lin_sizes,
                 activation=nn.ReLU(),
                 end = 'edge_mean'):
        super().__init__()
        self.end  = end
        layers=[]       
        for layer in range(len(nodes_sizes)-1):
            layers.append(nn_.GNN3(nodes_sizes[layer], 
                                        edges_sizes[layer], 
                                        edges_sizes[layer+1], 
                                        spectral_norm = True))

        self.layers = nn.Sequential(*layers)
        
        self.linear = nn_.mlp_sn(edges_sizes[layer+1], 
                                 1, [edges_sizes[layer+1], 
                                     edges_sizes[layer+1] ])
        
        
    def forward(self, edge_index, x,  edge_attr, norm = None):
        x = x.float()
        nodes, edge_attr = self.layers[0](edge_index,
                                      x, 
                                      edge_attr = edge_attr,
                                      norm=norm) 
        
        for layer in self.layers[1:-1]:
            nodes_, edge_attr_ = layer(edge_index, nodes, 
                                     edge_attr = edge_attr)
            nodes = .5*(nodes + nodes_)
            edge_attr = .5*(edge_attr+edge_attr_)
        nodes, edges = self.layers[-1](edge_index, nodes, 
                                       edge_attr = edge_attr)
        if self.end == 'node':
            x = nodes
        elif self.end == 'edge':
            x = edges
        elif self.end == 'node_mean':
            x = nodes.mean(1)
        elif self.end == 'edge_mean':
            x = edges.mean([1,2])
        else:
            raise Exception('Discriminator end not implemented')
        x = self.linear(x)
        return x.squeeze()

class Disc_node2(nn.Module):
    def __init__(self,
                 nodes_sizes, 
                 edges_sizes, 
                 lin_sizes,
                 activation=nn.ReLU(),
                 end = 'edge_mean'):
        super().__init__()
        self.end  = end
        layers=[]       
        for layer in range(len(nodes_sizes)-1):
            layers.append(nn_.GNN2(nodes_sizes[layer], 
                                        edges_sizes[layer], 
                                        edges_sizes[layer+1], 
                                        spectral_norm = True))

        self.layers = nn.Sequential(*layers)
        
        self.linear = nn_.mlp_sn(edges_sizes[layer+1], 
                                 1, [edges_sizes[layer+1], 
                                     edges_sizes[layer+1] ])
        
        
    def forward(self, edge_index, x, norm = None):
        x = x.float()
        nodes, edge_attr = self.layers[0](edge_index,
                                      x, 
                                      norm=norm) 
        
        for layer in self.layers[1:-1]:
            nodes_, edge_attr_ = layer(edge_index, nodes, edge_attr)
            nodes = .5*(nodes + nodes_)
            edge_attr = .5*(edge_attr+edge_attr_)
        nodes_, edges_ = self.layers[-1](edge_index, nodes, edge_attr)
        nodes = nodes + nodes_
        edge_attr = edge_attr + edge_attr_
        if self.end == 'node':
            x = nodes
        elif self.end == 'edge':
            x = edge_attr
        elif self.end == 'node_mean':
            x = nodes.mean(1)
        elif self.end == 'edge_mean':
            x = edge_attr.mean([1,2])
        else:
            raise Exception('Discriminator end not implemented')
        x = self.linear(x)
        return x

class Disc_edge4(nn.Module):
    def __init__(self,
                 nodes_sizes, 
                 edges_sizes, 
                 lin_sizes,
                 activation=nn.ReLU(),
                 end = 'edge_mean'):
        super().__init__()
        self.end  = end
        layers=[]       
        for layer in range(len(nodes_sizes)-1):
            layers.append(nn_.GNN(nodes_sizes[layer], 
                                        edges_sizes[layer], 
                                        edges_sizes[layer+1], 
                                        spectral_norm = True))

        self.layers = nn.Sequential(*layers)
        
        self.linear = nn_.mlp_sn(edges_sizes[layer+1], 
                                 1, [edges_sizes[layer+1], 
                                     edges_sizes[layer+1] ])
        
        
    def forward(self, edge_index, x,  edge_attr, norm = None):
        x = x.float()
        x, edge_attr = self.layers[0](edge_index,
                                      x, 
                                      edge_attr = edge_attr,
                                      norm=norm) 
        
        for layer in self.layers[1:-1]:
            x, edge_attr = layer(edge_index, x, 
                                     edge_attr = edge_attr)
        x, edges = self.layers[-1](edge_index, x, 
                                       edge_attr = edge_attr)
        if self.end == 'node':
            x = x
        elif self.end == 'edge':
            x = edges
        elif self.end == 'node_mean':
            x = x.mean(1)
        elif self.end == 'edge_mean':
            x = edges.mean([1,2])
        else:
            raise Exception('Discriminator end not implemented')
        x = self.linear(x)
        return x.squeeze()

class Disc_edge5(nn.Module):
    def __init__(self,
                 nodes_sizes, 
                 edges_sizes, 
                 lin_sizes,
                 activation=nn.ReLU(),
                 end = 'edge_mean'):
        super().__init__()
        self.end  = end
        layers=[]       
        for layer in range(len(nodes_sizes)-1):
            layers.append(nn_.GNN2(nodes_sizes[layer], 
                                        edges_sizes[layer], 
                                        edges_sizes[layer+1], 
                                        spectral_norm = True))

        self.layers = nn.Sequential(*layers)
        
        self.linear = nn_.mlp_sn(edges_sizes[layer+1], 
                                 1, [edges_sizes[layer+1], 
                                     edges_sizes[layer+1] ])
        
        
    def forward(self, edge_index, x,  edge_attr, norm = None):
        x = x.float()
        x, edge_attr = self.layers[0](edge_index,
                                      x, 
                                      edge_attr = edge_attr,
                                      norm=norm) 
        
        for layer in self.layers[1:-1]:
            x, edge_attr = layer(edge_index, x, 
                                     edge_attr = edge_attr)
        x, edges = self.layers[-1](edge_index, x, 
                                       edge_attr = edge_attr)
        if self.end == 'node':
            x = x
        elif self.end == 'edge':
            x = edges
        elif self.end == 'node_mean':
            x = x.mean(1)
        elif self.end == 'edge_mean':
            x = edges.mean([1,2])
        else:
            raise Exception('Discriminator end not implemented')
        x = self.linear(x)
        return x.squeeze()

class Disc_node3(nn.Module):
    def __init__(self,
                 nodes_sizes, 
                 edges_sizes, 
                 lin_sizes,
                 activation=nn.ReLU(),
                 end = 'edge_mean'):
        super().__init__()
        self.end  = end
        layers=[]       
        for layer in range(len(nodes_sizes)-1):
            layers.append(nn_.GNN2(nodes_sizes[layer], 
                                        edges_sizes[layer], 
                                        edges_sizes[layer+1], 
                                        spectral_norm = True))

        self.layers = nn.Sequential(*layers)
        
        self.linear = nn_.mlp_sn(edges_sizes[layer+1], 
                                 1, [edges_sizes[layer+1], 
                                     edges_sizes[layer+1] ])
        
        
    def forward(self, edge_index, x, norm = None):
        x = x.float()
        nodes, edge_attr = self.layers[0](edge_index,
                                      x, 
                                      norm=norm) 
        
        for layer in self.layers[1:-1]:
            nodes, edge_attr = layer(edge_index, nodes, edge_attr)
           
        nodes, edges = self.layers[-1](edge_index, nodes, edge_attr)
       
        if self.end == 'node':
            x = nodes
        elif self.end == 'edge':
            x = edge_attr
        elif self.end == 'node_mean':
            x = nodes.mean(1)
        elif self.end == 'edge_mean':
            x = edge_attr.mean([1,2])
        else:
            raise Exception('Discriminator end not implemented')
        x = self.linear(x)
        return x