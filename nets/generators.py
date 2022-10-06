#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:55:58 2021

@author: Yoann Boget
"""

import torch
import torch.nn as nn
import nets.nn2 as nn_

    
class Gen_edge(nn.Module):
    def __init__(self, 
                 nodes_sizes, 
                 edges_sizes, 
                 activation=nn.ReLU()):
        super().__init__()        
        layers = []  
        for layer in range(len(nodes_sizes)-1):
            layers.append(nn_.GNN(nodes_sizes[layer], 
                                  edges_sizes[layer], 
                                  nodes_sizes[layer+1]))
            
        self.layers = nn.Sequential(*layers)
    def forward(self, edge_index, x, z, norm = None):
        edge_attr = z
        x = x.float()
        for layer in self.layers:
            x, edge_attr = layer(edge_index, x, 
                                 edge_attr = edge_attr, norm = norm)
        return edge_attr
    
class Gen_node(nn.Module):
    def __init__(self, 
                 nodes_sizes, 
                 edges_sizes, 
                 activation=nn.ReLU()):
        super().__init__()        
        layers = []  
        for layer in range(len(nodes_sizes)-1):
            layers.append(nn_.GNN(nodes_sizes[layer], 
                                  edges_sizes[layer], 
                                  nodes_sizes[layer+1]))
            
        self.layers = nn.Sequential(*layers)
    def forward(self, edge_index, z, norm = None):
        x = z
        x, edge_attr = self.layers[0](edge_index, x, norm = norm)
        for layer in self.layers[1:]:
            x, edge_attr = layer(edge_index, x, edge_attr = edge_attr, norm = norm)
        return x

class Gen_edge2(nn.Module):
    def __init__(self, 
                 nodes_sizes, 
                 edges_sizes, 
                 activation=nn.ReLU()):
        super().__init__()        
        layers = []  
        for layer in range(len(nodes_sizes)-1):
            layers.append(nn_.GNN3(nodes_sizes[layer], 
                                  edges_sizes[layer], 
                                  nodes_sizes[layer+1]))
            
        self.layers = nn.Sequential(*layers)
    def forward(self, edge_index, x, z, norm = None):
        edge_attr = z
        x = x.float()
        x, edge_attr = self.layers[0](edge_index, x, 
                                 edge_attr = edge_attr, norm = norm)
        for layer in self.layers[1:-1]:
            x_, edge_attr_ = layer(edge_index, x, 
                                 edge_attr = edge_attr, norm = norm)
            x = .5*(x + x_)
            edge_attr = .5*(edge_attr+edge_attr_)
        x, edge_attr = self.layers[-1](edge_index, x, 
                                 edge_attr = edge_attr, norm = norm)
        return edge_attr


class Gen_node2(nn.Module):
    def __init__(self, 
                 nodes_sizes, 
                 edges_sizes, 
                 activation=nn.ReLU()):
        super().__init__()        
        layers = []  
        for layer in range(len(nodes_sizes)-1):
            layers.append(nn_.GNN2(nodes_sizes[layer], 
                                  edges_sizes[layer], 
                                  nodes_sizes[layer+1]))
            
        self.layers = nn.Sequential(*layers)
    def forward(self, edge_index, z, norm = None):
        x = z
        x, edge_attr = self.layers[0](edge_index, x, norm = norm)
        for layer in self.layers[1:-1]:
            x_, edge_attr_ = layer(edge_index, x, edge_attr = edge_attr, norm = norm)
            x = .5*(x + x_)
            edge_attr = .5*(edge_attr+edge_attr_)
        x, edge_attr = self.layers[-1](edge_index, x, edge_attr = edge_attr, norm = norm)
        return x



class Gen_node_old(nn.Module):
    def __init__(self, 
                 nodes_sizes, 
                 edges_sizes,             
                 activation=nn.ReLU(), 
                 ):
        super().__init__()        
        layers = []
        layers.append( nn_.gnn_no_edge(nodes_sizes[0], 
                                  edges_sizes[0], 
                                  nodes_sizes[1]))
        for layer in range(1, len(nodes_sizes)-1):
            layers.append(nn_.gnn(nodes_sizes[layer], 
                                  edges_sizes[layer], 
                                  nodes_sizes[layer+1]))
            #layers.append(non_linearity)      
        #layers.pop()
        self.layers = nn.Sequential(*layers)
              
    def forward(self, z, scaffold):
        scaffold = scaffold.permute(0,2,3,1)
        x, edges = self.layers[0](z, scaffold)
        x_new = torch.zeros(x.shape)       
        edges_new = torch.zeros(edges.shape)
        for layer in self.layers[1:-1]:
            x_new, edges_new = layer(x, edges, scaffold)
            x = x + x_new
            edges = edges + edges_new
        x, edges = self.layers[-1](x, edges, scaffold)
        return x

class Gen_edge4(nn.Module):
    def __init__(self, 
                 nodes_sizes, 
                 edges_sizes, 
                 activation=nn.ReLU()):
        super().__init__()        
        layers = []  
        for layer in range(len(nodes_sizes)-1):
            layers.append(nn_.GNN2(nodes_sizes[layer], 
                                  edges_sizes[layer], 
                                  nodes_sizes[layer+1]))
            
        self.layers = nn.Sequential(*layers)
    def forward(self, edge_index, x, z, norm = None):
        edge_attr = z
        x = x.float()
        x, edge_attr = self.layers[0](edge_index, x, 
                                 edge_attr = edge_attr, norm = norm)
        for layer in self.layers[1:-1]:
            x_, edge_attr_ = layer(edge_index, x, 
                                 edge_attr = edge_attr, norm = norm)
            x = x + x_
            edge_attr = edge_attr+edge_attr_
        x, edge_attr = self.layers[-1](edge_index, x, 
                                 edge_attr = edge_attr, norm = norm)
        return edge_attr

class Gen_node3(nn.Module):
    def __init__(self, 
                 nodes_sizes, 
                 edges_sizes, 
                 activation=nn.ReLU()):
        super().__init__()        
        layers = []  
        for layer in range(len(nodes_sizes)-1):
            layers.append(nn_.GNN2(nodes_sizes[layer], 
                                  edges_sizes[layer], 
                                  nodes_sizes[layer+1]))
            
        self.layers = nn.Sequential(*layers)
    def forward(self, edge_index, z, norm = None):
        x = z
        x, edge_attr = self.layers[0](edge_index, x, norm = norm)
        for layer in self.layers[1:-1]:
            x_, edge_attr_ = layer(edge_index, x, edge_attr = edge_attr, norm = norm)
            x = x + x_
            edge_attr = edge_attr+edge_attr_
        x, edge_attr = self.layers[-1](edge_index, x, edge_attr = edge_attr, norm = norm)
        return x