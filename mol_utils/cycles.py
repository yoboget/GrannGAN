#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:36:47 2022

@author: yo1
"""
import torch


def get_cycles(adj):
    'ng_k is the number of non-simple closed path of length k' 
   
    adj = adj.type(torch.FloatTensor)
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
    
    if cycles_5.sum()% 5 != 0:

        assert cycles_5.sum() % 5 == 0, 'Ca merde'
    
    
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
    adj_2_without_loops = adj_2*((torch.eye(adj.shape[-1])-1)*-1)
    squares_opposite = choose(adj_2_without_loops, 2)   
    in_squares = squares_opposite.sum(-1)
    #is_adj_2_in_square = (adj_2_without_loops > 1)*1.
    #squares_adj = adj@is_adj_2_in_square * adj
    squares_adj = (adj*adj_3) - adj@degree_matrix - degree_minus_one@adj

    square_paths = 2*6*in_squares    
    in_square_matrix = in_squares.diag_embed(dim1=-2, dim2=-1)
    
    all_in_squares = squares_opposite + in_square_matrix + squares_adj
    
    all_in_squares = squares_opposite + in_square_matrix + squares_adj
    
    
    square_path_out = (all_in_squares@adj-2*all_in_squares).sum(-1)*2
    is_insquare = (in_squares>0).type(torch.FloatTensor)
    adj2in_squares = (((is_insquare)*degree-2) * in_squares )*2

    
    'out path square'
    out_path_square = ((adj@in_square_matrix-squares_adj).sum(-1))*2

    'Miss or double counted'
    
    'adjust degree3'
    adjust = (squares_adj).sum(-2)


    'In triangles and square'
    squares_opposite_connected = squares_opposite*adj*adj_2
    counted_twice1 = 4*squares_opposite_connected.sum(-1)
    is_adj_in_square = ((adj*adj_2)>0).type(torch.FloatTensor)
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
def choose(n, k):
    k = torch.Tensor([k])
    log_coef = ((n + 1).lgamma() - (k + 1).lgamma() - ((n - k) + 1).lgamma())
    return log_coef.exp()
        
