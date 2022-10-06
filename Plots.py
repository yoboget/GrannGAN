#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 17:55:39 2022

@author: yo1
"""


import matplotlib.pyplot as plt
import numpy as np
from bokeh.palettes import Viridis
 

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def set_axis(dataset, training_step, ax, x):
    ax.set_ylabel(f'{training_step} distribution by type')
    ax.set_title(f'Proportion of edges - {dataset}')
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    return ax

def plot_distribution(labels, real, gen, dataset, training_step):
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, real, width, label='Real', color=Viridis[3][0])
    rects2 = ax.bar(x + width/2, gen, width, label='Generated', color=Viridis[3][2])
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    set_axis(dataset, training_step, ax, x)
    
    #autolabel(rects1, ax)
    #autolabel(rects2, ax)
    fig.tight_layout()
    plt.show()
    
training_step = 'Edge'
dataset = 'QM9'
labels = ['Single', 'Double', 'Tripple', 'Aromatic']
edge_real = [0.792, 0.074, 0.029, 0.103]
edge_gen = [0.809,  0.085,  0.027,  0.078]
plot_distribution(labels, edge_real, edge_gen, dataset, training_step)



training_step = 'Edge'
dataset = 'ZINC'
edge_real = [0.4871, 0.0674, 0.0028, 0.4427] 
edge_gen = [0.5383, 0.0616, 0., 0.4001]
plot_distribution(labels, edge_real, edge_gen, dataset, training_step)

'''
GRAPH_AF
'''
training_step = 'Atom'
dataset = 'ZINC'
[(6,), (15,), (53,), (8,), (17,), (7,), (16,), (35,), (9,), (None,)]
labels = ['C', 'P', 'I', 'O', 'Cl', 'N', 'S', 'Br', 'F', 'Empty']
edge_real = [0.3613041639328003, 2.0833333110203966e-05,  0.00010833333362825215, 0.050320833921432495, 0.004110416863113642, 0.05844583362340927, 0.01003333367407322, 0.0012083332985639572, 0.0071749999187886715, 0.5072728991508484]
edge_gen = [0.2824108898639679, 0.00400037644430995, 0.003576807212084532, 0.02663780190050602, 0.004800451919436455, 0.033995356410741806, 0.0051142070442438126, 0.003121862420812249, 0.0081890057772398, 0.6281532645225525]
plot_distribution(labels, edge_real, edge_gen, dataset, training_step)

training_step = 'Atom'
dataset = 'ZINC'
labels = ['C', 'P', 'I', 'O', 'Cl', 'N', 'S', 'Br', 'F']

edge_real = np.asarray(edge_real)[:-1]/(1- np.asarray(edge_real)[-1])

edge_gen = np.asarray(edge_gen)[:-1]/(1- np.asarray(edge_gen)[-1])
print(edge_real, edge_gen)
plot_distribution(labels, edge_real, edge_gen, dataset, training_step)


training_step = 'Edge'
dataset = 'ZINC'
labels = ['Single', 'Double', 'Tripple']
edge_real = [0.7280115485191345, 0.26912909746170044, 0.0028593617025762796]
edge_gen = [0.7742504477500916, 0.217464417219162, 0.008285139687359333]
plot_distribution(labels, edge_real, edge_gen, dataset, training_step)





