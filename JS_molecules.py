#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 09:54:52 2022

@author: yoann
"""

import numpy as np

'''
	real	model
Single	0.79	0.809
Double	0.074	0.085
Triple	0.029	0.027
Aromatic	0.103	0.078
		
		
Qm9		
	real	model
N	0.1146	0.1004
O	0.1551	0.1647
C	0.7048	0.7156
F	0.0028	0

'''

'''
qm9 - ours
node

pmf1 = np.asarray([0.1146, 0.1551, 0.7048, 0.0028]) 
pmf2 = np.asarray([0.1004, 0.1647, 0.7156, 0.00000000000000001]) 
'''

'''
qm9 - ours
edges

pmf1 = np.asarray([0.79, 0.074, 0.029, 0.1030]) 
pmf2 = np.asarray([0.7927491664886475, 0.07437869161367416, 0.02959427237510681, 0.10327785462141037] ) 
'''

'''
qm9 - gaf
nodes

pmf1 = np.asarray([0.75859592, 0.09704642, 0.11006374 ,0.03429392])
pmf2 = np.asarray([0.71886354, 0.11868234, 0.15963999, 0.00281412])
'''

'''
qm9 - gaf
edges

pmf1 = np.asarray([0.8683556914329529, 0.09397251158952713, 0.03767180070281029])
pmf2 = np.asarray([0.8554023504257202, 0.1149941235780716, 0.02960350550711155])
'''

'''
zinc - ours
node

pmf1 = np.asarray([0.000000000001, 0.0002, 0.1023, 0.0023, 0.0148, 0.0196, 0.7336,0.0082, 0.119]) 
pmf2 = np.asarray([0.0015, 0.0000000001, 0.0992, 0.0058, 0.0129, 0.0197, 0.7339, 0.0072, 0.1198]) 
'''

'''
zinc - ours
edges

pmf1 = np.asarray([0.4871, 0.0674, 0.0028, 0.4427]) 
pmf2 = np.asarray([0.5067, 0.0553, 0.0032, 0.4348] ) 
'''

'''
zinc - gaf
nodes

pmf1 = np.asarray([0.011, 0.0096 ,0.0716, 0.0084, 0.022, 0.014, 0.7595, 0.013, 0.091])
pmf2 = np.asarray([0.000000001, 0.0002, 0.1021, 0.0025 ,0.015, 0.02, 0.7333, 0.0083, 0.1186])
'''

'''
zinc - gaf
edges
'''
pmf1 = np.asarray([0.728, 0.2691, 0.0029])
pmf2 = np.asarray([0.7742, 0.2174, 0.008])



m = .5*(pmf1+pmf2)
KL1 = (pmf1 * np.log2( pmf1/m))
KL2 = (pmf2 * np.log2( pmf2/m))
JSD = .5*(KL1.sum() + KL2.sum())

print(JSD)