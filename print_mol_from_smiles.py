#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:41:18 2022

@author: yoann
"""
from rdkit import Chem
from rdkit.Chem import Draw



smiles_path = '/home/yoann/datasets/qm9/qm9.smi'
suppl = Chem.SmilesMolSupplier(smiles_path, delimiter=',', smilesColumn=0, 
                                              nameColumn=-1)

ms = [x for x in suppl if x is not None]
img=Draw.MolsToGridImage(ms[:8],molsPerRow=4,
                     subImgSize=(200,200),
                     legends=[Chem.MolToSmiles(x) for x in ms[:8]])    
img.show()