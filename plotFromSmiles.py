#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 14:12:27 2022

@author: yoann
"""

from rdkit import Chem
from rdkit.Chem import Draw


mol1 = ['CC1C(Br)C(Cl)C(F)C(Br)C1Oc1ccc(CN2C(=O)C(=O)C(=O)C(Cl)C2F)nc1',
'CC1C(Cl)C(=O)C(Br)C(=O)C1Nc1ccc(CN2C(=O)C(=N)C(=P)C(F)C2C)cc1',
'CC1C(F)C(Cl)C(F)C(=O)[SH]1Cc1ccc(NN2c(F)c(N)c(C)c(F)c2Cl)cc1',
'O=C1C(=O)C(Cl)C(F)C(=O)C1Cc1ccc(NN2C(=O)c(F)c(F)c(Cl)C2F)cc1',
'CC1C(Sc2ccc(CN3C(=O)C(F)C(=P)C(F)C3C)cc2)C(=O)C(N)C(Cl)C1F',
'C=C1C(Br)C(=O)C(=O)C(=O)C1Oc1ccc(CN2C(C)C(=O)C(F)C(Cl)C2C)nc1']



mol2 = ['CC(C)(C)N(Cc1cccc2c1=NC(=O)P=2)C(=O)CBr',
'CCC(O)N(Cc1occc2nc(=O)nc12)S(C)(C)C',
'CCC(=O)N(Cc1cccc2Oc(=O)nc12)[N+](C)([O-])O',
'Cc1nc2cccc(CN(C(=O)CF)[SH](C)(=O)O)c2n1',
'COC(=O)C(Nc1ncnc2sc(C)cc12)C(C)(C)C',
'CCC(=O)N(Cc1occc2nc(=O)nc12)S(C)(C)C']


mols = mol1 + mol2
ms= []
for mol in mols:
    mol = Chem.MolFromSmiles(mol, sanitize = True)
    #Chem.SanitizeMol(mol, catchErrors=True)
    ms.append(mol)
    img = Draw.MolToImage(mol)
    img.show()
img= Draw.MolsToGridImage(ms,molsPerRow=6)
img.show()