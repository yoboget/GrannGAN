import torch
from mol_utils.dataset import MyMolDataset
from torch_geometric.loader import DataLoader
from mol_utils.rdkitfuncs import MolConstructor
from rdkit.Chem.Draw import MolToImage
import matplotlib.pyplot as plt
import rdkit
from rdkit.Chem import Draw
from rdkit import Chem

def plot_mols(dataset, n_mols = 1, random = True):
    dataset.create_rdkit()
    for i in range(100):
        try:
            image = MolToImage(dataset.mols[i], size=(300, 300), kekulize=False, 
                               wedgeBonds=True, fitImage=False, 
                               options=None, canvas=None)
            image.save(f'./img/test_{i}.png')
        except: print(f'Image {i} not plotted')

meta = {'kekulize': False, 'total_mols': 168130, 'exclude_arom': False, 
        'exclude_charged': True, 'max_atoms': 38, 
        'atom_types': [(15,), (53,), (8,), (35,), (9,), (16,), (6,), (17,), (7,), (None,)], 
        'bond_types': [rdkit.Chem.rdchem.BondType(1), rdkit.Chem.rdchem.BondType(2), rdkit.Chem.rdchem.BondType(3), rdkit.Chem.rdchem.BondType(12), None]}

adj = torch.load('adjacencies.pt')[:100]
ann = torch.load('annotations.pt')[:100]


dataset = MolConstructor(ann.type(torch.long),
                          adj.type(torch.long), meta)

dataset.create_rdkit()
plot_mols(dataset)

img=Draw.MolsToGridImage(dataset.mols[:8],molsPerRow=4,
                     subImgSize=(200,200),
                     legends=[Chem.MolToSmiles(x) for x in dataset.mols[:8]])
img.show() 
    
