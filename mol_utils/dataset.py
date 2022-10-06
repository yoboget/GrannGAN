import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_dense_adj

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import PropertyMol
from rdkit import RDLogger

from pathlib import Path
import pickle
from mol_utils.cycles import get_cycles

import sys

class MyMolDataset(InMemoryDataset):

    def __init__(self, dataset, root, transform=None, pre_transform=None, pre_filter=None, 
                 filetype='smi', max_mols=1e7, kekulize=False, countH=True,
                 formal_charge=True, exclude_arom=False, exclude_charged=False):
        'Precondition: a folder "root" sould contain a file .csv with the name\
            of the dataset (qm9.csv or zinc.csv)'
        
        self.dataset = dataset
        if dataset == 'qm9':
            self.filename = 'qm9.csv'
        elif dataset == 'zinc':
            self.filename = 'zinc.csv'
        self.filetype = filetype
        self.max_mols=max_mols
        self.kekulize=kekulize
        self.countH=countH
        self.formal_charge=formal_charge
        self.exclude_arom=exclude_arom
        self.exclude_charged=exclude_charged
        
        self.meta = {'kekulize': self.kekulize, 
                      'total_mols': 0, 
                      'exclude_arom': self.exclude_arom,
                      'exclude_charged': self.exclude_charged,
                      'max_atoms': 0, 'atom_types': [], 'bond_types': []}
        
        self.processing = {'max_mols': self.max_mols, 
                           'molsnone': [], 'molserr': [],
                           'molsradical': [], 'molsdisconnect': [], 
                           'molsions': [], 'molsaromatic': [], 
                           'molscharged': [], 'n_unique': 0}
        
        self.strings = {'smiles': [], 'names': []}
        
        self.root = root
        fpath = Path(root)
        fspec = dataset
        fspec = fspec + '_excludeArom' if self.meta['exclude_arom'] else fspec
        fspec = fspec + '_excludeCharged' if self.meta['exclude_charged'] else fspec
        fspec = fspec + '_kekul' if self.meta['kekulize'] else fspec + '_arom'
        fspec = fspec + '_H' if countH else fspec + '_noH'
        fspec = fspec + '_noFormalCharge' if not formal_charge else fspec + '_FormalCharge'
        self.proc_dir = str(fpath / (fspec))
        print(self.proc_dir)
        
        self.atom_features = ["GetAtomicNum"]
        if formal_charge:
            self.atom_features.append("GetFormalCharge")
        if countH:
            self.atom_features.append("GetTotalNumHs")
        
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices, self.meta = torch.load(self.processed_paths[0])
        
       
  
    @property
    def processed_dir(self):
        return self.proc_dir
    
    @property
    def raw_dir(self):
        return self.root
    
    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        return 'data.pt'


    def process(self):
        # Read data into huge `Data` list.
        
        data_list = self.create_graphs()
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
       
        data, slices = self.collate(data_list)
        torch.save((data, slices, self.meta), self.processed_paths[0])
 
    
    def create_graphs(self):
        """Create molecule graphs from molecular file.

        Returns:
            (MolGraphset): data from molecular db translated to graphs
        """
        # disable printing out sanitization warnings

        #RDLogger.DisableLog('rdApp.*')
        print(f"Extracting molecular data from {self.raw_file_names}.")
        print('-'*60)
        print(self.raw_file_names)



        supplier = Chem.SmilesMolSupplier(self.raw_dir + '/' + self.raw_file_names,
                                          delimiter=',', smilesColumn=0, nameColumn=-1)

        # extract summary meta info about content of molecular file
        mol_list = self._get_meta(self.processing['max_mols'], supplier)
        
        
        data_list = []
        # populate graph annotation and adjacency tensors, and create molgraph dateset
        for mol_idx, mol in enumerate(mol_list):
            annot_matrix, edge_index, edge_attr = self._mol2graph(mol)
            if edge_index.nelement() != 0:
                adj = to_dense_adj(edge_index)
                cycles = get_cycles(adj).squeeze()
            else:
                cycles = torch.zeros(1, 5)

            data = Data(x=annot_matrix,
                        edge_index=edge_index, 
                        edge_attr=edge_attr, 
                        cycles=cycles
                        )
            data_list.append(data)
        print(mol_idx) 

        origout, origerr = sys.stdout, sys.stderr
        logfile = open(Path(self.processed_dir)/('mol_extraction.log'), 'w')
        sys.stdout = sys.err = logfile
        print(f"Dropped {self.processing['molsnone']} records evaluated as None by RDKit.")
        print(f"Dropped {len(self.processing['molserr'])} mols with ChemistryProblems in RDKit.")
        print(f"Dropped {len(self.processing['molsradical'])} mols with radical electrons.")
        print(f"Dropped {len(self.processing['molsdisconnect'])} mols with disconnected fragments.")
        print(f"Dropped {len(self.processing['molsions'])} mols with negative or positive charge (ions).")
        if self.meta['exclude_arom']:
            print(f"Dropped {len(self.processing['molsaromatic'])} aromatic mols.")
        else:
            print(f"Included {len(self.processing['molsaromatic'])} aromatic mols.")
        if self.meta['exclude_charged']:
            print(f"Dropped {len(self.processing['molscharged'])} mols with charged atoms.")
        else:
            print(f"Included {len(self.processing['molscharged'])} mols with charged atoms.")
        print(f"After all these, extracted {mol_idx+1} valid molecules of which {self.processing['n_unique']} are unique.")
        print('-'*60)
        print(self.meta)
        sys.stdout, sys.stderr = origout, origerr
 
        RDLogger.EnableLog('rdApp.*')
        return data_list
    
    def _get_meta(self, max_mols, supplier):
        """Extract summary meta info about content of molecular file.

        Args:
            max_mols (int): max number of molecules to process
            supplier (Chem.Supplier): supplier for molecules from sdf or smi files
        """
        print('Extracting meta information')
        mol_idx, max_atoms = 0, 0
        atom_types, bond_types = set(), set()
        mol_list = []
        for mol_count, mol in enumerate(supplier):
            mol_charged = False
            if self._check_validity(mol, mol_count, track=True):
                self.strings['smiles'].append(Chem.MolToSmiles(mol, isomericSmiles=False))
                if self.meta['kekulize']:
                    Chem.Kekulize(mol)
                max_atoms = max(max_atoms, mol.GetNumAtoms())
                try:
                    name = mol.GetProp('_Name')
                except KeyError:
                    name = ''
                finally:
                    self.strings['names'].append(name)
                    
                
                for atom in mol.GetAtoms():
                    if atom.GetFormalCharge() != 0:
                        mol_charged = True
                        if not self.meta['exclude_charged']:
                            atom_types.add(tuple(getattr(atom, feat)() for feat in self.atom_features))
                    else:
                        atom_types.add(tuple(getattr(atom, feat)() for feat in self.atom_features))
                if mol_charged and self.meta['exclude_charged']:
                    pass
                else:
                    for bond in mol.GetBonds():
                        bond_types.add(bond.GetBondType())
                    mol_list.append(PropertyMol.PropertyMol(mol))
                    mol_idx += 1
                    if mol_idx % 1000 == 0:
                        print(mol_idx)
            if mol_charged:
                self.processing['molscharged'].append(Chem.MolToSmiles(mol, isomericSmiles=False))
            if mol_idx == max_mols: break
        self.meta['total_mols'] = mol_idx
        self.meta['max_atoms'] = max_atoms
        self.meta['atom_types'] = list(atom_types)
        self.meta['bond_types'] = list(bond_types)
        self.processing['n_unique'] = len(set(self.strings['smiles']))
        
        self.rdkitfile = Path(self.processed_dir) / ('data_rdkit.pickle')
        pickle.dump(mol_list, open(self.rdkitfile, 'wb'))
        print(f"Stored {mol_idx} molecules as RDkit mol objects in {self.rdkitfile}.")
        print('-'*60)
        return mol_list
    
    def _check_validity(self, mol, mol_count, track=False):
        """Multiple checks of RDKit validity.

        Args:
            mol (Chem.Mol): molecule in RDKit format
            track: keep track of invalid molecules
        """
        valid = True
        if mol is None:
            valid = False
            if track:
                self.processing['molsnone'].append(mol)
        else:
            if len(Chem.DetectChemistryProblems(mol)) > 0:
                valid = False
                if track:
                    self.processing['molserr'].append(Chem.MolToSmiles(mol, isomericSmiles=False))
            elif len(Chem.GetMolFrags(mol)) > 1:
                valid = False
                if track:
                    self.processing['molsdisconnect'].append(Chem.MolToSmiles(mol, isomericSmiles=False))
            elif Chem.GetFormalCharge(mol) != 0:
                valid = False
                if track:
                    self.processing['molsions'].append(Chem.MolToSmiles(mol, isomericSmiles=False))
            elif Descriptors.NumRadicalElectrons(mol) > 0:
                valid = False
                if track:
                    self.processing['molsradical'].append(Chem.MolToSmiles(mol, isomericSmiles=False))
            elif len(Chem.Mol.GetAromaticAtoms(mol)) > 1:
                if self.meta['exclude_arom']:
                    valid = False
                if track:
                    self.processing['molsaromatic'].append(Chem.MolToSmiles(mol, isomericSmiles=False))
        return valid
    
    def _mol2graph(self, mol):
        """Convert RDKit molecule to graph.

        Args:
            mol (Chem.Mol): molecule in RDKit format
        Returns:
            (max_atoms, n_atom_types) torch.BoolTensor:
                one_hot encodings of atom_types in molecule
            (n_bond_types, max_atoms, max_atoms) torch.BoolTensor:
                one_hot encodings of bond types btw atoms in molecule
        """
        atom_types = self.meta['atom_types']
        bond_types = self.meta['bond_types']
        atom_list = []
        for atom in mol.GetAtoms():
            atom_type = tuple(getattr(atom, feat)() for feat in self.atom_features)
            atom_list.append(atom_types.index(atom_type))
        annot_matrix = torch.eye(len(atom_types))[atom_list]

        start_atoms = [bond.GetBeginAtomIdx() for bond in mol.GetBonds()]
        end_atoms = [bond.GetEndAtomIdx() for bond in mol.GetBonds()]
        edge_index = torch.LongTensor([start_atoms+end_atoms, end_atoms+start_atoms])
        edge_list = [bond_types.index(bond.GetBondType()) for bond in mol.GetBonds()]
        edge_list = edge_list + edge_list
        edge_attr = torch.eye(len(bond_types))[edge_list]
        return annot_matrix, edge_index, edge_attr
    
