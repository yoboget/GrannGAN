"""Leverage RDKit to translate between molecular graphs / db records / smiles."""


import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import PropertyMol
from rdkit import RDLogger
from pathlib import Path
import pickle
import sys


class MolConstructor():
    def __init__(self, data, meta):
        """Init MolConstructor object.
                Args:
                    data: torch_geometric batch of data
                    meta (dict): Meta information from the dataset
                """
        self.data = data
        self.meta = meta

class MolConstructorOld():
    """Class for constructing molecule representation from graph."""

    def __init__(self, annotations, adjacencies, meta, max_mols=1e7):
        """Init MolConstructor object.

        Args:
            molgraphset (data.MolGraphDataset): dataset with molecular graphs
            max_mols (int): max number of molecules to process
        """
        self.annotations = annotations
        self.adjacencies = adjacencies
        self.meta = meta
        self.max_mols = max_mols
        

    def create_rdkit(self):
        """Create list of mols in RDKit format from graphs.

        Returns:
            mols: list of molecules in RDKit format
            n_errs: number of graphs which were not translated to mols
        """
        # disable printing out sanitization warnings
        RDLogger.DisableLog('rdApp.*')
        print("Translating molecular graphs.")
        mols = []
        n_errs = 0
        annotations = self.annotations
        adjacencies = self.adjacencies
        total_mols = len(self.annotations)
        for mol_idx in range(total_mols):
            if mol_idx == self.max_mols: break
            annot_matrix = annotations[mol_idx, ...]
            adjacency_matrix = adjacencies[mol_idx, ...]
            try:
                mol = self._graph2mol(annot_matrix, adjacency_matrix)
                mols.append(mol)
            except (ValueError):
                n_errs += 1
            if (mol_idx + 1) % 1000 == 0:
                print(f'Processed {mol_idx + 1} graphs and keep going ....')
        print(f'Translated {mol_idx + 1 - n_errs} graphs into RDKit format,',
              f'not translated {n_errs} invalid graphs.')
        self.mols, self.n_errs = mols, n_errs

    def _graph2mol(self, annot_matrix, adjacency_matrix):
        """Convert molecule graph to RDKit molecule.

        Args:
            annot_matrix (max_atoms, n_atom_types) torch.BoolTensor:
                one_hot encodings of atom_types in molecule
            adjacency_matrix (n_bond_types, max_atoms, max_atoms) torch.BoolTensor:
                one_hot encodings of bond types btw atoms in molecule
        Returns:
            (data.Mol): molecule in RDKit format
        """
        mol = Chem.RWMol()  # empty molecule in RDKit format
        max_atoms = self.meta['max_atoms']
        atom_types = self.meta['atom_types']
        bond_types = self.meta['bond_types']
        
        for atom_idx in range(max_atoms):
            atomtype_onehot = annot_matrix[atom_idx, ...]
            atomtype_idx = torch.nonzero((atomtype_onehot == 1), as_tuple=False).item()
            atomtype = atom_types[atomtype_idx]
            if atomtype[0] is None:  # annot_matrix is padded to max_atoms with None types
                continue
            mol.AddAtom(Chem.Atom(atomtype[0]))
            if len(atomtype) > 1:
                mol.GetAtomWithIdx(atom_idx).SetFormalCharge(atomtype[1])
            if len(atomtype) == 3:
                mol.GetAtomWithIdx(atom_idx).SetNumExplicitHs(atomtype[2])
        for atom_idx1 in range(max_atoms):
            # Use upper triangular part of adjacency matrix to extract bonds
            for atom_idx2 in range(atom_idx1+1, max_atoms):
                bondtype_onehot = adjacency_matrix[:, atom_idx1, atom_idx2]
                bondtype_idx = torch.nonzero((bondtype_onehot == 1), as_tuple=False).item()
                bondtype = bond_types[bondtype_idx]
                if bondtype is not None:  # None means non bond
                    mol.AddBond(atom_idx1, atom_idx2, bondtype)
        if self.meta['kekulize'] or len(self.meta['atom_types'][0]) < 3:
            Chem.SanitizeMol(mol, catchErrors=True)

            #Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_SETAROMATICITY)
        # only AddHs if not added already explicitly through atom features
        # if len(self.molgraphset.meta['atom_types']) <= 2:
        #     Chem.AddHs(mol)
        try:
            mol.UpdatePropertyCache()
        except Exception:
            print('Could not update properties')
            pass
        # sops = Chem.SANITIZE_ADJUSTHS&Chem.SANITIZE_SETCONJUGATION&Chem.SANITIZE_SETHYBRIDIZATION&Chem.SANITIZE_SYMMRINGS
        # sops = Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE^Chem.SANITIZE_CLEANUP^Chem.SANITIZE_CLEANUPCHIRALITY
        # Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_KEKULIZE)
        # Chem.SanitizeMol(mol, sanitizeOps=sops)  # catchEerror=False will stop the processing on error
        # Chem.SanitizeMol(mol, sanitizeOps=sops, catchErrors=True)  # catchEerror=False will stop the processing on error

        return mol


class MolChecker():
    """Class for checking validity, unicity and novelty of molecules."""

    def __init__(self, mols, mols_db=[]):
        """Initialize MolChecker state.
        
        Args:
            mols: list of mols in RDKit format (Chem.Mol) to check
            mols_db: list of mols in RDKit format (Chem.Mol) to cross-check with
        """
        self.mols = mols
        self.mols_db = mols_db
        self.valid_mols = []
        self.unique_smiles = []
        self.novel_smiles = []
        n_mols = len(self.mols)
        self.validity = torch.full((n_mols, 5), False, dtype=bool)

    def _valid_chemprops(self, mol):
        """Check validity of chemical properties.
        
        Args:
            mol: mol in RDKit format (Chem.Mol)
        """

        # full monitoring of chemistryproblems for debugging
        # chemProblems = Chem.DetectChemistryProblems(mol)
        # if len(chemProblems):
        #     print(f"Molecule {Chem.MolToSmiles(mol, isomericSmiles=False)}")
        #     for p in chemProblems:
        #         print(p.GetType())
        #         print(p.Message())
        #     valid = False
        # else:
        #     valid = True
        
        valid = False if len(Chem.DetectChemistryProblems(mol)) > 0 else True
        return valid

    def _valid_disconnect(self, mol):
        """Check mol has no disconnected structures.
        
        Args:
            mol: mol in RDKit format (Chem.Mol)
        """

        valid = False if len(Chem.GetMolFrags(mol)) > 1 else True
        return valid

    def _valid_ion(self, mol):
        """Check mol is neutral (not ion with positive or negative charge).
        
        Args:
            mol: mol in RDKit format (Chem.Mol)
        """

        valid = False if Chem.GetFormalCharge(mol) != 0 else True
        return valid

    def _valid_radical(self, mol):
        """Check mol is not a radical.
        
        Args:
            mol: mol in RDKit format (Chem.Mol)
        """

        valid = False if Descriptors.NumRadicalElectrons(mol) > 0 else True
        return valid

    def check_validity(self):
        """Use RDKit to check validity of mols."""

        # disable printing out sanitization warnings
        RDLogger.DisableLog('rdApp.*')
        for mol_idx, mol in enumerate(self.mols):
            self.validity[mol_idx, 0] = (self._valid_chemprops(mol))
            if self.validity[mol_idx, 0] == True:
                self.validity[mol_idx, 1] = (self._valid_disconnect(mol))
                self.validity[mol_idx, 2] = (self._valid_ion(mol))
                self.validity[mol_idx, 3] = (self._valid_radical(mol))

        self.validity[:, -1] = (self.validity.sum(dim=1) == 4)
        self.valid_mols = [mol for idx, mol in enumerate(self.mols) if self.validity[idx, -1]]
        # enable printing out sanitization warnings
        RDLogger.EnableLog('rdApp.*')

    def check_unicity(self):
        """Use RDKit smiles to check unicity of mols.
        RDKit produces unique canonical smiles irrespective of atom indexing in mols.
        """

        if len(self.valid_mols) == 0:
            raise Exception(f'There are no valid molecules. Did you run check_validity() before?')
        smiles = [Chem.MolToSmiles(mol, isomericSmiles=False) for mol in self.valid_mols]
        self.unique_smiles = set(smiles)
        for smile in self.unique_smiles:
            print(smile)

    def check_novelty(self):
        """Use RDKit smiles to check novelty of mols compared to mol_db.
        RDKit produces unique canonical smiles irrespective of atom indexing in mols.
        """

        if len(self.mols_db) == 0:
            raise Exception(f'No mols_db to compare to. Make sure self.mols_db is not empty.')
        if len(self.unique_smiles) == 0:
            raise Exception(f'Set of unique smiles is empty. Did you run check_unicity() before?')

        db_smiles = [Chem.MolToSmiles(mol, isomericSmiles=False) for mol in self.mols_db]
        unique_smiles = set(db_smiles)
        self.novel_smiles = self.unique_smiles - unique_smiles
