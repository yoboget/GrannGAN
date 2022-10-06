"""Utility script for validity/ unicity/ novelty metrics for graphs in MolGraphDataset."""

import torch
from mol_utils import rdkitfuncs
#import rdkitfuncs
from pathlib import Path


def getmetrics_dsetpickle(molgraphset, picklefile_db):
    """Get metrics for graphs in molgrpahset compared to graphs in picklefile_db.

    Args:
        molgraphset: (data.MolGraphDataset) containing annotations and adjacencies
        picklefile_db: path to pickle file containing graphs of the molecular db
    Returns:
        stats_total: n_graphs = n_mols + n_errs (n_errs = num of graphs that couldn't be translated to RDKit)
        valid: score = num / n_graphs
        unique: score = num / valid['num']
        novel: score = num / valid['num']
    Example:
        getmetrics_dsetpickle(generatedgraphset, '/home/magda/datasets/MoleculeData/gdb9/gdb9_graphs_kekul_1000.pickle'):
    """

    print('Creating RDKit mols from graphs in molgraphset.')
    molconstructor_new = rdkitfuncs.MolConstructor(molgraphset)
    molconstructor_new.create_rdkit()

    print('-'*60)
    print('Creating RDKit mols from graphs in picklefile_db.')
    molgraphset = torch.load(Path(picklefile_db))
    molconstructor_db = rdkitfuncs.MolConstructor(molgraphset)
    molconstructor_db.create_rdkit()

    stats_total, valid, unique, novel = getmetrics(molconstructor_new.mols, molconstructor_new.n_errs, molconstructor_db.mols)
    return stats_total, valid, unique, novel


def getmetrics_picklepickle(picklefile, picklefile_db):
    """Get metrics for graphs in piclefile compared to graphs in picklefile_db.

    Args:
        picklefile: path to pickle file containing graphs to check
        picklefile_db: path to pickle file containing graphs of the molecular db
    Returns:
        stats_total: n_graphs = n_mols + n_errs (n_errs = num of graphs that couldn't be translated to RDKit)
        valid: score = num / n_graphs
        unique: score = num / valid['num']
        novel: score = num / valid['num']
    Example:
        getmetrics_dsetpickle('/home/magda/datasets/MoleculeData/gdb9/gdb9_graphs_generated.pickle', '/home/magda/datasets/MoleculeData/gdb9/gdb9_graphs_kekul_1000.pickle'):
    """

    print('Creating RDKit mols from graphs in picklefile.')
    molgraphset = torch.load(Path(picklefile))
    molconstructor_new = rdkitfuncs.MolConstructor(molgraphset)
    molconstructor_new.create_rdkit()

    print('-'*60)
    print('Creating RDKit mols from graphs in picklefile_db.')
    molgraphset = torch.load(Path(picklefile_db))
    molconstructor_db = rdkitfuncs.MolConstructor(molgraphset)
    molconstructor_db.create_rdkit()

    stats_total, valid, unique, novel = getmetrics(molconstructor_new.mols, molconstructor_new.n_errs, molconstructor_db.mols)
    return stats_total, valid, unique, novel


def getmetrics_dsetdset(molgraphset, molgraphset_db):
    """Get metrics for graphs in molgrpahset compared to graphs in molgraphset_db.

    Args:
        molgraphset: (data.MolGraphDataset) containing annotations and adjacencies
        molgraphset_db: (data.MolGraphDataset) containing graphs of the molecular db
    Returns:
        stats_total: n_graphs = n_mols + n_errs (n_errs = num of graphs that couldn't be translated to RDKit)
        valid: score = num / n_graphs
        unique: score = num / valid['num']
        novel: score = num / valid['num']
    Example:
        getmetrics_dsetpickle(generatedgraphset, molgraphset_db):
    """

    print('Creating RDKit mols from graphs in molgraphset.')
    molconstructor_new = rdkitfuncs.MolConstructor(molgraphset)
    molconstructor_new.create_rdkit()

    print('-'*60)
    print('Creating RDKit mols from graphs in molgraphset_db.')
    molconstructor_db = rdkitfuncs.MolConstructor(molgraphset_db)
    molconstructor_db.create_rdkit()

    stats_total, valid, unique, novel = getmetrics(molconstructor_new.mols, molconstructor_new.n_errs, molconstructor_db.mols)
    return stats_total, valid, unique, novel


def getmetrics_dset(molgraphset):
    """Get metrics for graphs in molgrpahset alone (no novelty)

    Args:
        molgraphset: (data.MolGraphDataset) containing annotations and adjacencies
    Returns:
        stats_total: n_graphs = n_mols + n_errs (n_errs = num of graphs that couldn't be translated to RDKit)
        valid: score = num / n_graphs
        unique: score = num / valid['num']
    Example:
        getmetrics_dsetpickle(generatedgraphset, molgraphset_db):
    """

    print('Creating RDKit mols from graphs in molgraphset.')
    molconstructor_new = rdkitfuncs.MolConstructor(molgraphset)
    molconstructor_new.create_rdkit()

    stats_total, valid, unique, novel = getmetrics(molconstructor_new.mols, molconstructor_new.n_errs)
    return stats_total, valid, unique, novel


def getmetrics(mols, n_errs, mols_db=None):
    """Get metrics for mols as compared to mols_db.
    The novelty checking can be skipped by not providing any mols_db.

    Normally, you do not call this func directly but it is used by all the other getmetrics funcs.
    Args:
        mols: list of mols in RDKit format (Chem.Mol)
        n_errs: number of errors on construction of mols from graphs
        mols_db: list of mols in RDKit format (Chem.Mol) to compare the novelty to
    Returns:
        stats_total: n_graphs = n_mols + n_errs (n_errs = num of graphs that couldn't be translated to RDKit)
        valid: score = num / n_graphs
        unique: score = num / valid['num']
        novel: score = num / valid['num']
    Example:
        getmetrics_dsetpickle(generatedgraphset, '/home/magda/datasets/MoleculeData/gdb9/gdb9_graphs_kekul_1000.pickle'):
    """

    stats_total = {'total_graphs': len(mols) + n_errs}
    stats_total['not_constructed'] = n_errs
    stats_total['constructed_mols'] = len(mols)

    molchecker = rdkitfuncs.MolChecker(mols, mols_db)
    print('-'*60)
    print('Checking additional validity problems of molecules in molgraphset.')
    molchecker.check_validity()
    chemprobs_pass, disconnect_pass, ion_pass, radical_pass, all_pass = molchecker.validity.sum(dim=0)
    print(f"From total {stats_total['total_graphs']} generated graphs constructed {stats_total['constructed_mols']} mols.")
    print(f"Out of these:")
    print(f" {' '*5} {chemprobs_pass} passed chemical property checks")
    print(f" {' '*5} {disconnect_pass} are not disconnected")
    print(f" {' '*5} {ion_pass} are not ion molecules")
    print(f" {' '*5} {ion_pass} are not radical molecules")
    valid = {'num': all_pass.item()}
    valid['score'] = valid['num'] / stats_total['total_graphs']
    print(f"After all these, there are {valid['num']} molecules. Validity score: {valid['score']}")

    print('-'*60)
    if valid['num'] > 0:
        print('Checking unicity of molecules in molgraphset.')
        molchecker.check_unicity()
        for smile in molchecker.unique_smiles:
            print(smile)
        unique = {'num': len(molchecker.unique_smiles)}
        unique['score'] = unique['num'] / valid['num']
        print(f"From the valid mols there are {unique['num']} unique. Unicity score: {unique['score']}")
    else: unique = None
    if mols_db:
        print('-'*60)
        print('Checking novelty of molecules in molgraphset when compared to mol_db.')
        molchecker.check_novelty()
        novel = {'num': len(molchecker.novel_smiles)}
        novel['score'] = novel['num'] / unique['num']
        print(f"From the valid unique mols there are {novel['num']} novel. Novelty score: {novel['score']}")
    else:
        novel = None

    return stats_total, valid, unique, novel
