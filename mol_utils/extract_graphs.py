"""Utility script for extracting graphs for molecular dbs,
creating MolGraphDataset, and pickling it to disk."""

import argparse
from utils import rdkitfuncs
#import rdkitfuncs
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "data_path", type=str,
    default='/home/yoann/datasets/qm9',
    help=('Full path to data file. ',
          'Graph pickle will be stored in the same folder.')
)
parser.add_argument(
    "filetype", choices=["smi", "sdf"],
    default="smi",
    help=("Type of molecular file. smi: SMILES or sdf: SDF molfile")
)
parser.add_argument(
    "--max_mols", type=int, default=1e8,
    help='Max number of molecules to extract / use (eg for testing).'
)
parser.add_argument(
    "--kekulize", action="store_true",
    help='Aromatic bonds shall be kekulized.'
)
parser.add_argument(
    "--noH", action="store_true",
    help='Do not include hydrogens counts into atom_types.'
)
parser.add_argument(
    "--noFormalCharge", action="store_true",
    help='Do not include the formal charge into atom_types.'
)
parser.add_argument(
    "--cheat", action="store_true",
    help='Simplified data for rdkit cheating. Same as setting kekulized, noH, no_formal_charge.'
)
parser.add_argument(
    "--exclude_arom", action="store_true",
    help='Exclude aromatic molecules from dataset.'
)
parser.add_argument(
    "--exclude_charged", action="store_true",
    help='Exclude mols that have some atoms with nonzero formal charge.'
)


args = parser.parse_args()
if args.cheat:
    args.kekulize = True
    args.noH = True
    args.noFormalCharge = True

# initiate extractor
extractor = rdkitfuncs.MolExtractor(Path(args.data_path), args.filetype, max_mols=args.max_mols,
                                    kekulize=args.kekulize, countH=(not args.noH),
                                    formal_charge=(not args.noFormalCharge),
                                    exclude_arom=args.exclude_arom,
                                    exclude_charged=args.exclude_charged,
                                    save=True)

# extract graphs from file
data_set = extractor.create_graphs()
