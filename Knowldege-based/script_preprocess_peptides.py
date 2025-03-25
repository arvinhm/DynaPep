#!/usr/bin/env python3
"""
This script reads a folder of PDB files and extracts chain B (peptide) from each.
Chain A is assumed to be the receptor.
For each PDB file, we extract:
  - The one–letter peptide sequence (using a standard 3–letter to 1–letter mapping)
  - (Optionally) the coordinates of chain B atoms.
The processed data are saved as a list of dictionaries and split into train/val/test.
"""

import os
import glob
import pickle
import random
from Bio import PDB
from tqdm import tqdm

# Define a dictionary for converting three-letter codes to one-letter codes.
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
    'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    # Extend if needed (e.g., for non–standard amino acids)
}

def get_chain_sequence(chain):
    """Extracts the one-letter amino acid sequence from a chain.
       Also returns a list of CA coordinates (if available)."""
    seq = ""
    coords = []
    for res in chain:
        # We check if the residue id is a standard residue
        if res.get_id()[0] != " ":
            continue
        resname = res.get_resname().upper()
        aa = THREE_TO_ONE.get(resname, 'X')  # Use X for unknown
        seq += aa
        # Get CA coordinate (if present)
        if 'CA' in res:
            coords.append(res['CA'].get_coord())
        else:
            coords.append([0.0, 0.0, 0.0])
    return seq, coords

def process_pdb_file(pdb_file):
    """Process one PDB file to extract chain A (receptor) and chain B (peptide) info.
       Returns a dictionary with file name, receptor (chain A) info and peptide (chain B) info."""
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(os.path.basename(pdb_file), pdb_file)
    except Exception as e:
        print(f"Error processing {pdb_file}: {e}")
        return None

    # Assume we want the first model only
    model = structure[0]
    # Get chain A and chain B (if present)
    chain_A = None
    chain_B = None
    for chain in model:
        if chain.get_id() == 'A':
            chain_A = chain
        elif chain.get_id() == 'B':
            chain_B = chain

    if chain_A is None or chain_B is None:
        print(f"Skipping {pdb_file}: missing chain A or chain B")
        return None

    peptide_seq, peptide_coords = get_chain_sequence(chain_B)
    # Optionally, you may also extract receptor coordinates from chain A
    receptor_seq, receptor_coords = get_chain_sequence(chain_A)

    return {
        'pdb_file': pdb_file,
        'receptor_seq': receptor_seq,
        'receptor_coords': receptor_coords,
        'peptide_seq': peptide_seq,
        'peptide_coords': peptide_coords
    }

def split_dataset(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Randomly split data into train, validation and test sets."""
    random.shuffle(data)
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_data = data[:n_train]
    val_data = data[n_train:n_train+n_val]
    test_data = data[n_train+n_val:]
    return train_data, val_data, test_data

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess PDB files for peptide generation")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing pdb files")
    parser.add_argument("--output_folder", type=str, default="preprocessed_data", help="Folder to save splits")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    pdb_files = glob.glob(os.path.join(args.input_folder, "*.pdb"))
    print(f"Found {len(pdb_files)} pdb files in {args.input_folder}")

    processed = []
    for pdb_file in tqdm(pdb_files, desc="Processing pdb files"):
        result = process_pdb_file(pdb_file)
        if result is not None:
            processed.append(result)

    if not processed:
        print("No valid pdb files processed!")
        return

    train_data, val_data, test_data = split_dataset(processed)

    with open(os.path.join(args.output_folder, "train_data.pkl"), "wb") as f:
        pickle.dump(train_data, f)
    with open(os.path.join(args.output_folder, "val_data.pkl"), "wb") as f:
        pickle.dump(val_data, f)
    with open(os.path.join(args.output_folder, "test_data.pkl"), "wb") as f:
        pickle.dump(test_data, f)

    print(f"Dataset split: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test examples.")

if __name__ == '__main__':
    main()
