#!/usr/bin/env python

"""
md_preprocessor.py

Creates a PKL file per cluster from MD-simulated protein-peptide folders.
Each PKL contains:
  - 3D structure data (chain A=protein, chain B=peptide)
  - MD metrics (docking score, binding energy, per-residue RMSDs, etc.)
  - Peptide SMILES (pep_smi).

'pep_scf' is removed to avoid duplication.
"""

import os
import sys
import csv
import argparse
import pickle
import numpy as np

from Bio.PDB import PDBParser

# RDKit imports for SMILES generation
from rdkit import Chem
from rdkit.Chem import rdmolfiles

# 0) Helper: generate full-peptide SMILES
def peptide_smiles(seq_str):
    """
    Converts a one-letter-coded peptide sequence into an RDKit Mol via MolFromFASTA,
    then returns the canonical SMILES. Returns "" if it fails.
    """
    try:
        mol = rdmolfiles.MolFromFASTA(seq_str)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except:
        return ""

# 1) Residue definitions & one-hot
AMINO_ACIDS_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}
RESIDUE_LIST = [
    "ALA","ARG","ASN","ASP","CYS","GLU","GLN","GLY","HIS",
    "ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"
]
RESIDUE_LIST_LEN = len(RESIDUE_LIST)  # 20

def one_hot_residue_20(resname):
    """
    Returns a one-hot vector of length 21 (the 20 standard AAs + 1 for 'UNK').
    """
    vec = np.zeros(RESIDUE_LIST_LEN + 1, dtype=np.float32)
    if resname in RESIDUE_LIST:
        idx = RESIDUE_LIST.index(resname)
    else:
        idx = RESIDUE_LIST_LEN  # 'UNK'
    vec[idx] = 1.0
    return vec

# 2) Build arrays from the structure
def build_peptide_data(structure):
    """
    Extract minimal arrays: 
      (ligand_type, ligand_coord, pocket_type, pocket_coord, adjacency, 
       pep_seq_str, pep_smi, ligand_n, pocket_cond, center_of_mass)
    """
    model = structure[0]
    chainA = model["A"]  # receptor/protein
    chainB = model["B"]  # peptide

    # -- (1) Peptide info --
    pep_residues = list(chainB.get_residues())
    pep_seq_list = []
    ligand_type_list = []
    ligand_coord_list = []

    for residue in pep_residues:
        rname3 = residue.get_resname().upper().strip()
        rname1 = AMINO_ACIDS_3TO1.get(rname3, "X")
        pep_seq_list.append(rname1)

        oh = one_hot_residue_20(rname3)
        ligand_type_list.append(oh)

        if "CA" in residue:
            ligand_coord_list.append(residue["CA"].coord)
        else:
            # fallback
            fallback_coord = None
            for atom in residue:
                if atom.element != "H":
                    fallback_coord = atom.coord
                    break
            if fallback_coord is None:
                fallback_coord = np.zeros(3)
            ligand_coord_list.append(fallback_coord)

    pep_seq_str  = "".join(pep_seq_list)
    ligand_type  = np.array(ligand_type_list, dtype=np.float32)
    ligand_coord = np.array(ligand_coord_list, dtype=np.float32)
    ligand_n     = ligand_type.shape[0]

    # adjacency: consecutive residues
    adjacency = np.zeros((ligand_n, ligand_n), dtype=np.int64)
    for i in range(ligand_n - 1):
        adjacency[i, i+1] = 1
        adjacency[i+1, i] = 1

    # full SMILES from pep_seq_str
    pep_smi = peptide_smiles(pep_seq_str)

    # -- (2) Protein info --
    prot_residues = list(chainA.get_residues())
    pocket_type_list = []
    pocket_coord_list = []

    for residue in prot_residues:
        rname3 = residue.get_resname().upper().strip()
        oh = one_hot_residue_20(rname3)
        pocket_type_list.append(oh)

        if "CA" in residue:
            pocket_coord_list.append(residue["CA"].coord)
        else:
            fallback_coord = None
            for atom in residue:
                if atom.element != "H":
                    fallback_coord = atom.coord
                    break
            if fallback_coord is None:
                fallback_coord = np.zeros(3)
            pocket_coord_list.append(fallback_coord)

    pocket_type  = np.array(pocket_type_list, dtype=np.float32)
    pocket_coord = np.array(pocket_coord_list, dtype=np.float32)
    pocket_cond  = np.zeros((pocket_type.shape[0], 1), dtype=np.float32)

    # center-of-mass shift for the peptide
    if ligand_coord.shape[0] > 0:
        center_of_mass = np.mean(ligand_coord, axis=0)
    else:
        center_of_mass = np.zeros(3)

    ligand_coord -= center_of_mass
    pocket_coord -= center_of_mass

    data_tuple = (
        ligand_type,      # 0
        ligand_coord,     # 1
        pocket_type,      # 2
        pocket_coord,     # 3
        adjacency,        # 4
        pep_seq_str,      # 5
        pep_smi,          # 6
        ligand_n,         # 7
        pocket_cond,      # 8
        center_of_mass    # 9
    )
    return data_tuple

# 3) Process a single row => dictionary
def process_one_cluster(row, csv_columns, parser):
    cluster_id     = row["cluster_id"]
    pdb_filename   = row["filename"]
    docking_score  = float(row["docking_score"])
    binding_energy = float(row["binding_energy_kcal_per_mol"])
    global_rmsd_nm = float(row["rmsd_nm"])
    mean_dist_nm   = float(row["mean_distance_nm"])

    all_res_rmsd = []
    for i in range(1, 200):
        colname = str(i)
        if colname in row and row[colname].strip():
            val = float(row[colname])
            all_res_rmsd.append(val)
        else:
            break
    md_per_residue_rmsd = np.array(all_res_rmsd, dtype=np.float32)

    structure = parser.get_structure("cluster", pdb_filename)
    data_tuple = build_peptide_data(structure)
    (ligand_type, ligand_coord, pocket_type, pocket_coord,
     ligand_adj, pep_seq_str, pep_smi, ligand_n,
     pocket_cond, center_of_mass) = data_tuple

    pep_len = len(pep_seq_str)
    if md_per_residue_rmsd.shape[0] < pep_len:
        needed = pep_len - md_per_residue_rmsd.shape[0]
        pad_zeros = np.zeros(needed, dtype=np.float32)
        md_per_residue_rmsd = np.concatenate([md_per_residue_rmsd, pad_zeros], axis=0)
    elif md_per_residue_rmsd.shape[0] > pep_len:
        md_per_residue_rmsd = md_per_residue_rmsd[:pep_len]

    data_dict = {
        "ligand_type": ligand_type,
        "ligand_coord": ligand_coord,
        "pocket_type": pocket_type,
        "pocket_coord": pocket_coord,
        "ligand_adj": ligand_adj,
        "pep_seq_str": pep_seq_str,
        "pep_smi": pep_smi,  # full SMILES only
        "ligand_n": ligand_n,
        "pocket_cond": pocket_cond,
        "center_of_mass": center_of_mass,
        # MD metrics
        "docking_score": docking_score,
        "binding_energy_kcal_per_mol": binding_energy,
        "md_global_rmsd_nm": global_rmsd_nm,
        "md_mean_distance_nm": mean_dist_nm,
        "md_per_residue_rmsd": md_per_residue_rmsd
    }

    return data_dict

# 4) Main
def process_md_folders(data_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    parser = PDBParser(QUIET=True)
    subfolders = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]

    for sf in subfolders:
        csv_path = os.path.join(data_dir, sf, "clusters.csv")
        if not os.path.exists(csv_path):
            print(f"[WARN] No clusters.csv found in {sf}, skipping.")
            continue

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            csv_columns = reader.fieldnames if reader.fieldnames else []

            for row in reader:
                data_dict = process_one_cluster(row, csv_columns, parser)
                cluster_id = row["cluster_id"]
                base_name = f"{sf}_cluster{cluster_id}.pkl"
                out_fn = os.path.join(save_dir, base_name)
                with open(out_fn, "wb") as fout:
                    pickle.dump(data_dict, fout)
                print(f"Saved {out_fn}")

    print("Finished processing all subfolders.")

# 5) Entry point
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to main MD directory (with subfolders).")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Where to store PKL files.")
    args = parser.parse_args()

    process_md_folders(args.data_dir, args.save_dir)

if __name__ == "__main__":
    main()
