#!/usr/bin/env python
import os
import sys
import glob
import time
import pickle
import traceback
import multiprocessing

import numpy as np

from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import rdmolfiles
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

###########################################
# 1) Configuration
###########################################

# A minimal set of standard residues for chain A
RESIDUE_LIST = [
    "ALA","ARG","ASN","ASP","CYS","GLU","GLN","GLY","HIS",
    "ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"
]

# Peptide dictionary for single-letter codes (Chain B)
THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLU':'E','GLN':'Q','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}

# Arbitrary limit for peptide length
MAX_PEP_LEN = 100

# Define interaction types if you still want a “pocket_cond” dimension
INTERACTION_TYPES = ["pipi","anion","cation","hbd","hba","hydro"]

# For demonstration, we treat chain A’s atoms as “pocket atoms” 
# and chain B’s as “ligand atoms.” No explicit interface extraction.
CHAIN_PROTEIN = 'A'
CHAIN_PEPTIDE = 'B'

###########################################
# 2) Helpers: one-hot and mock interactions
###########################################
def one_hot_residue_20(resname):
    """
    Returns a one-hot vector of length len(RESIDUE_LIST)+1 
    for chain-A residues. The +1 slot is for 'UNK'.
    """
    if resname not in RESIDUE_LIST:
        idx = len(RESIDUE_LIST)  # 'UNK' index
    else:
        idx = RESIDUE_LIST.index(resname)
    vec = np.zeros(len(RESIDUE_LIST)+1, dtype=np.float32)
    vec[idx] = 1.0
    return vec

def get_mock_pocket_cond(num_atoms):
    """
    For demonstration, creates a (num_atoms x (len(INTERACTION_TYPES)+1)) array,
    all 'none' interactions. 
    If you need real interaction detection, plug in your PLIP-based code here.
    """
    mat = np.zeros((num_atoms, len(INTERACTION_TYPES)+1), dtype=np.float32)
    # last index is 'none'
    mat[:, -1] = 1.0
    return mat

def minimal_peptide_scaffold(seq_str):
    """
    Example 'scaffold': same length as your peptide,
    with each residue replaced by 'G'.
    """
    if not seq_str:
        return ""
    return "G"*len(seq_str)

def peptide_smiles(seq_str):
    """
    Converts a one-letter peptide sequence into an RDKit Mol (MolFromFASTA)
    and returns the canonical SMILES. Returns "" if it fails.
    """
    try:
        mol = rdmolfiles.MolFromFASTA(seq_str)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except:
        return ""

###########################################
# 3) Main DataProcessor class
###########################################
class PeptideDataProcessor:
    """
    1) Iterates over all PDB files in data_dir.
    2) For each PDB: parse chain A (protein) and chain B (peptide).
    3) Build arrays for:
       - ligand_type (chain B, per-residue or per-atom),
       - ligand_coord,
       - pocket_type (chain A),
       - pocket_coord,
       - adjacency for chain B,
       - plus the new fields: (pep_seq, pep_smiles, pep_scaffold).
    4) Save final result as a tuple in PKL.
    """

    def __init__(self, data_dir, save_dir, pattern="*.pdb"):
        self.data_dir = data_dir
        self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.pdb_files = sorted(glob.glob(os.path.join(self.data_dir, pattern)))
        self.num_data = len(self.pdb_files)

    def run(self, idx):
        pdb_fn = self.pdb_files[idx]
        base_name = os.path.splitext(os.path.basename(pdb_fn))[0]
        try:
            data = self._process_one_pdb(pdb_fn)
            if data is not None:
                out_fn = os.path.join(self.save_dir, base_name + ".pkl")
                with open(out_fn, "wb") as f:
                    pickle.dump(data, f)
            else:
                print(f"Skipping {pdb_fn} => No data extracted.")
        except Exception as e:
            print(f"Error processing {pdb_fn}:\n{traceback.format_exc()}")

    def _process_one_pdb(self, pdb_file):
        """
        Returns a 14-element tuple (similar to your old pipeline) with 
        the new fields for peptide SMILES, sequence, and scaffold.
        """
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("complex", pdb_file)

        # If either chain is missing, skip
        model = structure[0]
        if CHAIN_PROTEIN not in model or CHAIN_PEPTIDE not in model:
            return None

        chainA = model[CHAIN_PROTEIN]  # protein
        chainB = model[CHAIN_PEPTIDE]  # peptide

        # 3A-1) Extract the peptide sequence from chain B
        pep_residues = list(chainB.get_residues())
        pep_seq_list = []
        for residue in pep_residues:
            r3 = residue.get_resname().upper().strip()
            one_letter = THREE_TO_ONE.get(r3, 'X')  # fallback 'X'
            pep_seq_list.append(one_letter)
        pep_seq = "".join(pep_seq_list)
        if len(pep_seq) == 0:
            return None
        if len(pep_seq) > MAX_PEP_LEN:
            # skip if it’s too large
            return None

        # 3A-2) Build ligand_type, ligand_coord from chain B
        # We'll do it at residue-level: one entry per residue (using CA coords).
        ligand_type_list = []
        ligand_coord_list = []
        for residue in pep_residues:
            # one-hot of 20 + 1 (like 'UNK'), but you could also do side-chain-level
            # if you prefer each heavy atom. 
            # For demonstration we do one vector per residue:
            rname = residue.get_resname().upper().strip()
            oh = one_hot_residue_20(rname)
            ligand_type_list.append(oh)

            # Use alpha carbon or fallback to the first heavy atom
            if "CA" in residue:
                at = residue["CA"]
                ligand_coord_list.append(at.coord)
            else:
                # fallback:
                coords = None
                for atom in residue:
                    if atom.element != 'H':
                        coords = atom.coord
                        break
                if coords is None:
                    coords = np.zeros(3)
                ligand_coord_list.append(coords)

        ligand_type  = np.array(ligand_type_list,  dtype=np.float32)  # shape (Nres, 21)
        ligand_coord = np.array(ligand_coord_list, dtype=np.float32)  # shape (Nres, 3)
        ligand_n = ligand_type.shape[0]

        # 3A-3) Build adjacency for chain B (e.g., link consecutive residues)
        ligand_adj = np.zeros((ligand_n, ligand_n), dtype=np.int64)
        for i in range(ligand_n - 1):
            ligand_adj[i,   i+1] = 1
            ligand_adj[i+1, i  ] = 1

        # 3B-1) Build pocket_type, pocket_coord from chain A
        prot_residues = list(chainA.get_residues())
        pocket_type_list = []
        pocket_coord_list = []
        for residue in prot_residues:
            rname = residue.get_resname().upper().strip()
            oh = one_hot_residue_20(rname)
            pocket_type_list.append(oh)

            # Use alpha carbon or fallback
            if "CA" in residue:
                pocket_coord_list.append(residue["CA"].coord)
            else:
                coords = None
                for atom in residue:
                    if atom.element != 'H':
                        coords = atom.coord
                        break
                if coords is None:
                    coords = np.zeros(3)
                pocket_coord_list.append(coords)

        pocket_type  = np.array(pocket_type_list,  dtype=np.float32)
        pocket_coord = np.array(pocket_coord_list, dtype=np.float32)

        # 3B-2) For demonstration, fill pocket_cond with "none"
        # If you want actual interactions, call your PLIP code or so.
        pocket_cond = get_mock_pocket_cond(pocket_type.shape[0])

        # 3C) Create new fields: pep_seq, pep_smi, pep_scaffold
        pep_smi = peptide_smiles(pep_seq)            # e.g., "NCC(=O)..." 
        pep_scf = minimal_peptide_scaffold(pep_seq)  # e.g., "GGG..."

        # 3D) Center-of-mass shift if you want, or skip
        # For demonstration, let's compute CoM and *optionally* shift everything:
        if ligand_coord.shape[0] > 0:
            center_of_mass = np.mean(ligand_coord, axis=0)
        else:
            center_of_mass = np.zeros(3)
        # SHIFT if desired:
        ligand_coord -= center_of_mass
        pocket_coord -= center_of_mass

        # 3E) Build final data tuple. 
        # Adapting your old 13-tuple structure plus new fields if desired:
        # 
        #   0) ligand_type
        #   1) ligand_coord
        #   2) pocket_type
        #   3) pocket_coord
        #   4) ligand_adj
        #   5) pep_seq_str
        #   6) pep_smi
        #   7) pep_scf
        #   8) ligand_n
        #   9) pocket_cond
        #  10) center_of_mass
        # (You can rearrange or add fields as needed.)
        data_tuple = (
            ligand_type,      # (0)
            ligand_coord,     # (1)
            pocket_type,      # (2)
            pocket_coord,     # (3)
            ligand_adj,       # (4)
            pep_seq,          # (5)
            pep_smi,          # (6)
            pep_scf,          # (7)
            ligand_n,         # (8) number of residues
            pocket_cond,      # (9)
            center_of_mass    # (10)
        )

        return data_tuple

###########################################
# 4) Main script
###########################################
def main():
    if len(sys.argv) < 4:
        print("Usage: python revised_preprocessor.py <DATA_DIR> <SAVE_DIR> <NCPU>")
        sys.exit(1)

    data_dir = sys.argv[1]
    save_dir = sys.argv[2]
    ncpu = int(sys.argv[3])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Optionally ask to remove old .pkl files
    if len(os.listdir(save_dir))>0:
        token = input(f"Remove files in {save_dir}? (y/n) ")
        if token.lower()=='y':
            for fn in os.listdir(save_dir):
                os.remove(os.path.join(save_dir, fn))

    processor = PeptideDataProcessor(data_dir, save_dir)

    print("Total PDB files found:", processor.num_data)
    time.sleep(1)

    pool = multiprocessing.Pool(ncpu)
    r = pool.map_async(processor.run, range(processor.num_data))
    r.wait()
    pool.close()
    pool.join()

    # Final summary
    out_count = len(os.listdir(save_dir))
    print(f"Processed {out_count} PKLs saved in {save_dir}.")

if __name__ == "__main__":
    main()
