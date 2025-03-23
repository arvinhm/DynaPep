#!/usr/bin/env python
import os
import sys
import glob
import time
import pickle
import traceback
import multiprocessing

import numpy as np
import torch

from Bio.PDB import PDBParser

###########################################
# 1) Configuration
###########################################

# Residue list for one-hot
RESIDUE_LIST = [
    "ALA","ARG","ASN","ASP","CYS","GLU","GLN","GLY","HIS",
    "ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"
]
INTERACTION_TYPES = ["pipi", "anion", "cation", "hbd", "hba", "hydro"]

# Token dictionary for the peptide tokens
TOKEN_DICT = {
    'unk': 0, 'pad': 1, 'start': 2, 'eos': 3,
    'A': 4, 'R': 5, 'N': 6, 'D': 7, 'C': 8, 'E': 9,
    'Q': 10, 'G': 11, 'H': 12, 'I': 13, 'L': 14, 'K': 15,
    'M': 16, 'F': 17, 'P': 18, 'S': 19, 'T': 20,
    'W': 21, 'Y': 22, 'V': 23
}

MAX_SEQ_LEN = 50  # maximum for your peptide tokens

###########################################
# 2) Helper Functions
###########################################
def one_hot_residue(resname):
    """
    Returns a one-hot vector of length len(RESIDUE_LIST)+1.
    The +1 slot is for 'UNK'.
    """
    if resname not in RESIDUE_LIST:
        idx = len(RESIDUE_LIST)  # 'UNK'
    else:
        idx = RESIDUE_LIST.index(resname)
    vec = np.zeros(len(RESIDUE_LIST)+1, dtype=np.float32)
    vec[idx] = 1.0
    return vec

# Simple 3-letter -> 1-letter mapping for standard residues
THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLU':'E','GLN':'Q','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}

class CustomTokenizer:
    def __init__(self, token_dict):
        self.token_dict = token_dict
        self.pad_idx = token_dict['pad']
        self.unk_idx = token_dict['unk']
        self.start_idx = token_dict['start']
        self.eos_idx = token_dict['eos']

    def encode(self, sequence, max_length):
        tokens = [self.token_dict.get(r, self.unk_idx) for r in sequence.upper()]
        # Add start/eos
        tokens = [self.start_idx] + tokens + [self.eos_idx]
        actual_length = len(tokens)

        # Truncate
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
            actual_length = max_length
        # Pad
        elif len(tokens) < max_length:
            tokens += [self.pad_idx]*(max_length - len(tokens))

        return torch.tensor(tokens, dtype=torch.long), actual_length

###########################################
# 3) Peptide SMILES + Scaffold
###########################################
from rdkit import Chem
from rdkit.Chem import rdmolfiles

def simple_peptide_smiles(seq_str):
    """
    Tries to create an RDKit Mol from a one-letter peptide sequence using MolFromFASTA,
    then returns canonical SMILES.
    Returns None if it fails.
    """
    if not seq_str:
        return None
    try:
        mol = rdmolfiles.MolFromFASTA(seq_str)
        if mol is None:
            return None
        smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        return smi
    except:
        return None

def minimal_scaffold(seq_str):
    """
    Example 'scaffold': same length, but every residue replaced with 'G'.
    If the user wants a more sophisticated approach (like removing side chains), adapt here.
    """
    return "G" * len(seq_str)

###########################################
# 4) Main Data Processor (13-tuple)
###########################################
class PeptideInterfaceDataProcessor:
    """
    Creates a 13-element tuple that mirrors your old pipeline:
        0) ligand_type
        1) ligand_coord
        2) pocket_type
        3) pocket_coord
        4) ligand_adj
        5) pep_tokens (replacing old placeholder)
        6) pep_smiles (replacing old None at index=6)
        7) ligand_n
        8) scaff_n
        9) pep_seq_str (replacing ligand_smi)
       10) scaffold_smi
       11) pocket_cond
       12) center_of_mass
    """
    def __init__(self, data_dir, save_dir, chain_protein='A', chain_peptide='B', pattern="*.pdb"):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.chain_protein = chain_protein
        self.chain_peptide = chain_peptide

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.pdb_files = sorted(glob.glob(os.path.join(self.data_dir, pattern)))
        self.num_data = len(self.pdb_files)

        self.tokenizer = CustomTokenizer(TOKEN_DICT)

    def run(self, idx):
        pdb_file = self.pdb_files[idx]
        base_name = os.path.splitext(os.path.basename(pdb_file))[0]
        try:
            data_tuple = self._process_single_pdb(pdb_file)
            if data_tuple is None:
                print(f"Skipping {base_name} (data is None).")
                return
            out_fn = os.path.join(self.save_dir, base_name + ".pkl")
            with open(out_fn, "wb") as f:
                pickle.dump(data_tuple, f)

        except Exception as e:
            print(f"Error processing {pdb_file}")
            print(traceback.format_exc())

    def _process_single_pdb(self, pdb_file):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("complex", pdb_file)
        model = structure[0]

        # If chain B doesn't exist => skip
        if self.chain_peptide not in model:
            return None

        # Extract chain B residues
        chainB = model[self.chain_peptide]
        chainB_residues = list(chainB.get_residues())
        if len(chainB_residues)==0:
            return None

        # Build one-letter sequence
        pep_seq = []
        for residue in chainB_residues:
            r3 = residue.get_resname().upper().strip()
            pep_seq.append(THREE_TO_ONE.get(r3, 'X'))  # fallback to X
        pep_seq_str = "".join(pep_seq)

        # Tokenize
        pep_tokens, pep_len = self.tokenizer.encode(pep_seq_str, MAX_SEQ_LEN)

        # Create pep_smiles
        pep_smiles_str = simple_peptide_smiles(pep_seq_str)
        if pep_smiles_str is None:
            pep_smiles_str = ""

        # Create scaffold sequence + smiles
        # For example, replace everything with 'G'
        scaff_seq = minimal_scaffold(pep_seq_str)
        scaff_smi_str = simple_peptide_smiles(scaff_seq)
        if scaff_smi_str is None:
            scaff_smi_str = ""

        # Build ligand_type / ligand_coord for each residue
        ligand_type_list = []
        ligand_coord_list = []
        for residue in chainB_residues:
            rname = residue.get_resname().upper().strip()
            oh = one_hot_residue(rname)
            ligand_type_list.append(oh)

            # Alpha carbon coords if exist
            ca_atom = residue['CA'] if 'CA' in residue else None
            if ca_atom:
                ligand_coord_list.append(ca_atom.coord)
            else:
                # fallback: first heavy atom or 0,0,0
                found_any = False
                for atom in residue:
                    if atom.element != 'H':
                        ligand_coord_list.append(atom.coord)
                        found_any = True
                        break
                if not found_any:
                    ligand_coord_list.append((0.,0.,0.))

        ligand_type  = np.array(ligand_type_list, dtype=np.float32)   # shape=(N, 21)
        ligand_coord = np.array(ligand_coord_list, dtype=np.float32)  # shape=(N, 3)
        ligand_n = ligand_type.shape[0]

        # Build adjacency by linking consecutive residues
        ligand_adj = np.zeros((ligand_n, ligand_n), dtype=np.int64)
        for i in range(ligand_n-1):
            ligand_adj[i, i+1] = 1
            ligand_adj[i+1, i] = 1

        # Build pocket_type / pocket_coord
        pocket_type_list = []
        pocket_coord_list = []
        if self.chain_protein in model:
            chainA = model[self.chain_protein]
            chainA_residues = list(chainA.get_residues())
            for residue in chainA_residues:
                rname = residue.get_resname().upper().strip()
                oh = one_hot_residue(rname)
                pocket_type_list.append(oh)

                ca_atom = residue['CA'] if 'CA' in residue else None
                if ca_atom:
                    pocket_coord_list.append(ca_atom.coord)
                else:
                    # fallback
                    found = False
                    for atom in residue:
                        if atom.element != 'H':
                            pocket_coord_list.append(atom.coord)
                            found=True
                            break
                    if not found:
                        pocket_coord_list.append((0.,0.,0.))

        pocket_type  = np.array(pocket_type_list,  dtype=np.float32)
        pocket_coord = np.array(pocket_coord_list, dtype=np.float32)

        # pocket_cond => fill with "none"
        # shape = (pocket_n, len(INTERACTION_TYPES)+1) => last index = "none"
        if len(pocket_type)>0:
            pcond_list = []
            for _ in range(pocket_type.shape[0]):
                cvec = np.zeros(len(INTERACTION_TYPES)+1, dtype=np.float32)
                cvec[-1] = 1.0  # 'none'
                pcond_list.append(cvec)
            pocket_cond = np.stack(pcond_list, axis=0)
        else:
            pocket_cond = np.zeros((0, len(INTERACTION_TYPES)+1), dtype=np.float32)

        # center_of_mass from ligand_coord
        if ligand_coord.shape[0]>0:
            center_of_mass = np.mean(ligand_coord, axis=0)
        else:
            center_of_mass = np.zeros((3,), dtype=np.float32)

        # scaff_n => length of scaffold
        scaff_n = len(scaff_seq)

        # Final 13-tuple
        # Indices:
        #   0=ligand_type, 1=ligand_coord, 2=pocket_type, 3=pocket_coord,
        #   4=ligand_adj, 5=pep_tokens, 6=pep_smiles_str, 7=ligand_n,
        #   8=scaff_n, 9=pep_seq_str, 10=scaff_smi_str, 11=pocket_cond,
        #   12=center_of_mass
        data_tuple = (
            ligand_type,      # index=0
            ligand_coord,     # index=1
            pocket_type,      # index=2
            pocket_coord,     # index=3
            ligand_adj,       # index=4
            pep_tokens,       # index=5
            pep_smiles_str,   # index=6 (was None)
            ligand_n,         # index=7
            scaff_n,          # index=8
            pep_seq_str,      # index=9 (old ligand_smi)
            scaff_smi_str,    # index=10 (old scaff_smi, now real string)
            pocket_cond,      # index=11
            center_of_mass    # index=12
        )

        return data_tuple

###########################################
# 5) Main Script
###########################################
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python scaffold_pep_preprocessor.py <DATA_DIR> <SAVE_DIR> <NCPU>")
        sys.exit(1)

    data_dir = sys.argv[1]
    save_dir = sys.argv[2]
    ncpu = int(sys.argv[3])

    print(f"Data directory: {data_dir}")
    print(f"Output PKL dir: {save_dir}")
    print(f"Using {ncpu} CPU cores")

    # Optionally remove existing PKLs
    if os.path.exists(save_dir) and len(os.listdir(save_dir))>0:
        token = input(f"Remove files in {save_dir}? (y/n) ")
        if token.lower()=="y":
            os.system(f"rm {save_dir}/*")

    processor = PeptideInterfaceDataProcessor(
        data_dir=data_dir,
        save_dir=save_dir,
        chain_protein='A',  # change if needed
        chain_peptide='B',  # change if needed
        pattern="*.pdb"
    )
    print("Found", processor.num_data, "PDB files.")
    time.sleep(0.5)

    pool = multiprocessing.Pool(ncpu)
    r = pool.map_async(processor.run, range(processor.num_data))
    r.wait()
    pool.close()
    pool.join()

    print("Done. PKLs in", save_dir)
