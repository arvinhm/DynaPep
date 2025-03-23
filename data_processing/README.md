Here we exatcrt informauton from PDB interface files and datastructure is as below as exmple

            ligand_type,      # index=0
            ligand_coord,     # index=1
            pocket_type,      # index=2
            pocket_coord,     # index=3
            ligand_adj,       # index=4
            pep_tokens,       # index=5
            pep_smiles_str,   # index=6
            ligand_n,         # index=7
            scaff_n,          # index=8
            pep_seq_str,      # index=9
            scaff_smi_str,    # index=10
            pocket_cond,      # index=11
            center_of_mass    # index=12

to run

         python peptide_preprocessor_all_in_one.py ./interface ./output_dir 8
         python clean.py


