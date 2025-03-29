#!/usr/bin/env python
import os
import subprocess
import torch
import torch.nn as nn
from rdkit import Chem

# For peptides/proteins, we use 20 standard amino acids plus UNK.
NUM_LIGAND_ATOM_TYPES = 21
NUM_POCKET_ATOM_TYPES = 21
INTERACTION_TYPES = ["pipi", "anion", "cation", "hbd", "hba", "hydro"]
NUM_INTERACTION_TYPES = len(INTERACTION_TYPES) + 1

ATOM_TYPES = ["C", "N", "O", "F", "P", "S", "Cl", "Br"]
ATOM_NUM = [6, 7, 8, 9, 15, 16, 17, 35]
ATOM_MASS = [12.0, 14.0, 16.0, 19.0, 31.0, 32.1, 35.5, 79.9]

def find_free_port():
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def get_abs_path(path):
    return os.path.expanduser(path)

def dic_to_device(dic, device):
    for key, value in dic.items():
        if isinstance(value, torch.Tensor):
            dic[key] = value.to(device)
    return dic

def get_cuda_visible_devices(num_gpus: int) -> str:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU drivers and CUDA installation.")
    available_gpus = torch.cuda.device_count()
    if available_gpus < num_gpus:
        raise RuntimeError(f"Available GPUs ({available_gpus}) are less than required ({num_gpus})!")
    # Return a comma-separated list of GPU IDs
    return ",".join(str(i) for i in range(num_gpus))

def stat_cuda(msg):
    print("--", msg)
    print("allocated: %dM, max allocated: %dM, cached: %dM, max cached: %dM" % (
        torch.cuda.memory_allocated() / 1024 / 1024,
        torch.cuda.max_memory_allocated() / 1024 / 1024,
        torch.cuda.memory_reserved() / 1024 / 1024,
        torch.cuda.max_memory_reserved() / 1024 / 1024,
    ))

def load_save_file(model, use_gpu, save_file=None):
    if save_file is None:
        return model
    else:
        if not use_gpu:
            save_file_dict = torch.load(save_file, map_location="cpu")
        else:
            save_file_dict = torch.load(save_file)
        new_save_file_dict = {k.replace("module.", ""): v for k, v in save_file_dict.items()}
        model.load_state_dict(new_save_file_dict, strict=False)
        return model

def initialize_model(model, use_gpu=True, load_save_file_path=False):
    if load_save_file_path:
        model = load_save_file(model, use_gpu, load_save_file_path)
    else:
        for param in model.parameters():
            if not param.requires_grad:
                continue
            if param.dim() == 1:
                nn.init.zeros_(param)
            else:
                nn.init.xavier_uniform_(param)
    return model

def write_result(fn, true_dict, pred_dict):
    lines = []
    for k, v in true_dict.items():
        assert pred_dict.get(k) is not None
        lines.append(f"{k}\t{float(v):.3f}\t{float(pred_dict[k]):.3f}\n")
    with open(fn, "w") as w:
        w.writelines(lines)

def write_xyz(types, coords, msg="", fn=None, is_onehot=True):
    import numpy as np
    if isinstance(types, torch.Tensor):
        types = np.array(types.detach().cpu())
    if isinstance(coords, torch.Tensor):
        coords = np.array(coords.detach().cpu())
    xyz = f"{coords.shape[0]}\n{msg}\n"
    for i in range(coords.shape[0]):
        if is_onehot:
            atom_type = ATOM_TYPES[np.argmax(types[i])] if types.shape[-1] == len(ATOM_TYPES) else "X"
        else:
            atom_type = types[i]
        xyz += f"{atom_type}\t{coords[i][0]}\t{coords[i][1]}\t{coords[i][2]}\n"
    if fn is not None:
        with open(fn, "w") as w:
            w.write(xyz.strip())
    return xyz.strip()

def xyz_to_sdf(xyz_fn, sdf_fn):
    os.system(f"obabel '{xyz_fn}' -O '{sdf_fn}'")

def xyz_to_pdb(xyz_fn, pdb_fn):
    os.system(f"obabel '{xyz_fn}' -O '{pdb_fn}'")

def text_filling(text, char="#", num_char=80):
    if len(text) > num_char:
        return text
    elif len(text) % 2 == 0:
        side = (num_char - len(text)) // 2 - 1
        return char * side + f" {text} " + char * side
    else:
        left = (num_char - len(text)) // 2
        right = (num_char - len(text)) // 2 - 1
        return char * left + f" {text} " + char * right

if __name__ == "__main__":
    pass
