#!/usr/bin/env python
# md_train.py
# ------------------------------------------------------------
# This is DynaPep created and developed by Arvin Haj-Mirzaian 
# ------------------------------------------------------------

import os
import sys
import glob
import time
import math
import copy
import traceback
import socket
import subprocess
import argparse
from math import pi as PI
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
from torch_geometric.data import Data, Batch, HeteroData
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm


# ############################################################################
# ###########################    arguments    ################################
# ############################################################################
torch.cuda.empty_cache() # to clear memory

def train_args_parser():
    parser = argparse.ArgumentParser()

    # DEFAULT SETTINGS
    parser.add_argument("--world_size", help="world size", type=int, default=1)
    parser.add_argument("--distributed", help="distributed", action="store_true", default=True)
    parser.add_argument("--autocast", help="autocast", action="store_true", default=False)
    parser.add_argument("--num_workers", help="number of workers", type=int, default=1)
    parser.add_argument("--batch_size", help="batch_size", type=int, default=64)

    # DIRECTORY SETTINGS
    parser.add_argument("--save_dir", help="save directory", type=str)
    parser.add_argument("--data_dir", help="data directory", type=str)
    parser.add_argument("--key_dir", help="key directory", type=str, default=None)

    # DATASET SETTINGS
    parser.add_argument("--k", help="k for k-NN parameter", type=int)

    # MODEL SETTINGS
    parser.add_argument("--num_layers", help="num layers", type=int, default=10)
    parser.add_argument("--num_dense_layers", help="num dense layers", type=int, default=5)
    parser.add_argument(
        "--num_ligand_atom_feature",
        help="ligand atom features",
        type=int,
        default=21,
    )
    parser.add_argument(
        "--num_pocket_atom_feature",
        help="pocket atom features",
        type=int,
        default=21,
    )
    parser.add_argument("--num_hidden_feature", help="num hidden features", type=int, default=512)
    parser.add_argument("--num_latent_feature", help="num latent features", type=int, default=512)
    parser.add_argument("--gamma1", type=float, default=1e1)
    parser.add_argument("--gamma2", type=float, default=5e1)
    parser.add_argument(
        "--dist_one_hot_param1",
        help="dist. one-hot param for representation",
        type=int,
        nargs="+",
        default=[0, 10, 25],
    )
    parser.add_argument(
        "--dist_one_hot_param2",
        help="dist. one-hot param for next distance",
        type=int,
        nargs="+",
        default=[0, 15, 300],
    )
    parser.add_argument("--conditional", help="conditional", action="store_true")
    parser.add_argument(
        "--num_cond_feature",
        help="num condition features",
        type=int,
        default=7,
    )
    parser.add_argument("--ssl", help="semi-supervised learning", action="store_true")

    # TRAINING SETTINGS
    parser.add_argument("--num_epochs", help="num epochs", type=int, default=100)
    parser.add_argument("--lr", help="lr", type=float, default=1e-3)
    parser.add_argument("--lr_decay", help="lr_decay", type=float, default=0.8)
    parser.add_argument("--lr_tolerance", help="lr_tolerance", type=int, default=4)
    parser.add_argument("--lr_min", help="lr_min", type=float, default=1e-6)
    parser.add_argument("--weight_decay", help="weight_decay", type=float, default=0.1)
    parser.add_argument(
        "--vae_loss_coeff",
        help="vae coeff for annealing",
        type=float,
        nargs="+",
        default=[0.0, 1.0],
    )
    parser.add_argument(
        "--vae_loss_beta",
        help="decaying coeff for vae loss annealing",
        type=float,
        default=0.2,
    )
    parser.add_argument("--restart_file", help="restart_file", type=str, default=None)
    parser.add_argument("--save_every", help="save every n epochs", type=int, default=1)

    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="Path to a pretrained model checkpoint to load at start")

    args = parser.parse_args()
    return args

# below is duplicated check please
def generate_args_parser():
    parser = argparse.ArgumentParser()

    # DEFAULT SETTINGS
    parser.add_argument("--ngpu", help="ngpu", type=int, default=0)
    parser.add_argument("--ncpu", help="ncpu", type=int, default=0)
    parser.add_argument("-y", help="delete", action="store_true")

    # DIRECTORY SETTINGS
    parser.add_argument("--data_dir", help="data directory", type=str)
    parser.add_argument("--key_dir", help="key directory", type=str)
    parser.add_argument("--result_dir", help="result directory", type=str)
    parser.add_argument("--restart_dir", help="restart model directory", type=str)

    # MODEL SETTINGS
    parser.add_argument("--num_layers", help="num layers", type=int, default=10)
    parser.add_argument("--num_dense_layers", help="num layers", type=int, default=5)
    parser.add_argument(
        "--num_ligand_atom_feature",
        help="ligand atom features",
        type=int,
        default=21,
    )
    parser.add_argument(
        "--num_pocket_atom_feature",
        help="pocket atom features",
        type=int,
        default=21,
    )


    parser.add_argument("--use_scaffold", help="use_scaffold", action="store_true")
    parser.add_argument("--use_condition", help="use condition", action="store_true")
    parser.add_argument(
        "--num_cond_feature",
        help="num condition features",
        type=int,
        default=7,
    )
    parser.add_argument("--ssl", help="semi-supervised learning", action="store_true")

    # GENERATING SETTINGS
    parser.add_argument("--k", help="k for k-NN parameter", type=int, default=8)
    parser.add_argument("--max_num_add_atom", type=int, default=30)
    parser.add_argument("--radial_limits", type=float, nargs="+", default=[0.9, 2.2])
    parser.add_argument("--num_sample", help="num samples", type=int, default=30)
    parser.add_argument("--add_noise", help="add noise", action="store_true")
    parser.add_argument("--pocket_coeff_max", type=float, default=10.0)
    parser.add_argument("--pocket_coeff_thr", type=float, default=2.5)
    parser.add_argument("--pocket_coeff_beta", type=float, default=0.91)
    parser.add_argument("--dropout", help="dropout parameter", type=float, default=0.3)
    parser.add_argument("--temperature_factor1", type=float, default=0.1)
    parser.add_argument("--temperature_factor2", type=float, default=0.1)
    parser.add_argument("--translation_coeff", type=float, default=0.2)
    parser.add_argument("--rotation_coeff", type=float, default=PI / 90)
    parser.add_argument("--verbose", help="verbal mode", action="store_true")

    args = parser.parse_args()
    return args

# ############################################################################
# ##########################      utils       ################################
# ############################################################################

ATOM_TYPES = ["C", "N", "O", "F", "P", "S", "Cl", "Br"]
INTERACTION_TYPES = ["pipi", "anion", "cation", "hbd", "hba", "hydro"]

NUM_LIGAND_ATOM_TYPES = len(ATOM_TYPES) + 1   # e.g. 8 + 1 = 9
NUM_POCKET_ATOM_TYPES = 21
NUM_INTERACTION_TYPES = len(INTERACTION_TYPES) + 1  # e.g. 6 + 1 = 7

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def get_abs_path(path):
    return os.path.expanduser(path)

def get_cuda_visible_devices(num_gpus: int) -> str:
    """Get available GPU IDs as a str (e.g. '0,1,2')."""
    max_num_gpus = 16
    idle_gpus = []
    if num_gpus:
        for i in range(max_num_gpus):
            cmd = ["nvidia-smi", "-i", str(i)]
            import sys
            major, minor = sys.version_info[0], sys.version_info[1]
            if major == 3 and minor > 6:
                proc = subprocess.run(cmd, capture_output=True, text=True)
            else:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, universal_newlines=True)
            if "No devices were found" in proc.stdout:
                break
            if "No running" in proc.stdout:
                idle_gpus.append(i)
            if len(idle_gpus) >= num_gpus:
                break
        if len(idle_gpus) < num_gpus:
            msg = "Available GPUs are less than required!"
            msg += f" ({num_gpus} required, {len(idle_gpus)} available)"
            raise RuntimeError(msg)
        idle_gpus = ",".join(str(i) for i in idle_gpus[:num_gpus])
    else:
        idle_gpus = ""
    return idle_gpus

def text_filling(text, char="#", num_char=80):
    if len(text) > num_char:
        return text
    elif len(text) % 2 == 0:
        right = left = (num_char - len(text)) // 2 - 1
        return char * left + f" {text} " + char * right
    else:
        right = (num_char - len(text)) // 2 - 1
        left = (num_char - len(text)) // 2
        return char * left + f" {text} " + char * right

def initialize_model(model, use_gpu=True, load_save_file_path=None):
    if load_save_file_path:  # if not None
        if not use_gpu:  # CPU
            save_file_dict = torch.load(load_save_file_path, map_location="cpu")
            new_save_file_dict = {}
            for k in save_file_dict:
                new_key = k.replace("module.", "")
                new_save_file_dict[new_key] = save_file_dict[k]
            model.load_state_dict(new_save_file_dict, strict=False)
        else:
            save_file_dict = torch.load(load_save_file_path)
            new_save_file_dict = {}
            for k in save_file_dict:
                new_key = k.replace("module.", "")
                new_save_file_dict[new_key] = save_file_dict[k]
            model.load_state_dict(new_save_file_dict, strict=False)
    else:
        # fresh initialization
        for param in model.parameters():
            if not param.requires_grad:
                continue
            if param.dim() == 1:
                nn.init.zeros_(param)
            else:
                nn.init.xavier_uniform_(param)
    return model

from torch_geometric.data import Batch

def custom_collate_fn(list_of_dicts):
    """
    This function takes a list of samples (each is a dictionary with keys:
      "whole": pre-batched HeteroData,
      "traj":  pre-batched HeteroData,
      "key":   a key string),
    and merges them into one dictionary:
      "whole": a merged Batch of whole graphs,
      "traj":  a merged Batch of traj graphs,
      "key":   a list of keys.
    """
    wholes = [d["whole"] for d in list_of_dicts]
    trajs = [d["traj"] for d in list_of_dicts]
    keys = [d["key"] for d in list_of_dicts]
    whole_batch = Batch.from_data_list(wholes)
    traj_batch = Batch.from_data_list(trajs)
    return {"whole": whole_batch, "traj": traj_batch, "key": keys}


# ############################################################################
# ##########################     layers      #################################
# ############################################################################

class SoftOneHot(nn.Module):
    """
    Gaussian expansion of given distance
    x_min, x_max, steps, gamma=10.0, normalize=False
    """
    def __init__(self, x_min, x_max, steps, gamma=10.0, normalize=False):
        super().__init__()
        assert x_min < x_max, "x_min is larger than x_max"
        self.x_min = x_min
        self.x_max = x_max
        self.steps = int(steps)
        self.center = torch.Tensor(
            [
                x_min * (self.steps - i - 1) / (self.steps - 1) + x_max * i / (self.steps - 1)
                for i in range(self.steps)
            ]
        )
        self.gamma = gamma
        self.normalize = normalize

    def forward(self, x):
        x_repeat = x.unsqueeze(-1).repeat(1, 1, self.steps)
        c_repeat = self.center.unsqueeze(0).unsqueeze(0).to(x_repeat.device)
        x_embed = torch.exp(-self.gamma * (x_repeat - c_repeat).pow(2))
        if self.normalize:
            x_embed = x_embed / torch.sum(x_embed, dim=-1, keepdim=True)
        return x_embed


class HardOneHot(nn.Module):
    def __init__(self, x_min, x_max, steps):
        super().__init__()
        assert x_min < x_max, "x_min is larger than x_max"
        self.x_min = x_min
        self.x_max = x_max
        self.steps = steps
        self.eye = torch.eye(self.steps)

    def forward(self, x):
        x = x - self.x_min
        x = x * (self.steps - 1) / (self.x_max - self.x_min)
        x = x.clamp(0, self.steps - 1).long()
        c = self.eye[x]
        return c


class E3II_Layer(nn.Module):
    """E(3)-invariant interaction layer with revised conditional branch."""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_feature = args.num_hidden_feature
        if args.conditional:
            self.cond_feature = args.num_cond_feature
            self.cond_mlp = nn.Linear(self.hidden_feature + self.cond_feature, self.hidden_feature, bias=False)
        else:
            self.cond_feature = 0

        self.inter_mlp = nn.Sequential(
            nn.Linear(args.dist_one_hot_param1[-1] + self.hidden_feature * 2, self.hidden_feature),
            nn.SiLU(),
            nn.Linear(self.hidden_feature, self.hidden_feature),
        )
        self.intra_mlp = nn.Sequential(
            nn.Linear(args.dist_one_hot_param1[-1] + self.hidden_feature * 2, self.hidden_feature),
            nn.SiLU(),
            nn.Linear(self.hidden_feature, self.hidden_feature),
        )
        self.inter_intra_gate = nn.Sequential(
            nn.Linear(2 * self.hidden_feature, 1),
            nn.Sigmoid()
        )
        self.l_node_update = nn.GRUCell(self.hidden_feature, self.hidden_feature)
        self.p_node_update = nn.GRUCell(self.hidden_feature, self.hidden_feature)
        self.radial_cutoff = args.dist_one_hot_param1[1]

    def filter_function(self, edge_weight):
        return 0.5 * (torch.cos(edge_weight * math.pi / self.radial_cutoff) + 1.0)

    def forward(self, data, cond=None):
        if self.args.conditional and cond is not None:

            if cond.dim() == 1:
                cond = cond.unsqueeze(0).expand(data["pocket"].h.size(0), -1)
            elif cond.dim() == 2 and cond.size(0) == 1 and data["pocket"].h.size(0) > 1:
                cond = cond.expand(data["pocket"].h.size(0), -1)
            if cond.size(-1) != self.args.num_cond_feature:

                factor = self.args.num_cond_feature // cond.size(-1)
                cond = cond.repeat(1, factor)

            data["pocket"].h_cond = torch.cat([data["pocket"].h, cond], dim=-1)
            data["pocket"].h = self.cond_mlp(data["pocket"].h_cond)

        e_ll_src, e_ll_tar = data["l2l"].edge_index
        e_pp_src, e_pp_tar = data["p2p"].edge_index
        e_pl_src, e_pl_tar = data["p2l"].edge_index

        h_ll_i, h_ll_j = data["ligand"].h[e_ll_src], data["ligand"].h[e_ll_tar]
        h_pp_i, h_pp_j = data["pocket"].h[e_pp_src], data["pocket"].h[e_pp_tar]
        h_pl_i, h_pl_j = data["pocket"].h[e_pl_src], data["ligand"].h[e_pl_tar]
        h_lp_i, h_lp_j = data["ligand"].h[e_pl_tar], data["pocket"].h[e_pl_src]

        ll_in = torch.cat([h_ll_i, data["l2l"].edge_attr, h_ll_j], dim=-1)
        pp_in = torch.cat([h_pp_i, data["p2p"].edge_attr, h_pp_j], dim=-1)
        pl_in = torch.cat([h_pl_i, data["p2l"].edge_attr, h_pl_j], dim=-1)
        lp_in = torch.cat([h_lp_i, data["p2l"].edge_attr, h_lp_j], dim=-1)

        filter_ll = self.filter_function(data["l2l"].edge_weight).unsqueeze(-1)
        filter_pp = self.filter_function(data["p2p"].edge_weight).unsqueeze(-1)
        filter_pl = self.filter_function(data["p2l"].edge_weight).unsqueeze(-1)

        msg_ll = self.inter_mlp(ll_in) * filter_ll
        msg_pp = self.inter_mlp(pp_in) * filter_pp
        msg_pl = self.intra_mlp(pl_in) * filter_pl
        msg_lp = self.intra_mlp(lp_in) * filter_pl

        from torch_scatter import scatter_mean
        aggr_msg_ll = scatter_mean(msg_ll, e_ll_tar, dim=0, out=data["ligand"].h.new_zeros(data["ligand"].h.shape))
        aggr_msg_pp = scatter_mean(msg_pp, e_pp_tar, dim=0, out=data["pocket"].h.new_zeros(data["pocket"].h.shape))
        aggr_msg_pl = scatter_mean(msg_pl, e_pl_tar, dim=0, out=data["ligand"].h.new_zeros(data["ligand"].h.shape))
        aggr_msg_lp = scatter_mean(msg_lp, e_pl_src, dim=0, out=data["pocket"].h.new_zeros(data["pocket"].h.shape))

        l_msg_cat = torch.cat([aggr_msg_ll, aggr_msg_pl], dim=-1)
        p_msg_cat = torch.cat([aggr_msg_pp, aggr_msg_lp], dim=-1)

        l_gate = self.inter_intra_gate(l_msg_cat)
        p_gate = self.inter_intra_gate(p_msg_cat)

        l_msg = l_gate * aggr_msg_ll + (1.0 - l_gate) * aggr_msg_pl
        p_msg = p_gate * aggr_msg_pp + (1.0 - p_gate) * aggr_msg_lp

        new_h_l = self.l_node_update(data["ligand"].h, l_msg)
        new_h_p = self.p_node_update(data["pocket"].h, p_msg)

        data["ligand"].h = new_h_l
        data["pocket"].h = new_h_p
        return data


class EGCL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.distance_expand = SoftOneHot(*args.dist_one_hot_param1, gamma=args.gamma1)
        self.num_edge_feature = args.dist_one_hot_param1[-1]
        self.num_node_feature = args.num_pocket_atom_feature
        self.num_hidden_feature = args.num_hidden_feature

        self.edge_mlp = nn.Sequential(
            nn.Linear(self.num_node_feature * 2 + self.num_edge_feature, self.num_hidden_feature),
            nn.SiLU(),
            nn.Linear(self.num_hidden_feature, self.num_hidden_feature),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(self.num_hidden_feature, self.num_hidden_feature),
            nn.SiLU(),
            nn.Linear(self.num_hidden_feature, self.num_node_feature),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(self.num_hidden_feature, self.num_hidden_feature),
            nn.SiLU(),
            nn.Linear(self.num_hidden_feature, 1),
        )

    def forward(self, h, x, edge_index):
        from torch_scatter import scatter_add
        src, tar = edge_index
        h_i, h_j = h[src], h[tar]
        x_i, x_j = x[src], x[tar]
        d_ij = (x_i - x_j).norm(dim=-1)
        e_ij = self.distance_expand(d_ij)[0]

        edge_msg = self.edge_mlp(torch.cat([h_i, h_j, e_ij], dim=-1))
        coord_msg = self.coord_mlp(edge_msg)
        div = d_ij.unsqueeze(-1) + 1e-8

        x_prime = x + scatter_add((x_i - x_j) * coord_msg / div, tar, dim=0)
        from torch_scatter import scatter_mean
        h_prime = h + scatter_mean(self.node_mlp(edge_msg), tar, dim=0,
                                   out=h.new_zeros(h.shape))
        return h_prime, x_prime


class ConstrainedCrossAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.qkv_proj = nn.Linear(args.num_hidden_feature, 3 * args.num_hidden_feature)
        self.b_proj = nn.Linear(args.num_hidden_feature, 1)
        self.g_proj = nn.Linear(args.num_hidden_feature, 1)
        self.t_proj = nn.Linear(args.dist_one_hot_param1[-1], 1)
        self.dist_expand = SoftOneHot(*args.dist_one_hot_param1)

    def attention(self, h, x, batch, length):
        from torch_scatter import scatter_add, scatter_softmax
        sqrt_len = torch.pow(length.float(), -0.5)[batch]

        qkv_proj = self.qkv_proj(h)
        q, k, v = qkv_proj.chunk(3, dim=-1)
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_softmax = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attn_softmax, v)
        return values, attn_softmax

    def forward(self, h, x, batch):
        from torch_scatter import scatter_add
        ones = batch.new_ones(batch.shape)
        length = scatter_add(ones, batch, 0).long()
        values, attn = self.attention(h, x, batch, length)
        return values, attn

# ############################################################################
# ######################   Reasoning Layer   #######################
# ############################################################################

class ReasoningLayer(nn.Module):
    """
    I added ReasoningLayer, which fuses sequence and MD metrics.

    It computes a per-graph meta vector from:
      [len(pep_seq_str), len(pep_smi_str), len(pep_scf_str), docking_score, binding_energy_kcal_per_mol]
    passes it through an MLP, and then “broadcasts” it to every node in the graph.
    If processing a batch of graphs, the per-graph meta vectors are selected for each node via the
    provided batch vector.

    Additionally, if a per-node RMSD vector is provided, it is passed through a linear layer with a
    sigmoid to obtain a scaling factor that is multiplied elementwise with the updated node embedding.
    """
    def __init__(self, hidden_dim, use_rmsd=True):
        super(ReasoningLayer, self).__init__()
        self.use_rmsd = use_rmsd
        self.meta_mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        if self.use_rmsd:
            self.rmsd_scaler = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.Sigmoid()
            )
    
    def forward(self, x, pep_seq, pep_smi, pep_scf, docking, binding, rmsd=None, batch=None):
        if batch is None:
            # Single graph case
            if isinstance(docking, torch.Tensor):
                docking = docking.item()
            if isinstance(binding, torch.Tensor):
                binding = binding.item()
            meta_values = [
                len(pep_seq),
                len(pep_smi),
                len(pep_scf) if pep_scf else 0,
                float(docking),
                float(binding)
            ]
            meta_vec = torch.tensor(meta_values, dtype=torch.float, device=x.device).unsqueeze(0)
            global_meta = self.meta_mlp(meta_vec)
            global_meta_expanded = global_meta.expand(x.size(0), -1)
        else:
            B = int(batch.max().item() + 1)
            if not isinstance(pep_seq, list):
                pep_seq = [pep_seq] * B
            if not isinstance(pep_smi, list):
                pep_smi = [pep_smi] * B
            if not isinstance(pep_scf, list):
                pep_scf = [pep_scf] * B
            meta_list = []
            for i in range(B):
                seq_len = len(pep_seq[i])
                smi_len = len(pep_smi[i])
                scf_len = len(pep_scf[i]) if pep_scf[i] else 0
                d_val = float(docking[i].item()) if isinstance(docking, torch.Tensor) else float(docking)
                b_val = float(binding[i].item()) if isinstance(binding, torch.Tensor) else float(binding)
                meta_list.append([seq_len, smi_len, scf_len, d_val, b_val])
            meta_tensor = torch.tensor(meta_list, dtype=torch.float, device=x.device)
            global_meta = self.meta_mlp(meta_tensor)
            global_meta_expanded = global_meta[batch]
        
        x_updated = x + global_meta_expanded
        
        if self.use_rmsd and rmsd is not None:
            if rmsd.dim() == 1:
                rmsd = rmsd.unsqueeze(-1)
            scale = self.rmsd_scaler(rmsd)
            x_updated = x_updated * scale
        
        return x_updated


# ############################################################################
# ##########################      model       ################################
# ############################################################################

class VariationalEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mean = nn.Linear(args.num_hidden_feature, args.num_latent_feature)
        self.logvar = nn.Linear(args.num_hidden_feature, args.num_latent_feature)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mean

    def vae_loss(self, mean, logvar):
        # KL Divergence
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1)

    def forward(self, data):
        # We just do a readout from data["ligand"].h and data["pocket"].h
        h_cat = torch.cat([data["ligand"].h, data["pocket"].h], dim=0)
        readout = h_cat.mean(dim=0, keepdim=True)
        mean = self.mean(readout)
        logvar = self.logvar(readout)
        latent = self.reparameterize(mean, logvar)
        vae_loss = self.vae_loss(mean, logvar)
        return latent, vae_loss


class Embedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.l_node_emb = nn.Sequential(
            nn.Linear(args.num_ligand_atom_feature, args.num_hidden_feature),
        )
        self.p_node_emb = nn.Sequential(
            nn.Linear(args.num_pocket_atom_feature, args.num_hidden_feature),
        )
        self.layers = nn.ModuleList([E3II_Layer(args) for _ in range(args.num_layers)])

    def forward(self, data, cond=None):
        data["ligand"].h = self.l_node_emb(data["ligand"].x)
        data["pocket"].h = self.p_node_emb(data["pocket"].x)

        for layer in self.layers:
            layer(data, cond=cond)
        return data


class NextType(nn.Module):
    def __init__(self, args, embedding=None):
        super().__init__()
        self.args = args
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Linear(args.num_ligand_atom_feature, args.num_hidden_feature, bias=False)

        self.act = nn.SiLU()
        self.last_act = None
        layers = []
        n_dims = list(np.linspace(args.num_hidden_feature, 1, args.num_dense_layers + 1).astype(int))
        for n_in, n_out in zip(n_dims[:-1], n_dims[1:]):
            layers.append(nn.Linear(n_in, n_out))
            layers.append(self.act)
        layers = layers[:-1]
        if self.last_act is not None:
            layers.append(self.last_act)
        self.dense = nn.Sequential(*layers)

        self.atom_type = torch.eye(args.num_ligand_atom_feature)
        self.atom_type = nn.Parameter(self.atom_type, requires_grad=False)

    def forward(self, data, key="ligand"):

        embed_type = self.embedding(self.atom_type)  # [num_types, hidden_dim]
        embed_type = embed_type.unsqueeze(0)          # [1, num_types, hidden_dim]
        repr_type = data[key].h.unsqueeze(1)          # [N, 1, hidden_dim]
        mul_type = embed_type * repr_type             # [N, num_types, hidden_dim]
        dense_type = self.dense(mul_type).squeeze(-1)   # [N, num_types]
        next_type = F.log_softmax(dense_type, dim=-1)   # [N, num_types]
        return next_type


class NextDist(nn.Module):
    """
    Predict p(Distance) from embeddings + next_type.
    """
    def __init__(self, args, embedding=None):
        super().__init__()
        self.args = args
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Linear(args.num_ligand_atom_feature, args.num_hidden_feature, bias=False)

        self.act = nn.SiLU()
        self.last_act = None
        layers = []
        n_dims = list(
            np.linspace(args.num_hidden_feature, args.dist_one_hot_param2[-1], args.num_dense_layers+1).astype(int)
        )
        for n_in, n_out in zip(n_dims[:-1], n_dims[1:]):
            layers.append(nn.Linear(n_in, n_out))
            layers.append(self.act)
        layers = layers[:-1]
        if self.last_act is not None:
            layers.append(self.last_act)

        self.dense = nn.Sequential(*layers)
        self.use_attention = False
        if self.use_attention:
            self.attn = ConstrainedCrossAttention(args)

    def forward(self, data, next_type, key):
        from torch_scatter import scatter_mean
        batch = data[key].batch
        type_embed = self.embedding(next_type)[batch]
        dist_embed = data[key].h * type_embed
        # If we wanted advanced attention: dist_embed, attn = self.attn(dist_embed, data[key].pos, batch)
        next_dist = F.log_softmax(self.dense(dist_embed), dim=-1)
        return next_dist


class SSLModel(nn.Module):
    """
    For semi-supervised learning condition prediction.
    """
    def __init__(self, args, embedding=None):
        super().__init__()
        self.args = args
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Linear(args.num_pocket_atom_feature, args.num_hidden_feature, bias=False)

        self.num_layers = args.num_dense_layers
        self.layers = nn.ModuleList([EGCL(args) for _ in range(self.num_layers)])
        self.fc_layer = nn.Linear(args.num_hidden_feature, args.num_cond_feature)

    def forward(self, data):
        edge_index_ = data["pocket"].edge_index.clone()
        h_ = data["pocket"].x.clone()
        x_ = data["pocket"].pos.clone()
        for layer in self.layers:
            h_, x_ = layer(h_, x_, edge_index_)
        y = self.fc_layer(h_)
        return y


# ###########################  The main Model: DynaPep  #######################

class DynaPep(nn.Module):
    """
    Now the revised DynaPep model integrates:
      - A conditional VAE for peptide encoding with KL annealing support.
      - A reasoning layer that fuses pocket features, the peptide latent vector, and RMSD information.
      - A decoder branch for peptide sequence generation (via NextType/NextDist) and an MD regression head.
      
    Key revisions:
      • The forward() method accepts a kl_weight parameter (e.g., from a KL annealing schedule).
      • The latent vector is concatenated with pocket features at every stage (ensuring continuous conditioning).
      • The MD head now receives a fused context that includes both pocket and peptide (latent) info.
      • Comments indicate where additional improvements (teacher-forcing dropout, RMSD normalization,
        dual-branch reasoning, etc.) could be inserted.
    """
    def __init__(self, args):
        super(DynaPep, self).__init__()
        self.args = args
        # Embedding modules for ligand and pocket.
        self.embedding = Embedding(args)
        # VAE encoder: produces latent representation (mean and logvar) from the ligand (peptide).
        self.vae = VariationalEncoder(args)
        self.conditional = args.conditional
        self.num_cond_feature = args.num_cond_feature if self.conditional else 0

        # Optional semi-supervised learning branch.
        if args.ssl:
            self.ssl_model = SSLModel(args)

        # Scale factors for loss components.
        self.scale_vae = getattr(args, "scale_vae", 1.0)
        self.scale_md = getattr(args, "scale_md", 1e-4)
        self.scale_ssl = getattr(args, "scale_ssl", 1.0)
        # Note: KL weight is now supplied externally via the forward() method.
        
        self.reasoning_layer = ReasoningLayer(hidden_dim=args.num_hidden_feature, use_rmsd=True)
        self.latent_mlp = nn.Linear(
            args.num_hidden_feature + args.num_latent_feature + self.num_cond_feature,
            args.num_hidden_feature
        )
        
        self.next_type_ll = NextType(args, self.embedding.l_node_emb)
        self.next_type_lp = NextType(args, self.embedding.l_node_emb)
        self.next_dist_ll = NextDist(args, self.embedding.l_node_emb)
        self.next_dist_lp = NextDist(args, self.embedding.l_node_emb)
        # KL divergence loss function.
        self.kl_loss_fn = nn.KLDivLoss(reduction="none")

        self.md_regressor = nn.Sequential(
            nn.Linear(args.num_hidden_feature + args.num_latent_feature + self.num_cond_feature, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.rmsd_head = nn.Linear(args.num_hidden_feature, 1)
        self.pocket_energy_head = nn.Linear(args.num_hidden_feature, 1)

    def forward(self, data_dict, kl_weight=1.0):

        whole = data_dict["whole"]
        traj = data_dict["traj"]

        raw_docking = getattr(whole, "docking_score", 0.0)
        if torch.is_tensor(raw_docking):
            raw_docking = raw_docking.mean().item() if raw_docking.numel() > 1 else raw_docking.item()
        raw_binding = getattr(whole, "binding_energy_kcal_per_mol", 0.0)
        if torch.is_tensor(raw_binding):
            raw_binding = raw_binding.mean().item() if raw_binding.numel() > 1 else raw_binding.item()

        gt_rmsd = getattr(whole, "md_per_residue_rmsd", None)
        if gt_rmsd is None:
            gt_rmsd = whole["ligand"].x.new_zeros(whole["ligand"].num_nodes)
        gt_pocket_energy = getattr(whole["pocket"], "pocket_energy", None)

        cond_whole = whole.pocket_prop if self.conditional else None
        cond_traj = traj.pocket_prop if self.conditional else None


        self.embedding(whole, cond=cond_whole)
        latent_vec, vae_loss_val = self.vae(whole)
        vae_loss = vae_loss_val.sum()

        pep_seq = getattr(whole, "pep_seq_str", "")
        pep_smi = getattr(whole, "pep_smi_str", "")
        pep_scf = getattr(whole, "pep_scf_str", "")  # May be empty if not available.

        if not hasattr(whole["ligand"], "batch"):
            whole["ligand"].batch = torch.zeros(whole["ligand"].x.size(0),
                                                 dtype=torch.long,
                                                 device=whole["ligand"].x.device)
        if not hasattr(whole["pocket"], "batch"):
            whole["pocket"].batch = torch.zeros(whole["pocket"].x.size(0),
                                                 dtype=torch.long,
                                                 device=whole["pocket"].x.device)

        whole["ligand"].h = self.reasoning_layer(
            whole["ligand"].h, pep_seq, pep_smi, pep_scf,
            docking=raw_docking, binding=raw_binding, rmsd=gt_rmsd,
            batch=whole["ligand"].batch
        )
        whole["pocket"].h = self.reasoning_layer(
            whole["pocket"].h, pep_seq, pep_smi, pep_scf,
            docking=raw_docking, binding=raw_binding, rmsd=gt_pocket_energy,
            batch=whole["pocket"].batch
        )

        self.embedding(traj, cond=cond_traj)
        if not hasattr(traj["ligand"], "batch"):
            traj["ligand"].batch = torch.zeros(traj["ligand"].x.size(0),
                                                dtype=torch.long,
                                                device=traj["ligand"].x.device)
        if not hasattr(traj["pocket"], "batch"):
            traj["pocket"].batch = torch.zeros(traj["pocket"].x.size(0),
                                                dtype=torch.long,
                                                device=traj["pocket"].x.device)
        traj["ligand"].h = self.reasoning_layer(
            traj["ligand"].h, pep_seq, pep_smi, pep_scf,
            docking=raw_docking, binding=raw_binding, rmsd=gt_rmsd,
            batch=traj["ligand"].batch
        )
        traj["pocket"].h = self.reasoning_layer(
            traj["pocket"].h, pep_seq, pep_smi, pep_scf,
            docking=raw_docking, binding=raw_binding, rmsd=gt_pocket_energy,
            batch=traj["pocket"].batch
        )


        psize = traj["pocket"].h.size(0)
        if self.conditional and cond_traj is not None:
            if cond_traj.size(-1) != self.num_cond_feature:
                cond_traj = cond_traj.repeat(1, self.num_cond_feature)
            cat_vec = torch.cat([traj["pocket"].h, cond_traj, latent_vec.repeat(psize, 1)], dim=-1)
        else:
            cat_vec = torch.cat([traj["pocket"].h, latent_vec.repeat(psize, 1)], dim=-1)
        traj["pocket"].h = self.latent_mlp(cat_vec)

        pred_type_lig = self.next_type_ll(traj, key="ligand")
        pred_type_poc = self.next_type_lp(traj, key="pocket")
        gt_type_lig = traj["ligand"].type_output
        gt_type_poc = traj["pocket"].type_output
        type_loss = self.kl_loss_fn(pred_type_lig, gt_type_lig).sum() + \
                    self.kl_loss_fn(pred_type_poc, gt_type_poc).sum()

        pred_dist_lig = self.next_dist_ll(traj, next_type=gt_type_lig, key="ligand")
        pred_dist_poc = self.next_dist_lp(traj, next_type=gt_type_poc, key="pocket")
        gt_dist_lig = traj["ligand"].dist_ll_output
        gt_dist_poc = traj["pocket"].dist_lp_output
        dist_loss = self.kl_loss_fn(pred_dist_lig, gt_dist_lig).sum() + \
                    self.kl_loss_fn(pred_dist_poc, gt_dist_poc).sum()
        kl_loss = type_loss + dist_loss


        md_preds = self.md_regressor(cat_vec)
        docking_pred = md_preds[:, 0].mean()
        binding_pred = md_preds[:, 1].mean()
        docking_mse = F.mse_loss(docking_pred, torch.tensor(raw_docking, device=cat_vec.device))
        binding_mse = F.mse_loss(binding_pred, torch.tensor(raw_binding, device=cat_vec.device))
        md_reg_loss = docking_mse + binding_mse


        pred_rmsd = self.rmsd_head(whole["ligand"].h).squeeze(-1)
        rmsd_loss = F.mse_loss(pred_rmsd, gt_rmsd, reduction="mean")
        if gt_pocket_energy is not None:
            pred_pe = self.pocket_energy_head(whole["pocket"].h).squeeze(-1)
            pocket_energy_loss = F.mse_loss(pred_pe, gt_pocket_energy, reduction="mean")
        else:
            pocket_energy_loss = 0.0
        raw_md_loss = md_reg_loss + rmsd_loss + pocket_energy_loss


        ssl_loss = 0.0
        if self.args.ssl:
            # Insert semi-supervised loss logic here.
            pass

        total_loss = (self.scale_vae * vae_loss +
                      kl_weight * kl_loss +
                      self.scale_md * raw_md_loss +
                      self.scale_ssl * ssl_loss)

        return total_loss, vae_loss, kl_loss, (self.scale_md * raw_md_loss), ssl_loss

# ############################################################################
# ##########################      dataset     ################################
# ############################################################################
import numpy as np
import torch
from torch_geometric.data import HeteroData, Batch
from torch_geometric.nn import radius_graph  # <<-- Added import
from torch import cdist
from torch import topk 

from torch.utils.data import Sampler
import math
import random
from collections import defaultdict

class SystemGroupingSampler(Sampler):

    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.system2indices = defaultdict(list)
        # Expect dataset.samples to be a list of tuples: (key, system_id)
        for idx in range(len(dataset)):
            _, system_id = dataset.samples[idx]
            self.system2indices[system_id].append(idx)
        self.system_ids = list(self.system2indices.keys())
        if self.shuffle:
            random.shuffle(self.system_ids)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.system_ids)
        for sys_id in self.system_ids:
            yield self.system2indices[sys_id]

    def __len__(self):
        return len(self.system_ids)


class DataProcessor(object):

    def __init__(self, args, mode="train"):
        self.args = args
        self.mode = mode
        self.radial_cutoff = args.dist_one_hot_param1[1]
        self.distance_expand1 = SoftOneHot(*args.dist_one_hot_param1, gamma=args.gamma1)
        self.distance_expand2 = SoftOneHot(*args.dist_one_hot_param2, gamma=args.gamma2, normalize=True)

    def get_input_from_data(self, data_dict):
        ligand_type  = data_dict["ligand_type"]
        ligand_coord = data_dict["ligand_coord"]
        pocket_type  = data_dict["pocket_type"]
        pocket_coord = data_dict["pocket_coord"]
        pep_seq_str  = data_dict["pep_seq_str"]
        pep_smi_str  = data_dict["pep_smi"]
        pocket_cond  = data_dict["pocket_cond"]

        docking_score  = float(data_dict["docking_score"])      # e.g., -9.2
        binding_energy = float(data_dict["binding_energy_kcal_per_mol"])  # e.g., -45.2
        md_rmsd_array  = np.array(data_dict["md_per_residue_rmsd"])  # shape [n_res]

        whole = HeteroData()
        whole["ligand"].x = torch.tensor(ligand_type, dtype=torch.float)
        whole["ligand"].pos = torch.tensor(ligand_coord, dtype=torch.float)
        whole["pocket"].x = torch.tensor(pocket_type, dtype=torch.float)
        whole["pocket"].pos = torch.tensor(pocket_coord, dtype=torch.float)

        whole.pep_seq_str = pep_seq_str
        whole.pep_smi_str = pep_smi_str
        whole.pep_scf_str = ""
        whole.docking_score = docking_score
        whole.binding_energy_kcal_per_mol = binding_energy
        whole.md_per_residue_rmsd = torch.tensor(md_rmsd_array, dtype=torch.float)
        whole.pocket_prop = torch.tensor(pocket_cond, dtype=torch.float)

        l2l_index = radius_graph(whole["ligand"].pos, r=self.radial_cutoff)
        p2p_index = radius_graph(whole["pocket"].pos, r=self.radial_cutoff)
        if l2l_index.numel() == 0:
            l2l_index = torch.zeros((2, 1), dtype=torch.long)
        if p2p_index.numel() == 0:
            p2p_index = torch.zeros((2, 1), dtype=torch.long)
        whole["ligand", "l2l", "ligand"].edge_index = l2l_index
        whole["pocket", "p2p", "pocket"].edge_index = p2p_index

        l2l_weight = (whole["ligand"].pos[l2l_index[0]] - whole["ligand"].pos[l2l_index[1]]).norm(dim=-1)
        p2p_weight = (whole["pocket"].pos[p2p_index[0]] - whole["pocket"].pos[p2p_index[1]]).norm(dim=-1)
        l2l_attr = self.distance_expand1(l2l_weight)[0]
        p2p_attr = self.distance_expand1(p2p_weight)[0]

        whole["l2l"].edge_weight = l2l_weight
        whole["p2p"].edge_weight = p2p_weight
        whole["l2l"].edge_attr = l2l_attr
        whole["p2p"].edge_attr = p2p_attr

        l_n = whole["ligand"].x.size(0)
        p_n = whole["pocket"].x.size(0)
        if p_n == 0 or l_n == 0:
            pl_index = torch.zeros((2, 1), dtype=torch.long)
            p2l_weight = torch.zeros((1,), dtype=torch.float)
            p2l_attr = torch.zeros((1, self.args.dist_one_hot_param1[-1]), dtype=torch.float)
        else:
            dist_mat = cdist(whole["ligand"].pos, whole["pocket"].pos)
            k_val = self.args.k if self.args.k else 8
            k_val = min(k_val, p_n)
            p_knn_index = topk(dist_mat, k=k_val, largest=False, dim=-1)[1]
            p_index = p_knn_index.view(-1)
            l_index = torch.arange(l_n).unsqueeze(-1).repeat(1, k_val).view(-1)
            pl_index = torch.stack([p_index, l_index], dim=0)
            p2l_weight = (whole["pocket"].pos[p_index] - whole["ligand"].pos[l_index]).norm(dim=-1)
            p2l_attr = self.distance_expand1(p2l_weight)[0]

        whole["pocket", "p2l", "ligand"].edge_index = pl_index
        whole["p2l"].edge_weight = p2l_weight
        whole["p2l"].edge_attr = p2l_attr

        whole = Batch.from_data_list([whole])
        traj = whole.clone()

        ligand_count = traj["ligand"].x.size(0)
        pocket_count = traj["pocket"].x.size(0)
        type_output_ligand = torch.ones(ligand_count, self.args.num_ligand_atom_feature)
        type_output_ligand = type_output_ligand / type_output_ligand.sum(dim=-1, keepdim=True)
        dist_output_ligand = torch.ones(ligand_count, self.args.dist_one_hot_param2[-1])
        dist_output_ligand = dist_output_ligand / dist_output_ligand.sum(dim=-1, keepdim=True)
        type_output_pocket = torch.ones(pocket_count, self.args.num_ligand_atom_feature)
        type_output_pocket = type_output_pocket / type_output_pocket.sum(dim=-1, keepdim=True)
        dist_output_pocket = torch.ones(pocket_count, self.args.dist_one_hot_param2[-1])
        dist_output_pocket = dist_output_pocket / dist_output_pocket.sum(dim=-1, keepdim=True)

        traj["ligand"].type_output = type_output_ligand
        traj["ligand"].dist_ll_output = dist_output_ligand
        traj["ligand"].mask = torch.ones(ligand_count)
        traj["pocket"].type_output = type_output_pocket
        traj["pocket"].dist_lp_output = dist_output_pocket
        traj["pocket"].mask = torch.ones(pocket_count)
        traj.pocket_prop = whole.pocket_prop.clone()
        traj["ligand"].batch = torch.zeros(ligand_count, dtype=torch.long)
        traj["pocket"].batch = torch.zeros(pocket_count, dtype=torch.long)

        traj = Batch.from_data_list([traj])
        return whole, traj

class MoleculeDataset(Dataset):
    """
    A general dataset parent. 
    """
    def __init__(self, args, mode="train"):
        super().__init__()
        self.args = args
        self.mode = mode
        self.data_dir = args.data_dir
        self.key_dir = args.key_dir

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class PDBbindDataset(MoleculeDataset):
    def __init__(self, args, mode="train"):
        super().__init__(args, mode)
        self.processor = DataProcessor(args, mode)
        key_file = os.path.join(self.key_dir, f"{mode}_keys.pkl")
        if not os.path.exists(key_file):
            print(f"No {mode}_keys.pkl found at: {key_file}")
            sys.exit()
        import pickle
        with open(key_file, "rb") as f:
            self.key_list = pickle.load(f)
        
        self.samples = []
        for key in self.key_list:
            # Example: key = "1bs6_F_C_cluster1.pkl"
            # Let's parse system_id = "1bs6_F_C"
            # (assuming the key always has "_clusterN.pkl" at the end)
            filename_no_ext = key.replace(".pkl", "")
            # split off "_cluster..."
            # e.g. "1bs6_F_C_cluster1" -> ["1bs6_F_C", "1"] if we split by "_cluster"
            parts = filename_no_ext.split("_cluster")
            if len(parts) == 2:
                system_id = parts[0]  # e.g. "1bs6_F_C"
            else:
                system_id = filename_no_ext
            
            self.samples.append((key, system_id))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key, system_id = self.samples[idx]
        full_path = os.path.join(self.data_dir, key)
        import pickle
        with open(full_path, "rb") as f:
            data = pickle.load(f)
        
        whole, traj = self.processor.get_input_from_data(data)
        # We store the system_id in the dictionary so we can group in the sampler if needed
        # or store it in 'whole' or 'traj' if you like.
        item = {"whole": whole, "traj": traj, "key": key, "system_id": system_id}
        return item

# ############################################################################
# ##########################     evaluate     ################################
# ############################################################################

def save_results(mol_list, fn):
    writer = Chem.SDWriter(fn)
    for mol in mol_list:
        writer.write(mol)
    writer.close()

def filter_generated_results(keys, result_dir, train_smiles=None, filter_level=0):
    total_mol_list = []
    val, uni, nov = 0, 0, 0
    N = len(keys)
    from rdkit.RDLogger import DisableLog
    DisableLog("rdApp.*")

    for key in keys:
        if key == "":
            from glob import glob
            gen_fns = glob(f"{result_dir}/*.sdf")
            key = "total"
        else:
            from glob import glob
            gen_fns = glob(f"{result_dir}/{key}*.sdf")

        tot_n = len(gen_fns)
        if tot_n == 0:
            print("No generated molecules for", key)
            continue

        val_mol_list = []
        val_smi_dict = {}
        for fn in gen_fns:
            name = os.path.basename(fn).split(".")[0]
            if "val" in fn or "uni" in fn or "nov" in fn:
                tot_n -= 1
                continue
            try:
                mol = Chem.SDMolSupplier(fn)[0]
                assert mol is not None
                smi = Chem.MolToSmiles(mol)
                smi = Chem.CanonSmiles(smi)
                assert "." not in smi
            except Exception:
                continue

            for a in mol.GetAtoms():
                a.SetNumExplicitHs(a.GetNumRadicalElectrons())
                a.SetNumRadicalElectrons(0)
            mol = Chem.AddHs(mol, addCoords=True)
            val_mol_list.append(mol)
            val_smi_dict[name] = Chem.CanonSmiles(Chem.MolToSmiles(mol))

        val_n = len(val_smi_dict)
        if tot_n == 0:
            tot_n = 1e-8
        print("Key:", key)
        print("# Generated molecule:", tot_n)
        print("Validity:", f"{100 * val_n / tot_n:.2f}%")
        val += 100 * val_n / tot_n

        if filter_level == 1:
            save_results(val_mol_list, f"{result_dir}/{key}_valid_results.sdf")
            total_mol_list += val_mol_list
            continue

        uni_mol_list = []
        uni_smi_dict = {}
        val_keys = list(val_smi_dict.keys())
        for i in range(len(val_keys)):
            FLAG = 0
            uni_keys = list(uni_smi_dict.keys())
            for j in range(len(uni_keys)):
                if val_smi_dict[val_keys[i]] == uni_smi_dict[uni_keys[j]]:
                    FLAG = 1
            if not FLAG:
                uni_mol_list.append(val_mol_list[i])
                uni_smi_dict[val_keys[i]] = val_smi_dict[val_keys[i]]
        uni_n = len(uni_smi_dict)
        if val_n == 0:
            val_n = 1e-8
        print("Uniqueness:", f"{100 * uni_n / val_n:.2f}%")
        uni += 100 * uni_n / val_n
        if filter_level == 2:
            save_results(uni_mol_list, f"{result_dir}/{key}_unique_results.sdf")
            total_mol_list += uni_mol_list
            continue

        nov_mol_list = []
        nov_smi_dict = {}
        uni_keys = list(uni_smi_dict.keys())
        for i in range(len(uni_keys)):
            FLAG = 0
            for ts in train_smiles:
                if uni_smi_dict[uni_keys[i]] == ts:
                    FLAG = 1
            if not FLAG:
                nov_mol_list.append(uni_mol_list[i])
                nov_smi_dict[uni_keys[i]] = uni_smi_dict[uni_keys[i]]
        nov_n = len(nov_smi_dict)
        if uni_n == 0:
            uni_n = 1e-8
        print("Novelty:", f"{100 * nov_n / uni_n:.2f}%")
        nov += 100 * nov_n / uni_n

        if filter_level == 3 or filter_level == 0:
            save_results(nov_mol_list, f"{result_dir}/{key}_novel_results.sdf")
            total_mol_list += nov_mol_list
            continue

    val /= (N if N>0 else 1)
    uni /= (N if N>0 else 1)
    nov /= (N if N>0 else 1)
    return total_mol_list, val, uni, nov


def main_evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", help="directory for generated molecules", type=str)
    parser.add_argument("--key_dir", help="directory for keys", type=str)
    parser.add_argument("--smi_dir", help="directory for training set smiles", type=str)
    parser.add_argument("--filter_level", help="filter level, 0:All,1:Valid,2:Unique,3:Novel", type=int, default=0)
    args = parser.parse_args()

    if args.key_dir is None:
        KEYS = [""]
    else:
        import pickle
        with open(args.key_dir, "rb") as f:
            KEYS = pickle.load(f)
        KEYS = sorted(KEYS)

    with open(args.smi_dir, "r") as f:
        SMIS = [Chem.CanonSmiles(s.strip().split()[-1]) for s in f.readlines()]

    tot_mol_list, val, uni, nov = filter_generated_results(KEYS, args.result_dir, SMIS, args.filter_level)
    print(f"TOTAL VALIDITY: {val:.2f}")
    if args.filter_level > 1 or args.filter_level == 0:
        print(f"TOTAL UNIQUENESS: {uni:.2f}")
    if args.filter_level > 2 or args.filter_level == 0:
        print(f"TOTAL NOVELTY: {nov:.2f}")

# ############################################################################
# ##########################      train       ################################
# ############################################################################

def sample_to_device(sample, device):
    sample["whole"].to(device)
    sample["traj"].to(device)


from contextlib import nullcontext
import torch.distributed as dist

import torch.distributed as dist
from contextlib import nullcontext

def train_one_epoch(model, args, optimizer, data_loader, train_mode: bool, device, scaler=None):

    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss_sum = 0.0
    vae_loss_sum   = 0.0
    type_loss_sum  = 0.0
    dist_loss_sum  = 0.0
    ssl_loss_sum   = 0.0
    local_count    = 0

    if not dist.is_initialized() or dist.get_rank() == 0:
        from tqdm import tqdm
        pbar = tqdm(data_loader, total=len(data_loader), desc=("Train" if train_mode else "Valid"), leave=False)
    else:
        pbar = data_loader

    from contextlib import nullcontext
    ctx = torch.no_grad() if not train_mode else nullcontext()
    with ctx:
        for sample in pbar:
            local_count += 1

            sample["whole"].to(device)
            sample["traj"].to(device)

            if train_mode and args.autocast:
                import torch.cuda.amp as amp
                with amp.autocast():
                    total_loss, vae_loss, type_loss, dist_loss, ssl_loss = model(sample)
            else:
                total_loss, vae_loss, type_loss, dist_loss, ssl_loss = model(sample)

            if train_mode:
                optimizer.zero_grad()
                if args.autocast:
                    scaler.scale(total_loss).backward()
                    # Unscale and clip gradients
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

            total_loss_sum += total_loss.item()
            vae_loss_sum   += vae_loss.item()
            type_loss_sum  += type_loss.item()
            dist_loss_sum  += dist_loss.item()
            if args.ssl and ssl_loss is not None:
                # Check if ssl_loss is a tensor or a float.
                if isinstance(ssl_loss, float):
                    ssl_loss_sum += ssl_loss
                else:
                    ssl_loss_sum += ssl_loss.item()

            if hasattr(pbar, "set_postfix"):
                avg_so_far = total_loss_sum / local_count
                pbar.set_postfix({'Loss': f"{avg_so_far:.3f}"})

    if dist.is_initialized():
        sum_tensor = torch.tensor(
            [total_loss_sum, vae_loss_sum, type_loss_sum, dist_loss_sum, ssl_loss_sum],
            device=device, dtype=torch.float)
        count_tensor = torch.tensor([local_count], device=device, dtype=torch.float)
        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        global_sum = sum_tensor.cpu().numpy()
        global_count = count_tensor.item()
        if global_count < 1e-8:
            total_mean = vae_mean = type_mean = dist_mean = ssl_mean = 0.0
        else:
            total_mean = global_sum[0] / global_count
            vae_mean   = global_sum[1] / global_count
            type_mean  = global_sum[2] / global_count
            dist_mean  = global_sum[3] / global_count
            ssl_mean   = global_sum[4] / global_count
    else:
        if local_count == 0:
            return (0, 0, 0, 0, 0)
        total_mean = total_loss_sum / local_count
        vae_mean   = vae_loss_sum   / local_count
        type_mean  = type_loss_sum  / local_count
        dist_mean  = dist_loss_sum  / local_count
        ssl_mean   = ssl_loss_sum   / local_count

    return total_mean, vae_mean, type_mean, dist_mean, ssl_mean


def main_worker(gpu, ngpus_per_node, args):
    total_gpus = torch.cuda.device_count()
    if gpu >= total_gpus:
        gpu = 0
    rank = gpu

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = args.master_port
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    print(text_filling(f"Finished Setting DDP: CUDA:{rank}"), flush=True)
    torch.cuda.set_device(gpu)

    train_dataset = PDBbindDataset(args, mode="train")
    valid_dataset = PDBbindDataset(args, mode="valid")

    train_sampler = SystemGroupingSampler(train_dataset, shuffle=True)
    valid_sampler = SystemGroupingSampler(valid_dataset, shuffle=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_sampler=valid_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    N_TRAIN_DATA = len(train_dataset)
    N_VALID_DATA = len(valid_dataset)
    if rank == 0 and not args.restart_file:
        print("Train dataset length:", N_TRAIN_DATA, flush=True)
        print("Validation dataset length:", N_VALID_DATA, flush=True)
        print(text_filling("Finished Loading Datasets"), flush=True)

    model = DynaPep(args)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    load_path = get_abs_path(args.pretrained_model) if args.pretrained_model is not None else None
    model = initialize_model(model, use_gpu=True, load_save_file_path=load_path)
    model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
    cudnn.benchmark = True
    if rank == 0 and not args.restart_file:
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of Parameters:", param_count, flush=True)
        print(text_filling("Finished Loading Model"), flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_decay, patience=args.lr_tolerance,
        min_lr=args.lr_min, verbose=True
    )
    scaler = None
    if args.autocast:
        import torch.cuda.amp as amp
        scaler = amp.GradScaler()

    vae_coeff_init, vae_coeff_final = args.vae_loss_coeff
    start_epoch = int(args.restart_file.split("_")[-1].split(".")[0]) if args.restart_file else 0

    best_valid_loss = float('inf')
    best_epoch = 0
    no_improve_counter = 0  # count epochs with no improvement

    for epoch in range(start_epoch, args.num_epochs):
        if epoch == 0 and rank == 0:
            print("Training Started ...", flush=True)

        train_total, train_vae, train_kl, train_md, train_ssl = train_one_epoch(
            model=model, args=args, optimizer=optimizer, data_loader=train_dataloader,
            train_mode=True, device=gpu, scaler=scaler
        )

        valid_total, valid_vae, valid_kl, valid_md, valid_ssl = train_one_epoch(
            model=model, args=args, optimizer=optimizer, data_loader=valid_dataloader,
            train_mode=False, device=gpu, scaler=scaler
        )

        scheduler.step(valid_total)


        if valid_total < best_valid_loss:
            best_valid_loss = valid_total
            best_epoch = epoch
            no_improve_counter = 0
            if rank == 0:
                best_model_path = os.path.join(get_abs_path(args.save_dir), "best_model.pt")
                torch.save(model.state_dict(), best_model_path)
        else:
            no_improve_counter += 1

        if no_improve_counter >= 50:
            print("Early stopping: no improvement in validation loss for 50 epochs.")
            break

        if rank == 0:
            print(f"Epoch {epoch}:")
            print(f"  Train Total Loss : {train_total:.3f}")
            print(f"  Valid Total Loss : {valid_total:.3f}")
            print(f"  Learning Rate    : {[group['lr'] for group in optimizer.param_groups][0]:.4f}")
            print(f"  Best Epoch So Far: {best_epoch}")
            print("-" * 80)

        if epoch % args.save_every == 0 and rank == 0:
            epoch_model_path = os.path.join(get_abs_path(args.save_dir), f"save_{epoch}.pt")
            torch.save(model.state_dict(), epoch_model_path)

    if rank == 0:
        print(f"Training complete. Best epoch: {best_epoch} with validation loss: {best_valid_loss:.4f}", flush=True)


def main_train():
    now = datetime.now()
    print(f"Train starts at {now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}")

    args = train_args_parser()
    d = vars(args)
    print(text_filling("PARAMETER SETTINGS"), flush=True)
    for a in d:
        print(a, "=", d[a])
    print("#" * 80, flush=True)

    total_gpus = torch.cuda.device_count()
    if args.world_size > 1 and total_gpus > 1:
        args.distributed = True
    else:
        args.distributed = False
        args.world_size = 1

    args.master_port = find_free_port()

    if args.distributed:
        mp.spawn(main_worker, nprocs=args.world_size, args=(args.world_size, args))
    else:
        main_worker(0, 1, args)



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python super_big_code.py [train|evaluate] <other args>")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "train":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        main_train()
    elif cmd == "evaluate":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        main_evaluate()
    else:
        print("Unknown command. Use 'train' or 'evaluate'.")
        sys.exit(1)
