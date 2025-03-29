#!/usr/bin/env python
# super_big_code.py
# -------------------------------------------------------
# Combined code from arguments.py, dataset.py, train.py,
# utils.py, model.py, layers.py, evaluate.py
# plus modifications to load pretrained model, handle
# new peptide data, and show a 'reasoning layer'.
# -------------------------------------------------------

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
# ##########################    arguments.py   ################################
# ############################################################################
torch.cuda.empty_cache()

def train_args_parser():
    parser = argparse.ArgumentParser()

    # DEFAULT SETTINGS
    parser.add_argument("--world_size", help="world size", type=int, default=1)
    parser.add_argument("--distributed", help="distributed", action="store_true", default=True)
    parser.add_argument("--autocast", help="autocast", action="store_true", default=False)
    parser.add_argument("--num_workers", help="number of workers", type=int, default=1)
    parser.add_argument("--batch_size", help="batch_size", type=int, default=8)

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
        default=18, # was utils.NUM_LIGAND_ATOM_TYPES
    )
    parser.add_argument(
        "--num_pocket_atom_feature",
        help="pocket atom features",
        type=int,
        default=51, # was utils.NUM_POCKET_ATOM_TYPES
    )
    parser.add_argument("--num_hidden_feature", help="num hidden features", type=int, default=256)
    parser.add_argument("--num_latent_feature", help="num latent features", type=int, default=256)
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
        default=7, # was utils.NUM_INTERACTION_TYPES
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

    # #### ADDED: optional pretrained model argument
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="Path to a pretrained model checkpoint to load at start")

    args = parser.parse_args()
    return args


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
    parser.add_argument("--use_scaffold", help="use_scaffold", action="store_true")
    parser.add_argument("--conditional", help="conditional", action="store_true")
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
# ##########################    utils.py       ################################
# ############################################################################

ATOM_TYPES = ["C", "N", "O", "F", "P", "S", "Cl", "Br"]
INTERACTION_TYPES = ["pipi", "anion", "cation", "hbd", "hba", "hydro"]

NUM_LIGAND_ATOM_TYPES = len(ATOM_TYPES) + 1   # e.g. 8 + 1 = 9
NUM_POCKET_ATOM_TYPES = 51
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
# ##########################    layers.py     #################################
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
            # Expect condition features of dimension num_cond_feature.
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
            # --- Revised conditional branch ---
            # Ensure cond is a tensor of shape (N, num_cond_feature)
            # Case 1: cond is 1D (e.g., shape [cond_dim]), assume it is global and repeat for every node.
            if cond.dim() == 1:
                # Now cond becomes (1, cond_dim) and then expand to (N, cond_dim)
                cond = cond.unsqueeze(0).expand(data["pocket"].h.size(0), -1)
            # Case 2: cond is 2D but the number of rows is 1 (global condition)
            elif cond.dim() == 2 and cond.size(0) == 1 and data["pocket"].h.size(0) > 1:
                cond = cond.expand(data["pocket"].h.size(0), -1)
            # Case 3: If cond is batched with a different number of rows, try to index using the pocket batch vector.
            if cond.size(-1) != self.args.num_cond_feature:
                # If the condition features have too few columns, repeat along the feature dimension.
                # (e.g., if cond has shape (N, 1) but we need (N, 7), repeat it 7 times.)
                factor = self.args.num_cond_feature // cond.size(-1)
                cond = cond.repeat(1, factor)
            # At this point, cond should have shape (N, num_cond_feature)
            # Concatenate the pocket node embeddings with the condition.
            data["pocket"].h_cond = torch.cat([data["pocket"].h, cond], dim=-1)
            # Now, self.cond_mlp expects an input dimension of hidden_feature+num_cond_feature.
            data["pocket"].h = self.cond_mlp(data["pocket"].h_cond)

        # --- Standard message passing below ---
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
    """Example GCL used in SSLModel or other modules."""
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
    """(Optional) for advanced attention in NextDist, etc."""
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
        # block diagonal mask ...
        # (this is advanced usage if you want cross-attn across batch)
        # We'll skip detailed implementation here for brevity.

        # Basic code for demonstration (no large block masks)
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
# ######################   The new "Reasoning Layer"   #######################
# ############################################################################

class ReasoningLayer(nn.Module):
    """
    Revised ReasoningLayer that fuses sequence and MD metrics.

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
            # Batched case
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
# ##########################    model.py      ################################
# ############################################################################

class VariationalEncoder(nn.Module):
    """Simple VAE encoder for a readout of the ligand+pocket features."""
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
    """
    Node embedding and E3II-layer propagation.
    """
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
    """
    Predict p(Type) for each node from node embeddings.
    This version returns per-node predictions of shape [N, num_types],
    matching the ground truth stored in traj["ligand"].type_output.
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
        # Create a series of dense layers that decrease the dimension
        n_dims = list(np.linspace(args.num_hidden_feature, 1, args.num_dense_layers + 1).astype(int))
        for n_in, n_out in zip(n_dims[:-1], n_dims[1:]):
            layers.append(nn.Linear(n_in, n_out))
            layers.append(self.act)
        layers = layers[:-1]  # Remove the last activation if not needed
        if self.last_act is not None:
            layers.append(self.last_act)
        self.dense = nn.Sequential(*layers)

        # Create a fixed embedding for each possible atom type.
        # This has shape [num_ligand_atom_feature, num_ligand_atom_feature]
        self.atom_type = torch.eye(args.num_ligand_atom_feature)
        self.atom_type = nn.Parameter(self.atom_type, requires_grad=False)

    def forward(self, data, key="ligand"):
        # data[key].h is the node representation: shape [N, hidden_dim]
        # Embed each atom type: shape becomes [num_types, hidden_dim]
        embed_type = self.embedding(self.atom_type)  # [num_types, hidden_dim]
        # Expand to broadcast over nodes:
        embed_type = embed_type.unsqueeze(0)          # [1, num_types, hidden_dim]
        repr_type = data[key].h.unsqueeze(1)          # [N, 1, hidden_dim]
        # Multiply node representations with each type embedding:
        mul_type = embed_type * repr_type             # [N, num_types, hidden_dim]
        # Pass through dense layers:
        dense_type = self.dense(mul_type).squeeze(-1)   # [N, num_types]
        # Apply log softmax to get log-probabilities for each node:
        next_type = F.log_softmax(dense_type, dim=-1)   # [N, num_types]
        # **Note:** We removed the scatter_add aggregation.
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
    For semi-supervised learning condition prediction (just an example).
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


# ###########################  The main Model: DeepICL  #######################

class DeepICL(nn.Module):
    """
    Revised DeepICL model that incorporates MD metrics:
      - Global docking score and binding energy.
      - Per-residue RMSD for the ligand (peptide).
      - Per-pocket-residue energy (via an extra head)

    It computes:
      total_loss = VAE_loss + KL_loss (NextType + NextDist) + MD_loss_total + SSL_loss

    MD_loss_total = MD_regression_loss (for docking & binding) + RMSD_loss (for ligand) + Pocket_energy_loss (for pocket)
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = Embedding(args)
        self.vae = VariationalEncoder(args)
        self.conditional = args.conditional
        self.num_cond_feature = args.num_cond_feature if self.conditional else 0

        if args.ssl:
            self.ssl_model = SSLModel(args)

        # Revised reasoning layer that can also optionally accept pocket energy
        self.reasoning_layer = ReasoningLayer(hidden_dim=args.num_hidden_feature, use_rmsd=True)
        self.latent_mlp = nn.Linear(
            args.num_hidden_feature + args.num_latent_feature + self.num_cond_feature,
            args.num_hidden_feature
        )
        self.next_type_ll = NextType(args, self.embedding.l_node_emb)
        self.next_type_lp = NextType(args, self.embedding.l_node_emb)
        self.next_dist_ll = NextDist(args, self.embedding.l_node_emb)
        self.next_dist_lp = NextDist(args, self.embedding.l_node_emb)
        self.kl_loss_fn = nn.KLDivLoss(reduction="none")
        self.md_regressor = nn.Sequential(
            nn.Linear(args.num_hidden_feature + args.num_latent_feature + self.num_cond_feature, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # outputs: predicted docking and binding
        )
        self.rmsd_head = nn.Sequential(
            nn.Linear(args.num_hidden_feature, 1)
        )
        # NEW: Head for per-residue pocket energy
        self.pocket_energy_head = nn.Sequential(
            nn.Linear(args.num_hidden_feature, 1)
        )

    def forward(self, data_dict):
        # Unpack the input dictionary.
        whole, traj, _ = data_dict.values()

        # Retrieve ground-truth MD metrics.
        gt_docking = getattr(whole, "docking_score", 0.0)
        gt_binding = getattr(whole, "binding_energy_kcal_per_mol", 0.0)
        gt_rmsd = getattr(whole, "md_per_residue_rmsd", None)
        if gt_rmsd is None:
            gt_rmsd = whole["ligand"].x.new_zeros(whole["ligand"].num_nodes)
        gt_pocket_energy = getattr(whole["pocket"], "pocket_energy", None)  # may be None

        cond_whole = whole.pocket_prop if self.conditional else None
        cond_traj = traj.pocket_prop if self.conditional else None

        # Embed the whole graph and compute latent representation.
        self.embedding(whole, cond=cond_whole)
        latent_vec, vae_loss = self.vae(whole)
        vae_loss_final = self.args.vae_loss_coeff[1] * vae_loss.sum()

        # Get peptide strings.
        pep_seq = getattr(whole, "pep_seq_str", "")
        pep_smi = getattr(whole, "pep_smi_str", "")
        pep_scf = getattr(whole, "pep_scf_str", "")

        # Ensure batch vectors exist.
        if not hasattr(whole["ligand"], "batch"):
            whole["ligand"].batch = torch.zeros(whole["ligand"].x.size(0), dtype=torch.long, device=whole["ligand"].x.device)
        if not hasattr(whole["pocket"], "batch"):
            whole["pocket"].batch = torch.zeros(whole["pocket"].x.size(0), dtype=torch.long, device=whole["pocket"].x.device)

        # Apply reasoning layer on whole graph.
        whole["ligand"].h = self.reasoning_layer(
            whole["ligand"].h, pep_seq, pep_smi, pep_scf,
            docking=gt_docking, binding=gt_binding, rmsd=gt_rmsd,
            batch=whole["ligand"].batch
        )
        whole["pocket"].h = self.reasoning_layer(
            whole["pocket"].h, pep_seq, pep_smi, pep_scf,
            docking=gt_docking, binding=gt_binding, rmsd=gt_pocket_energy,
            batch=whole["pocket"].batch
        )

        # Process trajectory graph.
        self.embedding(traj, cond=cond_traj)
        if not hasattr(traj["ligand"], "batch"):
            traj["ligand"].batch = torch.zeros(traj["ligand"].x.size(0), dtype=torch.long, device=traj["ligand"].x.device)
        if not hasattr(traj["pocket"], "batch"):
            traj["pocket"].batch = torch.zeros(traj["pocket"].x.size(0), dtype=torch.long, device=traj["pocket"].x.device)

        traj["ligand"].h = self.reasoning_layer(
            traj["ligand"].h, pep_seq, pep_smi, pep_scf,
            docking=gt_docking, binding=gt_binding, rmsd=gt_rmsd,
            batch=traj["ligand"].batch
        )
        traj["pocket"].h = self.reasoning_layer(
            traj["pocket"].h, pep_seq, pep_smi, pep_scf,
            docking=gt_docking, binding=gt_binding, rmsd=gt_pocket_energy,
            batch=traj["pocket"].batch
        )

        # Merge latent vector with trajectory pocket features.
        psize = traj["pocket"].h.size(0)
        if self.conditional and (cond_traj is not None):
            # Ensure cond_traj has shape (psize, num_cond_feature)
            if cond_traj.size(-1) != self.args.num_cond_feature:
                # If condition has only 1 feature, repeat it along the last dimension.
                cond_traj = cond_traj.repeat(1, self.args.num_cond_feature)
            cat_vec = torch.cat([traj["pocket"].h, cond_traj, latent_vec.repeat(psize, 1)], dim=-1)
        else:
            cat_vec = torch.cat([traj["pocket"].h, latent_vec.repeat(psize, 1)], dim=-1)
        traj["pocket"].h = self.latent_mlp(cat_vec)

        # Next-Type and Next-Dist losses.
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

        # --- Revised MD Regression Loss Block ---
        md_pred = self.md_regressor(cat_vec)  # shape: [psize, 2]
        if isinstance(gt_docking, torch.Tensor) and gt_docking.numel() > 1:
            pocket_batch = traj["pocket"].batch  # shape: [psize]
            target_docking = gt_docking[pocket_batch]
            target_binding = gt_binding[pocket_batch]
            db_sum = (target_docking + target_binding).mean().item()
        else:
            target_docking = md_pred[:, 0].new_full((psize,), float(gt_docking))
            target_binding = md_pred[:, 1].new_full((psize,), float(gt_binding))
            db_sum = float(gt_docking + gt_binding)
        alpha = 0.01
        weight_factor = math.exp(-alpha * db_sum)
        mse_dock = F.mse_loss(md_pred[:, 0], target_docking, reduction="mean")
        mse_bind = F.mse_loss(md_pred[:, 1], target_binding, reduction="mean")
        md_reg_loss = weight_factor * (mse_dock + mse_bind)

        # RMSD Loss for ligand nodes.
        pred_rmsd = self.rmsd_head(whole["ligand"].h).squeeze(-1)
        rmsd_loss = F.mse_loss(pred_rmsd, gt_rmsd, reduction="mean")

        # Pocket Energy Loss (if ground truth is provided).
        if gt_pocket_energy is not None:
            pred_pocket_energy = self.pocket_energy_head(whole["pocket"].h).squeeze(-1)
            pocket_energy_loss = F.mse_loss(pred_pocket_energy, gt_pocket_energy, reduction="mean")
        else:
            pocket_energy_loss = 0.0

        md_loss_total = md_reg_loss + rmsd_loss + pocket_energy_loss

        # Optional SSL loss.
        ssl_loss = torch.tensor(0.0, device=whole["ligand"].x.device)
        if self.args.ssl:
            ssl_out = self.ssl_model(whole)
            ssl_loss = self.ssl_loss_fn(ssl_out, cond_whole.argmax(-1)).mean()

        total_loss = vae_loss_final + kl_loss + md_loss_total + ssl_loss

        return total_loss, vae_loss_final, kl_loss, md_loss_total, ssl_loss


# ############################################################################
# ##########################    dataset.py    ################################
# ############################################################################

class DataProcessor(object):
    def __init__(self, args, mode="train"):
        self.args = args
        self.mode = mode
        self.radial_cutoff = args.dist_one_hot_param1[1]
        self.distance_expand1 = SoftOneHot(*args.dist_one_hot_param1, gamma=args.gamma1)
        self.distance_expand2 = SoftOneHot(*args.dist_one_hot_param2, gamma=args.gamma2, normalize=True)
        self.num_dim = 3
        self.use_scaffold = getattr(args, "use_scaffold", False)
        self.o_Z = torch.zeros(1, args.num_ligand_atom_feature)
        self.o_R = torch.zeros(1, self.num_dim)
        self.t_Z = torch.zeros(1, args.num_ligand_atom_feature)
        self.t_Z[:, args.num_ligand_atom_feature - 1] = 1.0
        self.f_Z = torch.zeros(1, args.num_ligand_atom_feature)
        self.f_Z[:, args.num_ligand_atom_feature - 1] = 1.0
        self.distance_dummy = torch.zeros(1, args.dist_one_hot_param2[-1])

    def get_input_from_data(self, data_dict):
        """
        data_dict is the dictionary loaded from PKL:
        {
            "ligand_type": <np.array>,
            "ligand_coord": <np.array>,
            "pocket_type": <np.array>,
            "pocket_coord": <np.array>,
            "ligand_adj": <np.array>,
            "pep_seq_str": <str>,
            "pep_smi": <str>,
            "ligand_n": <int>,
            "pocket_cond": <np.array>,
            "center_of_mass": <np.array>,
            "docking_score": <float>,
            "binding_energy_kcal_per_mol": <float>,
            "md_global_rmsd_nm": <float>,
            "md_mean_distance_nm": <float>,
            "md_per_residue_rmsd": <np.array>,
            ...
        }
        """
        import torch
        from torch_geometric.data import HeteroData, Batch
        from torch_geometric.nn import radius_graph
        from torch import cdist, topk
        
        # 1) Extract fields from the dictionary:
        ligand_type  = data_dict["ligand_type"]
        ligand_coord = data_dict["ligand_coord"]
        pocket_type  = data_dict["pocket_type"]
        pocket_coord = data_dict["pocket_coord"]
        ligand_adj   = data_dict["ligand_adj"]
        pep_seq_str  = data_dict["pep_seq_str"]
        # The PKL uses "pep_smi", not "pep_smi_str":
        pep_smi_str  = data_dict["pep_smi"]
        ligand_n     = data_dict["ligand_n"]
        pocket_cond  = data_dict["pocket_cond"]
        center_of_mass = data_dict["center_of_mass"]
        
        # (Optionally) also read these MD metrics from data_dict:
        docking_score = data_dict["docking_score"]
        binding_energy = data_dict["binding_energy_kcal_per_mol"]
        md_rmsd_array = data_dict["md_per_residue_rmsd"]  # shape [N_ligand]
        # You can store them in 'whole' for the model to see:
        
        # 2) Build your "whole" HeteroData
        whole = HeteroData()
        whole["ligand"].x   = torch.tensor(ligand_type,  dtype=torch.float)
        whole["ligand"].pos = torch.tensor(ligand_coord, dtype=torch.float)
        whole["pocket"].x   = torch.tensor(pocket_type,  dtype=torch.float)
        whole["pocket"].pos = torch.tensor(pocket_coord, dtype=torch.float)
        
        whole["ligand"].num_nodes = whole["ligand"].x.size(0)
        whole["pocket"].num_nodes = whole["pocket"].x.size(0)
        
        # 3) Store the string fields:
        whole.pep_seq_str   = pep_seq_str
        whole.pep_smi_str   = pep_smi_str
        whole.pep_scf_str   = ""  # if none or if you want
        whole.docking_score = float(docking_score)
        whole.binding_energy_kcal_per_mol = float(binding_energy)
        whole.md_per_residue_rmsd = torch.tensor(md_rmsd_array, dtype=torch.float)
        
        # Also store pocket_cond as a property:
        whole.pocket_prop   = torch.tensor(pocket_cond, dtype=torch.float)
        
        # 4) radius_graph, etc. (like your existing code)
        l2l_index = radius_graph(whole["ligand"].pos, r=self.radial_cutoff)
        p2p_index = radius_graph(whole["pocket"].pos, r=self.radial_cutoff)
        if l2l_index.numel() == 0:
            import torch
            l2l_index = torch.zeros((2,1), dtype=torch.long)
        if p2p_index.numel() == 0:
            import torch
            p2p_index = torch.zeros((2,1), dtype=torch.long)
        
        whole["ligand","l2l","ligand"].edge_index = l2l_index
        whole["pocket","p2p","pocket"].edge_index = p2p_index
        
        # distance expansions
        l2l_weight = (whole["ligand"].pos[l2l_index[0]] - whole["ligand"].pos[l2l_index[1]]).norm(dim=-1)
        p2p_weight = (whole["pocket"].pos[p2p_index[0]] - whole["pocket"].pos[p2p_index[1]]).norm(dim=-1)
        l2l_attr   = self.distance_expand1(l2l_weight)[0]
        p2p_attr   = self.distance_expand1(p2p_weight)[0]
        
        whole["l2l"].edge_weight = l2l_weight
        whole["p2p"].edge_weight = p2p_weight
        whole["l2l"].edge_attr   = l2l_attr
        whole["p2p"].edge_attr   = p2p_attr
        
        # ligand->pocket kNN
        l_n = whole["ligand"].x.size(0)
        p_n = whole["pocket"].x.size(0)
        if p_n == 0 or l_n == 0:
            pl_index = torch.zeros((2,1), dtype=torch.long)
            p2l_weight = torch.zeros((1,), dtype=torch.float)
            p2l_attr   = torch.zeros((1,self.args.dist_one_hot_param1[-1]), dtype=torch.float)
        else:
            dist_mat   = cdist(whole["ligand"].pos, whole["pocket"].pos)
            k_val      = self.args.k if self.args.k else 8
            k_val      = min(k_val, p_n)
            p_knn_index= topk(dist_mat, k=k_val, largest=False, dim=-1)[1]
            p_index    = p_knn_index.view(-1)
            l_index    = torch.arange(l_n).unsqueeze(-1).repeat(1,k_val).view(-1)
            pl_index   = torch.stack([p_index, l_index], dim=0)
            p2l_weight = (whole["pocket"].pos[p_index] - whole["ligand"].pos[l_index]).norm(dim=-1)
            p2l_attr   = self.distance_expand1(p2l_weight)[0]
        
        whole["pocket","p2l","ligand"].edge_index = pl_index
        whole["p2l"].edge_weight = p2l_weight
        whole["p2l"].edge_attr   = p2l_attr
        
        # pre-batch
        whole = Batch.from_data_list([whole])
        
        # 5) Build the "traj" placeholder
        traj = whole.clone()
        
        # Provide "type_output" and "dist_output" to lig/pocket
        ligand_count = traj["ligand"].x.size(0)
        pocket_count = traj["pocket"].x.size(0)
        
        # Suppose we fill them with uniform distributions
        type_output_ligand = torch.ones(ligand_count, self.args.num_ligand_atom_feature)
        type_output_ligand = type_output_ligand / type_output_ligand.sum(dim=-1, keepdim=True)
        
        dist_output_ligand = torch.ones(ligand_count, self.args.dist_one_hot_param2[-1])
        dist_output_ligand = dist_output_ligand / dist_output_ligand.sum(dim=-1, keepdim=True)
        
        type_output_pocket = torch.ones(pocket_count, self.args.num_ligand_atom_feature)
        type_output_pocket = type_output_pocket / type_output_pocket.sum(dim=-1, keepdim=True)
        
        dist_output_pocket = torch.ones(pocket_count, self.args.dist_one_hot_param2[-1])
        dist_output_pocket = dist_output_pocket / dist_output_pocket.sum(dim=-1, keepdim=True)
        
        traj["ligand"].type_output    = type_output_ligand
        traj["ligand"].dist_ll_output = dist_output_ligand
        traj["ligand"].mask           = torch.ones(ligand_count)
        
        traj["pocket"].type_output    = type_output_pocket
        traj["pocket"].dist_lp_output = dist_output_pocket
        traj["pocket"].mask           = torch.ones(pocket_count)
        
        # Keep the same pocket_prop as in 'whole'
        traj.pocket_prop = whole.pocket_prop.clone()
        
        # batch indices
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

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx):
        key = self.key_list[idx]
        full_path = os.path.join(self.data_dir, key)
        import pickle
        with open(full_path, "rb") as f:
            data = pickle.load(f)
        whole, traj = self.processor.get_input_from_data(data)
        return {"whole": whole, "traj": traj, "key": key}

# ############################################################################
# ##########################    evaluate.py   ################################
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

        # 1) check validity
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

        # 2) uniqueness
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

        # 3) novelty
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
# ##########################    train.py      ################################
# ############################################################################

def sample_to_device(sample, device):
    sample["whole"].to(device)
    sample["traj"].to(device)


from contextlib import nullcontext
import torch.distributed as dist

def train_one_epoch(
    model,
    args,
    optimizer,
    data_loader,
    train_mode: bool,
    device,
    scaler=None
):
    """
    train_mode=True  -> training step
    train_mode=False -> validation step (no gradient)
    """
    if train_mode:
        model.train()
    else:
        model.eval()

    import numpy as np

    # We store *sums* of local losses rather than a list.
    total_loss_sum = 0.0
    vae_loss_sum   = 0.0
    type_loss_sum  = 0.0
    dist_loss_sum  = 0.0
    ssl_loss_sum   = 0.0
    local_count    = 0   # how many batches this rank sees

    # Only rank 0 shows a progress bar
    if not dist.is_initialized() or dist.get_rank() == 0:
        from tqdm import tqdm
        pbar = tqdm(data_loader, total=len(data_loader),
                    desc=("Train" if train_mode else "Valid"), leave=False)
    else:
        pbar = data_loader

    # Use torch.no_grad() for validation, else do nothing
    ctx = torch.no_grad() if not train_mode else nullcontext()
    with ctx:
        for sample in pbar:
            local_count += 1

            # Move data to device
            sample["whole"].to(device)
            sample["traj"].to(device)

            # Forward pass
            if train_mode and args.autocast:
                import torch.cuda.amp as amp
                with amp.autocast():
                    total_loss, vae_loss, type_loss, dist_loss, ssl_loss = model(sample)
            else:
                total_loss, vae_loss, type_loss, dist_loss, ssl_loss = model(sample)

            # Backprop if training
            if train_mode:
                optimizer.zero_grad()
                if args.autocast:
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()

            # Accumulate local sums
            total_loss_sum += total_loss.item()
            vae_loss_sum   += vae_loss.item()
            type_loss_sum  += type_loss.item()
            dist_loss_sum  += dist_loss.item()
            if args.ssl and ssl_loss is not None:
                ssl_loss_sum += ssl_loss.item()

            # If rank 0 has a progress bar, update it
            if hasattr(pbar, "set_postfix"):
                avg_so_far = total_loss_sum / local_count
                pbar.set_postfix({'Loss': f"{avg_so_far:.3f}"})

    # AGGREGATE across all ranks (only matters if distributed)
    if dist.is_initialized():
        sum_tensor = torch.tensor([
            total_loss_sum,
            vae_loss_sum,
            type_loss_sum,
            dist_loss_sum,
            ssl_loss_sum
        ], device=device, dtype=torch.float)
        count_tensor = torch.tensor([local_count], device=device, dtype=torch.float)

        # all_reduce so every rank gets the grand total
        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

        # Compute global means
        global_sum = sum_tensor.cpu().numpy()  # shape [5]
        global_count = count_tensor.item()

        if global_count < 1e-8:  # avoid division by zero
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
    # Adjust GPU if necessary.
    total_gpus = torch.cuda.device_count()
    if gpu >= total_gpus:
        gpu = 0
    rank = gpu

    # Set up distributed environment.
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = args.master_port
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    print(text_filling(f"Finished Setting DDP: CUDA:{rank}"), flush=True)
    torch.cuda.set_device(gpu)

    # Build datasets.
    train_dataset = PDBbindDataset(args, mode="train")
    valid_dataset = PDBbindDataset(args, mode="valid")
    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=args.world_size, rank=rank, shuffle=False)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        sampler=valid_sampler,
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

    # Build the model.
    model = DeepICL(args)
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

    # Set up the optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = None
    if args.autocast:
        import torch.cuda.amp as amp
        scaler = amp.GradScaler()

    # VAE annealing parameters.
    vae_coeff_init, vae_coeff_final = args.vae_loss_coeff
    start_epoch = int(args.restart_file.split("_")[-1].split(".")[0]) if args.restart_file else 0

    # Initialize best model tracking.
    best_model_loss = float('inf')
    best_epoch = 0
    min_valid_loss = float('inf')
    lr_tick = 0

    # Epoch loop.
    for epoch in range(start_epoch, args.num_epochs):
        if epoch == 0 and rank == 0:
            print(
                "EPOCH || TRA_VAE | TRA_SSL | TRA_TYPE | TRA_DIST | TRA_TOT || VAL_VAE | VAL_SSL | VAL_TYPE | VAL_DIST | VAL_TOT || TIME/EPOCH | LR | BEST_EPOCH",
                flush=True
            )

        # Update the samplers for this epoch.
        train_sampler.set_epoch(epoch)
        valid_sampler.set_epoch(epoch)

        # Update VAE loss coefficient (KL annealing).
        args.vae_coeff = vae_coeff_final + (vae_coeff_init - vae_coeff_final) * ((1 - args.vae_loss_beta) ** epoch)

        st = time.time()
        # Run training and validation for one epoch.
        train_total, train_vae, train_type, train_dist, train_ssl = train_one_epoch(
            model=model, args=args, optimizer=optimizer, data_loader=train_dataloader,
            train_mode=True, device=gpu, scaler=scaler
        )
        valid_total, valid_vae, valid_type, valid_dist, valid_ssl = train_one_epoch(
            model=model, args=args, optimizer=optimizer, data_loader=valid_dataloader,
            train_mode=False, device=gpu, scaler=scaler
        )
        et = time.time()

        # Update best validation loss and save best model.
        if valid_total < min_valid_loss:
            min_valid_loss = valid_total
            best_epoch = epoch
            lr_tick = 0
            if rank == 0:
                best_model_path = os.path.join(get_abs_path(args.save_dir), "best_model.pt")
                torch.save(model.state_dict(), best_model_path)
        else:
            lr_tick += 1

        # Adjust learning rate if no improvement.
        if lr_tick >= args.lr_tolerance:
            for param_group in optimizer.param_groups:
                lr_current = param_group["lr"]
                if lr_current > args.lr_min:
                    param_group["lr"] = lr_current * args.lr_decay

        if lr_tick > 20:
            print("No more improvement, stopping training.")
            sys.exit()

        # Print epoch stats.
        if rank == 0:
            print(
                f"{epoch} || {train_vae:.3f} | {train_ssl:.3f} | {train_type:.3f} | {train_dist:.3f} | {train_total:.3f} || "
                f"{valid_vae:.3f} | {valid_ssl:.3f} | {valid_type:.3f} | {valid_dist:.3f} | {valid_total:.3f} || "
                f"{(et - st):.2f} | {[group['lr'] for group in optimizer.param_groups][0]:.4f} | {best_epoch}",
                flush=True
            )

        # Save model checkpoint for this epoch.
        if epoch % args.save_every == 0 and rank == 0:
            epoch_model_path = os.path.join(get_abs_path(args.save_dir), f"save_{epoch}.pt")
            torch.save(model.state_dict(), epoch_model_path)

    if rank == 0:
        print(f"Training complete. Best epoch: {best_epoch} with validation loss: {min_valid_loss:.4f}", flush=True)


def main_train():
    now = datetime.now()
    print(f"Train starts at {now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}")

    args = train_args_parser()
    d = vars(args)
    print(text_filling("PARAMETER SETTINGS"), flush=True)
    for a in d:
        print(a, "=", d[a])
    print("#" * 80, flush=True)

    # If you want multi-GPU, you must actually have more than 1 GPU available.
    # Otherwise, force single-GPU or CPU fallback.
    total_gpus = torch.cuda.device_count()
    if args.world_size > 1 and total_gpus > 1:
        args.distributed = True
    else:
        args.distributed = False
        args.world_size = 1

    # Always pick a free port using your function
    args.master_port = find_free_port()

    if args.distributed:
        mp.spawn(main_worker, nprocs=args.world_size, args=(args.world_size, args))
    else:
        main_worker(0, 1, args)



# ############################################################################
# ##########################   The Single Entry   ############################
# ############################################################################

if __name__ == "__main__":
    """
    This single file can do:
    - Train: python super_big_code.py train --save_dir ... 
    - Evaluate: python super_big_code.py evaluate --result_dir ...
    Or you can handle subcommands, etc.
    """
    if len(sys.argv) < 2:
        print("Usage: python super_big_code.py [train|evaluate] <other args>")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "train":
        # remove "train" from sys.argv for the parser
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        main_train()
    elif cmd == "evaluate":
        # remove "evaluate"
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        main_evaluate()
    else:
        print("Unknown command. Use 'train' or 'evaluate'.")
        sys.exit(1)
