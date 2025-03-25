#!/usr/bin/env python3
"""
Sampling script for peptide generation with receptor conditioning, reproducibility,
candidate ranking, diversity, and uniqueness filtering.
The script loads a fine–tuned model and generates candidate peptides (with nucleus sampling if desired),
enforcing that peptide lengths are within a specified range.
It then deduplicates the candidates (keeping the best scoring version of each unique peptide)
and outputs the top unique candidates.
"""

import os
import math
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import random
import numpy as np

# ---------------------------
# PositionalEncoding and PeptideTransformer definitions (as used in training)
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class PeptideTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=3, dropout=0.1):
        super(PeptideTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
    def forward(self, src, src_mask=None):
        embedded = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        embedded = self.pos_encoder(embedded)
        output = self.transformer_encoder(embedded, src_mask)
        decoded = self.decoder(output)
        return decoded

# ---------------------------
# Utility functions for model loading and peptide generation
# ---------------------------
def load_model(model, model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def nucleus_sample(probs, top_p):
    """
    Perform nucleus (top-p) sampling.
    probs: Tensor of probabilities.
    top_p: Cumulative probability threshold.
    Returns a tuple (token, token_probability).
    """
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
    # Identify tokens to keep: all tokens until cumulative probability exceeds top_p.
    cutoff = cumulative_probs > top_p
    if cutoff[0]:
        cutoff[0] = False  # ensure at least one token is kept
    filtered_probs = sorted_probs[~cutoff]
    filtered_indices = sorted_indices[~cutoff]
    # Renormalize probabilities.
    filtered_probs = filtered_probs / filtered_probs.sum()
    sample_idx = torch.multinomial(filtered_probs, num_samples=1).item()
    return filtered_indices[sample_idx].item(), filtered_probs[sample_idx].item()

def generate_peptide(model, min_length, max_length, start_token, end_token, device,
                     temperature=1.0, greedy=False, top_p=1.0):
    """
    Autoregressively generate a peptide sequence token-by-token,
    enforcing that the peptide (excluding the <start> token) is at least `min_length`
    and at most `max_length`. The <end> token is disallowed until min_length is reached.
    Accumulates the cumulative log probability of the generated tokens.
    Returns a tuple: (generated_tokens, cumulative_log_prob).
    """
    generated = [start_token]
    cumulative_log_prob = 0.0
    max_steps = max_length + 10  # safety net in case generation does not terminate
    for i in range(max_steps):
        if len(generated) - 1 >= max_length:
            generated.append(end_token)
            break

        input_seq = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(1)
        src_mask = nn.Transformer.generate_square_subsequent_mask(input_seq.size(0)).to(device)
        logits = model(input_seq, src_mask)
        next_logits = logits[-1, 0] / temperature

        # Disallow the <end> token until min_length is reached.
        if len(generated) - 1 < min_length:
            next_logits[end_token] = -float('inf')

        probs = torch.softmax(next_logits, dim=-1)
        if greedy:
            next_token = torch.argmax(probs).item()
            token_prob = probs[next_token].item()
        else:
            if top_p < 1.0:
                next_token, token_prob = nucleus_sample(probs, top_p)
            else:
                next_token = torch.multinomial(probs, num_samples=1).item()
                token_prob = probs[next_token].item()

        cumulative_log_prob += torch.log(torch.tensor(token_prob)).item()
        generated.append(next_token)

        if next_token == end_token and (len(generated) - 1) >= min_length:
            break

    return generated, cumulative_log_prob

def tokens_to_sequence(tokens, inv_vocab):
    """
    Convert a list of token indices to a peptide sequence string,
    skipping special tokens like <start> and <pad>, and stopping at <end>.
    """
    seq = ""
    for tok in tokens:
        if inv_vocab[tok] in ['<start>', '<pad>']:
            continue
        if inv_vocab[tok] == '<end>':
            break
        seq += inv_vocab[tok]
    return seq

# ---------------------------
# Functions for reading and processing a PDB file
# ---------------------------
def read_pdb_file(pdb_file):
    """
    Reads a PDB file and returns a list of atom coordinates.
    This simple parser extracts coordinates from ATOM and HETATM records.
    """
    atoms = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    atoms.append((x, y, z))
                except Exception:
                    continue
    return atoms

def extract_pocket(atoms, center, box_size):
    """
    Extract atoms within a cuboidal pocket defined by a center and box dimensions.
    center: tuple (center_x, center_y, center_z)
    box_size: tuple (size_x, size_y, size_z)
    """
    cx, cy, cz = center
    sx, sy, sz = box_size
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    pocket_atoms = []
    for atom in atoms:
        x, y, z = atom
        if (cx - hx <= x <= cx + hx and
            cy - hy <= y <= cy + hy and
            cz - hz <= z <= cz + hz):
            pocket_atoms.append(atom)
    return pocket_atoms

# ---------------------------
# Main script
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Peptide sampling with receptor conditioning, reproducibility, candidate ranking, diversity, and uniqueness filtering")
    
    # Model and generation parameters
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned model checkpoint")
    parser.add_argument("--min_length", type=int, default=4,
                        help="Minimum peptide length (number of amino acids)")
    parser.add_argument("--max_length", type=int, default=12,
                        help="Maximum peptide length (number of amino acids)")
    parser.add_argument("--greedy", action="store_true", help="Use greedy sampling")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    # Candidate generation and ranking parameters
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility")
    parser.add_argument("--num_candidates", type=int, default=100000000,
                        help="Number of candidate peptides to generate before ranking")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Number of top unique peptides to output")
    parser.add_argument("--exploration_depth", type=float, default=1.0,
                        help="Fraction of candidate space to explore (1.0 = full, 0.1 = 10%)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Nucleus sampling threshold (set <1.0 for more diversity)")
    
    # PDB-based receptor conditioning parameters
    parser.add_argument("--pdb_file", type=str, help="Path to the receptor PDB file")
    parser.add_argument("--center_x", type=float, help="X coordinate of the pocket center")
    parser.add_argument("--center_y", type=float, help="Y coordinate of the pocket center")
    parser.add_argument("--center_z", type=float, help="Z coordinate of the pocket center")
    parser.add_argument("--box_size", type=float, nargs=3, metavar=('size_x', 'size_y', 'size_z'),
                        help="Dimensions of the pocket grid (size_x, size_y, size_z)")
    
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Set up the amino acid vocabulary (same as used during training)
    aa_vocab = {
        '<pad>': 0, '<start>': 21, '<end>': 22,
        'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5,
        'E': 6, 'Q': 7, 'G': 8, 'H': 9, 'I': 10,
        'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15,
        'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20
    }
    inv_vocab = {v: k for k, v in aa_vocab.items()}
    vocab_size = len(aa_vocab)
    
    # Load the fine-tuned model
    model = PeptideTransformer(vocab_size=vocab_size)
    model = load_model(model, args.model_path, device)
    
    # Receptor conditioning: if PDB file and pocket parameters are provided, process the file.
    receptor_features = None
    if args.pdb_file and args.center_x is not None and args.center_y is not None \
       and args.center_z is not None and args.box_size:
        atoms = read_pdb_file(args.pdb_file)
        center = (args.center_x, args.center_y, args.center_z)
        pocket_atoms = extract_pocket(atoms, center, tuple(args.box_size))
        print(f"Number of atoms in the pocket: {len(pocket_atoms)}")
        # For demonstration, we use the count as a dummy receptor feature.
        receptor_features = len(pocket_atoms)
    
    # Determine the effective number of candidates to generate.
    effective_candidates = int(args.num_candidates * args.exploration_depth)
    print(f"Generating {effective_candidates} candidate peptides...")
    
    candidates = []
    start_token = aa_vocab['<start>']
    end_token = aa_vocab['<end>']
    
    # Generate candidate peptides with their cumulative log probability scores.
    for _ in tqdm(range(effective_candidates), desc="Generating candidates"):
        tokens, score = generate_peptide(
            model, args.min_length, args.max_length,
            start_token, end_token, device,
            temperature=args.temperature, greedy=args.greedy, top_p=args.top_p
        )
        seq = tokens_to_sequence(tokens, inv_vocab)
        candidates.append((seq, score))
    
    # Deduplicate: for each unique peptide, keep the best (highest) score.
    unique_candidates = {}
    for seq, score in candidates:
        if seq not in unique_candidates or score > unique_candidates[seq]:
            unique_candidates[seq] = score

    # Rank unique candidates by score (closer to zero is better).
    ranked_candidates = sorted(unique_candidates.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {args.top_k} unique candidate peptides:")
    for i, (pep, score) in enumerate(ranked_candidates[:args.top_k], start=1):
        print(f"{i}. Sequence: {pep}  |  Score: {score:.4f}")
    
    if len(unique_candidates) < args.top_k:
        print(f"\nNote: Only {len(unique_candidates)} unique peptides were generated.")

if __name__ == '__main__':
    main()
