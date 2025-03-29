#!/usr/bin/env python

"""
split_pkl_keys.py

Script to read all .pkl files from data_dir, shuffle them, and split
into train/valid/test sets (70/20/10 by default). Then save the lists of
keys (filenames) into .pkl files under key_dir:
    - train_keys.pkl
    - valid_keys.pkl
    - test_keys.pkl
"""

import os
import sys
import argparse
import random
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing .pkl files.")
    parser.add_argument("--key_dir", type=str, required=True,
                        help="Directory to store train/valid/test keys.")
    parser.add_argument("--train_ratio", type=float, default=0.70,
                        help="Fraction of data for train set.")
    parser.add_argument("--valid_ratio", type=float, default=0.20,
                        help="Fraction of data for valid set.")
    parser.add_argument("--test_ratio", type=float, default=0.10,
                        help="Fraction of data for test set.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling.")
    args = parser.parse_args()

    # Make sure the ratios sum to 1.0 (or close to it)
    total_ratio = args.train_ratio + args.valid_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Warning: The sum of train/valid/test ratio = {total_ratio}, not 1.0!")
        print("Continuing anyway...")

    # Gather all PKL files
    all_pkl_files = []
    for fname in os.listdir(args.data_dir):
        if fname.endswith(".pkl"):
            all_pkl_files.append(fname)
    all_pkl_files.sort()

    # Shuffle with a given seed
    random.seed(args.seed)
    random.shuffle(all_pkl_files)

    n_total = len(all_pkl_files)
    if n_total == 0:
        print("No .pkl files found in data_dir, exiting.")
        sys.exit()

    # Compute split sizes
    n_train = int(n_total * args.train_ratio)
    n_valid = int(n_total * args.valid_ratio)
    # Remainder for test:
    n_test = n_total - n_train - n_valid

    # Split
    train_keys = all_pkl_files[:n_train]
    valid_keys = all_pkl_files[n_train : n_train + n_valid]
    test_keys = all_pkl_files[n_train + n_valid :]

    # Print stats
    print("Total PKL files: ", n_total)
    print(f"Train: {len(train_keys)}  Valid: {len(valid_keys)}  Test: {len(test_keys)}")

    # Make sure key_dir exists
    os.makedirs(args.key_dir, exist_ok=True)

    # Save
    train_pkl_path = os.path.join(args.key_dir, "train_keys.pkl")
    valid_pkl_path = os.path.join(args.key_dir, "valid_keys.pkl")
    test_pkl_path  = os.path.join(args.key_dir, "test_keys.pkl")

    with open(train_pkl_path, "wb") as f:
        pickle.dump(train_keys, f)
    with open(valid_pkl_path, "wb") as f:
        pickle.dump(valid_keys, f)
    with open(test_pkl_path, "wb") as f:
        pickle.dump(test_keys, f)

    print("Saved key files to:", args.key_dir)

if __name__ == "__main__":
    main()

