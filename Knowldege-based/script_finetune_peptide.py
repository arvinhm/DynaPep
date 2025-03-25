#!/usr/bin/env python3
"""
Training script for peptide sequence generation.
It loads the preprocessed data (train/val), defines a dataset and dataloader,
loads a pre-trained transformer backbone (from checkpoint), modifies the final layer for peptide generation,
and runs a training loop with progress bars and loss logging.
"""

import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse

# Define a simple peptide dataset.
class PeptideDataset(Dataset):
    def __init__(self, data_file, max_len=100):
        with open(data_file, "rb") as f:
            self.data = pickle.load(f)
        self.max_len = max_len
        # For training, we use the peptide sequence as target.
        # We assume that you will tokenize the peptide sequence with start/end tokens.
        # Here we use a simple mapping for the 20 standard amino acids.
        self.aa_vocab = {
            'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5,
            'E': 6, 'Q': 7, 'G': 8, 'H': 9, 'I': 10,
            'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15,
            'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20,
            '<pad>': 0, '<start>': 21, '<end>': 22
        }
        self.inv_vocab = {v: k for k, v in self.aa_vocab.items()}

    def tokenize(self, seq):
        # add start and end tokens and pad to max_len
        tokens = [self.aa_vocab['<start>']]
        for aa in seq:
            tokens.append(self.aa_vocab.get(aa, 0))
        tokens.append(self.aa_vocab['<end>'])
        if len(tokens) < self.max_len:
            tokens += [self.aa_vocab['<pad>']] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # We focus on the peptide sequence (chain B)
        item = self.data[idx]
        seq = item['peptide_seq']
        # Optionally, you can also use the receptor (chain A) information for conditioning.
        receptor_seq = item['receptor_seq']
        # Tokenize peptide sequence
        target = self.tokenize(seq)
        # For a conditional generation model, you might also encode receptor info.
        # Here we simply return the target sequence.
        return target

# A dummy Transformer model for peptide generation.
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
        # src shape: (seq_len, batch)
        embedded = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        embedded = self.pos_encoder(embedded)
        output = self.transformer_encoder(embedded, src_mask)
        decoded = self.decoder(output)
        return decoded

# PositionalEncoding class similar to “Attention is All You Need”
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def load_pretrained_weights(model, checkpoint_path):
    """Load pre-trained weights into the model. You may need to adjust keys if necessary."""
    if os.path.isfile(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print("Loaded pre-trained weights from", checkpoint_path)
    else:
        print("Checkpoint not found at", checkpoint_path)
    return model

def train_model(model, train_loader, val_loader, epochs, lr, device, save_path):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    best_val_loss = float("inf")
    
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            # batch shape: (batch, seq_len)
            batch = batch.to(device).transpose(0, 1)  # Transformer expects (seq_len, batch)
            optimizer.zero_grad()
            # Input sequence excluding the last token
            src = batch[:-1, :]
            # Target sequence excluding the first token
            tgt = batch[1:, :]
            # Create a subsequent mask for decoder input
            src_mask = nn.Transformer.generate_square_subsequent_mask(src.size(0)).to(device)
            output = model(src, src_mask)
            loss = criterion(output.view(-1, output.size(-1)), tgt.reshape(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                batch = batch.to(device).transpose(0, 1)
                src = batch[:-1, :]
                tgt = batch[1:, :]
                src_mask = nn.Transformer.generate_square_subsequent_mask(src.size(0)).to(device)
                output = model(src, src_mask)
                loss = criterion(output.view(-1, output.size(-1)), tgt.reshape(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with val loss {best_val_loss:.4f} to {save_path}")
    return model

def main():
    parser = argparse.ArgumentParser(description="Train peptide generation model")
    parser.add_argument("--train_data", type=str, required=True, help="Path to train_data.pkl")
    parser.add_argument("--val_data", type=str, required=True, help="Path to val_data.pkl")
    parser.add_argument("--checkpoint", type=str, default="pretrained_checkpoint.pt", help="Path to pre-trained weights")
    parser.add_argument("--save_model", type=str, default="peptide_finetuned.pt", help="Path to save fine-tuned model")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create datasets and loaders
    train_dataset = PeptideDataset(args.train_data)
    val_dataset = PeptideDataset(args.val_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    vocab_size = len(train_dataset.aa_vocab)
    model = PeptideTransformer(vocab_size=vocab_size)
    model = load_pretrained_weights(model, args.checkpoint)
    print("Starting training...")
    model = train_model(model, train_loader, val_loader, args.epochs, args.lr, device, args.save_model)
    print("Training complete.")

if __name__ == '__main__':
    main()
