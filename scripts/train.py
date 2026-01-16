"""
Training script for the MLSA Transformer Project.
Architecture adapted from Lecture 6: Transformers.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
from tqdm import tqdm
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dataset import get_dataloaders
from src.model import EncoderLayer, DecoderLayer, EncoderTransf, DecoderTransf, EncoderDecoderTransf

def train(resume_epoch=None):
    # Hyperparameters
    batch_size = 16
    epochs = 10
    lr = 1e-4
    d_model = 256
    n_heads = 8
    n_layers = 4
    ff_units = 512
    dropout = 0.1
    max_code_len = 256
    max_summary_len = 128

    # Device configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Data loading
    train_loader, val_loader, _, tokenizer = get_dataloaders(
        batch_size=batch_size, 
        tokenizer_name="microsoft/codebert-base",
        train_subset=50000,
        val_subset=5000
    )
    vocab_size = tokenizer.vocab_size

    # Model initialization
    enclayer = EncoderLayer(n_heads=n_heads, d_model=d_model, ff_units=ff_units, dropout=dropout)
    declayer = DecoderLayer(n_heads=n_heads, d_model=d_model, ff_units=ff_units, dropout=dropout)
    
    encoder = EncoderTransf(enclayer, n_layers=n_layers, max_len=max_code_len)
    decoder = DecoderTransf(declayer, n_layers=n_layers, max_len=max_summary_len)
    
    model = EncoderDecoderTransf(encoder, decoder, vocab_size, vocab_size, max_len=max_summary_len)
    
    # RESUME LOGIC
    start_epoch = 0
    if resume_epoch is not None:
        ckpt_path = f'models/checkpoint_epoch_{resume_epoch}.pt'
        if os.path.exists(ckpt_path):
            print(f"--- Resuming from Epoch {resume_epoch} ---")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            start_epoch = resume_epoch
        else:
            print(f"!!! Checkpoint {ckpt_path} not found !!!")

    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs('models', exist_ok=True)

    # Training loop starting from resume point
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Prepare decoder inputs and targets
            # dec_in: BOS token + tokens 1 to N-1
            # targets: tokens 1 to N (EOS token)
            dec_in = decoder_input_ids[:, :-1]
            targets = labels[:, 1:].contiguous()

            # Forward pass
            optimizer.zero_grad()
            logits = model(input_ids, dec_in)
            
            # Loss calculation: Model ignores padding tokens (index 1) which 
            # are common due to 256/128 fixed length sequences.
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

            # Progress monitoring
            total_loss += loss.item()
            perplexity = math.exp(loss.item()) if loss.item() < 100 else float('inf')
            pbar.set_postfix({'loss': loss.item(), 'ppl': perplexity})

        avg_train_loss = total_loss / len(train_loader)
        avg_train_ppl = math.exp(avg_train_loss) if avg_train_loss < 100 else float('inf')
        print(f"Epoch {epoch+1} Average Loss: {avg_train_loss:.4f}, Perplexity: {avg_train_ppl:.4f}")

        # Basic validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                decoder_input_ids = batch['decoder_input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                dec_in = decoder_input_ids[:, :-1]
                targets = labels[:, 1:].contiguous()
                
                logits = model(input_ids, dec_in)
                loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), f'models/checkpoint_epoch_{epoch+1}.pt')

if __name__ == "__main__":
    # To start from scratch: train()
    # To resume from epoch 3: train(resume_epoch=3)
    train()
