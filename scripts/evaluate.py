"""
Evaluation script for the MLSA Transformer Project.
Architecture adapted from Lecture 6: Transformers.
LLM Help: Code snippets and structure assisted by Antigravity (AI).
"""

import torch
import torch.nn as nn
import math
from tqdm import tqdm
import os
import sys
from torchmetrics.text import BLEUScore, ROUGEScore

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dataset import get_dataloaders
from src.model import EncoderLayer, DecoderLayer, EncoderTransf, DecoderTransf, EncoderDecoderTransf

def evaluate(checkpoint_path):
    # Hyperparameters (must match training)
    batch_size = 16
    d_model = 256
    n_heads = 8
    n_layers = 4
    ff_units = 512
    dropout = 0.1
    max_code_len = 256
    max_summary_len = 128

    # Device configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loading
    _, _, test_loader, tokenizer = get_dataloaders(
        batch_size=batch_size, 
        tokenizer_name="microsoft/codebert-base"
    )
    vocab_size = tokenizer.vocab_size

    # Model initialization
    enclayer = EncoderLayer(n_heads=n_heads, d_model=d_model, ff_units=ff_units, dropout=dropout)
    declayer = DecoderLayer(n_heads=n_heads, d_model=d_model, ff_units=ff_units, dropout=dropout)
    
    encoder = EncoderTransf(enclayer, n_layers=n_layers, max_len=max_code_len)
    decoder = DecoderTransf(declayer, n_layers=n_layers, max_len=max_summary_len)
    
    model = EncoderDecoderTransf(encoder, decoder, vocab_size, vocab_size, max_len=max_summary_len)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    bleu = BLEUScore()
    rouge = ROUGEScore()
    predictions = []
    references = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)

            # Shift for loss calculation
            dec_in = decoder_input_ids[:, :-1]
            targets = labels[:, 1:].contiguous()
            
            logits = model(input_ids, dec_in)
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            total_loss += loss.item()

            # Greedy decoding for text metrics
            generated = torch.zeros((input_ids.size(0), 1), dtype=torch.long, device=device).fill_(tokenizer.bos_token_id)
            
            for _ in range(max_summary_len - 1):
                logits_inf = model(input_ids, generated)
                next_token_logits = logits_inf[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)
                
                if (next_token == tokenizer.eos_token_id).all():
                    break
            
            # Decode sequences
            pred_text = tokenizer.batch_decode(generated, skip_special_tokens=True)
            ref_text = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            predictions.extend(pred_text)
            references.extend([[ref] for ref in ref_text])

    avg_loss = total_loss / len(test_loader)
    ppl = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    
    bleu_score = bleu(predictions, references)
    rouge_score = rouge(predictions, references)
    
    print(f"\n--- Evaluation Results ---")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Perplexity: {ppl:.4f}")
    print(f"BLEU Score: {bleu_score.item():.4f}")
    print(f"ROUGE-L Score: {rouge_score['rougeL_fmeasure'].item():.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <checkpoint_path>")
    else:
        evaluate(sys.argv[1])
