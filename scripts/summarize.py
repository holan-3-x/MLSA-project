"""
Inference script for the MLSA Transformer Project.
Architecture adapted from Lecture 6: Transformers.
LLM Help: Code snippets and structure assisted by Antigravity (AI).
"""

import torch
import os
import sys
import argparse
from transformers import RobertaTokenizer

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import EncoderLayer, DecoderLayer, EncoderTransf, DecoderTransf, EncoderDecoderTransf

def summarize(code_snippet, checkpoint_path):
    """
    Generate a docstring summary for a given Python code snippet using a trained model.
    """
    # Hyperparameters
    d_model = 256
    n_heads = 8
    n_layers = 4
    ff_units = 512
    dropout = 0.1
    max_code_len = 256
    max_summary_len = 128

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
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

    # Preprocess
    inputs = tokenizer(code_snippet, return_tensors='pt', max_length=max_code_len, truncation=True, padding='max_length')
    input_ids = inputs['input_ids'].to(device)

    # Greedy decoding
    generated = torch.zeros((1, 1), dtype=torch.long, device=device).fill_(tokenizer.bos_token_id)
    
    with torch.no_grad():
        for _ in range(max_summary_len - 1):
            logits = model(input_ids, generated)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    summary = tokenizer.decode(generated[0], skip_special_tokens=True)
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a summary for a Python code snippet.")
    parser.add_argument("--input", type=str, required=True, help="Python code snippet or path to a .py file.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model checkpoint.")
    
    args = parser.parse_args()
    
    code = args.input
    if os.path.exists(code):
        with open(code, 'r') as f:
            code = f.read()
    
    result = summarize(code, args.model)
    print("\n--- Code Summary ---")
    print(result)
    print("--------------------")
