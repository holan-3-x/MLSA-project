import torch
import torch.nn as nn
from src.dataset import get_dataloaders
from src.model import EncoderLayer, DecoderLayer, EncoderTransf, DecoderTransf, EncoderDecoderTransf

def test():
    # Small parameters for testing
    d_model = 64
    n_heads = 4
    n_layers = 1
    ff_units = 128
    max_code_len = 256
    max_summary_len = 128

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Testing on device: {device}")

    train_loader, _, _, tokenizer = get_dataloaders(batch_size=2)
    vocab_size = tokenizer.vocab_size

    enclayer = EncoderLayer(n_heads=n_heads, d_model=d_model, ff_units=ff_units)
    declayer = DecoderLayer(n_heads=n_heads, d_model=d_model, ff_units=ff_units)
    
    encoder = EncoderTransf(enclayer, n_layers=n_layers, max_len=max_code_len)
    decoder = DecoderTransf(declayer, n_layers=n_layers, max_len=max_summary_len)
    
    model = EncoderDecoderTransf(encoder, decoder, vocab_size, vocab_size, max_len=max_summary_len)
    model.to(device)

    batch = next(iter(train_loader))
    input_ids = batch['input_ids'].to(device)
    decoder_input_ids = batch['decoder_input_ids'].to(device)
    
    # Forward pass
    logits = model(input_ids, decoder_input_ids[:, :-1])
    print(f"Logits shape: {logits.shape}")
    
    if logits.shape == (2, max_summary_len - 1, vocab_size):
        print("Model forward pass successful!")
    else:
        print(f"Unexpected shape: {logits.shape}")

if __name__ == "__main__":
    test()
