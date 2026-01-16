"""
Dataset handling for the MLSA Transformer Project.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import RobertaTokenizer

class CodeSummarizationDataset(Dataset):
    def __init__(self, split, tokenizer, max_code_len=256, max_summary_len=128, subset_size=None):
        # Configuration for CodeXGLUE code-to-text python
        self.dataset = load_dataset("code_x_glue_ct_code_to_text", "python", split=split)
        
        # SPEED OPTIMIZATION: Use a subset if specified
        if subset_size and subset_size < len(self.dataset):
            self.dataset = self.dataset.select(range(subset_size))
            
        self.tokenizer = tokenizer
        self.max_code_len = max_code_len
        self.max_summary_len = max_summary_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        code = item['code']
        summary = item['docstring']

        # Tokenize code (source)
        code_enc = self.tokenizer(
            code,
            max_length=self.max_code_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize summary (target)
        summary_enc = self.tokenizer(
            summary,
            max_length=self.max_summary_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = code_enc['input_ids'].squeeze(0)
        attention_mask = code_enc['attention_mask'].squeeze(0)
        labels = summary_enc['input_ids'].squeeze(0)
        
        # In a custom Transformer, we often need decoder_input_ids
        # typically shifted labels or labels with a BOS token.
        # RobertaTokenizer uses 0 for <s>, 2 for </s>, 1 for <pad>.
        decoder_input_ids = labels.clone()
        # For training, common practice is to shift labels for the decoder.
        # But for simplicity in the first draft, we'll return them like this 
        # and handle shifting in the training loop or model forward if needed.

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'labels': labels
        }

def get_dataloaders(batch_size=16, tokenizer_name="microsoft/codebert-base", train_subset=None, val_subset=None):
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    
    # Using small subset for initial testing could be helpful, 
    # but here we load the requested splits.
    train_ds = CodeSummarizationDataset('train', tokenizer, subset_size=train_subset)
    val_ds = CodeSummarizationDataset('validation', tokenizer, subset_size=val_subset)
    test_ds = CodeSummarizationDataset('test', tokenizer)

    print(f"Dataset Statistics:")
    print(f"  - Training samples: {len(train_ds)}")
    print(f"  - Validation samples: {len(val_ds)}")
    print(f"  - Test samples: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=0)

    return train_loader, val_loader, test_loader, tokenizer

if __name__ == "__main__":
    # Quick test
    train_loader, _, _, tokenizer = get_dataloaders(batch_size=2)
    batch = next(iter(train_loader))
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    print("Success!")
