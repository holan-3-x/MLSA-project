import os
import json
from datasets import load_dataset
from tqdm import tqdm

def download_subset(split, size, output_path):
    print(f"Downloading {split} split...")
    dataset = load_dataset("code_x_glue_ct_code_to_text", "python", split=split)
    
    # Take a representative subset
    subset = dataset.select(range(min(size, len(dataset))))
    
    data_list = []
    print(f"Processing {split} samples...")
    for item in tqdm(subset):
        data_list.append({
            'code': item['code'],
            'docstring': item['docstring']
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=2)
    print(f"Saved {len(data_list)} samples to {output_path}")

if __name__ == "__main__":
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # We save a smaller subset for the repository to stay lightweight
    # but still have "real data" present.
    # Set to 50,000 for training as per project specification
    download_subset('train', 50000, os.path.join(data_dir, 'train_subset.json'))
    download_subset('validation', 5000, os.path.join(data_dir, 'val_subset.json'))
    download_subset('test', 5000, os.path.join(data_dir, 'test_subset.json'))
    
    print("\n✅ Local data subset established in 'data/'")
