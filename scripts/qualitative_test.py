"""
Qualitative evaluation script to show model improvement over time.
Runs inference on fixed samples for all checkpoints in the models/ directory.
"""

import torch
import os
import sys
from transformers import RobertaTokenizer
import glob

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.summarize import summarize

TEST_SAMPLES = [
    "def add(a, b): return a + b",
    "def factorial(n): return 1 if n == 0 else n * factorial(n-1)",
    "def greet(name): print(f'Hello, {name}!')"
]

def run_evolution():
    checkpoints = sorted(glob.glob("models/checkpoint_epoch_*.pt"), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    if not checkpoints:
        print("No checkpoints found in 'models/'. Please train the model first.")
        return

    results = {sample: [] for sample in TEST_SAMPLES}

    for ckpt in checkpoints:
        epoch = ckpt.split('_')[-1].split('.')[0]
        print(f"Running inference for Epoch {epoch}...")
        for sample in TEST_SAMPLES:
            summary = summarize(sample, ckpt)
            results[sample].append((epoch, summary))

    print("\n=== Model Evolution Over Time ===")
    for sample, summaries in results.items():
        print(f"\nCode: {sample}")
        for epoch, summary in summaries:
            print(f"  Epoch {epoch:2}: {summary}")

if __name__ == "__main__":
    run_evolution()
