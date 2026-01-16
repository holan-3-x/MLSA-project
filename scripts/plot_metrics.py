"""
Script to dynamically generate performance metrics by evaluating all checkpoints.
This script performs REAL evaluation on the test subset to verify the evolution 
of Loss, Perplexity, BLEU, and ROUGE-L scores.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.evaluate import evaluate

def analyze_checkpoints(num_test_samples=32):
    """
    Evaluates key checkpoints to generate the Real Metrics Evolution graph.
    """
    all_ckpts = sorted(glob.glob("models/checkpoint_epoch_*.pt"), 
                       key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # We evaluate selective key epochs to be efficient
    target_epochs = [1, 3, 5, 7, 10]
    checkpoint_files = [c for c in all_ckpts if int(c.split('_')[-1].split('.')[0]) in target_epochs]

    if not checkpoint_files:
        print("❌ No checkpoints found.")
        return

    res_epochs = []
    res_losses = []
    res_ppls = []
    res_bleus = []
    res_rouges = []

    print(f"🔍 Starting real evaluation on keys: {target_epochs}...")

    for ckpt in checkpoint_files:
        epoch_num = int(ckpt.split('_')[-1].split('.')[0])
        print(f"\n--- Real-Time Audit: Epoch {epoch_num} ---")
        metrics = evaluate(ckpt, num_samples=num_test_samples)
        
        res_epochs.append(epoch_num)
        res_losses.append(metrics['loss'])
        res_ppls.append(metrics['ppl'])
        res_bleus.append(metrics['bleu'])
        res_rouges.append(metrics['rougeL'])

    # Plotting logic with real data
    plt.figure(figsize=(10, 6))
    plt.plot(res_epochs, res_losses, marker='o', color='tab:red', label='Test Loss (Real)')
    plt.plot(res_epochs, res_bleus, marker='s', color='tab:blue', label='BLEU (Real)')
    plt.plot(res_epochs, res_rouges, marker='^', color='tab:cyan', label='ROUGE-L (Real)')
    
    plt.title('Verified Model Performance Evolution', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Score / Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_path = 'models/real_metrics_evolution.png'
    plt.savefig(output_path, dpi=300)
    print(f"\n✅ SUCCESS: Verified plot saved to {output_path}")

if __name__ == "__main__":
    analyze_checkpoints(num_test_samples=32)
