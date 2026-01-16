"""
Script to evaluate all checkpoints and plot REAL metrics.
"""
import os
import matplotlib.pyplot as plt
from evaluate import evaluate

def run_history_evaluation(samples=100):
    model_dir = "models"
    checkpoints = sorted([f for f in os.listdir(model_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pt")], 
                         key=lambda x: int(x.split("_")[-1].split(".")[0]))
    
    epochs = []
    losses = []
    ppls = []
    bleus = []
    rouges = []
    
    print(f"Starting real evaluation of {len(checkpoints)} checkpoints (Samples per model: {samples})...")
    
    for ckpt_name in checkpoints:
        epoch_num = int(ckpt_name.split("_")[-1].split(".")[0])
        ckpt_path = os.path.join(model_dir, ckpt_name)
        
        print(f"\n>>> Evaluating Epoch {epoch_num}...")
        metrics = evaluate(ckpt_path, num_samples=samples)
        
        epochs.append(epoch_num)
        losses.append(metrics["loss"])
        ppls.append(metrics["ppl"])
        bleus.append(metrics["bleu"])
        rouges.append(metrics["rougeL"])

    # --- Plotting REAL Data ---
    plt.figure(figsize=(15, 5))

    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, losses, label='Test Loss', marker='o', color='blue')
    plt.title('Real Loss Trend')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Plot Perplexity
    plt.subplot(1, 3, 2)
    plt.plot(epochs, ppls, label='Perplexity', marker='s', color='green')
    plt.title('Real Perplexity Trend')
    plt.xlabel('Epochs')
    plt.ylabel('PPL')
    plt.grid(True)
    plt.legend()

    # Plot Text Metrics
    plt.subplot(1, 3, 3)
    plt.plot(epochs, bleus, label='BLEU', marker='^', color='red')
    plt.plot(epochs, rouges, label='ROUGE-L', marker='v', color='orange')
    plt.title('Text Metrics Trend')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(model_dir, 'real_metrics_evolution.png')
    plt.savefig(plot_path)
    print(f"\n✅ Real evolution plots saved to '{plot_path}'")
    
    # Final data for report
    print("\n=== Real Data for your Report ===")
    print("Epoch | Loss | PPL | BLEU | ROUGE-L")
    for i in range(len(epochs)):
        print(f"{epochs[i]:5d} | {losses[i]:.2f} | {ppls[i]:.2f} | {bleus[i]:.4f} | {rouges[i]:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to evaluate per checkpoint")
    args = parser.parse_args()
    
    run_history_evaluation(samples=args.samples)
