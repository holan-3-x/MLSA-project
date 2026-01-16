import torch
import datasets
import transformers
from torchmetrics import BLEUScore

print(f"PyTorch version: {torch.__version__}")
print(f"Datasets version: {datasets.__version__}")
print(f"Transformers version: {transformers.__version__}")

if torch.backends.mps.is_available():
    print("MPS is available. Using GPU acceleration.")
    device = torch.device("mps")
else:
    print("MPS not available. Using CPU.")
    device = torch.device("cpu")

print(f"Current device: {device}")
