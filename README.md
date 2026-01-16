# MLSA Transformer Project: Python Code Summarization

This project implements a custom Transformer model (Encoder-Decoder) for the task of Python code summarization (Code-to-Text). The model is designed to generate English docstrings from Python source code.

## Project Adherence & Documentation

### Adaptation & LLM Use
- **Custom Architecture**: The core Transformer architecture (`src/model.py`) is a custom implementation ported and adapted from **Lecture 6: Transformers**.
- **LLM Assistance**: Antigravity (AI) was used to assist in the coding process, structuring the repository, and implementing boilerplate logic.
- **Pretrained Components**: Consistent with the project rules, we use the **pretrained Roberta Tokenizer** (`microsoft/codebert-base`) and its vocabulary. No pretrained *full* language models or models already trained on Python syntax were used for the main core.

### Hardware Acceleration
- This project is optimized for **Apple Silicon** using the **MPS (Metal Performance Shaders)** backend for high-performance training on Mac hardware.

## Setup Instructions

1.  **Environment Setup**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install torch datasets transformers torchmetrics
    ```

2.  **Verify Setup**:
    ```bash
    python verify_setup.py
    ```

## Execution Steps

### Option A: Local Training (Mac/MPS)
If you want to train on your local machine with hardware acceleration (Apple Silicon):
```bash
python scripts/train.py
```

### Option B: Cloud Training (Google Colab)
If your local machine is overheating or lacks a powerful GPU:
1.  Upload `Transformer_Colab_Training.ipynb` to Google Colab.
2.  Enable a **GPU T4** or better runtime (Runtime -> Change runtime type).
3.  Run all cells. The notebook is self-contained and will install dependencies automatically.
4.  Download the generated `.pt` checkpoints to your local `models/` folder for evaluation/inference.

### 2. Evaluation
To evaluate a trained checkpoint on the test set (calculates Loss, Perplexity, BLEU, and ROUGE):
```bash
python scripts/evaluate.py models/checkpoint_epoch_10.pt
```

### 3. Inference (Summarization)
To generate a summary for a custom code snippet:
```bash
python scripts/summarize.py models/checkpoint_epoch_10.pt "def add(a, b): return a + b"
```

## Metrics Implemented
- **Cross-entropy loss**: Monitor training convergence.
- **Perplexity**: Qualitative measure of language model certainty.
- **BLEU score**: Overlap of n-grams between prediction and reference.
- **ROUGE score**: Recall-oriented assessment of summary quality.

## Repository Structure
- `src/`: Core implementation (`model.py`, `dataset.py`).
- `scripts/`: Execution logic (`train.py`, `evaluate.py`, `summarize.py`).
- `models/`: Weights storage.
- `verify_setup.py`: Environment verification utility.
