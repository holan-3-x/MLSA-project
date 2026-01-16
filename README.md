# 🐍 Transformer-Based Python Code Summarization

[![MLSA 2026](https://img.shields.io/badge/Course-MLSA_2026-blue.svg)](https://github.com/holan/mlsa-transformer)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Tokenizer-CodeBERT-yellow.svg)](https://huggingface.co/microsoft/codebert-base)

This repository contains a 100% custom implementation of a **Transformer-based Sequence-to-Sequence (Seq2Seq)** model designed to translate Python source code into concise, human-readable English docstrings.

---

## 🌟 Project Highlights
- **🏗️ custom Architecture**: A deep-learning model built from scratch using PyTorch tensors, implementing multi-head attention and causal masking as per **Lecture 6: Transformers**.
- **⚡ Performance Optimized**: High-performance training support for **Apple Silicon (MPS)** and **NVIDIA CUDA** environments.
- **🔄 Fault-Tolerant Training**: Robust checkpointing system with Google Drive integration for seamless session recovery on Google Colab.
- **📈 Data-Driven Analysis**: Comprehensive evaluation suite providing Loss, Perplexity, BLEU, and ROUGE-L metrics.

---

## 🚀 Getting Started

### 1. Environment Setup
We recommend using a virtual environment to manage dependencies.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
> [!NOTE]
> Major dependencies include `torch`, `transformers`, `datasets`, and `torchmetrics`.

### 2. Training (Colab Optimized)
For high-performance training, use the included **`Transformer_Colab_Training.ipynb`**. 
- **Setup**: One-click setup for Google Drive and dependencies.
- **Safety**: Built-in VRAM management to ensure stability on T4 GPUs.
- **Persistence**: Models are saved directly to `/content/drive/My Drive/MLSA_Transformer_Checkpoints`.

### 3. Inference & Analysis
Test the model or analyze trends locally:
- **Summarize Code**:
  ```bash
  python scripts/summarize.py --model models/checkpoint_epoch_10.pt --input "def add(x, y): return x + y"
  ```
- **Generate Performance Charts**:
  ```bash
  python scripts/plot_metrics.py
  ```

---

## 📁 Repository Map

| Directory | Description |
| :--- | :--- |
| **`src/`** | Core library: `model.py` (Architecture) & `dataset.py` (Data Pipeline). |
| **`scripts/`** | `train.py`, `evaluate.py`, `summarize.py`, and `plot_metrics.py`. |
| **`data/`** | Local subsets (**50k train**) serialized from CodeXGLUE. |
| **`models/`** | Storage for `.pt` checkpoints and evaluation visualizations. |
| **`final_report.md`** | Detailed technical documentation and academic analysis. |

---

## ⚖️ Academic Integrity & AI Assistance
This project was developed for the **MLSA 2026** course. In the spirit of transparency:
- **Core Implementation**: The Transformer architecture and data pipeline were manually implemented based on provided lecture notes.
- **AI Assistance**: **Antigravity (LLM)** was utilized as a technical pair-programmer to assist with boilerplate structuring, debugging complex tensor shapes, and technical documentation drafting.
- **Verification**: All outputs were manually reviewed and verified for alignment with course objectives and mathematical correctness.

---

## 📊 Documentation
For a deep dive into the architecture, mathematical justification, and real training results, please refer to the **[Final Project Report](final_report.md)**.
