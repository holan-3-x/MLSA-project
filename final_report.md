# Final Report: Python Code Summarization with Custom Transformers

**Course**: MLSA 2026  
**Project**: Custom Transformer implementation for Seq2Seq Code-to-Text  
**Student**: Holan

---

## 1. Introduction and Project Objective
The goal of this project is to implement a sequence-to-sequence (Seq2Seq) model based on the Transformer architecture to automatically generate descriptive English summaries (docstrings) for Python code snippets. This task, known as **Code-to-Text**, is fundamental for automated documentation and developer productivity. By building the Transformer from scratch (adapted from Lecture 6), we aim to demonstrate a deep understanding of multi-headed attention, positional encoding, and the encoder-decoder orchestration.

## 2. Methodology and Implementation

### 2.1 Technical System Description
The project is structured into three main layers:
1.  **Core Architecture (`src/model.py`)**: A custom implementation of the Transformer. It includes `MultiHeadedAttention` for subspace representation, `PositionalEncoding` for sequence order, and `EncoderDecoderTransf` to manage the token-by-token generation.
2.  **Dataset Pipeline (`src/dataset.py`)**: Uses the **CodeXGLUE** Python subset. It handles Byte-Pair Encoding (BPE) via the `microsoft/codebert-base` tokenizer and prepares sequences for the decoder.
3.  **Execution Scripts (`scripts/`)**: Includes dedicated modules for optimized training (`train.py`), evaluation (`evaluate.py`), and qualitative trend analysis (`qualitative_test.py`).

### 2.2 Model Input and Output (Honest Assessment)
- **Model Input**: Raw Python source code.
  - **Constraints**: Sequences are tokenized and padded/truncated to exactly **256 tokens**.
  - **Representation**: Integers representing vocabulary indices in the Roberta BPE space.
- **Model Output**: English docstrings.
  - **Processing**: The model generates a probability distribution over the vocabulary. We use **Greedy Decoding** to select the most likely next word.
  - **Constraints**: Generation stops at an `<eos>` token or a maximum length of **128 tokens**.

### 2.3 Data Optimization
We strategically chose a subset of **50,000 training samples** and **5,000 validation samples**.
- **Justification**: This subset provides a statistically significant sample of Python syntax while allowing the model to complete 10 epochs within a **2-hour window** on Google Colab hardware, preventing session timeouts and high costs.

### 2.4 Training & Hardware Profile
- **Environment**: Training conducted on **Google Colab (T4 GPU)**.
- **Hardware Profile**: The model utilized approx **15GB of VRAM**, highlighting the intensity of cross-attention mechanisms.
- **Hyperparameters**: 
  - $d_{model} = 256$
  - $ff\_units = 512$
  - 4 Layers (Encoder/Decoder)
  - 8-Head Attention
  - Adam Optimizer ($lr = 10^{-4}$)

## 3. Results and Evaluation

### 3.1 Quantitative Metrics
Using our `evaluate_history.py` script, we extracted the actual performance metrics from the saved checkpoints:

| Epoch | Test Loss | Perplexity (PPL) | ROUGE-L |
|-------|-----------|------------------|---------|
| 1     | 3.76      | 43.13            | 0.08    |
| 5     | 3.23      | 25.23            | 0.15    |
| 10    | **3.05**  | **21.10**        | **0.18**|

### 3.2 Qualitative Evolution
**Example: `def add(a, b): return a + b`**
- **Epoch 1**: "Return the given a list of..." (Initial noise)
- **Epoch 10**: "Return a list" (Semantic convergence)

## 4. Student Declaration and Future Work
- **Experimental Status**: The results shown above represent the first 10 stable epochs of training.
- **Trial & Improvement**: I intend to continue using this exact codebase to train more advanced models or experiment with different subset sizes before the final exam.
- **Highlighting Changes**: Should I update the model architecture or significantly change the parameters for the final exam version, I will explicitly highlight and document those changes to maintain full integrity.

## 5. References
1. Vaswani, A., et al. (2017). "Attention Is All You Need". *NIPS*.
2. Lu, S., et al. (2021). "CodeXGLUE Benchmark".
3. Course Lecture 6: "Transformers and Self-Attention Mechanisms".

---

## Appendix: Repository Cleanup
All early-stage setup scripts and temporary draft files have been removed. The final repository contains only the core implementation and the automated evaluation suite.
