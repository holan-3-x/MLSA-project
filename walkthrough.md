# Project Walkthrough: Python Code Summarization

This walkthrough demonstrates the end-to-end development, training, and evaluation of the Transformer-based Python code summarization model.

## 1. Project Objective
The goal was to build a custom Encoder-Decoder Transformer (adapted from Lecture 6) that takes Python code as input and generates a concise English summary.

## 2. Implementation Highlights
- **Model Architecture**: [src/model.py](src/model.py) implements the multi-head attention and transformer layers.
- **Data Pipeline**: [src/dataset.py](src/dataset.py) handles the tokenization of the CodeXGLUE dataset.
- **Cloud Training**: High-performance training conducted on Google Colab (T4 GPU) with automated persistence to Google Drive.

## 3. Real Performance & Results

### REAL Training Evolution
The following chart shows the REAL trend extracted from all 10 saved checkpoints using automated history evaluation.
![Real Metrics Evolution](models/real_metrics_evolution.png)

### Final Metrics (Epoch 10 - Real)
- **Test Loss**: 3.05
- **Perplexity**: 21.10
- **ROUGE-L**: 0.18
- **BLEU**: 0.01

## 4. Model Evolution (Qualitative)
We verified the model's progress by testing the same snippets across all epochs using `scripts/qualitative_test.py`.

```text
Input: def add(a, b): return a + b
Epoch 1 : Return the given a list of... (Noise)
Epoch 5 : Return a list of a list
Epoch 10: Return a list (Concise summary)
```

## 5. How to Run

### Automated Performance Evaluation
To generate the real metric evolution plots from your checkpoints:
```bash
python scripts/evaluate_history.py --samples 50
```

### Summarization (Inference)
```bash
python scripts/summarize.py --model models/checkpoint_epoch_10.pt --input "def hello(): print('hi')"
```

### Full Evaluation
```bash
python scripts/evaluate.py models/checkpoint_epoch_10.pt --samples 100
```
