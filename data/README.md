# Data Management and Storage

This directory contains the sampled datasets used for the **Transformer-based Seq2Seq** model evaluation and training verification. To ensure reproducibility and portability of the academic repository, we utilize a tiered data loading strategy.

### Directory Contents:
- **`train_subset.json`**: A representative sample of **50,000** Python/Docstring pairs extracted from the CodeXGLUE training split.
- **`val_subset.json`**: **5,000** validation samples for periodic performance monitoring.
- **`test_subset.json`**: **5,000** test samples used for qualitative and quantitative evaluation.

### Implementation Logic:
The system is designed to prioritize these local assets to ensure functionality in offline environments or restricted network conditions. The `CodeSummarizationDataset` class (found in `src/dataset.py`) automatically detects these local JSON files to bypass the need for external API calls to the Hugging Face Hub, while maintaining a fallback mechanism to the full dataset for large-scale high-performance training.

