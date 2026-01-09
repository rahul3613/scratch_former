# scratch_former
Building a Transformer from Scratch

## Overview
This project implements transformer model from scratch in PyTorch, progressing through three versions with increasing complexity and features.

## Versions

### [v0](v0): Foundation
- Basic attention mechanism
- Projection layer
- Minimal architecture for understanding core concepts

### [v1](v1): Complete Flow
- Single attention head
- Single transformer block
- Full forward pass pipeline
- Foundation for scaling

### [v2](v2): Production Architecture
- Multi-head attention
- Multiple stacked blocks
- Complete transformer architecture
- Optimized for performance

## [tokenizer](tokenizer)

- **Purpose:** Implements a simple BPE-style byte-pair encoding over UTF-8 bytes. `tokenizer.py` provides `encode(text)` and `decode(token_ids)` utilities that apply merges from `merges_spl.json`.
- **Building merges:** Run `bpe.py` to build merges from the dataset (it reads `corpus.txt`). `bpe.py` writes `merges.json` which can be used directly by `tokenizer.py`.
- **Usage (example):**

```python
from tokenizer.tokenizer import encode, decode

ids = encode("Hello World")
text = decode(ids)
```

## Dataset
Custom-prepared date conversion dataset used for training and evaluation across all versions.

## Getting Started
Explore each version directory to understand the progressive development of transformer architecture.

## File Structure
```markdown
scratch_former/
├── README.md           # Project documentation
├── requirements.txt    # Python dependencies including PyTorch, CUDA, Jupyter
├── test.json           # Test dataset for date conversion (JSON array of human-machine pairs)
├── train.json          # Training dataset for date conversion (large JSON array)
├── utils.py            # Utility functions for generating datetime datasets using Faker and Babel
├── tokenizer/
│   ├── tokenizer.py    # encode(text) and decode(token_ids) using merges (BPE over UTF-8 bytes)
│   ├── bpe.py          # Builds merges from corpus.txt and writes merges.json
│   └── merges.json     # BPE merges produced by bpe.py (used by tokenizer)
├── v0/
│   ├── basic.py        # Basic transformer implementation with single attention head
│   └── test.ipynb      # Jupyter notebook for testing v0 model: data loading, training, inference
├── v1/
│   ├── basic.py        # Enhanced transformer with layer norms, feed-forward, residuals
│   └── test.ipynb      # Jupyter notebook for testing v1 model
└── v2/
    ├── basic.py        # Multi-head attention transformer with stacked blocks
    └── test.ipynb      # Jupyter notebook for testing v2 model
```

