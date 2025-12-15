# scratch_former
Building a Transformer from Scratch

## Overview
This project implements transformer model from scratch in PyTorch, progressing through three versions with increasing complexity and features.

## Versions

### v1: Foundation
- Basic attention mechanism
- Projection layer
- Minimal architecture for understanding core concepts

### v2: Complete Flow
- Single attention head
- Single transformer block
- Full forward pass pipeline
- Foundation for scaling

### v3: Production Architecture
- Multi-head attention
- Multiple stacked blocks
- Complete transformer architecture
- Optimized for performance

## Dataset
Custom-prepared date conversion dataset used for training and evaluation across all versions.

## Getting Started
Explore each version directory to understand the progressive development of transformer architecture.

## File Structure
```
scratch_former/
├── README.md           # Project documentation
├── requirements.txt    # Python dependencies including PyTorch, CUDA, Jupyter
├── test.json           # Test dataset for date conversion (JSON array of human-machine pairs)
├── train.json          # Training dataset for date conversion (large JSON array)
├── utils.py            # Utility functions for generating datetime datasets using Faker and Babel
├── v0
│   ├── basic.py        # Basic transformer implementation with single attention head
│   └── test.ipynb      # Jupyter notebook for testing v0 model: data loading, training, inference
├── v1
│   ├── basic.py        # Enhanced transformer with layer norms, feed-forward, residuals
│   └── test.ipynb      # Jupyter notebook for testing v1 model
└── v2
    ├── basic.py        # Multi-head attention transformer with stacked blocks
    └── test.ipynb      # Jupyter notebook for testing v2 model
```