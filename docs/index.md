# Minimal LLM Documentation

Welcome to the Minimal LLM documentation. This guide will walk you through every step of training your own language model from scratch.

## Quick Navigation

| Step | Document | Description |
|------|----------|-------------|
| 1 | [Installation](01-installation.md) | Set up your environment |
| 2 | [Data Preparation](02-data-preparation.md) | Prepare training data |
| 3 | [Tokenizer](03-tokenizer.md) | Understand and use tokenizers |
| 4 | [Training](04-training.md) | Train your model |
| 5 | [Generation](05-generation.md) | Generate text with trained models |
| 6 | [Architecture](06-architecture.md) | Deep dive into model architecture |
| 7 | [Model Configs](07-model-configs.md) | Parameter relationships and scaling |

## Overview

This codebase implements a minimal but complete LLM training pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Raw Text ──► Tokenizer ──► Binary Data ──► DataLoader         │
│       │            │              │              │               │
│       │            ▼              │              ▼               │
│       │      vocab.json          │         Batches              │
│       │                          │              │               │
│       │                          │              ▼               │
│       │                          │      ┌──────────────┐        │
│       │                          │      │   GPT Model  │        │
│       │                          │      │  ┌────────┐  │        │
│       │                          │      │  │RMSNorm │  │        │
│       │                          │      │  │  RoPE  │  │        │
│       │                          │      │  │  GQA   │  │        │
│       │                          │      │  │SwiGLU  │  │        │
│       │                          │      │  └────────┘  │        │
│       │                          │      └──────────────┘        │
│       │                          │              │               │
│       │                          │              ▼               │
│       │                          │      Checkpoints (.pt)       │
│       │                          │              │               │
│       ▼                          ▼              ▼               │
│   Input Text ◄────────────── Generation ◄── Load Model         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## File Structure

```
minimal-llm/
├── model.py          # GPT model implementation
├── train.py          # Training script
├── tokenizer.py      # Tokenizer implementations
├── data.py           # Data preparation utilities
├── generate.py       # Text generation script
├── requirements.txt  # Python dependencies
└── docs/             # Documentation (you are here)
    ├── index.md
    ├── 01-installation.md
    ├── 02-data-preparation.md
    ├── 03-tokenizer.md
    ├── 04-training.md
    ├── 05-generation.md
    └── 06-architecture.md
```

## End-to-End Example

Here's a complete example to train a Shakespeare model in 5 minutes:

```bash
# 1. Install dependencies
pip install torch numpy

# 2. Prepare data
python data.py --dataset shakespeare

# 3. Train (small model, 2000 iterations)
python train.py \
    --data-path data/shakespeare.bin \
    --model-size small \
    --max-iters 2000 \
    --batch-size 32

# 4. Generate text
python generate.py \
    --checkpoint checkpoints/ckpt_2000.pt \
    --vocab-path data/shakespeare_vocab.json \
    --prompt "ROMEO:"
```

## Requirements

- Python 3.8+
- PyTorch 2.0+ (recommended for `torch.compile`)
- CUDA-capable GPU (optional but recommended)
- 4-16 GB VRAM depending on model size

## Getting Help

Each documentation page includes:
- **Inputs**: What you need before starting
- **Outputs**: What you'll have after completing the step
- **Commands**: Exact commands to run
- **Expected Output**: What you should see
- **Troubleshooting**: Common issues and solutions

Start with [Installation](01-installation.md) →
