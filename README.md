# Minimal LLM

A minimal, fully-functional LLM training codebase in pure PyTorch. Inspired by [nanochat](https://github.com/karpathy/nanochat) and based on [GPT-OSS 20B](https://huggingface.co/openai/gpt-oss-20b) architecture patterns.

## Features

- **Pure PyTorch** - No external dependencies beyond PyTorch and NumPy
- **Modern Architecture** - RMSNorm, RoPE, GQA, SwiGLU (GPT-OSS 20B style)
- **Production Ready** - Mixed precision, gradient accumulation, DDP support
- **Minimal & Readable** - ~1500 lines of well-documented code

### Architecture Comparison

| Feature | Traditional GPT | This Implementation |
|---------|----------------|---------------------|
| Normalization | LayerNorm | **RMSNorm** |
| Position Encoding | Learned Absolute | **RoPE** |
| Attention | MHA | **GQA** |
| Activation | GELU | **SwiGLU** |
| Linear Bias | Yes | **No** |

## Quick Start

### Installation

```bash
pip install torch numpy
pip install tiktoken  # optional, for BPE tokenizer
```

### Train on Shakespeare (5 min demo)

```bash
# Prepare data
python data.py --dataset shakespeare

# Train
python train.py \
    --data-path data/shakespeare.bin \
    --model-size small \
    --max-iters 2000 \
    --batch-size 32

# Generate text
python generate.py \
    --checkpoint checkpoints/ckpt_2000.pt \
    --vocab-path data/shakespeare_vocab.json \
    --prompt "ROMEO: "
```

## Project Structure

```
├── model.py        # GPT model with RMSNorm, RoPE, GQA, SwiGLU
├── train.py        # Training loop with mixed precision & DDP
├── tokenizer.py    # Character-level and BPE tokenizers
├── data.py         # Data preparation utilities
├── generate.py     # Text generation / inference
├── tutorial.md     # Detailed tutorial
└── requirements.txt
```

## Model Configurations

| Config | Layers | Dim | Heads | KV Heads | Params |
|--------|--------|-----|-------|----------|--------|
| `small` | 12 | 768 | 12 | 4 | ~125M |
| `medium` | 24 | 1024 | 16 | 4 | ~350M |
| `large` | 24 | 1536 | 16 | 8 | ~760M |
| `gpt-oss` | 24 | 2880 | 64 | 8 | ~2.7B |

## Usage

### Training

```bash
# Basic training with synthetic data (for testing)
python train.py --model-size small --max-iters 100

# Train on custom data
python train.py \
    --data-path path/to/data.bin \
    --model-size medium \
    --batch-size 16 \
    --seq-len 512 \
    --learning-rate 3e-4 \
    --max-iters 10000

# Resume from checkpoint
python train.py \
    --resume-from checkpoints/ckpt_5000.pt \
    --max-iters 20000

# Enable torch.compile (PyTorch 2.0+)
python train.py --compile
```

### Distributed Training

```bash
# Single node, multi-GPU
torchrun --nproc_per_node=4 train.py --model-size large --batch-size 8

# Multi-node
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=<MASTER_IP> train.py
```

### Data Preparation

```bash
# Download and prepare Shakespeare dataset
python data.py --dataset shakespeare

# Prepare custom text file
python -c "
from data import prepare_data
prepare_data('my_corpus.txt', 'data/my_corpus.bin')
"
```

### Text Generation

```bash
# Single prompt
python generate.py \
    --checkpoint checkpoints/ckpt_10000.pt \
    --prompt "Once upon a time" \
    --max-tokens 200 \
    --temperature 0.8

# Interactive mode
python generate.py --checkpoint checkpoints/ckpt_10000.pt
```

## Architecture Details

### RMSNorm

Root Mean Square Normalization - simpler and faster than LayerNorm:

```python
RMSNorm(x) = x / sqrt(mean(x²) + ε) * γ
```

### RoPE (Rotary Position Embeddings)

Encodes position through rotation in complex space:
- Naturally captures relative positions
- Extrapolates to longer sequences
- No learned position parameters

### GQA (Grouped Query Attention)

Multiple query heads share key/value heads:
- Reduces KV cache size for faster inference
- Maintains model quality
- Example: 64 Q heads with 8 KV heads (8:1 ratio)

### SwiGLU

Gated activation combining Swish and GLU:

```python
SwiGLU(x) = Swish(xW_gate) * (xW_up)
```

## Training Tips

1. **Learning Rate**: Start with 3e-4 for small models, 1e-4 for large
2. **Batch Size**: Use gradient accumulation for effective large batches
3. **Warmup**: Default 500 steps, increase for larger models
4. **Gradient Clipping**: Default 1.0, helps stability

## Memory Requirements (Approximate)

| Model | Training (bf16) | Inference |
|-------|-----------------|-----------|
| small | ~4 GB | ~1 GB |
| medium | ~8 GB | ~2 GB |
| large | ~16 GB | ~4 GB |

## API Reference

### Model

```python
from model import GPT, ModelConfig

config = ModelConfig.small()
model = GPT(config)

# Forward pass
logits, loss, kv_cache = model(input_ids, targets)

# Generation
output_ids = model.generate(input_ids, max_new_tokens=100)
```

### Tokenizer

```python
from tokenizer import CharTokenizer, BPETokenizer

# Character-level
tok = CharTokenizer.from_text("Hello World")
tokens = tok.encode("Hello")
text = tok.decode(tokens)

# BPE (requires tiktoken)
tok = BPETokenizer("cl100k_base")
tokens = tok.encode("Hello World")
```

## References

- [nanochat](https://github.com/karpathy/nanochat) - Karpathy's minimal chat model
- [GPT-OSS 20B](https://huggingface.co/openai/gpt-oss-20b) - OpenAI's open-source model
- [RoPE Paper](https://arxiv.org/abs/2104.09864) - Rotary Position Embedding
- [GQA Paper](https://arxiv.org/abs/2305.13245) - Grouped Query Attention
- [SwiGLU Paper](https://arxiv.org/abs/2002.05202) - GLU Variants

## License

MIT
