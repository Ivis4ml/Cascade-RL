# Model Configurations Deep Dive

This guide provides detailed analysis of different model configurations, parameter relationships, and how to choose the right settings for your use case.

## Available Presets

| Preset | Parameters | Layers | Hidden | Heads | KV Heads | FFN | Use Case |
|--------|------------|--------|--------|-------|----------|-----|----------|
| `small` | ~125M | 12 | 768 | 12 | 4 | 2048 | Learning, testing |
| `medium` | ~350M | 24 | 1024 | 16 | 4 | 2816 | Serious experiments |
| `large` | ~760M | 24 | 1536 | 16 | 8 | 4096 | Production |
| `gpt_oss_style` | ~2.7B | 24 | 2880 | 64 | 8 | 7680 | Large-scale |

---

## Parameter Relationships

### Core Formula

```
Total Parameters ≈ Embedding + Transformer Layers + LM Head

Where:
- Embedding:    vocab_size × d_model
- Per Layer:    Attention + FFN + Norms
- LM Head:      d_model × vocab_size
```

### Detailed Breakdown

```
┌─────────────────────────────────────────────────────────────────┐
│                    Parameter Distribution                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Embedding Layer                                                 │
│  └── vocab_size × d_model                                       │
│                                                                  │
│  Per Transformer Layer:                                          │
│  ├── Attention                                                   │
│  │   ├── W_q: d_model × (n_heads × head_dim)                   │
│  │   ├── W_k: d_model × (n_kv_heads × head_dim)                │
│  │   ├── W_v: d_model × (n_kv_heads × head_dim)                │
│  │   └── W_o: (n_heads × head_dim) × d_model                   │
│  │                                                               │
│  ├── FFN (SwiGLU)                                               │
│  │   ├── W_gate: d_model × d_ff                                 │
│  │   ├── W_up:   d_model × d_ff                                 │
│  │   └── W_down: d_ff × d_model                                 │
│  │                                                               │
│  └── RMSNorm (×2)                                               │
│      └── 2 × d_model (scale parameters only)                    │
│                                                                  │
│  Final RMSNorm                                                   │
│  └── d_model                                                     │
│                                                                  │
│  LM Head                                                         │
│  └── d_model × vocab_size                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration: Small (~125M)

```python
ModelConfig.small() = ModelConfig(
    vocab_size=32000,
    n_layers=12,
    n_heads=12,
    n_kv_heads=4,      # GQA ratio: 12/4 = 3
    d_model=768,
    d_ff=2048,
    max_seq_len=2048,
)
```

### Parameter Calculation

```
head_dim = d_model / n_heads = 768 / 12 = 64

Embedding:
  32000 × 768 = 24,576,000

Per Layer Attention:
  W_q: 768 × 768 = 589,824
  W_k: 768 × 256 = 196,608  (n_kv_heads × head_dim = 4 × 64 = 256)
  W_v: 768 × 256 = 196,608
  W_o: 768 × 768 = 589,824
  Attention Total: 1,572,864

Per Layer FFN (SwiGLU):
  W_gate: 768 × 2048 = 1,572,864
  W_up:   768 × 2048 = 1,572,864
  W_down: 2048 × 768 = 1,572,864
  FFN Total: 4,718,592

Per Layer Norms:
  2 × 768 = 1,536

Per Layer Total: 1,572,864 + 4,718,592 + 1,536 = 6,292,992

All Layers: 12 × 6,292,992 = 75,515,904

Final Norm: 768

LM Head: 768 × 32000 = 24,576,000

═══════════════════════════════════════
Total: 24,576,000 + 75,515,904 + 768 + 24,576,000
     = 124,668,672 parameters (~125M)
═══════════════════════════════════════
```

### Memory Requirements

| Precision | Parameters | Gradients | Optimizer | Activations* | Total |
|-----------|------------|-----------|-----------|--------------|-------|
| fp32 | 500 MB | 500 MB | 1 GB | ~1 GB | ~3 GB |
| bf16 | 250 MB | 250 MB | 1 GB | ~0.5 GB | ~2 GB |

*Activations depend on batch_size × seq_len

### Recommended Settings

```bash
python train.py \
    --model-size small \
    --batch-size 32 \
    --seq-len 512 \
    --learning-rate 1e-3 \
    --max-iters 10000
```

---

## Configuration: Medium (~350M)

```python
ModelConfig.medium() = ModelConfig(
    vocab_size=32000,
    n_layers=24,        # 2× layers vs small
    n_heads=16,
    n_kv_heads=4,       # GQA ratio: 16/4 = 4
    d_model=1024,       # 1.33× hidden dim
    d_ff=2816,          # ~2.75× d_model
    max_seq_len=2048,
)
```

### Parameter Calculation

```
head_dim = 1024 / 16 = 64

Embedding: 32000 × 1024 = 32,768,000

Per Layer Attention:
  W_q: 1024 × 1024 = 1,048,576
  W_k: 1024 × 256 = 262,144
  W_v: 1024 × 256 = 262,144
  W_o: 1024 × 1024 = 1,048,576
  Total: 2,621,440

Per Layer FFN:
  3 × (1024 × 2816) = 8,650,752

Per Layer Norms: 2,048

Per Layer Total: 11,274,240

All Layers: 24 × 11,274,240 = 270,581,760

Final Norm + LM Head: 1024 + 32,768,000 = 32,769,024

═══════════════════════════════════════
Total: 32,768,000 + 270,581,760 + 32,769,024
     = 336,118,784 parameters (~336M)
═══════════════════════════════════════
```

### Key Differences from Small

| Aspect | Small | Medium | Change |
|--------|-------|--------|--------|
| Layers | 12 | 24 | 2× depth |
| Hidden | 768 | 1024 | 1.33× width |
| Heads | 12 | 16 | More parallel |
| Params | 125M | 336M | 2.7× |

### Recommended Settings

```bash
python train.py \
    --model-size medium \
    --batch-size 16 \
    --seq-len 512 \
    --gradient-accumulation-steps 2 \
    --learning-rate 3e-4 \
    --max-iters 50000
```

---

## Configuration: Large (~760M)

```python
ModelConfig.large() = ModelConfig(
    vocab_size=32000,
    n_layers=24,
    n_heads=16,
    n_kv_heads=8,       # GQA ratio: 16/8 = 2 (less compression)
    d_model=1536,       # 1.5× medium
    d_ff=4096,
    max_seq_len=2048,
)
```

### Parameter Calculation

```
head_dim = 1536 / 16 = 96  (larger heads!)

Embedding: 32000 × 1536 = 49,152,000

Per Layer Attention:
  W_q: 1536 × 1536 = 2,359,296
  W_k: 1536 × 768 = 1,179,648  (n_kv_heads × head_dim = 8 × 96)
  W_v: 1536 × 768 = 1,179,648
  W_o: 1536 × 1536 = 2,359,296
  Total: 7,077,888

Per Layer FFN:
  3 × (1536 × 4096) = 18,874,368

Per Layer Norms: 3,072

Per Layer Total: 25,955,328

All Layers: 24 × 25,955,328 = 622,927,872

Final Norm + LM Head: 1536 + 49,152,000 = 49,153,536

═══════════════════════════════════════
Total: 49,152,000 + 622,927,872 + 49,153,536
     = 721,233,408 parameters (~721M)
═══════════════════════════════════════
```

### Recommended Settings

```bash
python train.py \
    --model-size large \
    --batch-size 8 \
    --seq-len 1024 \
    --gradient-accumulation-steps 4 \
    --learning-rate 1e-4 \
    --compile \
    --max-iters 100000
```

---

## Configuration: GPT-OSS Style (~2.7B)

```python
ModelConfig.gpt_oss_style() = ModelConfig(
    vocab_size=32000,
    n_layers=24,
    n_heads=64,         # Many heads
    n_kv_heads=8,       # GQA ratio: 64/8 = 8 (aggressive compression)
    d_model=2880,       # Large hidden dim
    d_ff=7680,          # ~2.67× d_model (SwiGLU optimal)
    max_seq_len=2048,
)
```

### Parameter Calculation

```
head_dim = 2880 / 64 = 45  (smaller heads, more of them)

Embedding: 32000 × 2880 = 92,160,000

Per Layer Attention:
  W_q: 2880 × 2880 = 8,294,400
  W_k: 2880 × 360 = 1,036,800   (8 × 45 = 360)
  W_v: 2880 × 360 = 1,036,800
  W_o: 2880 × 2880 = 8,294,400
  Total: 18,662,400

Per Layer FFN:
  3 × (2880 × 7680) = 66,355,200

Per Layer Norms: 5,760

Per Layer Total: 85,023,360

All Layers: 24 × 85,023,360 = 2,040,560,640

Final Norm + LM Head: 2880 + 92,160,000 = 92,162,880

═══════════════════════════════════════
Total: 92,160,000 + 2,040,560,640 + 92,162,880
     = 2,224,883,520 parameters (~2.2B)
═══════════════════════════════════════
```

### Recommended Settings

```bash
# Multi-GPU required
torchrun --nproc_per_node=4 train.py \
    --model-size gpt-oss \
    --batch-size 4 \
    --seq-len 2048 \
    --gradient-accumulation-steps 8 \
    --learning-rate 1e-4 \
    --compile
```

---

## Key Parameter Relationships

### 1. Head Dimension

```
head_dim = d_model / n_heads

Examples:
- Small:    768 / 12 = 64
- Medium:   1024 / 16 = 64
- Large:    1536 / 16 = 96
- GPT-OSS:  2880 / 64 = 45
```

**Trade-off**: Smaller head_dim = more heads = more parallelism, but each head sees less.

### 2. GQA Compression Ratio

```
gqa_ratio = n_heads / n_kv_heads

Examples:
- Small:    12 / 4 = 3×
- Medium:   16 / 4 = 4×
- Large:    16 / 8 = 2×
- GPT-OSS:  64 / 8 = 8×
```

**Effect on KV Cache**:
```
Standard MHA cache size: batch × seq × n_heads × head_dim × 2
GQA cache size:          batch × seq × n_kv_heads × head_dim × 2

Savings = n_heads / n_kv_heads
```

### 3. FFN Expansion Ratio

```
ffn_ratio = d_ff / d_model

For SwiGLU, optimal is ~2.67× (since it has 3 matrices vs 2)

Examples:
- Small:    2048 / 768 = 2.67×  ✓
- Medium:   2816 / 1024 = 2.75× ✓
- Large:    4096 / 1536 = 2.67× ✓
- GPT-OSS:  7680 / 2880 = 2.67× ✓
```

### 4. Depth vs Width

```
┌─────────────────────────────────────────────────────────────┐
│ Same parameter budget, different allocation:                 │
│                                                              │
│ Deep & Narrow:          Wide & Shallow:                      │
│ ┌───┐                   ┌─────────────┐                     │
│ │   │                   │             │                     │
│ ├───┤                   │             │                     │
│ │   │                   │             │                     │
│ ├───┤                   │             │                     │
│ │   │                   └─────────────┘                     │
│ ├───┤                   ┌─────────────┐                     │
│ │   │                   │             │                     │
│ ├───┤                   └─────────────┘                     │
│ │   │                                                        │
│ └───┘                   Better for parallelism               │
│ Better for reasoning    Better for knowledge                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Custom Configuration Guide

### Creating Custom Config

```python
from model import ModelConfig, GPT

config = ModelConfig(
    vocab_size=32000,      # Match your tokenizer
    n_layers=16,           # Depth
    n_heads=16,            # Must divide d_model evenly
    n_kv_heads=4,          # Must divide n_heads evenly
    d_model=1024,          # Hidden dimension
    d_ff=2816,             # ~2.67 × d_model for SwiGLU
    max_seq_len=4096,      # Context length
    dropout=0.1,           # Regularization
    rope_theta=10000.0,    # RoPE base (increase for longer context)
    use_sink_attention=True,  # GPT-OSS style sinks
    window_size=0,         # 0=full, >0=sliding window
)

model = GPT(config)
print(f"Parameters: {model.count_parameters():,}")
```

### Constraints

```python
# These must hold:
assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

# Recommendations:
assert d_ff >= 2 * d_model, "FFN should be at least 2× hidden"
assert d_ff <= 4 * d_model, "FFN larger than 4× is unusual"
```

### Scaling Laws

```
For optimal scaling, maintain these ratios:

d_model ∝ n_layers^0.5      (width grows slower than depth)
n_heads ∝ d_model / 64      (keep head_dim around 64-128)
d_ff ≈ 2.67 × d_model       (SwiGLU optimal)

Example scaling:
- 100M:  n_layers=12, d_model=768
- 300M:  n_layers=24, d_model=1024
- 1B:    n_layers=24, d_model=2048
- 7B:    n_layers=32, d_model=4096
```

---

## Memory Estimation Calculator

```python
def estimate_memory(config, batch_size, seq_len, precision="bf16"):
    """Estimate training memory requirements."""
    bytes_per_param = 2 if precision in ["bf16", "fp16"] else 4

    # Count parameters
    n_params = (
        config.vocab_size * config.d_model +  # embedding
        config.n_layers * (
            # attention
            config.d_model * config.d_model +  # W_q
            config.d_model * (config.n_kv_heads * (config.d_model // config.n_heads)) * 2 +  # W_k, W_v
            config.d_model * config.d_model +  # W_o
            # ffn
            3 * config.d_model * config.d_ff +
            # norms
            2 * config.d_model
        ) +
        config.d_model +  # final norm
        config.d_model * config.vocab_size  # lm_head
    )

    # Memory breakdown
    params_mem = n_params * bytes_per_param
    grads_mem = n_params * bytes_per_param
    optimizer_mem = n_params * 8  # Adam states in fp32

    # Activation memory (rough estimate)
    activation_mem = batch_size * seq_len * config.d_model * config.n_layers * bytes_per_param * 4

    total = params_mem + grads_mem + optimizer_mem + activation_mem

    return {
        "parameters": n_params,
        "params_gb": params_mem / 1e9,
        "grads_gb": grads_mem / 1e9,
        "optimizer_gb": optimizer_mem / 1e9,
        "activation_gb": activation_mem / 1e9,
        "total_gb": total / 1e9
    }

# Example usage:
config = ModelConfig.small()
mem = estimate_memory(config, batch_size=32, seq_len=512)
print(f"Estimated memory: {mem['total_gb']:.1f} GB")
```

---

## Comparison Summary

| Aspect | Small | Medium | Large | GPT-OSS |
|--------|-------|--------|-------|---------|
| **Parameters** | 125M | 336M | 721M | 2.2B |
| **Layers** | 12 | 24 | 24 | 24 |
| **Hidden** | 768 | 1024 | 1536 | 2880 |
| **Heads** | 12 | 16 | 16 | 64 |
| **KV Heads** | 4 | 4 | 8 | 8 |
| **Head Dim** | 64 | 64 | 96 | 45 |
| **GQA Ratio** | 3× | 4× | 2× | 8× |
| **FFN** | 2048 | 2816 | 4096 | 7680 |
| **FFN Ratio** | 2.67× | 2.75× | 2.67× | 2.67× |
| **Min VRAM** | 4 GB | 8 GB | 16 GB | 48 GB |
| **Training Time** | Hours | Days | Days | Weeks |

---

## When to Use Each

### Small (~125M)
- Learning and experimentation
- Quick iteration on ideas
- Limited hardware (single GPU, 4-8GB)
- Small datasets (< 100MB)

### Medium (~350M)
- Serious experiments
- Moderate hardware (single GPU, 16GB)
- Medium datasets (100MB - 1GB)
- When small is too weak but large is too slow

### Large (~760M)
- Production-quality models
- Good hardware (single GPU 24GB or multi-GPU)
- Large datasets (1GB+)
- When quality matters more than speed

### GPT-OSS Style (~2.7B)
- Research on large models
- Multi-GPU setup required
- Very large datasets
- Matching GPT-OSS architecture for comparison

---

← Back to [Documentation Index](index.md)
