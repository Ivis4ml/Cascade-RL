# Minimal LLM Training Tutorial

This tutorial provides a comprehensive guide to using this minimal LLM training codebase, covering the principles behind each component and practical usage instructions.

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Model Architecture (model.py)](#2-model-architecture-modelpy)
3. [Training Script (train.py)](#3-training-script-trainpy)
4. [Tokenizer (tokenizer.py)](#4-tokenizer-tokenizerpy)
5. [Data Preparation (data.py)](#5-data-preparation-datapy)
6. [Text Generation (generate.py)](#6-text-generation-generatepy)
7. [Complete Training Example](#7-complete-training-example)
8. [Architecture Deep Dive](#8-architecture-deep-dive)

---

## 1. Project Overview

This is a minimal LLM training codebase implemented entirely in PyTorch, inspired by [nanochat](https://github.com/karpathy/nanochat) and based on [GPT-OSS 20B](https://huggingface.co/openai/gpt-oss-20b) architecture patterns.

### Core Features

| Feature | Traditional GPT | This Implementation (GPT-OSS Style) |
|---------|-----------------|-------------------------------------|
| Normalization | LayerNorm | **RMSNorm** |
| Position Encoding | Learned Absolute | **RoPE (Rotary Position Embedding)** |
| Attention | MHA (Multi-Head Attention) | **GQA (Grouped Query Attention)** |
| Activation | GELU | **SwiGLU** |
| Linear Layer Bias | Yes | **No** |

### File Structure

```
├── model.py        # Model definition (GPT + all components)
├── train.py        # Training loop
├── tokenizer.py    # Tokenizer implementations
├── data.py         # Data loading and preprocessing
├── generate.py     # Inference and text generation
└── requirements.txt
```

---

## 2. Model Architecture (model.py)

### 2.1 Model Configuration (ModelConfig)

```python
from model import ModelConfig

# Use preset configurations
config = ModelConfig.small()   # ~125M parameters
config = ModelConfig.medium()  # ~350M parameters
config = ModelConfig.large()   # ~760M parameters
config = ModelConfig.gpt_oss_style()  # GPT-OSS 20B style

# Or customize
config = ModelConfig(
    vocab_size=32000,      # Vocabulary size
    n_layers=12,           # Number of transformer layers
    n_heads=12,            # Number of attention heads
    n_kv_heads=4,          # Number of KV heads (for GQA)
    d_model=768,           # Hidden dimension
    d_ff=2048,             # FFN intermediate dimension
    max_seq_len=2048,      # Maximum sequence length
    dropout=0.0,           # Dropout rate
    rope_theta=10000.0     # RoPE base frequency
)
```

### 2.2 RMSNorm (Root Mean Square Normalization)

RMSNorm is a simplified version of LayerNorm that only scales without shifting:

```python
class RMSNorm(nn.Module):
    """
    RMSNorm(x) = x / RMS(x) * γ
    where RMS(x) = sqrt(mean(x²))

    Advantages:
    - Faster computation (no mean calculation)
    - Fewer parameters (no bias)
    - Comparable training stability
    """
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight
```

### 2.3 RoPE (Rotary Position Embeddings)

RoPE encodes positional information through vector rotation:

```python
def precompute_rope_freqs(dim, max_seq_len, theta=10000.0):
    """
    Precompute RoPE frequencies

    Principle: Encode position as complex rotation
    - Encoding for position p: e^(i * p * θ)
    - θ = 10000^(-2k/d), where k is the dimension index

    Advantages:
    - Relative position information naturally embedded
    - Can extrapolate to longer sequences
    - No learned position parameters
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rope(q, k, freqs_cis):
    """Apply RoPE to Q and K"""
    q_complex = torch.view_as_complex(q.reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.reshape(*k.shape[:-1], -1, 2))
    # Complex multiplication = rotation
    q_rotated = torch.view_as_real(q_complex * freqs_cis)
    k_rotated = torch.view_as_real(k_complex * freqs_cis)
    return q_rotated.flatten(-2), k_rotated.flatten(-2)
```

### 2.4 GQA (Grouped Query Attention)

GQA allows multiple Query heads to share the same Key/Value heads:

```python
class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention

    Traditional MHA: n_heads Q, K, V heads (1:1:1)
    MQA:            n_heads Q heads, 1 K, V head (n:1:1)
    GQA:            n_heads Q heads, n_kv_heads K, V heads (n:k:k)

    Example: n_heads=64, n_kv_heads=8
    Every 8 Q heads share 1 KV head

    Advantages:
    - Reduced KV cache size (faster inference)
    - Maintains model quality
    """
    def __init__(self, config):
        self.n_rep = config.n_heads // config.n_kv_heads  # Q/KV ratio

        # Q projection: d_model -> n_heads * head_dim
        self.wq = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        # K, V projection: d_model -> n_kv_heads * head_dim (smaller)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
```

### 2.5 SwiGLU (Gated Linear Unit with Swish)

SwiGLU combines Swish activation with a gating mechanism:

```python
class SwiGLU(nn.Module):
    """
    SwiGLU(x) = Swish(xW_gate) ⊙ (xW_up)

    where Swish(x) = x * sigmoid(x) = SiLU(x)

    Compared to traditional FFN:
    - Traditional: GELU(xW1) @ W2
    - SwiGLU: (Swish(xW_gate) * xW_up) @ W_down

    Advantages:
    - Better gradient flow
    - Better performance in practice
    """
    def forward(self, x):
        gate = F.silu(self.w_gate(x))  # Swish activation
        up = self.w_up(x)
        return self.w_down(gate * up)  # Gating + projection
```

### 2.6 Complete Model Usage

```python
from model import GPT, ModelConfig
import torch

# Create model
config = ModelConfig.small()
model = GPT(config)
print(f"Parameters: {model.count_parameters():,}")  # ~125M

# Forward pass
input_ids = torch.randint(0, config.vocab_size, (2, 128))  # [batch, seq_len]
targets = torch.randint(0, config.vocab_size, (2, 128))

logits, loss, _ = model(input_ids, targets)
print(f"Logits shape: {logits.shape}")  # [2, 128, vocab_size]
print(f"Loss: {loss.item():.4f}")

# Text generation
generated = model.generate(
    input_ids[:, :10],  # First 10 tokens as prompt
    max_new_tokens=50,
    temperature=0.8,
    top_k=50
)
```

---

## 3. Training Script (train.py)

### 3.1 Training Configuration

```python
from train import TrainConfig

config = TrainConfig(
    # Data
    batch_size=32,                      # Batch size
    seq_len=512,                        # Sequence length
    gradient_accumulation_steps=4,      # Gradient accumulation steps

    # Training
    max_iters=10000,                    # Total iterations
    eval_interval=500,                  # Evaluation interval
    log_interval=10,                    # Logging interval

    # Optimizer
    learning_rate=3e-4,                 # Peak learning rate
    min_lr=3e-5,                        # Minimum learning rate
    weight_decay=0.1,                   # Weight decay
    beta1=0.9, beta2=0.95,             # Adam parameters
    grad_clip=1.0,                      # Gradient clipping

    # LR Schedule
    warmup_iters=500,                   # Warmup steps
    lr_decay_iters=10000,               # Decay end step

    # System
    device="cuda",
    dtype="bfloat16",                   # Mixed precision
    compile=False,                      # torch.compile
)
```

### 3.2 Learning Rate Schedule

Uses Warmup + Cosine Decay:

```
Learning Rate
  ^
  |      /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
  |     /                    \
  |    /                       \
  |   /                          \_____ min_lr
  |  /
  | /
  +----------------------------------------> Iteration
    warmup     constant      decay
```

```python
def get_lr(iter_num, config):
    # 1. Linear warmup
    if iter_num < config.warmup_iters:
        return config.learning_rate * iter_num / config.warmup_iters

    # 2. Cosine decay
    if iter_num > config.lr_decay_iters:
        return config.min_lr

    decay_ratio = (iter_num - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)
```

### 3.3 Command Line Usage

```bash
# Basic training (with synthetic data for testing)
python train.py --model-size small --max-iters 1000

# Using real data
python train.py \
    --data-path data/shakespeare.bin \
    --model-size small \
    --batch-size 16 \
    --seq-len 256 \
    --max-iters 5000 \
    --learning-rate 1e-3

# Large model training
python train.py \
    --model-size large \
    --batch-size 8 \
    --gradient-accumulation-steps 8 \
    --compile  # Use torch.compile for speedup

# Resume from checkpoint
python train.py \
    --resume-from checkpoints/ckpt_5000.pt \
    --max-iters 10000
```

### 3.4 Distributed Training

```bash
# Single node multi-GPU (DDP)
torchrun --nproc_per_node=4 train.py \
    --model-size medium \
    --batch-size 8

# Multi-node multi-GPU
torchrun --nnodes=2 --nproc_per_node=8 \
    --node_rank=0 --master_addr=<MASTER_IP> \
    train.py --model-size large
```

---

## 4. Tokenizer (tokenizer.py)

### 4.1 Character-level Tokenizer

Suitable for quick testing and small datasets:

```python
from tokenizer import CharTokenizer

# Create from text
text = "Hello, World!"
tokenizer = CharTokenizer.from_text(text)
print(f"Vocab size: {tokenizer.vocab_size}")  # Number of unique characters

# Encode/decode
tokens = tokenizer.encode("Hello")  # [7, 4, 11, 11, 14]
text = tokenizer.decode(tokens)     # "Hello"

# Save/load
tokenizer.save("vocab.json")
tokenizer = CharTokenizer.load("vocab.json")
```

### 4.2 BPE Tokenizer (tiktoken)

Suitable for production:

```python
from tokenizer import BPETokenizer

tokenizer = BPETokenizer("cl100k_base")  # GPT-4's encoding
print(f"Vocab size: {tokenizer.vocab_size}")  # ~100k

tokens = tokenizer.encode("Hello, World!")
text = tokenizer.decode(tokens)
```

---

## 5. Data Preparation (data.py)

### 5.1 Download Sample Data

```python
from data import download_shakespeare, prepare_shakespeare

# Download Shakespeare dataset (~1MB)
text_path = download_shakespeare("data/")

# Download and preprocess (tokenize + save as binary)
bin_path = prepare_shakespeare("data/")
# Generates: data/shakespeare.bin, data/shakespeare_vocab.json
```

### 5.2 Prepare Custom Data

```python
from data import prepare_data
from tokenizer import CharTokenizer

# Prepare text file
prepare_data(
    input_path="my_corpus.txt",
    output_path="data/my_corpus.bin",
    tokenizer=None  # Auto-create character-level tokenizer
)

# Or use custom tokenizer
from tokenizer import BPETokenizer
tokenizer = BPETokenizer()
prepare_data(
    input_path="my_corpus.txt",
    output_path="data/my_corpus.bin",
    tokenizer=tokenizer
)
```

### 5.3 Dataset Classes

```python
from train import TextDataset, SyntheticDataset

# Text dataset
dataset = TextDataset(
    data_path="data/shakespeare.txt",  # or .bin
    seq_len=512,
    split="train",      # "train" or "val"
    train_split=0.9     # 90% train, 10% val
)

# Synthetic dataset (for debugging)
dataset = SyntheticDataset(
    vocab_size=32000,
    seq_len=512,
    num_samples=10000
)

# Get a sample
x, y = dataset[0]  # x: input, y: target (shifted by 1)
```

---

## 6. Text Generation (generate.py)

### 6.1 Command Line Usage

```bash
# Single generation
python generate.py \
    --checkpoint checkpoints/ckpt_5000.pt \
    --prompt "To be or not to be" \
    --max-tokens 200 \
    --temperature 0.8

# Interactive mode
python generate.py \
    --checkpoint checkpoints/ckpt_5000.pt \
    --tokenizer char \
    --vocab-path data/shakespeare_vocab.json
```

### 6.2 Sampling Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `temperature` | Controls randomness, higher = more random | 0.7-1.0 |
| `top_k` | Sample only from top k most likely tokens | 40-100 |
| `top_p` | Nucleus sampling, cumulative probability threshold | 0.9-0.95 |

```python
# Deterministic generation (greedy)
output = model.generate(input_ids, temperature=0.0)

# Creative generation
output = model.generate(input_ids, temperature=1.0, top_k=50, top_p=0.9)

# Balanced generation
output = model.generate(input_ids, temperature=0.7, top_k=40)
```

### 6.3 Python API

```python
from generate import load_model, generate_text
from tokenizer import CharTokenizer

# Load model
model, config = load_model("checkpoints/ckpt_5000.pt", device="cuda")

# Load tokenizer
tokenizer = CharTokenizer.load("data/shakespeare_vocab.json")

# Generate
text = generate_text(
    model, tokenizer,
    prompt="ROMEO: ",
    max_new_tokens=200,
    temperature=0.8
)
print(text)
```

---

## 7. Complete Training Example

### 7.1 Training Shakespeare Model

```bash
# Step 1: Prepare data
python data.py --dataset shakespeare

# Step 2: Train
python train.py \
    --data-path data/shakespeare.bin \
    --model-size small \
    --batch-size 32 \
    --seq-len 256 \
    --max-iters 5000 \
    --learning-rate 1e-3 \
    --log-interval 100 \
    --eval-interval 500

# Step 3: Generate
python generate.py \
    --checkpoint checkpoints/ckpt_5000.pt \
    --vocab-path data/shakespeare_vocab.json \
    --prompt "HAMLET: To be or not"
```

### 7.2 Training Larger Models

```bash
# Medium model (~350M) - requires ~8GB VRAM
python train.py \
    --data-path data/my_corpus.bin \
    --model-size medium \
    --batch-size 8 \
    --gradient-accumulation-steps 4 \
    --max-iters 50000 \
    --compile

# Large model (~760M) - requires ~16GB VRAM
python train.py \
    --model-size large \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --compile
```

---

## 8. Architecture Deep Dive

### 8.1 Parameter Count Calculation

For a Transformer model, main parameter distribution:

```
Token embedding:    vocab_size × d_model
Per-layer attention: 4 × d_model × d_model (for MHA)
                    (1 + 1/n_rep × 2) × d_model² (for GQA)
Per-layer FFN:      3 × d_model × d_ff (for SwiGLU)
Output layer:       d_model × vocab_size

Total ≈ 12 × n_layers × d_model² (approximation)
```

### 8.2 Memory Estimation

Training memory usage:

```
Parameters:         P bytes (fp32: 4P, bf16: 2P)
Gradients:          P bytes
Optimizer states:   8P bytes (Adam: m + v)
Activations:        depends on batch_size × seq_len
```

Rough estimates:
- Small (~125M): ~4GB
- Medium (~350M): ~8GB
- Large (~760M): ~16GB

### 8.3 Training Tips

1. **Gradient Accumulation**: Simulate large batch with small batches
   ```bash
   --batch-size 4 --gradient-accumulation-steps 8  # effective batch=32
   ```

2. **Mixed Precision**: Uses bfloat16/float16 by default

3. **Gradient Clipping**: Default `grad_clip=1.0` prevents gradient explosion

4. **Learning Rate**:
   - Small models: 1e-3 ~ 3e-4
   - Large models: 3e-4 ~ 1e-4

5. **torch.compile**: PyTorch 2.0+ can speedup by 20-30%
   ```bash
   --compile
   ```

---

## FAQ

**Q: Why use RMSNorm instead of LayerNorm?**
A: RMSNorm is faster to compute, has fewer parameters, and performs comparably or better for LLMs.

**Q: How to choose n_kv_heads for GQA?**
A: Typically n_heads / n_kv_heads = 4~8. For example, 64 Q heads with 8 KV heads.

**Q: How to set d_ff for SwiGLU?**
A: Typically d_ff ≈ 2.67 × d_model (since SwiGLU has 3 matrices instead of 2).

**Q: Training not converging?**
A: Try lowering learning rate, increasing warmup, or checking data quality.

---

## References

- [nanochat](https://github.com/karpathy/nanochat) - Karpathy's minimal implementation
- [GPT-OSS 20B](https://huggingface.co/openai/gpt-oss-20b) - OpenAI's open-source model
- [RoPE Paper](https://arxiv.org/abs/2104.09864) - Rotary Position Embedding
- [GQA Paper](https://arxiv.org/abs/2305.13245) - Grouped Query Attention
- [SwiGLU Paper](https://arxiv.org/abs/2002.05202) - GLU Variants
