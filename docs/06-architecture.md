# Step 6: Model Architecture

This guide provides a deep dive into the model architecture based on GPT-OSS 20B patterns.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         GPT Model                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input IDs: [batch, seq_len]                                   │
│        ↓                                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │            Token Embedding (no position emb)             │   │
│   │            [vocab_size, d_model]                        │   │
│   └─────────────────────────────────────────────────────────┘   │
│        ↓                                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                  Transformer Block × N                   │   │
│   │  ┌───────────────────────────────────────────────────┐  │   │
│   │  │  RMSNorm → GQA (with RoPE) → Residual             │  │   │
│   │  └───────────────────────────────────────────────────┘  │   │
│   │  ┌───────────────────────────────────────────────────┐  │   │
│   │  │  RMSNorm → SwiGLU FFN → Residual                  │  │   │
│   │  └───────────────────────────────────────────────────┘  │   │
│   └─────────────────────────────────────────────────────────┘   │
│        ↓                                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                       RMSNorm                            │   │
│   └─────────────────────────────────────────────────────────┘   │
│        ↓                                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              LM Head (Linear, no bias)                   │   │
│   │              [d_model, vocab_size]                       │   │
│   └─────────────────────────────────────────────────────────┘   │
│        ↓                                                        │
│   Output Logits: [batch, seq_len, vocab_size]                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component 1: RMSNorm

### What It Does

Normalizes activations using Root Mean Square:

```
RMSNorm(x) = x / RMS(x) × γ

where RMS(x) = √(mean(x²) + ε)
```

### Comparison with LayerNorm

| Feature | LayerNorm | RMSNorm |
|---------|-----------|---------|
| Mean removal | Yes | No |
| Variance scaling | Yes | Yes (RMS) |
| Learnable bias | Yes | No |
| Learnable scale | Yes | Yes |
| Speed | Slower | Faster |

### Code

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # γ (scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # Scale
        return x * rms * self.weight
```

### Why RMSNorm?

1. **Faster**: ~15% speedup (no mean computation)
2. **Simpler**: Fewer parameters
3. **Effective**: Works as well or better for LLMs

---

## Component 2: RoPE (Rotary Position Embedding)

### What It Does

Encodes position information through rotation in complex number space.

### Intuition

```
Position 0: Rotate by 0°
Position 1: Rotate by θ
Position 2: Rotate by 2θ
Position 3: Rotate by 3θ
...

When computing attention:
Q_pos5 · K_pos3 → rotation captures relative distance (5-3=2)
```

### Mathematical Formulation

```
For dimension pair (2i, 2i+1) at position m:

θᵢ = 10000^(-2i/d)

[q₂ᵢ]     [cos(mθᵢ)  -sin(mθᵢ)] [q₂ᵢ]
[q₂ᵢ₊₁] = [sin(mθᵢ)   cos(mθᵢ)] [q₂ᵢ₊₁]
```

### Code

```python
def precompute_rope_freqs(dim, max_seq_len, theta=10000.0):
    # Compute frequencies: θᵢ = 10000^(-2i/d)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # Position indices
    t = torch.arange(max_seq_len)
    # Outer product: [seq_len, dim/2]
    freqs = torch.outer(t, freqs)
    # Complex representation: e^(iθ)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rope(q, k, freqs_cis):
    # View as complex numbers
    q_complex = torch.view_as_complex(q.reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.reshape(*k.shape[:-1], -1, 2))
    # Rotate
    q_rotated = q_complex * freqs_cis
    k_rotated = k_complex * freqs_cis
    # Back to real
    return torch.view_as_real(q_rotated).flatten(-2), \
           torch.view_as_real(k_rotated).flatten(-2)
```

### Why RoPE?

1. **Relative positions**: Naturally encodes relative distance
2. **Extrapolation**: Can extend to longer sequences than trained
3. **No learned parameters**: Purely mathematical
4. **Efficient**: Applied to Q and K only

---

## Component 3: GQA (Grouped Query Attention)

### What It Does

Groups multiple Query heads to share Key/Value heads, reducing memory and computation.

### Attention Variants

```
MHA (Multi-Head Attention):
Q heads: [H₁][H₂][H₃][H₄][H₅][H₆][H₇][H₈]
K heads: [H₁][H₂][H₃][H₄][H₅][H₆][H₇][H₈]
V heads: [H₁][H₂][H₃][H₄][H₅][H₆][H₇][H₈]
Ratio: 1:1:1

MQA (Multi-Query Attention):
Q heads: [H₁][H₂][H₃][H₄][H₅][H₆][H₇][H₈]
K heads: [        H₁ (shared)              ]
V heads: [        H₁ (shared)              ]
Ratio: 8:1:1

GQA (Grouped Query Attention):
Q heads: [H₁][H₂][H₃][H₄][H₅][H₆][H₇][H₈]
K heads: [  H₁  ][  H₂  ][  H₃  ][  H₄  ]
V heads: [  H₁  ][  H₂  ][  H₃  ][  H₄  ]
Ratio: 2:1:1 (2 Q heads per KV head)
```

### Code

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        self.n_heads = config.n_heads      # 64
        self.n_kv_heads = config.n_kv_heads  # 8
        self.n_rep = self.n_heads // self.n_kv_heads  # 8

        # Q projection: full size
        self.wq = nn.Linear(config.d_model, self.n_heads * self.head_dim, bias=False)
        # KV projections: reduced size
        self.wk = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)

    def forward(self, x, freqs_cis, mask, kv_cache):
        # Project
        q = self.wq(x)  # [B, S, n_heads * head_dim]
        k = self.wk(x)  # [B, S, n_kv_heads * head_dim]
        v = self.wv(x)  # [B, S, n_kv_heads * head_dim]

        # Reshape
        q = q.view(B, S, self.n_heads, self.head_dim)
        k = k.view(B, S, self.n_kv_heads, self.head_dim)
        v = v.view(B, S, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        q, k = apply_rope(q, k, freqs_cis)

        # Expand KV to match Q (repeat_interleave)
        k = k.repeat_interleave(self.n_rep, dim=2)  # [B, S, n_heads, head_dim]
        v = v.repeat_interleave(self.n_rep, dim=2)

        # Standard attention
        attn = (q @ k.transpose(-2, -1)) / sqrt(head_dim)
        attn = softmax(attn + mask)
        out = attn @ v
```

### Why GQA?

1. **Memory efficient**: KV cache is n_kv_heads × smaller
2. **Faster inference**: Less memory bandwidth
3. **Quality preserved**: Almost no quality loss vs MHA

### KV Cache Size Comparison

| Attention | KV Cache Size | Relative |
|-----------|---------------|----------|
| MHA (n=64) | 64 × 2 × head_dim | 100% |
| GQA (n=64, kv=8) | 8 × 2 × head_dim | 12.5% |
| MQA (n=64, kv=1) | 1 × 2 × head_dim | 1.6% |

---

## Component 4: SwiGLU (Swish-Gated Linear Unit)

### What It Does

A gated activation function that combines Swish with a linear gating mechanism.

### Formulas

```
Traditional FFN:
FFN(x) = GELU(xW₁) W₂

SwiGLU:
SwiGLU(x) = Swish(xW_gate) ⊙ (xW_up) W_down

where Swish(x) = x × sigmoid(x) = SiLU(x)
```

### Visualization

```
                    ┌─────────┐
              ┌────►│ W_gate  │────► Swish ────┐
              │     └─────────┘                │
Input ────────┤                                ⊙────► W_down ────► Output
              │     ┌─────────┐                │
              └────►│  W_up   │────────────────┘
                    └─────────┘
```

### Code

```python
class SwiGLU(nn.Module):
    def __init__(self, config):
        self.w_gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w_up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w_down = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))  # Swish activation
        up = self.w_up(x)
        return self.w_down(gate * up)
```

### Why SwiGLU?

1. **Better gradients**: Swish has smooth gradients everywhere
2. **Gating**: Selective information flow
3. **Empirically better**: Outperforms GELU in practice

### Parameter Count Note

SwiGLU has 3 matrices vs 2 for standard FFN:
- To keep parameter count similar, use d_ff ≈ 2.67 × d_model
- Standard: d_ff = 4 × d_model, 2 matrices = 8 d_model²
- SwiGLU: d_ff ≈ 2.67 × d_model, 3 matrices ≈ 8 d_model²

---

## Model Configurations

### Preset Configurations

| Config | n_layers | d_model | n_heads | n_kv_heads | d_ff | Params |
|--------|----------|---------|---------|------------|------|--------|
| small | 12 | 768 | 12 | 4 | 2048 | ~125M |
| medium | 24 | 1024 | 16 | 4 | 2816 | ~350M |
| large | 24 | 1536 | 16 | 8 | 4096 | ~760M |
| gpt-oss | 24 | 2880 | 64 | 8 | 7680 | ~2.7B |

### GPT-OSS 20B Reference

The actual GPT-OSS 20B uses:
- 24 layers
- 2880 hidden dimension
- 64 query heads, 8 KV heads (GQA)
- 32 MoE experts (not implemented in this codebase)
- 128k context length

---

## Parameter Count Calculation

### Formula

```
Embedding:     vocab_size × d_model
Per Layer:
  - Attention: d_model × d_model × (1 + 2/n_rep + 1)  # Q, K, V, O
             = d_model² × (2 + 2/n_rep)
  - FFN:      3 × d_model × d_ff  # gate, up, down
  - Norms:    2 × d_model  # 2 RMSNorms
LM Head:       d_model × vocab_size

Total ≈ 2 × vocab_size × d_model + n_layers × (2 × d_model² + 3 × d_model × d_ff)
```

### Example: Small Model

```python
vocab_size = 32000
d_model = 768
n_layers = 12
d_ff = 2048
n_heads = 12
n_kv_heads = 4
n_rep = 3  # n_heads / n_kv_heads

# Embedding
emb = vocab_size * d_model  # 24,576,000

# Per layer attention
attn_q = d_model * (n_heads * 64)      # 589,824
attn_k = d_model * (n_kv_heads * 64)   # 196,608
attn_v = d_model * (n_kv_heads * 64)   # 196,608
attn_o = (n_heads * 64) * d_model      # 589,824
attn_total = attn_q + attn_k + attn_v + attn_o  # 1,572,864

# Per layer FFN
ffn_gate = d_model * d_ff  # 1,572,864
ffn_up = d_model * d_ff    # 1,572,864
ffn_down = d_ff * d_model  # 1,572,864
ffn_total = ffn_gate + ffn_up + ffn_down  # 4,718,592

# Per layer norms
norms = 2 * d_model  # 1,536

# Total per layer
per_layer = attn_total + ffn_total + norms  # 6,292,992

# All layers
all_layers = n_layers * per_layer  # 75,515,904

# Final norm
final_norm = d_model  # 768

# LM head
lm_head = d_model * vocab_size  # 24,576,000

# Total
total = emb + all_layers + final_norm + lm_head
# ≈ 124,668,672 parameters
```

---

## Memory Requirements

### Training Memory Breakdown

```
Parameter memory:       P × dtype_size
Gradient memory:        P × dtype_size
Optimizer memory:       P × 8 (Adam: m and v states, fp32)
Activation memory:      B × S × d_model × n_layers × factor

Total ≈ 10P + activations
```

### Inference Memory

```
Parameters only:        P × dtype_size
KV Cache per layer:     B × S × 2 × n_kv_heads × head_dim × dtype_size
Total KV Cache:         n_layers × above
```

### Memory Optimization Tips

1. **Mixed precision**: Use bfloat16 (2 bytes vs 4)
2. **Gradient checkpointing**: Trade compute for memory
3. **GQA**: Reduces KV cache size
4. **Smaller batch**: Most direct way to reduce memory

---

## Initialization

### Weight Initialization

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        # Normal distribution with small std
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

### Residual Scaling

Output projections are scaled by 1/√(2×n_layers):

```python
for name, p in self.named_parameters():
    if name.endswith('wo.weight') or name.endswith('w_down.weight'):
        torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
```

This prevents gradient explosion in deep networks.

---

## Component 5: Attention Sink (StreamingLLM)

### What It Does

Enables processing infinite-length sequences with bounded memory by keeping:
1. **Sink tokens**: First few tokens that absorb attention mass
2. **Window tokens**: Recent tokens in a sliding window

### The Problem

Standard attention has O(n²) complexity and unbounded KV cache:

```
Token 1:     [K₁, V₁]
Token 2:     [K₁, K₂, V₁, V₂]
Token 3:     [K₁, K₂, K₃, V₁, V₂, V₃]
...
Token 10000: [K₁, K₂, ..., K₁₀₀₀₀, V₁, ..., V₁₀₀₀₀]  ← Memory explosion!
```

### The Solution

Keep only sink tokens and recent window:

```
┌─────────────────────────────────────────────────────────┐
│ Standard KV Cache (unbounded):                          │
│ [K₁][K₂][K₃][K₄][K₅][K₆][K₇]...[K₉₉₉₈][K₉₉₉₉][K₁₀₀₀₀] │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Sink Attention (bounded):                               │
│ [K₁][K₂][K₃][K₄]  ...discarded...  [K₉₉₉₇][K₉₉₉₈][K₉₉₉₉][K₁₀₀₀₀]│
│ └─sink tokens─┘                    └────window tokens────┘      │
└─────────────────────────────────────────────────────────┐
```

### Why Sink Tokens?

Research found that initial tokens accumulate attention even when semantically unimportant:

```
Attention pattern (typical):
Token:    [BOS] [The] [cat] [sat] [on] [the] [mat]
Attention: 0.4   0.1   0.1  0.15  0.1  0.05  0.1
           ↑
           "Sink" - high attention despite low semantic value
```

By keeping these sink tokens, the model maintains stable attention distribution.

### Usage

```python
from model import ModelConfig, GPT

# Enable sink attention
config = ModelConfig.small()
config.use_sink_attention = True
config.sink_size = 4        # Keep first 4 tokens as sinks
config.window_size = 1024   # Keep last 1024 tokens

model = GPT(config)

# Now model can handle infinite sequences with bounded memory
# KV cache max size = sink_size + window_size = 1028 tokens
```

### Memory Comparison

| Sequence Length | Standard Cache | Sink Cache (4 + 1024) |
|-----------------|----------------|----------------------|
| 1,000 | 1,000 tokens | 1,028 tokens |
| 10,000 | 10,000 tokens | 1,028 tokens |
| 100,000 | 100,000 tokens | 1,028 tokens |
| ∞ | ∞ (OOM) | 1,028 tokens |

---

## Summary Table

| Component | Purpose | Key Innovation |
|-----------|---------|----------------|
| RMSNorm | Normalization | No mean, faster |
| RoPE | Position encoding | Relative positions, no learned params |
| GQA | Attention | Shared KV heads, smaller cache |
| SwiGLU | Activation | Gated, better gradients |
| Sink Attention | Long sequences | Bounded memory, infinite context |

---

## Further Reading

- [RMSNorm Paper](https://arxiv.org/abs/1910.07467) - Root Mean Square Layer Normalization
- [RoPE Paper](https://arxiv.org/abs/2104.09864) - RoFormer: Enhanced Transformer with Rotary Position Embedding
- [GQA Paper](https://arxiv.org/abs/2305.13245) - GQA: Training Generalized Multi-Query Transformer Models
- [SwiGLU Paper](https://arxiv.org/abs/2002.05202) - GLU Variants Improve Transformer
- [StreamingLLM Paper](https://arxiv.org/abs/2309.17453) - Efficient Streaming Language Models with Attention Sinks
- [GPT-OSS 20B](https://huggingface.co/openai/gpt-oss-20b) - OpenAI's open-source model

---

← Back to [Documentation Index](index.md)
