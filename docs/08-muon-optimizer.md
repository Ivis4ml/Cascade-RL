# Muon Optimizer

The Muon optimizer (**M**oment**U**m **O**rthogonalized by **N**ewton-schulz) is a specialized optimizer for neural network matrix parameters. It combines SGD with momentum and Newton-Schulz orthogonalization to achieve better optimization dynamics.

## Overview

Muon internally runs standard SGD-momentum and then performs an orthogonalization post-processing step. Each 2D parameter's update is replaced with the nearest orthogonal matrix using Newton-Schulz iteration.

### Key Benefits

1. **Better optimization landscape**: Orthogonalized updates help maintain well-conditioned weight matrices
2. **Efficient computation**: Newton-Schulz iteration is fast and numerically stable in bfloat16
3. **Improved convergence**: Often converges faster than AdamW for matrix parameters
4. **Memory efficient**: Lower memory footprint than Adam (no second moment buffer)

## When to Use Muon

Muon is designed for **2D matrix parameters** in transformer blocks:

| Parameter Type | Optimizer | Reason |
|---------------|-----------|--------|
| Linear layer weights | Muon | 2D matrices benefit from orthogonalization |
| Attention Q/K/V weights | Muon | Core transformer computations |
| MLP weights | Muon | Dense matrix operations |
| Embeddings | AdamW | High-dimensional, sparse gradients |
| Output projection (lm_head) | AdamW | Tied with embeddings or special |
| Biases | AdamW | 1D vectors, not matrices |
| LayerNorm/RMSNorm | AdamW | 1D scaling parameters |

## Usage

### Basic Usage

Enable Muon with the `--use-muon` flag:

```bash
python train.py --use-muon --data-path data/train.txt
```

### With Custom Learning Rates

```bash
python train.py \
    --use-muon \
    --muon-lr 0.02 \
    --learning-rate 3e-4 \
    --data-path data/train.txt
```

### Recommended Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use-muon` | False | Enable Muon optimizer |
| `--muon-lr` | 0.02 | Learning rate for Muon (higher than AdamW) |
| `--muon-momentum` | 0.95 | Momentum for Muon optimizer |
| `--learning-rate` | 3e-4 | AdamW learning rate (for embeddings/biases) |

## How It Works

### 1. Newton-Schulz Orthogonalization

The key innovation is using Newton-Schulz iteration to orthogonalize gradient updates:

```python
def zeropower_via_newtonschulz5(G, steps=5):
    """
    Compute nearest orthogonal matrix using Newton-Schulz iteration.
    Returns something like U @ V.T from SVD: G = U @ S @ V.T
    """
    # Quintic polynomial coefficients (optimized for fast convergence)
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Normalize to spectral norm ~1
    X = G / G.norm()

    # Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    return X
```

This converges to an orthogonal matrix much faster than computing full SVD.

### 2. Parameter Routing

Our implementation automatically routes parameters:

```python
for name, param in model.named_parameters():
    if is_embedding or is_head or is_norm or is_bias or param.ndim < 2:
        # AdamW for embeddings, heads, norms, biases
        adamw_params.append(param)
    else:
        # Muon for 2D matrix parameters
        muon_params.append(param)
```

### 3. Momentum Scheduling

Muon uses momentum warmup for stability:

```python
def get_muon_momentum_schedule(step, warmup_steps=300):
    if step >= warmup_steps:
        return 0.95
    frac = step / warmup_steps
    return (1 - frac) * 0.85 + frac * 0.95
```

## Implementation Details

### Muon Optimizer Class

```python
class Muon(Optimizer):
    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        weight_decay=0.0
    ):
        ...

    def step(self):
        for p in params:
            # Standard momentum update
            buf.mul_(momentum).add_(grad)

            # Nesterov momentum (optional)
            update = grad.add(buf, alpha=momentum) if nesterov else buf

            # Orthogonalize 2D+ parameters
            if p.ndim >= 2:
                update = zeropower_via_newtonschulz5(update)
                scale = max(p.size(0), p.size(1)) ** 0.5
                update = update * scale

            # Apply update
            p.add_(update, alpha=-lr)
```

### Dual Optimizer Training

The training loop handles both optimizers:

```python
# Create optimizers
adamw_optimizer, muon_optimizer = configure_optimizers(model, config)
optimizers = [opt for opt in [adamw_optimizer, muon_optimizer] if opt]

# Training step
for opt in optimizers:
    opt.zero_grad()

loss.backward()

for opt in optimizers:
    scaler.unscale_(opt)

grad_norm = clip_grad_norm_(model.parameters(), config.grad_clip)

for opt in optimizers:
    scaler.step(opt)
```

## Comparison with AdamW

| Aspect | AdamW | Muon |
|--------|-------|------|
| Memory | 2x parameters (m, v) | 1x parameters (momentum only) |
| Typical LR | 1e-4 to 6e-4 | 0.01 to 0.05 |
| Best for | Embeddings, biases | Dense weight matrices |
| Update type | Adaptive per-parameter | Orthogonalized |

## References

- [Keller Jordan's Muon](https://github.com/KellerJordan/Muon) - Original implementation
- [Muon Blog Post](https://kellerjordan.github.io/posts/muon/) - Technical explanation
- [PyTorch 2.9+ Muon](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html) - Official PyTorch implementation
- [nanochat](https://github.com/karpathy/nanochat) - Karpathy's implementation

## Example Training Script

Complete example with Muon:

```bash
# Download data
python data.py

# Train with Muon
python train.py \
    --model-size small \
    --use-muon \
    --muon-lr 0.02 \
    --learning-rate 3e-4 \
    --batch-size 32 \
    --seq-len 512 \
    --max-iters 5000 \
    --data-path data/shakespeare/train.txt

# Generate text
python generate.py \
    --checkpoint checkpoints/ckpt_5000.pt \
    --prompt "To be or not to be"
```

## Troubleshooting

### Loss spikes with Muon

Try reducing `--muon-lr`:
```bash
python train.py --use-muon --muon-lr 0.01
```

### Slow convergence

Ensure you're using mixed precision:
```bash
python train.py --use-muon  # bfloat16 by default on supported GPUs
```

### Memory issues

Muon actually uses less memory than AdamW. If you have memory issues, reduce batch size:
```bash
python train.py --use-muon --batch-size 16 --gradient-accumulation-steps 8
```
