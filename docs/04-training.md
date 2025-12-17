# Step 4: Training

This guide covers how to train your language model.

## Inputs

- Prepared binary data file (`data/*.bin`)
- Vocabulary file (`data/*_vocab.json`) for char tokenizer

## Outputs

After this step, you will have:
- Model checkpoints (`checkpoints/ckpt_*.pt`)
- Training logs (printed to console)

---

## Quick Start

### Minimal Training Command

```bash
python train.py \
    --data-path data/shakespeare.bin \
    --model-size small \
    --max-iters 2000
```

### Expected Output

```
Model parameters: 124,668,672
Model config: ModelConfig(vocab_size=32000, n_layers=12, ...)
Train config: TrainConfig(batch_size=32, seq_len=512, ...)
train dataset: 1,003,854 tokens
val dataset: 111,540 tokens
iter      0 | loss 10.4523 | lr 0.00e+00 | grad_norm 72.30 | tok/s 45123
iter     10 | loss 9.8234 | lr 6.00e-06 | grad_norm 45.21 | tok/s 52341
iter     20 | loss 8.5123 | lr 1.20e-05 | grad_norm 38.92 | tok/s 51234
...
iter   2000 | loss 1.8234 | lr 3.00e-04 | grad_norm 0.82 | tok/s 48923
Saved checkpoint to checkpoints/ckpt_2000.pt
Training complete!
```

---

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                      Training Loop                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐      │
│  │ Data    │───►│ Forward │───►│  Loss   │───►│Backward │      │
│  │ Loader  │    │  Pass   │    │         │    │  Pass   │      │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘      │
│       │                                            │            │
│       │         ┌─────────────────────────────────┘            │
│       │         ▼                                              │
│       │    ┌─────────┐    ┌─────────┐    ┌─────────┐          │
│       │    │Gradient │───►│Optimizer│───►│ Update  │          │
│       │    │  Clip   │    │  Step   │    │ Weights │          │
│       │    └─────────┘    └─────────┘    └─────────┘          │
│       │                                        │               │
│       └────────────────────────────────────────┘               │
│                         (repeat)                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Command Line Arguments

### Model Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-size` | `small` | Preset: `small`, `medium`, `large`, `gpt-oss` |
| `--vocab-size` | 32000 | Vocabulary size |
| `--n-layers` | (preset) | Number of transformer layers |
| `--n-heads` | (preset) | Number of attention heads |
| `--d-model` | (preset) | Hidden dimension |

### Training Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | 32 | Batch size per step |
| `--seq-len` | 512 | Sequence length |
| `--max-iters` | 10000 | Total training iterations |
| `--learning-rate` | 3e-4 | Peak learning rate |
| `--gradient-accumulation-steps` | 4 | Gradient accumulation |

### Logging & Checkpoints

| Argument | Default | Description |
|----------|---------|-------------|
| `--log-interval` | 10 | Log every N iterations |
| `--eval-interval` | 500 | Evaluate every N iterations |
| `--checkpoint-dir` | `checkpoints` | Checkpoint directory |
| `--resume-from` | None | Resume from checkpoint |

### System

| Argument | Default | Description |
|----------|---------|-------------|
| `--compile` | False | Use torch.compile (PyTorch 2.0+) |
| `--data-path` | None | Path to training data |

---

## Training Configurations

### Quick Test (1-2 minutes)

```bash
python train.py \
    --data-path data/shakespeare.bin \
    --model-size small \
    --max-iters 100 \
    --batch-size 16 \
    --seq-len 128
```

### Shakespeare Demo (5-10 minutes)

```bash
python train.py \
    --data-path data/shakespeare.bin \
    --model-size small \
    --max-iters 2000 \
    --batch-size 32 \
    --seq-len 256 \
    --learning-rate 1e-3
```

### Serious Training (hours)

```bash
python train.py \
    --data-path data/shakespeare.bin \
    --model-size medium \
    --max-iters 10000 \
    --batch-size 16 \
    --seq-len 512 \
    --gradient-accumulation-steps 4 \
    --compile
```

### Large Scale Training (days)

```bash
python train.py \
    --data-path data/large_corpus.bin \
    --model-size large \
    --max-iters 100000 \
    --batch-size 8 \
    --seq-len 1024 \
    --gradient-accumulation-steps 8 \
    --compile
```

---

## Understanding Training Output

### Log Line Format

```
iter    100 | loss 5.2341 | lr 6.00e-05 | grad_norm 12.34 | tok/s 48923
```

| Field | Meaning |
|-------|---------|
| `iter` | Current iteration number |
| `loss` | Cross-entropy loss (lower is better) |
| `lr` | Current learning rate |
| `grad_norm` | Gradient norm (after clipping) |
| `tok/s` | Tokens processed per second |

### Evaluation Output

```
eval | train_loss 2.1234 | val_loss 2.3456
```

| Field | Meaning |
|-------|---------|
| `train_loss` | Loss on training data |
| `val_loss` | Loss on validation data (held-out) |

### Good Training Signs

1. **Loss decreasing**: Should drop from ~10 to ~2-3
2. **Stable grad_norm**: Should stabilize around 0.5-2.0
3. **train_loss ≈ val_loss**: No overfitting

### Bad Training Signs

1. **Loss not decreasing**: Learning rate may be wrong
2. **Loss exploding (NaN)**: Learning rate too high
3. **val_loss >> train_loss**: Overfitting

---

## Learning Rate Schedule

The training uses warmup + cosine decay:

```
Learning Rate
    │
3e-4├────────────╮
    │           ╱ ╲
    │          ╱   ╲
    │         ╱     ╲
    │        ╱       ╲
    │       ╱         ╲
3e-5├──────╱           ╲________
    │
    └────────────────────────────► Iteration
    0   500            10000
      warmup   decay
```

---

## Gradient Accumulation

Simulates larger batch sizes with limited memory:

```
Effective batch = batch_size × gradient_accumulation_steps

Example:
--batch-size 8 --gradient-accumulation-steps 4
→ Effective batch size = 32
```

### When to Use

| Your GPU | Desired Batch | Command |
|----------|---------------|---------|
| 8 GB | 32 | `--batch-size 8 --gradient-accumulation-steps 4` |
| 16 GB | 64 | `--batch-size 16 --gradient-accumulation-steps 4` |
| 24 GB | 64 | `--batch-size 32 --gradient-accumulation-steps 2` |

---

## Checkpointing

### Automatic Saves

Checkpoints are saved:
- Every `save_interval` iterations (default: 1000)
- At the end of training

### Checkpoint Contents

```python
checkpoint = {
    "model": model.state_dict(),      # Model weights
    "optimizer": optimizer.state_dict(),  # Optimizer state
    "iter_num": current_iteration,    # Progress
    "model_config": {...},            # Model configuration
    "train_config": {...},            # Training configuration
}
```

### Resume Training

```bash
python train.py \
    --resume-from checkpoints/ckpt_5000.pt \
    --max-iters 10000
```

---

## Distributed Training

### Single Node, Multiple GPUs

```bash
torchrun --nproc_per_node=4 train.py \
    --data-path data/shakespeare.bin \
    --model-size medium \
    --batch-size 8
```

### Multiple Nodes

**On node 0 (master):**
```bash
torchrun --nnodes=2 --nproc_per_node=8 \
    --node_rank=0 --master_addr=192.168.1.1 --master_port=29500 \
    train.py --model-size large
```

**On node 1:**
```bash
torchrun --nnodes=2 --nproc_per_node=8 \
    --node_rank=1 --master_addr=192.168.1.1 --master_port=29500 \
    train.py --model-size large
```

---

## torch.compile (PyTorch 2.0+)

### Enable Compilation

```bash
python train.py --compile
```

### Benefits

- 20-30% speedup on supported GPUs
- No code changes needed
- Automatic kernel fusion

### Requirements

- PyTorch 2.0+
- NVIDIA GPU (best support)
- First iteration is slow (compilation)

---

## Memory Optimization

### Reduce Memory Usage

1. **Smaller batch size**: `--batch-size 4`
2. **Shorter sequences**: `--seq-len 256`
3. **Gradient accumulation**: `--gradient-accumulation-steps 8`
4. **Smaller model**: `--model-size small`

### Estimate Memory Needs

| Model | batch=8, seq=512 | batch=16, seq=512 | batch=32, seq=512 |
|-------|------------------|-------------------|-------------------|
| small | ~3 GB | ~5 GB | ~8 GB |
| medium | ~6 GB | ~10 GB | ~16 GB |
| large | ~12 GB | ~20 GB | ~32 GB |

---

## Using Synthetic Data (Testing)

For quick testing without real data:

```bash
python train.py \
    --model-size small \
    --max-iters 100
```

When `--data-path` is not provided, synthetic data is used automatically.

---

## Monitoring Training

### Watch GPU Usage

```bash
# In another terminal
watch -n 1 nvidia-smi
```

### Expected GPU Utilization

- **Good**: 90-100% GPU usage
- **OK**: 70-90% (may be I/O bound)
- **Bad**: < 70% (bottleneck somewhere)

---

## Troubleshooting

### Issue: CUDA out of memory

**Solution:** Reduce memory usage:
```bash
python train.py --batch-size 4 --gradient-accumulation-steps 8
```

### Issue: Loss is NaN

**Causes:**
1. Learning rate too high
2. Gradient explosion

**Solution:**
```bash
python train.py --learning-rate 1e-4 --grad-clip 0.5
```

### Issue: Loss not decreasing

**Causes:**
1. Learning rate too low
2. Data issues

**Solution:**
```bash
python train.py --learning-rate 1e-3
```

### Issue: Training too slow

**Solutions:**
1. Use torch.compile: `--compile`
2. Increase batch size
3. Use mixed precision (default)

### Issue: val_loss much higher than train_loss

**Cause:** Overfitting

**Solutions:**
1. More training data
2. Add dropout
3. Early stopping

---

## Example: Full Training Run

```bash
# 1. Prepare data
python data.py --dataset shakespeare

# 2. Train with good defaults
python train.py \
    --data-path data/shakespeare.bin \
    --model-size small \
    --max-iters 5000 \
    --batch-size 32 \
    --seq-len 256 \
    --learning-rate 1e-3 \
    --log-interval 100 \
    --eval-interval 500 \
    --compile

# 3. Monitor output
# iter    100 | loss 6.2341 | lr 2.00e-04 | ...
# iter    200 | loss 4.5678 | lr 4.00e-04 | ...
# ...
# eval | train_loss 1.8234 | val_loss 1.9456
# ...
# Saved checkpoint to checkpoints/ckpt_5000.pt
# Training complete!
```

---

## Summary

| Goal | Command |
|------|---------|
| Quick test | `python train.py --max-iters 100` |
| Shakespeare demo | `python train.py --data-path data/shakespeare.bin --max-iters 2000` |
| Resume training | `python train.py --resume-from checkpoints/ckpt_2000.pt` |
| Multi-GPU | `torchrun --nproc_per_node=4 train.py ...` |

---

## Next Step

Once training is complete, learn how to [Generate Text](05-generation.md) →
