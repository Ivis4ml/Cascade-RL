"""
Minimal LLM Training Script in PyTorch

Features:
- Mixed precision training (bfloat16/float16)
- Gradient accumulation
- Learning rate scheduling (warmup + cosine decay)
- Gradient clipping
- Checkpointing
- Distributed training support (DDP)
"""

import argparse
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from model import GPT, ModelConfig


@dataclass
class TrainConfig:
    """Training configuration."""
    # Data
    batch_size: int = 32
    seq_len: int = 512
    gradient_accumulation_steps: int = 4

    # Training
    max_iters: int = 10000
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 10

    # Optimizer
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # LR Schedule
    warmup_iters: int = 500
    lr_decay_iters: int = 10000

    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    compile: bool = False  # PyTorch 2.0 compile

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 1000


class TextDataset(Dataset):
    """
    Simple text dataset for language modeling.
    Loads text from a file and creates fixed-length sequences.
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int,
        tokenizer=None,
        split: str = "train",
        train_split: float = 0.9
    ):
        super().__init__()
        self.seq_len = seq_len

        # Load and tokenize data
        if data_path.endswith('.bin'):
            # Pre-tokenized binary data
            self.data = torch.from_numpy(
                __import__('numpy').memmap(data_path, dtype=__import__('numpy').uint16, mode='r')
            ).long()
        else:
            # Text file - use simple character-level tokenization if no tokenizer
            with open(data_path, 'r', encoding='utf-8') as f:
                text = f.read()

            if tokenizer is not None:
                tokens = tokenizer.encode(text)
            else:
                # Character-level encoding
                chars = sorted(list(set(text)))
                self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
                self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
                tokens = [self.char_to_idx[ch] for ch in text]

            self.data = torch.tensor(tokens, dtype=torch.long)

        # Train/val split
        n = len(self.data)
        split_idx = int(n * train_split)
        if split == "train":
            self.data = self.data[:split_idx]
        else:
            self.data = self.data[split_idx:]

        print(f"{split} dataset: {len(self.data):,} tokens")

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for testing/debugging.
    Generates random sequences with simple patterns.
    """

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a sequence with a simple repeating pattern
        torch.manual_seed(idx)
        pattern_len = torch.randint(2, 10, (1,)).item()
        pattern = torch.randint(0, self.vocab_size, (pattern_len,))
        repeats = (self.seq_len + 1) // pattern_len + 1
        sequence = pattern.repeat(repeats)[:self.seq_len + 1]
        return sequence[:-1], sequence[1:]


def get_lr(iter_num: int, config: TrainConfig) -> float:
    """Calculate learning rate with warmup and cosine decay."""
    # Linear warmup
    if iter_num < config.warmup_iters:
        return config.learning_rate * iter_num / config.warmup_iters

    # Cosine decay
    if iter_num > config.lr_decay_iters:
        return config.min_lr

    decay_ratio = (iter_num - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def configure_optimizers(model: GPT, config: TrainConfig) -> torch.optim.Optimizer:
    """
    Configure optimizer with weight decay.
    Apply weight decay only to 2D parameters (weights), not 1D (biases, norms).
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        fused=torch.cuda.is_available()  # Use fused AdamW on CUDA
    )

    return optimizer


@torch.no_grad()
def estimate_loss(model: GPT, train_loader: DataLoader, val_loader: DataLoader,
                  config: TrainConfig, ctx) -> dict:
    """Estimate loss on train and validation sets."""
    model.eval()
    losses = {}

    for split, loader in [("train", train_loader), ("val", val_loader)]:
        total_loss = 0.0
        count = 0
        loader_iter = iter(loader)

        for _ in range(config.eval_iters):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x, y = next(loader_iter)

            x = x.to(config.device)
            y = y.to(config.device)

            with ctx:
                _, loss, _ = model(x, y)

            total_loss += loss.item()
            count += 1

        losses[split] = total_loss / count

    model.train()
    return losses


def save_checkpoint(model: GPT, optimizer: torch.optim.Optimizer,
                    iter_num: int, config: TrainConfig, model_config: ModelConfig):
    """Save training checkpoint."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    checkpoint = {
        "model": model.state_dict() if not isinstance(model, DDP) else model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter_num": iter_num,
        "model_config": model_config.__dict__,
        "train_config": config.__dict__,
    }

    path = os.path.join(config.checkpoint_dir, f"ckpt_{iter_num}.pt")
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path: str, model: GPT, optimizer: Optional[torch.optim.Optimizer] = None):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")

    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint.get("iter_num", 0)


def train(
    model_config: ModelConfig,
    train_config: TrainConfig,
    data_path: Optional[str] = None,
    resume_from: Optional[str] = None
):
    """Main training loop."""
    # Set up distributed training if available
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        ddp_rank = dist.get_rank()
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = dist.get_world_size()
        train_config.device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(train_config.device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        ddp_world_size = 1
        master_process = True
        seed_offset = 0

    # Set random seed
    torch.manual_seed(42 + seed_offset)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42 + seed_offset)

    # Create model
    model = GPT(model_config)
    model = model.to(train_config.device)

    if master_process:
        print(f"Model parameters: {model.count_parameters():,}")
        print(f"Model config: {model_config}")
        print(f"Train config: {train_config}")

    # Compile model if using PyTorch 2.0+
    if train_config.compile and hasattr(torch, 'compile'):
        if master_process:
            print("Compiling model...")
        model = torch.compile(model)

    # Wrap with DDP
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Create datasets
    if data_path is not None:
        train_dataset = TextDataset(data_path, train_config.seq_len, split="train")
        val_dataset = TextDataset(data_path, train_config.seq_len, split="val")
    else:
        if master_process:
            print("Using synthetic dataset for training...")
        train_dataset = SyntheticDataset(model_config.vocab_size, train_config.seq_len, 50000)
        val_dataset = SyntheticDataset(model_config.vocab_size, train_config.seq_len, 5000)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Create optimizer
    optimizer = configure_optimizers(
        model.module if ddp else model,
        train_config
    )

    # Mixed precision context
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[train_config.dtype]
    ctx = nullcontext() if train_config.device == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(train_config.dtype == 'float16'))

    # Resume from checkpoint
    start_iter = 0
    if resume_from is not None:
        start_iter = load_checkpoint(resume_from, model, optimizer)
        if master_process:
            print(f"Resumed from iteration {start_iter}")

    # Training loop
    model.train()
    train_iter = iter(train_loader)
    t0 = time.time()
    local_iter_num = 0

    for iter_num in range(start_iter, train_config.max_iters):
        # Update learning rate
        lr = get_lr(iter_num, train_config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Gradient accumulation loop
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for micro_step in range(train_config.gradient_accumulation_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(train_config.device)
            y = y.to(train_config.device)

            # Forward pass
            with ctx:
                _, loss, _ = model(x, y)
                loss = loss / train_config.gradient_accumulation_steps

            loss_accum += loss.item()

            # Backward pass
            scaler.scale(loss).backward()

        # Gradient clipping
        if train_config.grad_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        else:
            grad_norm = 0.0

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Logging
        if iter_num % train_config.log_interval == 0 and master_process:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            tokens_per_sec = (
                train_config.batch_size *
                train_config.seq_len *
                train_config.gradient_accumulation_steps *
                ddp_world_size
            ) / dt if dt > 0 else 0

            print(
                f"iter {iter_num:6d} | loss {loss_accum:.4f} | "
                f"lr {lr:.2e} | grad_norm {grad_norm:.2f} | "
                f"tok/s {tokens_per_sec:.0f}"
            )

        # Evaluation
        if iter_num > 0 and iter_num % train_config.eval_interval == 0 and master_process:
            losses = estimate_loss(model, train_loader, val_loader, train_config, ctx)
            print(f"eval | train_loss {losses['train']:.4f} | val_loss {losses['val']:.4f}")

        # Checkpointing
        if iter_num > 0 and iter_num % train_config.save_interval == 0 and master_process:
            save_checkpoint(model, optimizer, iter_num, train_config, model_config)

        local_iter_num += 1

    # Save final checkpoint
    if master_process:
        save_checkpoint(model, optimizer, train_config.max_iters, train_config, model_config)
        print("Training complete!")

    if ddp:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train a minimal LLM")

    # Model config
    parser.add_argument("--model-size", type=str, default="small",
                        choices=["small", "medium", "large", "gpt-oss"],
                        help="Model size preset")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)

    # Train config
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-iters", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=500)

    # Data
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to training data (text file or .bin)")

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume-from", type=str, default=None)

    # System
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")

    args = parser.parse_args()

    # Create model config
    if args.model_size == "small":
        model_config = ModelConfig.small()
    elif args.model_size == "medium":
        model_config = ModelConfig.medium()
    elif args.model_size == "large":
        model_config = ModelConfig.large()
    elif args.model_size == "gpt-oss":
        model_config = ModelConfig.gpt_oss_style()

    # Override with command line args
    if args.vocab_size:
        model_config.vocab_size = args.vocab_size
    if args.n_layers:
        model_config.n_layers = args.n_layers
    if args.n_heads:
        model_config.n_heads = args.n_heads
    if args.d_model:
        model_config.d_model = args.d_model
    model_config.max_seq_len = args.seq_len

    # Create train config
    train_config = TrainConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_iters=args.max_iters,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        checkpoint_dir=args.checkpoint_dir,
        compile=args.compile,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
    )

    train(model_config, train_config, args.data_path, args.resume_from)


if __name__ == "__main__":
    main()
