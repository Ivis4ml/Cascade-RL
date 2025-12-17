"""
Minimal LLM Implementation in PyTorch
Architecture based on GPT-OSS 20B design patterns:
- RMSNorm (instead of LayerNorm)
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA)
- SwiGLU activation
- No bias in linear layers
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Model configuration following GPT-OSS 20B architecture patterns."""
    # Model dimensions
    vocab_size: int = 32000  # Vocabulary size
    n_layers: int = 12       # Number of transformer blocks
    n_heads: int = 12        # Number of attention heads
    n_kv_heads: int = 4      # Number of KV heads (for GQA)
    d_model: int = 768       # Hidden dimension
    d_ff: int = 2048         # Feed-forward intermediate dimension (typically ~2.67x d_model for SwiGLU)
    max_seq_len: int = 2048  # Maximum sequence length
    dropout: float = 0.0     # Dropout rate
    rope_theta: float = 10000.0  # RoPE base frequency

    # Small model (~125M params)
    @classmethod
    def small(cls):
        return cls(
            n_layers=12,
            n_heads=12,
            n_kv_heads=4,
            d_model=768,
            d_ff=2048,
        )

    # Medium model (~350M params)
    @classmethod
    def medium(cls):
        return cls(
            n_layers=24,
            n_heads=16,
            n_kv_heads=4,
            d_model=1024,
            d_ff=2816,
        )

    # Large model (~760M params)
    @classmethod
    def large(cls):
        return cls(
            n_layers=24,
            n_heads=16,
            n_kv_heads=8,
            d_model=1536,
            d_ff=4096,
        )

    # GPT-OSS 20B style (scaled down for training)
    @classmethod
    def gpt_oss_style(cls):
        """GPT-OSS 20B style but smaller for practical training."""
        return cls(
            n_layers=24,
            n_heads=64,
            n_kv_heads=8,
            d_model=2880,
            d_ff=7680,  # ~2.67x d_model for SwiGLU efficiency
        )


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no learnable bias, just scale)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate RMS
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute Rotary Position Embedding frequencies."""
    # Compute inverse frequencies
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # Compute position indices
    t = torch.arange(max_seq_len, dtype=torch.float32)
    # Outer product: (max_seq_len, dim/2)
    freqs = torch.outer(t, freqs)
    # Convert to complex form for rotation
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # e^(i*theta)
    return freqs_cis


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embeddings to queries and keys."""
    # Reshape q and k to complex numbers: (B, n_heads, seq_len, head_dim/2, 2)
    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))

    # Reshape freqs for broadcasting: (1, 1, seq_len, head_dim/2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)

    # Apply rotation
    q_rotated = torch.view_as_real(q_complex * freqs_cis).flatten(-2)
    k_rotated = torch.view_as_real(k_complex * freqs_cis).flatten(-2)

    return q_rotated.type_as(q), k_rotated.type_as(k)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) as used in GPT-OSS 20B.
    Multiple query heads share the same key/value heads.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.n_rep = config.n_heads // config.n_kv_heads  # How many Q heads per KV head

        # Linear projections (no bias, following GPT-OSS)
        self.wq = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = self.wq(x)  # (B, seq_len, n_heads * head_dim)
        k = self.wk(x)  # (B, seq_len, n_kv_heads * head_dim)
        v = self.wv(x)  # (B, seq_len, n_kv_heads * head_dim)

        # Reshape to (B, n_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rope(q, k, freqs_cis)

        # Handle KV cache for inference
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_kv_cache = (k, v)

        # Expand KV heads to match Q heads (GQA)
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask
        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.wo(output)

        return output, new_kv_cache


class SwiGLU(nn.Module):
    """
    SwiGLU activation as used in GPT-OSS 20B.
    SwiGLU(x) = Swish(xW) * (xV)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = config.d_ff

        # Gate and up projections (no bias)
        self.w_gate = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: swish(gate) * up
        gate = F.silu(self.w_gate(x))  # Swish activation
        up = self.w_up(x)
        x = gate * up
        x = self.w_down(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with pre-normalization."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.ffn = SwiGLU(config)
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm attention with residual
        h, new_kv_cache = self.attention(self.norm1(x), freqs_cis, mask, kv_cache)
        x = x + h

        # Pre-norm FFN with residual
        x = x + self.ffn(self.norm2(x))

        return x, new_kv_cache


class GPT(nn.Module):
    """
    Minimal GPT model following GPT-OSS 20B architecture patterns.

    Key features:
    - RMSNorm (no learnable bias)
    - Rotary Position Embeddings (RoPE)
    - Grouped Query Attention (GQA)
    - SwiGLU activation
    - No bias in linear layers
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings (no position embeddings - using RoPE)
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final normalization and output projection
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Precompute RoPE frequencies
        head_dim = config.d_model // config.n_heads
        self.register_buffer(
            "freqs_cis",
            precompute_rope_freqs(head_dim, config.max_seq_len, config.rope_theta),
            persistent=False
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Special scaled init for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('wo.weight') or pn.endswith('w_down.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_cache: Optional[list] = None,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            targets: Target token IDs for loss computation
            kv_cache: List of KV cache tuples for each layer
            start_pos: Starting position for RoPE (used in inference)

        Returns:
            logits: Output logits, shape (batch_size, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided
            new_kv_cache: Updated KV cache
        """
        batch_size, seq_len = input_ids.shape

        # Get token embeddings
        x = self.tok_emb(input_ids)

        # Get RoPE frequencies for this sequence
        freqs_cis = self.freqs_cis[start_pos:start_pos + seq_len]

        # Create causal mask
        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device)
            mask = torch.triu(mask, diagonal=1)
            # Handle KV cache case
            if kv_cache is not None and kv_cache[0] is not None:
                cache_len = kv_cache[0][0].shape[2]
                mask = torch.cat([
                    torch.zeros((seq_len, cache_len), device=input_ids.device),
                    mask
                ], dim=1)

        # Pass through transformer blocks
        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, layer_kv = layer(x, freqs_cis, mask, layer_cache)
            new_kv_cache.append(layer_kv)

        # Final normalization and output projection
        x = self.norm(x)
        logits = self.lm_head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )

        return logits, loss, new_kv_cache

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Starting token IDs, shape (batch_size, seq_len)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter

        Returns:
            Generated token IDs including the input
        """
        self.eval()
        kv_cache = [None] * len(self.layers)

        for _ in range(max_new_tokens):
            # Get the sequence length for positioning
            seq_len = input_ids.shape[1]

            # Forward pass (use last token if we have cache)
            if kv_cache[0] is not None:
                curr_input = input_ids[:, -1:]
                start_pos = seq_len - 1
            else:
                curr_input = input_ids
                start_pos = 0

            logits, _, kv_cache = self(curr_input, kv_cache=kv_cache, start_pos=start_pos)

            # Get next token logits
            next_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')

            # Sample from distribution
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


if __name__ == "__main__":
    # Test the model
    config = ModelConfig.small()
    model = GPT(config)
    print(f"Model config: {config}")
    print(f"Total parameters: {model.count_parameters():,}")

    # Test forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, loss, _ = model(input_ids, targets)
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
