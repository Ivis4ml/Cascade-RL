"""
Muon Optimizer - MomentUm Orthogonalized by Newton-schulz

Reference:
- Keller Jordan's original implementation: https://github.com/KellerJordan/Muon
- PyTorch 2.9+ official implementation: torch.optim.Muon
- nanochat implementation: https://github.com/karpathy/nanochat

Key insight:
Muon internally runs standard SGD-momentum and then performs an orthogonalization
post-processing step using Newton-Schulz iteration. This replaces each 2D parameter's
update with the nearest orthogonal matrix.

Use Muon for 2D matrix parameters (linear layers) and AdamW for:
- 1D parameters (biases, norms)
- Embeddings
- Output projection (lm_head)
"""

import torch
from torch.optim import Optimizer
from typing import Optional, Tuple, Callable


def zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    This produces something like U @ V.T (from SVD: G = U @ S @ V.T), which is the
    nearest orthogonal matrix to G in Frobenius norm.

    The iteration converges to an orthogonal matrix when properly normalized.

    Args:
        G: Input matrix to orthogonalize
        steps: Number of Newton-Schulz iterations (5 is usually enough)
        eps: Small constant for numerical stability

    Returns:
        Orthogonalized matrix with same shape as G
    """
    assert G.ndim >= 2

    # Newton-Schulz coefficients (optimized for fast convergence)
    # These are the coefficients for the quintic polynomial approximation
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Normalize to have spectral norm ~1 for convergence
    # Use Frobenius norm as proxy for spectral norm
    X = G / (G.norm(dim=(-2, -1), keepdim=True) + eps)

    # Make it roughly isotropic if rectangular
    if X.size(-2) > X.size(-1):
        X = X.transpose(-2, -1)
        transposed = True
    else:
        transposed = False

    # Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.transpose(-2, -1)

    return X


class Muon(Optimizer):
    """
    Muon optimizer: SGD with momentum, orthogonalized via Newton-Schulz iteration.

    Muon is designed for 2D matrix parameters in neural networks. For other parameters
    (embeddings, biases, layer norms), use AdamW.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.02)
        momentum: Momentum factor (default: 0.95)
        nesterov: Whether to use Nesterov momentum (default: True)
        ns_steps: Number of Newton-Schulz iterations (default: 5)
        weight_decay: Weight decay (L2 penalty) (default: 0.0)

    Example:
        >>> # Separate parameters for Muon vs AdamW
        >>> muon_params = []
        >>> adamw_params = []
        >>> for name, param in model.named_parameters():
        ...     if param.ndim >= 2 and 'embed' not in name and 'head' not in name:
        ...         muon_params.append(param)
        ...     else:
        ...         adamw_params.append(param)
        >>> muon_opt = Muon(muon_params, lr=0.02)
        >>> adamw_opt = AdamW(adamw_params, lr=3e-4)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps value: {ns_steps}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            initial_lr=lr  # Store for LR scheduling
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]

                # Initialize momentum buffer
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad)

                # Compute update
                if nesterov:
                    update = grad.add(buf, alpha=momentum)
                else:
                    update = buf

                # Apply Newton-Schulz orthogonalization for 2D+ parameters
                if p.ndim >= 2:
                    # Reshape to 2D if needed (e.g., for conv layers)
                    original_shape = update.shape
                    if p.ndim > 2:
                        update = update.view(update.size(0), -1)

                    # Orthogonalize
                    update = zeropower_via_newtonschulz5(update, steps=ns_steps)

                    # Reshape back
                    if p.ndim > 2:
                        update = update.view(original_shape)

                    # Scale by sqrt(max(fan_in, fan_out)) for proper magnitude
                    # This helps maintain gradient scale across different layer sizes
                    scale = max(p.size(0), p.size(1)) ** 0.5
                    update = update * scale

                # Apply update
                p.add_(update, alpha=-lr)

        return loss


def setup_muon_adamw_optimizers(
    model: torch.nn.Module,
    muon_lr: float = 0.02,
    adamw_lr: float = 3e-4,
    muon_momentum: float = 0.95,
    weight_decay: float = 0.1,
    adamw_betas: Tuple[float, float] = (0.9, 0.95)
) -> Tuple[Optimizer, Optimizer]:
    """
    Set up Muon and AdamW optimizers for a GPT model.

    Following nanochat convention:
    - Muon: 2D matrix parameters in transformer blocks
    - AdamW: Embeddings, output head, biases, norms

    Args:
        model: The GPT model
        muon_lr: Learning rate for Muon optimizer
        adamw_lr: Learning rate for AdamW optimizer
        muon_momentum: Momentum for Muon optimizer
        weight_decay: Weight decay for both optimizers
        adamw_betas: Beta values for AdamW

    Returns:
        Tuple of (adamw_optimizer, muon_optimizer)
    """
    muon_params = []
    adamw_decay_params = []
    adamw_nodecay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if this is embedding or output head
        is_embedding = 'embed' in name.lower() or 'wte' in name.lower() or 'wpe' in name.lower()
        is_head = 'head' in name.lower() or 'lm_head' in name.lower()
        is_norm = 'norm' in name.lower() or 'ln' in name.lower()
        is_bias = name.endswith('.bias') or 'bias' in name.lower()

        # Route parameters to appropriate optimizer
        if is_embedding or is_head or is_norm or is_bias or param.ndim < 2:
            # AdamW for embeddings, heads, norms, biases, and 1D params
            if param.ndim >= 2 and not is_norm:
                adamw_decay_params.append(param)
            else:
                adamw_nodecay_params.append(param)
        else:
            # Muon for 2D matrix parameters in transformer blocks
            muon_params.append(param)

    # Create AdamW optimizer
    adamw_groups = []
    if adamw_decay_params:
        adamw_groups.append({
            'params': adamw_decay_params,
            'weight_decay': weight_decay,
            'initial_lr': adamw_lr
        })
    if adamw_nodecay_params:
        adamw_groups.append({
            'params': adamw_nodecay_params,
            'weight_decay': 0.0,
            'initial_lr': adamw_lr
        })

    adamw_optimizer = torch.optim.AdamW(
        adamw_groups,
        lr=adamw_lr,
        betas=adamw_betas,
        fused=torch.cuda.is_available()
    ) if adamw_groups else None

    # Create Muon optimizer
    muon_optimizer = Muon(
        muon_params,
        lr=muon_lr,
        momentum=muon_momentum,
        weight_decay=weight_decay
    ) if muon_params else None

    # Print parameter distribution
    n_muon = sum(p.numel() for p in muon_params)
    n_adamw_decay = sum(p.numel() for p in adamw_decay_params)
    n_adamw_nodecay = sum(p.numel() for p in adamw_nodecay_params)

    print(f"Optimizer parameter distribution:")
    print(f"  Muon (matrix params): {n_muon:,} parameters")
    print(f"  AdamW (decay): {n_adamw_decay:,} parameters")
    print(f"  AdamW (no decay): {n_adamw_nodecay:,} parameters")
    print(f"  Total: {n_muon + n_adamw_decay + n_adamw_nodecay:,} parameters")

    return adamw_optimizer, muon_optimizer


def get_muon_momentum_schedule(
    step: int,
    warmup_steps: int = 300,
    start_momentum: float = 0.85,
    end_momentum: float = 0.95
) -> float:
    """
    Momentum schedule for Muon optimizer.

    Linearly increases momentum from start to end over warmup_steps,
    then stays at end_momentum.

    Args:
        step: Current training step
        warmup_steps: Number of steps to warm up momentum
        start_momentum: Initial momentum value
        end_momentum: Final momentum value

    Returns:
        Current momentum value
    """
    if step >= warmup_steps:
        return end_momentum

    frac = step / warmup_steps
    return (1 - frac) * start_momentum + frac * end_momentum
