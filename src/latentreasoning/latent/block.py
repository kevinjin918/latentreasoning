"""The latent reasoning mechanism: a shared-weight recurrent block + adaptive halting.

Reasoning happens as repeated application of ONE block to a latent state (weight-shared
across iterations), so effective depth T is decoupled from parameter count. Adaptive halting
picks T per case: it emits a per-step halt probability and a PonderNet-style stopping
distribution, initialised to a **deep start** (bias -3, halt prob ~0.05) so training does not
collapse to shallow halting.

This is torch-only and imported lazily by callers / tests (``pytest.importorskip("torch")``);
the core package never imports it, so importing ``latentreasoning`` stays offline.
"""

from __future__ import annotations

import torch
from torch import nn


class RecurrentLatentBlock(nn.Module):
    """A single pre-norm transformer block applied T times to a latent state.

    ``forward(h, n_iters)`` iterates the same weights ``n_iters`` times over ``h`` of shape
    ``(batch, seq, d_model)``. With ``return_all=True`` it returns the list of intermediate
    states (one per iteration), which is what the groundedness-vs-depth readout consumes.
    """

    def __init__(self, d_model: int, n_heads: int = 8, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model)
        )

    def step(self, h: torch.Tensor) -> torch.Tensor:
        """One reasoning iteration (residual self-attention then residual MLP)."""
        x = self.norm1(h)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        h = h + attn_out
        h = h + self.mlp(self.norm2(h))
        return h

    def forward(
        self, h: torch.Tensor, n_iters: int, *, return_all: bool = False
    ) -> torch.Tensor | list[torch.Tensor]:
        if n_iters < 1:
            raise ValueError(f"n_iters must be >= 1, got {n_iters}")
        states: list[torch.Tensor] = []
        for _ in range(n_iters):
            h = self.step(h)
            if return_all:
                states.append(h)
        return states if return_all else h


def halting_distribution(step_probs: torch.Tensor) -> torch.Tensor:
    """PonderNet stopping distribution from per-step halt probabilities.

    ``step_probs`` is ``(batch, T)`` with each entry in (0, 1). Returns ``(batch, T)`` where
    entry ``t`` is the probability of halting exactly at step ``t``; the final step absorbs
    the remaining mass so each row sums to 1.
    """
    if step_probs.dim() != 2:
        raise ValueError(f"expected (batch, T), got shape {tuple(step_probs.shape)}")
    b, t = step_probs.shape
    cont = torch.cumprod(1.0 - step_probs, dim=1)  # prod_{j<=t} (1 - p_j)
    prior_cont = torch.cat([torch.ones(b, 1, device=step_probs.device), cont[:, :-1]], dim=1)
    dist = step_probs * prior_cont
    # Absorb leftover mass into the last step so the distribution is proper.
    dist[:, -1] = prior_cont[:, -1]
    return dist


def expected_steps(dist: torch.Tensor) -> torch.Tensor:
    """Expected number of reasoning steps under a halting distribution ``(batch, T)``."""
    steps = torch.arange(1, dist.shape[1] + 1, device=dist.device, dtype=dist.dtype)
    return (dist * steps).sum(dim=1)


class AdaptiveHalting(nn.Module):
    """Per-step halt-probability head, deep-start initialised.

    Pools each latent state over the sequence and emits a halt probability. ``deep_start_bias``
    (default -3) makes early halting unlikely so the model learns *when* it can stop rather
    than collapsing to ~2 steps at initialisation.
    """

    def __init__(self, d_model: int, deep_start_bias: float = -3.0) -> None:
        super().__init__()
        self.halt = nn.Linear(d_model, 1)
        nn.init.zeros_(self.halt.weight)
        nn.init.constant_(self.halt.bias, deep_start_bias)

    def step_prob(self, h: torch.Tensor) -> torch.Tensor:
        """Halt probability for one state ``(batch, seq, d_model)`` -> ``(batch,)``."""
        pooled = h.mean(dim=1)
        return torch.sigmoid(self.halt(pooled)).squeeze(-1)

    def forward(self, states: list[torch.Tensor]) -> torch.Tensor:
        """Per-step halt probabilities ``(batch, T)`` from a list of ``T`` states."""
        return torch.stack([self.step_prob(h) for h in states], dim=1)


class LatentReasoner(nn.Module):
    """Convenience wrapper: iterate the block up to ``max_iters`` and score halting.

    Returns the list of per-iteration states, the per-step halt probabilities, and the
    expected number of steps (the ponder-cost signal). Fixed-depth training uses only the
    states; adaptive-depth uses the halting outputs.
    """

    def __init__(self, d_model: int, *, max_iters: int = 8, n_heads: int = 8,
                 deep_start_bias: float = -3.0) -> None:
        super().__init__()
        self.max_iters = max_iters
        self.block = RecurrentLatentBlock(d_model, n_heads=n_heads)
        self.halting = AdaptiveHalting(d_model, deep_start_bias=deep_start_bias)

    def forward(
        self, h: torch.Tensor, n_iters: int | None = None
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        n = self.max_iters if n_iters is None else n_iters
        states = self.block(h, n, return_all=True)
        step_probs = self.halting(states)
        dist = halting_distribution(step_probs)
        return states, step_probs, expected_steps(dist)
