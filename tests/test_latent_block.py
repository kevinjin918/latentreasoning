"""Latent block + adaptive halting shape/behaviour tests (torch, skipped in CI)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from latentreasoning.latent.block import (  # noqa: E402
    AdaptiveHalting,
    LatentReasoner,
    RecurrentLatentBlock,
    expected_steps,
    halting_distribution,
)


def test_block_preserves_shape_and_iterates() -> None:
    block = RecurrentLatentBlock(d_model=16, n_heads=4)
    h = torch.randn(2, 5, 16)
    assert block(h, 3).shape == (2, 5, 16)
    states = block(h, 3, return_all=True)
    assert len(states) == 3 and all(s.shape == (2, 5, 16) for s in states)


def test_block_rejects_zero_iters() -> None:
    block = RecurrentLatentBlock(d_model=8, n_heads=2)
    with pytest.raises(ValueError):
        block(torch.randn(1, 3, 8), 0)


def test_halting_distribution_is_proper() -> None:
    probs = torch.rand(4, 6).clamp(0.05, 0.95)
    dist = halting_distribution(probs)
    assert dist.shape == (4, 6)
    assert torch.allclose(dist.sum(dim=1), torch.ones(4), atol=1e-5)
    steps = expected_steps(dist)
    assert ((steps >= 1) & (steps <= 6)).all()


def test_deep_start_bias_makes_early_halting_unlikely() -> None:
    halting = AdaptiveHalting(d_model=16, deep_start_bias=-3.0)
    probs = halting([torch.randn(3, 5, 16) for _ in range(4)])
    assert probs.shape == (3, 4)
    assert (probs < 0.2).all()  # deep start: ~sigmoid(-3) ~= 0.047


def test_latent_reasoner_end_to_end() -> None:
    reasoner = LatentReasoner(d_model=16, max_iters=4, n_heads=4)
    states, step_probs, exp_steps = reasoner(torch.randn(2, 6, 16))
    assert len(states) == 4
    assert step_probs.shape == (2, 4)
    assert exp_steps.shape == (2,)
