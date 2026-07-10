"""Phase 1: minimal latent block on frozen CXR features (grounding-vs-depth).

Iterate the shared-weight recurrent block over the frozen encoder's residual states, sweep
the number of iterations T, and read groundedness at each depth (via the two-stream probe on
the spliced-in states). This tests the latent MECHANISM: does more latent depth ground the
answer or amplify the prior?

Usage (VM):
    LR_RUN_GPU=1 python scripts/phase1_latent_block.py --model medgemma --max-iters 8
Dry run (offline, torch on random states, mechanism smoke):
    python scripts/phase1_latent_block.py --dry-run

TODO(Milestone C): wire capture (grounding.capture) -> block -> splice (latent.splice) into
the frozen encoder and run the two-stream probe per depth. The dry run exercises the block +
halting math only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

RESULTS = Path(__file__).resolve().parent.parent / "results"


def log(*a: object) -> None:
    print(*a, flush=True)


def dry_run(d_model: int, max_iters: int) -> dict:
    import torch  # noqa: PLC0415

    from latentreasoning.latent.block import LatentReasoner  # noqa: PLC0415

    torch.manual_seed(0)
    reasoner = LatentReasoner(d_model=d_model, max_iters=max_iters, n_heads=4)
    states, step_probs, exp_steps = reasoner(torch.randn(2, 6, d_model))
    return {
        "d_model": d_model, "max_iters": max_iters, "n_states": len(states),
        "step_probs_shape": list(step_probs.shape),
        "mean_expected_steps": float(exp_steps.mean()),
        "state_norm_by_depth": [float(s.norm()) for s in states],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="medgemma")
    p.add_argument("--dry-run", action="store_true", help="offline torch mechanism smoke")
    p.add_argument("--d-model", type=int, default=16)
    p.add_argument("--max-iters", type=int, default=8)
    args = p.parse_args(argv)

    if not args.dry_run:
        raise NotImplementedError(
            "GPU splice path is Milestone C. Use --dry-run for the block/halting smoke."
        )
    result = dry_run(args.d_model, args.max_iters)
    log("dry-run:", json.dumps(result, indent=2))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "phase1_latent_block_dryrun.json").write_text(json.dumps(result, indent=2))
    log("banked -> results/phase1_latent_block_dryrun.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
