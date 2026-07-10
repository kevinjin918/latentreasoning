"""Causal grounding audit of CheXOne's verbalized reasoning.

CheXOne's reasoning traces were LLM-generated from the reference report and prompted to look
image-derived (its Fig 9). This audit asks, causally: does its reasoning-mode answer actually
depend on the image? We run the three-stream measurement for CheXOne in reasoning mode vs
instruction mode and compare region_groundedness / prior_reliance. If reasoning mode is no
more grounded (or less) than instruction mode, its reasoning is report-anchored, not
image-grounded.

Usage (VM):
    LR_RUN_GPU=1 python scripts/audit_chexone.py
Dry run (offline, MockVLM stand-ins):
    python scripts/audit_chexone.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phase0_two_stream import RESULTS, log, run, synthetic_records  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    if args.dry_run:
        # Two mock stand-ins: a "reasoning" reader that leans on the prior vs a grounded one.
        from latentreasoning.core.mock import MockVLM  # noqa: PLC0415
        from latentreasoning.core.model import register_model  # noqa: PLC0415
        from latentreasoning.core.types import Finding  # noqa: PLC0415

        register_model("chexone_reasoning", lambda **k: MockVLM(report_prior=(Finding.EFFUSION,)))
        register_model("chexone_instruction", lambda **k: MockVLM(report_prior=()))
        recs = synthetic_records()
        reasoning = run("chexone_reasoning", recs, fill=0.0)["summary"]
        instruction = run("chexone_instruction", recs, fill=0.0)["summary"]
    else:
        from latentreasoning.core.model import get_model  # noqa: F401,PLC0415

        raise NotImplementedError(
            "GPU path: instantiate chexone in reasoning vs instruction mode and load NIH "
            "records (Milestone D). Use --dry-run for the offline smoke."
        )

    verdict = {
        "reasoning": reasoning, "instruction": instruction,
        "grounding_delta": reasoning["mean_region_groundedness"]
        - instruction["mean_region_groundedness"],
        "note": "delta <= 0 means reasoning mode is no more image-grounded than direct answering",
    }
    log("audit:", json.dumps(verdict, indent=2))
    RESULTS.mkdir(exist_ok=True)
    tag = "dryrun" if args.dry_run else "chexone"
    (RESULTS / f"audit_chexone_{tag}.json").write_text(json.dumps(verdict, indent=2))
    log(f"banked -> results/audit_chexone_{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
