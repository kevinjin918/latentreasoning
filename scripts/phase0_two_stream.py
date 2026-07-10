"""Phase 0: three-stream groundedness measurement + verbalized-CoT sanity.

For each record with a boxed finding, run three synchronized streams (evidence / occluded /
no_image), read P(finding) from each, and record region_groundedness and prior_reliance.
Aggregate and bank to results/phase0_two_stream_<model>.{json,md}. This tests the PHENOMENON
(does removing the image or its region drop the assertion?) and validates the two-stream
measurement using verbalized CoT. It does NOT test the latent mechanism (that is phase1).

Usage (VM):
    LR_RUN_GPU=1 python scripts/phase0_two_stream.py --model medgemma
    LR_RUN_GPU=1 python scripts/phase0_two_stream.py --model chexone
Dry run (offline, MockVLM, full pipeline smoke):
    python scripts/phase0_two_stream.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from latentreasoning.core.mock import MockVLM  # noqa: E402
from latentreasoning.core.model import get_model  # noqa: E402
from latentreasoning.core.types import BBox, CXRRecord, Finding, Label  # noqa: E402
from latentreasoning.grounding.readout import (  # noqa: E402
    confabulation_risk,
    is_grounded,
    prior_reliance,
    region_groundedness,
)
from latentreasoning.grounding.two_stream import stream_predictions  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def log(*a: object) -> None:
    print(*a, flush=True)


def synthetic_records() -> list[tuple[CXRRecord, Finding]]:
    """Two tiny records (effusion-present, boxed) for the offline dry run."""
    out: list[tuple[CXRRecord, Finding]] = []
    for i in range(2):
        img = np.zeros((32, 32), dtype=np.float32)
        img[20:30, 5:15] = 1.0
        rec = CXRRecord(
            id=f"synthetic-{i}", image=img, report="Effusion at the left base.",
            findings={Finding.EFFUSION: Label.POSITIVE},
            bboxes={Finding.EFFUSION: [BBox(5, 20, 10, 10)]}, source="synthetic",
        )
        out.append((rec, Finding.EFFUSION))
    return out


def load_records(limit: int) -> list[tuple[CXRRecord, Finding]]:
    """Real records with radiologist boxes (NIH ChestX-ray14 bbox subset).

    TODO(Milestone B): vendor scripts/download_nih.py + a chestxray14 loader that yields
    (record, boxed_finding) pairs for the ~801 bbox records. Until then the GPU run needs
    that loader; use --dry-run for the offline pipeline smoke.
    """
    raise NotImplementedError(
        "NIH bbox loader not vendored yet (Milestone B). Run with --dry-run for the smoke."
    )


def run(model_name: str, records: list[tuple[CXRRecord, Finding]], fill: float) -> dict:
    model = MockVLM(report_prior=(Finding.EFFUSION,)) if model_name == "mock" else get_model(
        model_name
    )
    rows = []
    for rec, finding in records:
        s = stream_predictions(model, rec, finding, fill=fill)
        rows.append({
            "id": rec.id, "finding": finding.value, **s,
            "region_groundedness": region_groundedness(s),
            "prior_reliance": prior_reliance(s),
            "grounded": is_grounded(s),
            "confabulation_risk": confabulation_risk(s),
        })
    n = len(rows)
    summary = {
        "model": model_name, "n": n,
        "mean_region_groundedness": sum(r["region_groundedness"] for r in rows) / n,
        "mean_prior_reliance": sum(r["prior_reliance"] for r in rows) / n,
        "frac_grounded": sum(r["grounded"] for r in rows) / n,
        "frac_confabulation_risk": sum(r["confabulation_risk"] for r in rows) / n,
    }
    return {"summary": summary, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="medgemma")
    p.add_argument("--dry-run", action="store_true", help="offline MockVLM smoke")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--fill", type=float, default=0.0)
    args = p.parse_args(argv)

    if args.dry_run:
        model_name, records = "mock", synthetic_records()
    else:
        model_name, records = args.model, load_records(args.limit)

    log(f"phase0: model={model_name} n={len(records)}")
    result = run(model_name, records, args.fill)
    log("summary:", json.dumps(result["summary"], indent=2))

    RESULTS.mkdir(exist_ok=True)
    tag = "dryrun" if args.dry_run else model_name
    (RESULTS / f"phase0_two_stream_{tag}.json").write_text(json.dumps(result, indent=2))
    log(f"banked -> results/phase0_two_stream_{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
