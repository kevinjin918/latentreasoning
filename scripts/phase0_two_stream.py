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

import latentreasoning.models  # noqa: F401,E402  (registers medgemma/chexone adapters)
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

    Needs the dataset on disk under ``data_root()/chestxray14`` (run
    ``scripts/download_nih.py --full`` on the box first).
    """
    from latentreasoning.data.chestxray14 import ChestXray14Dataset, bbox_records  # noqa: PLC0415

    return bbox_records(ChestXray14Dataset(), limit=limit)


def run(model_name: str, records: list[tuple[CXRRecord, Finding]], fill: float,
        max_new_tokens: int = 768) -> dict:
    if model_name == "mock":
        model = MockVLM(report_prior=(Finding.EFFUSION,))
    else:
        model = get_model(model_name, max_new_tokens=max_new_tokens)
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
    p.add_argument("--max-new-tokens", type=int, default=768)
    args = p.parse_args(argv)

    if args.dry_run:
        model_name, records = "mock", synthetic_records()
    else:
        model_name, records = args.model, load_records(args.limit)

    log(f"phase0: model={model_name} n={len(records)} max_new_tokens={args.max_new_tokens}")
    result = run(model_name, records, args.fill, max_new_tokens=args.max_new_tokens)
    log("summary:", json.dumps(result["summary"], indent=2))

    RESULTS.mkdir(exist_ok=True)
    tag = "dryrun" if args.dry_run else model_name
    (RESULTS / f"phase0_two_stream_{tag}.json").write_text(json.dumps(result, indent=2))
    log(f"banked -> results/phase0_two_stream_{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
