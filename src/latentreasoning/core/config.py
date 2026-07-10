"""Central registry of external resource identifiers and local paths.

Resource ids for gated models/datasets live behind one module so no unit hardcodes a
fragile URL inline. Values are resolved only when real downloads are explicitly requested;
the mock-first test path never touches them. Override local paths with ``LR_DATA_DIR`` /
``LR_CACHE_DIR``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResourceId:
    """A pointer to an external artifact. ``verified`` flags whether we confirmed it."""

    name: str
    locator: str
    kind: str  # "hf-model" | "hf-dataset" | "url" | "github"
    gated: bool = False
    verified: bool = False
    note: str = ""


# --- Models. Confirm ids at use-time before downloading. ---
MODELS: dict[str, ResourceId] = {
    "medgemma": ResourceId(
        "medgemma", "google/medgemma-1.5-4b-it", "hf-model", gated=True, verified=True,
        note=(
            "MedGemma 1.5 4B-IT (Gemma-3 decoder); gated. The variant CheXthought evaluated. "
            "The older google/medgemma-4b-it also exists — do not confuse them."
        ),
    ),
    "chexone": ResourceId(
        "chexone", "YBZh/CheXOne", "hf-model", gated=False, verified=False,
        note=(
            "Langlotz-lab reasoning-enabled CXR VLM (arXiv:2604.00493), open at "
            "github.com/YBZh/CheXOne. Base = Qwen2.5-VL-3B; verbalized-CoT reasoning mode via "
            "'Please reason step by step and put your final answer within \\boxed{}'. The "
            "verbalized-CoT baseline + causal-audit subject for this project. VERIFY the exact "
            "HF repo id, processor, and chat template on first GPU run."
        ),
    ),
    "chexagent": ResourceId(
        "chexagent", "StanfordAIMI/CheXagent-2-3b", "hf-model", gated=False, verified=False,
        note="Independently-trained size-matched CXR VLM (Phi-based). Verify id/API at use.",
    ),
}

# --- Datasets. ---
DATASETS: dict[str, ResourceId] = {
    "chestxray14": ResourceId(
        "chestxray14", "alkzar90/NIH-Chest-X-ray-dataset", "hf-dataset",
        note="Public HF mirror. ~801 bbox records (incl. Effusion, Pneumothorax) for occlusion.",
    ),
    "rexgroundingct": ResourceId(
        "rexgroundingct", "rajpurkarlab/ReXGroundingCT", "hf-dataset", verified=False,
        note="MICCAI 2026 challenge: text finding -> 3D CT mask (CT-RATE). The CT vehicle.",
    ),
}


def data_root() -> Path:
    """Local directory where real datasets are expected. Override with ``LR_DATA_DIR``."""
    return Path(os.environ.get("LR_DATA_DIR", Path.home() / ".cache" / "latentreasoning" / "data"))


def cache_root() -> Path:
    """Local dir for model weights / artifacts. Override with ``LR_CACHE_DIR``."""
    return Path(os.environ.get("LR_CACHE_DIR", Path.home() / ".cache" / "latentreasoning"))
