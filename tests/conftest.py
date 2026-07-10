"""Shared pytest configuration.

Tests tagged ``requires_data`` (need a real dataset) or ``requires_gpu`` (need a GPU / real
weights) are skipped by default so the mock-first suite runs anywhere. Opt in with
``LR_RUN_DATA=1`` / ``LR_RUN_GPU=1``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from latentreasoning.core.types import BBox, CXRRecord, Finding, Label


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_data = os.environ.get("LR_RUN_DATA") == "1"
    run_gpu = os.environ.get("LR_RUN_GPU") == "1"
    skip_data = pytest.mark.skip(reason="needs a real dataset; set LR_RUN_DATA=1")
    skip_gpu = pytest.mark.skip(reason="needs a GPU / real weights; set LR_RUN_GPU=1")
    for item in items:
        if "requires_data" in item.keywords and not run_data:
            item.add_marker(skip_data)
        if "requires_gpu" in item.keywords and not run_gpu:
            item.add_marker(skip_gpu)


@pytest.fixture
def effusion_record() -> CXRRecord:
    """A tiny synthetic record: effusion-positive with an annotated box (for occlusion)."""
    img = np.zeros((32, 32), dtype=np.float32)
    img[20:30, 5:15] = 1.0  # a bright "effusion" region where the box is
    return CXRRecord(
        id="synthetic-eff-1",
        image=img,
        report="Small pleural effusion at the left base.",
        findings={Finding.EFFUSION: Label.POSITIVE, Finding.PNEUMOTHORAX: Label.NEGATIVE},
        bboxes={Finding.EFFUSION: [BBox(x=5, y=20, w=10, h=10)]},
        source="synthetic",
    )
