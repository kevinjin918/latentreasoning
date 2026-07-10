"""Occlusion primitive: pure, box blacked out, no mutation."""

from __future__ import annotations

import numpy as np

from latentreasoning.core.types import BBox, Finding
from latentreasoning.data.occlusion import occlude, occluded_record


def test_occlude_blacks_box_and_does_not_mutate() -> None:
    img = np.ones((10, 10), dtype=np.float32)
    out = occlude(img, [BBox(x=2, y=3, w=4, h=5)], fill=0.0)
    assert out.shape == img.shape and out.dtype == img.dtype
    assert (out[3:8, 2:6] == 0.0).all()  # box region zeroed
    assert (img == 1.0).all()  # original untouched
    # Everything outside the box is unchanged.
    mask = np.ones_like(img, dtype=bool)
    mask[3:8, 2:6] = False
    assert (out[mask] == 1.0).all()


def test_occlude_empty_boxes_is_copy() -> None:
    img = np.arange(9, dtype=np.float32).reshape(3, 3)
    out = occlude(img, [])
    assert np.array_equal(out, img) and out is not img


def test_occluded_record_sets_metadata(effusion_record) -> None:
    rec = occluded_record(effusion_record, Finding.EFFUSION)
    assert rec.metadata["occluded"] == Finding.EFFUSION.value
    assert effusion_record.image is not None and rec.image is not None
    # The effusion region (box at 5,20,10,10) is blacked out in the copy only.
    assert (rec.image[20:30, 5:15] == 0.0).all()
    assert (effusion_record.image[20:30, 5:15] == 1.0).all()
