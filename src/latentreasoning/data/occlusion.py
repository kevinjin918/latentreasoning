"""Occlusion-mask utilities (vendored from tracecxr).

The evidence-removal primitive for the two-stream groundedness measurement: black out the
diagnostic region of a chest X-ray (from public bounding boxes) so we can ask whether the
model's answer changes when the evidence is removed.

All functions are pure: inputs are never mutated. ``occlude`` returns a fresh copy with the
boxed regions overwritten by ``fill``; ``occlude_finding`` and ``occluded_record`` build on
it to occlude a specific finding's annotated region(s).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from latentreasoning.core.types import BBox, CXRRecord, Finding


def occlude(image: np.ndarray, boxes: Iterable[BBox], fill: float = 0.0) -> np.ndarray:
    """Return a copy of ``image`` with every box region overwritten by ``fill``.

    The image is ``HxW`` (grayscale) or ``HxWxC``. Each box is clipped to the image bounds
    before being filled, so out-of-bounds boxes are handled silently. An empty ``boxes``
    iterable returns an unchanged copy.
    """
    out = image.copy()
    height, width = image.shape[0], image.shape[1]
    for box in boxes:
        x1 = int(np.clip(np.floor(box.x), 0, width))
        y1 = int(np.clip(np.floor(box.y), 0, height))
        x2 = int(np.clip(np.ceil(box.x2), 0, width))
        y2 = int(np.clip(np.ceil(box.y2), 0, height))
        if x2 <= x1 or y2 <= y1:
            continue
        out[y1:y2, x1:x2] = fill
    return out


def occlude_finding(record: CXRRecord, finding: Finding, fill: float = 0.0) -> np.ndarray:
    """Return ``record.image`` with ``finding``'s annotated boxes occluded.

    If the finding has no box, the image is returned unchanged (as a copy). Raises
    :class:`ValueError` if the record carries no image.
    """
    if record.image is None:
        raise ValueError(f"record {record.id!r} has no image to occlude")
    boxes = record.bboxes.get(finding, [])
    return occlude(record.image, boxes, fill=fill)


def occluded_record(record: CXRRecord, finding: Finding, fill: float = 0.0) -> CXRRecord:
    """Return a new :class:`CXRRecord` with ``finding``'s region occluded in the image.

    Sets ``metadata["occluded"] = finding.value``. The input record is not mutated.
    """
    image = occlude_finding(record, finding, fill=fill)
    metadata = {**record.metadata, "occluded": finding.value}
    return CXRRecord(
        id=record.id,
        image=image,
        report=record.report,
        findings=dict(record.findings),
        bboxes=dict(record.bboxes),
        source=record.source,
        metadata=metadata,
    )
