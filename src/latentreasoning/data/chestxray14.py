"""NIH ChestX-ray14 loader (vendored from tracecxr; dataset-registry coupling dropped).

ChestX-ray14 ships two CSVs under ``data_root()/chestxray14``:

* ``Data_Entry_2017.csv`` — image-level labels (``Finding Labels`` is a pipe-separated list).
* ``BBox_List_2017.csv`` — sparse radiologist boxes (~984 images). The released header names
  the coordinate columns ``Bbox [x``, ``y``, ``w``, ``h]`` (with stray brackets).

Parsing never needs images on disk, so it runs in CI against tiny CSV fixtures. The ~801
bbox records (incl. Effusion / Pneumothorax) are the substrate for the two-stream
groundedness measurement: :func:`bbox_records` yields ``(record, boxed_finding)`` pairs with
eager images so a region can be occluded.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from latentreasoning.core.config import data_root
from latentreasoning.core.types import BBox, CXRRecord, Finding, Label

if TYPE_CHECKING:
    import pandas as pd

#: Explicit NIH ChestX-ray14 finding name -> project ``Finding`` (unknowns skipped).
NIH_TO_FINDING: dict[str, Finding] = {
    "No Finding": Finding.NO_FINDING,
    "Atelectasis": Finding.ATELECTASIS,
    "Cardiomegaly": Finding.CARDIOMEGALY,
    "Effusion": Finding.EFFUSION,
    "Infiltration": Finding.LUNG_OPACITY,
    "Mass": Finding.LUNG_LESION,
    "Nodule": Finding.LUNG_LESION,
    "Pneumonia": Finding.PNEUMONIA,
    "Pneumothorax": Finding.PNEUMOTHORAX,
    "Consolidation": Finding.CONSOLIDATION,
    "Edema": Finding.EDEMA,
}

_DATA_ENTRY_CSV = "Data_Entry_2017.csv"
_BBOX_CSV = "BBox_List_2017.csv"
_IMAGE_COL = "Image Index"
_FINDING_LABELS_COL = "Finding Labels"
_FINDING_LABEL_COL = "Finding Label"


class ChestXray14Dataset:
    """Iterable of :class:`CXRRecord` parsed from NIH ChestX-ray14 CSVs."""

    name = "chestxray14"

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        image_dir: Path | str | None = None,
        eager_images: bool = False,
    ) -> None:
        self.root = Path(root) if root is not None else data_root() / "chestxray14"
        self.image_dir = Path(image_dir) if image_dir is not None else self.root / "images"
        self.eager_images = eager_images
        self._data_entry_path = self.root / _DATA_ENTRY_CSV
        self._bbox_path = self.root / _BBOX_CSV
        if not self._data_entry_path.exists():
            raise FileNotFoundError(
                f"ChestX-ray14 label file not found at {self._data_entry_path}; "
                "expected the NIH `Data_Entry_2017.csv` under the dataset root"
            )

    def __len__(self) -> int:
        import pandas as pd  # noqa: PLC0415

        return len(pd.read_csv(self._data_entry_path))

    def __iter__(self) -> Iterator[CXRRecord]:
        import pandas as pd  # noqa: PLC0415

        entries = pd.read_csv(self._data_entry_path)
        bboxes_by_image = self._load_bboxes()
        for _, row in entries.iterrows():
            image_index = str(row[_IMAGE_COL]).strip()
            findings = _parse_finding_labels(row[_FINDING_LABELS_COL])
            bboxes = bboxes_by_image.get(image_index, {})
            image_path = self.image_dir / image_index
            image = _load_image(image_path) if self.eager_images else None
            yield CXRRecord(
                id=image_index, image=image, report=None, findings=findings,
                bboxes=bboxes, source=self.name,
                metadata={"image_path": str(image_path)},
            )

    def records(self) -> list[CXRRecord]:
        return list(self)

    def _load_bboxes(self) -> dict[str, dict[Finding, list[BBox]]]:
        import pandas as pd  # noqa: PLC0415

        if not self._bbox_path.exists():
            return {}
        frame = pd.read_csv(self._bbox_path)
        x_col, y_col, w_col, h_col = _resolve_bbox_columns(frame)
        result: dict[str, dict[Finding, list[BBox]]] = {}
        for _, row in frame.iterrows():
            finding = NIH_TO_FINDING.get(str(row[_FINDING_LABEL_COL]).strip())
            if finding is None:
                continue
            box = BBox(float(row[x_col]), float(row[y_col]), float(row[w_col]), float(row[h_col]))
            image_index = str(row[_IMAGE_COL]).strip()
            result.setdefault(image_index, {}).setdefault(finding, []).append(box)
        return result


def bbox_records(
    dataset: ChestXray14Dataset,
    *,
    findings: tuple[Finding, ...] = (Finding.EFFUSION, Finding.PNEUMOTHORAX),
    limit: int | None = None,
) -> list[tuple[CXRRecord, Finding]]:
    """``(record, boxed_finding)`` pairs with eager images, for the two-stream measurement.

    Only records that carry a radiologist box for one of ``findings`` are returned (the ~801
    bbox subset). Images are decoded eagerly so the region can be occluded.
    """
    out: list[tuple[CXRRecord, Finding]] = []
    for rec in dataset:
        for f in findings:
            if not rec.bboxes.get(f):
                continue
            img = _load_image(Path(rec.metadata["image_path"]))
            out.append((replace(rec, image=img), f))
            if limit is not None and len(out) >= limit:
                return out
    return out


def _parse_finding_labels(raw: Any) -> dict[Finding, Label]:
    findings: dict[Finding, Label] = {}
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return findings
    for name in str(raw).split("|"):
        finding = NIH_TO_FINDING.get(name.strip())
        if finding is not None:
            findings[finding] = Label.POSITIVE
    return findings


def _resolve_bbox_columns(frame: pd.DataFrame) -> tuple[str, str, str, str]:
    """Locate x/y/w/h columns; tolerant of the quirky ``Bbox [x/y/w/h]`` header."""
    wanted = ("x", "y", "w", "h")
    resolved: dict[str, str] = {}
    for col in frame.columns:
        normalized = str(col).strip().lower().removeprefix("bbox").strip(" [],")
        if normalized in wanted and normalized not in resolved:
            resolved[normalized] = col
    missing = [k for k in wanted if k not in resolved]
    if missing:
        raise ValueError(
            f"could not locate bbox columns {missing} in {list(frame.columns)}; "
            "expected NIH `Bbox [x/y/w/h]` columns"
        )
    return resolved["x"], resolved["y"], resolved["w"], resolved["h"]


def _load_image(path: Path) -> np.ndarray:
    from PIL import Image  # noqa: PLC0415

    if not path.exists():
        raise FileNotFoundError(f"ChestX-ray14 image not found: {path}")
    with Image.open(path) as img:
        return np.asarray(img)
