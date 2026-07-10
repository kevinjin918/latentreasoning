"""NIH loader parses labels + the quirky bbox header (tiny CSV fixtures, no images)."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from latentreasoning.core.types import Finding, Label  # noqa: E402
from latentreasoning.data.chestxray14 import ChestXray14Dataset  # noqa: E402


def _write_csvs(root) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "Data_Entry_2017.csv").write_text(
        "Image Index,Finding Labels\n"
        "00000001_000.png,Effusion|Infiltration\n"
        "00000002_000.png,No Finding\n"
    )
    # Quirky NIH header with stray brackets, exactly as released.
    (root / "BBox_List_2017.csv").write_text(
        "Image Index,Finding Label,Bbox [x,y,w,h]\n"
        "00000001_000.png,Effusion,10,20,30,40\n"
    )


def test_parses_labels_and_boxes(tmp_path) -> None:
    root = tmp_path / "chestxray14"
    _write_csvs(root)
    recs = {r.id: r for r in ChestXray14Dataset(root)}
    assert recs["00000001_000.png"].label(Finding.EFFUSION) == Label.POSITIVE
    assert recs["00000001_000.png"].label(Finding.LUNG_OPACITY) == Label.POSITIVE  # Infiltration
    box = recs["00000001_000.png"].bboxes[Finding.EFFUSION][0]
    assert (box.x, box.y, box.w, box.h) == (10.0, 20.0, 30.0, 40.0)
    assert recs["00000002_000.png"].bboxes == {}


def test_missing_label_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        ChestXray14Dataset(tmp_path / "nope")
