"""MockVLM: report-prior fires without evidence; occlusion removes evidence."""

from __future__ import annotations

from latentreasoning.core.mock import MockVLM
from latentreasoning.core.types import Finding
from latentreasoning.data.occlusion import occluded_record


def test_grounded_mock_needs_the_image(effusion_record) -> None:
    m = MockVLM(report_prior=(), honest=True)
    assert m.generate(effusion_record, with_image=True).asserts(Finding.EFFUSION)
    # No image -> a grounded reader cannot assert the finding.
    assert not m.generate(effusion_record, with_image=False).asserts(Finding.EFFUSION)
    # Region occluded -> evidence removed -> cannot assert it.
    occ = occluded_record(effusion_record, Finding.EFFUSION)
    assert not m.generate(occ, with_image=True).asserts(Finding.EFFUSION)


def test_report_prior_fires_without_evidence(effusion_record) -> None:
    m = MockVLM(report_prior=(Finding.EFFUSION,), honest=True)
    # The prior asserts effusion regardless of image / occlusion.
    assert m.generate(effusion_record, with_image=False).asserts(Finding.EFFUSION)
    occ = occluded_record(effusion_record, Finding.EFFUSION)
    assert m.generate(occ, with_image=True).asserts(Finding.EFFUSION)
