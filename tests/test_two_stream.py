"""Three-stream measurement + groundedness readouts, against MockVLM.

A grounded reader and a confabulating reader produce opposite readouts on the same record,
which is the whole point: groundedness sees what confidence cannot.
"""

from __future__ import annotations

from latentreasoning.core.mock import MockVLM
from latentreasoning.core.types import Finding
from latentreasoning.grounding.readout import (
    confabulation_risk,
    is_grounded,
    prior_reliance,
    region_groundedness,
)
from latentreasoning.grounding.two_stream import stream_predictions

EFF = Finding.EFFUSION


def test_grounded_reader_reads_as_grounded(effusion_record) -> None:
    grounded = MockVLM(report_prior=(), honest=True)
    s = stream_predictions(grounded, effusion_record, EFF)
    assert s["evidence"] > s["occluded"]  # removing the region drops P
    assert region_groundedness(s) > 0.5
    assert prior_reliance(s) < 0.1
    assert is_grounded(s)
    assert not confabulation_risk(s)


def test_confabulator_reads_as_ungrounded(effusion_record) -> None:
    confab = MockVLM(report_prior=(EFF,), honest=True)
    s = stream_predictions(confab, effusion_record, EFF)
    # Asserts effusion equally with or without the evidence: no region dependence.
    assert region_groundedness(s) < 0.1
    assert prior_reliance(s) > 0.5
    assert not is_grounded(s)
    assert confabulation_risk(s)  # the dangerous quadrant confidence cannot see
