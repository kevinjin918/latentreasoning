"""A deterministic mock VLM for offline testing.

:class:`MockVLM` is the backbone of the mock-first test strategy. It is configurable so
failure modes can be exercised without any real weights:

* ``report_prior`` — findings asserted *regardless of the image* (drives no-image and
  occlusion behaviour, so the two-stream measurement can be tested offline).
* ``hallucination_rate`` — probability of spuriously asserting an unrelated finding.
* ``honest`` — when True the model reads ``record.findings`` truthfully when the image is
  present *and not occluded* (a well-grounded baseline to contrast against).

Everything is seeded by the record id, so outputs are reproducible.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from latentreasoning.core.model import ModelAdapter, register_model
from latentreasoning.core.types import CXRRecord, Finding, Label, ModelOutput


def _seed(record_id: str, salt: str) -> float:
    """Deterministic pseudo-random float in [0, 1) from a record id + salt."""
    digest = hashlib.sha256(f"{record_id}:{salt}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


@dataclass
class MockVLM(ModelAdapter):
    """A fake CXR reader with tunable, deterministic behavior."""

    name: str = "mock"
    #: findings asserted even when the image is absent or occluded (the report prior).
    report_prior: tuple[Finding, ...] = (Finding.EFFUSION,)
    #: confidence assigned to a positively asserted finding.
    confidence: float = 0.9
    #: probability of spuriously asserting a non-grounded finding.
    hallucination_rate: float = 0.0
    #: when True, truthfully reports ``record.findings`` when the image is present + not occluded.
    honest: bool = True
    #: findings the model may comment on at all.
    vocabulary: tuple[Finding, ...] = field(
        default_factory=lambda: (Finding.EFFUSION, Finding.PNEUMOTHORAX)
    )

    def generate(
        self,
        record: CXRRecord,
        *,
        with_image: bool = True,
        prompt: str | None = None,
    ) -> ModelOutput:
        preds: dict[Finding, float] = {}

        # The report prior fires whether or not the model can actually see the finding.
        for f in self.report_prior:
            preds[f] = self.confidence

        # Image evidence only counts when present AND that finding's region is not occluded.
        occluded = record.metadata.get("occluded")
        if with_image and record.image is not None and self.honest:
            for f in self.vocabulary:
                if occluded == f.value:
                    continue  # region blacked out -> evidence removed
                if record.label(f) == Label.POSITIVE:
                    preds[f] = self.confidence

        if self.hallucination_rate > 0:
            for f in self.vocabulary:
                if f in preds:
                    continue
                if _seed(record.id, f"hallu:{f.value}") < self.hallucination_rate:
                    preds[f] = self.confidence

        asserted = sorted(f.value for f, p in preds.items() if p >= 0.5)
        abstained = len(asserted) == 0
        text = "No acute findings." if abstained else "Findings: " + ", ".join(asserted) + "."
        return ModelOutput(text=text, finding_predictions=preds, abstained=abstained, raw=None)


def _factory(**kwargs: object) -> MockVLM:
    return MockVLM(**kwargs)  # type: ignore[arg-type]


register_model("mock", _factory)
