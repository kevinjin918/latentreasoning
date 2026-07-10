"""Two-stream (really three-stream) groundedness measurement.

Groundedness is a *comparison*: the answer with the evidence versus the answer without it.
So we run the model on three synchronized inputs and read P(finding) from each:

* ``evidence``  — the real image (evidence + prior).
* ``occluded``  — the finding's region blacked out (prior, minus that region's evidence).
* ``no_image``  — the image withheld entirely (pure prior).

The gap between ``evidence`` and ``occluded`` is how much the answer depends on the region's
image evidence; ``no_image`` is the pure-prior reference. This works over the
:class:`~latentreasoning.core.model.ModelAdapter` protocol, so it runs against ``MockVLM``
offline and against MedGemma / CheXOne on a GPU unchanged.
"""

from __future__ import annotations

from latentreasoning.core.model import ModelAdapter
from latentreasoning.core.types import CXRRecord, Finding
from latentreasoning.data.occlusion import occluded_record

#: Canonical stream keys, ordered most-to-least evidence.
STREAMS: tuple[str, ...] = ("evidence", "occluded", "no_image")


def _finding_p(model: ModelAdapter, record: CXRRecord, finding: Finding, *,
               with_image: bool) -> float:
    out = model.generate(record, with_image=with_image)
    return float(out.finding_predictions.get(finding, 0.0))


def stream_predictions(
    model: ModelAdapter,
    record: CXRRecord,
    finding: Finding,
    *,
    fill: float = 0.0,
) -> dict[str, float]:
    """P(``finding``) under each of the three streams.

    Requires ``record.bboxes[finding]`` to be non-empty for a meaningful ``occluded`` stream
    (otherwise occlusion is a no-op and ``occluded`` equals ``evidence``).
    """
    evidence = _finding_p(model, record, finding, with_image=True)
    occluded = _finding_p(
        model, occluded_record(record, finding, fill=fill), finding, with_image=True
    )
    no_image = _finding_p(model, record, finding, with_image=False)
    return {"evidence": evidence, "occluded": occluded, "no_image": no_image}
