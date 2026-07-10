"""Groundedness signals derived from the three-stream predictions.

These are pure functions over the ``{"evidence", "occluded", "no_image"}`` dict returned by
:func:`latentreasoning.grounding.two_stream.stream_predictions`. They turn the raw
probabilities into the two quantities the halting rule cares about:

* ``region_groundedness`` — how much removing the finding's region drops P(finding). High =
  the answer depends on that region's image evidence (grounded). Near zero = the region did
  not matter, the answer came from elsewhere (prior).
* ``prior_reliance`` — P(finding) with no image at all. High = the model asserts the finding
  from the prior alone (the confident-confabulation danger signal).
"""

from __future__ import annotations

from collections.abc import Mapping


def region_groundedness(streams: Mapping[str, float]) -> float:
    """P(evidence) - P(occluded): the drop from removing the region's evidence."""
    return float(streams["evidence"] - streams["occluded"])


def prior_reliance(streams: Mapping[str, float]) -> float:
    """P(no_image): how much the assertion survives with no image at all."""
    return float(streams["no_image"])


def is_grounded(streams: Mapping[str, float], *, min_drop: float = 0.2) -> bool:
    """Whether the answer depends on the region's evidence by at least ``min_drop``."""
    return region_groundedness(streams) >= min_drop


def confabulation_risk(streams: Mapping[str, float], *, min_prior: float = 0.5) -> bool:
    """Ungrounded-but-asserted: high no-image P and little region dependence.

    The dangerous quadrant: the model asserts the finding even without the image and does not
    rely on the region's evidence. This is what confidence-based halting cannot see.
    """
    return prior_reliance(streams) >= min_prior and not is_grounded(streams)
