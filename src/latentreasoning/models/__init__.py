"""Model adapters. Importing this package registers each adapter by name.

MockVLM is registered in ``latentreasoning.core.mock``; the real adapters register here.
"""

from __future__ import annotations

from latentreasoning.models.chexone import CheXOneAdapter
from latentreasoning.models.medgemma import MedGemmaAdapter

__all__ = ["CheXOneAdapter", "MedGemmaAdapter"]
