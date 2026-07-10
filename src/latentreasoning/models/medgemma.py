"""MedGemma adapter (vendored). The un-reasoning-tuned baseline VLM.

MedGemma 1.5 4B-IT (``google/medgemma-1.5-4b-it``) reasons only via prompted, verbalized
CoT. Construction is offline; weights lazy-load on first use.
"""

from __future__ import annotations

from latentreasoning.core.model import register_model

from ._base import HFVLMAdapter


class MedGemmaAdapter(HFVLMAdapter):
    """Reason-then-probe adapter for MedGemma."""

    name = "medgemma"
    config_key = "medgemma"

    def __init__(self, *, model_id: str | None = None, device: str | None = None,
                 max_new_tokens: int = 768) -> None:
        super().__init__(model_id=model_id, device=device, max_new_tokens=max_new_tokens)


register_model("medgemma", MedGemmaAdapter)
