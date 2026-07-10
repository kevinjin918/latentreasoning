"""Real adapters construct offline (no weights loaded, no torch required to build)."""

from __future__ import annotations

from latentreasoning.core.model import available_models, get_model
from latentreasoning.models.chexone import CHEXONE_REASONING_PROMPT, CheXOneAdapter
from latentreasoning.models.medgemma import MedGemmaAdapter


def test_adapters_register() -> None:
    for name in ("mock", "medgemma", "chexone"):
        assert name in available_models()
    assert get_model("mock").name == "mock"


def test_medgemma_constructs_without_loading() -> None:
    a = MedGemmaAdapter()
    assert a.name == "medgemma"
    assert a.model_id == "google/medgemma-1.5-4b-it"
    assert a._model is None  # lazy: weights not loaded on construction


def test_chexone_uses_verbalized_reasoning_prompt() -> None:
    a = CheXOneAdapter()
    assert a.name == "chexone"
    assert a.default_prompt == CHEXONE_REASONING_PROMPT
    assert "\\boxed{}" in a.default_prompt
    assert a._model is None
    # Instruction mode drops the reasoning elicitation.
    inst = CheXOneAdapter(reasoning=False)
    assert inst.default_prompt != CHEXONE_REASONING_PROMPT
