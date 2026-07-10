"""Core data contracts and the model registry."""

from __future__ import annotations

from latentreasoning.core.mock import MockVLM
from latentreasoning.core.model import (
    ModelAdapter,
    available_models,
    get_model,
    register_model,
)
from latentreasoning.core.types import (
    BBox,
    CXRRecord,
    Finding,
    Label,
    ModelOutput,
)

__all__ = [
    "BBox",
    "CXRRecord",
    "Finding",
    "Label",
    "ModelAdapter",
    "ModelOutput",
    "MockVLM",
    "available_models",
    "get_model",
    "register_model",
]
