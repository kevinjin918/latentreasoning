"""Splice a latent block into a frozen VLM via forward hooks.

Two seams, both adapted from tracecxr's ``MedGemmaCLTBackend`` / ``ClampedMedGemma``:

* :func:`capture_module_output` — read a module's output during a forward pass (used to grab
  the frozen encoder's per-layer residual states that feed the block).
* :func:`substitute_module_output` — replace a module's output with the block's result
  (splice the reasoning back in while keeping the encoder frozen).

Both are context managers that install a ``register_forward_hook`` and remove it on exit.
torch is imported lazily so this module is importable offline; the hooks only run on a real
model. Layer paths are model-specific: :func:`mlp_at` gives the Gemma-3 / MedGemma path.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def capture_module_output(module: Any) -> Iterator[list[Any]]:
    """Yield a list that receives ``module``'s output tensor on each forward pass."""
    captured: list[Any] = []

    def hook(_mod: Any, _inp: Any, out: Any) -> None:
        captured.append(out)

    handle = module.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


@contextmanager
def substitute_module_output(
    module: Any, replacement: Any | Callable[[Any], Any]
) -> Iterator[None]:
    """Replace ``module``'s output with ``replacement`` (a tensor or ``fn(orig_out)``).

    A forward hook that *returns* a value overrides the module's output in-place, so the rest
    of the frozen model runs on the spliced-in reasoning state.
    """

    def hook(_mod: Any, _inp: Any, out: Any) -> Any:
        return replacement(out) if callable(replacement) else replacement

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def mlp_at(model: Any, layer: int) -> Any:
    """The MLP submodule at decoder ``layer`` for a Gemma-3 / MedGemma model.

    Path: ``model.model.language_model.layers[layer].mlp``. VERIFY for other backbones
    (Qwen2.5-VL / CheXOne use ``model.model.layers`` — adjust before splicing there).
    """
    return model.model.language_model.layers[layer].mlp
