"""Shared scaffolding for real HuggingFace VLM adapters (vendored from tracecxr).

Finding extraction is **constrained yes/no probing**, not text parsing: the model reasons
freely (a "step-by-step diagnostic reasoning" prompt), then for each finding we read the
next-token logits of a closed "is there <finding>? yes/no" question and report
``P(yes) = softmax([logit_yes, logit_no])``. The free reasoning is returned in
``ModelOutput.text`` for the mechanistic stages.

Subclasses set a name, a config key, a default generation budget, and optionally a
``default_prompt`` (CheXOne's verbalized-CoT reasoning prompt differs from MedGemma's).
torch / transformers are imported lazily so importing this package is offline.
"""

from __future__ import annotations

import os
from typing import Any

from latentreasoning.core.config import MODELS
from latentreasoning.core.types import CXRRecord, Finding, ModelOutput

#: Findings probed per record (the CheXpert pathologies with occlusion/gaze ground truth).
PROBE_FINDINGS: tuple[Finding, ...] = (
    Finding.EFFUSION,
    Finding.PNEUMOTHORAX,
    Finding.CARDIOMEGALY,
    Finding.ATELECTASIS,
    Finding.CONSOLIDATION,
    Finding.EDEMA,
    Finding.PNEUMONIA,
    Finding.LUNG_OPACITY,
    Finding.LUNG_LESION,
    Finding.FRACTURE,
)

#: Default reasoning elicitation. Subclasses may override via ``default_prompt``.
REASONING_PROMPT: str = (
    "Interpret this chest X-ray and provide step-by-step diagnostic reasoning."
)


def yes_no_question(finding: Finding) -> str:
    """The closed probe question for one finding."""
    return (
        f"Based on your analysis of this chest X-ray, is there {finding.value.lower()}? "
        "Answer with a single word: yes or no."
    )


def resolve_yes_no_token_ids(tokenizer: Any) -> tuple[list[int], list[int]]:
    """First-token ids for yes/no answer variants, used to read the probe logits."""
    yes_words = ("yes", " yes", "Yes", " Yes", "YES")
    no_words = ("no", " no", "No", " No", "NO")

    def first_ids(words: tuple[str, ...]) -> list[int]:
        ids: set[int] = set()
        for w in words:
            enc = tokenizer.encode(w, add_special_tokens=False)
            if enc:
                ids.add(int(enc[0]))
        return sorted(ids)

    return first_ids(yes_words), first_ids(no_words)


def yes_probability(logits_row: Any, yes_ids: list[int], no_ids: list[int]) -> float:
    """P(yes) from a next-token logit vector, restricted to the yes/no answer tokens."""
    import torch  # noqa: PLC0415

    yes_logit = logits_row[yes_ids].max()
    no_logit = logits_row[no_ids].max()
    pair = torch.stack([yes_logit, no_logit]).float()
    return float(torch.softmax(pair, dim=0)[0])


class HFVLMAdapter:
    """Reason-then-probe :class:`~latentreasoning.core.model.ModelAdapter` over a HF VLM.

    Subclasses set :attr:`name`, :attr:`config_key`, a default ``max_new_tokens``, and
    optionally :attr:`default_prompt`. Construction is offline; weights load on first use.
    """

    name: str = "hf-vlm"
    config_key: str = ""
    #: Reasoning elicitation used when ``generate(prompt=None)``. Override per model.
    default_prompt: str = REASONING_PROMPT

    def __init__(
        self,
        *,
        model_id: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 1024,
    ) -> None:
        self.model_id = model_id or MODELS[self.config_key].locator
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model: Any | None = None
        self._processor: Any | None = None
        self._yes_ids: list[int] | None = None
        self._no_ids: list[int] | None = None

    # -- Lazy weight loading -----------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: PLC0415
        from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        # None -> use HF's default cache (HF_HOME); override with LR_HF_CACHE if needed.
        cache_dir = os.environ.get("LR_HF_CACHE")
        self._processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=cache_dir)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, dtype=dtype, cache_dir=cache_dir
        ).to(self.device)
        self._model.eval()
        self._yes_ids, self._no_ids = resolve_yes_no_token_ids(self._processor.tokenizer)

    # -- Inference: reason once, then probe each finding -------------------------------

    def generate(
        self,
        record: CXRRecord,
        *,
        with_image: bool = True,
        prompt: str | None = None,
    ) -> ModelOutput:
        """Reason over ``record``, then read a yes/no probe per finding.

        ``with_image=False`` withholds the image (the no-image probe). An occluded image is
        passed by handing in a record whose region is already blacked out.
        """
        self._ensure_loaded()
        include_image = with_image and record.image is not None
        image = record.image if include_image else None
        instruction = prompt if prompt is not None else self.default_prompt

        user_content = self._user_content(instruction, image)
        reasoning = self._reason(user_content)
        preds = {f: self._probe(user_content, reasoning, f) for f in PROBE_FINDINGS}
        abstained = all(p < 0.5 for p in preds.values())
        return ModelOutput(
            text=reasoning,
            finding_predictions=preds,
            abstained=abstained,
            raw={"model_id": self.model_id, "with_image": include_image},
        )

    def _user_content(self, instruction: str, image: Any | None) -> list[dict[str, Any]]:
        from PIL import Image  # noqa: PLC0415

        content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
        if image is not None:
            pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
            content.insert(0, {"type": "image", "image": pil})
        return content

    def _reason(self, user_content: list[dict[str, Any]]) -> str:
        import torch  # noqa: PLC0415

        messages = [{"role": "user", "content": user_content}]
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            out = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        new_tokens = out[0][input_len:]
        return self._processor.decode(new_tokens, skip_special_tokens=True).strip()

    def _probe(self, user_content: list[dict[str, Any]], reasoning: str, finding: Finding) -> float:
        import torch  # noqa: PLC0415

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": reasoning}]},
            {"role": "user", "content": [{"type": "text", "text": yes_no_question(finding)}]},
        ]
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self._model.device)
        with torch.inference_mode():
            logits_row = self._model(**inputs).logits[0, -1, :]
        return yes_probability(logits_row, self._yes_ids or [], self._no_ids or [])

    # -- Single-shot probe (no free reasoning) -----------------------------------------

    def probe_finding(
        self, record: CXRRecord, finding: Finding, *,
        with_image: bool = True, prompt: str | None = None,
    ) -> float:
        """Single-shot ``P(finding present)``: one user turn, no free reasoning; ~1 forward pass."""
        self._ensure_loaded()
        include_image = with_image and record.image is not None
        image = record.image if include_image else None
        question = prompt if prompt is not None else yes_no_question(finding)
        return self._probe_single(self._user_content(question, image))

    def _probe_single(self, user_content: list[dict[str, Any]]) -> float:
        import torch  # noqa: PLC0415

        messages = [{"role": "user", "content": user_content}]
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self._model.device)
        with torch.inference_mode():
            logits_row = self._model(**inputs).logits[0, -1, :]
        return yes_probability(logits_row, self._yes_ids or [], self._no_ids or [])
