"""CheXOne adapter — the verbalized-CoT baseline and causal-audit subject.

CheXOne (arXiv:2604.00493, Langlotz lab; open at github.com/YBZh/CheXOne) is a
reasoning-enabled CXR VLM built on **Qwen2.5-VL-3B** and RL-tuned (GRPO) to emit explicit,
report-derived reasoning traces. It is the strongest verbalized-CoT medical reasoner and the
subject of this project's central audit: does its reasoning actually depend on the image, or
is it report-anchored rationalization? (Its training traces were LLM-generated from the
reference report and prompted to *look* image-derived — see the paper's Fig 9.)

It reasons in "Reasoning mode", triggered by the prompt "Please reason step by step and put
your final answer within \\boxed{}". We reuse the shared reason-then-probe flow; only the
default reasoning prompt differs from MedGemma.

VERIFY on first GPU run (could not be executed offline — no CheXOne weights / GPU here):
  1. the exact HF repo id (config resolves ``YBZh/CheXOne``; the release may live elsewhere,
     e.g. under StanfordAIMI) and that ``AutoProcessor`` / ``AutoModelForImageTextToText``
     load the Qwen2.5-VL checkpoint (else override loading like the CheXagent adapter).
  2. the reasoning-mode prompt formatting and that ``\\boxed{}`` answers do not disrupt the
     separate yes/no probe turn.
  3. yes/no first-token ids resolve on the Qwen2.5 tokenizer (uses the shared resolver).
"""

from __future__ import annotations

from latentreasoning.core.model import register_model

from ._base import HFVLMAdapter

#: CheXOne's verbalized-CoT elicitation ("Reasoning mode").
CHEXONE_REASONING_PROMPT: str = (
    "Interpret this chest X-ray. Please reason step by step and put your final answer "
    "within \\boxed{}."
)


class CheXOneAdapter(HFVLMAdapter):
    """Reason-then-probe adapter for CheXOne (Qwen2.5-VL-3B, verbalized reasoning mode)."""

    name = "chexone"
    config_key = "chexone"
    default_prompt = CHEXONE_REASONING_PROMPT

    def __init__(self, *, model_id: str | None = None, device: str | None = None,
                 max_new_tokens: int = 768, reasoning: bool = True) -> None:
        super().__init__(model_id=model_id, device=device, max_new_tokens=max_new_tokens)
        #: When False, run "Instruction mode" (direct answer, no reasoning trace).
        self.reasoning = reasoning
        if not reasoning:
            self.default_prompt = HFVLMAdapter.default_prompt


register_model("chexone", CheXOneAdapter)
