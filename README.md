# latentreasoning

Latent-reasoning medical vision models that stay grounded in the image.

## The problem

Medical vision-language models increasingly reason before answering, and that reasoning can make them worse. As a model thinks longer, it drifts from the scan toward its priors (typical anatomy, the report context, disease base rates) and asserts findings it cannot actually see. In radiology that is the dangerous failure: a confident, fluent, ungrounded call. Prior work (`tracecxr`) measured it on chest imaging: no-image hallucination rate ~1.0, and findings still asserted 49% of the time after their region is blacked out.

## The approach

Do the reasoning in latent space (an iterated hidden state over a frozen medical encoder) rather than as generated chain-of-thought text. This buys adaptive test-time depth, but latent reasoning is opaque, so the work is making it **legible and controllable on the axis that matters clinically: grounding.** Mechanism details in `docs/PROPOSAL.md`.

## Grounding (the core)

**Groundedness** = does the answer actually depend on the image evidence? (Occlude the finding's region: does the prediction change, or does it come from the prior?) Two contributions:

- **A per-step readout** of whether each reasoning step is driven by image evidence or by prior, measured at runtime with two synchronized streams (evidence vs region-occluded).
- **Grounding-gated halting**: stop reasoning when it stops being grounded (before it confabulates), and abstain rather than guess. Standard halting stops on *confidence*, which is blind to confident confabulation; this stops on *groundedness*.

## Application

Chest CT and CXR finding interpretation. The target vehicle is the **ReXGroundingCT challenge (MICCAI 2026)**: given a 3D chest CT and a free-text finding, ground it to a 3D segmentation mask, where Dice against the mask *is* a grounding score. The clinical payoff is a model that localizes findings to real evidence and says "I cannot tell from this image" instead of confabulating, with a reasoning depth that adapts to case difficulty.

## Status

Early: scaffolding + proposal. First experiment is the premise/measurement probe (`docs/PROPOSAL.md`), reusing `tracecxr` tooling.

## Layout

```
docs/PROPOSAL.md      mechanism, plan, honest novelty
src/latentreasoning/  package
tests/                mock-first tests (no GPU/weights by default)
```

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest        # mock-first, no GPU/data required
ruff check .
```

Secrets live in `.env` (gitignored, copied from `tracecxr`).

## Relationship to tracecxr

`tracecxr` (sister repo) is the prior mechanistic-interp work on medical-VLM hallucination; its occlusion / attention-attribution tooling powers the groundedness readout here. This is the separate, forward-looking project, kept distinct on purpose.
