# latentreasoning

Grounding-gated latent reasoning for medical vision models.

**Thesis.** Latent reasoning models are efficient but opaque, and can confabulate: as a model "thinks" longer, it can drift from image evidence toward its prior. This project builds a latent reasoning loop that **stops when it stops looking at the image**, and a **per-step readout** of whether each reasoning step is driven by evidence or by prior.

## The idea in one line

Standard reasoning halts on *confidence*. This halts on *groundedness*, because a confidently-confabulating model is exactly the failure confidence cannot see.

- **Groundedness** = counterfactual dependence of the answer on the image (occlude the region, does the answer change?).
- **Measured at runtime** by running two synchronized streams, one with the evidence and one with the region occluded, and watching the gap between them across reasoning depth.
- **Halting rule** (2 axes): keep reasoning while steps still pull in image evidence; stop the moment the answer keeps moving but the movement is no longer image-driven (confabulation onset); abstain in the ungrounded cases.

## Status

Early. Repo scaffolding + proposal. First experiment is the premise/measurement probe (see `docs/PROPOSAL.md`), which reuses tooling from the sister repo `tracecxr`.

## Layout

```
docs/PROPOSAL.md      the current proposal (mechanism, plan, honest novelty)
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

Secrets live in `.env` (gitignored: Redivis + Anthropic tokens, copied from `tracecxr`).

## Relationship to tracecxr

`tracecxr` (sister repo) is the prior mechanistic-interp work on medical-VLM hallucination. Its occlusion / no-image / attention-attribution tooling is what makes the groundedness measurement here possible. This repo is the separate, forward-looking latent-reasoning project. Kept distinct on purpose.
