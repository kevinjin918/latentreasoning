# latentreasoning

Latent (non-verbal) reasoning for medical vision models: reasoning as iterated computation in latent space, instead of generated chain-of-thought text.

## Why latent reasoning over chain-of-thought

Verbalized CoT reasons by emitting tokens. It is slow (one forward pass per token), bottlenecked through language (every step must be expressible in words), and discrete (each step commits to a word). Latent reasoning iterates a hidden state instead:

- adaptive depth without the per-token cost,
- can hold uncertainty across steps rather than collapsing to words,
- not constrained to language-expressible reasoning.

The price of latent reasoning is opacity: you lose the readable trace. Making it legible and controllable is the project.

## Mechanism

```
frozen encoder -> shared-weight recurrent block (iterated T times in latent space) -> head
```

Adaptive halting picks T per case (think longer on hard cases, stop early on easy ones).

## The control lens: groundedness

Latent reasoning can drift toward the model's prior as it iterates. So halting is gated on **groundedness** (does the answer still depend on the image?) rather than confidence, with a per-step readout of whether each step is evidence-driven or prior-driven. Details in `docs/PROPOSAL.md`.

## Status

Early: scaffolding + proposal. First experiment is the premise/measurement probe (`docs/PROPOSAL.md`), reusing tooling from the sister repo `tracecxr`.

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
