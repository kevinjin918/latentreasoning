# Phase 0 (MedGemma) — three-stream groundedness baseline

**Run:** 2026-07-10, H100, `scripts/phase0_two_stream.py --model medgemma --limit 50 --max-new-tokens 256`.
**Data:** 50 NIH ChestX-ray14 records with a radiologist box for effusion / pneumothorax.
**Data/JSON:** `phase0_two_stream_medgemma.json` (50 rows).

## What this run is
The measurement-validation + premise-baseline step. For each boxed record, MedGemma's
reason-then-probe is run on three streams and P(finding) is read from each:
`evidence` (full image), `occluded` (finding's box blacked out), `no_image` (image withheld).
It tests the **phenomenon** (does verbalized reasoning depend on the image evidence?), not the
latent mechanism (that is phase1), at a single reasoning depth (256 tokens).

## Result

| metric | value |
|---|---|
| mean region_groundedness (P_evidence - P_occluded) | **0.322** |
| mean prior_reliance (P_no_image) | **0.864** |
| frac_grounded (region drop >= 0.2) | **0.40** |
| frac_confabulation_risk (prior >= 0.5 AND not grounded) | **0.60** |

## Reading
- **MedGemma is substantially prior-driven.** It asserts the boxed finding ~86% of the time
  with **no image at all** (consistent with the TraceCXR no-image ~1.0 result; lower here as
  it is per-boxed-finding with a 256-token reasoning budget).
- **Region grounding is modest.** Removing the finding's own region drops P by only ~0.32 on
  average; only 40% of cases clear the "grounded" bar. 60% are in the confabulation-risk
  quadrant (asserted, but the answer does not depend on the evidence).
- **The instrument works.** The three-stream measurement produces a real, non-degenerate
  spread on a real 4B model, not a degenerate all-0 / all-1. This validates the harness the
  rest of the project depends on.

## Role in the arc
This is the baseline bar. Next: the same measurement on **CheXOne** (does the lab's
verbalized-CoT reasoning model ground any better, or is its report-derived reasoning
similarly prior-anchored?), then **phase1** asks whether an adaptive latent-reasoning block
with grounding-gated halting moves region_groundedness above this baseline.

## Caveats
- 50-record sanity sample (not the full ~801 bbox subset); single reasoning depth (no
  depth sweep yet); MedGemma is the un-reasoning-tuned baseline VLM.
