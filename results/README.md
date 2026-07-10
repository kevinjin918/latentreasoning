# Results

Consolidated narrative. Each experiment banks raw JSON + a `.md` interpretation here.

## Phase 0 — three-stream groundedness (measurement validation + premise baseline)

The instrument: for a boxed finding, run the model on three streams (evidence / region-
occluded / no-image) and read P(finding); `region_groundedness = P_evidence - P_occluded`,
`prior_reliance = P_no_image`. See `../src/latentreasoning/grounding/`.

- **MedGemma** (`phase0_two_stream_medgemma.md`, 50 NIH bbox records, 2026-07-10): region
  groundedness **0.32**, prior reliance **0.86**, confabulation-risk **60%**. Verbalized
  reasoning is substantially prior-driven; the measurement is validated on a real 4B model.
- CheXOne — pending (the verbalized-CoT reasoning baseline + causal-audit subject).

## Next
- Phase 0 on CheXOne (latent-vs-verbalized baseline; the causal audit).
- Phase 1: adaptive latent block spliced over the frozen encoder; grounding-vs-depth.
