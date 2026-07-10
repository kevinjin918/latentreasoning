# Proposal: Grounding-Gated Latent Reasoning

Concise, fluffless. Date: 2026-07-10.

## The novel contribution
The mechanism (latent reasoning, adaptive halting, CT encoders) is all off-the-shelf. The new idea is one variable swap:

- **Standard halting stops on confidence. This stops on groundedness.** A confidently-confabulating model is exactly what confidence-halting misses.
- **Deliverable 1 (readout):** per latent step, decompose the update into image-evidence vs prior. Gives a grounding-vs-depth curve *inside* the reasoning.
- **Deliverable 2 (mechanism):** feed that readout back as the stopping rule; halt before the model confabulates. Prevents prior-amplification by construction.

Interpretability of latent reasoning is the moat: the field builds opaque latent reasoners and worries about their opacity; the prior tracecxr interp tooling can open the black box on the grounding axis. Interp here is the instrument, not the product.

## Groundedness, precisely
Groundedness = counterfactual dependence of the answer on the image evidence.
- Grounded: occlude the finding's region, the prediction changes.
- Ungrounded: occlude it, the prediction is unchanged (it came from the prior).

## Runtime measurement (two-stream)
Run the reasoning loop twice in lockstep, same weights:
- Stream E: reasons with the image + text.
- Stream P: reasons with the finding's region occluded (the "prior" condition).

`groundedness_t = divergence(y_t^E, y_t^P)`. Large gap = image is driving = keep going. Gap collapses = ungrounded = halt/abstain. Exact, online, at 2x forward passes (on-brand for a test-time-compute method). A learned probe on `h_t` (trained against the 2x signal offline) is the cheaper deployment approximation.

## Halting rule (2 axes)
Signals per step: **convergence** (is the answer still changing?) and **groundedness** (are the changes image-driven?).

| | grounded | ungrounded |
|---|---|---|
| converged | stop, trust | stop, abstain (pure prior guess) |
| still moving | keep reasoning | stop now, flag (confabulation onset) |

Different from confidence-halting (1D) and from GPRO (routes discrete paths; RL-trained). This is a per-iteration continuous stop on a measured groundedness signal.

## The mechanism (components, all established)
- Frozen encoder: CT-CLIP (in-domain for CT-RATE) for the CT track; MedGemma/RadDINO for the CXR track. Not JEPA.
- Shared-weight recurrent block iterated T times in latent space.
- Grounding-gated halting picks T per case (ACT/PonderNet head, deep-start bias -3 to avoid halting collapse).
- Head: segmentation mask (CT) or finding decision (CXR).

## Vehicle
ReXGroundingCT challenge, MICCAI 2026: text finding + 3D chest CT -> 3D mask, metric = Dice per finding. Grounding IS the metric (no accuracy-vs-grounding tension); CT gives reasoning real headroom; deadline Sept 2026; external pretrained models allowed; data public.

## Plan (phased, honest about what each tests)
1. **Verbalized sanity + measurement validation (1-2 days, existing CXR/MedGemma harness).** Confirm the *phenomenon* (reasoning drifts to prior) and validate the two-stream measurement. Tests the premise, NOT the latent mechanism.
2. **Minimal latent block on CXR (~1 week).** Frozen encoder + small recurrent block; vary iterations; run the groundedness probe. Tests latent reasoning *specifically*, on data we already have. The real go/no-go for the mechanism.
3. **CT / ReXGrounding (~2 weeks).** CT-CLIP + grounding head + grounding-gated halting on the 50 public val cases, then the challenge. The three curves: accuracy-vs-compute, grounding-vs-depth (Figure 1), off-domain transfer.

## Honest novelty / nearest neighbors
- Latent reasoning, halting, CT encoders: all taken.
- Closest: GPRO (perception-vs-reasoning router, RL-trained, discrete paths). This differs: per-iteration continuous stop on a measured groundedness signal inside the loop.
- Latent-reasoning interpretability is nascent, mostly negative results; this turns it into a positive method on grounding.
- All arXiv IDs from the scan are flagged VERIFY before formal citation.

## Risks
- Runtime groundedness measurement cost/quality (2x passes; probe as fallback). Key technical risk.
- Premise may not hold for *latent* reasoning even if it holds for verbalized (they may differ; that difference is part of the story). Step 2 tests it.
- CheXthought roadmap may already claim the grounding direction (ask Sonali).
- Engineering-heavy challenge; mid-pack Dice is fine, the paper's claim is the mechanism + science.

## Extensions (scope-guarded, not core)
- Hybrid latent + text: verbal channel as a faithfulness anchor auditing the latent channel.
- Coconut-style latent tokens (variant b) after the visual-refinement variant (a) validates.
