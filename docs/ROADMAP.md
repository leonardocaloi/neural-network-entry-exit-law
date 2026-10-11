# Research roadmap

These are proposed next steps based on the existing manuscript, review, and
extension notes. They are not completed experiments, new preregistrations,
scheduled jobs, or commitments to a submission date. No training campaign was
launched as part of the repository reorganization.

## Priorities and completion criteria

| Priority | Question | Existing starting point | What would count as progress |
|---|---|---|---|
| 1 — Mathematical consolidation | Are the corrected bootstrap step and entry-corner estimates sufficient for each claimed order? | [Review sections 3 and 6](REVIEW.md), manuscript appendices | A line-by-line review that identifies hypotheses used at every step, resolves the revised induction argument, and states whether the unresolved refinement affects only the sharper remainder. |
| 2 — Trainable readouts | Which bounds survive when the live subsystem is no longer affine? | [Scoping guide](research-notes/2026-07-27-trainable-readout-scoping.md), [draft](../theory/trainable-readout-draft.tex), [review 18](../experiments/affine-forcing/review-18.md) | A consistent statement, corrected readout dependence and guards, and a proof audit before treating the extension as a result. Selected numerical cases would supplement that audit. |
| 3 — Competing margins | What replaces a single signed-area clock when several data points influence release? | [Scoping guide](research-notes/2026-07-27-multiple-margin-scoping.md), [draft](../theory/multiple-margin-draft.tex) | Explicit rank and separation assumptions, a check of the proposed reduction, and a clear distinction between boundary initialization and an episode entered from positive margin. Recover or replace missing scratchpad evidence transparently. |
| 4 — Optimizer dependence | Does optimizer state change the relevant memory or the first-order release behavior? | [Adam pilot](../experiments/adam-memory/README.md), [protocol guide](../experiments/adam-memory/confirmatory/PROTOCOL.md) | A bounded study separating optimizer-state, step-size, and epsilon effects, with a frozen primary endpoint and saturation controls. Review existing artifacts before proposing further runs. |
| 5 — Measurement and external validity | Which observations survive an improved instrument or a new architecture? | [Invalidated instrument](../experiments/NEGATIVE-RESULTS.md), final MLP and transformer protocols | Resolve the GELU depth confound, identify a meaningful effect size and power target, and separate injected inactivity from natural dynamical deaths before scaling the experiment. |
| 6 — Broader theoretical connections | Can a kernel, distributional, or geometric limit add a prediction beyond the current local law? | [Research map](RESEARCH-MAP.md) and cited primary literature | One precisely stated connection at a time, with assumptions, a proposed measurable consequence, and a possible counterexample. Shared terminology alone is insufficient. |

## Practical sequence

First consolidate the mathematical statements already being shared. Then choose
one extension with a clear gap and an inexpensive discriminating check. Decide
whether a larger experiment is justified only after the instrument and endpoint
are specified. Preserve negative results and write a new protocol for a new
hypothesis rather than editing an old preregistration after observing results.

Before any full new campaign, record its compute budget, seeds, dependency
versions, stopping rule, data schema, and analysis commit. Existing historical
times are 4.9 hours for the MLP campaign and 13.9 hours for the transformer with
four CPU workers; these are not runtime estimates for the proposed extensions.

## Communication and submission

The public repository is ready to inspect as an independent research artifact.
A journal or preprint-server submission is a separate action. It should use the
current manuscript, a clear account of AI assistance, the remaining proof
limitations, and the destination's current requirements.

For a research or evaluation interview, a useful short presentation would trace
one question through its hypothesis, instrument, negative control, observed
failure, redesign, and final scope. This presentation would describe the work
already done; it does not require another training campaign.
