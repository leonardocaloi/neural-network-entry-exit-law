# Negative results and invalidated instruments

Negative results are part of the evidence. A later hypothesis does not overwrite
an earlier failed prediction.

## Initial MLP campaign: invalidated by its null control

The stage-4a campaign completed 600 runs under the locally preregistered protocol
identified by original commit `6411ac8`. Pure diffusion triggered the calibrated
test (the historical record reports p = 0.002 at width 16). Under the frozen
P-NULL rule, this invalidated the instrument and its mechanism verdicts.

The subsequent diagnosis was exploratory: rapid boundary crossings and
between-unit heterogeneity undermined the intended interpretation. The record
reports that 59% of episodes lasted at most five steps. The redesign considered
hysteresis, stratified controls, lower noise, and longer observation windows.
The later 400-run deterministic campaign is a separate study.

See the [instrument-failure guide](../docs/research-notes/2026-07-24-stage4a-invalid-instrument.md)
and the original records in `mlp/results/`.

## Earlier toy studies

- **ReLU toy v2:** the proposed deterministic race schedule was falsified;
  fluctuating reopening motivated a new model and a new preregistration.
- **Pure-fluctuation reference model:** rejected at the predictor stage, rather
  than counted as an executed experimental confirmation.
- **ReLU toy v1 and crossing toy:** rejected or archived at the design stage;
  proposed distinctions were already decidable or canceled at the relevant order.

These findings concern their respective models and instruments. They do not
establish a universal result about all inactive units, stochastic training,
or large language models.
