# Transformer confirmatory study E5

**Status:** completed historical campaign on a small one-layer transformer.
This is an English guide to the frozen study, not a new preregistration. The
[complete original protocol](https://github.com/leonardocaloi/neural-network-entry-exit-law/blob/a331371220b539a467a7302eac8ab580740b5014/05-experimentos/etapa5-transformer/README-e5-conf.md)
preserves the historical estimand labels, thresholds, disclosures, and chronology.

## Design and measurement

Four activation arms (softplus beta 3 and 10, GELU, hard ReLU), two attention
conditions (trainable/frozen), two injection depths (-0.3/-2.0), and 50 seeds
(2000–2049) give **800 runs**. The architecture uses dimension 16, two attention
heads, and four feed-forward units on 24 synthetic sequences of three tokens.
Training is deterministic gradient descent for 200,000 steps at learning rate 0.001.

One feed-forward unit is deliberately made inactive in each run. Calibrated
hysteresis (approximately 0.0476) suppresses threshold chatter. A sustained return
means the injected unit returns and is active at the end of the horizon.
Calibration used frozen-attention runs; this is a limitation of the instrument.

| ID | Question | Scope |
|---|---|---|
| E-A | Does frozen-input hard ReLU ever show a sustained return? | Any such return invalidates the experiment. |
| E-B | How does trainable attention change return probability? | Trainable-minus-frozen contrast with bootstrap clustering by seed. |
| E-C | Are sustained returns observed for shallow, frozen-attention softplus beta 3? | At least six returns meets the positive gate. The rejection threshold is too small to be reached by zero returns at N = 50; low counts can be inconclusive. |
| E-D | How do deep and shallow frozen-attention GELU cells differ? | Depth is a confound; the comparison does not prove a terminal derivative barrier. |
| E-E | How do softplus return proportions vary with beta? | Descriptive comparison, not a new theorem. |

## Archived observations

The shallow frozen-attention softplus-beta-3 cell has **22/50** sustained returns
(95% Clopper-Pearson interval **[0.300, 0.587]**), with area opening in **21/22**.
Hard ReLU has **20/100** sustained returns with attention trainable and **0/100**
with attention frozen. The MLP study conditions on dynamical deaths; this study
conditions on injected inactive units. Their denominators are not interchangeable.

The original E-C label refers to transfer of the theorem class. The current
manuscript reports the result observationally because trainable readouts and
other architectural differences go beyond the theorem's assumptions.

## Files and analysis

`config_e5.json` is the unchanged frozen configuration. `results-e5/` contains the
800 original run summaries and their manifest. `engine_e5_confirmatory.py` is the
instrumented engine, `run_e5.py` the campaign runner, and `analyze_e5.py` the frozen
analysis. Earlier engines and [pilot guides](rate-pilot.md) preserve development.

From the repository root:

```sh
python experiments/transformer/analyze_e5.py
```

This reads archived results without training. Full reproduction is documented
in the [reproduction guide](../../docs/REPRODUCIBILITY.md). The runner's `--smoke`
still executes two full training runs; it is not the lightweight artifact check.
