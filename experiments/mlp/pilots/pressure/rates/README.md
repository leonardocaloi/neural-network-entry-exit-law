# MLP confirmatory study 4a.3

**Status:** completed historical campaign; independent research, not peer-reviewed.
This English guide summarizes the frozen study. The
[complete original protocol](https://github.com/leonardocaloi/neural-network-entry-exit-law/blob/a331371220b539a467a7302eac8ab580740b5014/05-experimentos/etapa4-mlp-populacao/pilotos-4a2/pressao/taxa/README-4a3.md)
retains every original threshold, rationale, and pre-execution statement.
`config_4a3.json` and all saved per-run records are unchanged.

## Design

- Four arms: softplus at beta 3, 10, and 300, and hard ReLU.
- Widths 4 and 6; 50 seeds (1900–1949): **400 runs**.
- Sixteen synthetic regression examples; deterministic full-batch gradient descent,
  learning rate 0.001, and 500,000 steps.
- A separate 240-run rate pilot supplies sizing and eligibility parameters.

## Frozen questions

| ID | Quantity | Decision logic and limitation |
|---|---|---|
| C1 | Return probability conditional on an eligible dynamical death, primary beta-3 arm | Eligibility ends at H minus tau; tau = 192,464.5 steps. Fewer than 10 eligible deaths is inconclusive. The rejection threshold is approximately 0.07599 on the upper confidence bound; power to reject is limited at this size. |
| C2 | Fraction of all primary-arm dynamical returns with area opening during inactivity | At least six returns are required; the criterion compares the lower confidence bound with 0.5. Part of the temporal ordering is imposed by the event clock. The empirical failure is a return without recorded area opening. |
| C3 | Release gap, inactive depth, and activation sensitivity | Exploratory estimation; no confirmatory rejection rule. |
| C4 | Return proportions across beta | Descriptive comparison with uncertainty. |
| Validity | Hard-ReLU returns with fixed inputs | Any ground-truth return invalidates the experiment. |

The instrument uses one ground-truth event clock and excludes initially inactive
units from the dynamical-return endpoint. Repeated events are clustered by seed
in bootstrap analyses. Read event-level intervals with that dependence in mind.

## Archived observations

The primary arm has **16/35** returns per dynamical death, **15/30** per eligible
death, and recorded area opening in **13/16** returns. Hard ReLU has **0/17**
returns per dynamical death. These observations do not extend the fixed-readout
theorem to all trained MLPs.

## Files and analysis

| File | Role |
|---|---|
| `config_4a3.json` | Frozen design and decision parameters |
| `engine_4a3.py` | Training and ground-truth area instrumentation |
| `predict_4a3.py` | Historical pilot-derived configuration generator |
| `run_4a3.py` | Resumable campaign runner; expensive |
| `analyze_4a3.py` | Frozen analysis |
| `results-4a3/` | 400 original run summaries and manifest |
| `results/`, `rate-results.json` | Separate rate-pilot records and summary |
| [Post-hoc C3](c3-posthoc.md) | Exploratory timing analysis |

From the repository root, this reads the saved campaign without training:

```sh
python experiments/mlp/pilots/pressure/rates/analyze_4a3.py
```

The [reproduction guide](../../../../../docs/REPRODUCIBILITY.md) documents the
environment, quick artifact check, full campaign, and comparison tolerances.
