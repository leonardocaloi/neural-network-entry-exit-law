# MLP studies

This directory preserves the progression from a failed noisy population
instrument to a deterministic confirmatory study of dynamical inactivity.

| Location | Role |
|---|---|
| `results/`, `config_stage4.json`, `analysis.py` | Initial 600-run campaign, **invalidated** by P-NULL |
| `calibrate_instrument*.py`, `instrument_v3.py`, `stratified_null.py` | Historical instrument development |
| `pilots/` | Exploratory width, noise, and detection studies |
| `pilots/pressure/` | Capacity-pressure pilot |
| [Final campaign](pilots/pressure/rates/README.md) | Rate pilot and 400-run confirmatory study, ID 4a.3 |

The initial campaign's saved data remain available for studying measurement
failure. They must not be pooled with the final campaign as confirmatory evidence.
See [negative results](../NEGATIVE-RESULTS.md).

The [complete original stage-4 protocol](https://github.com/leonardocaloi/neural-network-entry-exit-law/blob/a331371220b539a467a7302eac8ab580740b5014/05-experimentos/etapa4-mlp-populacao/README.md)
is preserved at the pre-reorganization commit. This English directory guide
describes the final organization; it is not a retrospective preregistration.
For execution, use the [reproduction guide](../../docs/REPRODUCIBILITY.md).
