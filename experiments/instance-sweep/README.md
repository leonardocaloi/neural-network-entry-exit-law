# Sampled-instance sweep

`sweep_instances.jsonl` contains 600 sampled instances; 57 pass the implemented
hypothesis screening. For those retained instances, the recorded median relative
error in the first-order release coefficient is 2.68% at beta = 10,000.

This is a conditional numerical check, not a success rate over arbitrary networks.
`sweep_instances.py` is the historical generator and overwrites its output when
run. Use `python verification/smoke.py` from the repository root to inspect the
saved summary without regenerating the sweep. Full commands are in the
[reproduction guide](../../docs/REPRODUCIBILITY.md).
