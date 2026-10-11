# Affine forcing and extension studies

`verify_forcing.py` and the saved `forcing_m.jsonl` study the affine live subsystem
in manuscript Theorem 6. Companion studies examine sharpness, integration step
size, width dependence, and replication constructions. These files supply the
released manuscript's numerical checks.

`trainable_readout.py`, `readout_scale.py`, and `verify_multiple_margins.py`
belong to later extension drafts. Their presence is not evidence that every
claim in those drafts has passed review. See
[trainable-readout scoping](../../docs/research-notes/2026-07-27-trainable-readout-scoping.md),
[multiple-margin scoping](../../docs/research-notes/2026-07-27-multiple-margin-scoping.md),
and [review 18](review-18.md).

## Historical auxiliary campaigns

`review14.py`, `review15_o5.py`, `review15_o5b.py`, and
`generate_artifacts.py` are full-campaign research scripts, not quick tests.
Version 0.2.1 removes an obsolete output directory and an unused import path;
their required modules and `forcing_m.jsonl` are included here. Regenerated
reports from `review14.py` and the wrapper go to `build/affine-forcing/` at the
repository root, leaving the original records untouched. Their full numerical
campaigns were not rerun for this maintenance release.

The supported small check is `python verification/check_entry_exit.py` from
the repository root; it uses an adaptive integrator on selected saved cases.
See the [reproduction guide](../../docs/REPRODUCIBILITY.md) for the distinction
between quick checks and full campaigns. The [roadmap](../../docs/ROADMAP.md)
identifies the remaining proof obligations.
