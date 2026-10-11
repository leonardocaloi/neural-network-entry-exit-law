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

Some historical review scripts reference scratchpad files not included in the
release. The supported small check is `verification/check_entry_exit.py` from
the repository root; it uses an adaptive integrator on selected saved cases.
The [roadmap](../../docs/ROADMAP.md) identifies the remaining proof obligations.
