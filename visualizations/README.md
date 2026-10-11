# Visual explanations

Open [`memory-gate.html`](memory-gate.html) in a browser to inspect the existing
two-unit simulation. It is a self-contained file with no external dependencies.
It starts paused and performs no simulation updates until **Run** is selected.

The controls adjust speed, temperature, activation sharpness, and whether the
second unit is frozen. The three panels show parameter space, network output,
and the leading-margin/area history. **Save PNG** exports the current view.

This is an exploratory stochastic toy, not a reproduction of the deterministic
softplus theorem. In particular, noise can move a hard-ReLU unit whose own
gradient is zero. A return event in this display does not establish an area-zero
release law. The simulation equations are preserved from the original artifact;
the English labels and scope statement replace stronger historical wording.

For the conceptual connections across fields, use the
[research map](../docs/RESEARCH-MAP.md). For numerical evidence, use the
[saved analyses and verification scripts](../docs/REPRODUCIBILITY.md).
