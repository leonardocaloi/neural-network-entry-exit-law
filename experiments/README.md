# Experiments

The studies below form a sequence of increasingly demanding questions about
inactive artificial neurons. Their evidence levels differ. The current
[manuscript](../paper/paper-v4.pdf) and [review](../docs/REVIEW.md) define the
claims supported by the public release.

| Study | Purpose | Status |
|---|---|---|
| [Instance sweep](instance-sweep/) | Test the release coefficient on sampled instances | 600 sampled; 57 retained under hypothesis screening |
| [Affine forcing](affine-forcing/) | Width dependence and live-system forcing | Numerical checks of the manuscript extension; later drafts are exploratory |
| [MLP](mlp/) | Population instrumentation and dynamical returns | An invalidated 600-run instrument, pilots, and a separate 400-run confirmatory campaign |
| [Transformer](transformer/) | Separate own-parameter and upstream return channels | 800-run confirmatory campaign on a small one-layer architecture |
| [Adam memory](adam-memory/) | Optimizer state and alternative release clocks | Exploratory pilot and subsequent protocol artifacts |
| [Crossing toy](crossing-toy/) | Test the original contact analogy | Archived design; not evidence for a general contact law |
| [ReLU toy v1](relu-death-v1/) | Separate structural identities from predictions | Rejected design |
| [ReLU toy v2](relu-death-v2/) | Reduced stochastic race model | Falsified model |
| [ReLU toy v3](relu-death-v3/) | Follow-up stochastic mechanism | Historical toy study |
| [ReLU toy v4](relu-death-v4/) | Temperature-dependent residual | Historical toy follow-up |

Start with the [negative-results register](NEGATIVE-RESULTS.md), then the two
confirmatory study guides. Saved records are included; analysis does not require
retraining. Use the [reproduction guide](../docs/REPRODUCIBILITY.md) for commands.
Machine-readable keys and frozen run identifiers retain their historical spelling.
