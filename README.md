# An entry–exit law for the resurrection of dead neurons under smoothed ReLU activation

**Leonardo Caloi Santos — Independent Researcher, Brazil**

An independent mathematical and computational study of inactive units in
artificial neural networks. The manuscript is **not peer-reviewed**; its
proofs and experiments were developed with substantial LLM assistance.

[Read the manuscript (PDF, 66 pages)](06-notas/paper/paper-v4.pdf) ·
[LaTeX source](06-notas/paper/paper-v4.tex) ·
[Reproduction guide](docs/REPRODUCIBILITY.md) ·
[Review and corrections](docs/REVIEW.md) ·
[Release](https://github.com/leonardocaloi/neural-network-entry-exit-law/releases/tag/v0.1.1)

## Research question

A hard-ReLU unit with negative pre-activation on every training example has
zero gradient through its own parameters. With fixed inputs and ordinary
unregularized gradient descent, that channel cannot reactivate it. What
happens when ReLU is replaced by softplus and its sharpness increases?

The manuscript studies this question under deterministic gradient flow. Its
central result describes a limiting trajectory that becomes inactive,
retains a memory of the residual, and returns when a signed residual integral
reaches zero. The experiments investigate related behavior in small MLPs and
a one-layer transformer. They concern artificial units, not biological neurons.

## Contributions and scope

- **Mathematical results presented in the manuscript.** Theorem 2 gives an
  entry–exit law for an isolated episode in a shallow softplus network, with
  fixed readout and one-dimensional inputs. Theorem 6 extends it to an affine
  live subsystem with a fixed activation pattern. The manuscript establishes
  an $O(\log\beta/\beta)$ trajectory approximation away from the switching
  times and an explicit first-order release correction. Its stated bounds in
  the affine extension scale as $O(\sqrt m)$, with threshold $\beta_0=O(m)$,
  at fixed forcing and the other specified constants.
- **Numerical checks of the asymptotics.** A 600-instance sweep retains 57
  instances under the implemented hypothesis screening. Additional archived
  experiments examine the affine extension and discretization error.
- **Experiments beyond the proved setting.** Pre-registered local protocols
  govern a 400-run MLP campaign and an 800-run transformer campaign. These
  train additional parameters and do not constitute applications of the
  fixed-readout theorem without further assumptions.
- **An auditable research record.** The repository includes pre-registration
  commits, configurations, original saved results, manifests, failed
  hypotheses, invalidated instruments, and the publication review.

The proof arguments have not received independent mathematical verification.
The review records corrections to a remainder bound and a bootstrap step,
and narrows claims about Jacobian rank, nonsmooth branch selection, and GELU.
See [the review record](docs/REVIEW.md) for the exact changes and remaining
questions. Earlier drafts and Portuguese laboratory notes are historical
artifacts; `paper-v4.tex` and its PDF are the current manuscript.

## Mathematical intuition

For softplus,

$$g_\beta(z)=\frac{1}{\beta}\log(1+e^{\beta z}),\qquad g'_\beta(z)=\sigma(\beta z).$$

Let $A$ denote the unit that becomes inactive on a distinguished datum $j$.
The hybrid reference follows ReLU until $t_{\mathrm{hit}}$, freezes $A$, and
continues evolving the live subsystem. Its residual excluding $A$ and signed
area are

$$R^{\mathrm{hyb}}(t)=y_j-f_{-A}(x_j;t),\qquad
\mathcal A^{\mathrm{hyb}}(t)=\int_{t_{\mathrm{hit}}}^{t}R^{\mathrm{hyb}}(s)\,ds.$$

Under the stated isolation, margin, and transversality hypotheses, the area
is negative during the inactive episode. Its first subsequent zero defines
$t_{\mathrm{exit}}$, with $R^{\mathrm{hyb}}(t_{\mathrm{exit}})>0$. The
manuscript's softplus release law is

$$t_{\mathrm{release}}^\beta=t_{\mathrm{exit}}
-\frac{1}{cR^{\mathrm{hyb}}(t_{\mathrm{exit}})}\frac{1}{\beta}
+o(\beta^{-1}),\qquad c=\frac{x_j^2+1}{n}.$$

Here the dying unit's readout is fixed at $+1$. The correction includes the
entry delay from a shared initial state with positive margin. A prescribed
forcing problem initialized directly at zero has different first-order
behavior; these initializations must not be conflated. The signed area,
rather than the instantaneous sign of the residual alone, determines the
limiting release.

## Experimental evidence

The counts below can be recalculated from the supplied records without
training a network. The full grids, eligibility rules, confidence intervals,
and pre-registered decision criteria are preserved in the original analyses.

| Study | Design | Selected recorded results |
|---|---|---|
| Two-unit numerical sweep | 600 sampled instances; 57 retained; $\beta=10^4$ | Median relative coefficient error **2.68%**, 90th percentile **7.26%**, maximum **12.01%** among the 57 retained instances. This is not a success rate over arbitrary networks. |
| MLP confirmatory | 400 runs: four activation arms × widths 4 and 6 × 50 seeds; 500,000 steps | In the primary softplus-$\beta3$ arm: **16/35** returns per dynamical death, **15/30** per eligible death, and area opening in **13/16** returns. Hard ReLU: **0/17** returns per dynamical death. |
| Transformer confirmatory | 800 runs: four arms × attention trainable/frozen × two injection depths × 50 seeds; 200,000 steps | Shallow softplus-$\beta3$ with frozen attention: **22/50** sustained returns, 95% Clopper–Pearson interval **[0.300, 0.587]**; area opening in **21/22**. Hard ReLU with attention trainable: **20/100**, versus **0/100** with attention frozen. |

The MLP estimands condition on dynamical deaths; the transformer estimands
condition on deliberately injected inactive units. The transformer has model
dimension 16, two attention heads, and four feed-forward units, trained on
24 synthetic sequences of three tokens. Both confirmatories use deterministic
gradient descent on CPU, with no added noise.

The MLP area counter imposes part of the temporal ordering by construction;
the empirical question is whether the area opens during the inactive episode.
Repeated events are clustered by seed in the bootstrap analyses. The
event-level Clopper–Pearson intervals should be read alongside that dependence.

GELU's derivative changes sign near $z=-0.7518$, but the experiment does not
establish a general terminal barrier: the deep-injection result remains
confounded with depth. An Adam pilot is exploratory. Negative results include
a falsified stochastic schedule and a 600-run instrument invalidated by its
own null control. They are retained in
[`05-experimentos/NEGATIVOS.md`](05-experimentos/NEGATIVOS.md).

## Repository guide

Historical Portuguese paths and protocol text are retained to preserve
references and the original research record. The public introduction and
reproduction guide are in English.

| Path | Contents |
|---|---|
| [`06-notas/paper/`](06-notas/paper/) | Current manuscript, historical versions, figures, and figure scripts |
| [`06-notas/t2-prova/`](06-notas/t2-prova/) | Proof notes, standalone drafts, symbolic and numerical verification scripts |
| [`05-experimentos/t2-sweep-instancias/`](05-experimentos/t2-sweep-instancias/) | 600-instance sweep and its saved output |
| [`05-experimentos/t2-n-neuronios/`](05-experimentos/t2-n-neuronios/) | Affine-extension numerics, controls, and exploratory extensions |
| [`05-experimentos/etapa4-mlp-populacao/`](05-experimentos/etapa4-mlp-populacao/) | MLP pilots, invalidated instrument, and confirmatory `pilotos-4a2/pressao/taxa/` |
| [`05-experimentos/etapa5-transformer/`](05-experimentos/etapa5-transformer/) | Transformer protocols, engines, analyses, and saved results |
| [`verification/`](verification/) | Lightweight artifact checks and numerical comparison tools |
| [`docs/`](docs/) | Reproduction instructions, review record, and original-to-public commit mapping |

## Quick start

From the repository root, with Python 3.14:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python verification/smoke.py
```

The smoke check verifies the SHA256 manifests of **1,800 original saved
records**: 400 MLP confirmatory, 800 transformer confirmatory, and 600 from
the invalidated instrument. It also checks the complete confirmatory grids,
central counts, confidence intervals, and the sweep summary. The recorded
runtime was about **5 seconds** on an Intel Core i9-9900K. It does not train
networks or change archived results.

Optional small numerical checks:

```sh
python verification/check_entry_exit.py
```

This checks selected identities and three archived instances with an adaptive
integrator. It is not a proof certificate. PyTorch is needed only for training
and optional engine checks; see the
[reproduction guide](docs/REPRODUCIBILITY.md) for installation, analysis-only
commands, figure generation, and full campaigns.

Historical full-campaign times were **4.9 hours** for the MLP and **13.9 hours**
for the transformer, with four CPU workers. Neither campaign was rerun for
publication. The archived files are run and episode summaries, not complete
per-step trajectories. Regenerated gzip bytes can differ because of timing
metadata, library versions, and floating-point behavior; comparison tools and
explicit tolerances are documented.

## Limitations and research status

The mathematical results require their stated hypotheses, including fixed
readouts and isolated switching behavior. The trained MLP and transformer
experiments are small and go beyond those hypotheses. They do not establish
behavior in production-scale networks. The stochastic setting, broader
architectures, and a sharper asymptotic remainder remain unresolved.

The manuscript is independent research and has not been peer-reviewed.
Its original date is July 26, 2026; the public-release revision is dated
October 11, 2026. Pre-registration means commits made before the recorded
campaigns in the local research history, not registration with an external
service. Relevant history is preserved, with original hashes mapped in
[`docs/provenance/commit-map.tsv`](docs/provenance/commit-map.tsv).

The historical dependency lockfile is incomplete, and the original PyTorch
version was not recorded. Pinned release dependencies describe the tested
release environment, not a recovered historical lockfile.

## AI assistance

Large language models assisted with mathematical derivations, coding,
editing, and adversarial review. The author defined the hypotheses and
pre-registered decision criteria, made the research decisions, and read the
cited sources. Publication preparation was also AI-assisted. This disclosure
does not imply independent external validation of the mathematics.

## License and citation

Original software and its software documentation are licensed under
[MIT](LICENSE). The manuscript, research notes, original figures, and
experimental records are licensed under
[CC BY 4.0](LICENSES/CC-BY-4.0.txt), to the extent applicable. Third-party
dependencies and cited works retain their own licenses.

Use [`CITATION.cff`](CITATION.cff) and identify the release or commit used:

> Leonardo Caloi Santos (2026). *An entry–exit law for the resurrection of dead
> neurons under smoothed ReLU activation*. Independent research manuscript,
> public-release revision. https://github.com/leonardocaloi/neural-network-entry-exit-law
