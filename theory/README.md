# Theory and proof development

The current theorem statements and full proof presentation are in
[`paper/paper-v4.tex`](../paper/paper-v4.tex), with the
[PDF](../paper/paper-v4.pdf) as the reading version. This directory preserves
the route from the scalar normal form to the coupled entry-exit law.

| Stage | Reading guide | Verification script |
|---|---|---|
| Normal form and quadrature | [Lemma 1](lemma-L1-normal-form.md) | `verify_lemma_L1.py` |
| Live-system feedback | [Lemma 2](lemma-L2-feedback.md) | `verify_lemma_L2.py` |
| Entry corner | [Lemma 3](lemma-L3-entry-corner.md) | `verify_lemma_L3.py` |
| Sticking and release | [Lemma 4](lemma-L4-release-bounds.md) | `verify_lemma_L4.py` |
| Coupled bootstrap | [Lemma 5](lemma-L5-bootstrap.md) | `verify_lemma_L5.py` |
| Exit corner | [Lemma 6](lemma-L6-exit-corner.md) | `verify_lemma_L6.py` |

The earlier `step0`–`step6` guides and `draft-v0`–`draft-v3` sources preserve the
development sequence. `forcing-theorem-draft.tex` develops the affine extension.
`trainable-readout-draft.tex` and `multiple-margin-draft.tex` are subsequent
extension drafts, not additional results validated by this release.

The scripts check selected identities, estimates, and examples. They are not
formal proof certificates. See the [review](../docs/REVIEW.md),
[research map](../docs/RESEARCH-MAP.md), and [roadmap](../docs/ROADMAP.md) for scope.
The [quick check](../docs/REPRODUCIBILITY.md) is the recommended entry point;
there is no need to execute every historical proof script to inspect the release.
