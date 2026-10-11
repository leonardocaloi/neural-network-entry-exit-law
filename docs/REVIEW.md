# Publication review record (October 2026)

This file records what was done to the research project between the last
research commit (5 August 2026) and its public release on 11 October 2026.
The review was performed with AI assistance (large language models operating
the author's development environment) and is **not** independent peer review.
Nothing below should be read as external validation of the mathematics.

**Layout note (version 0.2.0):** sections 1–8 retain the paths and commands
used during the original review. The current manuscript is `paper/paper-v4.pdf`.
See [current reproduction commands](REPRODUCIBILITY.md) and the
[migration guide](MIGRATION.md) for the English layout.

The manuscript is `06-notas/paper/paper-v4.tex`; the compiled PDF is
`06-notas/paper/paper-v4.pdf`. Historical drafts (`paper-v1` to `paper-v3`,
`06-notas/t2-prova/draft-v0` to `draft-v3`) are kept unchanged as records.

## 1. Scope of the review

The review had four goals: (a) check the archived numerical evidence against
the manuscript without rerunning the two training campaigns; (b) run small,
independent numerical and symbolic checks of the central identities;
(c) correct scope statements that claimed more than the proofs or the data
support; (d) prepare the repository for public release (licensing, provenance,
security audit, reproducibility documentation).

It did **not** attempt a line-by-line re-derivation of the six lemmas and the
forcing theorem. The committed verification scripts in `06-notas/t2-prova/`
remain the only executable checks of the proofs, and they check selected
identities and rates, not every step.

## 2. Checks executed and their outcomes

All checks below were run on the release machine (the same Intel Core
i9-9900K used for the historical campaigns, Linux x86-64, CPU only,
single-threaded) with the pinned environment of `requirements.txt`
(Python 3.14.6, NumPy 2.4.2, SciPy 1.16.3, SymPy 1.14.0, Matplotlib 3.10.8);
the PyTorch check used torch 2.13.0 (a CUDA build running on CPU).

| Check | Command | Outcome |
|---|---|---|
| Integrity of the archived runs | `python verification/smoke.py` | PASS. All 1,800 per-run files (400 MLP confirmatory, 800 transformer confirmatory, 600 discarded-instrument runs) match their SHA256 manifests; both configuration hashes match. 4.6 s. |
| Frozen analysis recomputed | same command | PASS. Complete grids; the main denominators quoted in the manuscript (15/30 eligible returns, 13/16 area-first returns, 22/50 sustained own-channel returns, 20/100 vs 0/100 ReLU upstream contrast, 21/22 area-first trios, 0/100 validity control) and the Clopper–Pearson intervals are reproduced by the frozen analysis modules. |
| Two-neuron sweep | same command | PASS. 600 sampled instances, 57 retained under (H1)–(H2); median relative error of the release coefficient 2.68%. |
| Scalar quadrature identity | `python verification/check_entry_exit.py` | PASS (SymPy). The identity for dΨ(u)/dt including the other-margin remainder, see §3.1 below. |
| Jacobian-rank counterexample | same command | PASS. A zero unit block need not lower the rank of the full data Jacobian (motivates the rewording in §3.2). |
| Entry–exit coefficient, adaptive integrator | same command | PASS. Three archived instances (m = 4, 8, 16) integrated with DOP853 (rtol 2e-12, event interpolation) at β = 1e4 … 3e5; measured release coefficients converge to the predicted −1/(c R(t_exit)) (relative error 0.15%, 0.37%, 0.01% at β = 3e5). Report: `docs/verification/entry-exit-checks.json`. 2.6 s. |
| Training engines | `python verification/smoke.py --engines` | PASS. Bit-exact parity of the two MLP engines on five short cases; determinism of the transformer engine; frozen-ReLU null gives zero own-parameter movement. 8 s. |
| Lemma scripts (sample) | `06-notas/t2-prova/passo0_verificacao.py`, `prova_L1_verificacao.py` | PASS with the pinned environment (0.5 s and 44 s). The remaining lemma scripts were not rerun for the release. |
| Manuscript build | `pdflatex paper-v4.tex` (three passes) | Builds without errors or undefined references; 66 pages. |
| Secret scan | gitleaks 8.30.1 on the original history, the public history and the working tree | No findings. |

The two training campaigns (400 and 800 runs, 4.9 h and 13.9 h historically)
were **not** rerun. The historical dependency lockfile was not recorded, so
byte-exact regeneration cannot be promised; see `docs/REPRODUCIBILITY.md`.

## 3. Consequential changes to the manuscript

The leading statements (the entry–exit law, the O(log β/β) rate and the
first-order coefficient −1/(c R(t_exit)), the forcing theorem) are unchanged.
The changes below alter wording, scope or a proof detail. The pre-release text
is the committed `paper-v4.tex` at commit `b8bbd03`; `git diff b8bbd03 --
06-notas/paper/paper-v4.tex` shows every edit.

### 3.1 Mathematical corrections

- **Lemma 1 remainder (eq. `quadrature`, and its forcing form eq.
  `quadratureF`).** The other-margin contribution of hypothesis (H3) enters
  the margin equation additively as η_β with |η_β| ≤ C₃e^{−βδ₁}. Under the
  chain rule for Ψ(u) = log u + u it becomes −β(1+u)η_β, so the pointwise
  bound is C₃β(1+u)e^{−βδ₁}, not C₃βe^{−βδ₁} as previously stated. The fast-time
  equation now carries the term wη_β, and the multiplicative form is used
  only where R ≠ 0. A Grönwall estimate shows u = O(β) on the episode, so the
  accumulated error is O(β²e^{−βδ₁}), still below every algebraic order. The
  identity was checked symbolically (`verification/check_entry_exit.py`).
- **Lemma L5′, step (4.3) of the continuous induction.** The previous text
  argued that a factor-2 degradation of the constants is absorbed by a factor
  3/2. That comparison alone does not yield the strict improvement the
  induction needs. The step now uses the uniform smallness of the forcing
  error established in (4.2): by enlarging β₀ the window bounds can be taken
  arbitrarily close to the hybrid ones, which gives the constant 3/2·C₄.
  This argument was written during the review and has not been independently
  verified.
- Transversality of the area at t_hit and t_exit: the derivative of the area
  is R, not cR (typo in Appendix L1).

### 3.2 Scope statements

- **Rank drop.** The introduction and setup no longer state that a dead unit
  implies a rank drop of the full Jacobian J, or that it "sits on a stratum
  of the singular locus". A dead unit has a zero Jacobian block with respect
  to its own incoming parameters; other units may already span its output
  space. The law concerns that inactive block.
- **Branch selection, formerly "non-commutation".** Section 5 and
  Proposition `noncomm` compared a "field-first" and a "flow-first" limit and
  invoked the uniqueness theorem of Eberle, Jentzen, Riekert and Weiss. That
  theorem is proved for input distributions with a bounded density and for
  neurons with zero incoming weights and bias, and does not directly apply to
  the finite atomic dataset used here. The proposition now constructs an
  explicit absorbing hard-ReLU continuation (zero-derivative convention) and
  shows that the smooth-flow limit differs from it after exit; no uniqueness
  among nonsmooth continuations is asserted. The related-work paragraph was
  rewritten accordingly.
- **Transformer confirmatory.** "The theorem class transfers to the
  feed-forward block" was replaced throughout by the observational statement:
  own-channel returns are observed in the shallow-injection softplus arm
  (22/50 sustained returns, area first in 21/22). The pre-registered estimand
  keeps its original label, "transfer of the theorem class", in Table `e5`
  and in Appendix `estimands`, because the pre-registration wording is kept
  verbatim; the interpretive text no longer uses that phrase.
- **GELU "barrier".** The sign change of GELU′ at z* = −0.7518 is now
  described as a derivative sign change, not as a proven self-trapping
  barrier: the direction of the full margin dynamics also depends on the
  residual, the readout and the other data. The caption of Figure 7 and the
  figure's labels were changed to match, and the figure was regenerated from
  `figura_barreira_gelu.py`. The paragraph "The deep zero is what the channel
  law predicts" became "A channel-based heuristic for the deep zero": the
  factor-117 extrapolation is a heuristic and does not resolve the E-D
  confound.
- **Dead units and loss of plasticity.** The zero own-gradient of a hard-ReLU
  dead unit is described as one mechanism relevant to dormant units, not as
  the mechanism behind all observed loss of plasticity.
- **Denominators in the abstract.** The MLP count (13 of 16 returns) is now
  attributed to the primary softplus-β3 arm, and the transformer proportion is
  given with its denominator (22/50).
- **Conditioning of the two confirmatories.** The MLP confirmatory conditions
  on dynamical deaths; the transformer confirmatory conditions on deliberately
  injected born-dead units. The related-work paragraph on plasticity says so.
- **Limitations (iv) and (vi)** were reworded to match the above.

### 3.3 Reproducibility, data availability and LLM declaration

- The reproducibility section described a private repository and a policy of
  keeping per-run trace files out of version control. The public repository
  includes the 1,200 confirmatory per-run records (and the 600 records of the
  discarded instrument), so the section was rewritten: what is included, how
  the manifests were checked, why regenerated gzip bytes need not match, and
  that the historical lockfile is missing.
- The compute paragraph said "the only dependency is numpy"; the training
  engines require PyTorch. Historical timings are labelled as historical.
- The second paragraph of the LLM declaration claimed that the committed
  scripts re-derive each lemma "from scratch" and that the checks "test the
  law against the dynamics". It now states that the scripts check selected
  identities and rates, that LLM adversarial review is not independent
  validation, and that an AI-assisted publication review made the changes
  listed here.
- Front matter: "Independent research manuscript; not peer-reviewed." and the
  public-release revision date were added. The original date (26 July 2026)
  is kept.

## 4. Changes to code and repository layout

- Twelve experiment and figure scripts had absolute paths of the author's
  machine (`/home/caloi/Projects/j-space/...`) replaced by paths relative to
  the script's location. No numerical code changed. Historical logs and result
  markdown files that print such paths were left as they are.
- `figura_barreira_gelu.py`: docstring and two text labels changed (see §3.2);
  `fig7-barreira-gelu.pdf/.png` regenerated.
- New files: `README.md`, `LICENSE` (MIT, software), `LICENSES/CC-BY-4.0.txt`
  (manuscript and figures), `CITATION.cff`, `.gitignore`, `requirements.txt`,
  `requirements-experiments.txt`, `verification/` (three scripts),
  `docs/` (this file, `REPRODUCIBILITY.md`, provenance and verification
  records).
- The per-run result files (`*.jsonl.gz`, 1,800 files, about 35 MB) and three
  pilot result directories were untracked in the private repository and are
  now versioned, so that `verification/smoke.py` can check them.
- Portuguese file names, script names, experiment labels and internal notes
  were kept; renaming them would break the paths cited in the manuscript and
  in the commit history.

## 5. History and provenance

The public history was produced from the private research repository
(`j-space`) with `git filter-repo`, keeping only the paths listed in
`docs/provenance/included-paths.txt`. 99 commits were retained, from
23 July 2026 to 5 August 2026, including the pre-registration commits, the
negative results and the two reverts of 5 August. Commit dates and messages
are the originals, except that three personal e-mail addresses were replaced
by the author's GitHub no-reply address (in author fields and in message
bodies). `docs/provenance/commit-map.tsv` maps the original hashes, which
the manuscript cites, to their public counterparts. Original commit dates
establish order within the author's local history; they are not an
independently time-stamped registration.

## 6. Concerns identified and left open

1. The proofs were derived with LLM assistance and checked only by the
   author's committed scripts. No independent mathematical verification by a
   third party exists. The open step (i′) of the Limitations (the unquantified
   o(1) in the entry-corner lemma) remains open.
2. The rewritten step (4.3) of Lemma L5′ (§3.1) should be checked by a reader
   who works through Appendix B.
3. The transformer estimand E-C carries the pre-registered label "transfer of
   the theorem class" in the table and appendix while the text now reports it
   observationally. This is deliberate (the pre-registration is quoted
   verbatim) but readers should be aware of it.
4. Historical dependency versions are incomplete (PyTorch version not
   recorded). Regenerated runs may differ at floating-point level from the
   archived records; `verification/compare_runs.py` compares numerical content
   with explicit tolerances and ignores timing metadata.
5. The release checks used NumPy 2.4.2; the historical runs report NumPy 2.4
   without a patch version. The engine check used a CUDA build of PyTorch
   executing on CPU.
6. The confirmatories are small (MLP with at most 6 hidden units, transformer
   block with 4 feed-forward units and 2 heads). Nothing in this repository
   supports claims about production-scale networks.
7. Zero-byte `.bbl` files in `06-notas/paper/` are historical artifacts; the
   bibliography is inline (`thebibliography`), so no BibTeX step exists.

## 7. What was not done

- No training campaign was rerun; no new experiments were added.
- The literature notes, the third-party paper copies and the unrelated parts
  of the private research repository were not published.
- The text of the historical notes under `06-notas/` and the experiment READMEs
  (in Portuguese) was not edited or translated.

## 8. Delivery follow-up

A subsequent documentation review consulted the Claude session that completed
the public release, reused its recorded checks, and did not rerun the training
campaigns. It shortened the README, corrected the theorem references to
Theorems 2 and 6, and made the reproduction commands executable from the
repository root. The full-regeneration instructions now explicitly preserve
and move the archived results aside in a separate clone, because both runners
otherwise skip all existing records. The manuscript, experiment code,
configurations, results and version 0.1.0 PDF were unchanged by this follow-up.

The documentation follow-up is released as version 0.1.1. It also fixes an
unquoted colon in the citation title so that `CITATION.cff` parses as YAML,
and updates the software citation version. The manuscript PDF has the same
SHA256 as the version 0.1.0 release asset.

## 9. English organization and research navigation (version 0.2.0)

Directories, script entry-point filenames, study guides, and Markdown navigation
now use English. Historical Portuguese prose is represented by clearly labeled
English reading guides linking the complete original records at the exact
pre-migration commit. Mathematical derivations in the historical LaTeX sources,
saved data, configurations, and preregistration chronology remain available.

The research map connects the manuscript to dynamics, geometry, optimization,
plasticity, and evaluation methodology, distinguishing evidence levels. The
roadmap records proposed work rather than completed extensions. It does not
promote the trainable-readout or multiple-margin drafts to established results.

Code changes update filenames, imports, and paths. The current manuscript's
artifact references follow the new layout. The interactive toy has an English
interface, starts paused, and no longer describes noisy returns as verification
of an area-zero theorem. Its numerical update equations are unchanged.

The existing analysis-only smoke check passed after migration in approximately
3.5 seconds: all 1,800 manifested records, configuration hashes, main counts,
confidence intervals, and the sweep summary matched. An additional byte check
confirmed that 2,390 data, configuration, log, and PNG artifacts were unchanged.
All 99 Python files parsed successfully. No training campaign or new proof review
was performed for this reorganization.

The current PDF was rebuilt to 66 pages with no overfull boxes or unresolved
references. Representative changed pages were visually inspected. One long
inline equation was displayed on its own line without changing its content.
JavaScript syntax checks passed for the English toy. An interactive browser
was unavailable in the preparation environment, so a browser rendering check
of that HTML file was not performed.

## 10. Maintenance and automated checks (version 0.2.1)

The manuscript, README, and citation retain Leonardo Caloi Santos as author.
The project-scoped Claude Code setting `attribution.commit` is now empty to
disable the automatic coauthor trailer in future commits, following the
[Claude Code settings reference](https://code.claude.com/docs/en/settings-reference#attribution-commit).
The existing AI-assistance disclosure and all previous commit messages remain
intact. The local Git author uses the account-associated GitHub noreply address;
the public remote and default branch remain unchanged.

The README now presents the research question, contributions and hypotheses,
mechanism, results, and quick reproduction before the broader research map.
The new GitHub Actions workflow runs the two documented quick checks in a clean
environment on pushes to `main` and pull requests. It does not run training or
the historical full-campaign wrappers. Actions are pinned to commit hashes,
and the workflow has read-only repository permissions.

Inspection corrected the earlier description of auxiliary dependencies:
`review14.py` used its old scratchpad path as an output directory, and
`review15_o5b.py` contained an unused import search path. The required modules
and input records are already in the repository. The obsolete paths have been
removed; `review14.py` and `generate_artifacts.py` write regenerated reports to
`build/affine-forcing/` instead of the scratchpad or archived evidence. Their
numerical routines, configurations, and saved results are unchanged. These
historical campaigns remain outside the quick verification path and were not
rerun for this release.

The manuscript PDF and LaTeX source are unchanged from version 0.2.0. No proof,
scientific conclusion, archived result, preregistration commit, or provenance
map was revised. Version 0.2.1 is a separate release; previous releases and
tags are preserved. Its release notes link the actual automated check result.

The [first automated run](https://github.com/leonardocaloi/neural-network-entry-exit-law/actions/runs/38114297450)
passed both quick checks and the tracked-file integrity check in a 34-second
job, including setup. The README badge was added only after this successful
execution. Auxiliary import and output-routing checks, documentation links,
citation metadata, and a staged-change secret scan also passed locally.
