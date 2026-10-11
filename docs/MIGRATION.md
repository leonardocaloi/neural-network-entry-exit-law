# Version 0.2.0: English organization and research navigation

This release reorganizes the existing project for an international audience.
It does not introduce a new research result or rerun a training campaign.

| Previous directory | Current directory |
|---|---|
| `05-experimentos/` | `experiments/` |
| `06-notas/paper/` | `paper/` |
| `06-notas/t2-prova/` | `theory/` |
| Dated notes in `06-notas/` | `docs/research-notes/` |
| `06-notas/paper/padrao/` | `docs/editorial/` |
| `06-pranchas/` | `visualizations/` |

Nested directories and script filenames also use English names. For example,
the confirmatory engines are `engine_4a3.py` and `engine_e5_confirmatory.py`;
their analyses are `analyze_4a3.py` and `analyze_e5.py`. The complete
[path mapping](provenance/path-map-v0.2.0.tsv) covers moved files.

## Historical records and interfaces

English study guides and historical reading guides replace Portuguese prose
in the current Markdown navigation. They are explicitly identified as guides,
not verbatim translations. Each historical guide links the complete original
at commit `a331371220b539a467a7302eac8ab580740b5014`, which remains available
through version 0.1.1 and Git history. No original preregistration was rewritten.

Saved data, configurations, manifests, and historical logs retain their exact
bytes. Internal Python function names, CLI flags, dictionary keys, and frozen
event identifiers remain stable. Existing Portuguese console output and code
comments can therefore still appear when running historical implementations.
The English study guides explain the public interpretation and supported commands.

Python module imports, paths, manuscript artifact references, and reproduction
commands follow the new layout. The current PDF is rebuilt for those reference
changes; historical PDFs retain their original bytes and historical paths.

The historical review commands in `REVIEW.md` describe the layout in which they
were executed. Use `REPRODUCIBILITY.md` for current commands. The original
publication inclusion list and commit map retain their historical meaning.

## Scope of the visual explainer

The existing interactive toy is translated into English and starts paused.
Its descriptions now identify it as a stochastic illustration: a noisy ReLU
return does not verify the deterministic softplus theorem, and the display
does not assert that a return proves the residual area crossed zero.
The simulation's update equations are unchanged.

## Validation

The reorganization is checked through saved-record integrity, the existing
analysis-only smoke path, Python syntax and local imports, documentation links,
and a rebuild of the current manuscript. These are migration checks, not a new
scientific validation campaign. Actual outcomes are recorded in the release notes.
