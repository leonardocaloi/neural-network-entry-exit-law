# Transformer scoping: two return channels

**Status:** Exploratory study.

The scoping work motivates distinguishing own-parameter changes from upstream changes through attention. It also explores the GELU derivative sign change and injection depth. These observations informed the later controls; they do not prove a universal GELU barrier or a direct transfer of the manuscript theorem to transformers.

## Source and scope

This is an English reading guide prepared for version 0.2.0, not a verbatim
translation or a new preregistration. The complete
[original research record](https://github.com/leonardocaloi/neural-network-entry-exit-law/blob/a331371220b539a467a7302eac8ab580740b5014/06-notas/2026-07-25-etapa5-scoping-transformer.md) remains available at the exact
pre-reorganization commit, with its original wording, equations, tables, and
historical identifiers. Its original language is Portuguese.

For current scientific claims, use the
[manuscript](../../paper/paper-v4.pdf) and
[publication review](../REVIEW.md). The
[research map](../RESEARCH-MAP.md) explains the connections to other
fields; the [roadmap](../ROADMAP.md) separates unfinished work
from released results.
