# NEGATIVOS — registro sem dó

Regra da casa: negativas são resultado; registrar, nunca apagar, nunca
reescrever a hipótese antiga.

## 2026-07-24 — etapa-4a (exp05): INVÁLIDO por P-NULL (instrumento)

Pré-registro 6411ac8; 600/600 runs; análise congelada sem decisão pós-hoc.
O braço de difusão pura passou no teste calibrado (m=16, p=0.002) ⟹ pela regra
pré-registrada, instrumento viciado e NENHUM veredito de P-A/P-B/P-D vale.
Diagnóstico: a régua mede chatter de fronteira (59% dos episódios ≤5 passos;
ReLU-dura "ressuscita" igual a softplus em T>0 — assinatura de ruído, não de
gradiente); a física do teorema vive em T→0 (episódios ~1e3 passos, 99.6%
censurados em 1e5). Nulo calibrado segurou FPR≈α em mundos iid, mas o mundo
real tem heterogeneidade por neurônio (não-iid). Nota completa:
`06-notas/2026-07-24-etapa4a-invalido-instrumento.md`. Herança para 4a.2:
debounce/histerese + nulo estratificado por neurônio + T-sweep para baixo com
horizonte ≥5e5.

## Anteriores (registrados nas notas)

- 2026-07-24 — p5-relu-morte-v2: FALSIFICADO (cronograma determinístico; janela
  reabre por flutuação) — `06-notas/2026-07-24-v2-falsificacao-modelo-corrida.md`
- 2026-07-24 — modelo 2.0 (flutuação-pura): refutado no preditor, não rodado —
  `06-notas/2026-07-24-v3-resultado-mecanismo.md`
- 2026-07-23/24 — v1 e p5-cruz: reprovados no desenho (decidíveis a priori /
  mecanismo cancela na ordem adiabática); v1 nunca rodado, p5-cruz arquivado.
