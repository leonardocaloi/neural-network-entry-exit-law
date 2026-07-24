# Resultado do p5-relu-morte-v3: mecanismo atribuído, com resíduo que escala com T

**Data: 2026-07-24. Pré-registro: fd58ccd (3º parecer aplicado). Run com seeds
virgens (1000–1449), análise 100% pré-registrada, zero desvios.**

## Vereditos

- **P-3' (contraste pareado, o único MATA binário): PASSA** — pooled observado
  0.0481 vs previsto 0.0457, z = 0.42 (1350 pares). O canal que o freeze mata
  (reabertura pós-freeze) está **exatamente** como o modelo 3.0 prevê. O desenho
  pareado do 3º parecer entregou o que prometia. [EVIDÊNCIA NUMÉRICA]
- **P-2' (resíduo de taxa-tardia): CI [−0.016, +0.018] CONTÉM ZERO** (médio
  −0.001) — as capturas tardias em dados virgens batem com o 3.0. Nota
  exploratória: a taxa velha (12.9%, seeds 0–149) ficou acima da nova (~8%);
  padrão por célula sugere dependência de T e/ou flutuação entre lotes de seeds
  — registrado, não interpretado.
- **P-1' (resíduo de p_captura): 3.0-INSUFICIENTE, como pré-declarado** — médio
  +0.058, CI [+0.030, +0.093]. **A estrutura é a notícia**: o resíduo CRESCE com
  a temperatura (T=3e-4: +0.02/+0.00/+0.04; T=1e-3: +0.02/+0.05/+0.07;
  T=3e-3: +0.05/+0.09/+0.18). O que falta modelar escala com o ruído.

## Leitura de mecanismo (o arco completo)

| Mecanismo | Evidência |
|---|---|
| Cronograma determinístico (v1) | falsificado (v2) |
| Flutuação-apenas (2.0) | refutado no preditor (smear preserva média) |
| **Back-reaction (3.0)** | **carrega o grosso**: canal pós-freeze exato (P-3'),
  tardias exatas (P-2'), e ~52% do gap em p_captura |
| Resíduo restante | positivo, **crescente com T** — candidatos: migração de
  face 2D (dirigida por ruído), excursões repetidas compondo back-reaction além
  do 1D, aproximação de bloco degradando em T alto. Hipótese do próximo ciclo. |

## O que NÃO se conclui

Não se conclui que o 3.0 é "o modelo certo" (P-1' o rejeita como suficiente);
não se conclui mecanismo do resíduo (só sua assinatura em T). Instrumento:
censuras 1–10%, todas as células válidas.

## Artefatos

results/runs.jsonl (~5850 runs), analise.json, run-confirmatoria-log.txt.
