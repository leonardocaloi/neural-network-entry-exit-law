# Parecer do referee adversarial — draft-v2 (2026-07-24) e aplicação (v2.1)

**Veredito original: NÃO APTO — a um ciclo de edição do APTO.** 2 bloqueadores
pontuais + lista de 8. Os 6 scripts de prova re-executados pelo referee: 6/6
passam. Universalidade do coeficiente re-derivada e confirmada. DAG de
dependências L1→{L3,L4}→L5→L6 sem circularidade real (ciclo de CITAÇÃO L1↔L4
benigno — argumento em duas passadas, corr≥0 primeiro).

## Bloqueadores → APLICADOS no v2.1

- **B1** (taxa do resto do release: O(logβ/β²) dito, o(β⁻¹) provado — o o(1)
  do L3 não quantificado contamina): teorema enfraquecido para o(β⁻¹), com o
  refinamento via δ_z = logβ/β declarado como caminho. ✔
- **B2** (fator c espúrio na perturbação do Lemma 4 — eco do erro do v0):
  removido; cota = δ(t_exit−t_hit)/R_exit. ✔

## Lista aplicada (3–8)

Ciclo L1↔L4 desfeito no texto (duas passadas explícitas) ✔; taxa refinada do
Lemma 2 condicionada à camada (fornecida por L5) ✔; bootstrap do peso do L3
escrito por extenso (rota monótona, sem ponto fixo exponencial) ✔; colar de
entrada O(1/β) em P_s(b) do L5 ✔; β₀ rebaixado ("rastreável, não exibido") e
M₂ com C4 empírico (440.8) vs teórico (≤1233) declarados, a₀≈0.02 pinado ✔;
numerics honestos (9.5% exibido; gluing 0.205–0.227; proveniência do
0.888→0.982; sweep rotulado como re-run de grade fina; corolário vs −14.68 a
2.1%; vacuidade do check de camada para β≤3000) ✔; bibitem GH = preprint 2019
conforme ficha ✔.

## Estado

**APTO-PARA-CIRCULAR nos termos do referee** ("com 1–2 aplicados, APTO" — foram
aplicados os 8). Compila limpo. Refinamentos declarados (não bloqueantes):
quantificar o o(1) do L3 (→ resto O(logβ/β²)); exibir β₀; segunda instância.
As decisões de circulação (OSF, endorsement, onde/quando) são do pesquisador.
