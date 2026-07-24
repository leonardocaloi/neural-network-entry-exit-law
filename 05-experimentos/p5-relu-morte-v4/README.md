# p5-relu-morte-v4 — o resíduo-em-T: geometria 2D ou artefato?

**STATUS: RASCUNHO — AGUARDANDO PREDITOR E REVISÃO ADVERSARIAL. NÃO RODAR.**

## Hipótese

O v3 deixou resíduo de p_captura +0.058 CRESCENTE com T (até +0.18 em T=3e-3),
com back-reaction confirmada nos canais limpos (P-3' pareado z=0.42; tardias
CI∋0). **[CONJECTURA v4]** O resíduo é GEOMETRIA: o modelo 1D projeta a excursão
na normal de UMA face; em T alto, o passeio 2D de p_A migra entre faces e
compõe rotas de captura que o 1D não conta. Modelo 4.0 = dinâmica 2D exata de
p_A (BM no cone; fora, empurrões das faces ativas) acoplada aos modos de B —
remove a simplificação 1D, sem parâmetro livre novo.

**Competidor-artefato declarado:** o passo grosso COARSE=100 do preditor pode
degradar em T alto. Controle: predições 4.0 com COARSE ∈ {100, 20} têm de
concordar dentro do erro de MC (senão, artefato primeiro).

**Escada de ablação (atribuição):** 1.0 (det) → 3.0 (1D+coupling) → 4.0
(2D+coupling). O que o 4.0 fechar é geometria; o que sobrar é mecanismo ainda
não modelado (registrável como aberto).

## Predições (congelar via preditor_v4.py)

- **P-1'' (MATA binário, incerteza genuína):** p_captura por célula do 4.0.
  Como o 4.0 é quase o sistema completo (únicas diferenças: B linear-modal
  [exato sob H4] e gate relu no lugar de softplus-fino), a predição DEVE fechar
  — falhar aqui indicaria erro de régua ou física fora do reduzido (ex.:
  clipping de B). **MATA:** ≥3 células com |z|>1.96 OU ≥8 mesmo sinal
  (critérios do v1, agora com incerteza real: o 4.0 pode passar).
- **P-2'' (estimação):** decomposição do gap por degrau da escada (1D→2D por
  célula), com CI.
- Réguas idênticas ao v2/v3 (congeladas). Seeds virgens: 1500–1649.

## Números congelados (preditor 4.0, M=6000, paralelo)

4.0 crava as observações do v3 (resíduo-em-T eliminado; ex.: T=3e-3 d0=0.020:
previsto 0.910 vs observado-v3 0.913); controle COARSE=20 difere ≤0.004
(artefato descartado). Resíduo restante ~−0.016 (leve SOBREprevisão em 2–3
células) — candidato dominante: B-clipping (preditor propaga B linear-modal; a confirmatória usa relu-duro idêntico ao gate do preditor — não há softplus em nenhum braço; 5º parecer) + ≤0.004 de bloco. **Pergunta
explícita ao revisor:** com viés residual ~−0.016, o teste de sinais (≥8/9) de
P-1'' pode disparar por viés-de-fio-de-cabelo — avaliar piso |Δp| (parecer v2,
item M2) ou manter como está; decidir ANTES do commit.

## Pendências
1. preditor_v4.py (+ controle COARSE) → config_v4.json. 2. Revisão adversarial.
3. Commit = pré-registro. 4. Run (reuso do experimento.py do v3, sem braço
congelado). 5. Nota.
