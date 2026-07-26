# e7 — o que o Adam faz com o portão de memória (piloto)

Data: 2026-07-26 · Status: **[EXPLORATÓRIO]** · N = 13 instâncias (a de
referência do paper + 12 do sweep que passaram H1–H4) × 4 valores de ε.
Código: `05-experimentos/e7-adam-portao/` (hipóteses pré-escritas em
`README-e7.md` ANTES da execução). Nada aqui entra no paper como está.

## TL;DR

O revisor externo disse, sem medir, que "Adam destrói a assintótica". **Medido: não
destrói.** O portão continua existindo — o neurônio morre, estaciona atrás de uma
barreira e volta — mas o **relógio que o portão lê muda com ε**, e o mecanismo pelo
qual muda **não é o que H-SIGN previa**.

Três achados, em ordem de robustez:

1. **[EXPLORATÓRIO]** Há uma **transição limpa e monotônica controlada por ε**
   (H-EPS **CONFIRMADA**): o erro da lei ORIGINAL `∫R dt = 0` cai
   12,3% → 5,7% → 2,1% → 0,24% (mediana) para ε = 1e−8, 1e−4, 1e−2, 1; e a lei do
   sinal faz o caminho inverso, 4,8% → 5,9% → 3,1% → 13,1%. O sinal-de-troca é
   nítido no teste pareado: `sign` vence `area` em **10/10, 8/10, 5/10, 0/13**
   instâncias ao longo do grid de ε.
2. **[EXPLORATÓRIO]** A profundidade de estacionamento do morto obedece a
   `β·z_park ≈ −0,94·ln(1/ε) − 6,1` (resíduo ≤ 0,36 nos 4 pontos; GD dá
   `β·z_park = −4,96`). Ou seja: **ε desloca a barreira em ln(1/ε)/β**. O portão
   não é destruído, é **aprofundado** — em unidades de 1/β, e log em ε.
3. **[EXPLORATÓRIO]** O Adam **não satura em sign(g)** na fase morta:
   `|m̂/(√v̂+ε)|` mediano é **3,6e−4**, quatro ordens abaixo de 1. Motivo medido:
   o gradiente do morto decai exponencialmente rápido (o `z` mergulha atrás da
   barreira), e a memória de `v` (β₂ = 0,999 ≈ 10³ passos) é *mais longa* que a de
   `m` — então `m̂/√v̂` colapsa em vez de saturar. **A premissa mecânica de H-SIGN
   é falsa**, embora sua predição de tempo seja a melhor disponível em ε pequeno.

Nada disso muda T2′ como está enunciado (GD): a lei de área continua exata no
baseline GD das mesmas 13 instâncias — **erro mediano 0,017%, p90 0,032%**, contra
11,7% da lei do sinal. O discriminador funciona; é a física que muda sob Adam.

## Setup

Instância de referência do paper (`X=[0.5,1.5]`, `Y=[0.5,1.0]`,
`PA0=(−0.5,0.27)`, `PB0=(1.144,0.261)`), β = 1e4, dt = 1e−5, FACE = 0, motor e
harness de detecção (morte, t_exit híbrido, release, `n_mortes`) copiados de
`05-experimentos/t2-sweep-instancias/sweep_instancias.py`. Única troca: o passo GD
do sistema COMPLETO vira **Adam de verdade** (β₁ = 0,9, β₂ = 0,999, bias-correction,
todo θ = (w_A,b_A,w_B,b_B)) — não híbrido.

`lr` calibrado por bisseção em log(lr) para casar `t_hit^Adam = t_hit^GD = 0,1121`
na referência, **recalibrado para cada ε** (senão a transição de H-EPS ficaria
confundida com mudança de timescale). Resultado da calibração:
lr = 1,20e−6 (ε = 1e−8), 1,21e−6 (1e−4), 1,30e−6 (1e−2), 1,07e−5 (1). Que lr mal
mude até ε = 1e−2 é esperado e é um controle: na fase VIVA |g| ~ 1e−1 ≫ ε, então ε
não entra; ε só age na fase MORTA. Em ε = 1, lr ≈ dt e o Adam degenera em GD.

Métricas (as séries foram guardadas desta vez — lição do C3):
- `err_area`, `err_sign`: |t_pred − t_release| / (t_release − t_hit), com t_pred =
  primeiro zero de `∫R dt` e de `∫sign(R) dt` a partir de t_hit. Os relógios seguem
  sendo integrados **depois** do release (continuação contrafactual) — sem isso, um
  relógio que zera logo após o release contaria falsamente como "não disparou", que
  foi exatamente o artefato da primeira passada.
- `frac_R`, `frac_S`: fração do relógio ainda não cumprida no instante do release
  (`|a_rel| / |min a|`); 0 = a lei bate exatamente.

## Números (mediana sobre as instâncias com release limpo)

| ε | N c/ release | err_area | err_sign | frac_R | frac_S | β·z_park | \|m̂/(√v̂+ε)\| | sign vence |
|---|---|---|---|---|---|---|---|---|
| 1e−8 | 10/13 | 12,3% (p90 23,6) | **4,8%** (p90 16,9) | 0,313 | 0,091 | −23,6 | 3,6e−4 | 10/10 |
| 1e−4 | 10/13 | 5,7% | 5,9% | 0,146 | 0,117 | −14,6 | 3,0e−4 | 8/10 |
| 1e−2 | 10/13 | **2,1%** | 3,1% | 0,060 | 0,063 | −10,1 | 2,7e−4 | 5/10 |
| 1e0 | 13/13 | **0,24%** (p90 0,55) | 13,1% | 0,006 | 0,302 | −6,4 | 1,0e−4 | 0/13 |
| GD (baseline) | 13/13 | **0,017%** (p90 0,032) | 11,7% | 0,0006 | 0,266 | −5,0 | — | 0/13 |

Duração mediana da fase morta: 3,6–4,0 sob Adam com ε pequeno, 1,2 em ε = 1,
**0,72 sob GD** — o Adam prolonga a fase morta em ~5×, coerente com a barreira mais
profunda. As 3 instâncias sem release em ε ≤ 1e−2 (seeds 20260792, 20260876,
20260890) são **truncamento de horizonte**, não ausência de retorno: em ε = 1 todas
as 13 voltam.

## Veredito POR HIPÓTESE

**H-SIGN — SOBREVIVE COMO APROXIMAÇÃO, REFUTADA COMO MECANISMO.**
O critério pré-registrado ("mediana < 5% E melhor que a lei de área") passa por
pouco em ε = 1e−8: mediana 4,8%, e vence a lei de área em **10/10** instâncias
(pareado). Mas: (i) p90 = 16,9%, máx 17,1% — isso não é uma lei no padrão de T2′
(0,017% sob GD), é um preditor grosseiro; (ii) o **mecanismo postulado é falso** —
o update não satura em `lr·sign(g)`, ele é ~1e−4 do valor saturado, porque a
memória de `v` (10³ passos) sobrevive ao colapso exponencial de `|g|` do morto.
H-SIGN acerta o *ranking* pela razão errada. Um preditor que funciona 10/10 vezes
com 5% de erro merece o confirmatório; a explicação por saturação de sinal, não.

**H-EPS — CONFIRMADA.**
A transição existe, é monotônica nas quatro métricas independentes (err_area,
frac_R, err_sign, β·z_park) e no teste pareado (10/10 → 0/13). Em ε = 1 a lei
ORIGINAL `∫R dt = 0` reaparece com 0,24% de erro mediano, a um fator ~14 do GD
puro. O relógio de fato reescala: a fase morta encurta de 3,6 para 1,2 quando ε vai
de 1e−8 a 1. O bônus não previsto na hipótese é a **lei quantitativa da barreira**,
`β·z_park ≈ −0,94·ln(1/ε) − 6,1`, que é o que amarra a transição a um mecanismo.

**H-NULL — REFUTADA.**
Sobrevive sim uma lei de primeira ordem: em todo ε algum dos dois relógios prediz o
retorno com erro mediano ≤ 5%, e em ε = 1 com 0,24%. Não houve overshoot por
momento, nem múltiplas mortes (`n_mortes = 1` em todas), nem release espúrio. O
único modo de falha observado foi **truncamento de horizonte** em 3/13 instâncias
com ε pequeno, e mesmo essas estavam com o relógio de sinal a caminho de zerar.

## Desvios do pré-registro (todos declarados)

1. **Horizonte estendido de H = 1,8 para H = 8,0.** H = 1,8 era o horizonte do GD
   (release em 1,13 na referência). Sob Adam com ε = 1e−8 a fase morta dura ~3,6, e
   manter H = 1,8 reportaria "não volta" por artefato de truncamento. β foi mantido
   em 1e4 (não foi preciso reduzir); dt = 1e−5 idem.
2. **Relógios integrados após o release** (ver Métricas). A primeira passada usava
   "primeiro zero antes do release" e registrava `n_area = 0` em todos os ε —
   inclusive ε = 1, onde `|∫R|` no release é 3e−4, isto é, a lei praticamente
   exata. Era artefato do critério, corrigido antes de qualquer leitura de veredito.
3. **12 instâncias do sweep em vez de 20**, por custo de CPU (fase morta 5× mais
   longa sob Adam). N = 13 no total, com a de referência.
4. Métricas `frac_R`/`frac_S`, `β·z_park` e `|m̂/(√v̂+ε)|` foram acrescentadas
   durante a execução (não estavam no pré-registro). São **descritivas**, não
   entram em nenhum critério de morte pré-registrado; o veredito de H-SIGN e H-EPS
   usa `err_area`/`err_sign`, que estavam.

## O que um CONFIRMATÓRIO exigiria

- **Pré-registro novo** (hipótese nova = experimento novo): a hipótese confirmatória
  não é mais H-SIGN, é **H-BARREIRA**: `β·z_park = a·ln(1/ε) + b` com `a = 1`
  predito a priori, e o retorno governado pela lei de área com a barreira
  deslocada. Predição falsificável: `a ∉ [0,9; 1,1]` mata.
- N ≥ 100 instâncias do mesmo sorteador (o sweep tem 57 que passam H1–H4; ampliar o
  sorteio), grid de ε com ≥ 6 pontos, e um segundo β (1e3 e 1e4) para separar o que
  é 1/β do que é ε.
- **Baseline nulo**: GD com o mesmo lr efetivo, e Adam com β₁ = 0 (sem momento) e
  com β₂ ∈ {0,9; 0,999} — a explicação medida atribui o colapso de `m̂/√v̂` à
  memória de `v`, então variar β₂ é o teste direto do mecanismo, e é o que falta.
- Instrumentar a razão de Adam com resolução temporal (aqui só guardei a mediana e
  a série da referência).
- Derivação analítica da profundidade de estacionamento (balanço
  `ż = −(lr/dt)·ratio(ż)` com `|g| ∝ e^{βz}`), verificada por SymPy/numérico antes
  de virar afirmação.

## Arquivos

- `05-experimentos/e7-adam-portao/README-e7.md` — pré-registro
- `05-experimentos/e7-adam-portao/e7_adam.py` — motor (estágios `calib`/`run`/`serie`)
- `05-experimentos/e7-adam-portao/results/e7_runs.jsonl` — 52 registros por instância×ε
- `05-experimentos/e7-adam-portao/results/serie_ref.npz` — séries (z_A, R, ∫R, ∫sign R, razão de Adam)
- `05-experimentos/e7-adam-portao/fig-e7-relogios.{pdf,png}` — GD vs Adam, os dois relógios
