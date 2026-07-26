# e7 — piloto EXPLORATÓRIO: o que o Adam faz com o portão de memória

Status: **[EXPLORATÓRIO]**. Escrito ANTES de rodar (hipóteses e critérios de morte
pré-registrados nesta seção). Data: 2026-07-26.

## Motivação

Um revisor externo afirmou, **sem medir**, que "Adam destrói a assintótica" do
teorema T2′ (lei de entrada-saída: o neurônio morto retorna quando a área do
resíduo volta a zero, `∫_{t_hit}^{t} R dt = 0`, com release em
`t_exit^hyb + O(1/β)`). A casa não assume: mede. E há hipótese nova a testar —
Adam não necessariamente destrói a lei, pode **trocar o relógio que a lei lê**.

## Setup

- Instância de referência do paper: `X=[0.5,1.5]`, `Y=[0.5,1.0]`,
  `PA0=(-0.5,0.27)`, `PB0=(1.144,0.261)`, `FACE=0`, `β=1e4`, `H=1.8`, `dt=1e-5`
  (idênticos a `06-notas/t2-prova/passo4_verificacao.py`).
- Mais 20 instâncias do sweep `05-experimentos/t2-sweep-instancias` que passaram
  (H1)–(H4) — para não concluir de 1.
- Passo GD do sistema COMPLETO substituído por **Adam de verdade**
  (β1=0.9, β2=0.999, bias-correction, aplicado a TODO θ = (w_A,b_A,w_B,b_B),
  não híbrido).
- `lr` calibrado por bisseção em log(lr) para que `t_hit^Adam ≈ t_hit^GD` na
  instância de referência (a fase VIVA reproduz o timescale do GD). O mesmo
  procedimento de calibração é repetido para CADA ε de H-EPS, senão a transição
  ficaria confundida com mudança de timescale.
- Harness de detecção idêntico ao do sweep: morte = `z_A(x_face) ≤ 0` com
  `t > 0.05`; release = `z_A(x_face) ≥ 0` depois da morte; `n_mortes` contado.
- Séries registradas (subamostradas) desta vez — lição do C3: `z_A(x_face)`,
  `R(t) = Y_face − softplus(z_B(x_face))`, `∫R dt`, `∫sign(R) dt`.
  Os preditores (i) e (ii) são calculados na resolução cheia, DENTRO do loop,
  a partir da MESMA série R da corrida Adam (isolando "qual relógio o portão lê").

## Hipóteses (pré-escritas) e o que MATA cada uma

### H-SIGN — o portão vira lei de área do SINAL
Sob Adam, o update é `lr·m̂/(√v̂+ε)`. Com o gradiente do morto minúsculo mas de
sinal consistente, `m̂/√v̂ → ±1`: o update satura em `≈ lr·sign(g)`. A dinâmica
do morto integra o **sinal** do resíduo, e o portão vira
`∫_{t_hit}^{t} sign(R) dt = 0` (equivalente: |desvio acumulado|/lr).

Predição operacional: `t_ret ≈ t_pred_sign`, o primeiro instante em que
`∫sign(R)dt` retorna a zero. Como `sign(R)` é −1 e depois +1, isso é
`t_ret = 2·t_zeroR − t_hit`, sem nenhum parâmetro livre.

**MATA H-SIGN:** erro relativo de `t_pred_sign` (medido sobre a duração morta
`t_ret − t_hit`) pior que ~poucos % — concretamente, se a mediana do erro de
H-SIGN não for melhor que a do preditor de área E não estiver abaixo de 5%,
H-SIGN está refutada.

### H-EPS — existe transição controlada por ε
Quando `|g| ≪ ε`, `m̂/(√v̂+ε) ≈ g/ε`: o update volta a ser LINEAR no gradiente e
a lei ORIGINAL `∫R dt = 0` reaparece, com o relógio reescalado por 1/ε.
Varredura `ε ∈ {1e-8, 1e-4, 1e-2, 1}`, resto fixo (lr recalibrado por ε).

**MATA H-EPS:** se o erro do preditor de área não cair monotonicamente (nem
sequer no ε maior) e o de sinal não subir — isto é, se não houver transição
mensurável entre os dois regimes ao longo do ε varrido.

### H-NULL — nenhuma lei de primeira ordem sobrevive
O retorno sob Adam não é predito nem por `∫R` nem por `∫sign(R)` (por exemplo:
o momento β1 domina e cria overshoot; ou o morto não retorna dentro de H, ou
retorna múltiplas vezes). **Também é resultado** e é registrado como tal.

**H-NULL vence** se, no ε default, ambos os preditores tiverem erro mediano
> 20%, ou se a fração de instâncias com release limpo (`n_mortes = 1` e release
dentro de H) cair abaixo de 50%.

## Desvios (preenchidos DEPOIS da execução)

1. Horizonte estendido de H=1.8 (do GD) para **H=8.0**: sob Adam a fase morta dura
   ~3.6 (vs 0.72 no GD) e H=1.8 reportaria "não volta" por truncamento. β mantido
   em 1e4, dt mantido em 1e-5 (não foi preciso reduzir β).
2. Relógios passaram a ser integrados **também depois do release** (continuação
   contrafactual), senão um relógio que zera logo após o release conta como "não
   disparou" — artefato que a primeira passada produziu em ε=1, onde a lei de área
   está praticamente exata (|∫R| = 3e-4 no release).
3. 12 instâncias do sweep em vez de 20 (custo de CPU); N=13 com a de referência.
4. Métricas descritivas acrescentadas durante a execução (não usadas em nenhum
   critério de morte pré-registrado): `frac_R`/`frac_S`, `β·z_park`, `|m̂/(√v̂+ε)|`.

Resultado e veredito por hipótese: `06-notas/2026-07-26-e7-piloto-adam.md`.
Execução em estágios resumíveis: `python e7_adam.py calib`, depois
`python e7_adam.py run <eps>` para cada ε, depois `python e7_adam.py serie`.

## Como rodar
```
source venv/bin/activate
cd 05-experimentos/e7-adam-portao && nice -19 python e7_adam.py
```
Saídas: `results/*.jsonl`, `results/serie_*.npz`, `fig-e7-relogios.{pdf,png}`.
