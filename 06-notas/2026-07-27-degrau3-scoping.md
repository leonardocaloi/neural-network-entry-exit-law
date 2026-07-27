# Degrau 3 — scoping matemático do teorema MULTI-DADO

**Data:** 2026-07-27 · **Status:** scoping, NÃO commitado, paper NÃO editado
**Fontes de contexto:** `06-notas/paper/paper-v4.tex` (§2, §3.1, Ap. A/B),
`06-notas/t2-prova/teorema-degrau2-draft.tex`,
`05-experimentos/etapa4-mlp-populacao/pilotos-4a2/pressao/taxa/` (motor + results-4a3 + c3-posthoc.md)
**Scripts deste scoping** (scratchpad, não versionados):
`verifica_identidade.py`, `toy_multidado.py`, `toy_transicao.py`, `replay_lider.py`,
`analisa_replay.py`, `gate_vetorial.py`

---

## TL;DR

1. **A estrutura multi-dado é uma identidade exata, não uma conjectura.** [ESTABELECIDO — derivado
   e verificado numericamente aqui] O neurônio morto com todas as faces ativas obedece
   `ż = C[(R − g_β(z)) ⊙ σ(βz)]`, com `C = XXᵀ/n` a Gram PSD dos dados aumentados, de **posto
   d+1**. Em `θ_A` isso é um fluxo de **ascensão de gradiente** sobre o potencial de soma de
   exponenciais `V = (1/nβ)Σ_i R_i(t)e^{βz_i}`; nas variáveis de canal `q_i = e^{βz_i}` é um
   **sistema Lotka–Volterra não-autônomo** `q̇_i = βq_i Σ_k C_{ik}R_k(t)q_k` com matriz de
   comunidade simétrica PSD.

2. **A quadratura Ψ do L1 sobrevive, mas com um termo de acoplamento explícito.** [ESTABELECIDO
   — verificado com derivada numérica, erro relativo 1e-8]
   ```
   dΨ(u_i)/dt = −βC_ii(R_i − g_β(z_i)) − β Σ_{k≠i} C_{ik} · (1+u_i)/(1+u_k) · (R_k − g_β(z_k))
   ```
   com `Ψ(u)=log u + u`, `u_i=e^{−βz_i}`. O peso do acoplamento é **exatamente** `(1+u_i)/(1+u_k)`,
   cujo limite de park profundo é `e^{β(z_k − z_i)}` — a forma que o pedido chutou, aqui **derivada**.

3. **Corolário do líder estável (grátis).** Para `i* = argmax z_i`, todos os pesos `e^{β(z_k−z_{i*})}`
   são exponencialmente pequenos: a equação de `Ψ_{i*}` **desacopla** e o teorema escalar do paper
   vale com `R_{i*}`, com erro `O(β e^{−βδ_gap})`, `δ_gap = z_{i*} − z_{(2)}`. Isto **identifica a
   hipótese (H3)/(H4′) do paper como exatamente uma hipótese de folga do líder**, e diz qual é o
   parâmetro: **β·δ_gap**, não β sozinho.

4. **Quadratura fechada existe se e somente se as faces são C-ortogonais.** [ESTABELECIDO]
   Com `C` diagonal o sistema **decompõe exatamente em n cópias escalares do L1**, e o portão vira
   uma **lei de área por face com limiar próprio**:
   ```
   face i solta quando   𝒜_i(t) = ∫R_i = [Ψ(u_i⁰) − 1] / (βC_ii),   u_i⁰ = e^{−βz_i(t_morte)}
   t_release = min_i (esses tempos)
   ```
   Para o líder (`z⁰=0`, `u⁰=1`) o limiar é 0 e isto **é** o teorema do paper. Fora da diagonal
   **não há quadratura**: o objeto correto é um fluxo reduzido de dimensão `d+1` dirigido só pelo
   forcing.

5. **Toy com troca de líder forçada: o portão escalar erra 40%–670%, a forma vetorial acerta.**
   Em `C` diagonal e β·δ ≤ 6 o portão escalar do líder inicial erra por fator 2–7 (errS/t até 6.7),
   enquanto o portão vetorial derivado acerta com erro `7e-2 (β=3) → 1e-5 (β=1000)`, decaindo
   ~`β^{−1.9}`. **Este é o resultado-chave do scoping.**

6. **Os 3 episódios reais sem área não satisfazem (H3) — mas nenhum dos 16 satisfaz.** Replay
   determinístico com instrumentação de líder: em **9/9 episódios medidos** `β·δ_gap ∈ [0.003, 0.35]`
   — isto é, o segundo dado pesa 70%–99.7% do líder. **A hipótese central do teorema é violada em
   todos os episódios reais**; a lei escalar acerta 13/16 por robustez, não por hipótese satisfeita.
   2 dos 3 fracassos têm troca de líder genuína; o terceiro (1923) é um empate permanente.

7. **Veredito: teorema tratável, com um pedaço de matemática nova de peso médio.** Não quebra.
   Ver §6.

---

## 1. A estrutura

### 1.1 O campo exato

Rede da §2 do paper, unidade morta `A` com readout +1, `θ_A ∈ R^{d+1}`, `X_i=(x_i,1)`,
`z_i = X_i·θ_A`, resíduo-excluindo-A por dado `R_i = y_i − f_{−A}(x_i)`, `res_i = g_β(z_i) − R_i`.
O campo de gradiente sobre `θ_A` é `θ̇_A = −(1/n)Σ_i res_i σ(βz_i) X_i`, logo

```
ż_i = Σ_k C_{ik} (R_k − g_β(z_k)) σ(βz_k),      C_{ik} = ⟨X_i, X_k⟩ / n .        (D1)
```

`C = XXᵀ/n ⪰ 0` tem **posto d+1** (2 no experimento, entrada escalar). Verificado contra o
gradiente automático de `L`: erro 6e-11 (`verifica_identidade.py`, I1).

Isto **corrige a formulação do pedido**: a força não é `Σ_i R_i g'_β(z_i) x_i` com canais
independentes — os `z_i` não são coordenadas independentes, todos vêm do **mesmo** `θ_A ∈ R^{d+1}`.
A decomposição pedida "`z_i = z·a_i + d_i`" é, exatamente, `z = Xθ_A`: **não há parte "por dado"
livre**; o que é comum ao neurônio é `θ_A` inteiro, e o que é por dado é só a linha `X_i`. A
consequência é forte: com `n > d+1` dados ativos o sistema é **sobredeterminado**, e é essa
sobredeterminação que mata a quadratura escalar (§2.2).

### 1.2 Duas leituras estruturais (ambas exatas em ordem principal)

**(a) Fluxo de gradiente sobre soma de exponenciais.** No park (`g_β(z_k) ≈ 0`, `σ ≈ e^{βz_k}`):
```
θ̇_A = ∇_θ V(θ,t),     V(θ,t) = (1/nβ) Σ_i R_i(t) e^{β X_i·θ} .                 (D2)
```
Verificado: `|∇V − (1/n)Xᵀ(R⊙e^{βz})| = 4e-13`; o campo exato difere do de ordem principal por
4.6% na instância de teste com β=7 (I5). A direção de subida é dominada pelo termo de maior
`z_i + β^{−1}log|R_i|` — **uma estrutura tropical**: o locus de troca de líder é a hipersuperfície
tropical `z_i + β^{−1}log|R_i| = z_k + β^{−1}log|R_k|`. (Ponte direta com a lente tropical da tese
do projeto; [ESPECULAÇÃO] quanto a haver aí conteúdo além da observação.)

**(b) Lotka–Volterra não-autônomo.** Com `q_i = e^{βz_i}` (canal do dado i):
```
q̇_i = β q_i Σ_k C_{ik} R_k(t) q_k ,        C ⪰ 0 simétrica, posto d+1 .        (D3)
```
Interação simétrica ⟹ é um fluxo de gradiente na métrica de Shahshahani; taxas de crescimento
`R_k(t)` não-autônomas e de sinal variável; **volta = primeiro `q_i` atingir 1**. Este é o objeto
matemático do Degrau 3, e é um objeto **reconhecível**, com teoria disponível — o que é a base do
veredito de tratabilidade.

### 1.3 A quadratura Ψ, forma exata multi-dado

Com `u_i = e^{−βz_i}`, `Ψ(u)=log u + u` (o mesmo Ψ do L1), `u̇_i = −βu_i ż_i`, `σ(βz_k)=1/(1+u_k)`:

```
dΨ(u_i)/dt = −β (1+u_i) Σ_k C_{ik} (R_k − g_β(z_k)) / (1 + u_k)                 (D4)
           = −βC_ii (R_i − g_β(z_i))                            ← termo do L1 escalar
             − β Σ_{k≠i} C_{ik} · (1+u_i)/(1+u_k) · (R_k − g_β(z_k))   ← acoplamento
```
Verificado numericamente (I2): erro relativo 9.6e-9.

**O peso do acoplamento é `(1+u_i)/(1+u_k)`, exatamente.** Em park profundo (`u ≫ 1`) isso é
`u_i/u_k = e^{β(z_k − z_i)}` — verificado (I4): a razão entre o peso exato e `e^{β(z_k−z_i)}` vai a
1 conforme a profundidade cresce (1.06, 0.94 nos pares profundos; 1.6–1.7 nos pares rasos, onde a
aproximação ainda não vale). **A candidata do pedido está certa em forma e é aqui derivada, não
chutada** — com a ressalva de que o peso certo é `(1+u_i)/(1+u_k)`, e a exponencial é só o seu
limite profundo.

### 1.4 Regime de LÍDER ESTÁVEL — corolário direto do forcing

**Corolário (líder estável).** [ESTABELECIDO, modulo o bookkeeping do Ap. B do paper]
Seja `i*(t) ≡ i*` constante na janela e `δ_gap = min_{k≠i*} (z_{i*} − z_k) > 0` uniforme. Então em
(D4) para `i = i*` todo peso de acoplamento é `≤ e^{−βδ_gap}`, e

```
dΨ(u_{i*})/dt = −βC_{i*i*} R_{i*} + C_{i*i*} log(1+w_{i*}) + E,
|E| ≤ β · (Σ_{k≠i*}|C_{i*k}|) · (max_k|R_k| + max_k g_β(z_k)) · e^{−βδ_gap}
```
que é **literalmente** a identidade (eq. quadrature) do `lem:gauge` do paper, com o erro `𝓔_{(H3)}`
substituído por um erro com `δ_1 → δ_gap`. **Tudo a jusante (L2′–L6′, o bootstrap, a lei de release
`t_rel = t_exit − 1/(cR^hyb(t_exit))·β^{−1}`) passa sem alteração**, com `c = C_{i*i*}` e
`R = R_{i*}`. Nada precisa ser reprovado: a hipótese (H3) do paper *já é* a hipótese de líder
estável, e o Degrau 3 apenas a nomeia e a torna a única hipótese relevante.

Isto também **fixa o preço em β**: o teorema pede `β δ_gap ≫ log β`, isto é
```
β₀^{multi-dado} ≳ δ_gap^{−1} log(1/δ_gap) .
```
Para `n` dados em posição geral num intervalo limitado, o gap típico entre margens vizinhas é
`O(|w|·diam/n)`, logo **β₀ cresce linearmente em n**. Esta é uma previsão nova, quantitativa, e
falsificável — e é o que os dados reais medem (§3).

---

## 2. A forma vetorial

### 2.1 O portão vetorial exato (identidade de deslocamento)

Sem nenhuma aproximação, de `θ_A(t) − θ_A(t_morte) = (1/n)Σ_k X_k P_k(t)` com
`P_k(t) = ∫_{t_morte}^{t} (R_k − g_β(z_k)) σ(βz_k) ds` a **área ponderada pelo canal do dado k**:

```
z_i(t) = z_i(t_morte) + Σ_k C_{ik} P_k(t) ,
volta na face i quando   Σ_k C_{ik} P_k(t) = |z_i(t_morte)| ,   t_rel = min sobre i e t.   (D5)
```

Este é o **portão vetorial**: n portões, um por face, com pesos `C_{ik}` e limiares `|z_i(t_morte)|`.
A área escalar do paper é o caso `n=1`. `P` obedece o sistema fechado, dirigido **apenas pelo
forcing** `R(t)` e pelas margens iniciais:

```
Ṗ_k = R_k(t) · exp( β [ z_k⁰ + Σ_l C_{kl} P_l ] ) · (1 + o(1)) .                (D6)
```
(D6) é a **forma normal do Degrau 3**: um sistema de `n` EDOs — reduzível a `d+1` — de soma de
exponenciais, autônomo em `P` e não-autônomo em `t` via `R`. Para `n=1` integra-se em forma fechada
e devolve `e^{−βcP} = 1 − βc𝒜` ⟹ `z = −β^{−1}log(1 − βc𝒜)` ⟹ volta em `𝒜 = 0`, **recuperando o
teorema do paper exatamente**, incluindo a profundidade de park `u = cβ|𝒜|`.

### 2.2 Quando há quadratura fechada, e quando não há

**Teorema-candidato (faces C-ortogonais).** [ESTABELECIDO no toy; prova = n cópias do L1]
Se `C_{ik}=0` para `i≠k` nas faces ativas, (D4) desacopla **exatamente**:
`dΨ(u_i)/dt = −βC_ii R_i + C_ii log(1+w_i)` (verificado, I3: erro 1.3e-7). Integrando e impondo
`u_i = 1`:
```
face i solta em      𝒜_i(t) = [ Ψ(u_i⁰) − 1 ] / (β C_ii) ,   Ψ(u_i⁰) = −βz_i⁰ + e^{−βz_i⁰}
t_release = min_i { primeiro t com 𝒜_i(t) ≥ esse limiar } .                     (D7)
```
Interpretação física: **cada dado tem seu próprio relógio de área, e o dado parado a profundidade
δ precisa de uma área exponencialmente maior, `≈ e^{βδ}/(βC_ii)`.** É esse fator exponencial —
e não a hipótese (H3) — que faz o líder ganhar quando β é grande, e é ele que **falha em ser grande
quando βδ = O(1)**.

**Não existe quadratura fora da ortogonalidade.** Com `n > d+1` faces ativas, a condição para uma
`Ψ` com `dΨ/dt` função só de `t` é `∇Ψ·X_i = κ_i e^{−βz_i}` para todo `i` — n equações para `d+1`
componentes de `∇Ψ`, **genericamente insolúvel**. Confirmado no toy (§4, caso B). O substituto
correto é integrar (D6), que é barato (`d+1` dimensões) e usa só o forcing — portanto continua
sendo um teorema **sobre o forcing**, na linha do `thm:forcing`.

---

## 3. O mapa: os 3 episódios reais sem área

### 3.1 O que o instrumento gravou, e o que não gravou

`motor_4a3.py` calcula `jstar = z.argmax(0)` a cada passo e acumula
`Â_k += R̂_k(jstar(t))·η` — isto é, o instrumento do 4a.3 **já rastreia o líder instantâneo**, mas
**não é** nem a lei escalar (líder fixo) nem o portão vetorial: é uma terceira coisa, uma área de
líder-instantâneo concatenada. O índice `jstar` **não é persistido**; os `.jsonl.gz` guardam só
`(k, classe, t_morte, t_area, t_res, revived, z_park, σ_park)`. **Dos dados gravados não se pode
concluir nada sobre troca de líder.**

Como o regime é determinístico (T=0) e o config é byte-determinístico, **re-executei** os runs
`sp-b3` com instrumentação adicional (`replay_lider.py`): `jstar(t)`, gap entre a 1ª e a 2ª margem,
área por face para as 16 faces, área com líder congelado em `j0 = jstar(t_morte)`. 3 episódios
sem-área + 6 controles com-área.

### 3.2 Medida

```
 seed  m  k    rótulo   dur    j0  jfim  trocou  n_trocas  med(β·gap)  A_hat  A_fix  A_jfim
 1923  6  1  SEM-AREA  253803  15    15   False         0      0.0026  False  False   False
 1924  4  2  SEM-AREA  380709   9    15    True         1      0.0449  False   True   False
 1934  6  4  SEM-AREA   58858   1     4    True         1      0.0631  False  False    True
 1918  4  2  com-area    3944   9     7    True         1      0.0206   True   True    True
 1924  6  2  com-area   17059   9    15    True         1      0.0453   True  False    True
 1919  4  0  com-area   99320   9     9   False         0      0.1244   True   True    True
 1939  6  5  com-area   65511  11    11   False         0      0.1378   True   True    True
 1937  6  4  com-area    5706  14    14   False         0      0.1437   True   True    True
 1937  6  3  com-area   85191  14    14   False         0      0.3461   True   True    True
```
(`A_hat` = área do instrumento 4a.3; `A_fix` = área com líder congelado em j0; `A_jfim` = área da
face que efetivamente volta. `β·gap` = mediana no episódio de `β(z_{(1)} − z_{(2)})`.)

### 3.3 Leitura honesta

- **Há troca de líder em 2 dos 3 fracassos** (1924/m4 `j9→j15`; 1934 `j1→j4`, com o líder j1 só 58%
  do episódio e j4 42%). O terceiro (1923) **não troca**, mas tem `β·gap = 0.0026`: as faces j15 e
  j13 estão empatadas o episódio inteiro (peso relativo 0.997). Ou seja: **o rótulo "líder" não
  significa nada nesses episódios.** A frase do paper §6.4, "voltas conduzidas fora do dado-líder",
  fica **parcialmente confirmada** — 2/3 literalmente, o terceiro por degenerescência do conceito.

- **Mas troca de líder não é o discriminante.** 2 dos 6 controles com-área também trocam de líder.
  O que separa é o gap: os três maiores `β·gap` (0.124, 0.138, 0.144, 0.346) são todos com-área e
  todos sem troca; os três menores incluem os três fracassos — mas 1918 (com-área, 0.0206) fura a
  ordenação. **N=9, sem pré-registro: isto é geração de hipótese, não teste.**

- **O achado estrutural é mais forte que o discriminante.** Em **9/9** episódios reais
  `β·δ_gap ≤ 0.35`, isto é, `e^{−βδ_gap} ≥ 0.70`. A hipótese (H3)/(H4′) do paper — a que produz o
  erro `O(e^{−βδ_1})` e que sustenta a redução escalar do `lem:corner-out` — **está violada em
  todos os episódios medidos**, não em 3 deles. A lei escalar acerta 13/16 **fora do seu domínio de
  validade**. Isso é um dado sobre robustez, não sobre a hipótese; e explica por que a taxa de
  falha é ~19% e não 0% nem 100%.

- **O portão vetorial não erra, mas a β=3 quase não prevê tempo.** Aplicando (D5) post-hoc com as
  séries gravadas (`gate_vetorial.py`): o portão vetorial abre em **9/9** episódios, incluindo os
  **3/3 sem-área**, onde o escalar abre em só 1/3. Porém: nos 3 fracassos ele abre cedo demais
  (t=0, t=6, e 69436 contra `t_res=380709`) — a β=3 os limiares `|z_i(t_morte)|` são ~0 e o portão
  é quase vácuo. Nos episódios de líder estável ele é **apertado**: 98992 vs 99320 (0.3% cedo),
  80704 vs 85191 (5%), 61216 vs 65511 (6.6%). **Conclusão honesta: nos dados reais o portão
  vetorial é uma condição necessária que nunca falha e uma previsão de tempo que só tem conteúdo
  onde o escalar já funcionava.** A limitação é β=3, não a forma vetorial.

---

## 4. Toy mínimo com troca de líder forçada

Sistema: 1 unidade morta, `θ_A ∈ R²`, 2 dados, forcing prescrito (forma-forcing)
`R_1(t) = −1 + 0.5t`, `R_2(t) = −1 + 4t`; `z⁰ = (0, −δ)` (face 1 é o líder na morte).
Dinâmica exata (D1) integrada por RK4, `dt = min(1e-4, 1/(20β))`; cruzamento de zero interpolado.
Preditores: **S** = escalar do líder inicial (1º zero de `∫R_1`, sempre `t=4.000`);
**V** = portão vetorial diagonal (D7); **H** = fluxo híbrido vetorial (D6), forcing-only.

### 4.1 Caso A — `x = [1, −1]`, `C = I` (faces C-ortogonais)

```
  beta   delta  b*delta |    t_rel face |     errS  errS/t |    errV |    errH
     3     0.1     0.30 |  0.73455    1 |   3.2654   4.446 | 7.1e-02 | 1.4e-01
     3     0.3     0.90 |  0.96911    1 |   3.0309   3.127 | 4.4e-02 | 1.7e-01
     3     1.0     3.00 |  2.19440    1 |   1.8056   0.823 | 9.7e-03 | 1.4e-01
    10     0.1     1.00 |  0.70768    1 |   3.2923   4.652 | 1.2e-02 | 7.2e-02
    10     0.3     3.00 |  1.33309    1 |   2.6669   2.001 | 3.0e-03 | 7.5e-02
    10     1.0    10.00 |  4.07670    0 |   0.0767   0.019 | 7.7e-02 | 7.7e-02
    30     0.1     3.00 |  0.90715    1 |   3.0929   3.409 | 9.8e-04 | 4.0e-02
    30     0.2     6.00 |  2.87108    1 |   1.1289   0.393 | 1.3e-04 | 1.9e-02
    30     0.3     9.00 |  4.01148    0 |   0.0115   0.003 | 1.2e-02 | 1.2e-02
   100    0.03     3.00 |  0.66609    1 |   3.3339   5.005 | 2.7e-04 | 1.9e-02
   100     0.1    10.00 |  4.00129    0 |   0.0013   0.000 | 1.3e-03 | 1.3e-03
   300    0.01     3.00 |  0.56519    1 |   3.4348   6.077 | 6.8e-05 | 8.1e-03
   300    0.03     9.00 |  3.93524    1 |   0.0648   0.016 | 4.6e-05 | 2.0e-03
  1000   0.003     3.00 |  0.52120    1 |   3.4788   6.675 | 1.1e-05 | 2.8e-03
  1000    0.01    10.00 |  3.57870    1 |   0.4213   0.118 | 2.3e-05 | 7.5e-04
```

**Leitura.**
- O parâmetro de transição é **β·δ**, não β: a transição líder-estável ↔ troca-de-líder está em
  `β·δ ≈ 8–10`, **estável ao longo de duas ordens de grandeza em β** (linhas com `β·δ=3` erram
  igual em β=3, 30, 100, 300, 1000).
- **O portão escalar do líder inicial FALHA exatamente como nos 3/16**: erro relativo 0.4 a 6.7
  em todo o regime `β·δ ≲ 6` — a volta acontece a `t≈0.5–2.9` e ele prediz `t=4.0`. E prediz a
  **face errada**.
- **O portão vetorial derivado ACERTA**, e acerta melhor conforme β cresce: `7.1e-2 (β=3)`,
  `1.2e-2 (10)`, `9.8e-4 (30)`, `2.7e-4 (100)`, `6.8e-5 (300)`, `1.1e-5 (1000)` — taxa empírica
  `≈ β^{−1.9}`, compatível com o refinamento `O(log β/β²)` conjecturado no `rem:mechanism` do paper.
  **Este é o resultado-chave do scoping.**
- No regime de líder estável (`β·δ ≥ 10`) V e S coincidem, como devem.

### 4.2 Caso B — `x = [0.5, 1.2]`, `C = [[0.625, 0.8],[0.8, 1.22]]` (faces acopladas)

```
  beta   delta  b*delta |    t_rel face |    errS  errS/t |   errV  fV |   errH  fH
     3     0.1     0.30 |  1.01051    0 |  2.9895   2.958 | 3.7e-01  1 | 8.7e-02  0
    10     0.1     1.00 |  1.19593    0 |  2.8041   2.345 | 5.3e-01  1 | 1.4e-01  0
    10     0.3     3.00 |  2.81858    0 |  1.1814   0.419 | 1.6e+00  1 | 2.1e-01  0
    30     0.1     3.00 |  3.07761    0 |  0.9224   0.300 | 2.2e+00  1 | 1.3e-01  0
   100    0.03     3.00 |  3.34660    0 |  0.6534   0.195 | 2.7e+00  1 | 4.5e-02  0
   300    0.01     3.00 |  3.52390    0 |  0.4761   0.135 | 3.0e+00  1 | 1.5e-02  0
   300    0.02     6.00 |  3.97402    0 |  0.0260   0.007 | 2.9e+00  1 | 1.1e-03  0
```

**Leitura.** Com acoplamento, o portão diagonal **V prediz a face errada em todos os β** e seu erro
**não decai** — a quadratura fechada é falsa fora da C-ortogonalidade, como a álgebra da §2.2
previu. O fluxo reduzido **H acerta a face sempre e converge** (`1.4e-1 → 1.1e-3`).
Mecanismo interessante e novo: aqui `C_{01} = 0.8 > C_{00} = 0.625`, de modo que **empurrar a face 2
levanta a face 1 mais rápido do que a si mesma** — a volta é literalmente *conduzida* pelo dado 2 e
*acontece* no dado 1. É a leitura mais forte possível de "volta conduzida fora do dado-líder", e é
um regime que a área escalar não pode nem em princípio capturar.

---

## 5. O que ficou aberto

1. **(A1)** A prova do corolário de líder estável está escrita como "L1′ com δ_1 → δ_gap"; falta
   percorrer L2′–L6′ verificando que nenhum outro passo usa (H3) de forma mais forte. O `lem:corner-out`
   (redução escalar) é o único suspeito, e é literalmente onde (H3) já é consumida no paper.
2. **(A2)** O caso de troca: não há teorema. Há (D5)/(D6) — a forma normal — e a evidência numérica
   de que ela é o objeto certo. Falta o teorema de seleção de ramo na travessia da parede tropical
   `z_i + β^{−1}log|R_i| = z_k + β^{−1}log|R_k|` (o análogo multi-dado dos corners L3′/L6′). É aqui
   que mora a matemática nova.
3. **(A3)** O limiar do caso diagonal (D7) é exato em ordem principal mas não tem termo de correção
   `O(β^{−1})` derivado; o toy mede `≈β^{−1.9}` mas isso não é prova.
4. **(A4)** Toda a §3 é **post-hoc, N=9, exploratório**. O `β·gap` como covariável de fracasso não
   foi pré-registrado e não pode ser reportado como teste.
5. **(A5)** Tensão empírica séria, e é estratégica: o regime onde (H3) vale (`β·δ_gap ≳ 10`, isto é
   `β ≳ 100–3000` com n=16 dados) é **exatamente o regime onde o 4a.3 não observa voltas nenhuma**
   (`p_volta`: 0.375 em β=3, 0.22 em β=10, **0.0** em β=300, config_4a3 C4). O teorema pede β grande;
   o instrumento só vê fenômeno com β pequeno. Isto **não é** contradição lógica (β muda o treino
   inteiro, não só o canal), mas é o obstáculo prático central do Degrau 3 e precisa de um desenho
   experimental que o ataque — por exemplo n pequeno (n=2,3 dados) a β alto, onde `δ_gap` é grande
   por construção.

---

## 6. Veredito

**TEOREMA TRATÁVEL, com um bloco de matemática nova de peso médio — não quebra.**

Repartido:

- **Grátis (corolário, dias de trabalho):** o regime de líder estável. É o teorema do paper com
  `δ_1` renomeado para `δ_gap` e um erro `O(βe^{−βδ_gap})` explícito. Vale a pena escrever porque
  **muda a leitura da hipótese (H3)**: ela deixa de ser "bookkeeping removível" (`rem:hyp` do paper)
  e passa a ser a hipótese **essencial e mensurável** do caso multi-dado, com preço
  `β₀ ≳ δ_gap^{−1}log(1/δ_gap)` e portanto `β₀ = O(n)`. Este último é um resultado novo, publicável
  como refinamento, e é falsificável.

- **Barato e novo (semanas):** o caso C-ortogonal, (D7) — a lei de área **por face com limiar
  `[Ψ(u_i⁰)−1]/(βC_ii)`** e `t_rel = min_i`. É n cópias exatas do L1, verificado, e já contém o
  fenômeno de troca de líder em forma fechada. Enunciado limpo, prova por redução, toy que o
  confirma a `β^{−1.9}`. É a melhor relação resultado/esforço do Degrau 3.

- **Matemática nova de verdade (meses):** o caso acoplado. O objeto é (D3)/(D6) — **Lotka–Volterra
  não-autônomo com matriz de comunidade PSD de posto d+1, ou equivalentemente ascensão de gradiente
  sobre soma de exponenciais**. O que falta especificamente é o **corner de troca de líder**: a
  análise de camada na travessia da parede tropical, onde dois canais têm peso comparável e a
  redução escalar do `lem:corner-out` deixa de valer. Isso é uma perturbação singular geométrica
  de tipo *two-fold / dobra dupla* — não é fácil, mas é padrão na literatura de sistemas
  Filippov/entry-exit, e o sistema é de dimensão `d+1` (2 no experimento), o que o mantém em
  território planar. **Não é um problema aberto de fronteira; é um capítulo de tese.**

- **O que NÃO existe:** quadratura escalar fechada no caso acoplado. Provado por contagem
  (n equações, d+1 incógnitas) e exibido no toy (caso B, V erra a face em todo β). Qualquer versão
  do Degrau 3 que prometa "uma área escalar corrigida" está errada. O que substitui é um fluxo
  reduzido de dimensão d+1, ainda dirigido só pelo forcing — logo ainda na família `thm:forcing`.

**Recomendação de sequência:** (i) escrever o corolário de líder estável e a lei diagonal (D7) como
um Degrau 3a curto; (ii) desenhar um experimento com **n=2–3 dados e β alto**, onde `δ_gap` é
grande por construção, para sair do buraco (A5) e testar (D7) com pré-registro; (iii) só então
atacar o corner de troca.
