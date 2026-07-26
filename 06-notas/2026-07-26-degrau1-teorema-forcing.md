# Degrau 1 — T2′ como teorema de forcing (m vivos + 1 morto)

Data: 2026-07-26 · Autor: Leonardo Caloi Santos (com assistência de LLM)
Artefatos: `05-experimentos/t2-n-neuronios/verifica_forcing.py`,
`forcing_m.jsonl`, `controle_dt.jsonl`, `run_forcing.log`
Fonte auditada: `06-notas/paper/paper-v3.tex` §2–3 e Apêndice A (L1–L6),
`06-notas/t2-prova/prova_L1..L6_verificacao.py`

---

## TL;DR

`[ESBOÇO DE PROVA verificado numericamente]` — **não é teorema.**

A auditoria linha a linha dos seis lemas mostra que **cinco deles (L1, L3, L4,
L5, L6) não usam em nenhum passo que "o resto da rede é um neurônio"**: usam o
resíduo agregado `R(t) = y_j − f_{−A}(x_j; θ_B(t))` apenas através de quatro
propriedades — regularidade C¹, uma troca de sinal na janela, transversalidade
nos dois cantos, e uma cota de crescimento. O único lema que menciona a
estrutura de B é **L2 (Grönwall)**, e lá a estrutura entra por *conveniência*
(o campo de B é afim ⇒ constante de Lipschitz global explícita), não por
necessidade: Grönwall só precisa de Lipschitz local num tubo compacto.

Além disso — e isto é uma descoberta da auditoria, não uma concessão — na
classe do paper (uma camada, readout fixo, padrão de ativação congelado por
(H4)) **o campo de B continua exatamente afim para qualquer número m de
unidades vivas**, porque uma rede ReLU com padrão fixo e só a primeira camada
treinável é *linear nos parâmetros*. Ou seja: L2 atravessa **verbatim** no
Degrau 1. A troca "afim → Lipschitz local" só é necessária no Degrau 2
(readout treinado, segunda camada), onde o campo é bilinear.

Verificação numérica (β = 1e4, m ∈ {4, 8, 16} vivas + 1 morta, mesmo integrador
canônico, sem nenhum parâmetro ajustado, 66 instâncias que passam
(H1′)–(H4′)): a lei acompanha uma variação de fator 8 no coeficiente predito
(`[−109, −14]`) com **R² = 0.950 sem ajuste**. Erro relativo mediano
**0.80 % / 6.85 % / 5.58 %** para m = 4/8/16, contra **2.68 %** no baseline de
2 neurônios. **Em m = 4 o Degrau 1 está sustentado** (1.81 % contra 1.79 % em
faixa casada de R_exit); **em m ≥ 8 o erro degrada por fator 2–4 na mediana e
5 na cauda**, e o controle de discretização (dt₀/5) mostra que **não é do
integrador**. A degradação some estatisticamente ao controlar por `1/R_exit`
(t = 0.71 para log₂ m contra t = 2.56 para 1/R_exit), mas com n = 66 isso
**não é conclusivo**: a uniformidade em m não está demonstrada. Detalhes em §4.

**O que isto NÃO é:** não passou por parecer adversarial; as constantes não
foram acompanhadas em m (só a *taxa* foi); o experimento é 1-D com readout
fixo, que é exatamente a classe em que L2 é gratuito.

---

## 1. Auditoria lema a lema

Notação: `R(t)` é o resíduo agregado da face j excluindo A; `Ar(t) = ∫_{t_hit}^t R`;
`c = ‖(x_j,1)‖²/n`; `w = e^{βz}`, `u = 1/w`, `Ψ(u) = log u + u`.

### Hipóteses sobre o forcing — (H1′)–(H2′)

Enunciadas **sobre R, não sobre B**:

- **(H0′) regularidade.** `R ∈ C¹([t_hit, H])` com `‖Ṙ‖_∞ ≤ C_R`.
  *(consumida por: L2 passo 1, L4 passo 4, L5 passo 3)*
- **(H1′) entrada transversal.** `R(t_hit) < 0`, e a velocidade de chegada
  `v = c|R(t_hit)| > 0`. *(L3, L5)*
- **(H2′) saída transversal com área nula.** `Ar` tem primeiro zero em
  `t_exit < H`, com `R(t_exit) > 0`, `Ar < 0` em `(t_hit, t_exit)` e
  `|Ar| ≥ a_0 > 0` no bulk. *(L1 passo 4, L4 passos 3–4, L6)*
- **(H3′) resposta Lipschitz do forcing.** existe `Lip_R` com
  `|R[θ_B] − R[θ_B′]| ≤ Lip_R‖θ_B − θ_B′‖`, e o campo de B é Lipschitz `L_B`
  num tubo compacto em torno da trajetória híbrida; a sensibilidade do campo de
  B ao resíduo é limitada por `C_f`. *(L2, L5)*
- **(H4′) episódio isolado.** nenhuma outra unidade chaveia em `[0,H]`; as
  demais margens de A ficam `≤ −δ₁ < 0`. *(L1 erro E_(H3), L6 passo 4)*

Só isto. Nenhuma hipótese sobre a dimensão de `θ_B`, sobre a forma funcional
de `f_{−A}`, ou sobre quantas unidades vivas existem.

### L1 — forma normal e quadratura aditiva

| passo | onde B entra |
|---|---|
| 1 (equação da margem) | **não entra.** `res_j = g_β(z) − R` é a *definição* de R. O que se usa de B é só que ele foi agregado em R. Os termos `i ≠ j` vêm de (H4′), não de B. |
| 2 (regra da cadeia exata) | não entra: é álgebra em `Ψ, u, σ`. |
| 3 (identidade aditiva) | não entra. |
| 4 (cota da correção, três janelas) | usa `R(t_hit) ≠ 0`, `R(t_exit) ≠ 0` e `|Ar| ≥ a_0` no bulk — isto é (H1′)–(H2′), nada mais. A constante `C_4 = 2/(c·min(|R(t_hit)|, R(t_exit))) + T_ep/a_0` é escrita **só em termos de R**. |

**Veredito: passa com R genérico, prova idêntica.** Nem uma linha muda.

### L2 — subsistema B e feedback

Único lema com estrutura de B explícita, em dois lugares:

1. *"o campo de B é afim com constante de Lipschitz `L_B = λ_max(G_B)`"*.
   Onde é usado: só para ter `‖Δ̇‖ ≤ L_B‖Δ‖` em Grönwall. **Grönwall integral
   só pede Lipschitz local num compacto.** Substituição: (H3′). No Degrau 1
   isto é *gratuito*: com readout fixo em +1, só a primeira camada treinável e
   padrão de ativação congelado por (H4′), a rede é `f(x) = Σ_k (w_k x + b_k)·1_{k ativo}`,
   **linear nos parâmetros**, logo o campo de gradiente é afim e `G_B` é a
   matriz de Gram (2m × 2m) das features ativas, para *qualquer* m.
2. *"`R^hyb` é afim em `θ_B`"*. Idem: só é usado para `‖δR‖ ≤ Lip_R‖Δθ_B‖`
   (consumido em L5 passo 3). Lipschitz basta. E também é afim para todo m,
   pela mesma razão.

O passo 2 (cota de acoplamento `‖G_A‖ ≤ C_f g_β(z) + C_3 e^{−βδ₁}`) **não**
depende de m na *forma*: a perturbação que A injeta em `res_i` é `g_β(z_{A,i})`,
independentemente de quantas vivas existem. O que muda é a constante `C_f`
(norma da sensibilidade do campo de B ao resíduo) e `L_B`.

**Veredito: passa; enunciado tem de trocar "afim" por "Lipschitz local no
tubo" para valer fora da classe de camada única com readout fixo. Dentro da
classe do paper, passa verbatim para todo m.**
**Preço honesto:** `M_2 = e^{L_B H} C_f C_4 / c` cresce **exponencialmente em
`L_B = λ_max(G_B)`, e com readout +1 nas vivas `λ_max(G_B)` cresce ~m. A *taxa*
`O(log β/β²)` sobrevive para cada m fixo; a **uniformidade em m não está
provada** e o `β₀(m)` piora. Este é o principal buraco do Degrau 1.

### L3 — canto de entrada

Passos 1 e 2 são inteiramente escalares em `z`: usam a transversalidade
`z⁰(t) = v(t_hit − t)(1 + O(δ_z))` com `v = c|R(t_hit)|`, a identidade
`(1−σ)/σ = e^{−s}`, e a iteração monótona da sub-solução. **R entra só por
(H1′).** A estrutura de B aparece zero vezes.
**Veredito: passa com R genérico, prova idêntica.**

### L4 — sticking e release para forcing prescrito

**Já está enunciado como teorema de forcing**: *"Let R ∈ C¹ be prescribed and
satisfy (H1)–(H2)"*. É o lema-chave e não menciona B em lugar algum. O passo 1
(comparação de Chaplygin) usa apenas a monotonia do lado direito em Ψ; passos
2–4 usam (H1′)–(H2′) e o TFI.
**Veredito: passa com R genérico — literalmente já é o enunciado publicado.**

### L5 — bootstrap acoplado

Passos 1, 2, 4, 5 são continuação com margens estritas: topologia + as cotas de
L1/L2/L4. Passo 3 é o único acoplamento com B, e usa exatamente uma frase de
L2: *"since R is affine in θ_B, ‖δ‖ ≤ Lip_R · 2M₂ log β/β²"* — que é (H3′)
Lipschitz, não afinidade.
**Veredito: passa com R genérico, condicionado ao L2 generalizado.**
Consequência estrutural importante: o **cancelamento da velocidade de chegada**
no coeficiente de primeira ordem é aritmética entre `K = 1/(c|R(t_hit)|)` de L3
e a área `R(t_hit)K/β` cortada na entrada — **puramente em R**. É por isso que
o coeficiente `−1/(c R(t_exit))` não pode depender de m: nada em m sobrevive
até ele.

### L6 — canto de saída, seleção de ramo e redução escalar

- (i) espelho de L3 com `v′ = cR(t_exit) > 0`: (H2′). Genérico.
- (ii) unicidade para frente + seleção de ramo: precisa que o campo ReLU ativo
  seja **localmente Lipschitz** do lado `z ≥ 0` no ponto de reinício. Com m
  vivas isto vale desde que nenhuma viva esteja exatamente na sua quina em
  `t_exit` — que é (H4′). A alternativa de Filippov é descartada por L5,
  independentemente de m. Genérico sob (H4′).
- (iii) colagem por Grönwall clássico: precisa de Lipschitz do campo ReLU no
  intervalo final — (H3′). Genérico.
- (passo 4) redução escalar: usa **só** que as *outras margens de A* são
  `≤ −δ₁`. É uma afirmação sobre A, não sobre B. Genérico para todo m.

**Veredito: passa com R genérico.**

### Placar

| lema | passa com R genérico? | o que precisa ser trocado |
|---|---|---|
| L1 | **sim, verbatim** | — |
| L2 | **sim, com troca de hipótese** | "campo de B afim, `L_B = λ_max(G_B)`" → "campo de B Lipschitz `L_B` num tubo compacto; R Lipschitz `Lip_R`". Na classe do paper a afinidade sobrevive para todo m e nada muda. |
| L3 | **sim, verbatim** | — |
| L4 | **sim** (já é de forcing) | — |
| L5 | **sim** | consome L2 via Lipschitz, não via afinidade |
| L6 | **sim, verbatim sob (H4′)** | — |

**5/6 verbatim, 1/6 com enfraquecimento de hipótese que é gratuito na classe
testada.**

---

## 2. Riscos examinados

### (i) Regularidade de R com n unidades vivas

`R^hyb(t) = y_j − Σ_{k vivo} ReLU(z_k(x_j; θ_B(t)))`. Entre eventos de
chaveamento, o campo ReLU é **polinomial** nos parâmetros (grau 2), logo o
fluxo é analítico e `R^hyb` é analítico. Nos instantes de chaveamento de uma
viva, `R^hyb` tem uma quina: é `C⁰` e `C¹` por partes, **não** `C¹` global.

Portanto: com m unidades vivas, `R^hyb` é **analítico por partes**, com um
número finito de quinas (uma por evento de chaveamento). (H4′) elimina todas as
quinas por decreto. Isto é mais forte do que o necessário: as provas usam `C¹`
de R apenas (a) em `‖Ṙ‖_∞ ≤ C_R` — que sobrevive a quinas, basta `R`
Lipschitz; e (b) na transversalidade dos zeros de `Ar` em `t_hit` e `t_exit`.
Uma versão mais fraca, **(H4″): nenhuma viva chaveia numa vizinhança δ de
`t_hit` e de `t_exit`**, já basta, com R apenas Lipschitz no meio. Não fizemos
a contabilidade dessa versão — fica registrado como o afrouxamento natural.
`[ESPECULAÇÃO]` quanto a estar correta em detalhe.

### (ii) Acoplamento de volta com m unidades

A unidade morta injeta `g_β(z) = β^{−1}log(1+w) ≤ w/β = e^{βz}/β` em `res_j`,
e `O(e^{−βδ₁})` nos outros dados. **O tamanho da injeção não depende de m** —
é uma propriedade de A. O que depende de m é a *amplificação*: o efeito
integrado sobre `θ_B` passa por `e^{L_B H} C_f`, e daí de volta a R por
`Lip_R`. Concretamente, na cadeia de L2 → L5:

```
sup‖θ_B^β − θ_B^hyb‖ ≤ e^{L_B H} [ (C_f/(cβ)) ∫ c log(1+w) dt + H C_3 e^{−βδ₁} ]
                     = O(log β / β²)    com constante  e^{L_B H} C_f C_4 / c
```

A taxa `O(log β/β²)` é **inalterada por m**: nada em m entra no expoente. O que
degrada é a constante, e degrada mal — exponencialmente em `L_B ~ λ_max(G_B)`.
Com readout +1 nas vivas, `λ_max(G_B) ~ m`, e o `β` a partir do qual o regime
assintótico morde cresce com m. **O experimento de §4 NÃO mede esse canal**:
ele adota readout NTK `m^{−1/2}` nas vivas justamente para manter `L_B = O(1)`
e isolar o efeito do *número* de unidades. Medir `β₀(m)` na família de readout
+1 é experimento separado, e é onde este esboço é mais frágil.

Um segundo efeito de m, mais sutil e não coberto pelas constantes: com m
maior o subsistema vivo **converge mais rápido** (autovalores do Gram escalam
com m), o episódio encurta como ~1/m, e a razão
`(t_exit − t_hit)·β` — a "profundidade do regime assintótico" — cai. Registrado
como diagnóstico no jsonl.

### (iii) (H2′) vira condição verificável?

**Sim, e é a boa notícia metodológica.** (H1′)–(H2′) são condições sobre a
*trajetória híbrida de referência*, que é computável por uma única integração
ReLU com A congelado — não sobre o fluxo-β, e não sobre a inicialização. O
custo é uma integração. Elas são **abertas** (zeros transversais, desigualdades
estritas), logo estáveis a perturbação — o que é o que legitima usá-las como
filtro pré-registrado.

O que elas **não** são: preditíveis em forma fechada a partir de (X, Y, θ(0)).
Não há critério de inicialização que garanta (H2′). O mecanismo que a produz é
identificável — o resíduo da face cruza zero por acoplamento de kernel
(`ė_j = −(1/n)[K_jj e_j + Σ_{l≠j} K_jl e_l]`, com `K_jl > 0`, exige os outros
resíduos positivos quando `e_j = 0`) — mas isso é uma condição sobre a
trajetória, não sobre o dado inicial. **Honesto: (H2′) é verificável a
posteriori, não garantível a priori.** A taxa de aceitação medida em §4 é a
quantificação disso.

---

## 3. Esboço do teorema de forcing

> **Teorema (forcing; ESBOÇO).** Seja `A` uma unidade softplus-β com margem
> `z(t) = θ_A·(x_j,1)` na face j, `c = ‖(x_j,1)‖²/n`, e seja `R(t)` o resíduo
> agregado gerado por **qualquer** subsistema `θ_B` cujo fluxo satisfaça
> (H0′), (H3′), (H4′), e cujo resíduo satisfaça (H1′)–(H2′) na janela
> `[t_hit, H]`. Então, para todo `δ > 0`,
>
> ```
> sup_{[0,H] \ (B_δ(t_hit) ∪ B_δ(t_exit))} ‖θ^β(t) − θ^∞(t)‖ = O_δ(log β / β),
> ```
>
> e o primeiro retorno da margem a zero após a morte obedece
>
> ```
> t_release^β = t_exit − 1/(c R(t_exit)) · 1/β + o(1/β),
> ```
>
> com `t_exit` o primeiro zero de `∫_{t_hit}^t R`. A profundidade de sticking é
> `z ~ −β^{−1} log β`, e `sup‖θ_B^β − θ_B^hyb‖ = O(log β/β²)` com constante
> `M₂ = e^{L_B H} C_f C_4 / c`, `C_4 = 2/(c·min(|R(t_hit)|, R(t_exit))) + T_ep/a_0`.

**Esqueleto de prova.** Composição idêntica à do paper §3.3, com:
L1 idêntico (R genérico já é o que a prova usa); L3 idêntico; L4 idêntico
(já enunciado para R prescrito); L6 idêntico sob (H4′); L5 idêntico, trocando
"R afim em θ_B" por (H3′) Lipschitz no passo 3; L2 com Grönwall integral sobre
Lipschitz local no tubo em vez de afinidade global. O cancelamento da
velocidade de chegada (L5 passo 5) é aritmética entre `K = 1/(c|R(t_hit)|)` e a
área `R(t_hit)K/β`, ambas escritas só em R — é por isso que o coeficiente de
primeira ordem `−1/(cR(t_exit))` é *estruturalmente* independente do
subsistema vivo.

**O que o esboço NÃO entrega:** (a) uniformidade das constantes em m — `M₂`
depende de `e^{L_B H}`; (b) a versão com R apenas Lipschitz por partes
((H4″)); (c) `β₀` explícito, que já faltava no caso 2-neurônios.

---

## 4. Verificação numérica

`[RESULTADO 2026-07-26]`

**Desenho.** `f(x) = g_β(w₀x+b₀) + m^{−1/2} Σ_{k=1}^{m} g_β(w_kx+b_k)`, dados 1-D
(n = 4), readout **fixo** (a₀ = 1 na morta — só ele entra em `c` e na equação da
margem, logo a predição não muda; a_k = m^{−1/2} nas vivas), β = 1e4, fluxo de
gradiente, **mesmo integrador canônico do paper**: Euler explícito, dt₀ = 1e-5
na referência híbrida ReLU, dt = min(1e-5, 0.02/β) = 2e-6 no acoplado, H = 1.8.
As vivas se dividem em 3 classes de padrão de ativação, de modo que `f_B` é
linear-por-partes com 2 quinas — o subsistema vivo **não** é uma cópia
reescalada de um neurônio. Predição testada, **sem nenhum parâmetro livre**:
`β(t_release − t_exit^hyb) → −1/(c R^hyb(t_exit))`, com `t_exit` e `R_exit`
medidos na referência híbrida, nunca no fluxo-β. 700 sorteios por m, peneira
barata (dt = 4e-4) e depois as 25 primeiras candidatas por m avaliadas
integralmente; toda instância que falha (H1′)–(H4′) em dt₀ é reprovada.

**Resultado bruto** (66 instâncias que passaram):

| m vivas | n | erro rel. mediana | p90 | máx | R_exit mediana | β·(t_exit−t_hit) mediana |
|---|---|---|---|---|---|---|
| 1 (baseline, sweep 2-neur.) | 57 | **2.68 %** | 7.26 % | 12.0 % | — | — |
| 4 | 23 | **0.80 %** | 5.42 % | 16.2 % | 0.141 | 1670 |
| 8 | 21 | **6.85 %** | 17.3 % | 20.5 % | 0.073 | 1080 |
| 16 | 22 | **5.58 %** | 16.9 % | 23.2 % | 0.068 | 865 |

**A lei, não o ajuste.** Os valores preditos varreram `[−109, −14]` (fator 8) e
os medidos `[−132, −14]`. Sem nenhum ajuste, `R² = 0.950`; a regressão
`medido ~ a·predito + b` dá `a = 1.154`, `b = 6.12` (a lei é `a = 1`, `b = 0`).
Ou seja: a lei acompanha uma variação de fator 8 no coeficiente predito, com um
viés sistemático de ~15 % de inclinação. Isso é o que sustenta o Degrau 1;
não é o mesmo que "erro de 3 %".

**Controle de discretização** (m = 16, dt₀/5, 5 instâncias): 12.67→12.67 %,
2.22→2.05 %, 0.42→0.53 %, 2.10→1.87 %, 1.48→1.47 %. **O resíduo não é do
integrador.**

**O confundimento m × transversalidade.** Instâncias com m maior saíram
sistematicamente com `R_exit` menor (0.141 → 0.068), e o próprio teorema diz
que o resto `o(1/β)` é controlado por `1/R(t_exit)` e pela profundidade
assintótica. Restringindo à faixa comum `R_exit ∈ [0.05, 0.13]`:

| m | n | mediana | p90 | máx |
|---|---|---|---|---|
| 1 | 26 | 1.79 % | 3.26 % | 4.01 % |
| 4 | 9 | 1.81 % | 5.79 % | 6.86 % |
| 8 | 16 | 7.40 % | 15.0 % | 20.5 % |
| 16 | 18 | 4.21 % | 13.0 % | 23.2 % |

E a regressão OLS do erro sobre `(log₂ m, 1/R_exit, duração do episódio)`:

```
const      +0.0030  (t = +0.07)
log2 m     +0.0063  (t = +0.71)      <- NAO significativo
1/R_exit   +0.0044  (t = +2.56)      <- significativo
duracao    -0.1632  (t = -1.09)
```

**Veredito numérico, sem amaciar.**
1. Em m = 4 a lei reproduz o baseline de 2 neurônios (1.81 % contra 1.79 % em
   faixa casada de `R_exit`). Degrau 1 **sustentado** aí.
2. Em m = 8 e m = 16 o erro **degrada** — mediana 4–7 % contra 1.8 %, cauda
   20–23 % contra 4 % — e a degradação **não** é do integrador.
3. Controlando por `1/R_exit` e pela duração do episódio, o coeficiente em m
   **não é estatisticamente significativo** (t = 0.71), enquanto `1/R_exit` é
   (t = 2.56). Isto é consistente com a leitura "o que degrada é a
   transversalidade da saída e a profundidade assintótica, não m em si" — mas
   com n = 66 e uma comparação em faixa casada que **ainda mostra m = 8 pior**,
   **não é conclusivo**. Não se pode afirmar que a lei é uniforme em m.
4. A **taxa de aceitação de (H1′)–(H2′) é 3 %** em todos os m (675/700 caem na
   peneira). Motivos dominantes: "sem morte no horizonte" e "(H4′) unidade viva
   chaveia na janela". O episódio de entrada-saída é um evento raro no espaço
   de inicializações desta família.

**Achado colateral, honesto.** A primeira família tentada usava readout +1 em
todas as unidades (a do paper). Nela o subsistema vivo acelera ~m no espaço de
funções, o episódio encolhe ~1/m e a aceitação colapsa (0/1800 em m ∈ {4,8,16}
com T_MIN = 5e-3). Isso **não é uma falha do teorema** — é a escala de
parametrização: com readout +1 e m unidades, `λ_max(G_B) ~ m`. Adotou-se a
escala NTK (a_k = m^{−1/2}) exatamente para que o único fator que varia entre
os três m seja o número de unidades, e não a escala de tempo. Registre-se que
**a família de readout +1 não foi testada** e que nela a constante
`M₂ = e^{L_B H}C_fC_4/c` explode.

---

## 5. O que mataria isso

1. **Degradação sistemática do erro com m.** Se o erro relativo mediano
   crescer monotonicamente com m (por exemplo 3 % → 8 % → 20 % de m = 4 a 16)
   e o controle de discretização mostrar que não é do integrador, então a
   independência do coeficiente em relação ao subsistema vivo é falsa — ou o
   `β₀(m)` cresce tão rápido que a lei é vazia em qualquer β praticável. Este é
   o teste mais provável de matar.
2. **Um contraexemplo com R legítimo e conclusão falsa.** Construir `R(t)`
   satisfazendo (H0′)–(H2′) e mostrar `t_release` fora da lei mataria o
   teorema *como enunciado*, revelando uma hipótese oculta que a auditoria não
   pegou. Rota mais promissora: `R` com `Ṙ` muito grande perto de `t_exit`
   (transversalidade fraca com `C_R` grande), que estressa L4(iii).
2b. **R apenas Lipschitz por partes.** Se um chaveamento de unidade viva dentro
   da janela quebrar a lei (e não só a prova), (H4′) deixa de ser
   "bookkeeping" e vira essencial — contradizendo a Remark 3.3 do paper.
3. **A afinidade não ser dispensável.** Se no Degrau 2 (readout treinado,
   campo bilinear) a lei quebrar, então L2 usava afinidade de verdade e a
   auditoria estará errada no ponto que ela mesma marcou como o mais frágil.
4. **Uniformidade em m ser necessária para a interpretação.** Mesmo que a lei
   valha para cada m, se `β₀(m)` crescer exponencialmente a afirmação "vale
   para MLPs" é vazia. Medir `β₀(m)` é o próximo experimento, não este.
5. **O filtro (H1′)–(H2′) estar selecionando o resultado.** A aceitação é
   baixa e cai com m. Se as instâncias aceitas forem sistematicamente as de
   episódio longo (regime assintótico profundo), o sucesso da lei é um artefato
   de seleção. Diagnóstico registrado no jsonl: `(t_exit − t_hit)·β` por
   instância; correlação erro × duração do episódio é o teste.

---

## Status epistêmico (etiquetas)

- Auditoria lema a lema (§1): `[ESTABELECIDO]` como leitura das provas do
  Apêndice A do paper-v3 — é uma afirmação sobre *o que os textos das provas
  usam*, verificável relendo-os.
- Enunciado do teorema de forcing (§3): `[ESBOÇO DE PROVA]` — não é teorema.
  Nenhuma prova foi reescrita por extenso; nenhum parecer adversarial foi
  emitido.
- Afinidade sobreviver para todo m na classe do paper: `[ESTABELECIDO]`
  (rede linear nos parâmetros com padrão de ativação fixo e readout fixo).
- Números de §4: `[RESULTADO 2026-07-26]` — execução própria, script commitado.
- Suficiência de (H4″) e uniformidade em m: `[ESPECULAÇÃO]`.
