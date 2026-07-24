# T2' — Passo 4: fechamento acoplado (bootstrap) [TEOREMA-esboço]

**O problema circular:** L1.2 (proximidade de $\theta_B$) assume a camada de
sticking controlada; L3.2 (release) assume $R$ prescrito próximo de $R^0$. O
bootstrap fecha o círculo — é a peça sem análogo em DM-S/Kristiansen/Sensi
(todos têm forcing prescrito ou autônomo).

## Cronograma híbrido (a referência correta)

O limite $\beta\to\infty$ pré-hit tem A ATIVO: a referência é
$\theta^{hyb}$ = fluxo relu 2-neurônios até $t_{hit}^0$ (quando $z_A^{hyb}=0$),
depois A contribui $0$ e $\theta_B^{hyb}$ segue B-sozinho. Definições:
$R^{hyb}(t) = y_j - \mathrm{relu}(z_{B,j}^{hyb}(t))$,
$A^{hyb}(t) = \int_{t_{hit}}^t c\,R^{hyb}$, $t_{exit}^{hyb}$ = primeiro zero
transversal. [(H1)/(H2) reformuladas sobre $R^{hyb}$ — o passo 1 vale verbatim
com esta referência; o passo 3 usou B-sozinho-desde-0 como $R$ prescrito, o que
era legítimo para L3 (prescrito é prescrito) mas NÃO é a referência do teorema.]

## L4.1 (bootstrap por continuação)

Seja $\mathcal{P}_s$: "em $[t_{cross}, s]$ vale
$\sup\|\theta_B^\beta - \theta_B^{hyb}\| \le 2M/\beta$ e $z_A^\beta \le \beta^{-1}\log w_0$".
(i) $\mathcal{P}$ vale num vizinho de $t_{cross}$ (passo 2). (ii) Sob $\mathcal{P}_s$:
o acoplamento é $\le C\beta^{-1}$ nas janelas de quina e $\le C'\beta^{-2}$ no miolo
(L3.1-corolário), logo L1.2 devolve a cota MELHOR $M/\beta$ em $[t_{cross}, s]$
(com $M$ fixado pelas constantes incondicionais); e $R^\beta = R^{hyb} + O(\beta^{-1})$
(Lipschitz), logo L3.3 com $\delta = O(\beta^{-1})$ mantém a camada até
$t_{exit}^{hyb} - O(\beta^{-1})$. (iii) O conjunto de $s$ com $\mathcal{P}_s$ é
aberto e fechado em $[t_{cross}, t_{exit}^{hyb} - \epsilon]$ ⟹ vale até o fim. $\square$ (esboço)

## L4.2 (lei fechada do release acoplado — o enunciado central de T2'-min)

$$t_{release}^\beta \;=\; t_{exit}^{hyb} \;+\; O\!\left(\frac{1}{\beta}\right).$$

## Verificação (`passo4_verificacao.py`)

Sistema 4D ACOPLADO (softplus, sem nada prescrito) vs referência híbrida:
(a) $t_{exit}^{hyb}$ do cronograma híbrido; (b) $t_{release}^\beta$ medido no 4D
para $\beta \in \{300, 10^3, 3\cdot10^3, 10^4\}$; (c) $\beta\,(t_{release}^\beta -
t_{exit}^{hyb})$ limitado; (d) $\sup\|\theta_B^\beta - \theta_B^{hyb}\|\cdot\beta$
limitado no episódio (bootstrap consistente de ponta a ponta).

## Status

[TEOREMA-esboço] — script passou em 2026-07-24: $\beta(t_{rel}-t_{exit}^{hyb})
\to -14.4$ (converge a CONSTANTE: assintótica de 1ª ordem definida, mais forte
que o $O(1/\beta)$ pedido; o coeficiente é objeto de refinamento futuro via
análise de quina); $\beta\sup\|\Delta\theta_B\| \le 0.09$ de ponta a ponta;
$t_{hit}^{hyb}=0.1114$ consistente com o fit independente do passo 2 (0.11141).
Draft no passo 7.
