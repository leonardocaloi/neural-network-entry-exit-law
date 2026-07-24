# T2' — Passo 2: quina de entrada [TEOREMA-esboço]

**Objetivo:** a trajetória-β cruza $\{w = w_0\}$ (i.e. $z = \beta^{-1}\log w_0$) em
$t_{cross}^\beta = t_{hit} + O(\beta^{-1})$, com estado $\theta^\beta(t_{cross})$ a
$O(\beta^{-1})$ do estado-limite $\theta^0(t_{hit})$ — o dado inicial uniforme que o
passo 3 (sticking) consome.

## L2.1 (transversalidade da chegada = (H1), sem hipótese nova)

Com A ativo na face $j$ e as demais margens afastadas, a margem-limite satisfaz
$\dot z^0 = -c\,(\mathrm{relu}(z^0) - R^0(t))$; em $z^0 = 0^+$:

$$\dot z^0(t_{hit}^-) = c\,R^0(t_{hit}) < 0 \quad \text{por (H1)}.$$

A chegada é transversal com velocidade $|c\,R^0(t_{hit})|$ — **(H0) ≡ (H1)**;
nenhuma hipótese adicional é necessária. [Verificado simbolicamente.]

## L2.2 (aproximação pré-quina)

Em $[0, t_{hit}-\delta]$ com $z^0 \ge \delta_z > 0$:
$|\mathrm{softplus}_\beta(z) - z| = \beta^{-1}\log(1+e^{-\beta z}) \le \beta^{-1}e^{-\beta\delta_z}$
e $1-\sigma(\beta z) \le e^{-\beta\delta_z}$ — campos exponencialmente próximos;
Grönwall dá $\|\theta^\beta - \theta^0\| = O(e^{-\beta\delta_z'})$ até a janela final.

## L2.3 (travessia da janela)

Na janela $z \in [\beta^{-1}\log w_0,\ \delta_z]$: $\sigma(\beta z) \ge \frac{w_0}{1+w_0}$
e $R(t) = R^0(t_{hit}) + O(\delta_z + \beta^{-1})$, logo por comparação (sub/super-soluções
com campos constantes):

$$t_{cross}^\beta - t_{hit} \;=\; O\!\left(\frac{1}{\beta}\right)
\quad\text{com constante}\quad K \sim \frac{C_{janela}}{c\,|R^0(t_{hit})|},$$

e $\|\theta^\beta(t_{cross}) - \theta^0(t_{hit})\| = O(\beta^{-1})$ (velocidades
limitadas na janela). $\square$ (esboço; draft no passo 7)

## Verificação (`passo2_verificacao.py`)

1. SymPy: L2.1 e as cotas de L2.2.
2. Numérico: $t_{cross}^\beta(w_0=1)$ para $\beta \in \{300, 1000, 3000, 10^4\}$;
   ajuste $t_{cross} = t_\infty + K/\beta$ (Richardson) com resíduo de ajuste
   pequeno e $K$ estável; estado na travessia converge com $O(\beta^{-1})$.

## Status

[TEOREMA-esboço] — script passou em 2026-07-24: L2.1 simbólico; ajuste
$t_{cross} = 0.11141 + 7.13/\beta$ com resíduo relativo 0.002 em
$\beta \in [300, 10^4]$; estado na travessia com $\beta\cdot\|\Delta\| \le 5.6$
limitado e decrescente. Prova completa no draft (passo 7).
