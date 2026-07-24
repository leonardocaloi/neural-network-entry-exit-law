# Prova completa — Lema 1 (forma normal e quadratura aditiva) [TEOREMA]

**Primeiro lema do draft-v1 promovido de esboço a prova completa. Passos
numerados; identidades verificadas em `prova_L1_verificacao.py`.**

## Enunciado (L1, forma final)

Seja $z(t)$ a margem do neurônio $A$ na face $j$ sob GF-softplus com as demais
margens satisfazendo (H3)/(H4), $u = e^{-\beta z}$, $w = 1/u$,
$\Phi(u) = \log u + u$, $c = \|(x_j,1)\|^2/n$ e $R(t)$ o resíduo excluindo $A$.
Então vale a identidade **exata** (sem hipótese sobre sinais de $R$):

$$\frac{d\Phi(u)}{dt} \;=\; -\,c\,\beta\,R(t) \;+\; c\,\log(1+w) \;+\; \mathcal{E}_{(H3)}(t),$$

com $|\mathcal{E}_{(H3)}| \le C_3\,\beta\,e^{-\beta\delta_1}$, e a correção
aditiva satisfaz, sob (H1)/(H2), a cota integral

$$\int_{t_{hit}}^{t_{release}} c\,\log(1+w(s))\,ds \;=\; O\!\left(\frac{\log\beta}{\beta}\right).$$

## Prova

1. **Equação da margem (exata sob (H3)).** $\dot z = (x_j,1)\cdot\dot\theta_A$.
   O campo em $\theta_A$ é $-\frac1n\sum_i \mathrm{res}_i\,\sigma(\beta z_i)(x_i,1)$;
   o termo $i=j$ dá $-c\,(\mathrm{softplus}_\beta(z) - R)\,\sigma(\beta z)$
   (pois $\mathrm{res}_j = \mathrm{softplus}_\beta(z) - R$ por definição de $R$);
   os termos $i \ne j$ são limitados por $C_3' e^{-\beta\delta_1}$ (L6.1). Logo
   $\dot z = -c(\mathrm{softplus}_\beta(z) - R)\sigma(\beta z) + O(e^{-\beta\delta_1})$.

2. **Cadeia exata.** $\dot u = -\beta u \dot z$ e $\Phi'(u) = \frac{1+u}{u}$;
   além disso $\sigma(\beta z) = \frac{1}{1+u}$ (identidade do passo 0). Então

   $$\frac{d\Phi}{dt} = \frac{1+u}{u}\cdot(-\beta u)\cdot
   \Bigl[-c(\mathrm{softplus}_\beta(z)-R)\tfrac{1}{1+u}\Bigr]
   = c\,\beta\,\bigl(\mathrm{softplus}_\beta(z) - R\bigr).$$

3. **Identidade aditiva.** $\beta\,\mathrm{softplus}_\beta(z) = \log(1+e^{\beta z})
   = \log(1+w)$. Substituindo: $d\Phi/dt = -c\beta R + c\log(1+w)$, mais o erro
   (H3) do passo 1 multiplicado pelos mesmos fatores (limitado por
   $C_3\beta e^{-\beta\delta_1}$). **Nenhuma divisão por $R$ ocorre; o zero
   interior de $R$ (forçado por (H1)–(H2) via TVI) é inócuo.** $\;\checkmark$ SymPy.

4. **Cota da correção em três janelas.** No episódio, $w = 1/u$ com
   $u = c\beta|A(t)|(1+o(1))$ fora de vizinhanças dos extremos (L4):
   - *miolo* ($|A| \ge a_0(\delta)$): $\log(1+w) \le w \le \frac{1}{c\beta a_0}$;
     contribuição $\le \frac{T}{\beta a_0} = O(\beta^{-1})$;
   - *janelas dos extremos* (perto de $t_{hit}$ e $t_{exit}$, onde $A$ tem zeros
     transversais com derivadas $c\,R(t_{hit}) \ne 0$, $c\,R(t_{exit}) \ne 0$):
     $|A(t)| \ge \kappa\,|t - t_\ast|$ localmente, logo
     $\int \log(1+w) \lesssim \int_{1/\beta}^{\epsilon} \frac{ds}{c\beta\kappa s}
     = \frac{\log(\beta\epsilon)}{c\beta\kappa}$ — e nos trechos de largura
     $O(\beta^{-1})$ colados aos extremos, $\log(1+w) \le \log 2 + O(1)$
     (ali $w = O(1)$), contribuição $O(\beta^{-1})$.
   Somando: $O(\log\beta/\beta)$, com constante
   $C_4 = \frac{2}{c\,\min(|R(t_{hit})|, R(t_{exit}))} + \frac{T}{a_0}$. $\square$

## Consequência imediata (usada em L4/L5)

$\Phi(u(t)) = \Phi(u_0) - c\beta A(t) + O(\log\beta)$ uniformemente no episódio
— a forma que sustenta a profundidade $u = c\beta|A|(1+o(1))$ em compactos e o
release em $t_{exit} + O(\beta^{-1})$ (o erro $O(\log\beta)$ em $\Phi$ traduz-se
em $O(\log\beta/(\beta\,c\,R(t_{exit})))$... refinando pela transversalidade:
$O(\beta^{-1}\log\beta)$ bruto; a taxa $O(\beta^{-1})$ fina do release vem do
sanduíche de L4 aplicado só à janela terminal, onde a correção já é $O(\beta^{-1})$).

## Status

[TEOREMA] — `prova_L1_verificacao.py` passou em 2026-07-24: identidade aditiva
exata confirmada simbolicamente; correção integral com métrica normalizada
estável em [8.2, 10.6] < C₄ teórica ≈ 30, em β ∈ [300, 10⁴]. Primeiro lema do
draft com prova COMPLETA (não esboço).
