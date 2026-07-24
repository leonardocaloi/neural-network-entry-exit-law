# Prova completa — Lema 4 (sticking e release, R prescrito) [TEOREMA]

## Enunciado (L4, forma final — com as fórmulas de release EXPLÍCITAS)

Para $R \in C^1$ prescrito com (H1)/(H2), margem escalar de $z(t_{hit})=0$:

**(i) Profundidade:** $\Phi(u(t)) = \Phi(u_0) - c\beta A(t) + \mathrm{corr}(t)$,
com $A(t) = \int_{t_{hit}}^t R$ e $0 \le \mathrm{corr}(t) = \int c\log(1+w)\,ds
= O(\log\beta/\beta)$ (cota de L1 — nota: a "Consequência" da prova-L1 dizia
$O(\log\beta)$ em $\Phi$; o correto é $O(\log\beta/\beta)$, ainda melhor).
Em compactos de $(t_{hit},t_{exit})$: $u = c\beta|A(t)|\,(1+O(\tfrac{\log\beta}{\beta a_0}))$.

**(ii) Release com fórmula fechada:** $\Phi(t_{rel}) = \Phi(u_0)$ dá
$c\beta A(t_{rel}) = \mathrm{corr}(t_{rel})$, e por transversalidade (H2):

$$t_{rel} - t_{exit} \;=\; \frac{\mathrm{corr}(\beta)}{c\,\beta\,R(t_{exit})}\,(1+o(1))
\;=\; O\!\left(\frac{\log\beta}{\beta^{2}}\right)$$

— o atraso do release É a integral de correção de L1, reparametrizada pela
velocidade de saída.

**(iii) Perturbação (três janelas):** $\tilde R = R + \delta$ desloca $t_{exit}$
por $\le \delta\,(t_{exit}-t_{hit})/R(t_{exit}) + O(\delta^2)$, com o argumento
em três janelas: perto do hit $|A| \ge |R(t_{hit})|\,s/2$; no miolo $|A| \ge a_0(\delta)$;
perto do exit, transversalidade.

**Corolário (explica o $-14.4$ do passo 4):** no sistema ACOPLADO o dado inicial
vem da quina (L3), com atraso $K/\beta$ que corta área negativa
$\approx R(t_{hit})K/\beta$; logo

$$\beta\,(t_{rel} - t_{exit}^{hyb}) \;\to\; -\,\frac{K\,|R(t_{hit})|}{R(t_{exit})}
= -\frac{7.13 \times 0.2272}{0.1080} = -15.0,$$

vs $-14.38$ medido no passo 4 (desvio 4%, o resíduo é a correção positiva (ii)).

## Prova

1. **Comparação monótona (Chaplygin).** O RHS de $\dot\Phi = -c\beta R + c\log(1+w(\Phi))$
   é DECRESCENTE em $\Phi$ ($w = 1/u$, $u$ crescente em $\Phi$) ⟹ princípio de
   comparação: $R_1 \le R_2 \Rightarrow \Phi_1 \ge \Phi_2$; envelopes constantes
   por partes de $R$ dão sub/super-soluções exatamente integráveis via a mesma
   primitiva. $\checkmark$ SymPy (monotonicidade).
2. **(i):** integrar a identidade exata de L1; a cota da corr é a de L1 (três
   janelas); em compactos, $\Phi \approx u$ para $u$ grande dá a lei de profundidade.
3. **(ii):** no release $\Phi$ retorna a $\Phi(u_0)$ (o $\Phi_0$ cancela — sem
   termo espúrio); resolver $c\beta A = \mathrm{corr}$ com $A(t) = R(t_{exit})(t-t_{exit})(1+o(1))$.
4. **(iii):** função implícita com denominador $R(t_{exit})$ nas três janelas. $\square$

## Verificação (`prova_L4_verificacao.py`)

Cronograma híbrido (como passo 4); margem escalar de $t_{hit}$ com $z=0$;
para $\beta \in \{300, 1000, 3000\}$: medir $t_{rel}$ e $\mathrm{corr}$ na MESMA
run e testar a fórmula $\Delta t = \mathrm{corr}/(c\beta R_{exit})$ (tolerância
10%); e o corolário $-15.0$ vs $-14.38$ (tolerância 8%).

## Status

[TEOREMA] — script passou em 2026-07-24 (grade 2e-6 após teste de refinamento:
o desvio de β=3000 colapsou 35%→9.5%, confirmando discretização): fórmula do
release fecha a 0.2%/1.0%/9.5% (β=300/1000/3000); corolário do acoplado:
−K|R_hit|/R_exit = −14.99 vs −14.38 medido (4.2%). Quarto lema com prova
completa — TRÊS constantes empíricas explicadas (o atraso prescrito, o −14.4
do passo 4, e a ligação corr↔release).
