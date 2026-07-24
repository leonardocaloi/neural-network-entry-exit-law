# T2' — Passo 1: subsistema B, hipóteses (H1)/(H2) e cota de feedback [TEOREMA-esboço]

**Convenção de sinal (FIXADA aqui para toda a prova):** $R(t) = y_j - f_{-A}(x_j;\theta_B(t))$
($R>0$ = sub-ajuste = empuxo pró-ativação, para FORA do cone). O relatório do
Kristiansen e o ataque de 2026-07-24 usavam $R=\mathrm{res}=f-y$ (sinal oposto).
Nesta convenção:

- **(H1) morte transversal:** $R^0(t_{hit}) < 0$ (overshoot empurrando A para dentro);
- **(H2) saída transversal:** $A(t) = \int_{t_{hit}}^{t} c\,R^0(s)\,ds$ tem primeiro
  zero em $t_{exit} < T$, com $R^0(t_{exit}) > 0$ e $A(t)<0$ em $(t_{hit},t_{exit})$;
- **(H3) margem única ativa** no episódio (multi-margem: passo 6 / aberto).

## L1.1 (fluxo B-sozinho e regularidade de $R^0$) — [TEOREMA elementar]

Com A removido, $\dot\theta_B = -\nabla_{\theta_B} L_{-A}$ é GF de função
$C^\infty$ por partes (softplus dos demais neurônios é $C^\infty$; se B for
softplus também, global $C^\infty$): existência/unicidade em $[0,T]$ clássicas;
$R^0 \in C^1$ com $\|\dot R^0\|_\infty \le C_R$ explícito via constantes de Gram.
No caso-instância (2 neurônios, ativa-ambos), B-sozinho é linear e $R^0$ é
combinação de 2 exponenciais — (H1)/(H2) verificáveis por conta fechada.

## L1.2 (cota de feedback) — o lema de Grönwall do passo

**Enunciado.** Seja $[t_1,t_2]\subset[0,T]$ intervalo em que a trajetória-$\beta$
mantém $w(t) = e^{\beta z(t)} \le w_{\max}$ (camada de sticking; pelo passo 3,
$w_{\max} = O(1)$ sob (H1)/(H2)). Então a contribuição de A ao campo de B é
$\mathrm{softplus}_\beta(z_A) \le \beta^{-1}\log(1+w_{\max})$ e, com $L$ =
constante de Lipschitz do campo de B em compacto contendo as trajetórias:

$$\sup_{[t_1,t_2]}\bigl\|\theta_B^\beta - \theta_B^0\bigr\|
\;\le\; \frac{e^{L (t_2-t_1)} - 1}{L}\; C_f\,\frac{\log(1+w_{\max})}{\beta}
\;=\; O\!\left(\beta^{-1}\right),$$

com $C_f$ = fator de Gram do acoplamento. **Nota:** isto MELHORA a estimativa
$O(\beta^{-1}\ln\beta)$ do relatório de leitura — o $\ln\beta$ só seria necessário
se $w_{\max}$ crescesse polinomialmente em $\beta$, o que (H2) + lei de área
excluem ($\Phi(u)=\log u + u$ limitada $\Rightarrow$ $u,w = O(1)$ no episódio).
A melhora é condicional ao passo 3 fechar $w_{\max}=O(1)$; manter
$O(\beta^{-1}\log\beta)$ como fallback seguro no enunciado de T2'-min.

**Prova (esboço numerado).** (1) $\dot\Delta = F_B(\theta_B^\beta) -
F_B(\theta_B^0) + G_A$, $\|G_A\| \le C_f\,\mathrm{softplus}_\beta(z_A)$;
(2) Lipschitz + Grönwall integral; (3) cota de $\mathrm{softplus}_\beta$ na
camada. $\square$ (completa no draft LaTeX; aqui verificada numericamente.)

## L1.3 (estabilidade de $t_{exit}$ sob perturbação do forcing) — alimenta o passo 4

Se $\tilde R = R^0 + \delta(t)$ com $\|\delta\|_\infty \le \delta_0$ pequeno, e
(H2) vale para $R^0$, então $\tilde A$ tem primeiro zero em
$\tilde t_{exit}$ com

$$|\tilde t_{exit} - t_{exit}| \;\le\; \frac{c\,\delta_0\,(t_{exit}-t_{hit})}{|A'(t_{exit})|}
\,+\,O(\delta_0^2), \qquad A'(t_{exit}) = c\,R^0(t_{exit}) > 0.$$

(Função implícita em $A$; transversalidade (H2) dá o denominador não-nulo.)

## Verificação (`passo1_verificacao.py`)

1. SymPy: campo de B, $\partial R/\partial \theta_B$, constantes de L1.2.
2. Instância do ataque ($p_A=(-0.5,0.27)$, $p_B=(1.144,0.261)$, $X=(0.5,1.5)$,
   $Y=(0.5,1.0)$): (H1)/(H2) verificadas numericamente no $R^0$ de B-sozinho.
3. Grönwall empírico: $\beta\cdot\sup\|\theta_B^\beta-\theta_B^0\| \le 0.093$
   e DECRESCENTE em $\beta \in \{300, 1000, 3000\}$ — a cota $O(\beta^{-1})$ de
   L1.2 confirmada com folga (o feedback real decai ainda mais rápido; o lema é
   conservador, o que basta para os passos 3–4).

## Status

[TEOREMA-esboço] — script passou integralmente em 2026-07-24: (H1) t_hit≈0.114
com R⁰<0; (H2) t_exit≈1.130 com R⁰>0 e área<0 no episódio; L1.2 com folga;
L1.3 dentro da cota. Prova completa em LaTeX no draft (passo 7).
