# Prova completa — Lema 2 (subsistema B, (H4) e feedback) [TEOREMA]

## Enunciado (L2, forma final — REFINADO sobre o esboço)

Sob (H1)–(H4), no episódio $[t_{cross}, t_{release}]$:

**(a)** o campo de B-sozinho tem padrão de ativação constante (H4), logo é o campo
de mínimos quadrados LINEAR em $\theta_B$; $R^{hyb}$ é afim em $\theta_B$ e
$C^1$ em $t$, com $\|\dot R^{hyb}\|_\infty \le \|(x_j,1)\|\cdot\|G_B\|\cdot
\mathrm{diam} =: C_R$ explícito.

**(b)** o acoplamento de A no campo de B satisfaz
$\|G_A(t)\| \le C_f\,\mathrm{softplus}_\beta(z(t)) + C_3 e^{-\beta\delta_1}$, e
por Grönwall integral com a cota de L1:

$$\sup_{[t_{cross},\,t_{release}]}\bigl\|\theta_B^\beta - \theta_B^{hyb}\bigr\|
\;\le\; e^{L_B T}\Bigl[\tfrac{C_f}{c\beta}\!\int c\log(1+w)\,dt + T C_3 e^{-\beta\delta_1}\Bigr]
\;=\; O\!\left(\frac{\log\beta}{\beta^{2}}\right)$$

— **uma ordem melhor** que o $O(\beta^{-1})$ do esboço (L1.2). Consequência:
$R^\beta = R^{hyb} + O(\log\beta/\beta^2)$, o que folga ainda mais o sanduíche
de L4 e o bootstrap de L5.

## Prova

1. **(H4) ⟹ campo linear.** Padrão constante: o campo de B é
   $-G_B(\theta_B - \theta_B^\ast)$ com $G_B$ = Gram dos dados ativos —
   $C^\infty$, Lipschitz $L_B = \lambda_{max}(G_B)$ (na instância: $2.13$).
   $R^{hyb} = y_j - (x_j,1)\cdot\theta_B$ (afim) ⟹ (a). $\checkmark$
2. **Cota do acoplamento.** A perturbação de A em $\mathrm{res}_i$ é
   $\mathrm{softplus}_\beta(z_{A,i})$; sob (H3), $i \ne j$ dá $C_3 e^{-\beta\delta_1}$
   e $i = j$ dá $\mathrm{softplus}_\beta(z)$; fatores de Gram dão $C_f$. $\checkmark$
3. **Grönwall integral.** $\dot\Delta \le L_B\Delta + \|G_A\|$ ⟹
   $\Delta(t) \le \int e^{L_B(t-s)}\|G_A(s)\|ds$. Como
   $\mathrm{softplus}_\beta = \beta^{-1}\log(1+w)$ e L1 dá
   $\int c\log(1+w)\,dt = O(\log\beta/\beta)$: total $O(\log\beta/\beta^2)$. $\square$
4. **Verificação da predição refinada nos dados do passo 1** (não recomputados —
   os números de 2026-07-24): $\sup\|\Delta\theta_B\| = 0.0925/\beta, 0.0398/\beta,
   0.0161/\beta$ em $\beta = 300, 1000, 3000$ ⟹ métrica
   $\sup\cdot\beta^2/\log\beta = 4.87, 5.76, 6.03$ — estável, confirmando o
   expoente refinado (sob $O(\beta^{-1})$ puro a métrica cresceria como
   $\beta/\log\beta$, i.e. $\times 6.5$ na década; cresceu $\times 1.24$).

## (H4) na instância (margens explícitas)

Verificadas em `prova_L2_verificacao.py`: margens de B afastadas de zero no
híbrido; padrão de A constante pré-hit; sem segundo episódio até $T$.

## Status

[TEOREMA] — script passou em 2026-07-24: campo/R afins com $L_B = 2.133$;
taxa refinada $O(\log\beta/\beta^2)$ com métrica $\sup\cdot\beta^2/\log\beta$
estável em $[4.20, 5.70]$ (β ∈ [300, 3000]); margens de (H4): $\min z_B = 0.39$.
Segundo lema com prova completa. O refinamento deve ser propagado ao draft-v2
(o Lemma 2 do draft-v1 afirma $O(\beta^{-1})$ — correto mas não-ótimo).

**ERRATA 2026-07-26:** a linha da métrica $4.87, 5.76, 6.03$ acima é de run antiga; o script canônico `prova_L2_verificacao.py` re-executado dá [4.20, 5.70] (β∈[300,3000]) — o valor do Status e dos drafts. Mantida por registro.
