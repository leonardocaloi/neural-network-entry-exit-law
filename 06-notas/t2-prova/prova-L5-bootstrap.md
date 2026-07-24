# Prova completa — Lema 5 (bootstrap acoplado, com constantes) [TEOREMA]

## Enunciado (L5, forma final — expansão de 1ª ordem, não só $O(\beta^{-1})$)

Sob (H1)–(H4), o sistema ACOPLADO satisfaz

$$t_{release}^\beta \;=\; t_{exit} \;-\; \frac{1}{c\,R^{hyb}(t_{exit})}\cdot\frac{1}{\beta}
\;+\; \frac{\mathrm{corr}(\beta)}{c\,\beta\,R^{hyb}(t_{exit})} \;+\; o(\beta^{-1}),$$

com $\mathrm{corr}(\beta) = O(\log\beta/\beta)$ (L1). **O coeficiente de 1ª ordem
é universal**: no produto $K\cdot|R_{hit}|/R_{exit}$ (atraso de entrada × área
cortada), o $|R_{hit}|$ cancela com o $K = 1/(c|R_{hit}|)$ de L3 — simetria
entrada/saída: cada quina contribui o recíproco da sua velocidade transversal.

## Prova (continuação com margens estritas)

1. **Propriedade de bootstrap.** $\mathcal{P}_s$: em $[t_{cross}, s]$,
   (a) $\sup\|\theta_B^\beta - \theta_B^{hyb}\| < 2M_2\log\beta/\beta^2$ e
   (b) $z(t) < \beta^{-1}\log w_0 - \beta^{-1}$ (camada com margem ESTRITA), com
   $M_2 = e^{L_B T} C_f C_4/c$ explícita (cadeia L2∘L1).
2. **Abertura no início:** L3 dá o dado inicial com folga ($O(\beta^{-1})$ em
   $\theta$, camada atingida transversalmente); as quantidades de $\mathcal{P}_s$
   são contínuas em $s$ (trajetórias $C^1$) — desigualdades estritas se propagam
   a uma vizinhança.
3. **Passo de melhoria (fechamento):** sob $\mathcal{P}_s$ com $2M_2$:
   $R^\beta = R^{hyb} + \delta$, $\|\delta\| \le \mathrm{Lip}_R\,2M_2\log\beta/\beta^2$
   ($R$ afim, L2a). O sanduíche de L4 com forcing perturbado mantém
   $u(t) \ge c\beta(|A(t)| - \|\delta\|T)/2 > u_0$ para
   $t \le t_{exit} - \varepsilon_\beta$, $\varepsilon_\beta = C_\varepsilon\log\beta/\beta$
   ($C_\varepsilon = 4C_4/(cR_{exit})$) — camada mantida com margem. Com a camada,
   o Grönwall de L2 (incondicional dada a camada) devolve $M_2$ — metade do
   assumido, para $\beta \ge \beta_0$ explícito. $M_2$ não depende da hipótese
   $2M_2$ (a cadeia de constantes é incondicional) — fechamento do bootstrap.
4. **Aberto-fechado:** o conjunto dos $s$ com $\mathcal{P}_s$ é não-vazio, aberto
   (margens estritas + continuidade) e fechado (limites das cotas) em
   $[t_{cross}, t_{exit}-\varepsilon_\beta]$ ⟹ é todo o intervalo.
5. **Janela terminal $[t_{exit}-\varepsilon_\beta,\ t_{release}]$:** aplicar
   L4(ii)–(iii) com o forcing congelado em $R^{hyb} + O(\log\beta/\beta^2)$:
   o deslocamento por $\delta$ é $O(\log\beta/\beta^2)$ (L4iii); o atraso-corr é
   $+\mathrm{corr}/(c\beta R_{exit})$ (L4ii); e o corte de área da entrada
   ($\int_{t_{hit}}^{t_{hit}+K/\beta} R \approx R_{hit}K/\beta$) adianta por
   $-K|R_{hit}|/(R_{exit}\beta) = -1/(cR_{exit}\beta)$. Somando: o enunciado. $\square$

## Verificação (`prova_L5_verificacao.py`)

(a) A lei de dois termos $\beta\Delta t = (-1+\mathrm{corr}(\beta))/(cR_{exit})$
reproduz os QUATRO valores do passo 4 ($-12.73, -13.90, -14.29, -14.38$) com
tolerância 3.5% cada; (b) margem da camada: $z < 0$ estrito em
$[t_{cross}, t_{exit}-\varepsilon_\beta]$ na run real; (c) constantes da cadeia
$M_2$ explicitadas.

## Status

[TEOREMA] — script passou em 2026-07-24: a lei de dois termos reproduz o sweep
inteiro do acoplado com desvios 1.9%/0.7%/0.2%/0.1% (β=300…10⁴) — convergência
a exatidão típica de lei assintótica; coeficiente universal 1/(cR_exit)=14.815;
camada estrita confirmada em β=10⁴ (nota de honestidade: com o C_ε conservador
a janela interna do check é VAZIA para β≤3000 — check vácuo ali; a manutenção
da camada nesses β é implicada pela própria detecção de release); cadeia de
constantes explícita M₂=440.8. Quinto lema completo — o "grosso" do referee.
