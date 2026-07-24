# T2' — Passo 5: quina de saída, objeto-limite e enunciado final [TEOREMA-esboço]

## Def 5.1 (o objeto híbrido $\theta^\infty$)

Sob (H1)/(H2)/(H3), $\theta^\infty: [0,T] \to \mathbb{R}^d$:

1. $[0, t_{hit}]$: fluxo relu ativo (2 neurônios);
2. $[t_{hit}, t_{exit}]$: **$\theta_A$ CONSTANTE** ($= \theta_A(t_{hit})$, com
   $z_A = 0$ automaticamente) e $\theta_B$ = fluxo B-sozinho — o neurônio morto
   congela; nada de deslizamento (o campo sobre $\theta_A$ na camada é
   $O(\beta^{-1})$ pelo corolário de L3.1; deslizamento genuíno exigiria força
   tangencial $O(1)$, ex. weight decay — fora de T2'-min);
3. $[t_{exit}, T]$: fluxo relu ativo retomado de $(\theta_A(t_{hit}), \theta_B^{hyb}(t_{exit}))$.

Contraste com a C1 morta: o limite NÃO é complementaridade sem memória — é
**congela + portão com memória** ($\int R = 0$).

## L5.1 (quina de saída)

Simétrica ao passo 2: pós-release, $z$ cresce com velocidade $c\,R^{hyb}(t_{exit})>0$
(transversal por (H2)); em janela $O(\beta^{-1})$ o campo volta a ser
exponencialmente próximo do relu ativo; Grönwall cola o trecho 3. $\square$ (esboço)

## T2'-min (enunciado final)

**Teorema-alvo.** Rede rasa, GF com $\sigma_\beta$ = regularização sigmoide
monótona de relu; dados genéricos; (H1) $R^{hyb}(t_{hit}) < 0$ transversal;
(H2) primeiro zero transversal de $A^{hyb}$ em $t_{exit} < T$; (H3) margem única
ativa no episódio. Então, $\forall \delta > 0$:

$$\sup_{t \in [0,T] \setminus (B_\delta(t_{hit}) \cup B_\delta(t_{exit}))}
\bigl\|\theta^\beta(t) - \theta^\infty(t)\bigr\| = O_\delta(\beta^{-1}),
\qquad t_{release}^\beta = t_{exit} + O(\beta^{-1}).$$

**Seleção:** o limite é do GF-$\sigma_\beta$; relu-GD discreto congela para sempre
(limites não comutam) — o teorema é sobre a família regularizada.

## L5.2 (corolário de independência da regularização — inclui GELU)

A lei do release depende só de $\int R\,dt = 0$, não da forma de $\sigma$
(análogo codim-1 da independência-de-$\pi$ de Guglielmi–Hairer): para qualquer
sigmoide monótona com caudas rápidas, a primitiva $\Phi_\sigma$ muda, o critério
de cruzamento não. **Caso de peso: $\sigma = \Phi_{gauss}(\beta z)$ (classe erf),
cuja ativação $g_\beta(z) = z\Phi(\beta z) + \varphi(\beta z)/\beta$ é o GELU
regularizado** — se verificar, redes GELU obedecem à MESMA lei de área de
ressurreição. [Ponte direta com prática moderna.]

## Verificação (`passo5_verificacao.py`)

4D acoplado com ativação classe-erf (GELU-β): $t_{release}$ para
$\beta \in \{300, 3000\}$ vs $t_{exit}^{hyb}$ (o MESMO da referência relu do
passo 4) — gap $O(1/\beta)$ ⟹ L5.2 confirmada na instância.

## Status

[TEOREMA-esboço] — script passou em 2026-07-24: classe erf (GELU-β) acoplada
libera no MESMO $t_{exit}^{hyb}$ da referência relu, $\beta\cdot$gap ≈ −8
limitado (gap absoluto 0.027→0.003 em uma década) ⟹ L5.2 (independência da
regularização, incluindo GELU) confirmada na instância. Def 5.1 + T2'-min em
forma final. Draft no passo 7.
