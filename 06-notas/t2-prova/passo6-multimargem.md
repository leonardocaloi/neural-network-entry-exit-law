# T2' — Passo 6: multi-margem sob (H3) [TEOREMA-esboço]

## L6.1 (redução à escalar)

Campo sobre $\theta_A$: $-\frac1n\sum_i \mathrm{res}_i\,\sigma(\beta z_i)\,(x_i,1)$.
Sob **(H3)** ($z_{A,i} \le -\delta_i < 0$, $i \ne j$, no episódio-limite):

1. termos $i \ne j$: $|\cdot| \le C\,e^{-\beta\delta_i}$ — exponencialmente pequenos,
   absorvidos por Grönwall (não alteram nenhuma taxa $O(1/\beta)$);
2. o termo $j$ é **rank-1 ao longo de $(x_j,1)$** (lema do salto, ataque de
   2026-07-24) ⟹ a componente de $\theta_A$ tangencial à face congela a taxa
   exponencial; a normal é EXATAMENTE o problema escalar dos passos 2–5. $\square$ (esboço)

## L6.2 ((H3) é aberta e verificável)

(H3) é condição aberta sobre $(dados, \theta_0)$; na instância verifica-se
numericamente com margem $\delta_1$ explícita.

## Declarado ABERTO (fora de T2'-min)

Episódios com 2+ margens simultâneas $O(\beta^{-1})$-ativas e o ápice do cone
(corank ≥ 2) — a quina-de-quinas onde prox-regularidade morre; conexão com o
estrato mais singular da SLT (deslizamento por weight decay, ataque §5) fica
para o programa, ciclo ≥3.

## Verificação (`passo6_verificacao.py`)

(a) simbólico: decomposição do campo e cota $e^{-\beta\delta}$; (b) numérico
(instância, episódio acoplado): $\max_t z_{A,1}(t) \le -\delta_1 < 0$ (H3 vale);
deriva tangencial $|\Delta(\theta_A \cdot \hat t)|$ é zero em precisão de
máquina ($e^{-\beta\delta_1} \sim 10^{-65} \ll \epsilon$) em ambos os $\beta$.

## Status

[TEOREMA-esboço] — script passou em 2026-07-24: (H3) vale com margem
$\delta_1 = 0.508$; deriva tangencial < 1e-10 (zero numérico) em ambos os β.
Ápice/corank≥2 declarado aberto. Draft no passo 7.
