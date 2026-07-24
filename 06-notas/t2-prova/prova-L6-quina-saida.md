# Prova completa — Lema 6 (quina de saída, colagem e unicidade-para-frente) [TEOREMA]

## Enunciado (L6, forma final)

Sob (H1)–(H4), pós-release: **(i)** a trajetória-β atravessa a quina de saída em
tempo $\tfrac{1}{\beta v'}(1+o(1))$ com $v' = c\,R^{hyb}(t_{exit})$ (espelho de
L3 — mesma identidade $\tfrac{1-\sigma}{\sigma} = e^{-s}$, mesmo fator $3/2$ no
peso); **(ii)** o fluxo relu reiniciado de $(\theta_A(t_{hit}), \theta_B^{hyb}(t_{exit}))$
tem unicidade-para-frente: o ponto de reinício está na FRONTEIRA ($z=0$) com
campo apontando transversalmente para DENTRO da região ativa
($\dot z = v' > 0$ por (H2)) — a trajetória entra imediatamente no domínio onde
o campo é $C^\infty$-Lipschitz, e a solução "ficar morto" de Filippov é
descartada pela seleção do limite (Def. 5.1 do draft: $\theta^\infty$ SELECIONA
o ramo ativante — este é o conteúdo da não-comutação com EJRW23); **(iii)** em
$[t_{release}, T]$, Grönwall clássico cola: $\|\theta^\beta - \theta^\infty\| =
O(\beta^{-1}\log\beta)$ (herdando o erro acumulado do episódio, dominado pela
coordenada normal de $\theta_A$).

## Prova

1. **(i)** Espelho de L3 com $v \to v'$: transversalidade por (H2), identidade
   da travessia idêntica, peso $\tfrac{3/2}{\beta v'}$. $\checkmark$ (mesmas
   identidades SymPy de L3.)
2. **(ii)** No ponto de reinício, o campo ativo é localmente Lipschitz do lado
   $z \ge 0$ e $\dot z(t_{exit}^+) = v' > 0$: a solução entra em $\{z > \epsilon t\}$
   e a unicidade de Picard vale para frente a partir de qualquer $t > t_{exit}$;
   a única alternativa de Filippov (permanecer em $z \le 0$ com campo nulo)
   é incompatível com o limite dos $\theta^\beta$ (L5: release em
   $t_{exit} + O(1/\beta)$) — a seleção está PROVADA pelo próprio L5, não
   postulada. $\checkmark$
3. **(iii)** Dado inicial da colagem: $\|\theta^\beta(t_{release}) -
   \theta^\infty(t_{exit})\| = O(\beta^{-1}\log\beta)$ (posição de $\theta_A$
   na camada $O(\log\beta/\beta)$ + $\theta_B$ de L2 + deslocamento temporal
   $O(1/\beta)$ × velocidades limitadas); campos exponencialmente próximos no
   interior ativo (como L2.2 de passo 2); Grönwall em $[t_{release}, T]$. $\square$

## Verificação (`prova_L6_verificacao.py`)

Numérica direta da colagem: para $\beta \in \{1000, 3000\}$, medir
$\sup_{[t_{rel}+0.05,\ T]}\|\theta^\beta(t) - \theta^{relu}(t)\|$ onde
$\theta^{relu}$ é o fluxo relu reiniciado de $(\theta_A(t_{hit}^\beta),
\theta_B^{hyb}(t_{exit}))$ — deve escalar $O(\log\beta/\beta)$ (métrica
$\sup\cdot\beta/\log\beta$ limitada e não-crescente).

## Status

[TEOREMA] — script passou em 2026-07-24: colagem com sup = 1.6e-3 / 5.5e-4
(β = 1000/3000), métrica β/log β estável em 0.21–0.23 ⟹ O(log β/β). REGISTRO
INSTRUTIVO do debug: a 1ª versão da referência reiniciava com z_A = −1.1e-6
(overshoot de um passo de Euler do loop pré-hit) e caía no ramo CONGELADO de
Filippov (sup ≈ 0.039 constante — a não-unicidade de EJRW23 demonstrada por
acidente); a referência correta exige projeção exata na face + nudge ativante
acima do overshoot. Isto é evidência concreta de que a seleção do ramo é
condição REAL do enunciado (Def. 5.1), não pedantismo.
