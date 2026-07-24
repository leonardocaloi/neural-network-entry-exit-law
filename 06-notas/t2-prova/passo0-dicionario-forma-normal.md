# T2' — Passo 0: dicionário e forma normal [TEOREMA]

**Leitor-alvo: eu, nos passos 1–5. Cada identidade daqui é usada adiante; nada
entra sem verificação simbólica (`passo0_verificacao.py`, ao lado).**

## Objeto

Rede rasa softplus: `f_θ(x) = Σ_k softplus_β(w_k x + b_k)`, loss
`L = (1/2n) Σ_i (f_θ(x_i) − y_i)²`, gradient flow `θ̇ = −∇L` (taxas por bloco
absorvidas em reescalonamento de tempo por bloco; SGLD fora do escopo T2'-min).

Neurônio A morto na face do dado j: margem `z ≡ z_{A,j} = θ_A·(x_j, 1)`.
Resíduo sem A: `R(t) = y_j − f_{−A}(x_j)` (sinal: R>0 = sub-ajuste = empuxo
pró-ativação). Constante de Gram: `c = ‖(x_j,1)‖²/n`.

## Identidades (I1–I4, todas verificadas simbolicamente)

- **I1 (equação da margem):** enquanto só a face $j$ é relevante,

  $$\dot z = -c\,\bigl(\mathrm{softplus}_\beta(z) - R(t)\bigr)\,\sigma(\beta z),\qquad \sigma(s) = \frac{1}{1+e^{-s}}.$$

  No regime de face ($z \le O(\beta^{-1}\log\beta)$), $\mathrm{softplus}_\beta(z) = \beta^{-1}\log(1+e^{\beta z})$ contribui $O(\beta^{-1}\log(1+w))$, logo

  $$\dot z = c\,R(t)\,\sigma(\beta z)\,(1 + E_\beta),\qquad |E_\beta| \le \frac{\log(1+w)}{\beta\,|R|}$$

  (erro controlado do passo 1).
- **I2 (gauge $w$):** $w = e^{\beta z} \Rightarrow \sigma(\beta z) = \tfrac{w}{1+w}$ e, em tempo rápido $\tau = \beta t$:

  $$\frac{dw}{d\tau} = c\,R(t)\,\frac{w^2}{1+w}\,(1+E_\beta)$$

  — **forma normal QUADRÁTICA de De Maesschalck–Schecter** (1511.01815, eqs. 1.9–1.10) com $f \equiv 1$ (lenta: $dt/d\tau = 1/\beta = \varepsilon$), $g = c R/(1+w)$.
- **I3 (gauge $u$):** $u = e^{-\beta z} = 1/w \Rightarrow$

  $$\frac{du}{d\tau} = -c\,R(t)\,\frac{u}{1+u}\,(1+E_\beta)$$

  (a forma do ataque de 2026-07-24; I2 e I3 são o mesmo objeto).
- **I4 (lei de área, caso solúvel):** com $R$ prescrito e $E_\beta = 0$, separação de variáveis em I3 dá

  $$\Phi\bigl(u(\tau)\bigr) - \Phi\bigl(u(\tau_0)\bigr) = -c\!\int R\,dt,\qquad \Phi(u) = \log u + u$$

  — release ($u$ volta a $O(1)$) exatamente quando $\int R\,dt = 0$, a menos de $O(\beta^{-1}\log\beta)$.
  [Âncora exatamente integrável, papel do Apêndice B de Kristiansen 1603.01821.]

## Dicionário com a literatura (fixado)

| nosso | DM-S 1511.01815 | Kristiansen 1603.01821 |
|---|---|---|
| 1/β | ε (caso quadrático §1.9) | ε/2 (classe tanh, §3.21) |
| w = e^{βz} | variável rápida quadrática | q do Method 1 (var. exponencial) |
| face de morte (lado ≡0) | **fora de escopo** (turning point transversal) | **fora de escopo** (fold aditivo) |
| R(t) acoplado | fora de escopo (autônomo) | fora de escopo (autônomo) |
| release | eq. 1.4 (lei ∫=0) | Thm 3.5 (local, sem integral) |

## O que o passo 0 entrega aos próximos

- Passo 1 usa I1 (cota de E_β) e o fluxo de B (exato enquanto A morto — L1-v2).
- Passo 3 usa I2/I4 (DM-S quadrático ou quadratura direta).
- Passo 5 usa I2 para formular o híbrido em θ-espaço via `z = β⁻¹ log w`.

## Status

[TEOREMA] — `passo0_verificacao.py` passou integralmente em 2026-07-24
(identidades elementares, verificação simbólica commitada ao lado).
