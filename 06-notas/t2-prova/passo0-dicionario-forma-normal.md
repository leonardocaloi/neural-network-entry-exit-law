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

- **I1 (equação da margem):** enquanto só a face j é relevante,
  `ż = −c·(softplus_β(z) − R(t))·σ(βz)` com `σ(s) = 1/(1+e^{−s})`.
  No regime de face (`z ≤ O(β⁻¹ log β)`): `softplus_β(z) = β⁻¹log(1+e^{βz})`
  contribui `O(β⁻¹ log(1+w))` ⟹ `ż = c·R(t)·σ(βz)·(1 + E_β)` com
  `|E_β| ≤ β⁻¹log(1+w)/|R|` (erro controlado do passo 1).
- **I2 (gauge w):** `w = e^{βz}` ⟹ `σ(βz) = w/(1+w)` e, em tempo rápido `τ = βt`:
  `dw/dτ = c·R(t)·w²/(1+w)·(1+E_β)` — **forma normal QUADRÁTICA de
  De Maesschalck–Schecter (1511.01815, eqs. 1.9–1.10)** com `f ≡ 1` (lenta:
  `dt/dτ = 1/β = ε`), `g = c·R/(1+w)`.
- **I3 (gauge u):** `u = e^{−βz} = 1/w` ⟹ `du/dτ = −c·R(t)·u/(1+u)·(1+E_β)`
  (a forma do ataque de 2026-07-24; I2 e I3 são o mesmo objeto).
- **I4 (lei de área, caso solúvel):** com `R` prescrito e `E_β = 0`, separação de
  variáveis em I3 dá `Φ(u(τ)) − Φ(u(τ₀)) = −c∫R`, `Φ(u) = log u + u` — release
  (`u` volta a O(1)) exatamente quando `∫R dt = 0` a menos de `O(β⁻¹ log β)`.
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
