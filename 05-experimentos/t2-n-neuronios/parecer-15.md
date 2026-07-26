# Parecer 15 — Degrau 1, draft LaTeX do teorema de forcing

2026-07-26. Parecerista adversarial. Objeto: `06-notas/t2-prova/teorema-forcing-draft.tex`
(reescrita completa dos lemas L1–L6 do paper-v3, Apêndice A, com o residual `R` tratado como
forcing prescrito gerado por um subsistema vivo afim arbitrário de ordem `m`).
Scripts do parecer: `parecer15_o5.py`, `parecer15_o5b.py` (este diretório).

**Veredito: DRAFT COM REPAROS.**

## O5 — FECHADO

Conta refeita independentemente (não reproduzida do draft, refeita do zero):

- Pré-hit, o sistema completo é **identidade afim** com Gram `G ⪰ 0` (padrão de ativação congelado,
  readout fixo): `δθ̇ = −G δθ + E`, logo `‖δθ(t)‖ ≤ ∫₀ᵗ‖E‖` sem nenhum fator exponencial.
- Bloco `A` de `E`: `O(res · e^{−βz})` — **sem** `β⁻¹`. É defeito de *derivada* (σ(βz) vs. 1{z>0}),
  não de saída.
- Bloco `B` de `E`: **só** o defeito de *saída* de `A`, `C_f β⁻¹ e^{−βz⁰}`. O defeito de derivada de
  `A` **não aparece** em `∂L/∂θ_B` — os gradientes de `B` envolvem apenas os `σ` das unidades vivas.
  Este é o ponto que faz `ρ_B` ganhar um `β⁻¹` a mais que `ρ_A`.
- Forma pontual via `∫₀^{t_hit} e^{−βv(t_hit−s)} ds = (βv)⁻¹(1+o(1))`.
- Separação `−G_B Δ − G_{BA} δθ_A` com o cruzado tratado como **fonte legítima** (repetição do passo 3
  sobre `Δ` sozinho, usando `G_B ⪰ 0`).
- Conclusão: `ρ_A(t_hit) = O(β⁻¹)`, `ρ_B(t_hit) = O(β⁻²)`. **Confere com o draft.**

Verificação numérica (RK4, 12 instâncias, `m ∈ {4,8,16}`, `β ∈ {1e3, 2e3, 4e3, 8e3}`):

- `ρ_A(t_hit)·β` **satura** (~0.53) — expoente 1 confirmado.
- `ρ_B(t_hit)·β²` **satura** — expoente 2 confirmado.
- `β²·v·ρ_B ≈ 0.198 ± 0.05`, **invariante em m** — a constante é governada pela velocidade de
  chegada `v = c|R(t_hit)|`, não pelo número de unidades vivas.

## O1 — cadeia de β₀ extraída

Cinco condições, rastreadas uma a uma:

| # | condição | origem | ordem em `m` |
|---|----------|--------|--------------|
| 1 | `β₁ = δ_B⁻¹ log(m²Hβ²)` | L2′ passo 2b | `log m` |
| 2 | termo além-de-todas-as-ordens `≤ ½M₂logβ/β²` | L5′ (4.1) | `log m` |
| 3 | `‖δ‖∞ H ≤ a₀/2`, com `Lip_R·M₂ ≲ m` | L5′ (4.2) | `√m` |
| 4 | `κ C_ε logβ/β ≥ 2 Lip_R M₂ H logβ/β²` | L5′ (4.2), colar/terminal | **linear em `m` (binding)** |
| 5 | `β ≳ m` escondido no `(1+o(1))` de (GS) | L2′ passo 2b | linear em `m` |

Pior caso: **linear em `m`**. Logo `β₀ ≲ O(m)`, polinomial — dado o forcing fixo
(`a₀`, `R(t_exit)`, `T_ep`). **Achado**: a eq. (GS) esconde a condição `β ≳ m` dentro do `(1+o(1))`;
tem de ser explicitada como condição de `β₀`, não escondida em notação assintótica.

## O2 — adjudicado CONTRA o draft (não é dívida, é escopo)

Relaxar (H4′) para a versão por-pedaços **quebra a identidade afim**, que é a espinha do L2′:
se o `β`-fluxo e o híbrido chaveiam em instantes *diferentes*, a diferença dos drifts vira
`−G_B Δ − (G_B − G_B′)θ`, e o segundo termo é **linear em ρ** (não em `Δ` com sinal), reintroduzindo
Grönwall com expoente `Σ_chaveamentos ‖ΔG‖‖θ‖/v_k`. Com até `mn` chaveamentos isso é **exponencial em
`m`** no pior caso — exatamente o que o passo PSD tinha eliminado.

Portanto (H4′) não é dívida a pagar: é **definição da janela**. O teorema vale em qualquer janela
maximal de padrão vivo constante contendo `t_hit` e `t_exit`, verificável a posteriori por uma
integração híbrida. Reescrever (O2) nesses termos.

## Descasamento enunciado × prova no item (iv)

O enunciado dá `sup_{[0,H]}‖θ_B^β − θ_B^hyb‖ ≤ M₂ logβ/β²`, mas a prova entrega `2M₂` (constante da
hipótese de indução, melhorada a `3/2 M₂`) e só em `[t_hit, t_rel]`; em `[t_rel, H]` o L6(iii) dá
Grönwall clássico a taxa `O(logβ/β)`, que **não** sustenta o `sup` em `[0,H]` a `O(logβ/β²)`.

Reparo: (a) constante `2M₂` no enunciado; (b) refazer a estimativa PSD em `[t_rel, H]` — pós-release
o sistema volta a ser afim com Gram PSD e a repetição do passo 3 do L2′ dá a taxa certa. Escrever
por extenso. (Restringir o enunciado a `[0, t_rel]` seria a via barata; **não** é a recomendada.)

## Indução contínua e anti-circularidade

- Indução contínua (aberto/fechado/não-vazio em `I`): **VÁLIDA**. Nenhum objeto do passo 4 depende de
  dados posteriores a `s`; a causalidade temporal é real.
- Anti-circularidade do fator 2: **CONFIRMADA**. `M₂ = C_f C_4 / c` é construído de `C_f` (propriedade
  de `A` e das features), `C_4` (funcional de `R^hyb`) e `c`; o fator 2 da hipótese de indução nunca
  entra em `M₂`. A melhoria `2M₂ → 3/2 M₂` é incondicional.

## O4

Suficiente como **observação empírica sobre `β₀(instância)`**. A cadeia de `β₀` acima contém `1/a₀` e
`1/R(t_exit)` — bate com a estrutura observada (o erro rastreia `1/R(t_exit)`, `t=2.56`; `log₂m` não,
`t=0.71`). A última frase de (O4) ("sem (O1) não podemos certificar que β₀ é polinomial em m") fica
**falsa** depois do reparo 5 e tem de ser corrigida.

## O3

Continua **aberto** (herdado do paper). Nomear: é o suspeito do viés de inclinação `a = 1.154`
medido na regressão do parecer-14 — ali L4(ii) foi descartado por sinal simétrico (36/66,
mediana +0.042), sobrando o `o(1)` não quantificado da quina de entrada (L3′).

## Os 9 reparos

### Bloqueantes

1. **Item (iv) do Teorema**: constante corrigida para `2M₂` **e** estimativa PSD refeita em
   `[t_rel, H]` (pós-release o sistema volta afim com Gram PSD — repetição do passo 3 do L2′),
   escrita por extenso, para que o `sup` em `[0,H]` seja legítimo.
2. **(O2) reescrito como ESCOPO, não dívida**: (H4′) define a janela; o teorema vale em qualquer
   janela maximal de padrão vivo constante contendo `t_hit` e `t_exit`, verificável a posteriori.
   Incluir o argumento de que a relaxação reintroduz Grönwall com expoente
   `Σ ‖ΔG‖‖θ‖/v_k` (até `mn` chaveamentos ⟹ exponencial em `m` no pior caso).
3. **Tirar `β ≳ m` de dentro do `(1+o(1))` da eq. (GS)** — explicitar como condição de `β₀`.

### Fortes

4. **Rebaixar (O5) para FECHADO** (auditado + verificado numericamente), com três reparos de
   escrituração: (a) termo `H·C_5√m·C_res·e^{−βδ_B}` aditivo na eq. (prehitA);
   (b) ressalva de monotonicidade de `z⁰` (ou `+C·t·e^{−βδ_z}` longe do hit);
   (c) `‖G_BA‖ ≲ X²_max‖a‖₂/n ≲ √m` na remark de uniformidade.
5. **Exibir a cadeia de `β₀` como tabela** (as 5 condições acima). Conclusão: `β₀ ≲ O(m)` dado o
   forcing (`a₀`, `R(t_exit)`, `T_ep` fixos). Corrigir a última frase de (O4) e de (U6).
6. **(O3)**: acrescentar que é o suspeito nomeado do viés de inclinação `a = 1.154` medido
   (parecer-14: L4(ii) descartado por sinal simétrico 36/66).

### Cosméticos

7. **Passo 1 do L2′**: corrigir a frase — no híbrido a contribuição de `A` é **exatamente 0** em todos
   os dados; o `O(e^{−βδ₁})` pertence ao `β`-flow.
8. **Passo 6**: contabilizar o erro de área `O(C_R ε_β²) = O(log²β/β²)` do congelamento do forcing na
   janela terminal (inofensivo, `o(β⁻¹)`, mas presente).
9. **Nova subseção na parte numérica**: verificação direta do passo 4 do L2′ — `ρ_A(t_hit)·β` satura
   (~0.53), `ρ_B(t_hit)·β²` satura, `β²·v·ρ_B ≈ 0.198 ± 0.05` invariante em `m`
   (12 instâncias, RK4, `β ∈ {1e3..8e3}`); scripts `parecer15_o5.py` / `parecer15_o5b.py`.

## Estado após os reparos 1–6

Teorema com **hipóteses fortes declaradas** (não prova condicional): uniformidade até `O(m)` em `β₀`
e `O(√m)` nas constantes explícitas; a lei de primeira ordem `−1/(c R(t_exit))` permanece
estruturalmente `m`-livre. Único item genuinamente aberto: (O3), herdado.
