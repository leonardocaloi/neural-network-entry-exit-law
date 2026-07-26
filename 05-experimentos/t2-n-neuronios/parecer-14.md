# Parecer 14 — Degrau 1 (teorema de forcing "m vivos + 1 morto")
2026-07-26. Parecerista adversarial (Opus). Objeto: 06-notas/2026-07-26-degrau1-teorema-forcing.md + verificação numérica em 05-experimentos/t2-n-neuronios/. Scripts do parecer: parecer14.py, beta.py (este diretório).

**Veredito: (b) PROMOVÍVEL, com o enunciado CORRIGIDO — e a correção o torna mais forte, não mais fraco. §4 da nota contém afirmação factualmente falsa retratada antes da promoção.**

## Reparos à auditoria (concordo em 5/6)
- **Reparo A**: o e^{L_B·H} de L2 é artefato de cota. G_B é Gram, logo PSD: d/dt ½‖Δ‖² = −ΔᵀG_BΔ + Δ·G_A ≤ ‖Δ‖·‖G_A‖ ⟹ ‖Δ(t)‖ ≤ ∫‖G_A‖ ds, sem fator exponencial. M₂ = C_f·C_4/c. Dependência em m só via C_f ~ √m, polinomial.
- **Reparo B**: com m vivas há m·n discrepâncias sigmoide-vs-indicadora nas próprias vivas, cada O(e^{−βδ_B}) — exige (H4′) REFORÇADA: margens vivas uniformemente ≥ δ_B > 0 (não apenas "não chaveia").
- **Reparo C**: ciclo lógico L1→L6→L5→L1 no paper (continuidade implícita); prova escrita exige indução contínua explícita (conjunto aberto-fechado-não-vazio). C_3′ envolve sup_i|res_i| ⟹ (H5′): |res_i| ≤ C_res no tubo.

## Numérica do parecer
- Réplica exata por clonagem (readout a_k/√p, params (w,b)/√p): 23/23 instâncias com erro BIT-IDÊNTICO em m=4/8/16 — contribuição de m per se exatamente zero.
- Regressão conjunta n=133 (duas famílias, λ_max 2.0→37.7 ⟹ e^{L_B·H} varia 28 ordens): só 1/R_exit significativo (t=+7.44); log₂m t=−0.09; log λ_max t=−0.47.
- Varredura β nas 6 piores m=8: 20.45%→0.35%, 20.20%→4.93%, 17.30%→0.20%, 13.52%→4.23%, 12.62%→0.30%, 11.88%→0.50% (β=1e4→3e4). Cauda é o(1/β) genuíno; β₀ ~ 1/R_exit, não m.
- Família readout+1 (a do paper): NÃO colapsa — peneira 179/900 (19.9%), erros mediana 1.16%/4.52%/5.35% (m=4/8/16), λ_max 10.2/18.7/37.7. H×4: motivos de rejeição numericamente idênticos (352/207/162). Retratação do §4 da nota exigida; causa provável: divergência docstring×código no sorteio de z_A(0).
- (H2′) a posteriori: aceitável, estrutura padrão de teorema de EDO com hipótese sobre a solução do problema-limite; condições abertas, estáveis. PORÉM o filtro numérico (troca única de sinal de R) é mais forte que (H2′) — declarar.
- Viés a=1.154 sem explicação: L4(ii) descartado (sinal simétrico 36/66, mediana +0.042); candidato: o(1/β) de L3.

## Lista exata do que a prova escrita precisa cobrir
1. Hipóteses corrigidas: (H0′) R Lipschitz + ‖Ṙ‖∞ ≤ C_R; (H1′), (H2′) como estão; (H3′) Lipschitz local no tubo; (H4′) reforçada — margens das vivas ≥ δ_B > 0; (H5′) |res_i| ≤ C_res no tubo.
2. L2 refeito com a estrutura PSD: ‖Δ‖ ≤ ∫‖G_A‖, sem e^{L_B·H}. Constante M₂ = C_f·C_4/c. Onde o campo de B não for gradiente de convexa (Degrau 2), aí sim e^{L_B·T_ep} — com T_ep, não H.
3. Contabilizar o termo m·n·O(e^{−βδ_B}) em L2 passo 2 e propagá-lo por L5.
4. Quebrar o ciclo L1/L5/L6 com enunciado explícito de continuação (aberto, fechado e não-vazio em [t_hit,H]).
5. Declarar a uniformidade que sobrevive: coeficiente de 1ª ordem −1/(cR(t_exit)) estruturalmente independente do subsistema vivo; M₂ e β₀ dependem de m só via C_f (≲ √m) e via 1/R(t_exit), T_ep, a₀ — não exponencialmente. Sem restrição "m·H pequeno".
6. Retratar §4 "Achado colateral" e corrigir o docstring de sorteia (z₀ ∈ [10⁻⁴,10⁻¹·³]).
7. Registrar que a numérica testa um filtro mais estrito que (H2′) (troca de sinal única de R).
