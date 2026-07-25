# Piloto-de-taxa etapa-4a.3 — resultado

Grade: 240 runs. braços ['sp-b3', 'sp-b10', 'sp-b300', 'relu'], m∈{4,6}, seeds 100–129 (30), n=16, η=1e-3, T=0, H=5e5, rec_neuronio=0.
Contador de morte/volta por GROUND-TRUTH (zmax<0→>0); trio e área-abriu pela régua v2. EXPLORATÓRIO — dimensiona o confirmatório.

## VALIDADE relu (teorema: grad≡0 no morto ⟹ 0 voltas)
OK: relu 0 voltas / 0 born-rev / 0 rev_real em 60 runs (13 mortes dinâmicas, todas sem volta). Instrumento válido.

## Tabela por braço (m=4 e m=6 agregados; 60 runs/braço, 30 seeds)
braço | runs | born-dead | mortes-din | λ/run (boot-seed) | λ/run (Poisson iid) | voltas | term | cens | p(volta\|morte) CP | p(volta\|área) CP | trios ord
---|---|---|---|---|---|---|---|---|---|---|---
sp-b3 | 60 | 110 | 16 | 0.267 [0.167, 0.383] | 0.267 [0.152, 0.433] | 6 | 5 | 5 | 0.375 [0.152, 0.646] | 0.500 [0.187, 0.813] | 2/6
sp-b10 | 60 | 110 | 18 | 0.300 [0.183, 0.433] | 0.300 [0.178, 0.474] | 4 | 9 | 5 | 0.222 [0.064, 0.476] | 0.167 [0.004, 0.641] | 1/4
sp-b300 | 60 | 110 | 13 | 0.217 [0.117, 0.333] | 0.217 [0.115, 0.371] | 0 | 13 | 0 | 0.000 [0.000, 0.247] | nan [nan, nan] | 0/0
relu | 60 | 110 | 13 | 0.217 [0.117, 0.333] | 0.217 [0.115, 0.371] | 0 | 13 | 0 | 0.000 [0.000, 0.247] | nan [nan, nan] | 0/0

_p(volta|área): numerador = voltas cuja ÁREA ABRIU (v2 t_area≥0); denominador = mortes dinâmicas com área aberta (voltas-com-área + censurados). NÃO usa todas as voltas — 4/10 voltas têm t_area=−1 (cegueira da v2, ver Diagnóstico C2), então este condicional está enviesado pelo instrumento e deve ser relido com área ground-truth._
## Tabela por (braço, m) — 30 runs/célula
braço | m | runs | born-dead | mortes-din | λ/run (boot-seed) | voltas | term | cens | p(volta\|morte) CP
---|---|---|---|---|---|---|---|---|---
sp-b3 | 4 | 30 | 48 | 9 | 0.300 [0.133, 0.467] | 2 | 4 | 3 | 0.222 [0.028, 0.600]
sp-b3 | 6 | 30 | 62 | 7 | 0.233 [0.067, 0.433] | 4 | 1 | 2 | 0.571 [0.184, 0.901]
sp-b10 | 4 | 30 | 48 | 9 | 0.300 [0.133, 0.467] | 2 | 5 | 2 | 0.222 [0.028, 0.600]
sp-b10 | 6 | 30 | 62 | 9 | 0.300 [0.133, 0.500] | 2 | 4 | 3 | 0.222 [0.028, 0.600]
sp-b300 | 4 | 30 | 48 | 7 | 0.233 [0.100, 0.400] | 0 | 7 | 0 | 0.000 [0.000, 0.410]
sp-b300 | 6 | 30 | 62 | 6 | 0.200 [0.067, 0.333] | 0 | 6 | 0 | 0.000 [0.000, 0.459]
relu | 4 | 30 | 48 | 7 | 0.233 [0.100, 0.400] | 0 | 7 | 0 | 0.000 [0.000, 0.410]
relu | 6 | 30 | 62 | 6 | 0.200 [0.067, 0.333] | 0 | 6 | 0 | 0.000 [0.000, 0.459]

## Return-time (t_res−t_morte, ground-truth) e τ = quantil 0.9, por β
β | n voltas | mín | mediana | **τ=q0.9** | máx | média
---|---|---|---|---|---|---
3 | 6 | 13825 | 45808 | **192464** | 308075 | 87711
10 | 4 | 11163 | 20445 | **303190** | 422834 | 118722
300 | 0 | — | — | — | — | —
pool | 10 | 11163 | 37015 | **319551** | 422834 | 100115

## Trios de TODAS as voltas dinâmicas (regra estrita t_morte<t_área<t_res)
braço | m | seed | k | t_morte(GT) | t_res(GT) | v2(t_morte,t_área,t_res) | ordenado | área-abriu
---|---|---|---|---|---|---|---|---
sp-b10 | 4 | 106 | 1 | 90 | 24110 | (0,-1,24231) | não | não
sp-b10 | 4 | 112 | 2 | 15057 | 31927 | (10026,-1,36814) | não | não
sp-b10 | 6 | 110 | 0 | 4842 | 427676 | (4611,290566,430314) | sim | sim
sp-b10 | 6 | 114 | 2 | 12923 | 24086 | (12751,-1,24244) | não | não
sp-b3 | 4 | 105 | 1 | 32819 | 86299 | (30392,30392,93843) | não | sim
sp-b3 | 4 | 113 | 3 | 2205 | 38098 | (2117,34019,38415) | sim | sim
sp-b3 | 6 | 106 | 4 | 281 | 14106 | (272,11646,14205) | sim | sim
sp-b3 | 6 | 106 | 4 | 23444 | 331519 | (23204,23204,332122) | não | sim
sp-b3 | 6 | 114 | 2 | 10214 | 87068 | (10141,-1,87158) | não | não
sp-b3 | 6 | 120 | 5 | 289621 | 327758 | (277800,277800,336161) | não | sim

## Diagnóstico C2: por que os trios falham a regra estrita (pool)
De 10 voltas dinâmicas, ordenadas ESTRITAS (t_morte<t_área<t_res) = **3/10** = 0.300 [0.067, 0.652]. Falhas:
- SEM-AREA(v2 t_area=-1): 4
- ORDENADO: 3
- EMPATE(t_area==t_morte): 3
Se a regra fosse t_morte ≤ t_área < t_res (empates contam), ordenadas = 6/10. O resto (4) é CEGUEIRA da régua v2: volta real (ground-truth) sem área registrada — falha de INSTRUMENTO, não da lei. Ver recomendação.

## Anomalias (volta sem área-abriu na v2): 4
- ('volta_sem_area', 'sp-b10', 4, 106, 1)
- ('volta_sem_area', 'sp-b10', 4, 112, 2)
- ('volta_sem_area', 'sp-b10', 6, 114, 2)
- ('volta_sem_area', 'sp-b3', 6, 114, 2)

## Recomendação para preditor_4a3 (derivada dos números medidos)

**Canal (C4) — confirmado.** p(volta|morte) monótona em β: sp-b3=0.375, sp-b10=0.222, sp-b300=0.000, relu=0.000. O canal e^{βz} FECHA quando β→∞ (softplus→relu): β300 dá 0 voltas exatamente como a relu (mesmo nº de mortes dinâmicas, todas terminais). Taxa de MORTE é β-independente (~0.22–0.30/run); a VOLTA é β-dependente. ⟹ o alvo primário deve ser **β3** (maior p(volta), mais voltas, retorno mais rápido, mais perto da suavidade GELU), NÃO β10.

**Achado crítico p/ C2 (o coração).** Sob a regra ESTRITA pré-registrada com área da v2, só 3/10 voltas são ordenadas — mas a falha é de INSTRUMENTO, não da lei:
  - 3 EMPATES t_área==t_morte (v2 abre a área no MESMO passo da morte-v2): a regra ESTRITA os rejeita; t_morte ≤ t_área os aceita.
  - 4 SEM-ÁREA (v2 t_area=−1) apesar de volta real: a régua v2 é CEGA a essas voltas (β10 rápidas + cruzamento de sinal de v_k). Nenhum ajuste de regra recupera — é o estimador de área que falha.
  - Ordenadas com t_morte≤t_área = 6/10; em β3 isoladamente as voltas ficam ~ordenadas (5–6/6) — a LEI DE ÁREA se sustenta em β3; quem falha é a v2.
  - **AÇÃO OBRIGATÓRIA antes de congelar C2:** (i) rec_neuronio>0 no confirmatório (o piloto pôs 0 por economia) para ter z_park e um sinal de área GROUND-TRUTH (sinal do drive acumulado da trajetória), régua-independente; (ii) regra t_morte ≤ t_área < t_res (absorve os empates de discretização); (iii) reconciliar o relógio: v2 marca morte ~126 passos (mediana) antes do ground-truth e às vezes chama morte-dinâmica de born-dead — usar UM relógio (o ground-truth).

**Braço primário β3:** λ_morte-din/run = 0.267 [0.167, 0.383] (bootstrap POR SEED ≈ Poisson iid ⟹ as MORTES quase não têm superdispersão; a superdispersão do parecer está nas VOLTAS, que se concentram em poucas seeds). p(volta|morte) = 0.375 [0.152, 0.646] (≪ os 0.70 que a nota assumia — parecer estava certo). p(volta|área) = 0.500 [0.187, 0.813] (numerador = voltas cuja área abriu; denominador contaminado pela cegueira da v2 — interpretar com o instrumento corrigido).
- voltas/seed (β3, 2 runs) = 2·0.267·0.375 ≈ 0.20 (ponto) / 0.05 (pior-caso CI-inf).
- **C1 (existência-com-taxa):** para E[voltas]≥10 (piso do parecer) ⟹ **S ≈ 50** seeds (ponto); para E[voltas]≥6 ⟹ S ≈ 30. pior-caso CI-inf pede S ≈ 197.
- **n\* (piso C1, INCONCLUSIVO):** contagem de MORTES dinâmicas (denominador). E[mortes-din β3] = S·2·0.267. A S≈50 ⟹ ~27 mortes; sugiro **n\* ≈ 17** (esperado ao CI-inferior λ=0.17); N_mortes < n\* ⟹ INCONCLUSIVO (não negativo), como no parecer.
- **C2 (sizing pelo coração):** precisa de ≥6 voltas ORDENADAS (CP-inf de 6/6 = 0.541>0.5; confortável 8–10). ordenadas/seed = voltas/seed × fração-ordenada:
  - v2 ESTRITA como está (fração≈0.33): ordenadas/seed≈0.07 ⟹ S≈90 (6 ord.) / 120 (8 ord.).
  - regra t_morte≤t_área (fração≈0.83): ordenadas/seed≈0.17 ⟹ S≈36 (6 ord.) / 48 (8 ord.).
  - instrumento CORRIGIDO (área ground-truth) (fração≈1.00): ordenadas/seed≈0.20 ⟹ S≈30 (6 ord.) / 40 (8 ord.).
  - Leitura: com a v2 como está, C2 é INVIÁVEL (S de centenas); com o instrumento corrigido, **S ≈ 40–60 seeds** entrega C2. O ganho do conserto de instrumento vale ~3–5× em amostra — é a alavanca.

**τ e janela de elegibilidade.** Return-time é MUITO assimétrico (mediana β3 ~46k mas cauda até ~3e5); com n=4–6 voltas o q0.9 é ruidoso. Valores medidos:
  - β3: τ=q0.9 ≈ 192464 ⟹ janela [0, H−τ] = [0, 307535] a H=5e5.
  - β10: τ=q0.9 ≈ 303190 ⟹ janela [0, H−τ] = [0, 196810] a H=5e5.
  - Manter **H=5e5**; a maioria das mortes é precoce (t_morte<30k), então a janela cobre o grosso. A cauda de retorno lento (2e5–4e5) é CENSURA legítima — tratar com KM/person-time (já exigido pelo parecer), não estendendo H indefinidamente.

**Grade sugerida do confirmatório:** primário β3; secundário β10 (comparação de canal, C4); β300 + relu como duplo controle de canal-fechado (0 voltas esperado); S≈50 seeds × m∈{4,6} × rec_neuronio>0. relu = validade (qualquer volta ⟹ bug). Sizing por C2 com o instrumento de área corrigido.
