# Piloto-de-taxa etapa-5 (transformer FFN) — RESULTADO

**Pergunta:** medir, com CI, as taxas de morte/ressurreição de unidades da FFN de um
transformer mínimo, para dimensionar o confirmatório e5 (S, n\*, janela, braços,
MATAs). Herda a disciplina do piloto-de-taxa 4a.3 (contador GROUND-TRUTH, CP/bootstrap,
τ=q0.9, exploratório-separado-de-confirmatório).

**Status epistêmico: [EVIDÊNCIA NUMÉRICA exploratória]** — 160 runs, seeds 100–119
(20/célula), SEM pré-registro. Braços {sp-β3, sp-β10, gelu, relu} × {attn-treina,
attn-congela} × m_ffn=4, GD puro T=0 determinístico, H=8e4. Injeção controlada de 2
unidades nascidas-mortas por run: **rasa** z_park=−0.3 (acima da barreira GELU
z\*=−0.7518) e **profunda** z_park=−1.0 (abaixo). Motor: `motor_e5b.py` (17 unit-tests
passam: 8 de regressão do motor_e5 + 9 do braço softplus/contador). Reprodução:
`piloto_taxa_e5.py --run --workers 2` → `results-taxa/*.json`; análise de-chatter:
`analise_taxa.py`.

---

## 0. Nota de instrumento — o CHATTER e a correção (ler antes da tabela)

O contador usou `z_dead=z_alive=0` (régua do 4a.3: zmax<0→>0). No MLP isso é limpo
(entrada fixa ⟹ morto permanece morto). **No transformer a entrada da FFN é
não-estacionária** (h=e+attn(e) se move): uma unidade empurrada para zmax≈0 **cruza o
limiar milhares de vezes** — cada micro-oscilação contada como morte+volta.
Prova (relu-train-s103, k=2): 1ª travessia GENUÍNA em t=66846 (upstream, own_move=0
exato) seguida de 159 oscilações em zmax=0.0000, z_park=−0.0007. Só **8/20** seeds de
relu-train explodem (até 4901 episódios/unidade); os demais braços têm ≤3 episódios
extra/célula. Sinal cru contaminado: relu-train aparecia com 5539 "voltas".

**Correção (uniforme a todos os braços): COLAPSO por unidade** — cada (célula,seed,
unidade) vira UM registro: revived = cruzou 0 alguma vez; t_res = MENOR t_res (1ª
travessia = return-time genuíno); z_park=min, classe-1ª = born/dyn do 1º episódio.
Assim cada unidade conta uma vez. **Sustentada** = a unidade terminou H no estado VIVO
(último episódio fechado por volta, sem reabrir); **hover** = revived mas reabriu
(chatter na barreira). `zmax_hi` NÃO serve de robustez (é limitado ao ponto de
travessia por construção — o episódio fecha na volta).

**Lição p/ o confirmatório:** usar **histerese** no instrumento (z_dead=−δ, z_alive=+δ,
δ≈0.05) para eliminar o chatter na origem — sem ela, qualquer taxa de volta de canal
não-estacionário fica inflada.

---

## 1. Tabela mestra (de-chatter, colapso por unidade; 20 seeds/célula, 40 unid. injetadas)

| célula | platô | λ_dyn/run [CI boot] | bornN/run | inj | voltas | sustent. | hover | cens | p(volta\|inj) [CP] | own_med(volta) |
|---|---|---|---|---|---|---|---|---|---|---|
| sp-b3-train  | 0.178 | 0.05 [0.00,0.15] | 0 | 40 | 11 | 10 | 1 | 29 | 0.28 [0.15,0.44] | 0.031 |
| sp-b3-frozen | 0.209 | 0.05 [0.00,0.15] | 0 | 40 |  5 |  5 | 0 | 35 | 0.12 [0.04,0.27] | 0.086 |
| sp-b10-train | 0.177 | 0.00 [0.00,0.00] | 0 | 40 |  9 |  8 | 1 | 31 | 0.23 [0.11,0.38] | 0.014 |
| sp-b10-frozen| 0.209 | 0.05 [0.00,0.15] | 0 | 40 |  3 |  3 | 0 | 37 | 0.07 [0.02,0.20] | 0.050 |
| gelu-train   | 0.170 | 0.05 [0.00,0.15] | 0 | 40 | 14 | 13 | 1 | 26 | 0.35 [0.21,0.52] | 0.047 |
| gelu-frozen  | 0.204 | 0.05 [0.00,0.15] | 0 | 40 |  8 |  6 | 2 | 32 | 0.20 [0.09,0.36] | 0.101 |
| relu-train   | 0.181 | 0.00 [0.00,0.00] | 0 | 40 |  8 |  4 | 4 | 32 | 0.20 [0.09,0.36] | **0.000** |
| relu-frozen  | 0.209 | 0.10 [0.00,0.25] | 0 | 40 |  0 |  0 | 0 | 40 | **0.00 [0.00,0.09]** | — |

**λ_morte-dinâmica (natural) ≈ 0 em TODOS os braços** (CI inclui 0; máx 0.10/run). Zero
born-dead natural (bornN=0). ⟹ **unidades de FFN não morrem sozinhas neste regime,
nem raso nem profundo** — confirma o scoping. Toda a estatística de volta vem da
injeção. Platô m4 com 2 mortas ≈ 0.17–0.21 (pressão real; controle de folga já no
scoping: m32→0.015).

_own_med(volta): mediana do deslocamento dos pesos próprios na volta. relu-train =
**0.000 exato** ⟹ voltas de relu são PURO montante (grad-próprio≡0). softplus/gelu > 0
⟹ own+montante._

---

## 2. p(volta\|injeção) e return-time por PROFUNDIDADE (só sustentadas p/ τ)

| célula | rasa (z=−0.3) volta/n [CP] | prof (z=−1.0) volta/n [CP] |
|---|---|---|
| sp-b3-train  | 10/20 [0.27,0.73] | 1/20 [0.00,0.25] |
| sp-b3-frozen |  2/20 [0.01,0.32] | 3/20 [0.03,0.38] |
| sp-b10-train |  9/20 [0.23,0.68] | 0/20 [0.00,0.17] |
| sp-b10-frozen|  3/20 [0.03,0.38] | 0/20 [0.00,0.17] |
| gelu-train   | 13/20 [0.41,0.85] | 1/20 [0.00,0.25] |
| gelu-frozen  |  4/20 [0.06,0.44] | 4/20 [0.06,0.44] |
| relu-train   |  8/20 [0.19,0.64] | 0/20 [0.00,0.17] |
| relu-frozen  |  0/20 [0.00,0.17] | 0/20 [0.00,0.17] |

**Return-time t_res (1ª travessia GT, sustentadas, pool train+frozen):**

| braço | n | mín | mediana | **τ=q0.9** | máx |
|---|---|---|---|---|---|
| sp-b3  | 15 | 4  | 14482 | **66072** | 79663 |
| sp-b10 | 11 | 5  | 43003 | **64193** | 74849 |
| gelu   | 19 | 9  | 21900 | **60132** | 71313 |
| relu   |  4 | 5  |   345 | **24784** | 35122 |

τ ≈ 60–66k para softplus/gelu ⟹ **com H=8e4 a janela [0,H−τ] fica ~[0,14–20k]:
apertadíssima.** O confirmatório precisa de H maior (§7).

---

## 3. Atribuição de canal: drive-próprio (frozen) vs montante (train) — P-iii/P-iv

| braço | frozen voltas (próprio) | train voltas | **Δ=montante** | own_med(train) |
|---|---|---|---|---|
| sp-b3  | 5 | 11 | **+6** | 0.031 |
| sp-b10 | 3 |  9 | **+6** | 0.014 |
| gelu   | 8 | 14 | **+6** | 0.047 |
| relu   | 0 |  8 | **+8** | **0.000** |

**Δ>0 em TODOS os braços** ⟹ attn-treina ADICIONA voltas (canal de montante).
relu isola-o na pureza: 0 (frozen) → 8 (train) com own_move=0.000 exato ⟹ volta 100%
de montante. Este é o achado **mais limpo e robusto** do piloto.

---

## 4. GT-ÁREA (A_drive) e S1 (sign-flip GELU)

- **Trios das voltas:** 58 voltas; A_drive aberta em/antes da travessia
  (t_area_drive ≤ t_res) em **50/58**. Mas **A_drive NÃO prediz a volta com confiança**:
  várias voltas têm A_drive<0 (ex.: sp-b3-train s105 −0.37→volta; gelu-train s105
  −0.069→volta). O drive-próprio explica bem os casos frozen-próprios; falha quando o
  montante domina (A_drive só mede o self-canal). Para relu, A_drive≈±1e-4 (g'=0) e a
  volta é montante — coerente (área-própria lê ~0, corretamente).
- **S1 confirmado (não-colapsado):** sinal(A_drive)≠sinal(A_mlp) em **19/20 (95%)** dos
  episódios GELU **profundos** (train) e **16/20 (80%)** (frozen); só **9%** nos rasos.
  ⟹ o porte cego da área do MLP (sem g') erra o sinal na região profunda — a
  área-com-inclinação A_drive é obrigatória, como o scoping antecipou.

---

## 5. Vereditos das predições da casa — SEM AMACIAR

**P-i (sp-β3/β10 attn-frozen reproduz a física do MLP: mortes/voltas por área, escada
de β) — PARCIALMENTE SUSTENTADA, fraca.** O canal próprio da classe-do-teorema REVIVE
sob attn-frozen (sp-b3 5/40, sp-b10 3/40 voltas, todas sustentadas) e a escada aparece
na direção certa (p: β3=0.12 > β10=0.07 frozen; 0.28>0.23 train), mas com **n pequeno e
CIs sobrepostos**. A área-própria A_drive orienta os casos frozen. PORÉM p é BAIXO
(0.07–0.12) em H=8e4 ⟹ o canal próprio é LENTO (τ≈64–66k); a "física do MLP" transfere,
mas anêmica neste horizonte.

**P-ii (GELU attn-frozen: morte-profunda TERMINAL por barreira, rasa revivível) —
NÃO SUSTENTADA no depth testado.** gelu-frozen **profunda 4/20 = rasa 4/20** (empatadas;
sustentadas: profunda 4 vs rasa 2). Injeção a z=−1.0 (só um pouco abaixo de z\*=−0.752)
**revive tanto quanto a rasa** — a FFN+cabeça co-adaptam e levam a unidade profunda a 0
apesar de g'<0. A assinatura da barreira existe (S1: 80–95% de sinal-oposto nos
profundos) mas **não é terminalidade** a −1.0. Decidibilidade: para testar a barreira
como MATA é preciso injetar MAIS FUNDO (z_park ≤ −2, longe de z\*); −1.0 é raso demais.

**P-iii (attn-treina ADICIONA voltas em todos os braços, inclusive relu) — SUSTENTADA
(forte).** Δ(train−frozen) = +6/+6/+6/+8. relu: 0→8 com own_move=0.000 ⟹ montante puro.
O canal de montante é real, mensurável e universal.

**P-iv (0 voltas relu attn-frozen; validade dupla grad≡0 E entrada fixa) — SUSTENTADA
(limpa).** relu-frozen 0/40, p=0.00 [0.00,0.09], own_move=0. Terminal por teorema.
Serve de MATA de validade do instrumento (qualquer volta aqui ⟹ bug).

**Resumo:** transferem-se (P-iii forte, P-iv limpa) o **canal de montante** e a
**terminalidade do relu-congelado**; transfere fraco (P-i) o canal próprio da
classe-do-teorema; **não** se confirma (P-ii) a terminalidade-por-barreira do GELU no
depth −1.0. Natural death ≈ 0: o confirmatório é obrigatoriamente injeção-condicional.

---

## 6. p(volta\|morte): volta / terminal / censurado + decidibilidade

- **Terminal DECIDÍVEL:** só relu-frozen (grad-próprio≡0 + entrada fixa ⟹ terminal por
  teorema; 40/40). Nos demais, "não-voltou-em-H" é **censura**, não terminalidade
  (o canal próprio/montante pode reviver com mais H) — exceto a barreira GELU, que
  seria terminal-decidível SÓ com injeção profunda (≤−2), não testada aqui.
- **Censurado (train, própria+montante):** softplus/gelu 26–31/40; relu-train 32/40.
- A cauda de return-time (τ≈66k, máx 80k≈H) ⟹ boa parte dos censurados são
  cauda-lenta legítima, não terminais. Tratar por KM/person-time no confirmatório.

---

## 7. Recomendação para o desenho confirmatório e5

O piloto **não valida** um confirmatório ingênuo em H=8e4: (i) o canal próprio
(o análogo-MLP, attn-frozen) é lento e raro (p≈0.07–0.12); (ii) a janela [0,H−τ] fica
apertada (τ≈66k vs H=8e4); (iii) a barreira GELU não é decidível a −1.0. Conserto:

**Instrumento.** Histerese z_dead=−0.05, z_alive=+0.05 (mata o chatter); relógio
ground-truth único; classificar volta em **sustentada** (viva no fim) vs hover.
Manter A_drive (com g' por ativação) como primário; A_mlp só como sonda do sign-flip.

**Horizonte e janela.** H ≥ **2e5** (≈3×τ) para dar tempo ao canal próprio e uma janela
real [0, H−τ] = [0, ~1.34e5]. Censura por KM (cauda de retorno lenta é legítima).

**Braços × fatores (pré-registrar o fator de canal):**
{sp-β3 (PRIMÁRIO, classe-do-teorema), sp-β10 (escada β), gelu (fora-de-classe: barreira
como estimando), relu (isola montante / validade)} × {attn-treina, attn-congela}.
**Injeção pré-registrada em DUAS profundidades:** rasa z_park=−0.3 (revivível) **e
profunda z_park≤−2** (para tornar a barreira-GELU DECIDÍVEL — a −1.0 não é).

**MATAs candidatos (com decidibilidade):**
- **MATA-A (validade):** qualquer volta sustentada em relu-frozen ⟹ bug. Decidível,
  limpa (0/40 aqui). Obrigatória.
- **MATA-B (montante real):** Δ(train−frozen) ≤ 0 no pool ⟹ canal de montante falso.
  Decidível já a S=20 (relu 0→8 decisivo). Confirmatório só aperta o CI.
- **MATA-C (transferência da classe-do-teorema):** sp-β3 attn-frozen raso = 0 voltas
  sustentadas a H≥2e5 ⟹ a lei de área NÃO transfere à FFN. Decidível com H/S adequados.
- **MATA-D (barreira-GELU / P-ii):** com injeção profunda ≤−2, p(volta) gelu-frozen
  profunda **não** < rasa ⟹ barreira não é terminal. **Só decidível com depth ≤−2**
  (o piloto a −1.0 NÃO decidiu: profunda≈rasa).
- **MATA-E (escada β):** p(volta) não-monótona em β ⟹ história do canal e^{βz} errada.
  Fraca (β3>β10 só na direção, n pequeno); precisa de S≥50 para virar MATA.

**Sizing (por estimando, 2 injetadas/seed = 1 rasa + 1 profunda):**
- Montante (MATA-B) — o mais forte: já separa a S=20 no relu; para separar train×frozen
  em softplus/gelu (CP sem sobreposição, 0.28 vs 0.12) ⟹ **S ≈ 40–50**.
- Canal próprio / lei-de-área (MATA-C, o coração-tipo-C2): p≈0.10 (frozen raso) ⟹ para
  E[voltas]≥6 (piso) a H=8e4 pediria S≈60–100; **subindo H→2e5** o p sobe e **S≈50** dá
  ~6–10 voltas próprias sustentadas. Recomendo **S=50** + H=2e5.
- Escada β (MATA-E): **S≈50** para monotonicidade com CI.
- Barreira (MATA-D): **S≈30–40** com injeção profunda ≤−2 basta (contraste esperado é
  grande se a barreira for terminal).

**n\* (piso de INCONCLUSIVO).** Como λ_natural≈0, n\* NÃO é sobre mortes naturais
(há zero) e sim sobre **voltas sustentadas por célula/estimando**: **n\* = 6** voltas
sustentadas ordenadas (t_área_drive ≤ t_res). Célula com < 6 ⟹ INCONCLUSIVO para aquele
estimando (não negativo) — igual à disciplina do 4a.3.

**Grade sugerida:** 4 braços × 2 fatores × **S=50** × injeção {−0.3, −2.0} × H=2e5,
histerese, rec_neurônio p/ z_park ground-truth, KM. Custo ≈ 400 runs × ~4.5 min (H=2e5)
/ 2 workers ≈ 15 h — **rodar em lotes por braço**, começando pelo PRIMÁRIO sp-β3 e pelo
par relu (MATA-A/B, os decisivos e baratos de interpretar).

---

## 8. Arquivos

- `motor_e5b.py` — motor estendido (softplus-β classe-do-teorema; z_dead=0 unificado;
  classificação deep/rasa na barreira z\*; injeção 2-profundidades; atribuição de canal;
  17 unit-tests: `python motor_e5b.py --testes`).
- `piloto_taxa_e5.py` — runner retomável (`--run --workers 2`) + análise crua (`--analise`).
- `analise_taxa.py` — análise DE-CHATTER (colapso por unidade) — a fonte desta nota.
- `results-taxa/*.json` — 160 runs; `ANALISE-dechatter.txt` — saída completa (trios).
