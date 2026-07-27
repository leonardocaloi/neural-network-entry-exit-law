# 18º parecer — `06-notas/t2-prova/teorema-degrau2-draft.tex`

**Data:** 2026-07-27
**Objeto:** draft do Teorema Degrau 2 (lei entrada–saída com readout treinável),
14 páginas na versão submetida ao parecer.
**Veredito: PROMOVÍVEL, com reparos.** Não há erro matemático que derrube o
teorema; o lema novo está correto e a numérica decide o item (a). Os reparos
abaixo são de enunciado, de contabilidade de constantes, de nomenclatura e de
honestidade estatística.

**Scripts do parecer** (neste diretório):
`parecer18_sym.py` (simbólico independente), `parecer18_data.py` (re-análise do
`degrau2_v0.jsonl`), `parecer18_rep.py` (réplica com sementes frescas),
`parecer18_m2.py` (piloto de Λ em m). Log: `parecer18.log`.

---

## O que foi auditado e passou

* **Lema novo (curvatura negativa bloco-diagonal).** Reverificado de forma
  independente em `parecer18_sym.py`: a estrutura bloco-diagonal de
  `S = ∇²U − JᵀJ/n` com blocos-flecha `[[0,0,c₁],[0,0,c₂],[c₁,c₂,0]]` é
  **estrutural**, confirmada simbolicamente em (m,n) = (3,3), (3,2), (2,4) —
  não é um acidente de m=n=2. Polinômio característico:
  `μ(c₁²+c₂²−μ²)`, logo espectro {0, ±‖c‖}. A prova do draft está correta e é
  completa.
* **Identidade `Δ̇ = −H̄Δ + G_A + G_S`.** Correta, e a observação da
  `rem:noswitch` (U de padrão congelado é polinômio global, o TVM não tem
  problema de padrão no segmento) é o argumento certo — é mais limpo que o
  caso (O2) e o draft acerta ao dizer isso.
* **Redução a L1″–L6″.** A alegação de que os passos 2a/2b do L2′ sobrevivem
  intactos foi conferida contra o Apêndice B do paper-v4 e procede: são
  afirmações sobre a saída de A e sobre o defeito softplus, não sobre a classe
  de B.
* **Item (a) e o teste discriminante.** O desenho é correto, a condição de
  falsificação foi fixada antes, e o resultado é decisivo. Réplica com
  sementes frescas e disjuntas (`parecer18_rep.py`, 16/16 aceitas):
  μ/naive 1.950 e 0.4995; μ/corrigido 0.975 e 0.999; pareado 15/16. Confirma.

---

## BLOQUEANTES

### B1 — a potência de 1/|v₀| no item (b) do Teorema é 2, não 1

O enunciado escreve `M₂ ↦ e^{Λ_ep T_ep} M₂/|v₀(t_hit)|`, uma potência. Mas o
próprio draft define `M₂″ = e^{Λ_ep T_ep} C_f C₄ /(c|v₀|)` e, em (U2), diz que
**C₄ já carrega 1/|v₀|** no seu primeiro termo. Logo a dependência no guarda é
`φ_v^{-2}`, exatamente como (U4) escreve. O item (b) e (U4) estão em
contradição literal. Corrigir o enunciado (não (U4), que está certo).

### B2 — (H6′) deve ser REDUZIDA ao readout do morto

(H6′) exige `min_{0≤k≤m} min_t |v_k| ≥ φ_v`. **Nenhum passo da prova consome
cota inferior nos v_k vivos.** Auditoria:

* Λ = max_k‖c_k‖ com `c_k = (1/n)Σ r_i 1_{k,i}(x_i,1)` **não menciona v_k**; a
  cota usada, Λ ≤ X_max·C_res, é livre de readouts. A identidade
  Λ = max_k‖∇_{(w_k,b_k)}U‖/|v_k| é uma *leitura*, não um passo.
* C_f, ‖G_BA‖, C₅ são cotas **superiores** em normas de readout — cota superior
  não precisa de piso.
* v, v′, C₄ e o coeficiente envolvem v₀ e **só** v₀.

Portanto: reduzir a hipótese a `|v₀(t)| ≥ φ_v` e v₀ sem troca de sinal. Um
readout vivo pode anular-se ou trocar de sinal sem quebrar nada — o bloco vivo
apenas perde uma direção da própria parte Gauss–Newton, que é um termo PSD que
a prova só descarta.

**Registrar o corolário:** com isso a discussão de balancedness de (P3) colapsa
a **uma** unidade. A condição suficiente vira `B₀(0) = v₀(0)² − ‖(w₀,b₀)(0)‖² > 0`,
sobre um único trio, e o reescalamento balanceado aplicado só à unidade 0 **não
toca nenhuma margem viva** — a competição com (H4′) que inviabilizava a rota
some. É uma rota tratável (não checada), e isso deve ser dito.

### B3 — renomear o `L_B` do draft/números/código

O draft mede `L_B := max(0, −λ_min(∇²U))`. No paper-v4, `L_B := λ_max(G_B)`.
São objetos diferentes, e **a distinção é o conteúdo inteiro da `rem:O6`**:
usar o mesmo símbolo para os dois destrói o argumento na leitura. Renomear para
`Λ_meas` no draft, na seção numérica e nas chaves do jsonl/código (manter
`L_B` como alias depreciado para não quebrar artefatos já gravados).

---

## FORTES

* **F4 — `err_naive` é piso aritmético, não erro empírico.** Se a lei corrigida
  vale, então `|μ−naive|/|naive| ≈ |1 − 1/v₀|` = 100% (v₀=0,5) e 50% (v₀=2,0),
  identicamente. É a distância a que *nós* colocamos as duas hipóteses, não a
  qualidade do ajuste da naive. Reportá-la como "a naive erra 51%" é inflar um
  parâmetro de desenho em resultado. As estatísticas informativas são a
  **contagem pareada** e as **razões μ/preditor**. Declarar isso na tabela.
* **F5 — razão mediana pooled errada.** `median(err_corr/err_naive)` sobre
  N=48 é **0,0460**, não 0,043 (esse é o subgrupo v₀=2,0). Corrigir.
  (Teste de sinal unilateral: p = 1,7·10⁻¹³.)
* **F6 — `|v₀(t_hit)−v₀(0)| ≤ 3,6·10⁻⁶` NÃO é teste de (eq:v0const).**
  (eq:v0const) fala de `[t_hit, t_rel]`, com a unidade **morta**. A medida é
  sobre `[0, t_hit]`, com a unidade **viva**. São fatos distintos, e o
  observado é um fato empírico separado (v₀ mal se move desde a init nesta
  família, porque z_A(0) é pequeno). Ele é *útil* — é o que faz o desenho
  discriminante funcionar, pois permite fixar v₀(t_hit) fixando v₀(0) — mas
  enunciá-lo como confirmação de (eq:v0const) é erro de atribuição.
* **F7 — declarar a normalização do "relative 3.0e-15".** O relativo pontual
  `max(|Λ_meas−Λ|/Λ)` é **1,59·10⁻¹⁵**; o 3,0·10⁻¹⁵ é `max|·|/min Λ`, outra
  normalização (cota superior da primeira). Reportar as duas, rotuladas.

---

## COSMÉTICOS

* **C1** — o fator pré-hit e o de episódio **compõem** (o pré-hit multiplica o
  dado inicial *dentro* do colchete que o de episódio escala). A tabela da
  `rem:honest` sugere três fatores independentes; não são.
* **C2** — escrever (eq:Brate) com `ρ(t_hit)` explícito como dado inicial em
  `[t_hit, s]`, em vez de embutido como "grupo pré-hit".
* **C3** — a liberdade em m é **da cota** `Λ ≤ X_max C_res`. O Λ *realizado* é
  empírico. Piloto (`parecer18_m2.py`): Λ mediana **5,03 / 5,70 / 5,18** em
  m = 4/8/16, contra λ_max(G_B) = **10,2 / 18,7 / 37,7**. N = 4/4/1 — indicativo,
  citar como tal.
* **C4** — Λ_ep deve ser definido sobre **o mesmo intervalo `[t_hit, s]` do
  bootstrap**, não sobre `[t_hit, t_rel]` fixo; a indução do L4″ fecha em s.
* **C5** — a seleção é condicional ao tratamento (8/30 rejeições em v₀=0,5
  contra 4/30 em v₀=2,0). Declarar em §5.2, e notar que a comparação pareada
  intra-instância é imune.
* **C6** — v₀ = 0 é singularidade **genuína** do coeficiente (a unidade some
  funcionalmente; não há release a prever) — degradação graciosa da mesma
  espécie do (O1). v_k vivo cruzando zero é **inócuo**. Dizer os dois.

---

## ABERTOS a acrescentar

* **P7 — varredura de Λ em m.** N ≈ 20 aceitas por nível, m ∈ {4,8,16,32},
  mesma família e dt fino, reportando mediana e IQR de Λ e de λ_max(G_B) por
  nível, com a predição pré-registrada "o primeiro é plano, o segundo linear".
  m=32 exige Hessiano 96×96 e `HESS_EVERY` mais grosso.
* **P8 — medir o quociente de Rayleigh `q(t) = −ΔᵀH̄Δ/‖Δ‖²`.** É o que o passo 3
  realmente paga (`exp∫q`), e `q ≤ Λ_ep` com igualdade só se Δ se alinha ao
  autovetor minimizante. **Separa as duas explicações vivas:** se q ≈ Λ, o
  expoente está certo e a estimativa é apertada; se q ≪ Λ, o Grönwall é
  simplesmente frouxo e Λ é artefato da pior direção. Δ e H̄ já são calculados
  em `degrau2_readout.py`; é barato.
  *Mecanismo afiado da saturação, para a `rem:sat`:* Weyl fecha com igualdade
  exatamente quando o autovetor minimizante de S é também vetor nulo da parte
  Gauss–Newton. Esse autovetor é **explícito**: `(e_c − e₃)/√2` dentro do bloco
  ligante `k* = argmax_k‖c_k‖`, zero nos demais. Como `rank(JᵀJ/n) ≤ n < 3m`
  (aqui 4 < 12), o núcleo GN tem dimensão ≥ 3m−n = 8; a afirmação é que *esse*
  vetor cai nele — o que não é automático do déficit de posto.
* **P9 — seleção condicional ao tratamento** (= C5).

---

## Imprecisão adicional a corrigir

`rem:honest` e (U7) dizem que as cláusulas (i)–(iii) "carregam só
`e^{Λ T_ep}`". Isso é verdade das **constantes exibidas nos enunciados**, mas
**β₀ é um número só** — o máximo sobre as cinco linhas da Tab. 5 do paper-v4 —
e uma dessas linhas é alimentada pela cláusula (ivB). Logo **β₀ herda o maior
fator (`e^{X_max C_res H}`) para todas as cláusulas**, inclusive (i)–(iii). Os
enunciados são limpos em episódio; o limiar acima do qual valem, não. Separar
as duas coisas em vez de arredondar a favor.
