# etapa-4a.3 — CONFIRMATÓRIO: estatística da ressurreição-por-mecanismo em MLP (T=0, pressão)

**STATUS: RASCUNHO DE DESENHO — para 9º parecer. NÃO congelado, NÃO rodar.**
Herdeiro do piloto-pressão (nota 06-notas/2026-07-25, commit b48b527): mecanismo
observado (2 ressurreições dinâmicas softplus, trio ordenado 2/2, relu 0/25 em
16 seeds), gargalo = taxa de eventos (~0.1–0.25/run) e censura (retorno
~1.4–1.5e5 vs H=2e5).

Etiqueta: [ESPECULAÇÃO⟶EVIDÊNCIA]: a existência tem evidência exploratória; o
confirmatório busca N e CI. Regime DETERMINÍSTICO (T=0) — o estocástico é
explicitamente fora de escopo (paper próprio).

## 1. Perguntas pré-registradas

- **C1 (EXISTÊNCIA COM TAXA — primário).** Contagem de ressurreições DINÂMICAS
  (morte com t_morte>0 e ground-truth zmax<0 em todos os dados; volta pelo
  ground-truth) no braço softplus-β10, agregada nas S seeds. Estatística: taxa
  Poisson com CI exato; pareado com relu (mesma seed/init/dados) cujo zero é
  ESTRUTURAL (grad≡0 — controle de instrumento, não MATA: decidível a priori,
  lição do P-C). **MATA de C1:** com esperança de eventos ≥10 (dimensionada do
  piloto), observar 0–2 eventos ⟹ taxa incompatível com o piloto ⟹ o
  mecanismo não é reprodutível na escala desenhada — negativo registrado.
- **C2 (LEI DE ÁREA — o coração).** Para CADA ressurreição dinâmica: o trio
  (t_morte, t_área, t_res) da régua v2 (T=0 ⟹ sem chatter; v3 redundante).
  Estatística: fração de trios ORDENADOS (t_morte < t_área ≤ t_res) com CI
  binomial exato. **MATA:** fração ordenada com CI-inferior < 0.5 ⟹ o
  zero-de-área NÃO governa a volta em rede real (mata a elevação da lei).
  Piloto: 2/2.
- **C3 (GAP DE SUBIDA — estimação).** gap = t_res − t_área vs profundidade
  z_park e canal σ(β·z_park): regressão exploratória do tempo de subida
  (correção a β finito). Sem MATA; alimenta o paper.
- **C4 (CANAL-β — estimação).** Taxas por β ∈ {3, 10, 300}: β=3 ≈ suavidade
  GELU (ponte com prática); predição qualitativa: taxa(β3) ≥ taxa(β10) ≫
  taxa(β300) — reportar com CI, piso de contagem para qualquer comparação.
- **VALIDADE (obrigatória):** QUALQUER ressurreição no braço relu ⟹ bug ⟹
  experimento inválido (é teorema: grad≡0). Verificação dupla: ground-truth
  independente de régua + inspeção do episódio.

## 2. Desenho (números a congelar em preditor_4a3 pós-parecer)

Sistema do piloto (inalterado — já mostrou o fenômeno): n=16 dados,
y=sin(2πx)+0.3sin(5πx), init padrão, η=1e-3, T=0, full-batch, CPU pinada,
determinismo. **H=5e5** (3.3× o retorno típico ~1.5e5 ⟹ descensura mortes até
~3.5e5). Braços: softplus-β3, softplus-β10, softplus-β300, relu.
m ∈ {4, 6}. Seeds virgens 1900+ — S dimensionado pelo piloto: taxa de morte
dinâmica ~0.3/run (m4+m6 combinados) e P(volta | morte, H=5e5) estimada ≥0.7
⟹ ~0.2 eventos/run/braço ⟹ **S=60 seeds** dá E[eventos β10] ≈ 60×2×0.2 = 24
(≥10 com folga 2×; recalcular no preditor com os números exatos do piloto).
Total: 4 braços × 2 m × 60 seeds = 480 runs × ~4.7 min ≈ 38 core-h ⟹
**~9.5 h no Pool 4 nice-19** (noturno). Régua: v2 + ground-truth (T=0);
episódio dinâmico = t_morte>0; flags v_k herdados; filtro-limpo NÃO se aplica
(m=4–6, poucos corpos — reportar switches por episódio como descritivo).

## 3. Análise

analise_4a3.py testada em sintético (mundos: taxa-zero, taxa-piloto,
trio-desordenado) ANTES da run; números todos do config; Poisson/binomial
exatos (sem scipy); RNGs novos (não reusar 777/779/5xxx). Figuras: taxa por
braço com CI; trios em linha do tempo; gap vs z_park.

## 4. Perguntas ao 9º parecer

(a) C1 com relu-pareada como controle e MATA por contagem — resta algo
decidível a priori? (b) o dimensionamento S=60 do piloto (n=5 mortes
dinâmicas!) é frágil — exigir piloto-de-taxa maior antes do freeze ou aceitar
o risco declarado (MATA de C1 cobre)? (c) C2 com fração-ordenada e piso 0.5 é
o corte certo (t_área=t_res empatados? censura de t_área?)? (d) braço β=3
entra sem calibração da régua a β=3 (limiar independe de β; canal muda) — ok?
(e) o que falta congelar.
