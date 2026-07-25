# etapa4-mlp-populacao (exp05 / etapa-4a) — estatística populacional de morte/ressurreição numa MLP real

**STATUS: CANDIDATO A PRÉ-REGISTRO — aguardando re-revisão dos blockers do 6º
parecer. NÃO RODAR runs confirmatórias antes do commit de pré-registro.**

Supersede `README-rascunho.md` (mantido como registro do desenho pré-parecer).
Parecer integral: `parecer-desenho-etapa4.md`. Respostas item-a-item:
`respostas-parecer.md`. Etiqueta global: **[ESPECULAÇÃO]** (hipótese
populacional própria); as leis do toy são **[TEOREMA-instância]** (draft-v2.tex).

**Escopo 4a (esta pré-registração): SÓ 1D-regressão.** 2D/MNIST viram etapa-4b
— o parecer mostrou que o filtro de estabilidade de líder só é dispensável em
1D (0 trocas medidas); antes de 2D ele precisa ser recalibrado.

---

## 1. Hipótese e competidores

**H [ESPECULAÇÃO]:** numa MLP rasa real (m neurônios, dados 1D, SGLD), a
morte/ressurreição de cada neurônio é governada pela lei de área por neurônio
do T2′ (entry–exit com memória), projetada via R̂_k = v_k·[(y−f)_{j*} +
v_k·act(z_{k,j*})]; e isso ergue estatísticas populacionais previsíveis apesar
do acoplamento de m corpos (H4 do teorema quebra por construção).

Competidores (cada um mata um pedaço): (1) complementaridade SEM memória —
ressuscita quando R̂>0 instantâneo; (2) escape térmico puro — só T manda, T=0
⟹ zero ressurreição; (3) predição-atrasada mecânica — a área "vence" só porque
dispara mais tarde (t_area ≥ t_sinal por construção; apontado no 6º parecer);
(4) lavagem por m corpos — a lei só vale em episódios isolados; (5) artefato de
régua — o instrumento acusaria lei até em dinâmica sem gradiente.

## 2. Instrumento (régua v2 — CALIBRADO, commit bd09a6b)

Classe `Regua` de `harness_mlp.py`: morte/ressurreição pelo limiar calibrado
`2√(nT)/‖(x_{j*},1)‖` na margem líder; R̂_k com a act da própria rede;
Â_k=Σ R̂_k·η por episódio; observação PRÉ-update a cada passo (Δsnapshot=1);
flags por episódio: `v_cruzou` (v_k trocou de sinal), `v_min_abs` (morte-por-
saída se < φ_v), `j_trocas` (líder), log global de switchings.

Calibração do instrumento EXATO (`calibracao_regua_v2.py`, toy-em-harness com
v treinável): (a) dinâmica converge em tempo contínuo (Δ3.4%); erro da
predição-área CONSISTENTE com o atraso O(1/β) do próprio teorema (4.54%→1.91%
quando β 300→1000, razão 0.42; piso residual ~0.8pp não explicado — rótulo do
7º parecer); (b) sob ruído T=1e-3: erro relativo mediano da área 0.534 vs 0.851
do sinal ("área vence sinal 34/34" é DESCRITIVO: t_sinal≤t_area≤t_res por
construção do instrumento, que para de medir na ressurreição — 7º parecer);
(c) unit-tests dos flags. V1 (`calibracao_regua.py`) fica como registro.

**Truncamento do instrumento (fato central para a análise):** a régua só
acumula Â_k enquanto morto ⟹ o atraso observado da área é censurado na duração
do episódio; episódio em que a área não disparou = censura à direita. TODA
inferência de P-A respeita isso (nulo calibrado abaixo).

**Episódio limpo** (`episodio_limpo()`, pré-registrado): ≤ q switchings de
outros neurônios no intervalo, sem v_cruzou, v_min_abs ≥ φ_v, ≤ q_j trocas de
líder. Censura: episódio aberto no horizonte NUNCA conta como permanente-por-
mecanismo; entra como censurado.

## 3. Predições congeladas (números em `config_etapa4.json` via `preditor.py`)

- **P-A (MATA — memória, o núcleo; reconstruído após o 7º parecer).** Episódios
  limpos-completos, braço softplus T=1e-3, por célula m. O 7º parecer provou:
  (i) "área vence sinal" é mecânico (t_sinal≤t_area≤t_res sempre) — vira
  report DESCRITIVO; (ii) o placebo de atraso-permutado cru viola o nível
  (FPR até 58× α) porque não herda o truncamento. **Teste congelado — nulo
  calibrado por bootstrap paramétrico:** estatística T = média entre seeds da
  fração-de-vitórias área-vs-placebo-permutado (dentro-da-seed, RNG 777);
  distribuição nula de T simulada sob H₀ = atraso ⊥ duração, atrasos iid de F̂
  estimada por **Kaplan-Meier** (não-disparo da área = censura à direita),
  COM o truncamento do instrumento reproduzido (B=1000, RNG 779);
  p=(1+#{T_b≥T_obs})/(B+1); passa se p<0.01. Nível e potência DEMONSTRADOS em
  `analise.py --sintetico`: FPR 0.000/0.000/0.010 para 8/40/200 episódios/seed;
  TPR 1.000 sob memória. A fração de limpos-completos SEM disparo da área é
  reportada por célula (evidência contra a lei se alta; 17% no piloto).
  MATA: p≥0.01 nas células válidas (agregação: demonstra com ≥2/3 incluindo
  m=16; MATA se falha em todas).
- **P-B (MATA — universalidade L5.2 DENTRO da classe).** Braço erf-smoothing
  (g′=Φ(βz), classe do teorema) vs softplus, mesmo β=300, pareado por seed.
  Estatísticas: fração-censurada, mediana da DURAÇÃO do episódio (t_res−t_morte;
  a validação sintética mostrou que t_res cru é dominado pelos instantes de
  morte), nº de episódios limpos. Teste de sinais bilateral entre seeds,
  Bonferroni α/3. MATA: |Δ| relativo > piso (5%) E significativo em ≥2 das 3
  células de m ⟹ L5.2 não ergue para rede real.
- **P-B′ (EXPLORATÓRIO, além do teorema).** GELU PADRÃO z·Φ(z) (derivada
  não-monótona — FORA da família; draft §reg). Report apenas; NENHUM "devem
  coincidir". É a ponte com a prática (LLMs), não teste do teorema.
- **P-C (ÂNCORA/ESTIMAÇÃO — rebaixado pelo 6º parecer).** relu-hard vs suaves:
  fração-censurada com CI por célula. O lado hard é quase-construção (gradiente
  exatamente nulo no morto-em-todos-os-dados); serve de âncora da não-comutação,
  não de MATA. Report inclui a separação permanente-por-mecanismo (hard, grad=0
  verificado) vs censurado (suave, horizonte).
- **P-D (MATA — back-reaction populacional).** Braço freeze-parcial (metade
  determinística dos nascidos-mortos congelada; RNG alinhado ⟹ pareamento
  exato com o controle da mesma seed). Para os nascidos-mortos NÃO congelados,
  Δt_res (freeze − controle) pareado POR NEURÔNIO; censura → diferença de
  fração-ressuscitada. MATA: CI 95% (bootstrap estratificado por seed) de
  mediana(Δt_res) contém 0 E |mediana| < piso (1% da duração mediana de
  episódio do controle) ⟹ sem back-reaction populacional detectável.
- **P-E (ESTIMAÇÃO).** Superfícies com CI, sem threshold: fração-censurada ×
  (T, m) [T-sweep no softplus]; medianas de t_res e duração × m; FRAÇÃO-LIMPA
  × m (risco-1: se ~0 em m=256, a lei é fenômeno de baixa densidade — é
  resultado); qualidade da predição-área (vitórias, erro relativo) × m — a
  curva de degradação por m corpos; frações de morte-por-saída e v_cruzou × m.
- **P-NULL (validade do instrumento, pré-registrado).** No braço `sem_grad`
  (difusão pura, sem gradiente): os critérios de P-A NÃO podem passar (área não
  pode vencer o placebo). Se passarem, o instrumento tem viés mecânico e TODO o
  experimento é inválido — nota negativa.

## 4. Desenho congelado

Dados: 1D-reg `y=sin(2x)+0.3x`, x~U[0,2], **n=32** (o rascunho dizia n≈256;
corrigido para o instrumento calibrado — desvio documentado), full-batch.
Arquitetura: f(x)=Σ v_k·act(w_k x+b_k); init w,b~N(0,1), v~N(0,1/m).
**Device: CPU pinado** (B4; modelos minúsculos — GPU não paga o overhead e
quebra o pareamento por seed; `use_deterministic_algorithms(True)`,
`torch.set_num_threads(1)` por worker; paralelismo entre runs via Pool nice-19).

| # | Braço | ativ | T | Papel |
|---|---|---|---|---|
| 1 | softplus | softplus β=300 | 1e-3 | primário (P-A, base de P-B/C/D) |
| 2 | erf | erf-smoothing β=300 | 1e-3 | P-B in-class |
| 3 | gelu | GELU padrão | 1e-3 | P-B′ exploratório |
| 4 | relu | ReLU dura | 1e-3 | P-C âncora |
| 5 | T0 | softplus β=300 | 0 | competidor-2 (escape térmico) |
| 6 | freeze | softplus β=300 | 1e-3 | P-D (freeze-parcial) |
| 7 | embaralha | softplus β=300 | 1e-3 | controle de geometria (report) |
| 8 | semgrad | softplus β=300 | 1e-3 | P-NULL (difusão pura) |
| 9,10 | T-sweep | softplus β=300 | 3e-4, 3e-3 | P-E superfície |

Grade: braços × **m ∈ {16, 64, 256}** = 30 células × **20 seeds virgens
(1700–1719)** = 600 runs. η=3e-4 (βη=0.09), horizonte **100 000 passos**
(τ=30; dimensionamento: ~1 min/run ⟹ ~10 core-h ⟹ ~1.7 h no Pool 6).
φ_v=1/(4√m); q=2; q_j=0; mínimo 30 episódios limpos/célula para MATA
(abaixo: célula "limpos-insuficientes" — vira resultado P-E).

## 5. Análise e potência

`analise.py` (testada em dados SINTÉTICOS no smoke-gate — regra do 5º parecer)
lê `results/*.jsonl.gz` + config e emite `resultados.json` + figuras.
Agregação: seed é a unidade (Bouthillier); IQM/mediana + bootstrap
estratificado por seed (B=10 000, RNG fixo). Potência/nível P-A: demonstrados
por SIMULAÇÃO no `--sintetico` (FPR≈α sob o nulo truncado em 8/40/200
episódios/seed; TPR=1.0 sob memória), condicionais a episódios/seed
suficientes (piloto: ~170 completos/seed já em 20k passos, m=64 — folga sobre
o mínimo de 3). Piso de |Δ| (P-B 5%, P-D 1% com S≥5 seeds) pré-declarado
contra viés-de-fio-de-cabelo (herança do parecer v2). Gate de 30 limpos por
célula vale para P-A E P-B (7º parecer).

## 6. Riscos e desvios

Riscos herdados do rascunho §8 (lavagem por m corpos = risco-1; censura;
chatter na fronteira) mitigados por filtro-limpo + censura + limiar calibrado.
**Desvios pós-execução: (nenhum ainda — seção obrigatória).**

## 7. Ordem de execução

1. `preditor.py` → escreve `config_etapa4.json` (freeze). 2. Re-revisão
adversarial focada nos blockers. 3. Commit = pré-registro. 4. `run.py`
(Pool 6, nice-19, vigia de load). 5. `analise.py` → nota em 06-notas +
NEGATIVOS.md se for o caso. Registro: git local (OSF adiado por decisão do
pesquisador — circulação suspensa).
