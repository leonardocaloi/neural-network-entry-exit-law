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
predição-área identificado como o atraso O(1/β) do PRÓPRIO teorema (4.54%→1.91%
quando β 300→1000, razão 0.42); (b) sob ruído T=1e-3: área vence sinal 34/34,
erro relativo mediano 0.534 vs 0.851 da duração; (c) unit-tests dos flags.
V1 (`calibracao_regua.py`) fica como registro do instrumento anterior.

**Episódio limpo** (`episodio_limpo()`, pré-registrado): ≤ q switchings de
outros neurônios no intervalo, sem v_cruzou, v_min_abs ≥ φ_v, ≤ q_j trocas de
líder. Censura: episódio aberto no horizonte NUNCA conta como permanente-por-
mecanismo; entra como censurado.

## 3. Predições congeladas (números em `config_etapa4.json` via `preditor.py`)

- **P-A (MATA — memória, o núcleo).** Episódios limpos, braço softplus T=1e-3,
  por célula m. Pareado por episódio: erro |t_res−t_área| vs |t_res−t_sinal|.
  Estatística primária: fração-de-vitórias POR SEED; teste Wilcoxon entre
  seeds contra 0.5, unilateral, α=0.01, nas 3 células de m.
  **Nulo assimétrico (competidor 3):** placebo de atraso-permutado — para cada
  episódio, predição placebo t_morte_i + (t_área_j − t_morte_j) com j sorteado
  entre os OUTROS episódios limpos da mesma célula (RNG fixo no config). A área
  tem de vencer TAMBÉM o placebo (Wilcoxon por seed, α=0.01). MATA: se a área
  não vence o sinal (memória morre — colapsa em complementaridade), OU vence o
  sinal mas não o placebo (era só viés-de-atraso, não predição).
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
estratificado por seed (B=10 000, RNG fixo). Potência P-A: com 20 seeds e a
variância por seed do toy recalibrado, Wilcoxon unilateral α=0.01 detecta
fração-de-vitórias ≥ ~0.62 com potência ≥0.8 (MDE documentado no preditor;
toy deu 1.00). Piso de |Δ| (P-B 5%, P-D 1%) pré-declarado contra
viés-de-fio-de-cabelo (herança do parecer v2).

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
