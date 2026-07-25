# etapa-4a.3 — CONFIRMATÓRIO: estatística da ressurreição-por-mecanismo em MLP (T=0)

**STATUS: PRÉ-REGISTRADO — este commit é o pré-registro.** Trilha: 9º parecer
CORRIGIR-DESENHO → piloto-de-taxa (240 runs) → pacote candidato (a622224) →
10º parecer NÃO-APTO (estimando C2 estrutural — `parecer-10-pacote-4a3.md`) →
correções textuais + fiação alpha/B → lista mínima do parecer atendida ⟹ APTO.
z_park = min de zmax no episódio (leitura confirmada; C3 exploratório).
Supersede `README-4a3-rascunho.md`. Todos os números de
desenho vêm do **piloto-de-taxa** (240 runs, EXPLORATÓRIO) e foram derivados
PROGRAMATICAMENTE por `preditor_4a3.py` para `config_4a3.json` (nada digitado à mão).

Etiqueta: **[ESPECULAÇÃO⟶EVIDÊNCIA]** — a existência tem evidência exploratória; o
confirmatório busca N e CI. Regime **DETERMINÍSTICO (T=0)**; o estocástico é
explicitamente fora de escopo (paper próprio).

---

## 0. O que mudou desde o rascunho (resposta ao 9º parecer)

O 9º parecer deu **CORRIGIR-DESENHO** (instrumento são; mecanismo existe; mas o
sizing S=60 era contradito pelos próprios dados — a nota assumia P(volta|morte)=0.70,
o piloto mostra ~0.13–0.38). Correções aplicadas: (i) **piloto-de-taxa** rodado; (ii)
estimando de C1 recondicionado à contagem de mortes; (iii) **instrumento GT-ÁREA**
substitui a régua v2 no caminho crítico (relógio único ground-truth); (iv) C1/C2
reescritos com MATA condicional e denominador correto; (v) `motor/preditor/analise/run`
4a3 criados. As 10 respostas item-a-item estão na §5.

## 1. Perguntas pré-registradas

- **C1 (EXISTÊNCIA-COM-TAXA — primário, braço β3).** Estimando =
  **p(volta | morte dinâmica ELEGÍVEL)**, Clopper-Pearson. Elegível = t_morte ≤ H−τ,
  τ = q0.9 do return-time β3 do piloto. **MATA:** com N_elig ≥ n*, CP-superior de
  p(volta) < ½·(CI-inferior do piloto) = **0.076** ⟹ mecanismo não reproduz na escala.
  **N_elig < n* ⟹ INCONCLUSIVO** (nunca negativo). relu = controle de validade
  (zero é TEOREMA, grad≡0; não é braço-pareado — divergem do passo 1).
- **C2 (LEI-DE-ÁREA — o CORAÇÃO, braço β3).** Estatística = fração de voltas
  dinâmicas cuja **área do drive-líder ABRE (t_área ≥ 0) durante o episódio
  morto, antes da volta**. **Honestidade estrutural (10º parecer):** no relógio
  único GT, t_morte ≤ t_área < t_res é GARANTIDO POR CONSTRUÇÃO sempre que a
  área abre (t_área só é setada em passos mortos; o episódio fecha na volta) —
  o ÚNICO modo de falha empírico é **volta sem área**, e é ele que carrega toda
  a potência do MATA (1 falha em 6 voltas já mata; voltas conduzidas
  fora-do-líder produziriam exatamente isso). NÃO reivindicar "lei de ordenação
  temporal" como achado. **Disclosure (paridade com C1):** o piloto prevê
  fração≈1.0 sob o instrumento GT ⟹ SUSTENTA é o desfecho esperado. **MATA:**
  CP-inferior < 0.5 com ≥6 voltas. **<6 voltas ⟹ INCONCLUSIVO.** Denominador =
  TODAS as voltas dinâmicas; born-dead excluídos.
- **C3 (GAP-DE-SUBIDA — estimação, sem MATA).** gap = t_res − t_área vs profundidade
  **z_park** (min de zmax no episódio GT-morto) e canal **σ(β·z_park)**. Alimenta o paper.
- **C4 (CANAL-β — estimação, sem MATA).** Escada p(volta|morte) por β∈{3,10,300}
  com CIs; predição qualitativa β3 ≥ β10 ≫ β300≈0 (piloto: 0.375 / 0.222 / 0.000).
- **VALIDADE (obrigatória).** QUALQUER volta ground-truth no braço relu ⟹ bug ⟹
  experimento INVÁLIDO (teorema grad≡0).

## 2. Instrumento GT-ÁREA (a correção central de C2)

O piloto expôs que a **régua v2 era o gargalo de C2**, não a lei: das 10 voltas
dinâmicas, só 3 eram ordenadas ESTRITAS — mas 3 eram EMPATES de discretização
(t_área==t_morte) e 4 eram CEGUEIRA da v2 (volta real sem área registrada). O
instrumento novo elimina os dois:

- **relógio ÚNICO = ground-truth:** morte = zmax_k<0 em todos os dados; volta =
  retorno a zmax_k>0 (mesmo contador-mestre do piloto). Sem a v2 marcando morte
  ~126 passos antes e chamando morte-dinâmica de born-dead.
- **área ancorada no relógio GT:** durante o episódio GT-morto acumula
  Â_k += R̂_k·η, com R̂_k = v_k·[(y−f)_{j*} + v_k·a_{j*,k}] (a MESMA projeção-líder da
  série, sem régua v2 no caminho crítico); **t_área = 1º passo com Â_k>0.**
- regra **t_morte ≤ t_área** absorve o empate de discretização; a cegueira some
  porque a área é da própria trajetória, não de um estimador paralelo.

`motor_4a3.py` prova **paridade bit-a-bit da dinâmica e do contador-mestre** com
`motor_taxa` (Δloss = 0.0e+00 em 5 casos; teste `_paridade` incluído). O instrumento
é aditivo e read-only ⟹ dispensa `rec_neuronio>0` (o pedido do parecer para ter
z_park via v2 fica superado — GT-ÁREA dá z_park régua-independente inline).

## 3. Desenho e números CONGELADOS (de `config_4a3.json`, origem no piloto)

Sistema do piloto (inalterado): n=16 dados, y=sin(2πx)+0.3sin(5πx), init padrão,
η=1e-3, **T=0**, full-batch, CPU, determinismo. **H=5e5.** Braços {sp-b3(primário),
sp-b10, sp-b300, relu} × m∈{4,6} × **S=50 seeds virgens 1900–1949** = **400 runs**.

| número | valor | origem (chave em `taxa-resultado.json`) |
|---|---|---|
| τ = q0.9 return-time β3 | **192 464.5** | `tau.sp-b3` |
| janela elegível [0, H−τ] | **[0, 307 535.5]** | H − τ, H=5e5 |
| λ_CI-inf β3 (bootstrap por seed) | **0.16667** | `por_braco.sp-b3.rate_boot[1]` |
| mortes esperadas β3 (100 runs) | **16.667** | λ_CI-inf × 100 |
| **n\*** (gate INCONCLUSIVO) | **10** | quantil-inf-95% de Poisson(16.667) |
| p(volta) CI-inf piloto β3 | **0.15198** | `por_braco.sp-b3.p_volta_morte[1]` |
| **MATA C1 thresh** | **0.07599** | ½ × CI-inf piloto |
| C2 piso / min voltas | **0.5 / 6** | 9º parecer |
| escada C4 piloto (β3/β10/β300/relu) | 0.375 / 0.222 / 0.000 / 0.000 | `p_volta_morte[0]` |
| RNGs bootstrap (novos) | rate=314159, prop=271828 | — (proíbe 777/779/5xxx/9090) |

## 4. Resultados do piloto-de-taxa que fundamentam o sizing

Grade: 240 runs, 4 braços × m{4,6} × 30 seeds 100–129, H=5e5, T=0.
**VALIDADE:** relu 0 voltas / 0 born-rev / 0 rev_real em 60 runs (13 mortes
dinâmicas, todas terminais) ⟹ instrumento válido.

| braço | mortes-din | λ/run (boot) | voltas | p(volta\|morte) CP |
|---|---|---|---|---|
| sp-b3  | 16 | 0.267 [0.167, 0.383] | 6 | **0.375 [0.152, 0.646]** |
| sp-b10 | 18 | 0.300 [0.183, 0.433] | 4 | 0.222 [0.064, 0.476] |
| sp-b300| 13 | 0.217 [0.117, 0.333] | 0 | 0.000 [0.000, 0.247] |
| relu   | 13 | 0.217 [0.117, 0.333] | 0 | 0.000 [0.000, 0.247] |

**Canal (C4) confirmado:** p(volta|morte) monótona em β; o canal e^{βz} FECHA quando
β→∞ (β300 = relu = 0 voltas). Taxa de MORTE é β-independente (~0.22–0.30/run); a VOLTA
é β-dependente ⟹ **alvo primário β3** (maior p(volta), retorno mais rápido, mais perto
da suavidade GELU). Return-time β3: n=6, mediana 45.8k, **τ=192k**, máx 308k — cauda
assimétrica ⟹ censura tratada com a janela [0,H−τ], não estendendo H.

## 5. As 10 respostas ao 9º parecer (freeze item a item)

| # | item de freeze (9º parecer) | resposta da casa (onde) |
|---|---|---|
| 1 | Piloto-de-taxa (pré-requisito) | **FEITO**: 240 runs, `taxa-resultado.{md,json}` |
| 2 | Estimando C1 = p(volta\|morte elegível) CP | `config_4a3.C1.estimando`; `analise_4a3.avalia_C1` (CP exato sem scipy) |
| 3 | MATA condicional (n*, piso, α; N<n*⟹inconclusivo) | n\*=10, thresh=0.076, α=0.05 congelados; regra em `avalia_C1` |
| 4 | Janela [0,H−τ] + terminal-vs-censurado + dois p + person-time | janela [0,307 535]; `classifica_run` separa terminal/censurado; reporta p(volta\|morte) **e** p(volta\|elegível); ret-times por episódio p/ KM |
| 5 | C2: todas-as-voltas, estritas, born-out, sizing por C2 | denominador = todas voltas dyn; estimando reescrito no 10º parecer: **área-abriu (t_área ≥ 0)** — a ordenação temporal é estrutural no relógio GT, não achado; born-dead fora; sizing por C2 no piloto |
| 6 | Contador ground-truth + regra dyn/born | contador-mestre GT (zmax<0→>0); classe born(t_morte=0)/dyn(t_morte>0) em `motor_4a3` |
| 7 | Grade β∈{3,10,300}; C4 sem MATA com piso | grade nos 4 braços; `escada_C4` sem MATA |
| 8 | relu = validade | `avalia_validade`: qualquer volta GT no relu ⟹ INVÁLIDO |
| 9 | RNGs virgens + clustering por seed | seeds 1900–1949; bootstrap por seed (rate/prop), RNGs 314159/271828 (fora de 777/779/5xxx/9090) |
| 10 | config_4a3 + preditor_4a3 + runner (rebind v2+GT, CPU) | `preditor_4a3.py`→`config_4a3.json`; `run_4a3.py` (Pool 4, nice-19, setsid, manifest SHA256+git+config) |

Achados novos do parecer também endereçados: contador é **ground-truth** (não v3);
CIs com **clustering por seed**; runtime ~4.5 min/run ⟹ 400 runs ≈ 8 h Pool 4 noturno.

## 6. Pipeline e verificações já rodadas (pré-registro)

- `motor_4a3.py --paridade`: **loss Δ=0.0e+00** vs motor_taxa em 5 casos (dinâmica +
  contador-mestre bit-idênticos); consistência interna dos episodios_gt.
- `preditor_4a3.py`: gera `config_4a3.json` (todos os números com origem no piloto).
- `analise_4a3.py --sintetico`: **passa TUDO** — 5 mundos com asserts nos dois sentidos
  (i taxa-piloto+lei-ordenada → C1 reproduz, C2 sustenta; ii taxa-zero → C1 MATA;
  iii poucos-eventos → inconclusivos; iv trios desordenados → C2 MATA; v relu-com-volta
  → INVÁLIDO) + CP/Poisson exatos batendo com valores conhecidos (CP(0/10)=0.3085,
  CP(6/6)=0.5407, PoisCI(13,60), n*=10).
- `run_4a3.py --smoke`: 2 runs H=2e4 (seeds 42-43, fora das virgens) — mecânica OK,
  manifest com SHA256+git+config. **Nenhuma run confirmatória foi executada.**

## 7. Como rodar (SÓ após pré-registro + 10º parecer)

```
cd pilotos-4a2/pressao/taxa
setsid nice -19 ../../../../../venv/bin/python run_4a3.py --workers 4 \
       > run_4a3.log 2>&1 < /dev/null &      # 400 runs, retomável, destacado
../../../../../venv/bin/python analise_4a3.py   # lê results-4a3/ + config
```

## 8. Pendências para o 10º parecer

- **C1-MATA é intencionalmente difícil de disparar** neste N: β3 rende ~27 mortes
  elegíveis esperadas (ponto) em 100 runs, mas CP-superior(0, N) só cai abaixo de
  0.076 com **N ≳ 47 mortes elegíveis de zero-volta**. Logo, na prática, **C2 é o
  teste que decide**; C1 tende a "reproduz" ou "inconclusivo". É consequência honesta
  do freeze (parecer já apontava C1 fraco) — sinalizado, não escondido.
- **z_park** está definido como min de zmax no episódio GT-morto (fundo do park);
  alternativa (zmax no instante de abertura da área) é possível — decisão de C3
  (exploratório, sem MATA). Confirmar leitura.
- **λ_CI-inf** usado no gate é o **bootstrap-por-seed** (`rate_boot[1]`=0.167), mais
  conservador/honesto a clustering que o Poisson-iid (`rate_pois[1]`=0.152). Confirmar.
- Return-time τ vem de **n=6 voltas** no piloto (ruidoso); a janela é robusta porque a
  maioria das mortes é precoce (t_morte<30k), mas o τ em si tem incerteza larga.
