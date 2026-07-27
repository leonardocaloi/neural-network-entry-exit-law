# Confirmatório H-BARREIRA — pré-registro

Status: **[PRÉ-REGISTRO]** — escrito ANTES de qualquer execução do confirmatório.
Data: 2026-07-26. **Emendado 2026-07-27 pelo 17º parecer** (`parecer-17.md`),
ainda antes de qualquer execução. Herda do piloto **[EXPLORATÓRIO]** e7
(`../README-e7.md`, `06-notas/2026-07-26-e7-piloto-adam.md`).
Nada deste diretório foi rodado além dos *smoke tests* declarados em §8.

Config congelado: `results/config_hbarreira.json`
**sha256 `49c3051e323271d3428d025ae03f44587a9e77e3e7afd43aaca37843cf673ec7`**
(regerado 2× byte-igual após a emenda; o SHA anterior à emenda está morto).

---

## 1. AS PREDIÇÕES, ANTES DE TUDO

**Convenção de sinal (fixada aqui, uma vez).** A variável ajustada é a
**profundidade** da barreira, `y := −β·z_park > 0`. O piloto reportou
`β·z_park ≈ −0,94·ln(1/ε) − 6,1`; nesta convenção isso é
`y ≈ +0,94·ln(1/ε) + 6,1`.

### PRIMÁRIO — H-BARREIRA (ÚNICO; não há segundo primário)

Na regressão sobre todas as instâncias e todos os ε da **grade primária** da
célula-âncora (β = 1e4, β₂ = 0,999):

```
    y(i, ε) = −β·z_park(i, ε) = a·ln(1/ε) + b + u_i + ruído
```

com **grade primária ε ∈ {1e−8, 1e−6, 1e−4, 1e−2}** — **ε = 1 fica FORA do
ajuste primário** (§1.1). O mecanismo proposto — o release acontece quando o
canal `e^{βz}` fica **proporcional** ao piso `ε` (`|g|_park = θ·ε`, θ constante;
ver Apêndice A.3, que corrige o "≈ ε" do texto original) — prediz `a = 1` como
lei física, e o Apêndice A.4/A.5 mostra que **sob este desenho** o valor
esperado é ligeiramente menor:

> ### PREDIÇÃO PRINCIPAL DECLARADA: **`a ≈ 0,97`**.
> A leitura esperada é **"SUSTENTA a lei, mas NÃO é `a = 1` exato"** — as duas
> coisas terão de ser ditas no texto do resultado, não uma só.

**Janela de decisão: [0,90; 1,10] — MANTIDA CONGELADA** (é a do pré-registro
original; não foi mexida pela emenda).

**MATA H-BARREIRA:** o IC95 de `a` (bootstrap de cluster sobre instâncias)
**excluir** o intervalo [0,90; 1,10], isto é, ser disjunto dele.

| resultado | veredito congelado |
|---|---|
| IC95(a) ⊆ [0,90; 1,10] | **SUSTENTA** |
| IC95(a) ∩ [0,90; 1,10] = ∅ | **MATA** |
| IC95(a) cruza a fronteira | **INCONCLUSIVO** (não é vitória) |
| curvatura significativa (§3) | **INCONCLUSIVO-CURVATURA** (sobrepõe as três acima) |

> **Predição derivada sobre a guarda de curvatura: ela NÃO deve disparar.**
> Sem o joelho de ε=1, o piloto dá `|q|·L²max` = **0,075** (recomputado por
> `derivacao_a1.py ajuste`) / **0,096** (verificação independente do 17º
> parecer) — uma ordem de grandeza abaixo de `CURV_TOL = 1,0`. Se ela disparar
> mesmo assim, é achado, não artefato conhecido.

**Resolução real do teste (corrigida — 17º parecer, forte 12; os DOIS números
em §7.3).** A meia-largura do IC95 de `a` é **≈ 0,0066** no mundo sintético
(N = 100, 4 ε, `sd_ruido = 0,4`) e projeta-se em **≈ 0,004** para o dado real
(resíduo within medido no piloto = 0,200). Não é 0,01 (número alegado na versão
anterior deste arquivo, folgado demais). **Os dois números estão declarados em
§7.3 justamente para que um IC real de ~0,004 não seja lido depois como
"desvio".** Qualquer um dos dois faz a regra ser bem posta: nem aceita qualquer
coisa, nem exige igualdade numérica impossível. Em particular, um
`a = 0,97 ± 0,007` (ou ±0,004) **SUSTENTA a lei e EXCLUI `a = 1`** ao mesmo
tempo — é exatamente esse o resultado que a derivação prevê.

#### 1.1 Por que ε = 1 sai do primário (adjudicação (c) restrita do 17º parecer)

Em ε = 1 o Adam já degenerou em GD: a própria fase VIVA está no regime linear, a
recalibração de `lr` salta ×8,9 e `K_morto` cai pela metade (Apêndice A.4). O
"joelho" que isso produz **não é curvatura da lei física — é vazamento de
projeto**, e derrubaria o primário por artefato (`|q|L²max ≈ 1,7`, guarda
disparando por motivo errado). Tirar ε = 1 do ajuste é a correção mínima:

- **ε = 1 e o baseline GD continuam sendo RODADOS e REPORTADOS**, como
  **pontos de saturação da secundária (iii)** — nunca no ajuste primário.
- **O custo do experimento fica inalterado** (nada deixou de ser executado).

A emenda proposta na versão anterior (§A.5) criava um **segundo** primário sobre
`y* = y − ln(lr·K_morto)` com janela própria [0,95; 1,05]. **Foi negada**: dois
primários com duas janelas são duas chances de sustentar. Um primário, uma
janela, um veredito.

#### 1.2 A lição da janela [0,95; 1,05] (registrada, e a janela abandonada)

A janela [0,95; 1,05] foi apresentada como "derivada de um orçamento de erro".
Ela **não era independente do piloto**: o orçamento que a produziu foi calibrado
no mesmo dado (as 13 instâncias exploratórias) que produziu o `a* = 0,9985` que
ela iria julgar. Isso é **auto-cumprimento do orçamento** — uma janela desenhada
para caber no número que já se tinha. **Lição registrada; janela abandonada.**
A janela vigente é a original [0,90; 1,10], escrita antes de qualquer conta de
orçamento.

### DIAGNÓSTICO (não mata) — decomposição de `a` e o RESTO INEXPLICADO

`y*` não sumiu: virou **diagnóstico**, com um teste novo que tem resolução.
Reportamos, com cada termo **medido**:

```
    a_diag = 1 + d ln(lr)/dL + d ln(K_morto)/dL + resto
```

> **NOTAÇÃO — `a_diag` NÃO é o `a` do primário** (2ª rodada do 17º parecer,
> item 2). `a_diag` (também dito `a_within`) é a inclinação estimada por
> **efeitos fixos within-instância** sobre o **subconjunto sem os runs
> truncados**; o `a` do primário é **OLS empilhado** sobre **todos** os runs com
> platô válido. **São estimadores diferentes em amostras diferentes: diferem por
> construção.** Nenhum dos dois "confirma" o outro, e a diferença entre eles não
> é resultado. Na saída de `analise_hbarreira.py` a chave é `a_diag` e vem
> acompanhada da própria definição; o `soma_confere` (resíduo da identidade
> acima) vem rotulado **"~0 POR CONSTRUÇÃO (não é evidência de nada)"**.

- `d ln(lr)/dL` — vazamento da recalibração de `lr` por ε (variável de desenho,
  registrada e medida; **não** é grau de liberdade de ajuste).
- `d ln(K_morto)/dL` — vazamento da duração da fase morta.
- `resto` — **o que a derivação não explica.** No piloto: `resto = −0,018`.

> **TESTE NOVO, PRÉ-REGISTRADO.** Se `|resto| > 0,01` **com IC95 excluindo 0**,
> o veredito do diagnóstico é **ESTIMAÇÃO-COM-ALERTA**: achado real, a
> investigar. **NÃO é MATA** — a física do resto não está derivada, e matar uma
> hipótese por um termo cuja física não se sabe seria trocar ignorância por
> veredito.
>
> **Franqueza obrigatória (disclosure §7.9):** `RESTO_TOL = 0,01` foi fixado
> **sabendo** que o piloto dá `−0,018`. **O resultado esperado é, portanto,
> ESTIMAÇÃO-COM-ALERTA.** O que se aprende não é "acendeu", e sim se o resto
> **persiste** nas 100 instâncias novas e **de que tamanho e sinal** ele sai.

**`K_morto`, definição operacional (obrigatória, 17º parecer item 3):**

```
    K_morto := (t_release − t_hit) / dt          [em PASSOS]
```

Logo **runs truncados (sem release) não têm `K_morto`** e são **EXCLUÍDOS DO
DIAGNÓSTICO**, com a contagem `n_trunc` reportada. Eles **nunca** saem do
primário — o primário usa `y` bruto, que existe para todo run com platô válido.

**Estimador do resto: WITHIN-instância (efeitos fixos), e por quê.** Achado do
próprio sintético ao aplicar a emenda: o OLS empilhado é não-viesado mesmo com
desbalanceamento, mas quando as instâncias deixam de cobrir todos os ε — e a
exclusão dos truncados é exatamente um desbalanceamento — o efeito de instância
`u_i` (sd ≈ 3) deixa de ser ortogonal a `L` e **vaza para a variância** da
inclinação: o IC95 do resto abre de ±0,0075 para **±0,02**, e um resto de
−0,018 deixaria de ser detectável. O estimador within elimina `u_i` por
construção. **Usado SÓ no diagnóstico; o primário continua sendo o OLS empilhado
congelado.** Verificado nos dois sentidos no sintético (§8).

### SECUNDÁRIA (i) — invariância em β

O piloto rodou **β = 1e4 apenas** (confirmado em `../README-e7.md` §Setup e em
`e7_adam.py: BETA = 1e4`). O mecanismo é uma condição sobre `β·z`, adimensional:
**`a` não pode depender de β**. Rodamos β ∈ {1e3, 1e4} com β₂ = 0,999.
β = 1e3 é **teste novo**, nunca visto por este projeto sob Adam.

> **Predição:** `a(β=1e3) = a(β=1e4)`. `b` pode mudar (é a constante da
> proporcionalidade `|g| = C·e^{βz}`, e `C` depende de β).

**Confound declarado e tratado (17º parecer, blocker 6).** `Δa` e `Δb` entre
células **misturam a recalibração de `lr`**: cada célula tem seu próprio `lr(ε)`,
logo seu próprio vazamento `d ln(lr)/dL`. Comparar células sobre `y` bruto é
comparar mecanismo + calibração. **Escolha implementada e documentada: a
comparação pré-registrada é feita sobre a GRADE DESCONTADA `y*`**, que remove
`ln(lr·K)` ponto a ponto. (A alternativa — forçar `lr` comum por par — exigiria
abandonar a calibração de `t_hit`, que é justamente o que impede confundir a
transição em ε com mudança de timescale; ficaria pior.) Consequência declarada:
as secundárias herdam a exclusão dos truncados do diagnóstico.

**Guarda de potência (nova, 17º parecer):** se a **meia-largura do IC95 de `Δa`
> 0,1**, o teste é declarado **SEM POTÊNCIA** e não decide nada. Antes, um IC
largo declarava INVARIANTE por omissão — o mesmo defeito que a (ii) já tratava.

**MATA (i):** IC95 pareado de `Δa = a(1e3) − a(1e4)` excluir 0 **e** |Δa| ≥ 0,1,
com potência suficiente. Se as duas condições não valerem juntas → INVARIANTE.

### SECUNDÁRIA (ii) — varredura de β₂ (a memória de `v`)

O piloto atribuiu o colapso de `|m̂/(√v̂+ε)|` (3,6e−4, quatro ordens abaixo da
saturação) à memória longa de `v` (τ₂ = 1/(1−β₂) ≈ 10³ passos). Varremos
β₂ ∈ {0,9; 0,99; 0,999} em β = 1e4. Também sobre a grade descontada `y*`.

**O que o mecanismo do ε diz que muda — e o que ele diz que NÃO muda.** Sob o
mecanismo do piso, o gate é `√v̂ ≲ ε`. Como `v` é uma média exponencial com
τ₂ passos de memória, `√v̂` lê o gradiente de ~τ₂ passos ATRÁS: a condição de
parada dispara com uma **defasagem** de τ₂ passos, o que desloca a profundidade
por um termo **aditivo**, não multiplicativo:

```
    Δ(β z_park) ≈ β·|ż|_morto · dt · Δτ₂
```

Com os números da corrida de referência (β·z desce ~23 unidades em ~3,5e5
passos ⟹ β|ż|dt ≈ 7e−5 por passo) e Δτ₂ = τ₂(0,999) − τ₂(0,9) = 10³ − 10 = 990:

> **Δb ≈ 0,07 unidades de β·z** — uma ordem de grandeza ABAIXO do resíduo do
> piloto (0,36).

Logo a predição pré-registrada **decisória** é **de equivalência**:

> **`a` invariante em β₂** (tolerância 0,1) **e** `|b(0,9) − b(0,999)| < 2`.

**MATA (ii):** IC95 pareado de `Δb = b(0,9) − b(0,999)` sair de [−2, +2], ou
`Δa` além de 0,1 com IC excluindo 0. Se a meia-largura do IC95 de Δb for maior
que 1, o teste é declarado **SEM POTÊNCIA** e não decide nada.

**A predição mais apertada `|Δb| < 0,1` é PREDIÇÃO-SOB-TESTE, não critério
(17º parecer, blocker 6).** O Apêndice A.3.3 deriva, de `κ = 1,00` medido, que
β₂ quase não entra em `b`: `|Δb| < 0,1`. Mas **o smoke test já deu
`Δb = −0,40`** (1 instância, ≈2σ pareado) — **contradiz a predição**. Registro
honesto: `|Δb| < 0,1` fica como **predição-sob-teste com evidência contrária
prévia declarada**; o critério **decisório é e continua sendo o teste de
equivalência ±2 congelado**. Não se usa uma predição já contrariada como
critério de morte, nem se apaga a contradição.

> **ETIQUETA COMPLETA (2ª rodada do 17º parecer, item 3):** `|Δb| < 0,1` é
> **predição-sob-teste, SEM POTÊNCIA POR DESENHO** — a meia-largura do IC95 de
> `Δb` é **~0,4**, quatro vezes a tolerância de 0,1. **Este desenho não consegue
> decidir 0,1 nem em princípio**, com ou sem dado. Adjudicado: a etiqueta é
> preferível a re-parear o experimento (re-pareamento custaria um desenho novo
> para responder uma pergunta que ninguém pré-registrou como decisória).
> **O ±2 segue decisório e nada mais muda.** A etiqueta está também na saída de
> `analise_hbarreira.py`, no campo `predicao_sob_teste_db_menor_0p1`.

### SECUNDÁRIA (iii) — SATURAÇÃO: ε = 1 e GD como âncoras

Formulação: quando `ε ≫ √v̂`, o update de Adam vira `lr·m̂/ε`, **linear em g** —
o piso deixa de existir como gate e o passo volta a ser (uma versão suavizada
por momento de) GD. Nesse sentido **GD é a célula ε → ∞**, e a lei
`y = a·ln(1/ε) + b` deve **saturar** nele, não extrapolar através dele:
`ln(1/ε) → −∞` daria `y → −∞`, o que é fisicamente impossível (a profundidade é
limitada pelo balanço de GD). O piloto viu exatamente isso: ε=1 → y = 6,4;
GD → y = 5,0. **É este o papel que ε = 1 tem agora: ponto de saturação, âncora
da ponta direita — não ponto do ajuste primário.**

> **Predição (direcional, não é o teste principal):** em **todo** ε da grade
> EXECUTADA (inclusive ε = 1), a mediana de `y_Adam(ε)` é MAIOR que a mediana de
> `y_GD`: Adam estaciona mais fundo que GD em qualquer ε.

**MATA (iii):** algum ε com mediana `y_Adam < y_GD`. Isso não mata H-BARREIRA;
mata a leitura "o piso APROFUNDA o portão", que é a história física.

O "controle de vazamento `y_GD` não depende de ε" foi **REMOVIDO** (17º parecer,
cosmético 16): é vácuo — o harness de GD não recebe `ε` por construção, então a
verificação não pode falhar. Um controle que não pode falhar não é controle.

### P-NULL de instrumento (obrigatório)

Se menos de **60%** das células (instância × ε) **da grade primária** da âncora
entregarem `z_park` válido (§2, critério de platô), nada é decidido: veredito
**INSTRUMENTO-NULL**, registrado como negativo de instrumento, sem reciclar
hipótese.

---

## 2. Desenho

- **Instâncias: N = 100**, sorteadas do **mesmo prior** do sweep
  `05-experimentos/t2-sweep-instancias` (`sorteia()`, faixas idênticas) e
  filtradas por (H1)–(H4) pelo **mesmo código** (`sweep_instancias.uma`).
  Taxa histórica de aprovação: 57/600 = 9,5% ⟹ `N_SEEDS_MAX = 1600` seeds.
- **Bloco de seeds NOVO: 20270000+**, disjunto do bloco do piloto
  (20260726–20261325). A **primeira** instância aprovada é a de **CALIBRAÇÃO de
  lr** e fica FORA da análise; as 100 seguintes são o confirmatório.
- **Grade de ε EXECUTADA: {1e−8, 1e−6, 1e−4, 1e−2, 1}** — 5 pontos, espaçamento
  uniforme de 2 décadas em `ln(1/ε)` ∈ {18,4; 13,8; 9,2; 4,6; 0}.
  **Grade PRIMÁRIA (do ajuste): {1e−8, 1e−6, 1e−4, 1e−2}**, 4 pontos.
  `EPS_GRID` e `EPS_PRIMARIO` são chaves separadas do config (entram no sha256).
- **Células** (um-fator-por-vez a partir da âncora, ver §5):
  `ancora` (β=1e4, β₂=0,999) · `beta-baixo` (1e3, 0,999) ·
  `b2-099` (1e4, 0,99) · `b2-090` (1e4, 0,9). Mais o **baseline GD** nos dois β.
- **lr recalibrado por (β, β₂, ε)** por bisseção em log(lr) para casar
  `t_hit^Adam = t_hit^GD` na instância de calibração — mesmo procedimento do
  piloto, e é o que impede que a transição em ε seja confundida com mudança de
  timescale. Nenhum ajuste posterior. **É também a fonte do vazamento
  `d ln(lr)/dL`**, medido e reportado no diagnóstico (§1).
- **Horizonte H = 10,0** (piloto usou 8,0; fase morta mediana ~3,6 no ε pequeno).
  dt = 1e−5, β₁ = 0,9 fixo, FACE = 0 — todos herdados do piloto.
- **`z_park` desacoplado do release** (correção de viés introduzida aqui, não
  havia no piloto): `z_park` é válido (`plato = True`) se (a) houve release —
  então o mínimo é interior por construção, `z` desceu e voltou a 0 — ou (b) o
  run truncou no horizonte MAS o mínimo já se sustentou por 0,5 unidade de tempo
  dentro de 0,5/β. Sem isso, as instâncias truncadas dariam `z_park` censurado
  (viés para fundo) e as células de ε grande, de fase morta curta, seriam
  excluídas em bloco.
- **`estacionario`: verificação de platô calculada SEMPRE, inclusive nos runs COM
  release** (17º parecer, forte 9). É **diagnóstico**, reportado por ε: mede se
  o mínimo se sustentou por 0,5 dentro de 0,5/β, sem exigir isso de quem teve
  release. **Fato já observado no smoke: ε = 1 FALHA a estacionariedade**
  (`bz_retorno = 0,545` contra tolerância 0,50). Isso agora é **reportado**, e é
  **coerente com o papel de ponto de saturação** que ε = 1 passou a ter — não
  contamina o primário, porque ε = 1 não está no primário.
- **Filtro de validade: apenas `plato`.** A guarda `n_mortes ≠ 1` foi
  **REMOVIDA** (17º parecer, forte 7): com o motor como está, a flag `morto` só
  vira `True` uma vez, logo `n_mortes` nunca passa de 1 e a guarda **nunca podia
  disparar** — era decorativa e dava falsa sensação de filtro. `n_mortes`
  continua sendo GRAVADO em cada registro.
- **IC por bootstrap de CLUSTER sobre instâncias** (reamostra a instância
  inteira com seus ε), B = 10000, semente 20270001. Comparações entre células
  usam bootstrap **pareado** (as células compartilham as mesmas instâncias).

## 3. Análise (congelada em `analise_hbarreira.py`)

**Tudo que este README decide existe em código congelado**, e o sintético
exercita cada regra nos dois sentidos (17º parecer, blocker 5).

- **Primário:** OLS empilhado de `y` sobre `ln(1/ε)`, **grade primária** (sem
  ε=1); `a`, `b`; IC por bootstrap de cluster. Janela [0,90; 1,10].
- **Guarda de curvatura pré-registrada:** ajusta-se também `y = q·L² + a·L + b`;
  se `|q|·L²max > CURV_TOL` **e** o IC95 de `q` excluir 0, a lei linear é
  declarada inadequada e o primário vira **INCONCLUSIVO-CURVATURA**.
  `CURV_TOL = 1,0` **agora vive no `CFG`** (cosmético 13), logo entra no sha256
  do config — antes era uma constante solta no arquivo de análise, fora do hash.
- **Diagnóstico** (§1): `d ln(lr)/dL`, `d ln(K)/dL`, `resto`, IC95 do resto,
  `n_usado`, `n_trunc`, e a checagem numérica de que a identidade fecha.
  Veredito `ESTIMAÇÃO-COM-ALERTA` ou `RESTO DENTRO DA TOLERANCIA`.
- **Secundárias (i) e (ii):** sobre `y*`, com guardas de potência.
- **Secundária (iii):** medianas por ε na grade EXECUTADA contra a mediana de GD.
- **Estacionariedade por ε** é reportada junto do resultado real.
- **Subcomando obrigatório** (forte 11): `python analise_hbarreira.py` sem
  argumento, ou com argumento desconhecido, **ERRA** com código 1. Antes o
  default era `sintetico` e o `else` caía em `real` — um typo (`rael`) viraria um
  **peek** não declarado no dado do confirmatório.

Exclusão de células: `plato = False`. A fração excluída é reportada por ε (e
alimenta o P-NULL de instrumento).

**Tabela de vereditos — uma linha por pergunta, sem dupla chance:**

| pergunta | base | veredito possível |
|---|---|---|
| PRIMÁRIO H-BARREIRA | `y` bruto, grade primária | SUSTENTA / MATA / INCONCLUSIVO / INCONCLUSIVO-CURVATURA / INSTRUMENTO-NULL |
| diagnóstico do resto | `y*`, within-instância | RESTO DENTRO DA TOLERANCIA / ESTIMAÇÃO-COM-ALERTA (nunca MATA) |
| secundária (i) β | `y*` | INVARIANTE / NAO-INVARIANTE / SEM POTENCIA |
| secundária (ii) β₂ | `y*` | MECANISMO-V SUSTENTADO / REFUTADO / SEM POTENCIA |
| secundária (iii) saturação | grade executada + GD | SATURACAO COERENTE / SATURACAO VIOLADA |

## 4. Custo (medido onde diz medido)

Constantes desta máquina em 2026-07-26, **1 núcleo, `nice -19`**, na instância de
referência, H = 8:

| ε | s/run | origem |
|---|---|---|
| 1e−8 | 28,4 | **MEDIDO** |
| 1e−6 | 26,0 | **INTERPOLADO** (declarado — 17º parecer, cosmético 14) |
| 1e−4 | 22,0 | **INTERPOLADO** |
| 1e−2 | 16,0 | **INTERPOLADO** |
| 1    |  9,0 | **MEDIDO** |
| GD   |  5,0 | **MEDIDO** |

As três entradas intermediárias são interpolação a olho entre os dois extremos
medidos, **não** cronometragens. Se o custo real divergir, vai para §9.
Padding ×1,6 para H = 10, instâncias mais lentas que a de referência e a fração
que trunca (paga o horizonte inteiro).

| item | core-h |
|---|---|
| sweep de instâncias (1600 seeds) | 10,1 |
| calibração de lr (4 células × 5 ε) | 0,9 |
| grade Adam (4 células × 5 ε × 100 inst) | 18,0 |
| baseline GD (2 β × 100 inst) | 0,4 |
| **TOTAL** | **29,4 core-h** |

Parede: **9,8 h** com Pool(3) · **7,4 h** com Pool(4) · **4,9 h** com Pool(6).
Reproduzir: `python preditor_hbarreira.py custo`.
**O custo não mudou com a emenda**: ε = 1 continua sendo rodado (como ponto de
saturação); o que mudou foi só o que entra no ajuste primário.

## 5. Por que NÃO o fatorial cheio (justificativa escrita)

O fatorial 2β × 3β₂ × 5ε × 100 custaria 27,0 core-h só na grade Adam (total
~38 core-h ⟹ ~13 h de parede com Pool(3), no limite do orçamento). O desenho
adotado é **um-fator-por-vez a partir da âncora**: 18,0 core-h na grade Adam,
33% mais barato. Justificativa: **nenhuma hipótese pré-registrada usa a
interação β × β₂** — a secundária (i) pergunta se `a` depende de β com β₂ fixo,
e a secundária (ii) pergunta se `a`/`b` dependem de β₂ com β fixo. Um fatorial
cheio compraria apenas a interação, que ninguém predisse e que não entraria em
nenhum critério de morte. **Consequência aceita e declarada:** se a secundária
(ii) refutar (β₂ importa muito), o confirmatório NÃO saberá dizer se o efeito de
β₂ é o mesmo em β = 1e3 — isso viraria um experimento novo.

## 6. Ordem de execução

```
source venv/bin/activate
cd 05-experimentos/e7-adam-portao/confirmatorio
nice -19 python preditor_hbarreira.py config    # congela results/config_hbarreira.json (com sha256)
nice -19 python preditor_hbarreira.py inst      # 1600 seeds -> 101 instâncias
nice -19 python preditor_hbarreira.py calib     # 20 lr
nice -19 python preditor_hbarreira.py gd
for c in ancora beta-baixo b2-099 b2-090; do
  for e in 1e-8 1e-6 1e-4 1e-2 1; do nice -19 python preditor_hbarreira.py run $c $e; done
done
nice -19 python analise_hbarreira.py sintetico  # TEM de passar antes de olhar o real
nice -19 python analise_hbarreira.py real
```
Estágios resumíveis (o ambiente reinicia processos de fundo). `run` e `gd` fazem
*append* em `results/hbarreira_runs.jsonl` — e agora há **TRAVA
ANTI-DUPLICAÇÃO** (17º parecer, cosmético 17): se o arquivo já contém linhas da
mesma chave (célula, ε) ou de GD, o estágio **ABORTA** com mensagem. Antes o
README pedia "truncar antes de repetir", que é um pedido de disciplina onde cabe
uma trava. Se uma rodada anterior foi parcial ou inválida, as linhas têm de ser
removidas EXPLICITAMENTE e o fato registrado em §9.

## 7. DISCLOSURES

1. **As 12–13 instâncias do piloto e7 são EXPLORATÓRIAS e ficam FORA deste
   confirmatório.** Seeds novas (bloco 20270000+), disjuntas de 20260726–20261325.
   A instância de referência do paper (`X=[0.5,1.5]`, `Y=[0.5,1.0]`, …) também
   fica fora da análise: ela gerou a hipótese.
2. A hipótese testada aqui (**H-BARREIRA**) é **nova**: não é H-SIGN nem H-EPS
   do piloto. H-SIGN foi *refutada como mecanismo* e H-EPS foi confirmada
   exploratoriamente; nenhuma das duas é reciclada. Hipótese nova = experimento
   novo, com pré-registro novo — este arquivo.
3. `a = 1` é a predição do mecanismo; `a ≈ 0,97` é a predição **derivada sob
   este desenho** (§1), e é ela que está declarada como resultado esperado. A
   regra de decisão [0,90; 1,10] foi escrita como equivalência **antes** de
   qualquer dado novo, e o piloto (0,973 na grade sem ε=1) está dentro dessa
   janela: **isto é assimétrico a favor de sustentar, e está declarado como
   tal.** O que protege o teste é o IC estreito: um `a` de 0,97 com IC
   [0,963; 0,977] SUSTENTA a lei mas **exclui** `a = 1` — o texto do resultado
   terá de dizer as duas coisas.

   **OS DOIS NÚMEROS DA MEIA-LARGURA, declarados AGORA para que nenhum deles
   vire "desvio" depois:**

   | número | de onde vem |
   |---|---|
   | **0,0066** | mundo **sintético**, N=100, 4 ε, com `sd_ruido = 0,4` |
   | **≈ 0,004** | **projeção para o dado real**: o resíduo within medido no piloto é **0,200**, não 0,4 — o sintético foi deliberadamente conservador |

   Ambos são **15–25× mais estreitos que a janela ±0,10**. Portanto: se o
   confirmatório entregar uma meia-largura de ~0,004, isso é **o esperado**, e
   **não** um desvio a registrar em §9. O que iria para §9 seria uma
   meia-largura **maior** que os 0,0066 do sintético — aí sim o instrumento
   teria menos resolução do que o pré-registro previu.
4. A calibração de lr usa uma instância que fica fora da análise, mas é a MESMA
   para as 100 — o lr não é ótimo por instância. Herança do piloto, declarada.
   É também a origem do vazamento `d ln(lr)/dL` medido no diagnóstico.
5. **Peek declarado (smoke, §8):** o smoke test rodou 1 instância (a de
   referência, EXCLUÍDA da análise) em 3 configurações. Ele mostrou
   `b(0,9) − b(0,999) = −0,40` em ε=1e−8 — **sinal oposto** ao do argumento de
   defasagem de §1(ii) e **magnitude 4× a predição `|Δb| < 0,1` do Apêndice
   A.3.3**. Esta contradição está agora registrada onde importa: `|Δb| < 0,1`
   é **predição-sob-teste com evidência contrária prévia**, e **não** é critério
   de morte. O critério decisório continua sendo a equivalência ±2, derivada a
   priori (0,07) e congelada. Registrado para que ninguém descubra depois.
6. A regra de platô (§2) foi ajustada **depois** do smoke e **antes** de
   qualquer dado do confirmatório. **DIREÇÃO DA MUDANÇA: ela AFROUXOU**
   (17º parecer, forte 8). A primeira versão exigia sustentação por 0,5 também
   nas corridas com release, e **reprovava ε=1** (`bz_retorno = 0,545` contra
   tolerância 0,50); a versão vigente aceita release como prova de mínimo
   interior, e portanto **admite ε=1**. Mudança de instrumento na direção de
   incluir mais dados tem de ser declarada como tal — está. A justificativa
   física (mínimo interior é garantido pelo release) continua valendo, e o custo
   epistêmico foi pago duas vezes: (a) ε = 1 saiu do primário por outro motivo
   (§1.1); (b) a estacionariedade voltou como diagnóstico reportado, e ε = 1
   aparece explicitamente como **não-estacionário**.
7. **Este pré-registro foi emendado pelo 17º parecer** (`parecer-17.md`) em
   2026-07-27, **antes de qualquer execução** do confirmatório. As mudanças
   estão listadas item a item no parecer e refletidas aqui e no código. O
   config foi regerado (SHA novo, no cabeçalho) e o sintético foi rodado
   inteiro depois da emenda (§8).
8. O estimador within-instância do diagnóstico do resto (§1) foi escolhido
   **depois** de o sintético mostrar que o OLS empilhado não tinha resolução
   para o teste — e **antes** de qualquer dado real. É uma escolha de estimador
   feita contra um mundo simulado, não contra o dado. Declarada.
9. **`RESTO_TOL = 0,01` foi fixado SABENDO que o piloto dá `resto = −0,018`.**
   Não foi escolhido às cegas: foi escolhido logo abaixo do valor que já se
   tinha na mão. **Logo o resultado ESPERADO do diagnóstico é
   ESTIMAÇÃO-COM-ALERTA** — o teste está montado para acender a luz, e acendê-la
   não será surpresa nem descoberta. O que o teste de fato mede é (a) se o resto
   **persiste** nas 100 instâncias novas e (b) **de que tamanho e sinal** ele
   sai, com IC. Um resto que ficasse **dentro** de 0,01 é que seria informação
   nova. Mesma franqueza da lição de §A.5.2: uma tolerância calibrada no dado
   que ela vai julgar tem de ser declarada como tal — e é precisamente por isso
   que o resto **não mata, alerta**.

## 8. Smoke tests e validação sintética (só isto foi executado)

- `python preditor_hbarreira.py smoke` — 3 trajetórias, 1 processo, ~1–2 min.
  Verifica que o motor roda, que o critério de platô dispara e que o config é
  determinístico. **Não** é o experimento. Resultado da 1ª passada (ver
  disclosures 5 e 6), da 2ª (regra de platô corrigida) e da 3ª (pós-emenda, com
  o campo `estacionario`) estão em `results/`. A 3ª passada confirma
  `estac=False` em ε=1 (`bz_ret = 0,545`) e `estac=True` nos dois ε=1e−8.
- `python analise_hbarreira.py sintetico` — **TUDO PASSA** (pós-emenda), com os
  mundos novos:
  - mundo `a = 1` → SUSTENTA; mundo `a = 0,97` (a predição declarada) → SUSTENTA;
  - `a = 0,7`, `a = 1,4`, SEM LEI → MATA; ruído grande com n = 6 → INCONCLUSIVO;
  - **mundo com CURVATURA REAL `q = 0,005`** (efeito 1,71) →
    INCONCLUSIVO-CURVATURA — a guarda **pode** disparar;
  - **mundo do JOELHO** (curvatura só em ε=1, o mundo do desenho velho) →
    SUSTENTA, `|q|L²max = 0,057`; **contraprova impressa pelo próprio teste:**
    na grade VELHA o mesmo mundo daria `a = +1,133` e `|q|L²max = 6,96`
    (**DISPARARIA**). É a demonstração de que a grade nova resolve o artefato;
  - diagnóstico do resto nos dois sentidos: mundo sem resto → dentro da
    tolerância; mundo com `resto = −0,018` (o do piloto) → **ESTIMAÇÃO-COM-
    ALERTA** (`IC95 = [−0,026; −0,011]`); mundo com `+0,05` → ALERTA;
  - guardas de potência das secundárias (i) e (ii) → SEM POTÊNCIA quando o IC é
    largo (antes, a (i) diria INVARIANTE);
  - **confound de `lr` removido pela grade descontada:** duas células com
    vazamentos diferentes dão `Δa = −0,301` sobre `y` bruto (NÃO-INVARIANTE,
    falso positivo) e `Δa = +0,007` sobre `y*` (INVARIANTE, correto);
  - secundária (iii) nos dois sentidos e P-NULL de instrumento.

## 9. Desvios (preencher DEPOIS da execução)

_(vazio — nada foi executado além dos smoke tests e do sintético de §8)_

---

# Apêndice A — Derivação de `a = 1`

Escrito 2026-07-26, **antes** do 17º parecer e **antes** de qualquer execução do
confirmatório. Verificação reexecutável: `derivacao_a1.py`
(`algebra` = SymPy · `crossover` = re-execução instrumentada da instância de
referência do **piloto** · `ajuste` = reajuste do `results/e7_runs.jsonl` do
**piloto**). Nada aqui usa dado do confirmatório — ele não existe. A instância
usada é a de referência, que já está **excluída** da análise (§7.1).

**§A.5 foi reescrita pela emenda** (era a proposta do primário′, negada). O
resto do apêndice está como foi escrito.

## A.1 A conta

Notação da casa: `y := −β·z` (profundidade, `y > 0` no morto), `L := ln(1/ε)`.

**(1) O canal.** Com o neurônio A morto, `σ(β z_A) ≈ e^{β z_A}`, logo a
componente do gradiente que o move é

```
    |g| = C · e^{β z} = C · e^{−y},        C ∝ |R| · escala de X.    [ESTABELECIDO — é a expansão da sigmoide]
```

**(2) O passo do Adam.** `Δθ = −lr·m̂/(√v̂+ε)`. Duas medições no piloto
(reproduzidas em `derivacao_a1.py crossover`, coluna "mediana no platô"):
`m̂/g = 1,000` e `√v̂/|g| = 1,00` em **todos** os ε. Isto é, no platô o Adam **não
tem memória efetiva**: o gradiente do morto varia devagar demais (ρ ≈ 7e−5 por
passo) para que β₁ ou β₂ o defasem. O SymPy dá o fator de memória em forma
fechada e o mesmo limite:

```
    κ² := v̂/g² = (1−β₂)/(1−β₂·e^{2ρ}),      κ → 1 quando ρ → 0.
```

**(3) A EDO do sticking.** Com `dz = dw·x_face + db` e `c := |x_face|+1`,
`A := β·lr·c`:

```
    dy/dk = A · |g| / (κ|g| + ε) = A·C·e^{−y} / (κ·C·e^{−y} + ε)
```

separável, com solução **fechada** (SymPy, `y(0)=0` na morte):

```
    κ·y + (ε/C)·(e^{y} − 1) = A·k.                                   (★)
```

Os dois regimes estão ambos dentro de (★): `√v̂ ≫ ε` ⟹ `dy/dk ≈ A/κ` (descida
saturada, velocidade independente de quão pequeno é `g`); `√v̂ ≪ ε` ⟹
`dy/dk ≈ (A C/ε) e^{−y}` (congelamento exponencial).

**(4) O estacionamento.** Ao fim da fase morta (`k = K_morto` passos), o termo
exponencial de (★) domina (`A·K ≫ κ·y`, verificado: 6,2e3 contra 23), logo

```
    (ε/C)·e^{y_park} = A·K_morto  ⟹  y_park = ln(1/ε) + ln(C·β·lr·c·K_morto)
```

> ### **a = 1 EXATO**, e `b = ln C + ln(β · lr · c · K_morto)`.

**(5) O termo de correção**, derivado por diferenciação implícita de (★) em `L`
(SymPy fecha a conta):

```
    a = dy/dL = 1 − (κ + ε/C)/(A·k) + O((A·k)^{−2}).
```

Medido no piloto: `(κ+ε/C)/(A·K)` = 1,6e−4 (ε=1e−8) … 1,9e−3 (ε=1).
**Intrinsecamente, `a` = 1 com erro ≤ 0,002.** A discretização não aparece
sequer nesse nível: um passo desloca `y_park` em ρ ≈ 7e−5.

## A.2 A leitura alternativa (`a = 1/2`) e a invariância de limiar

- Se o portão fosse **`v̂ ~ ε`** (ler o denominador como `v̂+ε`), então
  `g² ~ ε` ⟹ `y = ½·ln(1/ε)`: **`a = 1/2`**. As duas leituras diferem por um
  **fator 2** na inclinação e são trivialmente distinguíveis. O piloto dá 0,94 —
  `a = 1/2` está **excluído** (não é uma disputa de casas decimais). Corolário
  registrado: `a = 1/2` é morta por qualquer `a > 0,75`.
- Se o limiar for `√v̂ = θ·ε` para **qualquer** constante `θ` (θ=1, θ=2 do
  `√v̂+ε = 2ε`, ou o `θ ≈ 2e−4` que de fato se mede), isso muda `b` por `ln θ`
  e **não muda `a`**. `a = 1` é **invariante à convenção de limiar** — é o que dá
  robustez à predição.

## A.3 O crossover MEDIDO (e uma correção honesta ao texto de §1)

`derivacao_a1.py crossover`, instância de referência, no **platô** (todo o trecho
com `y ≥ y_park − 1`):

| ε | y_park | `\|g\|/ε` | `√v̂/ε` | κ=`√v̂/\|g\|` | `\|u\|` |
|---|---|---|---|---|---|
| 1e−8 | 23,33 | 2,28e−4 | 2,28e−4 | 1,00 | 2,28e−4 |
| 1e−4 | 14,36 | 2,10e−4 | 2,10e−4 | 1,00 | 2,10e−4 |
| 1e−2 |  9,87 | 1,91e−4 | 1,91e−4 | 1,00 | 1,91e−4 |
| 1    |  6,11 | 5,15e−5 | 5,16e−5 | 1,00 | 5,15e−5 |

**Três leituras, uma delas corrigindo o pré-registro:**

1. **O crossover NÃO é `|g| ~ ε` nem `√v̂ ~ ε`.** É `|g| = θ·ε` com
   **θ ≈ 2e−4**, cinco ordens abaixo. A frase de §1 ("o release acontece quando
   o canal cruza o piso ε") está **quantitativamente errada como localização** e
   **certa como estrutura**: o que a lei `a=1` exige é apenas que
   `|g|_park ∝ ε`, e é isso que se mede (θ constante a ±9% entre ε=1e−8 e
   ε=1e−2, oito décadas). O `ln θ` mora em `b`. **Esta correção não altera
   nenhum critério de morte pré-registrado** — `a=1` e a regra [0,90; 1,10] são
   exatamente as mesmas —, mas o texto da história física precisa dizer
   `|g| ∝ ε`, não `|g| ≈ ε`. (§1 já foi corrigida nesse sentido.)
2. **θ é identificado.** A EDO prevê `θ = 1/(A·K_morto) = 1/(β·lr·c·K_morto)`.
   Medido/predito = **1,42 · 1,49 · 1,38 · 1,19** (ε = 1e−8 … 1). A razão é
   constante a ±12% **através do joelho de ε=1** — a *estrutura* está
   confirmada; o fator ~1,4 é o `c` (conhecido só a menos de O(1)).
3. **κ = 1,00 em todo ε ⟹ β₂ NÃO entra em `b`.** A justificativa de §1(ii)
   (defasagem de τ₂ passos, Δb ≈ 0,07) chega ao mesmo veredito de equivalência
   por um caminho diferente; a derivação é mais forte: κ entra em (★) apenas
   pelo termo `κ·y/(A·K)` ≈ 3e−3. **Predição derivada: `|Δb(β₂)| < 0,1`.**
   > **RESSALVA OBRIGATÓRIA (17º parecer, blocker 6):** o smoke test **já
   > contradiz esta predição** — deu `Δb = −0,40`, quatro vezes a tolerância
   > (≈2σ pareado, 1 instância). Portanto `|Δb| < 0,1` **NÃO é critério de
   > morte**: é **predição-sob-teste com evidência contrária prévia declarada**.
   > O critério decisório é e continua sendo a equivalência ±2 congelada.

## A.4 O que realmente produz o `0,94` do piloto — e o risco para o confirmatório

`b = ln C + ln(β·lr·c·K_morto)`. Logo, se `lr` ou `K_morto` **dependerem de ε**,
essa dependência **vaza para dentro da inclinação**:

```
    a_medido = 1 + d ln(lr)/dL + d ln(K_morto)/dL + d ln(C)/dL − (κ+ε/C)/(A·K)
```

E o desenho **recalibra `lr` por ε** (§2) — vazamento **por construção**.
Medido no piloto (`derivacao_a1.py ajuste`):

| ε | L | y (mediana) | lr | K_morto (mediana) | inclinação local |
|---|---|---|---|---|---|
| 1     |  0,00 |  6,448 | 1,067e−5 | 120 493 | — |
| 1e−2  |  4,61 | 10,056 | 1,300e−6 | 380 006 | **0,784** |
| 1e−4  |  9,21 | 14,567 | 1,206e−6 | 398 974 | 0,979 |
| 1e−8  | 18,42 | 23,577 | 1,205e−6 | 363 760 | 0,978 |

`lr` é plano para ε ≤ 1e−2 e **salta ×8,9 em ε=1** (lá a própria fase VIVA já
está no regime linear, e a calibração de `t_hit` tem de compensar); `K_morto`
cai pela metade. Consequência, tudo derivado e depois verificado:

| ajuste | `a` | `q` (curvatura) | `\|q\|·L²max` |
|---|---|---|---|
| y bruto, 4 ε **incluindo ε=1** | **0,941** | +0,0050 | **1,697 → a guarda DISPARARIA** |
| **y bruto, ε ≤ 1e−2 (a GRADE PRIMÁRIA nova)** | **0,973** | −0,0002 | **0,075 (não dispara)** |
| y* = y − ln(lr·K_morto), 4 ε | 0,9985 | — | — |
| y*, ε ≤ 1e−2 | 0,982 | — | — |

> **Descontado o vazamento de projeto, `a = 0,9985` sobre as quatro décadas
> inteiras, joelho incluído.** O joelho de ε=1 **não é curvatura da lei
> física**: é `lr(ε)` e `K_morto(ε)`.

**Este era o achado operacional decisivo** e foi o que motivou a emenda: o
desenho original produziria `a ≈ 0,94` **e** dispararia a própria guarda de
curvatura por artefato. **A emenda o resolve tirando ε=1 do ajuste primário**
(§1.1), não tirando o vazamento por baixo do pano: na grade primária nova o
vazamento residual é `d ln(lr·K)/dL = −0,009`, a predição vira `a ≈ 0,97`, e a
guarda não dispara (0,075). O que sobra do `y*` é o **diagnóstico do resto**
(§1) — que é onde a informação de fato mora.

## A.5 Orçamento de erro, e a janela que foi ABANDONADA

**[Esta seção foi reescrita pela emenda do 17º parecer. A proposta original —
um "PRIMÁRIO′" sobre `y*` com janela [0,95; 1,05] — foi NEGADA. Ver §1.1–1.2.]**

Orçamento de erro derivado, para a **grade primária** (sem ε=1):

| fonte | efeito em `a` | conta |
|---|---|---|
| termo intrínseco da EDO `(κ+ε/C)/(A·K)` | −0,002 | medido, A.1(5) |
| discretização (dt) | −1e−4 | um passo = ρ ≈ 7e−5 em `y` |
| camada de sticking / memória de v̂ (`ln κ`) | 0 | **κ = 1,00 medido** |
| deriva de `C(ε)` (|R| na parada) | ±0,02 | resíduo A.4 (0,982 vs 1) |
| **vazamento `d ln(lr·K)/dL`, grade primária** | −0,009 | medido |
| _(vazamento na grade COM ε=1, para comparação)_ | _−0,06_ | _0,9985 → 0,941_ |
| estatística (N=100, 4 ε, bootstrap de cluster) | ±0,0066 | sintético, §8 |

**Duas conclusões, e uma autocrítica.**

1. O centro derivado sob **a grade primária** é **≈ 0,97**, e a janela congelada
   [0,90; 1,10] o contém com folga dos dois lados. Ela é **larga** para o poder
   que o experimento tem (IC de ±0,0066) — e isso está declarado como
   assimetria a favor de sustentar (disclosure §7.3). A compensação é que o
   texto do resultado é **obrigado** a reportar o IC e a dizer se `a = 1` foi
   ou não excluído: "SUSTENTA" sozinho não é reporte aceitável.
2. **A janela [0,95; 1,05] foi ABANDONADA e a lição registrada.** Ela foi
   apresentada como "derivada", mas o orçamento que a produziu usava a mesma
   linha `y*, 4 ε → 0,9985` que ela iria julgar: um orçamento **calibrado no
   dado que ele deveria testar**. É auto-cumprimento, e o fato de a conta
   "fechar" não é evidência — é circularidade. A janela vigente é a original,
   escrita antes de qualquer orçamento.

## A.6 Etiquetas

- `[ESTABELECIDO]` — expansão `σ(βz) ≈ e^{βz}`; forma do passo do Adam.
- `[RESULTADO — derivado e verificado aqui]` — solução fechada (★), `a = 1`,
  `b = ln C + ln(β·lr·c·K_morto)`, `a = 1 − (κ+ε/C)/(A·k)`, κ→1, `θ = 1/(A·K)`.
  Verificação: SymPy (álgebra) + re-execução instrumentada (crossover, κ, θ) +
  reajuste do piloto (`a = 0,9985` descontado o vazamento).
- `[NÃO VERIFICADO]` — que `c = |x_face|+1` seja exatamente o coeficiente
  geométrico (a razão medida é 1,2–1,5, não 1,0); nada depende disso, pois `c`
  vive só em `b`.
- `[ESPECULAÇÃO]` — a física do **resto inexplicado** (§1). Se ele aparecer
  acima de 0,01 no confirmatório, é achado a investigar, não explicação pronta.
- **Fora de escopo:** tudo isto é sobre a instância de referência (1 instância) e
  as 13 instâncias EXPLORATÓRIAS do piloto. A derivação é geral; a **verificação
  numérica não é** — ela mostra que a conta descreve o piloto, não que descreve
  as 100 instâncias novas. Isso é o confirmatório.
