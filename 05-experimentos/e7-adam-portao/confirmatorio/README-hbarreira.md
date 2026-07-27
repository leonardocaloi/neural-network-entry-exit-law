# Confirmatório H-BARREIRA — pré-registro

Status: **[PRÉ-REGISTRO]** — escrito ANTES de qualquer execução do confirmatório.
Data: 2026-07-26. Herda do piloto **[EXPLORATÓRIO]** e7
(`../README-e7.md`, `06-notas/2026-07-26-e7-piloto-adam.md`).
Nada deste diretório foi rodado além dos *smoke tests* declarados em §8.

---

## 1. AS PREDIÇÕES, ANTES DE TUDO

**Convenção de sinal (fixada aqui, uma vez).** A variável ajustada é a
**profundidade** da barreira, `y := −β·z_park > 0`. O piloto reportou
`β·z_park ≈ −0,94·ln(1/ε) − 6,1`; nesta convenção isso é
`y ≈ +0,94·ln(1/ε) + 6,1`.

### PRINCIPAL — H-BARREIRA

Na regressão sobre todas as instâncias e todos os ε da célula-âncora
(β = 1e4, β₂ = 0,999):

```
    y(i, ε) = −β·z_park(i, ε) = a·ln(1/ε) + b + u_i + ruído
```

o mecanismo proposto — **o release acontece quando o canal `e^{βz}` cruza o piso
`ε`**, isto é, `|g| ∝ e^{β z_park} ≈ ε` ⟹ `β·z_park = ln ε + const` — prediz

> ### **a = 1 EXATO.** Sem parâmetro livre. `b` é livre (constante da proporcionalidade).

**MATA H-BARREIRA:** o IC95 de `a` (bootstrap de cluster sobre instâncias)
**excluir** o intervalo [0,9; 1,1], isto é, ser disjunto dele.

| resultado | veredito congelado |
|---|---|
| IC95(a) ⊆ [0,9; 1,1] | **SUSTENTA** |
| IC95(a) ∩ [0,9; 1,1] = ∅ | **MATA** |
| IC95(a) cruza a fronteira | **INCONCLUSIVO** (não é vitória) |
| curvatura significativa (§3) | **INCONCLUSIVO-CURVATURA** (sobrepõe as três acima) |

O piloto deu `a = 0,94` com 4 pontos e 13 instâncias exploratórias. Note que
`a = 0,94` **SUSTENTA** pela regra acima se o IC95 couber em [0,9; 1,1] — é um
teste de **equivalência**, não de "≠ 1". Com N = 100 e 5 ε o IC de `a` deve ter
meia-largura ~0,01 (verificado no sintético), então a regra é bem posta: nem
aceita qualquer coisa, nem exige igualdade numérica impossível.

### SECUNDÁRIA (i) — invariância em β

O piloto rodou **β = 1e4 apenas** (confirmado em `../README-e7.md` §Setup e em
`e7_adam.py: BETA = 1e4`). O mecanismo é uma condição sobre `β·z`, adimensional:
**`a` não pode depender de β**. Rodamos β ∈ {1e3, 1e4} com β₂ = 0,999.
β = 1e3 é **teste novo**, nunca visto por este projeto sob Adam.

> **Predição:** `a(β=1e3) = a(β=1e4)`. `b` pode mudar (é a constante da
> proporcionalidade `|g| = C·e^{βz}`, e `C` depende de β).

**MATA (i):** IC95 pareado de `Δa = a(1e3) − a(1e4)` excluir 0 **e** |Δa| ≥ 0,1.
Se as duas condições não valerem juntas → INVARIANTE.

### SECUNDÁRIA (ii) — varredura de β₂ (a memória de `v`)

O piloto atribuiu o colapso de `|m̂/(√v̂+ε)|` (3,6e−4, quatro ordens abaixo da
saturação) à memória longa de `v` (τ₂ = 1/(1−β₂) ≈ 10³ passos). Varremos
β₂ ∈ {0,9; 0,99; 0,999} em β = 1e4.

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

Logo a predição pré-registrada é **de equivalência**:

> **`a` invariante em β₂** (tolerância 0,1) **e** `|b(0,9) − b(0,999)| < 2`.

Um efeito GRANDE de β₂ em `b` refutaria o mecanismo do piso de ε (mostraria que
quem governa a profundidade é a memória de `v`, não o piso). Um efeito de β₂ em
`a` refutaria mais fortemente ainda.

**MATA (ii):** IC95 pareado de `Δb = b(0,9) − b(0,999)` sair de [−2, +2], ou
`Δa` além de 0,1 com IC excluindo 0. Se o IC95 de Δb for mais largo que ±1, o
teste é declarado **SEM POTÊNCIA** e não decide nada (registrado como tal).

### SECUNDÁRIA (iii) — o baseline GD como âncora `ε → ∞`

Formulação: quando `ε ≫ √v̂`, o update de Adam vira `lr·m̂/ε`, **linear em g** —
o piso deixa de existir como gate e o passo volta a ser (uma versão suavizada
por momento de) GD. Nesse sentido **GD é a célula ε → ∞**, e a lei
`y = a·ln(1/ε) + b` deve **saturar** nele, não extrapolar através dele:
`ln(1/ε) → −∞` daria `y → −∞`, o que é fisicamente impossível (a profundidade é
limitada pelo balanço de GD). O piloto viu exatamente isso: ε=1 → y = 6,4;
GD → y = 5,0.

> **Predição (direcional, não é o teste principal):** em **todo** ε da grade,
> a mediana de `y_Adam(ε)` é MAIOR que a mediana de `y_GD` (Adam estaciona mais
> fundo que GD em qualquer ε), e `y_GD` não depende de ε (controle de vazamento:
> a corrida GD ignora ε por construção — se depender, há bug no harness).

**MATA (iii):** algum ε com mediana `y_Adam < y_GD`. Isso não mata H-BARREIRA;
mata a leitura "o piso APROFUNDA o portão", que é a história física.

### P-NULL de instrumento (obrigatório)

Se menos de **60%** das células (instância × ε) da âncora entregarem `z_park`
válido (§2, critério de platô), nada é decidido: veredito **INSTRUMENTO-NULL**,
registrado como negativo de instrumento, sem reciclar hipótese.

---

## 2. Desenho

- **Instâncias: N = 100**, sorteadas do **mesmo prior** do sweep
  `05-experimentos/t2-sweep-instancias` (`sorteia()`, faixas idênticas) e
  filtradas por (H1)–(H4) pelo **mesmo código** (`sweep_instancias.uma`).
  Taxa histórica de aprovação: 57/600 = 9,5% ⟹ `N_SEEDS_MAX = 1600` seeds.
- **Bloco de seeds NOVO: 20270000+**, disjunto do bloco do piloto
  (20260726–20261325). A **primeira** instância aprovada é a de **CALIBRAÇÃO de
  lr** e fica FORA da análise; as 100 seguintes são o confirmatório.
- **Grade de ε: {1e−8, 1e−6, 1e−4, 1e−2, 1}** — 5 pontos (o piloto teve 4),
  espaçamento uniforme de 2 décadas em `ln(1/ε)` ∈ {0; 4,6; 9,2; 13,8; 18,4}.
- **Células** (um-fator-por-vez a partir da âncora, ver §5):
  `ancora` (β=1e4, β₂=0,999) · `beta-baixo` (1e3, 0,999) ·
  `b2-099` (1e4, 0,99) · `b2-090` (1e4, 0,9). Mais o **baseline GD** nos dois β.
- **lr recalibrado por (β, β₂, ε)** por bisseção em log(lr) para casar
  `t_hit^Adam = t_hit^GD` na instância de calibração — mesmo procedimento do
  piloto, e é o que impede que a transição em ε seja confundida com mudança de
  timescale. Nenhum ajuste posterior.
- **Horizonte H = 10,0** (piloto usou 8,0; fase morta mediana ~3,6 no ε pequeno).
  dt = 1e−5, β₁ = 0,9 fixo, FACE = 0 — todos herdados do piloto.
- **`z_park` desacoplado do release** (correção de viés introduzida aqui, não
  havia no piloto): `z_park` é válido se (a) houve release — então o mínimo é
  interior por construção, `z` desceu e voltou a 0 — ou (b) o run truncou no
  horizonte MAS o mínimo já se sustentou por 0,5 unidade de tempo dentro de
  0,5/β. Sem isso, as instâncias truncadas dariam `z_park` censurado (viés para
  fundo) e as células de ε grande, de fase morta curta, seriam excluídas em
  bloco — perdendo justamente a ponta direita da regressão (visto no smoke, §8).
- **IC por bootstrap de CLUSTER sobre instâncias** (reamostra a instância
  inteira com seus 5 ε), B = 10000, semente 20270001. Comparações entre células
  usam bootstrap **pareado** (as células compartilham as mesmas instâncias).

## 3. Análise (congelada em `analise_hbarreira.py`)

OLS empilhado de `y` sobre `ln(1/ε)`; `a` e `b` como acima. **Guarda de
curvatura pré-registrada:** ajusta-se também `y = q·L² + a·L + b`; se
`|q|·L²max > 1,0` (a curvatura desloca o ajuste em mais de 1 unidade de β·z ao
longo da grade) **e** o IC95 de `q` excluir 0, a lei linear é declarada
inadequada e o resultado primário vira **INCONCLUSIVO-CURVATURA**. Motivo de
existir: o piloto ajustou 4 pontos incluindo ε=1, onde o Adam já está
degenerando em GD; se a lei tem joelho, o `a` do ajuste linear é um artefato e
não se deve declarar vitória nem derrota.

Exclusão de células: `plato = False` ou `n_mortes ≠ 1`. A fração excluída é
reportada por ε (e alimenta o P-NULL de instrumento).

## 4. Custo (medido, não estimado no chute)

Constantes medidas nesta máquina em 2026-07-26, **1 núcleo, `nice -19`**, na
instância de referência, H = 8: Adam ε=1e−8 → **28,4 s**; Adam ε=1 → **9,0 s**;
GD → **5,0 s** (o run para assim que release e os dois preditores dispararam).
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
Estágios resumíveis (o ambiente reinicia processos de fundo); `run` faz *append*
em `results/hbarreira_runs.jsonl` — reexecutar uma célula duplica linhas, então
truncar antes de repetir.

## 7. DISCLOSURES

1. **As 12–13 instâncias do piloto e7 são EXPLORATÓRIAS e ficam FORA deste
   confirmatório.** Seeds novas (bloco 20270000+), disjuntas de 20260726–20261325.
   A instância de referência do paper (`X=[0.5,1.5]`, `Y=[0.5,1.0]`, …) também
   fica fora da análise: ela gerou a hipótese.
2. A hipótese testada aqui (**H-BARREIRA**) é **nova**: não é H-SIGN nem H-EPS
   do piloto. H-SIGN foi *refutada como mecanismo* e H-EPS foi confirmada
   exploratoriamente; nenhuma das duas é reciclada. Hipótese nova = experimento
   novo, com pré-registro novo — este arquivo.
3. `a = 1` é predição a priori do mecanismo, **não** o valor ajustado do piloto
   (0,94). A regra de decisão foi escrita como equivalência em [0,9; 1,1]
   **antes** de qualquer dado novo, e o piloto está dentro dessa janela: isto é
   assimétrico a favor de sustentar, e está declarado como tal. O que protege o
   teste é o IC estreito (meia-largura ~0,01 com N=100): um `a` de 0,94 com IC
   [0,935; 0,948] SUSTENTA a lei mas **não** é `a = 1` — o texto do resultado
   terá de dizer as duas coisas.
4. A calibração de lr usa uma instância que fica fora da análise, mas é a MESMA
   para as 100 — o lr não é ótimo por instância. Herança do piloto, declarada.
5. **Peek declarado (smoke, §8):** o smoke test rodou 1 instância (a de
   referência, EXCLUÍDA da análise) em 3 configurações. Ele mostrou
   `b(0,9) − b(0,999) = −0,40` em ε=1e−8 — sinal oposto ao do argumento de
   defasagem de §1(ii), com magnitude compatível com o ruído. **A predição de
   §1(ii) não foi ajustada a esse ponto**; foi escrita como teste de
   equivalência com tolerância ±2 derivada a priori (0,07), e o ponto do smoke
   cai dentro dela. Registrado aqui para que ninguém descubra depois.
6. A regra de platô (§2) foi ajustada **depois** do smoke e **antes** de
   qualquer dado do confirmatório: a primeira versão exigia sustentação por 0,5
   também nas corridas com release, e reprovava ε=1 (bz_retorno = 0,545 contra
   tolerância 0,5). O ajuste é uma correção de instrumento com justificativa
   física (mínimo interior é garantido pelo release), não uma escolha guiada por
   resultado.

## 8. Smoke tests já rodados (só isto foi executado)

`python preditor_hbarreira.py smoke` — 3 trajetórias, 1 processo, 2,4 min.
Verifica que o motor roda, que o critério de platô dispara e que o config é
determinístico. **Não** é o experimento. Resultado da 1ª passada (ver disclosure
5 e 6) e da 2ª (com a regra de platô corrigida) está em `results/`.

## 9. Desvios (preencher DEPOIS da execução)

_(vazio — nada foi executado além dos smoke tests de §8)_

---

# Apêndice A — Derivação de `a = 1`

Escrito 2026-07-26, **antes** do parecer e **antes** de qualquer execução do
confirmatório. Verificação reexecutável: `derivacao_a1.py`
(`algebra` = SymPy · `crossover` = re-execução instrumentada da instância de
referência do **piloto** · `ajuste` = reajuste do `results/e7_runs.jsonl` do
**piloto**). Nada aqui usa dado do confirmatório — ele não existe. A instância
usada é a de referência, que já está **excluída** da análise (§7.1).

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
  `a = 1/2` está **excluído** (não é uma disputa de casas decimais).
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
   nenhum critério de morte pré-registrado** — `a=1` e a regra [0,9;1,1] são
   exatamente as mesmas —, mas o texto da história física precisa dizer
   `|g| ∝ ε`, não `|g| ≈ ε`.
2. **θ é identificado.** A EDO prevê `θ = 1/(A·K_morto) = 1/(β·lr·c·K_morto)`.
   Medido/predito = **1,42 · 1,49 · 1,38 · 1,19** (ε = 1e−8 … 1). A razão é
   constante a ±12% **através do joelho de ε=1** — a *estrutura* está
   confirmada; o fator ~1,4 é o `c` (conhecido só a menos de O(1)).
3. **κ = 1,00 em todo ε ⟹ β₂ NÃO entra em `b`.** A justificativa de §1(ii)
   (defasagem de τ₂ passos, Δb ≈ 0,07) chega ao mesmo veredito de equivalência
   por um caminho diferente; a derivação é mais forte: κ entra em (★) apenas
   pelo termo `κ·y/(A·K)` ≈ 3e−3. **Predição derivada: `|Δb(β₂)| < 0,1`**, dez
   vezes mais apertada que a tolerância ±2 pré-registrada. (Mantém-se a regra
   frozen de ±2; esta é uma predição adicional, mais exigente, registrada aqui.)

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
| y bruto, 4 ε | **0,941** | +0,0050 | **1,697 → a guarda de §3 DISPARA** |
| y bruto, ε ≤ 1e−2 | 0,973 | −0,0002 | 0,075 (ok) |
| **y* = y − ln(lr·K_morto), 4 ε** | **0,9985** | — | — |
| y*, ε ≤ 1e−2 | 0,982 | — | — |

> **Descontado o vazamento de projeto, `a = 0,9985` sobre as quatro décadas
> inteiras, joelho incluído.** O joelho de ε=1 **não é curvatura da lei
> física**: é `lr(ε)` e `K_morto(ε)`.

**AVISO OPERACIONAL (é o achado que mais importa para o parecer).** O
confirmatório usa a mesma grade com ε=1 e a mesma recalibração de `lr`. A
derivação prediz, portanto, que ele deve produzir `a ≈ 0,93–0,95` **e** disparar
a própria guarda de curvatura de §3 (`|q|L²max ≈ 1,7 > 1,0`), caindo em
**INCONCLUSIVO-CURVATURA** — não por a lei ser falsa, mas por um artefato de
desenho que agora está derivado e é corrigível.

## A.5 A janela: [0,9; 1,1] é folgada, apertada ou certa?

Orçamento de erro **derivado**, para a grade do confirmatório:

| fonte | efeito em `a` | conta |
|---|---|---|
| termo intrínseco da EDO `(κ+ε/C)/(A·K)` | −0,002 | medido, A.1(5) |
| discretização (dt) | −1e−4 | um passo = ρ ≈ 7e−5 em `y` |
| camada de sticking / memória de v̂ (`ln κ`) | 0 | **κ = 1,00 medido** |
| deriva de `C(ε)` (|R| na parada) | ±0,02 | resíduo A.4 (0,982 vs 1) |
| **vazamento `d ln(lr·K)/dL`, grade sem ε=1** | −0,009 | medido |
| **vazamento `d ln(lr·K)/dL`, grade COM ε=1** | **−0,06** | 0,9985 → 0,941 |
| estatística (N=100, 5 ε, bootstrap de cluster) | ±0,01 | §1 |

**Veredito.**

- **Sobre `y` bruto na grade completa (o teste primário como está congelado):**
  a janela [0,9; 1,1] está **CERTA em largura mas descentrada e certa pelo
  motivo errado**. O centro derivado não é 1,00, é **0,94**; a janela só
  funciona porque é larga o bastante para absorver o vazamento. E ela é
  **assimétrica sem necessidade**: **todo** termo do orçamento é negativo — a
  derivação não admite `a > 1,00 + 0,01`. Um `a = 1,08` "sustentaria" a lei pela
  regra atual sendo, pela derivação, tão anômalo quanto `a = 0,7`.
- **Sobre `y* = y − ln(lr·K_morto)` (o teste com poder):** o orçamento total é
  ±0,03 e a janela **derivada** é **[0,95; 1,05]**.

**Janela proposta (emenda a adjudicar ANTES de rodar — não aplicada
unilateralmente):**

```
  PRIMÁRIO (mantido, frozen): a sobre y bruto, janela [0,90; 1,10].
     Reinterpretação registrada: o valor derivado sob ESTE desenho é 0,94,
     não 1,00. Sustentar aqui é fraco.
  PRIMÁRIO', pré-registrado agora: a sobre y* = y − ln(lr·K_morto),
     janela DERIVADA [0,95; 1,05], e a guarda de curvatura §3 aplicada a y*.
     MATA: IC95(a*) disjunto de [0,95; 1,05].
  Corolário: a leitura a = 1/2 é MORTA por qualquer a > 0,75 nas duas versões.
```

Justificativa de a emenda ser legítima: (i) é escrita antes de qualquer execução
do confirmatório; (ii) `lr` e `K_morto` são **variáveis de desenho registradas e
medidas**, não graus de liberdade de ajuste — `y*` não tem parâmetro livre novo;
(iii) a regra antiga **não é removida**, é mantida e reportada em paralelo.

## A.6 Etiquetas

- `[ESTABELECIDO]` — expansão `σ(βz) ≈ e^{βz}`; forma do passo do Adam.
- `[RESULTADO — derivado e verificado aqui]` — solução fechada (★), `a = 1`,
  `b = ln C + ln(β·lr·c·K_morto)`, `a = 1 − (κ+ε/C)/(A·k)`, κ→1, `θ = 1/(A·K)`.
  Verificação: SymPy (álgebra) + re-execução instrumentada (crossover, κ, θ) +
  reajuste do piloto (`a = 0,9985` descontado o vazamento).
- `[NÃO VERIFICADO]` — que `c = |x_face|+1` seja exatamente o coeficiente
  geométrico (a razão medida é 1,2–1,5, não 1,0); nada depende disso, pois `c`
  vive só em `b`.
- **Fora de escopo:** tudo isto é sobre a instância de referência (1 instância) e
  as 13 instâncias EXPLORATÓRIAS do piloto. A derivação é geral; a **verificação
  numérica não é** — ela mostra que a conta descreve o piloto, não que descreve
  as 100 instâncias novas. Isso é o confirmatório.
