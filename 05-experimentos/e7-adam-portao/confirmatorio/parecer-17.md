# 17º parecer — pacote confirmatório H-BARREIRA

Recebido 2026-07-26, ANTES de qualquer execução do confirmatório. Transcrição
das ordens conforme repassadas ao executor. Este arquivo é o registro de
auditoria: o README-hbarreira.md é o pré-registro, este é o parecer que o
emendou.

Objeto: `README-hbarreira.md`, `preditor_hbarreira.py`, `analise_hbarreira.py`,
`derivacao_a1.py`.

---

## Adjudicação da emenda do Apêndice A.5 — opção (c) RESTRITA

O pacote submetido propunha (§A.5) manter o primário sobre `y` bruto na grade
completa e ACRESCENTAR um "primário′" sobre `y* = y − ln(lr·K_morto)` com janela
derivada [0,95; 1,05]. **Negado como desenhado** (dupla chance: dois primários,
duas janelas, duas oportunidades de sustentar). Adjudicado no lugar:

### 1. PRIMÁRIO ÚNICO

- `a` sobre `y` **bruto**, **grade primária ε ∈ {1e−8, 1e−6, 1e−4, 1e−2}**
  (ε=1 FORA do ajuste primário), janela **[0,90; 1,10] mantida congelada**.
- **Predição derivada declarada no README:** `a ≈ 0,97`, e a **guarda de
  curvatura NÃO dispara** — verificado pelo parecer: sem o joelho de ε=1,
  `|q|·L²max = 0,096`, bem abaixo de CURV_TOL = 1,0.
- **Remover o primário′ sobre `y*` como confirmação.** `y*` reaparece só como
  diagnóstico (item 3).

### 2. ε=1 e GD como PONTOS DE SATURAÇÃO

`ε = 1` e o baseline GD continuam sendo **rodados e reportados**, mas
exclusivamente como **pontos de saturação da secundária (iii)**, **nunca** no
ajuste primário. O custo do experimento fica inalterado.

### 3. `y*` vira DIAGNÓSTICO de decomposição — com um teste novo

Reportar a decomposição, cada termo medido:

```
    a = 1 + d ln(lr)/dL + d ln(K_morto)/dL + resto
```

e o **TESTE NOVO, com resolução**: o **RESTO INEXPLICADO**, com tolerância
derivada ANTES. O piloto dá `resto = −0,018` com IC ±0,003 (≈6σ).

> **Pré-registrado:** se `|resto|` do confirmatório > 0,01 **com IC excluindo
> 0**, isso é **achado real a investigar** — rotular como
> **ESTIMAÇÃO-COM-ALERTA**, **não MATA** (a física do resto não está derivada).

Implementação obrigatória em código: `carrega()` extrai `lr` e `K_morto`;
`K_morto` **definido operacionalmente no README** como `t_release − t_hit` (em
passos); **runs truncados EXCLUÍDOS do diagnóstico**, com a contagem reportada —
**nunca** excluídos do primário.

### 4. Lição sobre a janela [0,95; 1,05]

Registrar no README que **[0,95; 1,05] NÃO era independente do piloto**: o
orçamento de erro que a produziu foi calibrado no mesmo dado que produziu o
0,9985 — auto-cumprimento. Lição registrada, **janela abandonada**.

---

## BLOCKERS EM CÓDIGO (1–6)

**5.** Tudo que o README decide **TEM que existir** em `analise_hbarreira.py`
congelada: o primário novo (grade sem ε=1), o diagnóstico de decomposição, o
resto com tolerância, e a tabela de vereditos **SEM dupla chance** (o primário
único já resolve o blocker 3).

**6.** Secundárias (i)/(ii):

- **(i)** adicionar **guarda de potência**: IC mais largo que X ⟹
  **SEM-POTÊNCIA**, e não INVARIANTE. (Hoje um IC largo declara invariância por
  omissão — é o mesmo defeito que a (ii) já tratava.)
- Registrar que `Δa`/`Δb` entre células de β/β₂ **misturam recalibração de lr**.
  A comparação pré-registrada passa a ser **sobre a grade descontada OU com lr
  comum por par** — escolher a implementável e **documentar a escolha**.
- **blocker 6:** a predição `|Δb(β₂)| < 0,1` do Apêndice A.3.3 — **REGISTRAR que
  o smoke deu 0,40** (≈2σ pareado, **contradiz**) e **reformular**: manter o
  teste de equivalência **±2 congelado como decisório**; `|Δb| < 0,1` vira
  **predição-sob-teste**, com o smoke declarado como **evidência contrária
  prévia**.

## FORTES (7–12)

**7.** Remover a guarda vazia `n_mortes ≠ 1` (ou consertá-la para poder
disparar). Como o motor está, `n_mortes` nunca passa de 1 — a guarda é
decorativa e dá falsa sensação de filtro.

**8.** Declarar a **DIREÇÃO** da mudança da regra de platô: ela **AFROUXOU**
(passou a admitir ε=1). Mudança de instrumento numa direção que favorece incluir
dados tem de estar escrita como tal.

**9.** Para runs **com release**, adicionar **verificação de platô** (janela de
estabilidade de `z` antes do release). Se ε=1 falhar, ele é **reportado como
não-estacionário** — o que é coerente com o papel de âncora que ele passou a ter
(item 2), não um problema.

**10.** (coberto no item 6.)

**11.** `analise_hbarreira.py`: **exigir subcomando explícito**, default = ERRO.
Hoje o default é `sintetico`, mas um typo (`analise_hbarreira.py rael`) cai no
ramo `real` e vira um **peek** não declarado.

**12.** Atualizar a §1 com a **meia-largura real do IC (~0,003–0,004**, não os
0,01 alegados) e com a **predição principal declarada**: `a ≈ 0,97`, cuja
leitura esperada é **"SUSTENTA mas não a = 1 exato"**.

## COSMÉTICOS (13–17)

**13.** `CURV_TOL` para dentro do `CFG` (e portanto dentro do sha256).

**14.** Declarar quais das entradas da tabela de custo são **interpoladas** (3
delas) e quais foram medidas.

**15.** Adicionar o mundo sintético com `q ≈ 0,005` / efeito 1,7 — deve dar
**INCONCLUSIVO-CURVATURA** (é o mundo do desenho velho); e, **com a grade nova
sem ε=1**, verificar que a guarda **NÃO** dispara quando a curvatura mora
apenas no joelho.

**16.** Remover o **"controle de vazamento y_GD"** — é vácuo: o harness de GD
não recebe ε por construção, então "y_GD não depende de ε" não pode falhar.

**17.** **Trava anti-duplicação** no `estagio_run` (e no `estagio_gd`): abortar
se `results/hbarreira_runs.jsonl` já tem linhas da mesma chave. O README hoje
pede "truncar antes de repetir", o que é um pedido de disciplina onde cabe uma
trava.

---

## Fecho exigido

Ao final: regerar o config (2× byte-igual, SHA novo), rodar o sintético completo
(agora com os mundos novos — **TUDO deve passar nos dois sentidos**), atualizar
o README coerente de ponta a ponta (predições ANTES, disclosures atualizados), e
reportar SHA novo, saída do sintético e diff-resumo por item.
**NÃO rodar o experimento. NÃO commitar.**
