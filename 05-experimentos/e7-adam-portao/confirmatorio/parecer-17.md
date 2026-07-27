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

## Fecho exigido (1ª rodada)

Ao final: regerar o config (2× byte-igual, SHA novo), rodar o sintético completo
(agora com os mundos novos — **TUDO deve passar nos dois sentidos**), atualizar
o README coerente de ponta a ponta (predições ANTES, disclosures atualizados), e
reportar SHA novo, saída do sintético e diff-resumo por item.
**NÃO rodar o experimento. NÃO commitar.**

---

# 2ª RODADA — VEREDITO: **APTO**

Recebido 2026-07-27, sobre o pacote já emendado pela 1ª rodada. Ainda **antes de
qualquer execução** do confirmatório.

> ## **APTO** — com 4 itens de registro (redação/etiqueta) a aplicar ANTES do
> commit de pré-registro. Nenhum item exige mudar desenho, critério, estimador
> ou custo.

## Verificações feitas nesta rodada

1. **Grade primária sem ε=1 e janela [0,90; 1,10] congelada** — conferido no
   `CFG` (`EPS_PRIMARIO`), no `decide()` e no README. Um primário, uma janela.
2. **Guarda de curvatura não dispara na grade nova** — reconferido: 0,075
   (recomputo do pacote) / 0,096 (recomputo do parecer). A contraprova impressa
   pelo próprio sintético (mundo do joelho: 0,057 na grade nova contra 6,96 na
   grade velha) é a demonstração pedida.
3. **Sintético 24/24 nos dois sentidos**, incluindo os mundos novos e as duas
   guardas de potência.
4. **Config byte-determinístico**, regerado 2× igual; SHA
   `49c3051e323271d3428d025ae03f44587a9e77e3e7afd43aaca37843cf673ec7`.
5. **Travas de processo** — subcomando obrigatório e anti-duplicação — testadas
   e efetivas.
6. **Dois achados do executor que corrigem números do próprio parecer**, ambos
   aceitos e considerados a favor do pacote:
   - a meia-largura real do IC95(a) **não** era 0,003–0,004 como o parecer
     estimou (ver adjudicação (a));
   - o teste do resto **não tinha resolução** com OLS empilhado sob o
     desbalanceamento criado pela exclusão dos truncados; a troca para o
     estimador **within (efeitos fixos)** é a correção certa, foi feita contra
     mundo simulado e antes de qualquer dado real, e está declarada.

## Adjudicações

**(a) Meia-largura do IC95(a) — os DOIS números.** O executor mediu 0,0066 no
sintético (`sd_ruido = 0,4`) e o parecer havia estimado 0,003–0,004. **Ambos
estão certos, em mundos diferentes:** o resíduo within medido no piloto é
**0,200**, não 0,4 — o sintético foi deliberadamente conservador, e a projeção
para o dado real é **≈ 0,004**. Adjudicado: **declarar os dois**, com a origem
de cada um. Motivo: os dois são 15–25× mais estreitos que a janela ±0,10 (a
conclusão não muda), e sem a declaração um IC real de ~0,004 seria lido depois
como "desvio" quando é exatamente o previsto. → **item de registro 1**.

**(b1) Colisão de notação `a`.** O diagnóstico chamava sua inclinação de `a`
(chave `a_bruto_no_subconjunto`), mesma letra do primário. São **estimadores
diferentes em amostras diferentes** — FE-within sem truncados contra OLS
empilhado com todos os runs de platô — e **diferem por construção**. Risco real
de alguém, meses depois, ler a diferença como resultado. Adjudicado: renomear
para **`a_diag`** (ou `a_within`) no README §1 e **na saída** do código, dizendo
onde o diagnóstico é definido que a diferença é por construção. Manter o rótulo
"por construção" do `soma_confere`. → **item de registro 2**.

**(b2) `|Δb(β₂)| < 0,1` — etiqueta em vez de re-pareamento.** O pacote já a
rebaixou a predição-sob-teste com o smoke (−0,40) declarado como evidência
contrária prévia. Falta o essencial: a meia-largura do IC95 de `Δb` é **~0,4**,
**quatro vezes** a tolerância — **este desenho não decide 0,1 nem em princípio**,
com ou sem dado. Considerou-se exigir re-pareamento para ganhar potência;
**negado**: custaria um desenho novo para responder algo que ninguém
pré-registrou como decisório. Adjudicado: **a etiqueta é preferível ao
re-pareamento** — rotular como *"predição-sob-teste, SEM POTÊNCIA POR DESENHO
(meia-largura ~0,4 contra tolerância 0,1)"*. **O ±2 segue decisório e nada mais
muda.** → **item de registro 3**.

**(c) `RESTO_TOL = 0,01` e o resto de −0,018 do piloto.** A tolerância foi
fixada sabendo o valor que ela iria julgar. Isso **não invalida** o teste — o
resto é diagnóstico e não mata —, mas é a **mesma estrutura** da circularidade
que o próprio pacote denunciou e registrou em §A.5.2 sobre a janela
[0,95; 1,05]. Exigir a mesma franqueza nos dois lugares é questão de coerência,
não de rigor extra. Adjudicado: declarar que o **resultado esperado é
ESTIMAÇÃO-COM-ALERTA**, e que o que se aprende é a persistência, o tamanho e o
sinal do resto nas 100 instâncias novas. → **item de registro 4**.

## Os 4 itens de registro

1. **README §7.3:** declarar os **dois** números da meia-largura do IC95(a) —
   "0,0066 é o valor do mundo sintético com `sd_ruido = 0,4`; o resíduo within
   medido no piloto é 0,200, o que projeta ≈0,004 para o dado real; ambos
   15–25× mais estreitos que a janela ±0,10" — para que um IC real de ~0,004 não
   vire falso "desvio" depois.
2. **Colisão de notação:** renomear a inclinação do diagnóstico para **`a_diag`**
   (ou `a_within`) no README §1 e **na saída** de `analise_hbarreira.py` — é
   estimador FE em subconjunto sem truncados, difere do `a` do primário **por
   construção**, e isso tem de estar dito onde o diagnóstico é definido. Manter
   o rótulo "por construção" do `soma_confere`.
3. **Secundárias (i)/(ii):** adicionar a etiqueta *"predição-sob-teste, SEM
   POTÊNCIA POR DESENHO (meia-largura ~0,4 contra tolerância 0,1)"* ao
   `|Δb| < 0,1`. O ±2 segue decisório e nada mais muda.
4. **README §7:** declarar que `RESTO_TOL = 0,01` foi fixado **sabendo** que o
   piloto dá −0,018 — logo o resultado esperado é **ESTIMAÇÃO-COM-ALERTA** —,
   com a mesma franqueza da lição de §A.5.2.

## Fecho exigido (2ª rodada)

Aplicar os 4 itens; regerar o config **se algo do `CFG` mudou** (a expectativa é
que não — são etiquetas e redação; se o SHA mudar, reportar o novo); rodar o
sintético 1× para confirmar **24/24**; atualizar este arquivo com o veredito
APTO. **NÃO commitar** — o pré-registro é commitado na sequência, pelo
coordenador.

