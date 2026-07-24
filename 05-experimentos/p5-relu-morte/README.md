# p5-relu-morte — morte de neurônio ReLU como contato unilateral genuíno

**STATUS: RASCUNHO REPROVADO NA REVISÃO ADVERSARIAL DE 2026-07-23 — NÃO COMMITAR
COMO PRÉ-REGISTRO DE TESTE. NÃO RODADO (só smoke de mecânica).**
Ver `revisao-adversarial-2026-07-23.md`: as predições P-a/b/c abaixo são decidíveis
a priori (mesma patologia do p5-cruz, em nível mais sutil) e não discriminam
sweeping/complementaridade de "loss com kink". O documento permanece como registro do
desenho e dos lemas verificados; os caminhos para um teste válido estão no parecer
(rebaixar a demonstração, ou redesenhar com ≥2 neurônios / fronteira móvel C(t)).

## Contexto e relação com P5

O experimento p5-cruz (auditado em 2026-07-23) mostrou que em loss SUAVE a leitura de
contato não tem conteúdo: `r_N` era escravo algébrico do gap (`r_N·d = 2L`), sem
complementaridade possível. **Hipótese nova = experimento novo** (regra anti-viés): este
experimento recoloca P5 no habitat onde a não-suavidade é genuína — a fronteira de
ativação do ReLU. O p5-cruz permanece registrado como está (patch test SLT).

## Objeto

Modelo: `f(x) = relu(w·x + b)`, parâmetros `p=(w,b) ∈ R²`.
Dados fixos: `X = (0.5, 1.5)`, `Y = (0.5, 1.0)`, `n=2`.
Loss: `L(w,b) = (1/2n) Σ_i (relu(z_i) − y_i)²`, com `z_i = w·x_i + b`.
Dinâmica: SGLD full-batch `p ← p − η∇L + √(2ηT)ξ`, `ξ~N(0,I)`, convenção `relu'(0)=0`.

**Região de morte** `D = {p : z_i ≤ 0 ∀i}` — cone poliedral com ápice na origem,
ângulo interno `π − (atan x_max − atan x_min) = 150.26°`. Dentro de `D` o neurônio está
morto: `f ≡ 0` nos dados.

## Lemas (valem POR CONSTRUÇÃO — não são predições; verificados em SymPy + numérico)

Verificação executável: `scratchpad → verificacao-lemas-relu-morte.py` (a mover para
`ferramentas/` no commit). Etiqueta: **[TEOREMA]** (enunciados elementares, prova de
1 linha cada, checagem simbólica e por diferenças finitas commitada junto).

- **L1 (piso plano):** em `int(D)`, `∇L = 0` exatamente e `L = (1/2n)Σy_i²`. Sob SGLD,
  a dinâmica em `int(D)` é passeio browniano puro (deriva exatamente nula).
- **L2 (salto de força):** cruzando a face `{z_j=0}` (demais `z_i<0`), `∇L` salta de `0`
  para `−(y_j/n)(x_j, 1)`: descontinuidade de magnitude `Δ_j = (y_j/n)·√(x_j²+1)`.
  Aqui: `Δ_0 = 0.2795` (face x=0.5), `Δ_1 = 0.9014` (face x=1.5). A loss é C⁰ mas
  não C¹ — o kink vem do termo cruzado `−2y·v·relu(z)`, linear em relu.
- **L3 (complementaridade estrita):** com gap `s_j = max(−z_j, 0)` e reação
  `r_j = ‖contribuição do dado j a −∇L‖`: `s_j > 0 ⟹ r_j = 0` e `r_j > 0 ⟹ s_j = 0`
  identicamente, com `r_j → Δ_j > 0` no contato. `0 ≤ s_j ⊥ r_j ≥ 0` vale com reação
  O(1) na fronteira — a estrutura que o p5-cruz não tinha.

**Lição da auditoria do p5-cruz aplicada:** os lemas acima NÃO são evidência de nada —
valem por construção. O conteúdo empírico está exclusivamente em P-a/b/c abaixo, que a
dinâmica pode refutar.

## Hipótese ([CONJECTURA])

A trajetória de treino deste sistema é a de um sistema de complementaridade dinâmica:
morte = abertura de folga (força nula, voo livre browniano), ressurreição = fechamento
de contato com a "parede de dados" (evento não-suave com salto de força), e a fronteira
de `D` funciona como catraca unilateral (sai-se de `D` e a força expulsa; não há
re-entrada além do que o ruído sozinho explicaria descontado o empuxo).

**Competidores declarados:** (i) descrição suave — a mesma fenomenologia sem salto,
representada pelo braço rival softplus_β em escala pareada; (ii) ruído puro — braço
nulo `L=0`; (iii) artefato de estimador — calibração do estimador de deriva contra
processo com deriva conhecida ANTES da run (ver `--smoke`).

## Predições falsificáveis (congeladas antes de rodar; braço primário T = 1e-3)

- **P-a (salto de força no evento).** Na ressurreição (primeira saída de `D`, face j
  identificada geometricamente), o estimador de deriva em janelas de K=200 passos
  (antes: dentro de D; depois: só passos com active set exatamente {j}) mede salto
  `‖v̂_dep − v̂_ant‖` cujo IQM (20 seeds) fica a **≤25%** do alvo protocolar
  `salto*_j = √((Δ_j·κ_j)² + 4σ_c²)` — NÃO do `Δ_j` idealizado. Derivação a priori
  (verificada por conta em `config.py`): (i) na janela pós a coordenada normal segue
  ODE linear discreta com taxa `ρ_j = η(x_j²+1)/n`, logo a força média na janela é
  `Δ_j·κ_j`, `κ_j = (1−(1−ρ_j)^K)/(K·ρ_j)`; (ii) a norma do vetor com ruído infla:
  `E‖v+ε‖² = ‖v‖² + 4σ_c²`, `σ_c = √(2T/(ηK))` (confirmado na calibração do smoke).
  No rival softplus (β=2, mesma η, T, init, seeds), o mesmo estimador dá salto
  **≥3× menor** (`salto_relu/salto_softplus2 ≥ 3`).
  **MATA:** IQM fora da tolerância em torno de `salto*`, OU razão < 3 (salto
  indistinguível do suave).
- **P-b (catraca unilateral).** Após a primeira ressurreição, a fração de tempo em `D`
  no horizonte de 100k passos satisfaz `f_SGLD < f_nulo/5` (IQM sobre seeds), onde
  `f_nulo` vem do braço `L=0` (mesmos seeds, mesma init) após a SUA primeira saída
  de `D` — passeio browniano 2D re-entra recorrentemente; a catraca não.
  **MATA:** `f_SGLD ≥ f_nulo/5`.
- **P-c (limite de rigidez — a ponte MEF correta).** No rival softplus_β
  (β ∈ {2, 8, 32}), o tempo mediano de ressurreição `t_med(β)` é **crescente em β** e
  converge ao do ReLU: `|t_med(32) − t_med(relu)|/t_med(relu) ≤ 25%`. (Softplus tem
  deriva > 0 dentro de `D` que decai como σ(βz); penalidade→restrição com κ=β→∞ —
  o limite que no p5-cruz tinha o sinal invertido.)
  **MATA:** não-monotonicidade além do CI, ou não-convergência.

**P-b(i) — validação de instrumento, NÃO predição:** dentro de `D` a deriva estimada
por seed deve satisfazer `‖deriva_morta‖ ≤ 3·√(2T/(η·n_mortos))` (verdade exata = 0
por L1; uma norma nunca "inclui 0", então o critério é o teto de 3 erros-padrão).
Se falhar em >2 dos 20 seeds, o experimento é INVÁLIDO (falha de instrumento ≠
falsificação).

## O que este experimento NÃO testa

Não testa que redes ReLU em geral são sweeping processes; não testa escala; não testa
a identificação fronteira-de-ativação ↔ locus SLT de queda de posto de J (isso é o
passo seguinte do programa, com J = Jacobiano da parametrização). Testa a instância
mínima em que a leitura de contato tem conteúdo quantitativo refutável.

## Parâmetros (congelados em `config.py`)

η=1e-3; T primária 1e-3 (braços 3e-3 e 1e-2 exploratórios, rotulados); N_PASSOS=500k;
20 seeds; init determinística p0=(0, −0.5) ∈ int(D) para todos os braços;
K_EVENTO=200; HORIZONTE_POS=100k; β ∈ {2,8,32}; censura: run sem ressurreição no
horizonte conta como censurada; >30% de censura no braço primário ⟹ braço inconclusivo.
Análise: IQM + bootstrap percentil (10k reamostras) por braço; eventos com menos de
K/2 passos válidos na janela pós são descartados e contados.

## Baseline nulo

`L=0`, mesmos seeds, mesma init: idêntico ao SGLD dentro de `D` até a primeira saída
(consistência), e referência de re-entrada para P-b.

## Execução

```
python experimento.py --smoke   # só mecânica: gradiente vs dif. finitas, determinismo,
                                # calibração do estimador de deriva em OU conhecido.
                                # NÃO produz estatística de P-a/b/c.
python experimento.py           # run confirmatória (só após commit deste pré-registro)
```

## Desvios do pré-registro

(vazio — preencher se algo mudar após o commit, com data e justificativa)

## Prior art a verificar antes de qualquer nota pública ([NÃO VERIFICADO])

Aydinoglu & Posa (arXiv:2011.07626) — ReLU↔LCP no mapa forward; Lu et al.
(arXiv:1903.06733) — dying ReLU/dead set; arXiv:2305.12467 — dinâmica multi-fase de
ReLU (sandbox vizinho); arXiv:2305.15912 — characteristic activation boundaries;
Bolte & Pauwels (arXiv:1909.10300) — formalismo de campos conservativos. A reivindicação
desta nota, se P-a/b/c sobreviverem: apenas a peça DINÂMICA/preditiva (eventos de treino
como switches de active set com reação mensurável e limite de rigidez verificado).
