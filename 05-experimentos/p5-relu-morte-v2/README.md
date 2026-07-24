# p5-relu-morte-v2 — parede que amolece: corrida entre ressurreição e morte permanente

**STATUS: RASCUNHO DE DESENHO — AGUARDANDO REVISÃO ADVERSARIAL. NÃO COMMITAR COMO
PRÉ-REGISTRO, NÃO RODAR.** Regra nova do protocolo (dois precedentes: p5-cruz,
relu-morte v1): o desenho só vira pré-registro depois de um revisor adversarial
independente confirmar que as condições-que-MATAM são atingíveis por um mundo possível.

## Por que v2

No ciclo 1 (`../p5-relu-morte/`), com 1 neurônio o conjunto de restrição é fixo e o
sistema é exatamente solucionável — todas as predições eram decidíveis a priori
(reprovado; ver `../p5-relu-morte/revisao-adversarial-2026-07-23.md`). Com 2
neurônios, o piso do neurônio morto continua exatamente plano (L1 sobrevive), mas a
reação na fronteira decai com o tempo, porque o outro neurônio consome o resíduo:
`Δ_j(t) ∝ |resíduo_j(t)|`. A "parede" amolece enquanto o morto vaga. Isso instala uma
corrida genuinamente aberta e uma estrutura de complementaridade com conjunto efetivo
dependente do estado — vizinha do sweeping process state-dependent
(Brogliato–Tanwani §2; a conexão formal é [ANALOGIA] até segunda ordem).

## Objeto

Modelo: `f(x) = relu(w_A x + b_A) + relu(w_B x + b_B)`, saídas fixas em +1,
`p = (w_A, b_A, w_B, b_B) ∈ R⁴`. Dados: `X = (0.5, 1.5)`, `Y = (0.5, 1.0)`.
Loss quadrática; SGLD com **taxas separadas** `η_A = 1e-3` (coords de A) e
`η_B = 1e-4` (coords de B), ruído por coordenada `√(2η_i T)` — Langevin
pré-condicionado diagonal, que preserva a MESMA distribuição de Gibbs `e^{−L/T}`.
Justificativa de desenho (2026-07-24, pré-commit): com `η_B = η_A` a janela de
captura fecha em ~600–900 passos (o resíduo da face 1 decai dominado pelo modo
RÁPIDO de B, 469 passos — não pelo lento, como estimado de início) e a corrida fica
apertada demais para a grade alcançar p≈0.5/0.75; `η_B = η_A/10` alarga a janela ×10
e torna a grade informativa. Fisicamente: camadas treinando em ritmos distintos.

Fatos estruturais (a verificar em `verificacao-lemas-v2.py` antes do commit,
mesmo padrão do ciclo 1 — SymPy + diferenças finitas):

- **L1-v2:** enquanto A está morto (`z_A,i ≤ 0 ∀i`), `∇_{p_A} L = 0` exatamente;
  a marginal de `p_A` é passeio browniano puro no MESMO cone `D` do ciclo 1 (150.26°),
  e a marginal de `p_B` é exatamente o SGLD do problema "B sozinho" (acoplamento zero
  enquanto A morto).
- **L2-v2:** `B` sozinho tem ajuste exato `(w_B*, b_B*) = (0.5, 0.25)` com ambos os
  dados ativos; na região ativa-ambos a dinâmica de B é OU 2D com taxas
  `η·eig(G)`, `G = média[(x_i,1)(x_i,1)ᵀ]` (autovalores ≈ 2.13 e 0.12 — timescale
  lento ≈ `1/(η·0.12) ≈ 8.5k` passos).
- **L3-v2:** o salto de força na saída de A pela face j no instante τ é
  `Δ_j(τ) = |r_j(τ)|/n · √(x_j²+1)`, onde `r_j = f(x_j) − y_j` é o resíduo corrente
  — decai para 0 conforme B ajusta. (Consequência da regra da cadeia; decidível.)

## Estrutura de sinal do resíduo (achado de 2026-07-24, verificado)

O caminho GD de B a partir do init declarado `(0.2, 0.05)` tem **um** cruzamento de
sinal no componente rápido (`res_0` cruza zero; com `η_B=1e-4`, em t≈14.4k, overshoot
máx ≈0.014) e nenhum no lento (`res_1 < 0` decaindo, timescale ≈85k passos com
`η_B=1e-4`). Consequência física:
pós-cruzamento, a face 0 empurra A para DENTRO (anti-captura) e a face 1 para FORA
(pró-captura). **Capturas acontecem essencialmente pela face 1** — assimetria por
face que a dinâmica 4D real pode violar (A ativo muda os sinais do resíduo): entra
como verificação estrutural junto de P-1.

## O modelo reduzido pré-registrado (onde mora o risco)

**Modelo de corrida (reduced-order, aproximação declarada; implementado em
`preditor.py`):** enquanto A está morto, (i) `p_A` faz BM puro no cone D — tempo/face
de primeira saída por MC de BM (instrumento, exato por L1-v2; MC canônico reescalado
por difusão, com self-test contra simulação direta); (ii) o resíduo segue o caminho
DETERMINÍSTICO de B-sozinho. Atenção de honestidade (auditoria de premissas,
2026-07-24): o que é exato por L1-v2 é a marginal de B ser o *SGLD* de B-sozinho; o
cronograma determinístico é APROXIMAÇÃO — as flutuações de Gibbs de B têm sd de
resíduo ~`|proj_lento|·√(T/λ_min)` (≈0.04 em T=1e-3). Correção de narrativa pelo 2º
parecer (M3, verificado por simulação): a flutuação é um smear que PRESERVA A MÉDIA
da curva q — viés em p(captura) desprezível (−0.002 médio) ⟹ ela quase não afeta
P-1; seu sinal vive em P-2 (alargamento de q ~50% da largura de transição). O risco
real de P-1 vem de back-reaction de A e migração de face — sinais genuinamente
incertos (é isso que faz P-1 não-decidível). Na saída em τ1 pela face j, a
excursão é modelada em 1D na coordenada normal: OU discreta rumo ao equilíbrio móvel
`x*(t) = max(−res_j(t), 0)/‖(x_j,1)‖`, taxa `ρ_j = η(x_j²+1)/n`, ruído `√(2ηT)`.
Constantes congeladas a priori: **captura** quando `x ≥ x*(t)/2` E `x*(t) ≥ 4σ_x`
(`σ_x = √(nT)/‖(x_j,1)‖`, sd estacionária da OU); **re-morte** = retorno a `x<0`
após exceder `2σ_x`, contada numa janela fixa de `W_P3 = 50k` passos pós-saída;
horizonte local da excursão 70k passos (a janela de captura fecha quando
`x* < 4σ_x`, ≤16k passos em todas as T da grade — truncamento consistente com o
próprio critério). O que o modelo reduzido DESPREZA, e por isso pode errar: a
não-linearidade acoplada pós-saída (A ativo muda o resíduo que o empurra), as
flutuações estocásticas de B em torno do caminho determinístico, excursões 2D com
migração entre faces, e o regime de *grazing* (empuxo marginal → chatter).

**Lema de pooling (do modelo reduzido, verificado por construção):** dado o evento de
saída `(τ1, face j)`, a excursão reduzida NÃO depende de `d0` — a profundidade só
desloca a distribuição de τ1. Logo é válido POOLAR os eventos das 3 células de mesmo
T para P-2/P-3, e a curva `q(τ1) = P(captura | saída em τ1, face 1)` é o objeto
congelado por T (tabela de bins). Se na dinâmica real `q` depender de `d0` além da
amostragem, isso é violação do modelo reduzido e aparece em P-2.

## Predições falsificáveis (podem falhar; congelar valores numéricos no commit)

- **P-1 (lei de corrida, primária).** Grade de condições 3×3 (profundidade inicial de
  A × temperatura), com células ESCOLHIDAS PELO PREDITOR A PRIORI para que a captura
  prevista caia em {~0.25, ~0.5, ~0.75} — o regime marginal onde o modelo reduzido
  tem erro máximo é obrigatório na grade. `N=150` seeds/célula (subido de 50 pelo
  2º parecer, para alimentar a faixa de transição de P-2; SE≈0.04 em p=0.5). Régua idêntica ao preditor: `p̂ = capturas/N`
  com run censurada contando como NÃO capturada. Censura é predição EXATA (L1-v2:
  dentro de D a dinâmica é BM): célula com censura observada > 30% é INVÁLIDA por
  instrumento (fora da contagem); <7 células válidas ⟹ P-1 inconclusivo.
  **MATA:** ≥3 células válidas com `|z| > 1.96` (binomial contra p_prevista), OU
  ≥8 desvios de mesmo sinal (teste de sinais).
- **P-2 (calibração da curva de captura + falsificador exato — o teste do grazing;
  REDESENHADA após o 2º parecer adversarial de 2026-07-24).** Eventos face-1 de
  TODAS as células e temperaturas pooled (lema de pooling; a normalização entre T
  vem de binar por `q` PREDITO, não por τ1). Cada evento recebe `q_pred = q_T(τ1)`
  da tabela congelada do seu T. Bins fixos em q: `[0,0.1), [0.1,0.5), [0.5,0.9),
  [0.9,1]`; por bin, `Z = (Σcap − Σq)/√(Σq(1−q))` (Poisson-binomial). A física do
  grazing/flutuação de B (que ALARGA q — smear ~50% da largura de transição, conta
  do parecer) aparece como par direcional: bin (0.5,0.9) abaixo do previsto e bin
  (0.1,0.5) acima. **Falsificador exato acoplado:** o modelo reduzido prevê ZERO
  capturas nos τ1 com `q_pred(τ1) = 0` (janela fechada NA TABELA congelada — não
  `τ1 > τ10`, que é o cruzamento de q=0.1 e deixava fatia com q=0.089 gerando
  falso-MATA ~9%; correção do 2º re-check). Capturas tardias na dinâmica real
  (reabertura por flutuação de B) violam o modelo de forma qualitativa: sob o
  modelo, E[tardias] = 0 exato.
  **MATA:** `|Z| > 1.96` em ≥2 dos 4 bins, OU ≥2 capturas tardias no agregado
  (≥2, e não ≥1, por robustez a edge-case de implementação — declarado).
  [Histórico: v1 desta predição comparava τ50/largura com estimadores diferentes e
  ~36 eventos/célula (régua desigual + potência); v2 usava 3 faixas de τ1 em que a
  faixa discriminante tinha ~4 eventos esperados e a faixa q=0 era estruturalmente
  cega (parecer: FATAL F1 + GRAVE F2/G1). Diffs no git.]
- **P-3 — REBAIXADA A EXPLORATÓRIA pelo 2º parecer (sem MATA).** As faixas com
  eventos são ~decidíveis (estatística de recorrência de BM/OU) e o modelo reduzido
  tem força restauradora espúria quando `res≈0` (OU com `x*→0` onde o 4D tem BM
  livre) — a comparação por faixa é REPORTADA como exploratória, rotulada, sem
  poder confirmatório. A parte confirmatória do programa fica em P-1 e P-2.

**Checks de consistência (decidíveis a priori — NÃO são predições; falha = instrumento
ou implementação):** deriva nula na fase morta; distribuição de (τ, face) de saída
compatível com o MC de BM no cone; decaimento do salto conforme L3-v2; braço nulo
`L=0` idêntico dentro de D com mesmos seeds.

## Braços

Confirmatório: relu (grade 3×3) + nulo (mesmos seeds, para o check de BM).
Exploratório rotulado: softplus_β=8 em 3 células (a ressurreição determinística sem
cruzamento a T baixa é decidível — vai como demonstração de contraste, sem MATA).

## O que este experimento NÃO testa

Não testa "treino = sweeping process de Moreau" em geral; testa se um modelo reduzido
de contato com parede-que-amolece prevê quantitativamente a estatística de
captura/re-morte de um sistema que ele não resolve exatamente. Não testa escala nem
a identificação com o locus SLT de queda de posto de J (programa, ciclo ≥3).

## Definição operacional de captura (mesma na predição e na medição)

Na run real: A é "capturado" quando sua coordenada normal à face de saída j satisfaz
`x ≥ x*(t)/2` com `x*(t) ≥ 4σ_x`, onde `x*(t) = max(−res_{−A,j}(t), 0)/‖(x_j,1)‖` e
`res_{−A,j} = f(x_j) − relu(z_A,j) − y_j` é o resíduo corrente EXCLUINDO a
contribuição do próprio A (é o equilíbrio da coordenada normal de A dado o resto
fixo; com o resíduo total a definição colapsaria assim que A ativasse). A face é a
da saída corrente (se A re-entra e re-sai por outra face, usa-se a nova — o modelo
reduzido não tem migração de face; divergência daí conta contra ele via P-1, como
deve). Assim a régua é a mesma nos dois lados e a divergência medida é atribuível ao
modelo reduzido, não à régua.

## Pendências antes do commit do pré-registro

1. ~~`verificacao-lemas-v2.py`~~ FEITO 2026-07-24 (todos os lemas passam, incluindo
   o lema de estrutura de sinal corrigido pela evidência).
2. `preditor.py` ESCRITO; rodando — congelar `preditor-output.json` + `config.py`
   (grade 3×3, p previstos, curvas P-2, faixas P-3, censura por célula).
3. Escrever `experimento.py` (dinâmica 4D + medições espelhando as definições acima).
4. Revisão adversarial independente do desenho completo ("as MATAs são atingíveis?
   algo ainda é decidível? texto↔código consistentes?").
5. Só então: commit = pré-registro; run confirmatória depois do commit.
