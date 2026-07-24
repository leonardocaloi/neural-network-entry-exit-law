# etapa4-mlp-populacao — a primeira rede de verdade: estatística populacional de morte/ressurreição

**STATUS: RASCUNHO DE DESENHO — não pré-registrado, não congelado, NÃO RODAR.**
(Protocolo da casa: preditor congela números → revisão adversarial do desenho →
commit = pré-registro → run. Este documento é o passo −2.)

Etiqueta global do desenho: **[ESPECULAÇÃO]** (a hipótese populacional é própria);
as *leis do toy* que ela tenta erguer são **[TEOREMA-instância]** (draft-v2.tex,
provado numa única instância de 2 neurônios; largura/dados gerais = trabalho
futuro, é exatamente o que esta etapa começa a atacar).

---

## 0. Onde isto entra (escada do programa relu-morte)

| Etapa | Objeto | Veredito |
|---|---|---|
| 1 (ciclo 1 / v1) | 1 neurônio ReLU, R² | REPROVADO no desenho (P decidível a priori) |
| 2 (v2) | 2 neurônios, cronograma determinístico | FALSIFICADO (reabre por flutuação) |
| 3 (v3) | 2 neurônios, gate SGLD, back-reaction 3.0 | 3.0 carrega o grosso; resíduo cresce com T |
| — (v4) | 2 neurônios, geometria 2D do cone | rascunho, não rodado |
| T2′ | rede rasa softplus, GF, β→∞ | [TEOREMA-instância]: **lei de área** ∫R dt=0 |
| **4 (este)** | **MLP rasa largura m, dados 1D-2D/MNIST, SGD/SGLD** | **estatística POPULACIONAL** |

O teorema T2′ descreve UM neurônio morto isolado (hipótese H4: nenhum outro
switching, margem única ativa, episódio isolado). Numa rede real, m neurônios
compartilham o resíduo e as faces migram — **H4 quebra por construção**. A
etapa 4 não re-verifica o teorema; ela mede **quanto a lei sobrevive ao
acoplamento de m corpos** e se as leis viram estatística populacional preditível.

---

## 1. Quais leis do toy viram predição populacional

| Lei do toy (T2′) | Predição populacional candidata | Natureza |
|---|---|---|
| **Lei de área**: morto reativa quando A(t)=∫_{t_morte}^t R dt volta a 0 (não quando R>0 instantâneo) — memória | Por neurônio: t_ressurreição = 1º zero de Â_k=∫R̂_k dt, R̂_k = resíduo projetado na direção do neurônio | **TESTE** (H4 quebra) |
| **Não-comutação dos limites**: ReLU-GD discreto congela morto para sempre; GF-suave ressuscita | Fração de mortos-PERMANENTES: alta no braço ReLU-hard, cai em softplus/GELU e com T,η | **TESTE** (parcial: ver §5) |
| **L5.2 (universalidade)**: lei independe da sigmoide monótona — inclui erf/GELU | Braço GELU e braço softplus dão a MESMA estatística populacional (dentro do erro de MC) | **TESTE** |
| **Back-reaction (v3, 3.0)**: ativações seguram o resíduo compartilhado | Populacional: mortes de uns adiam/antecipam ressurreição de outros (via resíduo comum) | **TESTE** |
| Corolário: ∫R̂ dt é a variável de estado, não R | Assinatura "ressurreição atrasada pela história": Â_k prediz melhor que a mudança de sinal de R̂_k | **TESTE** (é o núcleo de P-A) |

As quatro candidatas do briefing mapeiam assim: (a) fração-permanente vs T,η →
linha 2; (b) distribuição de t_ressurreição vs lei de área com R agregado →
linha 1; (c) assinatura por neurônio via ∫R̂ dt → linhas 1+5; (d) back-reaction
populacional → linha 4.

---

## 2. Hipótese e competidores declarados

**H [ESPECULAÇÃO]:** numa MLP rasa real, a morte/ressurreição de cada neurônio
é governada pela **lei de área por neurônio** (entry–exit com memória) do T2′,
projetada na direção do neurônio via R̂_k; e essa lei ergue-se a estatísticas
populacionais (distribuição de t_ressurreição, fração-permanente vs T/η/m)
quantitativamente previsíveis a partir de R̂_k medido — apesar do acoplamento
de m corpos.

**Competidores declarados (cada um mata um pedaço de H):**
1. **Complementaridade SEM memória** (limite C1-morto): o neurônio ressuscita no
   instante em que R̂_k>0, não quando ∫R̂_k=0. Discrimina: contraste pareado
   por neurônio (zero-de-área vs mudança-de-sinal).
2. **Escape térmico puro**: ressurreição é difusão para fora do cone, taxa
   fixada só por T, independente da história de R̂. Discrimina: braço T=0
   (drift determinístico ainda fecha a área ⟹ ainda ressuscita; escape-puro
   prevê ZERO ressurreição em T=0) + nulos.
3. **ReLU-hard congela para sempre**: se softplus/GELU também congelam (sem
   ressurreição), a lei de área não tem domínio na rede real — não há o que
   explicar. Discrimina: contraste hard vs suave.
4. **Lavagem por m corpos**: com m neurônios e faces migrando, R̂_k não é bem
   definido (qual dado? faces trocam — a preocupação 2D do v4, agora em m-D) e a
   lei só vale em média/em episódios isolados. Discrimina: fração de episódios
   "limpos" (poucos switchings simultâneos) e degradação de R² da predição vs m.
5. **Artefato de régua**: definição de morte muda com n grande (MNIST); ruído de
   minibatch confundido com ruído térmico. Discrimina: régua de morte por
   fração-ativa com piso pré-declarado; braço full-batch vs minibatch.

---

## 3. Predições / MATAs candidatas (a congelar via `preditor.py`)

Números exatos e erro de MC vêm do preditor (pendência §9). Aqui, a ESTRUTURA
— separando MATA binário de estimação, lição dos 4 pareceres (§5).

- **P-A (MATA binário — memória; contraste PAREADO por neurônio).** Para cada
  episódio de morte→ressurreição limpo, comparar o erro de predição de
  t_ressurreição por dois critérios no MESMO neurônio: (i) 1º zero de Â_k=∫R̂_k dt
  [lei de área] vs (ii) 1ª mudança de sinal de R̂_k [sem memória]. O pareamento
  cancela a física comum (como o P-3′ do v3). **MATA:** se (ii) prevê tão bem ou
  melhor que (i) no pooled (|z|>1.96 a favor de ii, ou CI do ganho de (i) contém
  0), a memória/entry–exit morre na rede real — colapsa em complementaridade sem
  memória.
- **P-B (MATA binário — universalidade L5.2).** Estatística populacional
  (fração-permanente, quantis de t_ressurreição) do braço GELU vs braço softplus,
  setup pareado. **MATA:** diferença > erro de MC em ≥k células ⟹ L5.2 não
  ergue para rede real. (Predição forte: DEVEM coincidir — é a ponte com a
  prática moderna.)
- **P-C (MATA binário — existência de domínio).** Fração-permanente dos braços
  suaves (softplus-β, GELU) vs braço ReLU-hard, pareado por init. **MATA:** se
  suave ≈ hard (ambos congelam), não há ressurreição a modelar; toda a etapa é
  vacuosa e vira nota negativa.
- **P-D (MATA binário — back-reaction populacional).** Congelar um subconjunto
  aleatório de neurônios mortos (braço "freeze-parcial", análogo ao B-congelado
  do v3) e medir a taxa de ressurreição dos NÃO-congelados vs controle pareado.
  **MATA:** se congelar os outros não altera a taxa (CI∋0), não há back-reaction
  populacional pelo resíduo comum.
- **P-E (ESTIMAÇÃO, não MATA).** Fração-mortos-permanentes como função de (T, η,
  m) e distribuição de t_ressurreição — reportar superfícies com CI, sem
  threshold binário (evita o vício a-priori: as FORMAS não são decidíveis, mas o
  sinal de cada tendência isolada pode ser — ver §5). Decomposição do R² da lei
  de área por largura m (1→∞ neurônios) = quanto o acoplamento lava.

Régua/instrumento (não-predição): censura de episódios não-encerrados no
horizonte (como v2/v3); piso de |Δ| para P-B/P-C (evitar disparo por
viés-de-fio-de-cabelo — item explícito do parecer v2 e da pergunta aberta do v4);
mínimo de episódios limpos por célula.

---

## 4. Desenho mínimo honesto

**Arquitetura:** MLP de 1 camada oculta, f(x)=Σ_k a_k σ(w_k·x+b_k), largura
**m ∈ {16, 64, 256}** (a lei de m corpos deve degradar com m — m é o eixo do
teste, não conveniência).

**Dados (3 regimes de estrutura):**
- **1D-regressão** sintética (ex.: soma de bumps; n≈256) — mais próximo do toy,
  "posição geral" garantível.
- **2D-classificação** sintética (2 luas / xor suave; n≈512) — faces em 2D.
- **MNIST-subset** 2 classes (ex.: 3 vs 5), PCA→~20 dims, n≈2000 — dados
  ESTRUTURADOS (viola posição geral do teorema de propósito: testa robustez).

**Braços (pareados por init/seed onde aplicável):**
| Braço | Papel |
|---|---|
| softplus-β (β alto) | tratamento primário (o regularizado do teorema) |
| **GELU (erf-β)** | **testa universalidade L5.2 em rede real (P-B)** |
| ReLU-hard | referência da não-comutação (P-C); congela mortos |
| SGLD T>0 | gate estocástico (calor fecha a área) |
| SGD T=0 | escape-puro deve dar ZERO ressurreição (competidor 2) |
| rótulos embaralhados | nulo: geometria do locus muda ⟹ estatística muda como previsto |
| pesos aleatórios (sem treino) | nulo: detector NÃO pode acusar lei de área |
| freeze-parcial de mortos | separador de back-reaction (P-D) |

**Grade (T, η):** 2 níveis de η × 3 de T (incl. T=0), a fixar no preditor.

**N seeds:** ≥ 20 por célula (padrão da casa; cada seed dá m neurônios ⟹ muitos
episódios, mas a variância ENTRE-seed domina — Bouthillier; agregação IQM +
bootstrap estratificado por seed, contraste pareado onde há braço espelhado).
Power/MDE por célula a fixar no preditor (como v3 fez MDE≈0.053).

---

## 5. Decidível a priori vs genuinamente incerto (lição dos 4 pareceres)

Os 4 pareceres anteriores mataram desenhos cujas predições eram decidíveis antes
de rodar (lemas L1–L3 valem POR CONSTRUÇÃO; v1 e v2-original tinham P
quase-decidível; o 3º parecer do v3 rebaixou P-1′/P-2′ a estimação e transformou
P-3′ em contraste pareado). Aplicando a lição:

**Decidível a priori — NÃO é teste, serve de calibração/âncora:**
- ~metade dos neurônios "mortos-ish" no init simétrico (geometria do cone).
- No braço ReLU-hard, neurônio morto em TODOS os dados tem gradiente exatamente
  nulo ⟹ congela para sempre (teorema; é quase construção). Serve de referência
  de P-C, não de resultado.
- A lei de área do neurônio ISOLADO (m=1) já é [TEOREMA-instância] verificado —
  re-medir numa rede de 1 neurônio não é experimento novo.
- Sinal de tendências marginais isoladas (mais T ⟹ mais ressurreição) pode ser
  quase-decidível ⟹ tratar como ESTIMAÇÃO (P-E), com piso de efeito, NUNCA como
  MATA binário de fio-de-cabelo.

**Genuinamente incerto — é aqui que o compute se justifica:**
- Se a lei de área mantém PODER PREDITIVO de t_ressurreição sob acoplamento de m
  corpos (P-A) — H4 quebra, nada no teorema dita a resposta.
- As FORMAS das superfícies fração-permanente(T,η,m) e da distribuição de
  t_ressurreição (P-E).
- Back-reaction como efeito POPULACIONAL, não de 2 corpos (P-D).
- Se GELU real obedece (P-B) e se MNIST estruturado (fora de posição geral)
  preserva a lei.

---

## 6. Instrumentação: detectar morte/ressurreição por neurônio (régua barata)

O problema: margens por dado são **m×n** por passo — caro logar/armazenar tudo.
Régua operacional O(m) por passo, reusando o forward do treino:

Por neurônio k, sobre o conjunto de treino (avaliado a cada K passos, full-set,
para não confundir com ruído de minibatch):
- **fração ativa** a_k(t) = (1/n)·#{i : z_k(x_i) > 0}; **margem líder**
  m_k(t) = max_i z_k(x_i) e **dado líder** j*_k = argmax_i z_k(x_i).
- **MORTO** se a_k(t) ≤ φ (piso pré-declarado; φ=0 = morte estrita; φ pequeno
  robustece contra 1 outlier — item de risco §8). **Evento de morte:** a_k cruza
  φ↓; **ressurreição:** φ↑.
- **Resíduo projetado** R̂_k(t): resíduo da rede EXCLUINDO k, no dado líder j*,
  assinado pelo readout a_k, de modo que H1 dê R̂_k<0 na morte. Forma exata (qual
  face, readout, agregação sobre faces ativas) = escolha de modelagem a congelar
  no preditor — e é justamente onde a lavagem de m corpos entra.
- **Área acumulada** Â_k(t) = Σ_{s=t_morte}^t R̂_k(s)·Δt (um escalar por morto).
  Predição: ressuscita no 1º t com Â_k ≥ 0.

Custo: as pré-ativações z_k(x_i) já são computadas no forward; extrair
max/argmax/fração e o resíduo do líder é O(m) além do forward. Armazena-se por
morto apenas (a_k, m_k, j*, Â_k) — não a matriz m×n. Log append-only em
`results.jsonl` a cada K passos + eventos.

**Calibração obrigatória (anti-viés §2 da skill):** rodar a régua na rede de
m=1 (onde a lei de área é [TEOREMA-instância]) e confirmar que ela recupera o
t_release conhecido antes de confiar nela em m grande — o patch test da régua.

---

## 7. Compute (RTX 3070 Ti / i9 — Regime 1–2)

Objetos minúsculos para GPU: MLP rasa m≤256, n≤2000, entrada ≤20-D, saída ≤2.
Full-batch (regime limpo, sem ruído de minibatch) ou minibatch pequeno.
- Por passo: O(n·(d·m + m)) ≈ 10⁷ flops para o maior (m=256, n=2000, d=20).
  Treino 10⁴–2·10⁵ passos ⟹ ~10¹²–10¹³ flops/run ⟹ **< 1 s a poucos segundos/run
  em GPU** (3070 Ti ~20 TFLOPS); minutos em CPU no pior caso.
- Grade cheia (ordem de grandeza): 3 larguras × 3 dados × ~4 (T,η) × ~6 braços ×
  20 seeds ≈ 4·10³ runs. A ~10–30 s/run (incl. eval full-set periódico +
  logging) ⟹ **~10–40 GPU-horas** = uma a duas madrugadas na 3070 Ti. **Piloto
  reduzido** (1 largura, 1 dado, braços essenciais) em **uma tarde**.
- Confortavelmente Regime 1 (máquina local cobre todo o Regime 1 da área, per
  TODO). devinterp/LLC NÃO é necessário aqui (fase LLC é outra); só numpy/torch.

---

## 8. Riscos declarados

1. **Lavagem por m corpos lava a lei (risco principal).** Com faces migrando em
   m-D, R̂_k mal definido e Â_k ruidoso; a lei pode sobreviver só em episódios
   isolados. Mitigação pré-declarada: filtro de "episódio limpo" (≤ q switchings
   simultâneos numa janela) + reportar a FRAÇÃO limpa e o R² vs m; se quase nada
   for limpo em m=256, isso É o resultado (a lei é fenômeno de baixa-densidade).
2. **Régua de morte ambígua com dados grandes.** Em MNIST, neurônio morto em 99%
   dos dados mas vivo em 1 outlier: max_i z_k dominado por outlier. Mitigação:
   piso φ na fração-ativa, pré-declarado; sensibilidade a φ como análise de
   robustez, não escolha pós-hoc.
3. **Posição geral violada (MNIST).** Dados estruturados ⟹ faces não-genéricas;
   o teorema não cobre. Por isso MNIST entra como TESTE de robustez, não como
   confirmação; 1D/2D sintéticos garantem posição geral como controle.
4. **Censura/horizonte finito.** Morto que não ressuscita pode ser permanente
   (hard) OU só não teve a área cruzar zero ainda. Tratar como censura (v2/v3):
   célula com censura > limiar = inválida; não contar como "permanente".
5. **Ruído de minibatch × ruído térmico SGLD confundidos.** Braço full-batch
   isola o térmico; minibatch é braço separado, não default.
6. **Viés-de-fio-de-cabelo nos MATAs binários** (P-B/P-C): resíduo populacional
   pequeno pode disparar teste de sinais por viés sistemático — piso |Δ|
   pré-declarado (pergunta aberta herdada do v4).

---

## 9. Pendências antes do commit (pré-registro)

1. `preditor.py` — congela: números de P-A..P-E, forma exata de R̂_k, grade
   (T,η), φ, filtro de episódio limpo, erro de MC, piso |Δ|, seeds virgens.
   → `config.json`.
2. Calibração da régua em m=1 contra o t_release [TEOREMA-instância] (§6).
3. `run.py` + smoke (gradiente vs dif. finitas; overfit 1 batch; init com fração
   de mortos esperada; determinismo — flags do experimento-protocolo).
4. **Revisão adversarial do desenho** (regra permanente da casa; foco: P-A/P-B/
   P-C são decidíveis a priori? piso |Δ| resolve o fio-de-cabelo?).
5. Commit = pré-registro (timestamp; considerar OSF, já que ISTO escala além do
   toy — TODO item 9). 6. Run. 7. Nota + índice; negativo (P-C vacuoso) vai para
   `NEGATIVOS.md` sem dó.
8. **Decisão de numeração/nome:** integrar ao esquema `p5-relu-morte-vN` ou
   promover a `expNN` sequencial do protocolo — resolver no commit (o
   experimento-protocolo pede expNN; o programa usa vN).
