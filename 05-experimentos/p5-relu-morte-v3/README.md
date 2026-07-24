# p5-relu-morte-v3 — o gate estocástico

**STATUS: RASCUNHO DE DESENHO — AGUARDANDO REVISÃO ADVERSARIAL. NÃO COMMITAR COMO
PRÉ-REGISTRO, NÃO RODAR.** (Protocolo da casa: revisão do desenho antes do commit.)

## Contexto e hipótese

O v2 FALSIFICOU o modelo reduzido com cronograma determinístico (9/9 desvios
positivos; 139 capturas tardias onde E=0; ver `06-notas/2026-07-24-v2-falsificacao`).
Interpretação registrada (então [EXPLORATÓRIO], agora a hipótese deste ciclo):

**RESULTADO DE PREDITOR (2026-07-24, antes de qualquer run):** o modelo 2.0
(flutuação-apenas) prevê ≈ o determinístico morto (smear preserva a média;
tardias 10× abaixo do observado) — a conjectura "flutuação-apenas" está
REFUTADA em nível de teoria (números em `config_v3.json`, seção `celulas`,
mantidos como REFERÊNCIA). Errata registrada na nota de falsificação do v2.

**[CONJECTURA v3']** A estatística de ressurreição é governada pela
**BACK-REACTION**: ativações transitórias de A seguram o resíduo compartilhado
(B desacelera), estendendo a janela. O modelo reduzido **3.0** — BM no cone ×
excursão 1D ACOPLADA aos modos de B (o termo de acoplamento é a projeção modal
exata da contribuição de A, sem parâmetro livre) — prevê $p(\text{captura})$,
tardias e o braço B-congelado. No braço congelado a back-reaction sobre B é
impossível por construção ⟹ lá o 3.0 colapsa no 1.0/2.0 — o braço separa os
mecanismos DENTRO do mesmo experimento.

**Competidores declarados:** (i) back-reaction de A (mecanismo rival da nota de
falsificação — ISOLADO pelo braço B-congelado, onde flutuação não existe por
construção); (ii) artefato de régua (mesmas definições operacionais congeladas
do v2 — inalteradas); (iii) modelo determinístico v1 (já falsificado; referência).

## Desenho

Grade: as MESMAS 9 células (d0, T) congeladas em `../p5-relu-morte-v2/config_v2.json`.
Seeds NOVOS: 1000–1149 (N=150/célula; os dados do v2, seeds 0–149, são
motivação exploratória — nunca alvo). Braços: relu (9 células) + nulo (mesmos
seeds) + **B-CONGELADO** (3 células da coluna alvo-0.5: B congelado em
$t_{freeze} = 25000$ passos, cronograma constante dali em diante — o separador).

## Números congelados do preditor (modelo 3.0; 2.0 mantido como referência)

O 3.0 fecha em média 52% do gap do v2 (faixa 0.15–0.67 por célula) e multiplica
tardias ~10× sobre o 2.0 — mas NÃO fecha tudo (resíduo +0.02..+0.16 contra os
dados VELHOS, que são motivação exploratória). Consequência de desenho honesto: um
MATA binário de P-1' contra dados novos seria quase-decidível (o vício que
reprovou v1 e o desenho-v2-original). Estrutura final:

- **P-1' (ESTIMAÇÃO, não MATA):** resíduo $p_{obs} - p_{3.0}$ por célula com CI
  95% (IQM/bootstrap sobre as células) — atribuição quantitativa: quanto da
  anomalia do v2 a back-reaction carrega em dados virgens. Pré-declarado: se o
  CI do resíduo médio excluir 0 por cima, o 3.0 é insuficiente
  ([FALSIFICADO-como-suficiente], esperado); se incluir 0, o 3.0 basta
  (surpresa registrável).
- **P-2' (ESTIMAÇÃO — corrigido pelo 3º parecer):** o revisor calculou z
  esperado 5.4 contra dados velhos (quase-decidível — tardias vêm da MESMA
  física subprevista de P-1'); vira estimação do resíduo de taxa-tardia por
  célula + CI, tratamento idêntico a P-1'.
- **P-3' (MATA binário — CONTRASTE PAREADO, corrigido pelo 3º parecer):** como
  92–96% das capturas ocorrem ANTES do freeze, o nível absoluto do braço
  congelado herda o viés do 3.0. O teste correto cancela o viés comum: o
  contraste pareado por seed $(\text{livre} - \text{congelado})$ observado vs.
  previsto pelo 3.0 (0.053/0.042/0.043 por célula) — isola o canal de
  reabertura PÓS-freeze, que o freeze mata por construção. Seeds ampliados nas
  3 células pareadas (1000–1449, N=450) e MATA no POOLED (1350 pares):
  $|z|>1.96$. **Potência registrada:** MDE ≈ 0.053 a 80% (conservador, sem
  crédito de pareamento); por célula é descritivo.
- **P-1' (CI honesto):** o CI propaga o SE binomial por célula (~0.033 em
  N=150) além do bootstrap sobre 9 células (n=9 admitido como limitação);
  pré-declarado que "3.0-INSUFICIENTE" é o desfecho esperado.

Instrumento (não-predição): censura = BM exato (como v2); >30% ⟹ célula
inválida; <7 válidas ⟹ inconclusivo.

## O que este experimento NÃO testa

Back-reaction quantitativa (o braço separador a DETECTA, não a modela);
multi-neurônio/população (etapa 4 do programa); escala.

## Pendências antes do commit

1. `preditor_v3.py` rodar e congelar `config_v3.json` (p por célula, taxas de
   tardias, predições do braço congelado, com erro de MC declarado).
2. `experimento.py` (reuso do v2 + braço congelado + estatística de tardias).
3. Revisão adversarial do desenho. 4. Commit = pré-registro. 5. Run.
