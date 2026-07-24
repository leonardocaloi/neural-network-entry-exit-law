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

## Predições (congelar números via `preditor_v3.py` antes do commit)

- **P-1' (lei de corrida estocástica).** $p(\text{captura})$ por célula prevista
  pelo modelo 2.0. Régua idêntica ao v2 (captura operacional, censura = não
  captura). **MATA:** ≥3 células válidas com $|z|>1.96$ OU ≥8 desvios de mesmo
  sinal — os MESMOS critérios que mataram o v1: o modelo 2.0 enfrenta
  exatamente o teste que o antecessor reprovou, com dados novos.
- **P-2' (capturas tardias — de falsificador a predição).** O modelo 2.0 prevê
  taxa NÃO-nula de capturas com $\tau_1 > \tau_{zero}^{det}$ (janela
  determinística fechada; $\tau_{zero}$ congelados do v2). **MATA:**
  $|z|>1.96$ no agregado pooled (aprox. normal da Poisson-binomial por célula).
- **P-3' (braço separador).** Nas células B-congeladas, o cronograma é constante
  pós-freeze (valor estocástico do instante do freeze, modelado pelo OU até
  $t_{freeze}$): captura exatamente integrável. **MATA:** ≥2 de 3 células
  congeladas com $|z|>1.96$. Desvio AQUI (sem flutuação) implica back-reaction —
  o desfecho é informativo nos dois sentidos e nenhum confirma a conjectura
  gratuitamente.

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
