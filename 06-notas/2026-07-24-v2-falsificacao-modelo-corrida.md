# Resultado do p5-relu-morte-v2: modelo reduzido de corrida [FALSIFICADO]

**Data: 2026-07-24. Pré-registro: commit `190d94a` (aprovado em 2 rodadas de revisão
adversarial). Run confirmatória executada APÓS o commit, protocolo íntegro, zero
desvios do pré-registro. Leitor-alvo: eu-em-1-semana, desenhando o ciclo 3.**

## Veredito pré-registrado

- **P-1: FALHA** — 8 de 9 células rejeitam (|z| até 7.8) e **9 de 9 desvios são
  POSITIVOS** (teste de sinais estoura): a dinâmica 4D real captura (ressuscita)
  sistematicamente MAIS que o modelo reduzido prevê. Ex.: T=3e-3, d0=0.159:
  p_obs=0.47 vs p_prev=0.21.
- **P-2: FALHA pelo falsificador exato** — **139 capturas tardias** onde o modelo
  prevê exatamente ZERO (q_pred=0, janela fechada); o bin [0,0.1) teve 150 capturas
  vs 1.4 previstas (z=+131). Crucial: o bin (0.5,0.9) está BEM calibrado (z=−0.26)
  — onde o modelo diz que captura é provável, ele acerta; o erro está inteiro no
  lado "impossível".
- **P-3 (exploratório):** re-morte muito MENOR que a prevista nas faixas 2–3
  (ex.: obs 0.07 vs prev 0.90) — uma vez ressuscitado, o neurônio real segura a
  vida muito melhor que o modelo.
- **Instrumento VALIDADO:** censura observada (2–7%) bate com a predição exata de
  BM no cone (L1-v2) em todas as 9 células — a parte-teorema do modelo confirma;
  a falha é da hipótese (cronograma determinístico), não da régua. (Distinção
  hipótese/instrumento/regime exigida pelo protocolo §6: é hipótese.)

## Interpretação (pós-hoc, [EXPLORATÓRIO] — NÃO estava pré-registrada)

A física dominante que o modelo despreza e os dados gritam: **a parede que amolece
não fecha monotonicamente — ela tremula e REABRE.** O cronograma determinístico do
resíduo ignora as flutuações de Gibbs de B, que têm tempo de correlação do modo
lento `1/(η_B·λ_min) ≈ 85k passos` — excursões de sub-ajuste são PERSISTENTES
(dezenas de milhares de passos), reabrindo a janela de captura muito depois do
fechamento determinístico. O 2º parecer adversarial antecipou qualitativamente o
disparo do falsificador; a magnitude (139 eventos, +0.03 a +0.26 em p_captura)
mostra que a reabertura estocástica é MECANISMO DE PRIMEIRA ORDEM, não correção.
Candidato adicional (não separado neste desenho): back-reaction de A (ao ativar,
A segura o resíduo compartilhado — consistente com a re-morte menor de P-3).

**Enunciado que emerge (candidato a hipótese do ciclo 3, [CONJECTURA]):** a
ressurreição de neurônio morto é governada pela ESTATÍSTICA DE FLUTUAÇÃO do
resíduo (gate estocástico com memória longa), não pelo seu decaimento médio —
"morte permanente" em rede treinada com ruído é evento de cauda, muito mais raro
que o cronograma médio sugere, e a catraca pós-ressurreição é mais forte que a
recorrência de BM/OU prevê.

## O que esta falsificação NÃO exclui (protocolo §6)

Não exclui a leitura de contato/complementaridade em si — os lemas L1–L3-v2
continuam teoremas e a estrutura unilateral segue genuína; o que caiu é o modelo
reduzido com cronograma determinístico. Não exclui o modelo de corrida com
cronograma ESTOCÁSTICO (OU do resíduo) — é o candidato natural do ciclo 3, com
predição de capturas tardias ∝ estatística de excursão do modo lento. Não decide
back-reaction vs flutuação (precisa desenho que os separe: ex. braço com B
CONGELADO após t*, que zera flutuação e isola back-reaction).

## Próximo ciclo (v3 — novo pré-registro, hipótese nova = experimento novo)

1. Modelo reduzido 2.0: cronograma estocástico (OU exata do resíduo de B, que é
   gaussiana conhecida — sem parâmetro livre novo) → predições de p_captura,
   capturas tardias E[·]>0 quantificada, e re-morte com catraca corrigida.
2. Braço separador: B congelado após atingir vizinhança do fit (isola
   back-reaction de flutuação).
3. Mesmo protocolo: preditor a priori → revisão adversarial → commit → run.

## Artefatos

`05-experimentos/p5-relu-morte-v2/results/`: runs.jsonl (2700 runs), analise.json,
falsificacao-p1-p2.png, run-confirmatoria-log.txt. Análise 100% pré-registrada;
nenhuma estatística pós-hoc entrou no veredito.
