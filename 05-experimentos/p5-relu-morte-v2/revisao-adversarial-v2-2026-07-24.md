# Revisões adversariais do pré-registro v2 — 2026-07-24 (2 rodadas + delta)

Revisor independente (Opus), mesmo papel que reprovou o v1. Nenhuma trajetória 4D
confirmatória executada em nenhuma rodada (só contas analíticas, SymPy e simulações
de processos genéricos: BM no cone, OU com flutuação de Gibbs injetada).

## Rodada 1 — NÃO APTO (resumo)

**Progresso central: P-1 PASSOU no teste de decidibilidade que matou o v1.** A lei
(τ1, face) é exata-compartilhada (L1-v2); p_captura difere só via q_real≠q_reduzido;
a flutuação de Gibbs de B NÃO decide (viés −0.002, verificado por simulação); os
efeitos restantes (back-reaction, migração de face, grazing) têm sinal e magnitude
genuinamente incertos ⟹ P-1 é teste real. Calibração sob H0 exata: falso-MATA
combinado ~4.7%.

**Reprovas:** F1 — a faixa de transição de P-2 (a única que discrimina) tinha ~4
eventos esperados em 2 das 3 T (MATA inatingível — inanição estatística no lugar da
decidibilidade). F2/G1 — a faixa q=0 era estruturalmente cega ao falsificador mais
limpo (captura tardia). G2 — P-3: faixas testáveis ~decidíveis (recorrência BM/OU) e
o modelo reduzido tem força restauradora espúria em res≈0. M1 — assimetria de
horizonte de busca de captura (500k vs 70k). M2 — teste de sinais frágil a viés
uniforme (aceitável pós-correções). M3 — narrativa do README superdimensionava a
flutuação de B em P-1 (o smear preserva a média; o sinal dela vive em P-2). M4 —
w_local não documentado.

Também confirmados corretos: lemas, régua modelo-vs-medida de x*, streams do
roda_lote (bit a bit), lema de pooling, searchsorted, censura (após correção da
régua no meio da revisão: censurada = não-capturada nos dois lados).

## Correções aplicadas (mesma data)

1. P-2 redesenhada: calibração da curva inteira — eventos face-1 de TODAS as células
   e T pooled, binados por q PREDITO (bins fixos 0/0.1/0.5/0.9/1), Poisson-binomial
   por bin; MATA: ≥2 de 4 bins OU ≥2 capturas tardias.
2. N_SEEDS_CELULA: 50 → 150.
3. P-3 rebaixada a exploratória (sem MATA), artefato documentado.
4. M1/M4: busca de captura limitada a t1+W_LOCAL (70k), na config.
5. M3: narrativa corrigida.

## Rodada 2 (re-check dirigido do delta) — NÃO APTO por 1 item; APTO após correção

- (a) Potência com N=150 (contas do revisor): ~921 eventos pooled; bins da transição
  23–34 eventos; potência 0.77 no bin (0.5,0.9) contra o smear-puro (a alternativa
  mínima), P(≥2 bins)=0.32; miscalibração grosseira detectada com folga. Aceito
  N=150 com a potência registrada (opcional não-bloqueante: ~450/célula p/ 80%
  contra smear-puro).
- (b) **BLOQUEANTE (novo, introduzido pelas edições): régua do falsificador tardio.**
  `τ1 > τ10` contava fatia com q_pred=0.089 (T=1e-3) ⟹ E[tardias|modelo]=0.48,
  P(≥2)=8.5% de falso-MATA por contabilidade. Correção: tardia ⟺ `q_pred == 0`
  (τ_zero por T: 10650/6997/3972) ⟹ E=0 exato. **Aplicada.**
- (c) Teste de sinais com M1: calibrado (3.9% sob H0). Aceito.
- (d) Preditor NÃO precisa re-rodar (quantidades populacionais independentes de
  N_SEEDS). Ramos de OR correlacionados (bin [0,0.1) × tardias) registrados como
  inofensivos. Sem referências órfãs na config.

**Estado final: APTO PARA COMMIT após a correção (b), que está aplicada neste
mesmo commit.** O commit deste arquivo + README + config_v2.json + preditor-output
+ código constitui o pré-registro; a run confirmatória só depois.
