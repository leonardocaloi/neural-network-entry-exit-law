# Etapa-5 scoping: duas leis novas no transformer — barreira do GELU e canal de montante (2026-07-25, exploratório)

**TL;DR (3 linhas):** No transformer 1-camada, (S1) o GELU tem $g'(z)<0$ para
$z<z^*=-0.7518$ (VERIFICADO por mim): unidade morta-profunda é empurrada PARA
LONGE da volta pelo próprio gradiente — auto-ressurreição só na faixa rasa
$(-0.752,0)$; 12/12 episódios com o porte cego da área do MLP erram o SINAL.
(S2) A entrada da FFN é não-estacionária: ReLU-dura morta ressuscita por
DRIVE DE MONTANTE (attn movendo-se; controle attn-congelada: Δmargem=0.000
exato) — canal inexistente no MLP. Pressão (H-c) sustentada (platô 8×).

**Status epistêmico: [EVIDÊNCIA NUMÉRICA exploratória]** — 18 runs, 3
seeds/braço, sem pré-registro; S1 é fato de cálculo (verificado analiticamente).

## O que muda no programa

1. O teorema T2′ transfere ao transformer **dentro da classe monótona**
   (softplus-β/erf-β na FFN) — é o braço em que o confirmatório e5 testa a lei.
2. **GELU real é outra física**: morte profunda é auto-terminal POR BARREIRA
   (não por fome de gradiente como β=300) — mas o transformer compensa com o
   canal de montante. A história "por que transformers recuperam unidades" =
   faixa rasa auto-revivível + montante. Estimandos próprios, não MATA da lei.
3. Instrumento GT-ÁREA generalizado (c_k=⟨w_out,W2[:,k]⟩ como v_k; líder =
   (amostra,posição)) precisa da **área-com-inclinação** (A_drive) — o porte
   sem $g'$ é incorreto para ativação não-monótona.
4. FFN é difícil de matar naturalmente (entrada d=16 sobre 72 tokens) —
   confirmatório precisa de piloto-de-taxa próprio (disciplina do 4a.3) e/ou
   injeção rasa pré-registrada.

## Setup e números

05-experimentos/etapa5-transformer/ (motor_e5.py com 8 unit-tests, 18 runs).
attn causal H=2, d=16, S=3, N=24; tarefa routing+sin richa; m_ffn 4 (pressão,
platô 0.119) vs 32 (folga, 0.015, 0 mortes). GELU: 0 mortes profundas
naturais; 0/24 ressurreições em episódios injetados (sobem até z=−0.56 e
estagnam na barreira). relu-treina: margem +0.69 médio com grad-próprio≡0
(montante); relu-frozen: 0.000 terminal exato.

## Herança para o confirmatório e5

Braços: FFN-softplus-β (classe do teorema; a lei deve transferir) + FFN-GELU
(fora-de-classe: barreira/faixa-rasa/montante como ESTIMANDOS) + FFN-relu ×
{attn-treina, attn-congelada} (fator do canal de montante). Piloto-de-taxa
antes de qualquer freeze; janela/KM herdados do 4a.3. Detalhe:
etapa5-transformer/RELATORIO-scoping.md.
