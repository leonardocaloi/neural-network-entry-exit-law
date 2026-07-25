# 4a.3 CONFIRMADO: a lei de área governa a ressurreição em MLP real (2026-07-25, pré-reg a8f8f8c, 400 runs)

**TL;DR (3 linhas):** No confirmatório pré-registrado (400 runs, análise
congelada, zero decisão pós-hoc): C1 REPRODUZ (p(volta|morte elegível) = 0.500
[0.313, 0.687], 30 mortes elegíveis, limiar-MATA 0.076), **C2 SUSTENTA — em
13/16 ressurreições dinâmicas a área do drive-líder ABRE antes da volta**
(CP-inferior 0.544 > piso 0.5), escada do canal monotônica (β3 0.457 > β10
0.370 > β300 0.211 > relu 0.000) e validade limpa (relu 0/17).

**Status epistêmico: [EVIDÊNCIA NUMÉRICA confirmatória]** — pré-registro
a8f8f8c após 9º e 10º pareceres adversariais; instrumento GT-área com paridade
bit-exata; config 100% derivado do piloto-de-taxa; seeds virgens 1900–1949.

## O que foi demonstrado (escopo honesto)

1. Em MLP subparametrizada (pressão) a T=0, unidades softplus que morrem
   dinamicamente RESSUSCITAM a taxas reprodutíveis, e ReLU-dura NUNCA (zero
   estrutural, grad≡0 — a não-comutação do T2′ medida com N=100).
2. **A área do resíduo projetado no dado líder abre antes da ressurreição em
   13/16 casos** — o portão de memória do teorema opera na rede real. As 3
   voltas-sem-área são reais e contadas contra (voltas conduzidas fora do
   líder — objeto de estudo, não escondidas). Honestidade estrutural: a
   ordenação temporal em si é garantida pelo instrumento; o achado é a
   ABERTURA da área (10º parecer).
3. **Escada do canal** p(volta|morte) monotônica em β com CIs — a suavidade da
   ativação controla o canal de volta; β=3 (a suavidade da família GELU) é o
   mais vivo. Ponte direta com a prática.

## Limites

T=0 (determinístico), 1 arquitetura pequena, alvo 1D rico, m∈{4,6}; o
estocástico segue aberto (paper próprio); GELU real tem física adicional
(barreira z*=−0.752 + canal de montante — etapa-5 em curso).

## Reprodução

pilotos-4a2/pressao/taxa/: config_4a3.json (congelado em a8f8f8c),
run_4a3.py → results-4a3/ (400 gz + manifest), analise_4a3.py →
resultado-4a3.md. Cadeia: teorema (draft-v3) → piloto-pressão (b48b527) →
piloto-de-taxa (995a87c) → ESTE confirmatório.
