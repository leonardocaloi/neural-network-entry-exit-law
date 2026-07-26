# e5 CONFIRMADO: o mecanismo do teorema opera num transformer; canal de montante reproduz nos 4 braços (2026-07-26, pré-reg 1b45498, 800 runs)

**TL;DR (3 linhas):** Análise congelada, zero decisão pós-hoc: **E-C TRANSFERE**
(sp-β3-frozen raso 22/50, CP 0.440 [0.300, 0.587] ≫ portão n*=6) — a classe do
teorema ressuscita numa FFN de transformer pelo canal próprio, com **lei de
área 21/22 (95%)** nos trios; **E-B REPRODUZ nos 4 braços** (montante Δ≈0.20-0.22,
CIs excluem 0; até relu 20/100 vs 0/100); E-A limpo (relu-frozen 0/100).

**Status epistêmico: [EVIDÊNCIA NUMÉRICA confirmatória]** — pré-registro
1b45498 após 11º parecer (2 rodadas); instrumento com histerese δ_h=0.0476
calibrada de dados; seeds virgens 2000–2049; manifest SHA256.

## Vereditos congelados

1. **E-A VALIDADE**: relu-frozen 0 voltas sustentadas em 100 runs (2 profundidades).
2. **E-B MONTANTE (reprodução)**: Δ(train−frozen) = 0.210 [0.120,0.300] (β3),
   0.200 [0.110,0.280] (β10), 0.220 [0.120,0.320] (GELU), 0.200 [0.120,0.280]
   (relu). A entrada não-estacionária da FFN é um canal de ressurreição real e
   comparável em magnitude entre ativações.
3. **E-C TRANSFERÊNCIA**: no braço isolado do canal próprio (attn congelada),
   a classe do teorema revive 22/50 injeções rasas; trios com área-antes-da-volta
   21/22. O portão de memória do T2′ opera na FFN.
4. **E-D BARREIRA**: rótulo de honestidade estrutural disparou como desenhado —
   "CONSISTENTE-COM-BARREIRA (confundido com profundidade)": gelu-frozen raso
   29/50 vs profundo 0/50, MAS sp-β3-frozen −2.0 também 0/50 ⟹ a profundidade
   sozinha explica o zero. A assinatura S1 permanece: inversão de sinal
   A_drive vs A_mlp em 92% dos profundos vs 4% dos rasos — a barreira existe
   no campo de drive; a terminalidade não é separável da profundidade neste
   desenho (herança: injeção depth-matched mais rasa que -2 dentro da região
   de barreira, ou tempo-de-subida como estimando).
5. **E-E ESCADA**: β3 > β10 nos dois fatores (frozen 0.220 vs 0.090; train
   0.430 vs 0.290) — suavidade abre o canal, de novo.
6. **GELU real**: raso frozen 0.580 [0.432,0.718] — o mais vivo de todos os
   braços no canal próprio raso; com montante, 0.860. A prática (GELU) é o
   regime MAIS revivível — fecha a ponte com plasticidade.

## Reprodução

05-experimentos/etapa5-transformer/: config_e5.json (1b45498), run_e5.py →
results-e5/ (800 gz + manifest-e5.json), analise_e5.py → resultado-e5.md.
