# Resultado do p5-relu-morte-v4: 4.0 sobrevive; a escada fecha o toy

**Data: 2026-07-24. Pré-registro: 75e0779 (5º parecer aplicado). Seeds virgens
1500–1649; análise 100% pré-registrada.**

## Vereditos

- **P-1'' (MATA com piso): PASSA** — 0 rejeições (|z| ≤ 1.85 em todas as 9
  células); sinais-com-piso 7/9 (< 8). O modelo 4.0 (geometria 2D acoplada, sem
  parâmetro livre) sobrevive ao mesmo teste que matou o determinístico.
  [EVIDÊNCIA NUMÉRICA]
- **P-2'' (escada):** 1.0 → +0.206; 3.0 → +0.120; 4.0 → **+0.046
  [+0.032, +0.056]** — cada mecanismo corta ~metade do resíduo; o restante é
  pequeno, positivo NESTE lote, e o CI exclui zero.

## Honestidades obrigatórias

1. **O sinal do resíduo do 4.0 flutua entre lotes de seeds:** contra as
   observações do v3 (seeds 1000–1449) era −0.016; neste lote é +0.046. As
   observações por célula variaram até +0.06 entre lotes (N=150, SE≈0.04 —
   compatível com flutuação, z≲1.5). Conclusão honesta: |resíduo| ≲ 0.05, sinal
   instável, consistente com mistura de ponte-B-clipping (declarada) e
   flutuação de lote. NÃO se afirma "4.0 subprevê" nem "sobreprevê" — afirma-se
   que o que resta é da ordem do erro entre lotes.
2. 7/9 células acima do piso com mesmo sinal é indício sistemático que o MATA
   (≥8) não converteu em rejeição — registrado como está, sem promover a
   conclusão pós-hoc.
3. Leitor de análise teve KeyError cosmético (chave P1_celulas_validas removida
   no patch do 5º parecer) — só na impressão, não na análise.

## O toy está FECHADO

Arco completo em 4 ciclos e 5 pareceres adversariais: determinístico
[FALSIFICADO] → flutuação [REFUTADO no preditor] → back-reaction [EVIDÊNCIA:
canal pós-freeze z=0.42, tardias CI∋0] → geometria 2D [PASSA no MATA; escada
1.0→3.0→4.0 quantificada]. Resíduo final ≲0.05 com candidato nomeado
(B-clipping) e assinatura de lote. Encerrado o programa toy; próximo:
etapa 4 (MLP largura m — rascunho pronto em 05-experimentos/etapa4-mlp-populacao).
