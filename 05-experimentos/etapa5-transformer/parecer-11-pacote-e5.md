# 11º parecer — pacote confirmatório e5 (NÃO-APTO → corrigido) — 2026-07-25

Revisor de verificação (Opus, 3 quedas por 529 no caminho — retomado; sondas
próprias em train e gelu-frozen-−2.0). Mecânica 100% verde (7/7 testes do
motor, sintético 21/21, config byte-idêntico SHA 34a0b5f9…). NÃO-APTO por 2
itens TEXTUAIS (mesma classe do 10º parecer):

1. **E-D confound profundidade×barreira**: a separação intra-gelu rasa-vs-
   profunda não isola a barreira — se sp-b3-frozen a −2.0 (sem barreira)
   também não revive, o zero mede DISTÂNCIA. Sonda: a unidade gelu a −2.0 SE
   MOVE (own_move 1.4e-2→1.9e-3; g'(−2)=−0.085 inverte o drive) — não é célula
   vazia; e o controle depth-matched JÁ está na grade. Exigência: ligar o
   cross-arm ao veredito; intra-só ⟹ "CONSISTENTE-COM-BARREIRA (confundido)".
2. **δ_h**: TRAIN flutua q0.99≈0.070–0.081 (~1.7× o δ_h frozen 0.0476) — a
   base é frozen-only e deve ser declarada assim; sonda operacional mostra
   n_natural_dyn≈0 sob δ_h (fantasma fechado; estimandos decidíveis vivem em
   frozen). Completar disclosure.

Decisões mantidas pelo parecer: thresh E-C = ½·CI-inf(célula própria) —
REJEITADO pooling (contaminaria o estimando; precedente C1/4a.3); E-B sob
histerese OK (δ_h pequeno para train favorece REPRODUZ mas sonda mostra 0
espúrias); pressão 1-injetada conservadora OK; τ n-pequeno OK; run_e5 são.

CORREÇÕES APLICADAS (Fable, sem re-run, sem mudar número): regra E-D com
cross-arm depth-matched + rótulo três-vias; mundo sintético (vii)
profundidade-pura (valida o rótulo nos dois sentidos; mundo (i) recoerido);
disclosures δ_h e E-D em config+README. Sintético re-rodado: TUDO PASSA ⟹
lista mínima atendida ⟹ APTO.
