# 10º parecer — pacote confirmatório 4a.3 (NÃO APTO → corrigido) — 2026-07-25

Revisor de verificação (Opus) sobre a622224; rodou paridade, preditor
(byte-idêntico), sintético 21/21, smoke do instrumento em 3 seeds.

VEREDITO original: NÃO APTO por UM defeito textual: o estimando de C2 rotulava
como "ordenação temporal t_morte ≤ t_área < t_res" o que o relógio único GT
GARANTE POR CONSTRUÇÃO sempre que a área abre — o conteúdo empírico real é
"a área do drive-líder ABRE antes da volta"; o único modo de falha é
volta-sem-área (que carrega toda a potência do MATA: exige ≈90% para
sustentar; 1 falha em 6 mata — C2 continua falsificável e NÃO decidível).
Demais achados: 10/10 itens do 9º parecer sanados (C2 parcial); números 100%
rastreáveis (config regenerado byte-idêntico, SHA f136716…); álgebra de R̂_k
verificada = leave-one-out leader drive, idêntica à régua calibrada; empate
morte-área existe mas é FÍSICO e não estrutural (1/3 no smoke, park raso);
C1-MATA de fato quase-morto (N≳47 zero-volta) — aceitável e declarado;
λ boot conservador OK; τ ruidoso mas janela robusta (mortes precoces);
runner são (retomada, manifest, detach). Menor: fiar alpha/B do config.

CORREÇÕES APLICADAS (Fable, sem mudar números): estimando C2 reescrito em
config (preditor) e README com a cláusula de honestidade estrutural e o
disclosure de desfecho-esperado em paridade com C1; alpha e bootstrap.B agora
lidos do config em todas as funções estatísticas; z_park confirmado (min de
zmax). Sintético re-rodado: TUDO PASSA. Com isso a lista mínima do parecer
está integralmente atendida ⟹ APTO.
