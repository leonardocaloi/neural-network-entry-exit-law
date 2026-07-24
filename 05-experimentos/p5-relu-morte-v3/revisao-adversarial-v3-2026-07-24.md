# Revisão adversarial do desenho v3 — 2026-07-24

**Veredito original: NÃO APTO** — raiz única: P-2'/P-3' testavam predições-ponto
de um modelo (3.0) sabidamente enviesado (+0.075), herdando o viés. Contas do
revisor: P-2' com z esperado 5.4 (quase-decidível); P-3' com 22% de
falso-"mecanismo-novo" (92–96% das capturas ocorrem ANTES do freeze — o braço
congelado herda a back-reaction pré-freeze subprevista).

**Verificados corretos:** projeção modal do acoplamento (fator a fator, sinal do
feedback positivo); mecânica do freeze consistente experimento↔preditor (linha a
linha); herança de réguas do v2; HARKing limpo (preditor lê só config, nunca
runs); números da config; x̄·COARSE exato; aproximação de bloco conservadora.

**Correções (TODAS aplicadas antes do commit):**
1. P-2' → ESTIMAÇÃO (resíduo de taxa-tardia por célula + CI, como P-1').
2. P-3' → CONTRASTE PAREADO por seed: (livre − congelado) observado vs previsto
   (0.053/0.042/0.043) — cancela o viés comum, isola o canal pós-freeze; MATA no
   pooled (1350 pares), |z|>1.96.
3. Seeds ampliados nas células pareadas (1000–1449, N=450); potência registrada:
   MDE pooled ≈ 0.053 a 80% (conservador).
4. CI de P-1' com limitações declaradas (n=9; SE binomial ~0.033/célula);
   desfecho esperado pré-declarado.
5. Redação: gap médio 52% (faixa 0.15–0.67); resíduo +0.02..+0.16; código morto
   (cronogramas) mantido apenas como referência do modelo 2.0.

**Estado: APTO nos termos do parecer ("feito isso, o desenho fica APTO").
O commit deste arquivo + README + config_v3.json + preditor + experimento
constitui o pré-registro; run confirmatória só depois.**
