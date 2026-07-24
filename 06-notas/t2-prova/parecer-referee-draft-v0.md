# Parecer do referee adversarial — draft-v0 (2026-07-24)

**Veredito: APTO-PARA-EVOLUIR-A-DRAFT-V1**, com 3 bloqueadores e lista de 10
correções. Todos os 7 scripts de verificação re-executados pelo referee: passam.

## Bloqueadores e desfecho

1. **Taxa do teorema FALSA como escrita** — $O_\delta(\beta^{-1})$ contradiz o
   próprio Lema 4: a coordenada normal de $\theta_A$ no miolo é
   $\Theta(\log\beta/\beta)$ (o referee confirmou com nossos dados do passo 3:
   $\beta|z_{min}|/\log\beta = 0.334\ldots0.598$ casa com
   $1+\log(c|A|_{max})/\log\beta$). **APLICADO no v1**: taxa
   $O_\delta(\log\beta/\beta)$ no vetor completo, com a decomposição fina
   ($\theta_B$, tangencial e $t_{release}$ em $O(\beta^{-1})$).
2. **Falta (H4)** (nenhum outro evento de comutação em $[0,T]$). **APLICADO no v1.**
3. **Provenance vs repositório** — o referee afirmou a pasta v2 untracked.
   **REFUTADO na verificação**: `git ls-files` mostra 12 arquivos trackeados e o
   commit de pré-registro (190d94a) precede a run; o referee generalizou o `??`
   de um único log pós-run (`run-confirmatoria-log.txt`, agora commitado).
   Provenance reescrito no v1 com o episódio registrado.

## Demais correções aplicadas no v1

Forma aditiva da quadratura (sem divisão por $R$; zero interior de $R$ é
inócuo — a camada é mantida pela ÁREA); qualificador de subintervalos compactos
no Lema 4; fator $c$ duplicado em $A$ corrigido ($A=\int R$); $t_{release}$
definido; GELU rebaixado a "GELU-type erf smoothing" (GELU tem derivada
não-monótona, não pertence à família); "0.98" → "0.888→0.982 rumo ao limite 1"
com proveniência (variante R-prescrito, $t_{exit}=1.1352$); bibitems GH/Kri17/
SDR23 corrigidos conforme fichas; \cite ancorados no texto; não-comutação
hedgeada (literatura de não-unicidade deferida até verificação); nota de
instância única.

## Gaps para prova completa (registrados, não aplicáveis por edição)

L1 horas; L2 (H4 formal) 1–2 dias; L3 (travessia com peso exponencial) 2–4 dias;
L4 (sanduíche Chaplygin + L1.3 em três janelas) 3–5 dias; **L5 (continuação com
constantes explícitas + janela terminal) 1–2 semanas — o grosso**; L6 (quina de
saída espelhada + unicidade-para-frente) 3–5 dias. Total realista: **4–8 semanas**.

## Pendências de literatura criadas

Verificar linha de não-unicidade do GF ReLU (Wojtowytsch; Jentzen–Riekert) e a
origem Sotomayor–Teixeira antes de circular qualquer versão.
