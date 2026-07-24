# p5-cruz — assinatura de contato na singularidade mínima

**ARQUIVADO 2026-07-24 — reprovado em auditoria adversarial SEM rodar** (predições
satisfeitas por construção em loss suave: r_N·d = 2L; mecanismo de P-c cancela na
ordem adiabática — derivações na nota 06-notas/2026-07-24-relu-morte-lemas-e-reprova.md).
Mantido como registro histórico íntegro do desenho. O potencial `w1²w2²` (λ=1/2
exato) fica reutilizável como âncora analítica do futuro `llc-patch-test`
(experimento NOVO, a pré-registrar na fase LLC). Não emendar, não rodar.

**Registrado antes de rodar: 2026-07-23. Status original: NÃO RODADO.**

## Objeto

`L(w1, w2) = w1² w2²`. O conjunto ótimo é a cruz `{w1=0} ∪ {w2=0}` — a singularidade
não-trivial mais simples que existe (RLCT λ=1/2, multiplicidade m=2, valor exato conhecido
[ESTABELECIDO], Watanabe 2009). Gradiente: `∇L = (2 w1 w2², 2 w1² w2)`.

Dinâmica: SGLD — `w ← w − η ∇L + √(2ηT) ξ`, ξ ~ N(0, I), temperatura T.

## Leitura de contato (hipótese P5, [ESPECULAÇÃO])

Tomando o estrato `{w2=0}` como "parede": distância `d = |w2|`, e a "reação de apoio"
(proxy) é a componente do gradiente normal ao estrato, `r_N = |∂L/∂w2| = 2 w1² |w2|`,
com rigidez local de contato `k(w1) = ∂²L/∂w2²|_{w2=0} = 2 w1²`.

## Predições falsificáveis (escritas ANTES de rodar)

**P-a (pouso e deslize).** Trajetórias atingem `d < δ` em tempo finito e PERMANECEM
(fração de tempo em contato → 1 após o pouso), em vez de cruzar e recruzar como difusão livre.

**P-b (complementaridade ocupacional).** No plano (d, r_N), a ocupação concentra nos dois
braços (d≈0 com r_N variável ao longo do estrato; ou r_N≈0) e o quadrante "proibido"
(d e r_N simultaneamente grandes, relativo ao baseline) fica vazio ordens de magnitude
além do que o baseline nulo produz.

**P-c (deriva entrópica para o estrato mais degenerado).** Em contato com `{w2=0}`, a força
tangencial média é zero (`∂L/∂w1 = 2 w1 w2²` some quando w2=0), mas a rigidez normal
`k(w1)=2w1²` decresce quando `|w1|` → 0: o volume de quase-ótimos cresce em direção à origem.
Predição: deriva sistemática `d⟨w1²⟩/dt < 0` puramente induzida por ruído — a partícula
"escorrega" pela parede até o ponto mais singular. Magnitude deve crescer com T.

**Condição que MATA a leitura de contato:** se a ocupação do quadrante proibido em P-b for
estatisticamente indistinguível do baseline nulo, ou se em P-c ⟨w1²⟩ não decair (ou crescer),
a analogia de contato não descreve nem o caso mínimo → registrar negativa e reavaliar P5.

## Baseline nulo

Mesma dinâmica com `L=0` (passeio aleatório puro, mesma T, mesmos seeds): define a ocupação
de (d, r_N) e a deriva de ⟨w1²⟩ esperadas SEM mecanismo — tudo em P-a/b/c é medido contra isso.

## Execução

```
python experimento.py          # roda SGLD + baseline, escreve results/*.jsonl e figuras
```

Parâmetros e seeds em `config.py`. Análise honesta: nada de escolher janela a posteriori —
janelas de medição definidas no config.
