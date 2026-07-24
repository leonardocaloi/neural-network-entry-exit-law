# Nota — ciclo 1 do p5-relu-morte: lemas verificados, pré-registro reprovado

**Data: 2026-07-24 (trabalho de 2026-07-23/24). Leitor-alvo: eu-em-6-meses, retomando
por que o P5 mudou de habitat e por que o primeiro desenho de teste foi descartado.**

## Contexto

A auditoria adversarial de 2026-07-23 estabeleceu que o p5-cruz (loss suave
`L=w1²w2²`) não pode testar o conteúdo diferencial de P5 nem em princípio: em loss
C^∞ a "reação" é escrava algébrica do gap (`r_N·d = 2L`), sem complementaridade
possível, e o mecanismo declarado de P-c cancela na ordem adiabática
(`d⟨w1²⟩/dt = −4w1²⟨w2²⟩ + 2T = 0` com `⟨w2²|w1⟩ = T/2w1²`; derivação verificada).
P5 foi então recolocada no habitat onde a não-suavidade é genuína: a fronteira de
ativação do ReLU. **Hipótese nova = experimento novo**: nasceu o p5-relu-morte;
o p5-cruz permanece registrado como está.

## Resultados que ficam

**[TEOREMA] Lemas L1–L3** (modelo `f(x)=relu(wx+b)`, dados `X=(0.5,1.5)`,
`Y=(0.5,1.0)`, loss quadrática; verificação simbólica + numérica commitada em
`05-experimentos/p5-relu-morte/verificacao-lemas.py`):

1. **L1 (piso plano):** no cone de morte `D = {p: wx_i+b ≤ 0 ∀i}` (ângulo
   `π−(atan x_max − atan x_min) = 150.26°`), `∇L = 0` exatamente; sob SGLD a
   dinâmica em `int(D)` é passeio browniano puro.
2. **L2 (salto de força):** cruzando a face `{z_j=0}`, `∇L` salta de `0` para
   `−(y_j/n)(x_j,1)`; a loss é C⁰ mas não C¹ (o kink vem do termo cruzado
   `−2y·relu(z)`, linear em relu). Magnitudes: `Δ = (0.2795, 0.9014)`.
3. **L3 (complementaridade estrita):** `0 ≤ s_j ⊥ r_j ≥ 0` vale identicamente com
   reação O(1) no contato (`r_j → Δ_j` quando `z_j → 0⁺`) — a estrutura que o
   p5-cruz não tinha e que os céticos da auditoria exigiram.

**[EXPLORATÓRIO] Harness de demonstração** em `05-experimentos/p5-relu-morte/`:
dinâmica SGLD + rival softplus_β + braço nulo, estimador de deriva calibrado contra
processo de deriva conhecida (smoke), determinismo verificado. Correção de protocolo
encontrada no caminho: o alvo de P-a foi movido do `Δ_j` idealizado para o alvo
protocolar `salto* = √((Δ_j κ_j)² + 4σ_c²)` (decaimento na janela + inflação de norma
com ruído) — sem isso, um desvio de até +18%/−12% teria contaminado o veredito.

## O resultado principal do ciclo: a reprova

**[FALSIFICADO — o desenho, não a hipótese].** A revisão adversarial independente
(`05-experimentos/p5-relu-morte/revisao-adversarial-2026-07-23.md`) reprovou o
pré-registro: as três predições eram decidíveis a priori (P-b: re-entrada exigiria
11–22σ; P-c: limite uniforme softplus→relu; P-a: tolerância ~12× o desvio esperado e
razão vs rival calibrada em 3.98 contra limiar 3), e nenhuma discrimina
sweeping/complementaridade de "loss com kink". Mesma patologia do p5-cruz, um nível
acima. **A run confirmatória não foi executada** — teria "passado" sem informar nada.

Lição de método, agora com dois casos: *a condição-que-MATA precisa ser verificada
como atingível por um mundo possível antes de valer como pré-registro* — checagem que
entra como etapa fixa do protocolo (adversarial review do desenho ANTES do commit).

## Próximo ciclo (v2) — de onde vem o conteúdo aberto

No caso de 1 neurônio o conjunto de restrição é fixo e o sistema é exatamente
solucionável — por isso nada é indecidível. Com **≥2 neurônios**, o piso do neurônio
morto continua exatamente plano (gradiente próprio nulo — L1 sobrevive), mas o
resíduo muda com o aprendizado do outro neurônio: a reação na fronteira decai com o
tempo (`Δ_j(t) ∝ |resíduo(t)|`), o que instala a corrida genuinamente aberta
*ressurreição por ruído vs morte permanente por esvaziamento do resíduo* — e a
estrutura de conjunto de restrição efetivo dependente do estado (parede que amolece),
vizinha do sweeping process state-dependent do survey Brogliato–Tanwani.
Desenho pré-registrável em `05-experimentos/p5-relu-morte-v2/` (rascunho, aguardando
revisão adversarial própria).

## Pendências que esta nota NÃO resolve

Prior art ainda [NÃO VERIFICADO] (Aydinoglu–Posa 2011.07626, Lu 1903.06733,
2305.12467, 2305.15912, Bolte–Pauwels 1909.10300) — baixar para `04-literatura/`
antes de qualquer texto público.
