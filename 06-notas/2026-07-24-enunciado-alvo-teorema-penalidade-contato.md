# Enunciado-alvo: treino ReLU como limite de penalidade de um sistema de complementaridade

**ATUALIZAÇÃO 2026-07-24 (mesmo dia): C1 FALSA-COMO-ENUNCIADA — reprovada no ataque
adversarial ao enunciado, com contraexemplo quantitativo verificado (scripts em
`03-verificacao/2026-07-24-ataque-enunciado-c1/`). O ataque encontrou a lei correta
no lugar: na FACE DE MORTE (lado morto com campo ≡ 0 — o caso degenerado que a
teoria PWS genérica exclui), o limite β→∞ não é complementaridade sem memória: é
contato COM COLA — o release obedece à lei de ENTRADA–SAÍDA `∫R dt = 0` sobre o
episódio (way-in/way-out de sistemas lento–rápido), verificada numericamente com
erro < 0.2%, independente de β em 4 décadas. Também estabelecido pelo ataque:
salto de ∇L através de qualquer face é rank-1 paralelo à normal (⟹ nas faces
bilaterais, Filippov = seleção LCP = limite softplus — clássico Sotomayor–Teixeira);
sticking genuíno com multiplicador > 0 existe em aberto de inits (não-vácuo);
relu-GD discreto congela para sempre (a SELEÇÃO da solução depende da aproximação —
a inclusão é não-única); o campo de weight decay é TANGENTE a toda face (por
z ser homogêneo em θ), e durante sticking dá deslizamento genuíno rumo ao ÁPICE do
cone (o canto de corank máximo — ponte direta com o estrato mais singular da SLT).
C2 mantida como experimento, exigindo co-escalonamento explícito (T→0 mata a
captura; enunciado-limite requer (d0, T, η_A/η_B) co-escalados).

**NOVO ALVO — T2 [CONJECTURA], estimativa de prova 2–4 semanas:** para rede relu
rasa, dados genéricos, horizonte finito, o limite β→∞ do GF-softplus existe e é o
sistema híbrido com (a) cruzamento/deslizamento Filippov=LCP nas faces bilaterais
[conhecido]; (b) na face de morte, sticking com release pela lei de entrada–saída
`∫R dt = 0` (multiplicador adesivo no intervalo); (c) seleção dependente da
aproximação (relu-GD congela). Corolário físico falsificável: *ressurreição
determinística é atrasada pela HISTÓRIA do resíduo* (lei de área). PENDÊNCIA
OBRIGATÓRIA antes da prova [NÃO VERIFICADO]: checar se Kristiansen–Hogan/Jeffrey
("exit from sliding: determinacy-breaking"; "limit solution of regularized PWS",
Hairer) já cobrem a face com lado identicamente nulo — as ferramentas vivem lá; a
aplicação a relu + a lei do resíduo não foram encontradas publicadas na busca.

**CHECK DE PRIOR ART DE T2 (2026-07-24, 4 leitores; fichas em 04-literatura,
aresta pws-entrada-saida-t2):**
1. **A lei ∫R dt = 0 é CLÁSSICA e tem nome** — é a função de entrada–saída /
   Pontryagin delay (De Maesschalck–Schecter, arXiv:1511.01815, eq. 1.4 é
   exatamente ela; nossa variável rápida cai no caso linear deles com 1/β de
   pequeno parâmetro). [RESULTADO 2024-26, lido]. A contribuição NÃO é a lei —
   é onde ela aparece. Rebaixar qualquer linguagem de "lei nova" nos drafts.
2. **O caso lado-identicamente-nulo está explicitamente EM ABERTO**: Jeffrey 2016
   (exit from sliding) exige folds genéricos — fora de escopo; Guglielmi–Hairer
   (limit-reg, lido integral) têm a lei integral num caso degenerado VIZINHO
   (dupla tangência isolada, ambos os campos ≠ 0) e declaram que os casos
   não-genéricos "have not been investigated". A ferramenta para atacá-lo existe:
   Kristiansen, "Blowup for flat slow manifolds" (arXiv:1603.01821, Nonlinearity
   2017) — variedades lentas exponencialmente achatadas sob regularização tanh,
   exatamente nosso objeto [NÃO VERIFICADO além do abstract].
3. **A ponte para ReLU/dying-neurons/treino NÃO EXISTE em lugar nenhum** (16
   buscas). Nem a versão não-autônoma com forcing R(t) acoplado (vizinho mais
   próximo: Sensi–Desroches–Rodrigues 2023, input dependente do tempo, só abstract).
4. **Consistência teoria↔nossos números**: Guglielmi–Hairer preveem saída
   independente da regularização em codim-1 — bate com nossa independência de β
   em 4 décadas (0.2%).

**T2 REENQUADRADO (honesto):** não é lei nova; é (i) teorema de caso aberto —
entrada–saída sobre variedade exponencialmente plana com forcing não-autônomo
(ferramentas: Kristiansen + DM-S + Sensi), e (ii) PRIMEIRA aplicação da caixa
entrada–saída/PWS a treino de redes, com corolário físico verificado (ressurreição
atrasada pela história do resíduo). É exatamente "abrir a caixa": importar, com
teorema de cola onde a literatura declarou lacuna.

O enunciado original abaixo fica como registro histórico do ciclo (não editar).

---

**Etiqueta global original: [CONJECTURA] — enunciado preciso, sem prova; escrito
ANTES de tentar provar, para ataque adversarial ao ENUNCIADO (método da casa:
reprovar cedo é barato). Leitor-alvo: eu-em-2-semanas, decidindo se a prova vale
1–3 meses.**

## Objeto

Rede rasa `f_θ(x) = Σ_k v_k σ(w_k·x + b_k)`, `σ = relu` ou `σ_β = softplus_β`
(`σ_β → σ` uniformemente, `β→∞`), loss quadrática em dados finitos, gradient flow
(determinístico primeiro; SGLD depois) com taxas por bloco `η_k` (pré-condicionador
diagonal fixo).

## Conjectura C1 (limite de rigidez / penalidade→complementaridade)

Para horizonte finito `[0, T_h]` e dados em posição geral: as trajetórias do gradient
flow com `σ_β` convergem, quando `β→∞` (uniformemente em `[0,T_h]`, a menos de
eventos de medida nula), para a solução de uma **inclusão diferencial com estrutura
de complementaridade por neurônio-dado**:

- para cada par (neurônio k, dado i), gap `s_{ki} = max(−z_{ki}, 0)` e reação
  `r_{ki}` = contribuição do dado i ao campo em `θ_k` quando ativo;
- `0 ≤ s_{ki} ⊥ r_{ki} ≥ 0` ao longo da trajetória-limite (L3/L3-v2 são o caso
  n=1,2 disso, já verificados);
- na região de morte do neurônio k (todos `s_{ki} > 0`), o campo em `θ_k` é
  exatamente nulo (piso plano, L1/L1-v2);
- o conjunto de restrição EFETIVO de cada neurônio morto é movido pelo resíduo dos
  demais (estrutura state-dependent; caso 2-neurônios = "parede que amolece" do v2).

**Forma forte (a que abriria a caixa):** a trajetória-limite é solução de um
processo tipo sweeping state-dependent perturbado, e os teoremas de
existência/unicidade da literatura de contato (via prox-regularidade por face,
longe dos cruzamentos de faces) transferem, dando well-posedness que a formulação
suave não fornece no limite β→∞.

## Conjectura C2 (consequência quantitativa — o que só esta leitura prevê)

No regime SGLD com separação de escalas (`η_morto ≫ η_resto` ou resíduo lento): a
estatística de ressurreição/captura de um neurônio morto é dada pelo modelo de
corrida (primeira passagem de BM no cone de morte × cronograma do resíduo), com a
lei de captura do preditor-v2 como caso mínimo. C2 é exatamente o que o experimento
v2 testa em N=2; a versão-teorema é um limite adiabático a provar.

## Rota de prova candidata (esboço, não prova)

1. Fora das fronteiras de ativação: convergência clássica (campo suave, Grönwall).
2. Perto de uma face isolada: `σ_β` = penalidade regularizada do vínculo unilateral;
   comparar com a rota Filippov/convexificação e com a rota Moreau-Yosida; o campo
   limite em cruzamento transversal é o de Filippov — a complementaridade entra como
   SELEÇÃO do campo no modo deslizante.
3. Colar trechos por indução em eventos (número finito de cruzamentos em [0,T_h]
   fora de um conjunto de dados/inits de medida nula — precisa prova).
4. SGLD: convergência fraca de processos (tightness + identificação do gerador);
   mais difícil, deixar como C2-fraca se necessário.

## Riscos declarados (por que isto pode NÃO ser uma descoberta)

- **R1 (colapso em Filippov/Bolte–Pauwels, risco ~40–50%):** a inclusão-limite pode
  ser exatamente a dinâmica de campo conservativo já estabelecida (1909.10300 — já
  em 04-literatura). A contribuição teria que sobreviver como: (i) a estrutura de
  complementaridade EXPLÍCITA com reação/gap nomeados e mensuráveis; (ii) o conjunto
  móvel state-dependent (ausente lá); (iii) C2 (lei quantitativa). Se nada disso
  sobrar além de vocabulário → nota limitativa, não teorema.
- **R2:** o limite β→∞ pode não comutar com t→∞ nem com ruído (ordem dos limites) —
  restringir a horizonte finito desde o enunciado.
- **R3:** cruzamentos de faces (corank ≥ 2) quebram prox-regularidade — enunciar
  "longe dos cruzamentos" e tratar o caso geral como aberto.
- **R4 (prior art não lido):** Aydinoglu–Posa cobre ReLU=LCP no mapa forward; a
  seção §4/§6 de Bolte–Pauwels está [não extraído]; Amari/Wei 2008 não lido.
  OBRIGATÓRIO ler antes de investir na prova.

## Próximo passo desta nota

Submeter ESTE enunciado ao ataque adversarial (o mesmo protocolo dos experimentos):
"C1 é falsa como enunciada? é trivial? já existe? qual é o menor contraexemplo
candidato?" — só depois disso começar prova.

---

## VEREDITO DA LEITURA DE KRISTIANSEN (2026-07-24; ficha 1603.01821): INVESTIR, ESCOPO REDUZIDO

- Nosso GF-softplus É a classe tanh dele (ε = 2/β); flatness idêntica em espécie.
- Descoberta operacional: no gauge w = e^{βz} nosso sistema já é ALGÉBRICO — motor
  da prova = forma quadrática de DM-S + quadratura exata + Grönwall; Kristiansen
  entra como prior art da classe, justificativa do gauge e templates de canto
  (Lemma C.1 entrada; C.6–C.7 saída — a contração exponencial de fibras explica
  nossa independência de β). Ele NÃO tem lei ∫=0 (lacuna reconfirmada).
- Novo de verdade (ninguém tem): entrada–saída com lado ≡0 sob forcing acoplado
  não-autônomo; bootstrap R^β ↔ neurônio morto; enunciado híbrido; multi-margem.
- **T2'-min (alvo de prova):** sob (H1) morte transversal com margem única ativa e
  (H2) primeiro zero transversal de A(t)=∫cR⁰ em t_exit<T: a trajetória GF-softplus
  converge (β→∞, uniforme fora de vizinhanças dos 2 cantos) ao sistema híbrido
  {ativo → sticking → release em t_exit}, com t_release = t_exit + O(β⁻¹ln β).
  Verificação numérica: erro·β constante por 3 décadas (O(1/β) exato).
- **Cronograma honesto: 6.5–9 semanas** (5–6 com H3 assumida e DM-S citado);
  a estimativa anterior de 2–4 era otimista por fator ~2. Dificuldades reais:
  bootstrap de acoplamento (passo 4) e topologia do enunciado-limite (passo 5).
  Abertos declarados: cone corank≥2 (ápice), SGLD, horizonte infinito.
- Pendência antes do passo 3: ler o corpo de Sensi–Desroches–Rodrigues 2023.
