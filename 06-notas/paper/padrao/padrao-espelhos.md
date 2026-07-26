# Padrão dos papers-espelho — convenções de escrita extraídas por leitura direta

Data: 2026-07-25
Escopo: quatro papers da nossa bibliografia usados como *espelho de estilo* para o paper-v1
(`06-notas/paper/paper-v1.tex`). Objetivo: fixar convenções REAIS observadas, com citação
literal como evidência de cada uma. **Zero invenção**: tudo abaixo foi lido nesta sessão em
arXiv (`/abs`, `/abs/vN`, `export.arxiv.org/abs`) e/ou na versão HTML `ar5iv.labs.arxiv.org/html/<id>`.

## Convenção de etiqueta usada NESTE documento

Este é um dossiê documental de estilo, não de matemática. Mapeamento explícito das etiquetas
do `CLAUDE.md` para este contexto:

- `[ESTABELECIDO]` — observação feita na fonte aberta e lida nesta sessão; a citação literal
  logo abaixo é a evidência.
- `[NÃO VERIFICADO]` — dado obtido de snippet de busca ou de campo que a fonte não exibiu;
  a página do artigo publicado NÃO foi aberta.

Nada aqui é `[ESPECULAÇÃO]` exceto onde marcado explicitamente como recomendação nossa
(seção 6, "aplicação ao nosso paper"), que é inferência a partir das observações.

## Ressalvas de honestidade (ler antes de usar)

1. **Correção de autoria no pedido.** O pedido chamava 1511.01815 de "Dumortier–De Maesschalck".
   A fonte diz outra coisa. `[ESTABELECIDO]`
   > **Authors:** Peter De Maesschalck, Stephen Schecter
   (arXiv:1511.01815 abs e export.arxiv.org/abs/1511.01815). **Não há Dumortier neste paper.**
   Se o paper-v1 cita "Dumortier–De Maesschalck" para este arXiv-id, a citação está errada e
   precisa ser corrigida para De Maesschalck–Schecter (ou o id trocado, se a intenção era o
   livro/artigo de Dumortier–Roussarie sobre canards — **não verificado, não assumir**).

2. **Artefatos de transcrição.** As fórmulas vieram do render ar5iv via WebFetch e chegaram
   com unicode corrompido em alguns pontos (ex.: `0​p​t` no lugar de um macro de dimensão em
   2108.08106; `z̄` e subscritos remontados). Onde isso ocorre, marquei `[render ar5iv]`.
   **Não copiar fórmula daqui para o paper** — copiar só a *forma retórica*.

3. **Elisões.** Onde a transcrição veio truncada com `...`, a reticência é do processo de
   extração, **não do original**. Marcado como `[trecho elidido na extração]`.

4. Contagens de página de versão *publicada* vêm de resultado de busca, não da página do
   editor aberta → etiquetadas `[NÃO VERIFICADO]`.

---

# 1. arXiv:1511.01815 — De Maesschalck & Schecter, *entry-exit*

## 1.1 Metadados literais `[ESTABELECIDO]`

- Título: **"The entry-exit function and geometric singular perturbation theory"**
- Autores: **Peter De Maesschalck, Stephen Schecter**
- Subjects: **"Dynamical Systems (math.DS)"**
- Comments: **ABSENT** (campo vazio no abs; verificado em `export.arxiv.org/abs/1511.01815`)
- Journal reference no arXiv: **ABSENT**
- Submissão: **[v1] Thursday, 5 November 2015 17:00:30 UTC (1,063 KB)** — versão única.
- Publicado em J. Differential Equations 260(8) (2016) 6697–6715, DOI 10.1016/j.jde.2016.01.008
  → ~19 páginas de revista. `[NÃO VERIFICADO]` (snippet de busca; página da ScienceDirect não aberta)

## 1.2 Como o título é construído `[ESTABELECIDO]`

Fórmula: **[objeto matemático nomeado] + "and" + [ferramenta/teoria]**.
"The entry-exit function and geometric singular perturbation theory".

Observações: sem dois-pontos, sem subtítulo, sem verbo, sem adjetivo de valor
("novel", "unified", "a new approach"). Nomeia o objeto herdado da literatura
(*entry-exit function*, artigo definido "The") e a moldura teórica (*GSPT*).
O título **não anuncia o resultado**; anuncia o encontro entre um objeto e um método.

## 1.3 Primeiras duas frases do abstract `[ESTABELECIDO]`

> "For small $\epsilon>0$, the system $\dot x = \epsilon$, $\dot z = h(x,z,\epsilon)z$, with
> $h(x,0,0)<0$ for $x<0$ and $h(x,0,0)>0$ for $x>0$, admits solutions that approach the $x$-axis
> while $x<0$ and are repelled from it when $x>0$. The limiting attraction and repulsion points
> are given by the well-known entry-exit function."

Tom e tempo verbal:
- **Frase 1 = o sistema, escrito por extenso, com as desigualdades de sinal explícitas.**
  Zero motivação, zero "recently there has been much interest". O abstract abre com a EDO.
- **Presente do indicativo, terceira pessoa, sujeito matemático** ("the system ... admits
  solutions"). O agente é o objeto, não os autores.
- A frase 2 conecta ao objeto já conhecido da literatura, com o hedge invertido
  ("the **well-known** entry-exit function") — sinaliza herança, não novidade.
- Só na frase 3 aparece "we": "we explain this phenomenon using geometric singular
  perturbation theory". O "we" chega **depois** do fenômeno estar posto.

Abstract completo (4 frases, ~75 palavras):
> "... For $h(x,z,\epsilon)z$ replaced by $h(x,z,\epsilon)z^2$, we explain this phenomenon
> using geometric singular perturbation theory. We also show that the linear case can be
> reduced to the quadratic case, and we discuss the smoothness of the return map to the line
> $z=z_0$, $z_0>0$, in the limit $\epsilon\to0$."

Verbos de resultado usados: **"explain"**, **"show"**, **"discuss"** — escalonados por força.
"discuss" para a parte menos conclusiva. Isso é hedge por *escolha de verbo*, não por advérbio.

## 1.4 Introdução: como abre `[ESTABELECIDO]`

Primeiras três frases:
> "Consider the slow-fast planar system (1.1) ẋ=εf(x,z,ε), (1.2) ż=g(x,z,ε)z, with x∈ℝ, z∈ℝ,
> (1.3) f(x,0,0)>0; g(x,0,0)<0 for x<0 and g(x,0,0)>0 for x>0. For ε=0, the x-axis consists of
> equilibria; see Figure 1.2(a) below. These equilibria are normally attracting for x<0 and
> normally repelling for x>0."

Convenção: **"Consider ..." + equações numeradas já na primeira frase**. A hipótese estrutural
recebe número próprio ((1.3)) para poder ser *citada por número* nos teoremas depois — é o
mecanismo de "hipóteses nomeadas" deste paper: numeradas como equações, não como (H1)/(A2).

## 1.5 Enunciado do teorema principal `[ESTABELECIDO]`

> **Theorem 1.1.** "Consider (1.9)–(1.10), where f and g are C∞ and satisfy (1.3). Choose x₀\*<0
> such that p₀(x₀\*) can be defined using (1.4) [trecho elidido na extração] If z₀>0 and ε₀>0 are
> sufficiently small, and I₀ is a sufficiently small neighborhood of x₀\*, then:
> (1) For 0<ε<ε₀ and x₀∈I₀, the solution of (1.1)–(1.2) through (x₀,z₀) first reintersects the
> line z=z₀ in a point (x,z)=(pε(x₀),z₀).
> (2) Define p:I₀×[0,ε₀)→ℝ [trecho elidido na extração] Then there is a C∞ function p̃ of three
> variables such that p(x₀,ε)=p̃(x₀,ε,ε log ε).
> (3) If g(x,z,ε)/f(x,z,ε) has property ℱ in z, then p is a C∞ function of (x₀,ε)."

Convenções extraídas:
- **Hipóteses por referência numérica**, não por sigla: "Consider (1.9)–(1.10) ... satisfy (1.3)".
- **Constantes existenciais anônimas, quantificadas por "sufficiently small"**: "If z₀>0 and ε₀>0
  are sufficiently small, and I₀ is a sufficiently small neighborhood of x₀\*". Não há constante
  explícita (nada como "C = 4L²/μ"). O rigor está na ordem dos quantificadores, não em números.
- **Conclusão em itens numerados (1)(2)(3)**, cada um uma afirmação independentemente citável.
- **Item condicional dentro do teorema** ((3) só vale sob "property ℱ") — a hipótese extra fica
  dentro do item, não vira teorema separado.
- **Teorema principal enunciado na §1 (Introdução)**, antes de qualquer maquinaria.

Definição técnica é isolada e nomeada com letra caligráfica:
> "We shall say that a function h(x₁,…,xₘ,z) has property ℱ in z if h(x₁,…,xₘ,z)−h(x₁,…,xₘ,0)
> is flat in z as z→0, i.e., |h(x₁,…,xₘ,z)−h(x₁,…,xₘ,0)|=𝒪(zᴺ) as z→0 for all N>0."

Convenção: "We shall say that ... if ...", seguido de "i.e.," com a versão quantitativa. Definição
informal **e** formal na mesma frase.

## 1.6 Comentário pós-teorema e Remark `[ESTABELECIDO]`

Logo depois do teorema vem **uma frase de interpretação mecânica** (por que o resultado tem a
forma que tem), não um parágrafo de auto-elogio:
> "The dependence of p on ε log ε results from the fact that, after blow-up, solutions must pass
> by a line of saddle equilibria with positive and negative eigenvalues of equal magnitude."

E um Remark que **admite explicitamente que a hipótese é de conveniência**:
> "**Remark 1.** The assumptions in Theorem 1.1 that f and g only depend on (x,z,ε) and are C∞
> are there to simplify the proof. The theorem remains true if f and g are functions of
> (x,z,ε,α), where α is a finite-dimensional additional parameter."

Convenção forte e diretamente aplicável a nós: **dizer qual hipótese é essencial e qual é
comodidade técnica, por escrito, imediatamente após o teorema.**

## 1.7 Como escrevem provas `[ESTABELECIDO]`

Estrutura de seções:
> 1. Introduction
> 2. Blow-up
>  2.1. Polar coordinates
>  2.2. Affine coordinates for z̄>0
>  2.3. Passage from S₀ᵃ to S₁ᵃ
>  2.4. Passage from S₁ᵃ to S₂ᵃ
>  2.5. Passage from S₂ᵃ to S₃ᵃ
>  2.6. Proof of Theorem 1.1
> 3. Passage by the line of saddles

E a Introdução declara a arquitetura da prova, incluindo **onde a parte chata foi jogada**:
> "The organization of the paper is as follows: we prove Theorem 1.1 using blow-up in Section 2,
> but delay the somewhat technical part, a treatment of the flow past a line of resonant saddles,
> to Section 3."

Convenções:
- Prova **decomposta em passagens geométricas** (§2.3–2.5), cada trecho do caminho vira subseção.
- A prova do teorema é uma **subseção-costura** (§2.6) que compõe os pedaços:
  > "To prove Theorem 1.1, we define Pa:S₀→S₃ by Pa=P₃ᵃ∘Pᵃ=P₃ᵃ∘P₂ᵃ∘P₁ᵃ."
- **Não há apêndice**: a parte técnica vira uma *seção final* (§3), não um apêndice.
  Isto é uma diferença real em relação ao padrão ML.

## 1.8 Como citam `[ESTABELECIDO]`

**Numérico entre colchetes, agrupado, ordem não-alfabética (ordem de relevância/uso):**
> "Pontryagin delay and the entry-exit function are discussed in [12, 9, 14, 4]."
> "Pontryagin delay is also encountered in the codimension-one bifurcation of slow-fast systems
> that gives rise to the solutions known as canards; see [1, 5]."

Convenções: (i) padrão `[n]` puro; (ii) "see [1, 5]" como forma de apontar sem afirmar;
(iii) a frase carrega o *conteúdo* e a citação só localiza — nunca "As shown by Smith [3], ...".

## 1.9 Numéricos / figuras `[ESTABELECIDO]`

Não há experimento numérico. As figuras são **esquemas de fluxo**, e as legendas são
telegráficas, referenciando equações por número:
> "Figure 1.1. Nullclines of (1.5)–(1.6), and a typical solution for small ε>0."
> "Figure 1.2. Dynamics for ε=0 and an orbit for small ε>0 in dashed line. (a) For system
> (1.1)–(1.2). (b) For system (1.9)–(1.10). (c) Case of multiple turning points."
> "Figure 2.1. Flow of (2.6)–(2.8)."
> "Figure 2.2. Flow of (2.10)–(2.12)."

Convenção: **legenda de figura = ponteiro para as equações, ≤ 2 linhas, sem conclusão embutida.**
(Contraste violento com o padrão ML — ver §4.9.)

## 1.10 Hedge `[ESTABELECIDO]`

Praticamente ausente. O único localizado é sobre *atribuição histórica*, não sobre matemática:
> "As far as we know, it was originally discovered in a different context..."

Convenção: hedge é permitido para *quem descobriu o quê*; proibido para *o que é verdade*.

---

# 2. arXiv:1603.01821 — Kristiansen, *Blowup for flat slow manifolds* (Nonlinearity)

## 2.1 Metadados literais `[ESTABELECIDO]`

- Título final (v5/publicado): **"Blowup for flat slow manifolds"**
- Título da v3: **"Blowup for flat slow manifolds with applications to regularization of piecewise
  smooth systems using tanh and a model of aircraft ground dynamics"**
- Autor: **Kristian Uldall Kristiansen** (autor único)
- Subjects: **Dynamical Systems (math.DS)**
- Comments (versão final): **"Shortened title and updated references. To appear in Nonlinearity"**
- Comments (v3): **"Corrected Prop 4.4"**
- Cinco versões:
  > [v1] Sun, 6 Mar 2016 12:47:48 UTC (477 KB) / [v2] Tue, 8 Mar 2016 05:30:28 UTC (477 KB) /
  > [v3] Tue, 15 Mar 2016 20:23:15 UTC (479 KB) / [v4] Wed, 16 Nov 2016 09:43:33 UTC (581 KB) /
  > [v5] Mon, 27 Mar 2017 11:24:36 UTC (554 KB)
- Nonlinearity 30(5) (2017) 2138–2184 → 47 páginas. `[NÃO VERIFICADO]` (snippet de busca)

## 2.2 Como o título é construído `[ESTABELECIDO]`

Fórmula final: **[método] + "for" + [classe de objeto]** → "Blowup for flat slow manifolds".
Quatro palavras. Sem dois-pontos.

**Evidência documental de uma convenção de processo, não só de forma:** o título foi
*encurtado* na revisão, e a mudança foi registrada no campo Comments —
"Shortened title and updated references". O título longo v3 continha as aplicações
("with applications to ... tanh ... and a model of aircraft ground dynamics"); o título
publicado corta todas elas. **Aplicações saem do título; ficam no abstract.**

## 2.3 Primeiras duas frases do abstract `[ESTABELECIDO]`

> "In this paper we present a method for extending the blowup method, in the formulation of
> Krupa and Szmolyan, to flat slow manifolds that lose hyperbolicity beyond any algebraic order.
> Although these manifolds have infinite co-dimension, they do appear naturally in certain
> settings."

Tom e tempo verbal:
- **"In this paper we present ..."** — presente, primeira pessoa do plural, autor único usando
  "we". (Confirmado no corpo: "we present a method", "we will apply", "we shall consider".)
- **Frase 1 credita a formulação alheia dentro do próprio enunciado do que se faz**:
  "in the formulation of Krupa and Szmolyan" — atribuição por nome, no abstract, sem número.
- **Frase 2 é uma concessão-e-defesa**: "Although these manifolds have infinite co-dimension,
  they do appear naturally". Antecipa a objeção óbvia (isso é patológico?) e responde no mesmo
  período, com o enfático "do appear".

Resto do abstract (a estrutura "por que isso importa" é uma lista de 3 aplicações rotuladas):
> "For example in (a) the regularization of piecewise smooth systems by tanh, (b) a model of
> aircraft landing dynamics, and finally (c) in a model of earthquake faulting. We demonstrate
> the approach on a simple model system and the examples (a) and (b)."

Convenção notável de honestidade: **as aplicações são rotuladas (a)(b)(c) e a última frase diz
explicitamente quais foram feitas e qual não foi** — (c), earthquake faulting, é citada como
motivação e **excluída** do que foi demonstrado. Isso é declarar escopo no abstract.

## 2.4 Introdução `[ESTABELECIDO]`

> "In this paper we focus on slow-fast systems on (x,y)∈ℝns×nf of the form x′=εf(x,y,ε),
> y′=g(x,y,ε), where ()′=d/dτ denotes differentiation with respect to the fast time τ.
> System (1.1) is called the fast system. Near points where g(x,y,ε)=𝒪(1) the variables x∈ℝns
> and y∈ℝnf vary on separate time scales for 0<ε≪1." [render ar5iv]

Mesmo padrão de 1511.01815: sistema por extenso na primeira frase, notação declarada na hora,
nomenclatura fixada na frase seguinte ("System (1.1) is called the fast system").

## 2.5 O "teorema principal" é uma RECEITA numerada `[ESTABELECIDO]`

Antes dos teoremas, a §1.1 Overview enuncia um objeto chamado **"Method 1"**:
> "Consider a system (x,y)∈ℝn+1 with a small parameter ϵ≥0. Suppose that there exists a critical
> manifold for ϵ=0 with an eigenvalue λ(y) that decays exponentially as y→0+. To study the system
> using blowup (and hyperbolic methods of dynamical systems theory) then do the following:
> [Step 1-4 outlined]"

Convenção muito relevante para nós: **quando a contribuição é um procedimento e não uma
implicação, ela é empacotada num ambiente numerado próprio ("Method 1") com "Suppose that ..."
+ passos**, e os teoremas (3.3, 4.4, ...) são as *instâncias* onde o procedimento é executado.

Teorema concreto (aplicação tanh), com constante explícita:
> **Theorem 3.3** — "Consider ϕ∈CS​Tn−1, n≥2. Let r2=ε1/(2​n−1). Then the slow manifold Sa,ε
> intersects {x=θ} in (θ,yθ(ε)) with yθ(ε)=θ2+ε−r22n(2ϕ[n])2/(2n−1)η(n)2(1+r2F(r2))."
> [render ar5iv — fórmula corrompida na extração, NÃO copiar]

Apesar do render ruim, a *forma* é legível e é a convenção: **"Consider [classe]. Let [escala
explícita r₂ = ε^{1/(2n−1)}]. Then [objeto] intersects [seção] in [ponto] with [expansão
assintótica explícita, incluindo o fator de correção (1+r₂F(r₂))]."** Ou seja: aqui **as
constantes/expoentes SÃO explícitos** (o expoente 1/(2n−1) é o resultado), e o resto fica num
`F` genérico com dependência declarada. Contraste com De Maesschalck–Schecter (§1.5), onde tudo
é "sufficiently small".

## 2.6 Provas `[ESTABELECIDO]`

Abertura e fechamento típicos:
> "**Proof.** We demonstrate this result by considering our model system (3.16)."
> "... This expression coincides with (3.4). ∎"

Convenções:
- "Proof." em negrito, **primeira frase declara a estratégia** ("We demonstrate this result by
  considering our model system (3.16)"), não entra em conta direto.
- Fecha por **reconciliação com uma equação anterior** ("coincides with (3.4)") + `∎`.
- **Existe apêndice, e ele contém uma prova inteira**:
  > "Appendix A Proof of Theorem 3.3 using blowup"
  Isto é: o corpo dá o resultado + argumento principal; o apêndice dá a prova alternativa/completa
  pelo método. **Convenção: apêndice = prova longa por método, nomeada no título do apêndice.**

Índice de seções completo (para calibrar granularidade):
> 1. Introduction / 1.1. Overview / 2. A method for flat slow manifolds / 2.1. Blowup /
> 2.2. A flat slow manifold / 2.3. How to deal with flat slow manifolds / 2.4. Chart κ₁ /
> 2.5. Conclusion / 3. Regularization of PWS systems by tanh / 3.1. Regularization: Slow-fast
> analysis / 3.2. Geometry of regularization / 4. A flat slow manifold in a model for aircraft
> ground dynamics / Appendix A Proof of Theorem 3.3 using blowup

Note: **§2.5 chama-se "Conclusion" e é conclusão *da seção 2*, não do paper.** Conclusão local.

## 2.7 Como cita `[ESTABELECIDO]`

Numérico com **localizador interno obrigatório quando aponta para um resultado específico**:
> "[11, 12]"
> "[33, Proposition 6.3]"
> "[2, Eq. (1)]"

Convenção: nunca citar "[33]" seco quando se usa uma proposição de [33]; escrever
`[33, Proposition 6.3]`. Isso é auditável e é exatamente o que o nosso protocolo de
`verificacao-fontes` exige.

## 2.8 Hedge `[ESTABELECIDO]`

Duas frases localizadas, ambas *sobre o alcance do trabalho*, e ambas com admissão explícita:
> "We do not aim to provide a general geometric framework for our approach. This must be part of
> future research."
> "Although this can potentially lead to confusion we also believe that it stresses the
> standardization of the method."

Convenções: (i) o limite do escopo é declarado em voz ativa e curta ("We do not aim to..."),
não escondido; (ii) "believe" só aparece em juízo *editorial* (sobre notação), nunca sobre um
fato matemático; (iii) "potentially lead to confusion" — admite o custo da própria escolha.

## 2.9 Numéricos `[ESTABELECIDO]`

Não foi localizada nenhuma menção a simulação numérica ou estudo computacional no conteúdo
extraído; as legendas de figura são descritivas de geometria. Paper puramente analítico com
aplicações modeladas.

---

# 3. arXiv:2108.08106 — Eberle, Jentzen, Riekert & Weiss (ReLU / gradient flows)

## 3.1 Metadados literais `[ESTABELECIDO]`

- Título: **"Existence, uniqueness, and convergence rates for gradient flows in the training of
  artificial neural networks with ReLU activation"**
- Autores: **Simon Eberle, Arnulf Jentzen, Adrian Riekert, Georg S. Weiss**
- Subjects: **Machine Learning (cs.LG); Dynamical Systems (math.DS); Numerical Analysis (math.NA)**
- Comments: **"30 pages. arXiv admin note: text overlap with arXiv:2107.04479, arXiv:2108.04620"**
- Journal-ref: **"Electronic Research Archive 2023, Volume 31, Issue 5: 2519-2554"**
- Submissão: **[v1] Wed, 18 Aug 2021 12:06:19 UTC (32 KB)** — versão única, 32 KB (só fonte, sem figuras).

Sinal forte: **32 KB de submissão e nenhuma figura.** Paper 100% texto/matemática.

## 3.2 Como o título é construído `[ESTABELECIDO]`

Fórmula: **[lista dos três resultados provados, em série] + "for" + [objeto] + "in" + [contexto de
aplicação] + "with" + [restrição técnica]**:
"Existence, uniqueness, and convergence rates | for gradient flows | in the training of artificial
neural networks | with ReLU activation".

Convenções: título **descreve exatamente os teoremas**, não a mensagem; enumera as três
propriedades na mesma ordem em que aparecem no paper; termina cravando a limitação técnica
("with ReLU activation"). 17 palavras — longo é aceitável quando é *taxonômico*.

## 3.3 Primeiras duas frases do abstract `[ESTABELECIDO]`

> "The training of artificial neural networks (ANNs) with rectified linear unit (ReLU) activation
> via gradient descent (GD) type optimization schemes is nowadays a common industrially relevant
> procedure. Till this day in the scientific literature there is in general no mathematical
> convergence analysis which explains the numerical success of GD type optimization schemes in the
> training of ANNs with ReLU activation."

Tom e tempo verbal:
- **Frase 1 = fato de prática, presente, sem citação, sem "we".** Estabelece que o objeto é real
  e importante ("industrially relevant procedure").
- **Frase 2 = declaração de lacuna, em negativo forte**: "there is in general no mathematical
  convergence analysis which explains ...". O gap é o motor do paper e é dito na segunda frase.
- **Todas as siglas são definidas na primeira ocorrência** — ANNs, ReLU, GD, e depois GF — dentro
  do próprio abstract. Convenção rígida deste grupo.

Estrutura restante do abstract (é uma máquina de 5 movimentos): motivação → lacuna → estratégia
("it seems to be a natural direction of research to first aim to develop a mathematical convergence
theory for time-continuous GF differential equations and, thereafter, to aim to extend ...") →
**"In this article we establish two basic results ..."** → **"In the first main result of this
article we establish ... under the assumption that ..."** / **"In the second main result of this
article we prove ... under the assumption that ..."**.

Convenção de altíssimo valor para nós: **no abstract, cada resultado principal vem com sua
hipótese colada** ("under the assumption that the probability distribution of the input data ...
is absolutely continuous with a bounded density function"; "under the assumption that the target
function and the density function ... are piecewise polynomial"). Nunca um resultado sem sua
condição, nem no abstract.

E a taxa é dita com número: **"converges with rate 1 to the risk of the critical point"**.

## 3.4 Como enunciam o teorema principal `[ESTABELECIDO]`

Anúncio antes:
> "In the first main result of this article, see Theorem 1.1 below, we establish in the training
> of such ANNs..."

Enunciado (Theorem 1.1) — **totalmente auto-contido, sem referência a hipótese externa**, no
estilo "let-chain" característico do grupo Jentzen `[render ar5iv, fórmulas corrompidas]`:
> "Let d,0​p​t,𝔡∈ℕ, 𝒶∈ℝ, 𝒷∈(𝒶,∞), f∈C([𝒶,𝒷]^d,ℝ) satisfy 𝔡=d0pt+20pt+1, let 𝔭:[𝒶,𝒷]^d→[0,∞)
> be bounded and measurable, let ℜ_r∈C(ℝ,ℝ), r∈ℕ∪{∞}, satisfy for all x∈ℝ that
> (⋃_{r∈ℕ}{ℜ_r})⊆C^1(ℝ,ℝ), ℜ_∞(x)=max{x,0}, sup_{r∈ℕ}sup_{y∈[-|x|,|x|]}|(ℜ_r)'(y)|<∞, and
> limsup_{r→∞}(|ℜ_r(x)−ℜ_∞(x)|+|(ℜ_r)'(x)−𝟙_{(0,∞)}(x)|)=0, ... let θ∈ℝ^𝔡, and let 𝒢:ℝ^𝔡→ℝ^𝔡
> satisfy for all ϑ∈{v∈ℝ^𝔡:((∇ℒ_r)(v))_{r∈ℕ} is convergent} that 𝒢(ϑ)=lim_{r→∞}(∇ℒ_r)(ϑ). Then
> (i) it holds that 𝒢 is locally bounded and measurable and
> (ii) there exists a unique Θ∈C([0,∞),ℝ^𝔡) which satisfies for all t∈[0,∞), s∈[t,∞) that
> 𝐃^{Θ_t}⊆𝐃^{Θ_s} and Θ_t=θ−∫_0^t 𝒢(Θ_u) du."

Convenções:
- **"Let ... let ... satisfy ... Then (i) ... and (ii) ..."** — um período único de hipóteses,
  conclusão em itens romanos. Sem (H1)/(A1); **as hipóteses não são nomeadas, são declaradas
  inline no let-chain.**
- **Todo quantificador escrito por extenso**: "satisfy for all x∈ℝ that", "which satisfies for all
  t∈[0,∞), s∈[t,∞) that". Nunca "∀" no corpo do texto.
- **A definição do objeto derivado ($\mathcal G$ como limite dos gradientes suavizados) entra como
  hipótese do teorema**, não como definição prévia. O teorema carrega seu próprio setup.
- Segue-se **uma frase de rastreabilidade**: "Theorem 1.1 is a direct consequence of Theorem 3.3
  below." → o teorema da introdução é sempre corolário de um teorema técnico posterior.

E existe o ambiente **"Setting 2.1"**, que empacota o cenário compartilhado por várias
proposições (mesmo let-chain, declarado uma vez) — mecanismo do grupo para não repetir 20 linhas
de hipóteses a cada resultado.

## 3.5 Como escrevem provas `[ESTABELECIDO]`

Abertura e fechamento:
> "**Proof of Proposition 2.2.** Observe that, e.g., [24, Proposition 2.2] establishes [equação 2.7]."
> "... The proof of Proposition 2.2 is thus complete. ∎"

Convenções (rígidas e mecânicas):
- Cabeçalho **nomeia o objeto provado**: "Proof of Proposition 2.2." (não só "Proof.").
- Primeira palavra da prova costuma ser **"Observe that"** / "Note that", seguida de citação
  precisa com localizador: "e.g., [24, Proposition 2.2] establishes ...".
- **Fecho ritual explícito, com o nome do objeto repetido**: "The proof of Proposition 2.2 is thus
  complete. ∎". Nunca só `∎`.
- **Não há apêndice**: as seções são a própria escada de provas (§2 regularidade → §3 existência e
  unicidade → §4 semialgébrico → §5 taxas). Cada seção prepara a próxima; o teorema da introdução
  é fechado na última.

Estrutura completa (para calibrar granularidade — 5 seções, 15 subseções):
> 1 Introduction / 2 Properties of the risk function and its generalized gradient function
> (2.1 Mathematical description of artificial neural networks (ANNs) / 2.2 Differentiability
> properties of the risk function / 2.3 Local Lipschitz continuity of active neuron regions /
> 2.4 Local Lipschitz continuity properties for the generalized gradient function /
> 2.5 Subdifferentials) / 3 Existence and uniqueness properties for solutions of gradient flows
> (GFs) (3.1 Existence ... / 3.2 Uniqueness ... / 3.3 Existence and uniqueness ...) /
> 4 Semialgebraic sets and functions (4.1 ... / 4.2 On the semialgebraic property of certain
> parametric integrals / 4.3 On the semialgebraic property of the risk function) /
> 5 Convergence rates for solutions of GF differential equations (5.1 Generalized Łojasiewicz
> inequality for the risk function / 5.2 Local convergence ... / 5.3 Global convergence ...)

Convenção de nomes de subseção: **frase nominal completa e literal do que é provado ali**
("Local Lipschitz continuity of active neuron regions"), nunca rótulos vagos tipo "Preliminaries",
"Analysis", "Main lemma".

Roteiro anunciado na introdução:
> "The remainder of this article is organized in the following way. In Section 2 we establish
> several regularity properties for the risk function of the considered supervised learning problem
> and its generalized gradient function. In Section 3 we employ the findings from Section 2 to
> establish existence and uniqueness properties for solutions of GF differential equations."

## 3.6 Como citam `[ESTABELECIDO]`

Numérico com localizador, e **agrupamento por tema com "e.g."**:
> "[24, Proposition 2.2]" / "[25, Setting 2.1 and Proposition 2.3]" / "[39, Definition 8.3]"
> "For convex objective functions, the convergence of GF and GD processes to the global minimum in
> different settings has been proved, e.g., in [5, 23, 34, 35, 38]."
> "For further abstract convergence results ... we refer, e.g., to [6, 20, 32, 37, 40]."

Convenções: (i) `e.g., in [...]` sinaliza lista *não exaustiva* — hedge honesto sobre cobertura
bibliográfica; (ii) "we refer, e.g., to [...]" para literatura adjacente; (iii) localizador sempre
que se usa o resultado. Bibliografia com ~41 entradas ([1]–[41]).

## 3.7 Numéricos `[ESTABELECIDO]`

**Nenhum.** Sem experimentos, sem figuras, sem tabelas. Paper puramente teórico, 30 páginas.

## 3.8 Hedge `[ESTABELECIDO]`

Localizados:
> "it seems to be a natural direction of research to first aim to develop a mathematical
> convergence theory ..." — hedge sobre *estratégia de pesquisa*, no abstract.
> "roughly speaking, both the target function ... and the unnormalized density function should
> partially coincide" — "roughly speaking" marcando paráfrase informal de uma hipótese técnica.

Convenção: hedge concentrado em (a) escolha de programa de pesquisa e (b) glosa informal de
hipótese formal. **Nunca no enunciado do teorema.** A frase "Till this day in the scientific
literature there is in general no mathematical convergence analysis" mostra o oposto do hedge:
afirmação de lacuna feita com "in general" como única atenuação.

---

# 4. arXiv:2302.12902 — Sokar, Agarwal, Castro & Evci, *dormant neuron* (ICML 2023)

## 4.1 Metadados literais `[ESTABELECIDO]`

- Título: **"The Dormant Neuron Phenomenon in Deep Reinforcement Learning"**
- Autores: **Ghada Sokar, Rishabh Agarwal, Pablo Samuel Castro, Utku Evci**
- Subjects: **Machine Learning (cs.LG)**
- Comments: **"Oral at ICML 2023"** (a página abs longa exibe "Oral presentation at ICML 2023")
- Journal-ref: **ABSENT**
- Submissões: **v1: February 24, 2023 (1,093 KB) / v2: June 13, 2023 (4,845 KB)** — o salto de
  1,1 MB → 4,8 MB entre v1 e camera-ready é sinal de figuras/apêndice acrescidos.

## 4.2 Como o título é construído `[ESTABELECIDO]`

Fórmula: **"The" + [nome cunhado do fenômeno] + "in" + [campo]**.
"The Dormant Neuron Phenomenon in Deep Reinforcement Learning". Title Case (padrão ICML).

Convenções: (i) o título **batiza um fenômeno** e usa artigo definido "The", como se o fenômeno
já existisse; (ii) o método proposto (ReDo) **não aparece no título** — o fenômeno é o produto
principal, a intervenção é secundária; (iii) sem dois-pontos, sem número, sem claim de SOTA.

## 4.3 Primeiras duas frases do abstract `[ESTABELECIDO]`

> "In this work we identify the dormant neuron phenomenon in deep reinforcement learning, where an
> agent's network suffers from an increasing number of inactive neurons, thereby affecting network
> expressivity. We demonstrate the presence of this phenomenon across a variety of algorithms and
> environments, and highlight its effect on learning."

Tom e tempo verbal:
- **"In this work we identify ..."** — presente, primeira pessoa, verbo de descoberta na primeira
  palavra útil. Zero parágrafo de contexto: o abstract abre com a contribuição.
- **A definição do fenômeno vem embutida na oração relativa** ("where an agent's network suffers
  from an increasing number of inactive neurons") — o leitor sai da primeira frase sabendo o que é.
- **Frase 2 = escopo da evidência**, não resultado numérico: "across a variety of algorithms and
  environments". Números ficam para o corpo.
- Sequência do abstract: identificar → demonstrar presença → propor método ("To address this issue,
  we propose a simple and effective method (ReDo) that Recycles Dormant neurons throughout
  training.") → resultado ("Our experiments demonstrate that ReDo maintains the expressive power of
  networks by reducing the number of dormant neurons and results in improved performance.").
- **O acrônimo é construído por capitalização dentro da frase**: "**Re**cycles **Do**rmant neurons".
- Autodescrição "simple and effective" — a única palavra avaliativa, e aplicada à *simplicidade*.
- Abstract inteiro **sem um único número**.

## 4.4 Introdução `[ESTABELECIDO]`

> "The use of deep neural networks as function approximators for value-based reinforcement learning
> (RL) has been one of the core elements that has enabled scaling RL to complex decision-making
> problems (Mnih et al., 2015; Silver et al., 2016; Bellemare et al., 2020). However, their use can
> lead to training difficulties that are not present in traditional RL settings. Numerous
> improvements have been integrated with RL methods to address training instability, such as the use
> of target networks, prioritized experience replay, multi-step targets, among others
> (Hessel et al., 2018)."

Convenção: **funil clássico ML** — sucesso do campo (com 3 citações) → "However," + problema →
o que já se tentou (com citação). Contraste direto com os papers de math.DS, que abrem com a EDO.

Contribuições em bullets, verbo em primeira pessoa, um por linha:
> "Our contributions in this work can be summarized as follows:
> • We demonstrate the existence of the dormant neuron phenomenon in deep RL.
> • We investigate the underlying causes of this phenomenon and show its negative effect on the
>   learning ability of deep RL agents.
> • We propose Recycling Dormant neurons (ReDo), a simple method to reduce the number of dormant
>   neurons and maintain network expressivity during training.
> • We demonstrate the effectiveness of ReDo in maximizing network utilization and improving
>   performance."

## 4.5 O "teorema" é uma DEFINIÇÃO OPERACIONAL com limiar `[ESTABELECIDO]`

Não há teorema. O papel estrutural do teorema é ocupado por uma **definição mensurável e
normalizada**, com um hiperparâmetro de limiar explícito:
> "We define the score of a neuron i (in layer ℓ) via the normalized average of its activation as
> follows: s^ℓ_i = 𝔼_{x∈D}|h^ℓ_i(x)| / (1/H^ℓ ∑_{k∈h} 𝔼_{x∈D}|h^ℓ_k(x)|).
> We say a neuron i in layer ℓ is **τ-dormant** if s^ℓ_i ≤ τ."

Convenções: (i) **normalizar pela média da camada** antes de limiar (score adimensional);
(ii) **parametrizar o limiar e carregá-lo no nome do objeto** ("τ-dormant"), de modo que toda
afirmação posterior é indexada por τ; (iii) "We say ... if ..." — mesma fórmula verbal da
definição matemática em 1511.01815 ("We shall say that ... if ...").

E o método em uma frase, com auto-desqualificação de sofisticação:
> "The main idea of ReDo, outlined in Algorithm 1, is rather simple: during regular training,
> periodically check in all layers whether any neurons are τ-dormant; for these, reinitialize their
> incoming weights and zero out the outgoing weights."

## 4.6 Headings como FRASES-RESULTADO `[ESTABELECIDO]`

Seções: **1 Introduction / 2 Background / 3 The Dormant Neuron Phenomenon / 4 Recycling Dormant
Neurons (ReDo) / 5 Empirical Evaluations / 6 Related Work / 7 Discussion and Conclusion.**

Mas as **subseções da §3 são afirmações completas, com sujeito e verbo** — o índice é o resultado:
> "The dormant neuron phenomenon is present in deep RL agents."
> "Target non-stationarity exacerbates dormant neurons."
> "Input non-stationarity does not appear to be a major factor."
> "Dormant neurons remain dormant."
> "More gradient updates leads to more dormant neurons."
> "Dormant neurons make learning new tasks more difficult."

Convenção fortíssima e diretamente copiável: **cada bloco experimental tem por título a conclusão
que ele sustenta, e o hedge é embutido no próprio título quando o resultado é fraco** — compare
"Target non-stationarity **exacerbates**" (positivo, forte) com "Input non-stationarity **does not
appear to be** a major factor" (negativo, hedgeado). Um negativo é reportado com o mesmo destaque
tipográfico que um positivo.

Subseções da §5 são mistas — algumas nominais, outras interrogativas:
> "Consequences for Sample Efficiency" / "Impact on multi-step learning." / "Varying architectures."
> / "Varying agents." / "Learning Rate Scaling" / "**Is Over-parameterization Enough?**" /
> "Comparison with Related Methods" / "Neuron Selection Strategies"

## 4.7 Como reportam numéricos/experimentos `[ESTABELECIDO]`

> "We report results using 5 seeds, while DrQ(ϵ) use 10 seeds; error bars report 95% confidence
> intervals."
> "we report the Interquartile Mean (IQM), recommended by Agarwal et al. (2021) as a more
> statistically reliable alternative to median or mean, using 5 independent seeds for each DQN
> experiment, 10 for the DrQ and SAC experiments, and reporting 95% stratified bootstrap confidence
> intervals."

Convenções (o padrão pós-Agarwal-2021 de RL, que é o padrão que o nosso 4a.3 já segue):
- **Estimador robusto declarado e justificado por citação** (IQM, "recommended by Agarwal et al.
  (2021) as a more statistically reliable alternative to median or mean").
- **Número de seeds explícito e por família de experimento** (5 DQN / 10 DrQ e SAC).
- **Intervalo declarado com o método**: "95% stratified bootstrap confidence intervals".
- **Agregação declarada**: "aggregated across 17 Atari games and 5 runs per game".

Legenda de figura = parágrafo completo, com método + agregação + conclusão:
> "Figure 1: Sample efficiency curves for DQN, with a replay ratio of 1, when using network resets
> (Nikishin et al., 2022), weight decay (WD), and our proposed ReDo. Shaded regions show 95% CIs.
> The figure shows interquartile mean (IQM) human-normalized scores over the course of training,
> aggregated across 17 Atari games and 5 runs per game. Among all algorithms, DQN+ReDo performs
> the best."

Convenção: **a última frase da legenda é a leitura da figura** ("Among all algorithms, DQN+ReDo
performs the best") — oposto exato da legenda telegráfica de math.DS (§1.9).

Tabela de hiperparâmetros existe e é banal por design:
> "Table 1: Common Hyper-parameters for DQN and DrQ(ϵ)."

## 4.8 Como citam `[ESTABELECIDO]`

**Autor-ano (natbib/ICML)**, tanto textual quanto parentético:
> "(Mnih et al., 2015; Silver et al., 2016; Bellemare et al., 2020)"
> "recommended by Agarwal et al. (2021)"
> "network resets (Nikishin et al., 2022)"
> "Kumar et al., 2021a"

Convenções: (i) parentético para apoio de afirmação, textual quando a pessoa é o agente
("recommended by X (2021)"); (ii) sufixo a/b para mesmo autor-ano; (iii) citação colada ao termo
técnico que ela batiza ("network resets (Nikishin et al., 2022)").

## 4.9 Hedge `[ESTABELECIDO]`

Padrão explícito e disciplinado:
> "we **hypothesize** that the non-stationarity of training deep RL agents is one of the causes for
> the dormant neuron phenomenon."
> "This finding **could account for** the difficulty in training RL agents with higher replay ratios."
> "an effect we **coin** the 'dormant neuron phenomenon'."

E limitações ditas em voz ativa, no fim:
> "Although the simple approach of recycling neurons we introduced yielded good results, it is
> possible that better approaches exist."
> "ReDo reduces dormant neurons significantly but **it doesn't completely eliminate them**."
> "The dormancy threshold is a hyperparameter that requires tuning; having an adaptive threshold
> over the course of training could improve performance even further."

Conclusão (§7) reafirma exatamente o que foi *demonstrado*, com o verbo no passado:
> "In this work we identified the dormant neuron phenomenon whereby, during training, an RL agent's
> neural network exhibits an increase in the number of neurons with little-or-no activation."
> "We demonstrated that this phenomenon is present across a variety of algorithms and domains, and
> **provided evidence that** it does result in reduced expressivity and inability to adapt to new
> tasks."

Convenção: **"demonstrated" para o que foi medido, "provided evidence that" para o que foi
inferido** — dois níveis epistêmicos, na mesma frase, marcados por verbo. É exatamente a nossa
distinção fato ≠ inferência.

## 4.10 Extensão e apêndices `[ESTABELECIDO]`

Apêndices nomeados por função:
> "Appendix A: Experimental Details / Appendix B: The Dormant Neuron Phenomenon in Different
> Domains / Appendix C: Recycling Dormant Neurons / Appendix D: Performance Per Game"

Convenção: **apêndice A = reprodutibilidade (hiperparâmetros); apêndices B–D = replicação do
resultado em domínios/ablações/por-ambiente.** O corpo tem uma figura-resumo agregada; o apêndice
tem a decomposição completa por ambiente ("Performance Per Game").

---

# 5. Síntese comparativa

| Dimensão | 1511.01815 (DS) | 1603.01821 (DS) | 2108.08106 (ML teórico) | 2302.12902 (ML empírico) |
|---|---|---|---|---|
| Título | objeto + "and" + teoria | método + "for" + objeto (encurtado na revisão) | lista de resultados + objeto + restrição | "The" + fenômeno batizado + campo |
| Abstract abre com | a EDO | "In this paper we present a method..." | fato de prática + lacuna | "In this work we identify..." |
| Pessoa | 3ª → "we" na 3ª frase | "we" (autor único) | "we" | "we" |
| Hipóteses | por número de equação, "(1.3)" | "Consider ϕ∈..., Let r₂=..." | let-chain inline, sem siglas | definição operacional com limiar τ |
| Constantes | "sufficiently small" (anônimas) | expoentes explícitos (ε^{1/(2n−1)}) | rate 1, ordem explícita | τ, nº de seeds, IC 95% |
| Provas | seções-passagem + §técnica final | "Proof." + estratégia; **apêndice A = prova** | "Proof of Prop 2.2." + fecho ritual | n/a |
| Apêndice | não | sim (uma prova) | não | sim (A–D: detalhes + réplicas) |
| Citações | `[12, 9, 14, 4]` | `[33, Proposition 6.3]` | `[24, Proposition 2.2]`, "e.g., in [...]" | (Mnih et al., 2015) |
| Legenda de figura | telegráfica, aponta equações | descritiva | (sem figuras) | parágrafo com método + conclusão |
| Hedge | quase zero; só sobre história | sobre escopo ("We do not aim to") | sobre programa e glosa | "hypothesize", "could account for" |
| Extensão | ~19 pp `[NÃO VERIFICADO]` | ~47 pp `[NÃO VERIFICADO]` | **30 pp** (Comments) | ICML + apêndices A–D |

## 5.1 Convenções que os QUATRO compartilham `[ESTABELECIDO]`

1. **Nenhum título tem dois-pontos, subtítulo ou adjetivo de valor.** Quatro títulos, zero
   "Towards", "A Novel", "Rethinking", "Unified".
2. **Nenhum abstract contém uma comparação com concorrentes.** Nenhum diz "outperforms".
   (2302.12902 diz "results in improved performance" — melhora, sem alvo nomeado.)
3. **O objeto de estudo é nomeado e a nomeação é assumida**: "property ℱ", "flat slow manifolds",
   "τ-dormant". Todos criam ou herdam um nome curto e o usam consistentemente.
4. **Escopo negativo dito explicitamente**: (c) earthquake faulting não demonstrado; "with ReLU
   activation" no título; "does not appear to be a major factor"; "assumptions ... are there to
   simplify the proof".
5. **Cada afirmação forte vem com sua condição colada** — por número de equação, por let-chain,
   por τ, ou por "under the assumption that".

## 5.2 Onde os dois mundos DIVERGEM (e nosso paper vive nos dois) `[ESTABELECIDO]`

- **Abertura**: math.DS abre com a equação; ML abre com o funil de campo + "However,".
- **Citação**: math.DS/ML-teórico usam `[n]` com localizador; ML empírico usa autor-ano.
- **Legenda**: math.DS = ponteiro; ML = parágrafo autocontido com a conclusão no fim.
- **Heading**: math.DS = frase nominal do objeto ("Passage by the line of saddles"); ML empírico =
  frase declarativa com verbo ("Dormant neurons remain dormant.").
- **Hedge**: math.DS proíbe hedge sobre matemática, permite sobre escopo/história;
  ML empírico usa "hypothesize"/"could account for" como marcadores epistêmicos de rotina.

---

# 6. Aplicação ao nosso paper `[ESPECULAÇÃO — inferência nossa a partir do observado]`

O paper-v1 é híbrido (teorema T2′ + escada experimental toy→MLP→transformer), portanto o espelho
correto é: **estrutura e enunciado de 2108.08106, retórica de resultado de 1511.01815,
disciplina de escopo de 1603.01821, e reporte experimental de 2302.12902.** Concretamente:

- Título sem dois-pontos, taxonômico, terminando na restrição real do resultado
  (o análogo de "with ReLU activation").
- Abstract: prática → lacuna → o que estabelecemos, **cada resultado com sua hipótese colada**,
  e a natureza P5/contato dita como uma frase estrutural, nunca como resultado.
- T2′ enunciado na introdução como consequência declarada de um teorema técnico posterior
  ("Theorem 1.1 is a direct consequence of Theorem 3.3 below").
- Um Remark logo após T2′ dizendo **qual hipótese é essencial e qual é comodidade** (Remark 1 de
  De Maesschalck–Schecter é o modelo literal).
- Subseções experimentais com título-conclusão, incluindo os negativos com o mesmo destaque
  (nossos dois negativos de instrumento e a falsificação SGLD ganham heading declarativo).
- Reporte: seeds por família, estimador robusto nomeado + citado, IC com método explícito,
  legenda de figura com a leitura no fim.
- "demonstrated" só para o medido; "provided evidence that" para o inferido; nunca as duas
  categorias na mesma oração.
- Placeholders E5-PENDING **não** podem virar hedge no abstract: ou o resultado entra com sua
  condição, ou fica fora (padrão de escopo do Kristiansen: (c) foi citado e explicitamente excluído).

---

# 7. Procedência (o que foi aberto nesta sessão)

- `arxiv.org/abs/1511.01815`, `arxiv.org/abs/1511.01815v1`, `export.arxiv.org/abs/1511.01815`,
  `ar5iv.labs.arxiv.org/html/1511.01815` (3 passagens com prompts distintos)
- `arxiv.org/abs/1603.01821`, `www.arxiv.org/abs/1603.01821v3`,
  `ar5iv.labs.arxiv.org/html/1603.01821` (2 passagens)
- `arxiv.org/abs/2108.08106`, `ar5iv.labs.arxiv.org/html/2108.08106` (2 passagens)
- `arxiv.org/abs/2302.12902`, `export.arxiv.org/abs/2302.12902`,
  `ar5iv.labs.arxiv.org/html/2302.12902` (2 passagens)
- Buscas (só para page-range publicado, etiquetado `[NÃO VERIFICADO]`): JDE 260(8):6697–6715;
  Nonlinearity 30(5):2138–2184.

Não abertos: PDFs completos, páginas de editor (ScienceDirect / IOP / PMLR), bibliografias
integrais. Contagens de referência de 2108.08106 ([1]–[41]) vêm do render ar5iv.
