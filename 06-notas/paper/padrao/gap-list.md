# gap-list — auditoria de conformidade editorial de `paper-v1.tex`

Data: 2026-07-25. Arquivo auditado: `/home/caloi/Projects/j-space/06-notas/paper/paper-v1.tex`
(633 linhas, 11 pp compiladas, 11 referências, 1 teorema, 1 definição, 6 lemas, 3 figuras, 1 tabela).

Padrão de referência: os três dossiês em `06-notas/paper/padrao/`
(`padrao-venues-ds.md` = **[V]**, `padrao-ml-theory.md` = **[M]**, `padrao-espelhos.md` = **[E]**).
Toda regra citada abaixo aponta para a seção do dossiê que a sustenta; onde a regra é
`[PRATICA COMUM]` (frequência medida, não norma escrita), isso está dito.

**Nenhuma edição foi feita no paper.** Este arquivo é só o diagnóstico.

## Como ler a gravidade

- **BLOQUEIA-SUBMISSÃO** — o item impede a submissão/circulação, quebra a compilação no destino,
  ou é uma falha de integridade (afirmação sem suporte, referência não verificável, número
  inconsistente entre partes do texto). Nada disso pode sair da máquina como está.
- **FORTE** — não bloqueia, mas um revisor da área vai marcar. Divergência de convenção
  documentada, escopo mal declarado, ou lacuna que o checklist normativo cobra por escrito.
- **COSMÉTICO** — forma. Custa minutos, e a soma deles é o que separa "escrito por alguém da
  área" de "gerado".

Contagem: **14 BLOQUEIA-SUBMISSÃO · 23 FORTE · 17 COSMÉTICO** (54 itens).

---

# BLOQUEIA-SUBMISSÃO

## B1. Caminhos de figura com `../` — o pacote arXiv não compila

**Trecho atual (linhas 162, 307):**
```latex
\includegraphics[width=0.60\linewidth]{../t2-prova/fig1-mecanismo.pdf}
\includegraphics[width=0.56\linewidth]{../../05-experimentos/etapa4-mlp-populacao/fig-diagnostico-duracoes.pdf}
```

**Padrão violado:** a submissão do arXiv é um tarball com raiz única; caminhos que sobem de
diretório (`..`) não resolvem no servidor. [V §12] registra `info.arxiv.org/help/prep.html`
como fonte lida sobre preparação de submissão. O `.tex` compila localmente e **falha no destino** —
é o modo de falha mais barato de evitar e o mais caro de descobrir tarde.

**Correção concreta:** copiar os três PDFs para o diretório do paper com nomes achatados e
referenciá-los sem caminho:

```bash
cp ../t2-prova/fig1-mecanismo.pdf        fig1-mecanismo.pdf
cp ../t2-prova/fig2-constantes.pdf       fig2-constantes.pdf
cp ../../05-experimentos/etapa4-mlp-populacao/fig-diagnostico-duracoes.pdf fig3-diagnostico.pdf
```
e no `.tex`: `\includegraphics[width=0.6\linewidth]{fig1-mecanismo}` (sem extensão, sem caminho).
Acrescentar ao `figuras_paper.py` / `figuras_draft.py` a cópia automática, para a figura no paper
nunca divergir da figura gerada.

---

## B2. O paper enuncia um teorema e não contém prova nenhuma

**Trecho atual:** `\begin{theorem}` (linha 138) sem `\begin{proof}`; §3 tem um
`\paragraph{Proof outline.}` (linha 173); o Apêndice A (linha 526) chama-se "The six lemmas" e
contém **enunciados** de L1–L6 com métricas de verificação numérica, **sem um único argumento**.
`grep -c 'begin{proof}' = 0`.

**Padrão violado:**
- [M §5.4] checklist NeurIPS item 3, texto literal: "did you state the **full set of assumptions**
  of all theoretical results, and did you include **complete proofs**?"
- [M §4.5] ML Reproducibility Checklist v2.0: "For any theoretical claim: A clear statement of the
  claim; **A complete proof of the claim**."
- [V §6] o ambiente `proof` aparece em **17/25** das notas curtas de math.DS medidas.
- [E §1.7 / §2.6 / §3.5] os três espelhos matemáticos põem a prova ou em seções-passagem (§2.3–2.5
  de De Maesschalck–Schecter), ou em apêndice nomeado ("Appendix A **Proof of Theorem 3.3** using
  blowup", Kristiansen), ou na escada de seções (Eberle et al., §§2–5). **Nenhum enuncia sem provar.**

Isto contamina o abstract: "we **prove** that ... death and resurrection are governed by a hybrid
system with memory" (linha 30) é uma afirmação que o documento não sustenta. É o item que um
revisor de math.DS devolve primeiro.

**Correção concreta — duas saídas, escolher uma antes de circular:**

1. **(recomendada)** Apêndice A vira `Proofs of Lemmas 1–6` e o §3 ganha
   `\begin{proof}[Proof of Theorem 1]` costurando os seis lemas (é o idioma de Kristiansen §2.6:
   "Proof. We demonstrate this result by considering our model system (3.16)." → fecho por
   reconciliação + `\qed`). O material já existe nos scripts `prova_L1--L6.py`; o que falta é a
   redação.
2. **(mínima, se v1 tem de sair antes)** rebaixar o registro por escrito, em três lugares:
   - abstract: trocar `we prove that` por `we derive, and verify step by step, that`;
   - `\begin{theorem}` → `\begin{claim}` ou manter `theorem` mas inserir imediatamente após ele:
     > *Status of the proof.* The argument is organized as the six lemmas of Appendix~A. This note
     > states the lemmas and reports, for each, an independent symbolic/numerical verification
     > (scripts `prova_L1`–`prova_L6`); the complete written proofs are deferred to a companion
     > note. We flag this explicitly rather than let "Theorem" carry more weight than the present
     > document supports.
   - `\paragraph{Proof outline.}` → `\paragraph{Structure of the argument.}`

Não misturar as duas: ou prova, ou o aviso. Enunciar como teorema, chamar de "proof outline" e
não provar é a combinação que não sobrevive à leitura.

---

## B3. Bloco de autor sem afiliação e sem e-mail; nome divergente

**Trecho atual (linha 20):**
```latex
\author{L.~Caloi Santos}
```

**Padrão violado:** [V §3] — o conteúdo exigido pelos três venues é sempre *nome, afiliação onde o
trabalho foi feito, e-mail de pelo menos um autor, e o financiamento*.
- IOP: `\ead{}` com o e-mail de pelo menos um autor `[VERIFICADO em IOP LaTeX guidelines §3.2]`;
- Elsevier/Physica D: autor correspondente explicitamente indicado com e-mail;
- SIAM: endereço + e-mail de cada autor no `\thanks` do próprio autor.

Hoje o bloco não tem nenhum dos três. Além disso o nome no `.tex` (`L. Caloi Santos`) não bate com
o nome usado no repositório (`Leonardo Caloi`, git user) — para indexação (MathSciNet/zbMATH/arXiv
author ID) o nome tem de ser **um só**, escolhido agora e mantido.

**Ressalva de honestidade:** os três dossiês **não mediram** como pesquisadores independentes se
apresentam — não há amostra de afiliação "independent" em nenhum deles. Logo, a *obrigação* de
afiliação+e-mail é `[VERIFICADO]`, mas a string exata "Independent researcher" é
`[NÃO VERIFICADO neste levantamento]`: é forma corrente, não medida aqui. Se isso importar, vale
uma passada de verificação sobre autores sem vínculo em math.DS antes de fixar.

**Correção concreta (article/amsart, satisfaz IOP e Elsevier de uma vez):**
```latex
\author{Leonardo Caloi Santos\thanks{Independent researcher, Maring\'a, Paran\'a, Brazil.
\texttt{<e-mail>}. No funding was received for this work.}}
```
Decisões que só você pode tomar: (a) forma canônica do nome; (b) qual e-mail (o do repositório é
pessoal — se for circular, considerar um endereço dedicado); (c) cidade a declarar.
A frase de financiamento vai aí se o alvo for SIAM, e migra para `Acknowledgments` se for
IOP/Elsevier [V §11].

---

## B4. `\date` proíbe a circulação do documento que se quer circular

**Trecho atual (linha 21):**
```latex
\date{2026-07-25 \quad DRAFT v1 --- not for circulation}
```

**Padrão violado:** [V §3] — "Colocar 'DRAFT — not for circulation' na linha de data é uso interno,
não é convenção de venue — **nenhuma amostra tinha isso**" (`[PRATICA COMUM observada]`, 47 fontes).
E é autocontraditório com o objetivo declarado (draft circulável).

**Correção concreta:** `\date{}` (padrão amsart: data vazia, o carimbo do arXiv fala) ou
`\date{July 25, 2026}`. Se quiser marcar o estágio sem carimbar "não circule", o lugar é o campo
Comments do arXiv ("Preprint; comments welcome") — [V §11] mede 15/282 entradas de math.DS usando
"comments welcome".

---

## B5. Colisão de notação: `T` = horizonte do teorema **e** `T` = temperatura

**Trechos atuais:**
- linha 41 (abstract): "a confirmatory MLP run ($400$ runs, **$T=0$**)" — `T` aparece no abstract
  **sem definição**;
- linha 124 (H2): "$A^{\mathrm{hyb}}$ has a first zero at $\texit<T$", linha 127 (H4): "no other
  switching event occurs in $[0,T]$", linha 139: `\sup_{t\in[0,T]...}` — `T` é o horizonte de tempo;
- linhas 298, 318, 328, 349, 400, 409, 451, 483: "$T>0$", "$T=10^{-5}$", "$T=0$", "the $T=0$ arm" —
  `T` é temperatura de ruído do SGLD.

**Padrão violado:** [V §4] o abstract tem de ser *stand-alone* nos três venues (SIADS: "the abstract
must be able to stand independently"; IOP: "self-contained"). Um símbolo sem definição no abstract
já falha. E [E §5.1] a convenção compartilhada pelos quatro espelhos é "o objeto de estudo é nomeado
e a nomeação é assumida ... todos criam ou herdam um nome curto e o usam **consistentemente**".
Um `T` com dois significados no mesmo paper é erro de manuscrito, não de estilo.

**Correção concreta:** manter `T` só para o horizonte do teorema e trocar a temperatura por um
símbolo livre — o projeto já usa `\eta` para taxa e `\beta` para rigidez, então:
- definir uma vez em §5: "we write $\sigma$ for the SGLD noise scale (the *temperature*);
  $\sigma=0$ is deterministic gradient descent";
- substituir todo `$T=0$` → `$\sigma=0$`, `$T>0$` → `$\sigma>0$`, `$T=10^{-5}$` → `$\sigma=10^{-5}$`;
- no abstract, escrever por extenso e sem símbolo: "a confirmatory MLP run (400 runs, **noiseless
  gradient descent**)".

Verificar também `H` (horizonte em passos, linhas 331/351/444) contra `H^\ell` — não colide hoje,
mas declarar `H` explicitamente na primeira ocorrência.

---

## B6. Colisão de notação: `A` = neurônio **e** `A(t)` = área; `\Phi` com três sentidos

**Trechos atuais:**
- linha 109: "Fix one neuron $A$ and one datum $j$" → depois linha 118:
  "$A^{\mathrm{hyb}}(t)=\int_{\thit}^t R^{\mathrm{hyb}}$ for its signed area", linha 428:
  "$A_{\mathrm{drive}}$". O mesmo `A` é o neurônio, a área e a área ponderada.
- linha 48: "$\Phi:\mathcal{W}\to\mathcal{F}$ for the parametrization"; linha 180 e 532:
  "$\Phi(u)=\log u+u$"; linhas 226/228: "$\Phi_{\mathrm{gauss}}$". Três objetos, um símbolo.
  (E `CLAUDE.md` reserva `Φ` para a parametrização — a colisão é também interna ao projeto.)

**Padrão violado:** mesma fonte de B5 ([E §5.1]; [V §4] stand-alone). Em [E §1.5] a convenção de
De Maesschalck–Schecter é isolar e batizar o objeto técnico com símbolo próprio ("property $\mathcal F$").

**Correção concreta:**
- neurônio: `A` → `\ast` indexado, i.e. "fix one unit $k^\ast$ and one datum $j$"; todas as
  ocorrências `\theta_A`, `f_{-A}`, `z=\theta_A\cdot(x_j,1)` viram `\theta_{k^\ast}`, `f_{-k^\ast}`;
- área: manter `A(t)`, `A^{\mathrm{hyb}}(t)`, `A_{\mathrm{drive}}` (é o nome que o resultado carrega
  — "the area law");
- quadratura: `\Phi(u)=\log u+u` → `\Psi(u)=\log u+u` (linhas 180 e 532);
- gaussiana: `\Phi_{\mathrm{gauss}}` → `\mathcal N` ou `\mathrm{erf}`-based, e definir na primeira
  ocorrência (hoje entra em §3 sem definição).

---

## B7. `p=0.50` no abstract/intro vs `0.457` na tabela — dois condicionamentos, um símbolo

**Trechos atuais:**
- abstract (linha 41): "reproduces resurrection at **$p=0.50$**";
- C1 (linha 379–381): "$p(\mathrm{ret}\mid\mathrm{eligible})=0.500\ [0.313,0.687]$";
- Tabela 1 (linha 364): coluna "$p(\mathrm{ret}\mid\mathrm{death})$ CP" → sp-$\beta3$ = **0.457**.

O leitor que for do abstract à tabela encontra 0.50 e 0.457 para o mesmo braço e conclui que um dos
dois está errado. Não está — são condicionamentos diferentes (elegíveis vs. todas as mortes) — mas
**isso não é dito em lugar nenhum do abstract nem da intro**, e a definição de "elegível"
(janela $[0,307{,}536]$) aparece só na linha 353.

**Padrão violado:** [M §5.4] checklist NeurIPS item 1, literal: "Do the main claims made in the
abstract and introduction **accurately reflect the paper's contributions and scope**?"
[E §3.3] convenção de Eberle–Jentzen–Riekert–Weiss: "no abstract, **cada resultado principal vem com
sua hipótese colada**" — nunca um número sem sua condição.

**Correção concreta (abstract):**
> "...and a confirmatory MLP run (400 runs, noiseless gradient descent) reproduces resurrection in
> half of the deaths that occur early enough to be observable within the horizon
> ($p=0.50$, 15/30), with the residual area opening before 13 of 16 returns, hard ReLU giving zero,
> and a monotone channel staircase in the activation stiffness."

E na Tabela 1, renomear as colunas para que os dois estimandos coexistam sem ambiguidade:
`$\hat p_{\rm death}$` e `$\hat p_{\rm elig}$`, com a definição de elegibilidade na legenda.

---

## B8. Os cinco `[E5-PENDING]` são marcadores de TODO interno visíveis ao leitor

**Trecho atual (linhas 454–471):** cinco itens da forma
```latex
\item \textbf{[E5-PENDING: E-A validity]} --- any sustained return in relu-frozen invalidates the run
```
mais a linha 451: "**is running at the time of writing**; the following five estimands and their
results are pending", e a última frase do abstract (linha 43): "A transformer confirmatory run is
in progress."

**Padrão violado:** [E §6], precedente literal de Kristiansen: as três aplicações são rotuladas
(a)(b)(c) no abstract e a última frase diz **quais foram feitas e qual não foi** — (c) é citada como
motivação e **explicitamente excluída**. Conclusão do dossiê, textual: *"Placeholders E5-PENDING
**não** podem virar hedge no abstract: ou o resultado entra com sua condição, ou fica fora."*
Nenhum dos quatro espelhos reporta **status de execução** de um job; papers não têm campo de
progresso. E [M §9.2] mostra que o idioma correto para protocolo-sem-dados é **tempo futuro**
("We plan to conduct...", "we will train and test on...") numa seção nomeada como protocolo — foi
assim que os dois papers pré-registrados do PMLR v148/v181 fizeram.

**Correção concreta:**

1. Renomear a subseção: `\subsection{Transformer: two new mechanisms, and a pre-registered protocol}`
2. Trocar o parágrafo de status (linhas 448–452) por:
   > "A confirmatory run for the transformer was pre-registered at commit `1b45498` (800 runs;
   > $\{$sp-$\beta3$, sp-$\beta10$, GELU, ReLU$\}\times\{$attn-train, attn-frozen$\}\times$
   > injection $\{-0.3,-2.0\}\times50$ seeds 2000--2049; $H=2\cdot10^5$, noiseless).
   > **Its results are not part of this note.** We state the five pre-registered estimands here so
   > that the protocol is on record ahead of the data, and report them in a separate note."
3. Trocar cada `\textbf{[E5-PENDING: E-A validity]}` por `\textbf{E-A (validity).}` e pôr o item em
   **futuro**: "any sustained return in relu-frozen **will invalidate** the run (theorem:
   own-gradient $\equiv0$ and fixed input $\Rightarrow$ terminal); the pilot gave 0/40."
4. Abstract: trocar "A transformer confirmatory run is in progress." por uma declaração de escopo no
   molde Kristiansen:
   > "The transferability of the law to a transformer FFN is stated as a pre-registered protocol and
   > is **not** established here."

Assim o draft fica circulável sem uma única marca de rascunho, e a honestidade fica **mais** visível,
não menos.

---

## B9. Referência `[GH]` não é verificável e mistura duas obras numa entrada

**Trecho atual (linhas 605–607):**
```latex
\bibitem{GH} N.~Guglielmi, E.~Hairer, \emph{On the limit solution of regularized
piecewise-smooth dynamical systems}, preprint, Univ.\ Geneva, 2019; foundational
results in SIAM J.~Appl.~Dyn.~Syst.\ 14 (2015) 1454--1477.
```
Um "preprint, Univ. Geneva, 2019" sem DOI, sem arXiv-id e sem número de relatório não é
localizável; e a entrada emenda **duas obras diferentes** ("foundational results in ...") sob uma
chave só, citada no corpo como `~\cite{Jef16,GH}` (linha 191) sem localizador.

**Padrão violado:**
- [V §9] SIADS: **"Authors are required to provide DOIs."**
- [V §9] Elsevier: "Preprints should be clearly marked as such ... **The preprint DOI should also be
  provided.**"
- [V §9] IOP §6.7: "**if the article concerned has been published in a peer-reviewed journal, that
  reference should take precedence**. If only a preprint reference can be given, it is helpful to
  include the article title."
- [E §2.7] Kristiansen: nunca citar "[33]" seco quando se usa uma proposição de [33] —
  `[33, Proposition 6.3]`.
- E `CLAUDE.md`/skill `verificacao-fontes`: nada entra sem fonte aberta lida.

**Correção concreta:** abrir a fonte e desmembrar em duas entradas com DOI:
```latex
\bibitem{GH15} N.~Guglielmi, E.~Hairer, \emph{Classification of hidden dynamics in
discontinuous dynamical systems}, SIAM J. Appl. Dyn. Syst. 14 (2015) 1454--1477.
doi:10.1137/15M1013969.
\bibitem{GH19} N.~Guglielmi, E.~Hairer, \emph{<título exato da fonte aberta>},
<periódico/preprint com arXiv-id ou DOI>, 2019.
```
**Atenção:** o título que coloquei em `GH15` vem da minha memória do par (Guglielmi–Hairer, SIADS 14,
2015) e está `[NÃO VERIFICADO]` — **abrir a página do DOI antes de usar**. O que esta auditoria
estabelece é só que a entrada atual, como está, não é verificável.

---

## B10. Termos técnicos centrais usados sem nenhuma citação

**Trechos atuais:** "the frozen **Filippov** branch" (linha 593), "**SGLD**" (linhas 83, 262, 291,
484), "**Clopper--Pearson**" (linhas 372, 384, 404), "**GELU**" (primeira ocorrência linha 57),
"the **Brownian-cone** prediction" (linha 274), "bootstrap clustered by seed" (linha 372).
Nenhum tem referência.

**Padrão violado:** [E §3.6] convenção de Eberle et al.: citação numérica **com localizador** sempre
que se *usa* um resultado, e `e.g., in [...]` para literatura adjacente. [E §1.8] em
De Maesschalck–Schecter, a frase carrega o conteúdo e a citação localiza — mas ela *existe*.
Um paper de math.DS que escreve "the frozen Filippov branch" sem citar Filippov é o tipo de coisa
que faz o revisor duvidar do resto.

**Correção concreta:** acrescentar à bibliografia (todas a **verificar na fonte** antes de entrar —
a lista abaixo é a de lacunas, não referências prontas):
Filippov (1988, *Differential Equations with Discontinuous Righthand Sides*, Kluwer);
Welling & Teh (ICML 2011) para SGLD; Clopper & Pearson (Biometrika 26, 1934) para o intervalo;
Hendrycks & Gimpel (arXiv:1606.08415) para GELU; Krupa & Szmolyan para blow-up (já é o vocabulário
de [E §2.3], onde Kristiansen credita "in the formulation of Krupa and Szmolyan" **no abstract**).
Para o "Brownian-cone", ou entra a fonte, ou a expressão vira uma frase autocontida do que foi
calculado.

---

## B11. Legendas de CI sem estimador e sem fator de variabilidade nos números do corpo

**Trechos atuais:**
- linha 380: "$p(\mathrm{ret}\mid\mathrm{eligible})=0.500\ [0.313,0.687]$" — nenhum estimador nomeado;
- linha 364–367: coluna "$\lambda$/run [boot]" com `0.350 [0.220,0.500]` — **`\lambda` nunca é
  definido no paper**, "[boot]" não diz o número de reamostragens nem o nível;
- linha 384: "lower Clopper--Pearson bound $0.544>0.5$" — aqui o estimador é nomeado (bom), mas não
  se diz sobre o que ele varia.

**Padrão violado:** [M §8.1], guidelines literais do item 7 do checklist NeurIPS:
"**The factors of variability that the error bars are capturing should be clearly stated**";
"The method for calculating the error bars should be explained"; "It is OK to report 1-sigma error
bars, but **one should state it**". E [M §4.5], checklist v2.0: "The exact number of training and
evaluation runs" e "A clear definition of the specific measure or statistics used".
[E §4.7] Sokar et al. é o modelo: "we report the Interquartile Mean (IQM), recommended by Agarwal et
al. (2021) ... using 5 independent seeds for each DQN experiment, 10 for the DrQ and SAC
experiments, and reporting **95% stratified bootstrap confidence intervals**."

**Correção concreta:** um parágrafo de instrumento estatístico, uma vez, no início de §5, e todos os
números passam a herdá-lo:
> "**Statistics.** Proportions are reported as point estimate with a two-sided 95\% Clopper--Pearson
> interval, whose variability is over the Bernoulli count of returns among deaths within an arm
> ($n$ given in each cell). Event rates $\lambda$ (deaths per run) are reported with a 95\%
> percentile bootstrap interval over $B=10^4$ resamples **clustered by seed**, so that the
> variability captured is seed-to-seed, not episode-to-episode. No aggregate is pooled across arms."

e definir `\lambda` na legenda da Tabela 1: "$\lambda$: dynamical deaths per run".

---

## B12. Bibliografia sem DOI e em ordem que não é nem alfabética nem de aparição

**Trecho atual (linhas 600–631):** 11 entradas, **zero DOIs**. Ordem da lista: DMS16, Kri17, GH,
Jef16, SDR23, ST98, LST07, EJRW23, Sokar23, Lyle23, Dohare24. Ordem de **primeira citação** no texto:
Sokar23, Lyle23, Dohare24 (linha 55) → EJRW23 (77) → DMS16 (178) → Jef16, GH (191) → Kri17 (192) →
ST98, LST07 (193) → SDR23 (196).

**Padrão violado:** [V §9] — Physica D: "**Number the references in the order in which they appear
in the text**"; SIADS: "References should be listed in **either alphabetical order or order of
citation**"; SIADS: "Authors are required to provide DOIs". A lista atual não satisfaz nenhuma das
duas ordens admitidas.

**Correção concreta:** reordenar por primeira citação (satisfaz Physica D e SIADS de uma vez) e
acrescentar DOI a cada entrada. Duas entradas incompletas mesmo depois disso:
- `Kri17` — "Nonlinearity (2017)" sem volume/páginas, quando [E §2.1] registra
  **Nonlinearity 30(5) (2017) 2138–2184** (etiquetado `[NÃO VERIFICADO]` no dossiê: veio de snippet;
  confirmar no DOI antes de colocar). IOP §6.7 exige que a referência publicada tenha precedência.
- `Sokar23`, `Lyle23` — "ICML 2023" deve virar PMLR volume + páginas.

---

## B13. Contadores de teorema independentes: "Theorem 1" e "Definition 1" coexistem

**Trecho atual (linhas 9–11):**
```latex
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}{Lemma}
\newtheorem{definition}{Definition}
```
No PDF compilado: **Definition 1**, **Theorem 1**, **Lemma 1**…**Lemma 6** — três "1" no mesmo
documento.

**Padrão violado:** [V §6], que trata isso como divergência dura e já diagnosticou este arquivo:
AMS numera por seção com contador **compartilhado** (`\newtheorem{lemma}[theorem]{Lemma}`); SIAM
numera em **sequência única** compartilhada (`siamart220329.cls`, linhas 1448/1458). "Isso não
corresponde nem ao padrão AMS nem ao SIAM." Prática medida: contador compartilhado em **14/25**
das notas curtas de math.DS.

**Correção concreta (mantendo `article` + sequência única, estilo SIAM):**
```latex
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}[theorem]{Lemma}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}
```
Resultado: Definition 1, Theorem 2, Lemmas 3–8. Ajustar as 20+ referências `\ref` é automático.
(O `remark` é necessário para B14 e F5.)

---

## B14. Nome interno do projeto vazado para dentro do teorema

**Trecho atual (linha 138):**
```latex
\begin{theorem}[$T2'$-min]\label{thm:main}
```
`T2'` é o código do teorema nas notas internas (aparece nos commits: "teorema T2′ (espinha do
draft-v3)"). O leitor externo não tem como saber o que é, e "-min" sugere que existe uma versão
maior não mostrada.

**Padrão violado:** [V §6] o argumento opcional do ambiente **nomeia** o teorema, e o exemplo do
`docsiamart.tex` é `\begin{theorem}[Mean Value Theorem]` — um nome que o leitor reconhece.
[E §5.1] "o objeto de estudo é nomeado e a nomeação é assumida": os quatro espelhos criam ou herdam
um **nome curto de conteúdo** ("property $\mathcal F$", "flat slow manifolds", "$\tau$-dormant").

**Correção concreta:** `\begin{theorem}[Entry--exit law for the smoothed dead unit]\label{thm:main}`
— e uma varredura por outros vazamentos internos: `sec:falsif`, `sec:negativos`, `sec:mecanismo`,
`sec:confirm` (labels em português num paper em inglês; ver C11), "etapa-4a", "etapa-5",
"piloto-de-taxa".

---

# FORTE

## F1. O intro promete mais do que o teorema entrega (classe monotone-sigmoid)

**Trecho atual (linhas 67–70):** "the law is independent of the smoothing within the
**monotone-sigmoid class** (softplus, erf)" — apresentado como parte do teorema.
Mas §2 define o modelo só com softplus (linha 102) e §3 trata erf num
`\paragraph{Regularization independence.}` cujo suporte é "(verified)" numérico (linha 227), não
uma prova.

**Padrão violado:** [M §5.4] item 1 do checklist NeurIPS (claims vs. scope) e [E §3.3]
(cada resultado com sua hipótese colada).

**Correção concreta:** no bullet da intro, separar os dois registros —
"...and, **for the erf smoothing, we verify numerically** that the release criterion is unchanged;
the proof is written for softplus."

## F2. "Shallow softplus networks" descreve mais do que o modelo é

**Trecho atual (linha 102):** `f_\theta(x)=\sum_k g_\beta(w_kx+b_k)` — **sem pesos de saída**, com
$x$ escalar. O abstract (linha 30) e a intro dizem "shallow softplus networks", que o leitor lê como
$\sum_k a_k g_\beta(w_k^\top x + b_k)$ com $a_k$ treináveis e $x\in\R^d$.

**Padrão violado:** [E §3.2] o título/abstract de Eberle et al. termina cravando a limitação técnica
("with ReLU activation"); [E §5.1] convenção compartilhada #4: "escopo negativo dito
explicitamente". [M §5.4] item 1.

**Correção concreta:** no abstract, "for a shallow network with scalar input and **unit output
weights**"; e em §2, uma frase dizendo que a restrição é essencial ou de conveniência — que é
exatamente o Remark 1 de De Maesschalck–Schecter ([E §1.6]: "The assumptions ... are there to
simplify the proof").

## F3. Falta o Remark que separa hipótese essencial de comodidade técnica

**Trecho atual:** (H1)–(H4) (linhas 122–127) entram como `\paragraph{Hypotheses.}` e nunca mais são
qualificadas. O leitor não sabe se (H4) ("no other switching event occurs in $[0,T]$") é a alma do
resultado ou um andaime.

**Padrão violado:** [E §1.6], convenção classificada no dossiê como "forte e diretamente aplicável a
nós": **dizer qual hipótese é essencial e qual é comodidade técnica, por escrito, imediatamente após
o teorema** (Remark 1 de De Maesschalck–Schecter, citado literal).

**Correção concreta:** logo após o teorema:
```latex
\begin{remark}
(H1)--(H2) are essential: they are the entry--exit geometry itself, and the law fails without the
sign change of $R^{\mathrm{hyb}}$. (H3)--(H4) are there to simplify the proof: (H3) makes the other
data contribute $O(e^{-\beta\delta_1})$ and (H4) isolates a single episode. We expect both to be
removable at the cost of bookkeeping; we have not done it.
\end{remark}
```

## F4. Prosa e comentário dentro do ambiente `theorem`

**Trecho atual (linhas 148–157):** dentro do `\begin{theorem}` há sete linhas de interpretação —
"The $\log\beta$ in the sup is sharp and is carried solely by...", "each corner contributes the
reciprocal of its transversal speed", "**The remainder is $o(\beta^{-1})$ as proved.**",
"the limits do not commute, §4".

**Padrão violado:** [E §1.6] o comentário mecânico vem **depois** do teorema, como frase de
interpretação e como Remark — nunca dentro do enunciado. [E §3.4] o enunciado de Eberle et al. é
"Let ... Then (i) ... (ii) ...", puro. "As proved" dentro de um teorema que não tem prova (B2) é o
pior caso.

**Correção concreta:** cortar o enunciado em: (i) hipóteses + as duas conclusões (o `sup` e a lei de
release), em dois displays **numerados**; (ii) um `\begin{remark}` com todo o resto.

## F5. Zero equações numeradas em 11 páginas

**Trecho atual:** `grep -c 'begin{equation}' = 0`, `label{eq...} = 0`. Todos os displays são `\[ \]`,
inclusive o do teorema (linhas 140–146), que empilha **duas fórmulas independentes** separadas por
`\qquad`.

**Padrão violado:** [V §7] mediana de **21 equações rotuladas por paper** na amostra de 47 fontes.
[E §1.4] em De Maesschalck–Schecter a hipótese estrutural **recebe número próprio** ((1.3))
justamente para ser citada por número nos teoremas — é o mecanismo de hipóteses nomeadas do gênero.
Sem numeração, (H1)–(H4) e a lei de release não são citáveis por um revisor ("na sua eq. (7)...").

**Correção concreta:** numerar, no mínimo: a rede (§2), a definição de $R(t)$, as quatro hipóteses,
e **separar** as duas conclusões do teorema em (1) e (2).

## F6. Legenda de tabela embaixo da tabela

**Trecho atual (linhas 357–374):** `\begin{table}` → `\centering` → `\small` → `tabular` →
**`\caption`** → `\label`.

**Padrão violado:** [V §8] "Legenda de tabela vai **EM CIMA**" `[VERIFICADO em docsiamart.tex
§"Tables"]`. [M §1.5] NeurIPS, literal: "The table number and title always appear **before** the
table". (Figura é o contrário, e as três figuras do paper já estão certas.)

**Correção concreta:** mover `\caption{...}\label{tab:4a3}` para **antes** do `\begin{tabular}`.

## F7. Não existe seção de Related Work nem parágrafo de literatura

**Trecho atual:** a literatura entra só como citações soltas na intro (linha 55) e no
`\paragraph{Proof outline.}` (linhas 190–196).

**Padrão violado:** [M §11 item 4] e [M §2.2] — nos quatro papers medidos há uma seção de Related
Work, em §2 (empíricos) ou tarde, §6/§9 (quando a espinha é um teorema). Como este paper é híbrido e
tem teorema, o padrão observado manda **tarde**.

**Correção concreta:** criar `\section{Related work}` entre §4 e §5, com três parágrafos:
(i) entry–exit/GSPT e por que o caso degenerado não é coberto (o material da linha 190–196 já é
isso, e está enterrado num `\paragraph`); (ii) fluxos de gradiente com ReLU (EJRW23 com localizador);
(iii) dormant neurons / loss of plasticity. Ganho colateral: `\paragraph{Proof outline.}` volta a ser
só a prova.

## F8. Não existe Conclusion

**Padrão violado:** [V §10] seção de conclusão/discussão/outlook em **13/22** dos publicados nos três
venues; "Conclusion(s)" em 3/25 das notas curtas. [M §2.3] o esqueleto do exemplar mais próximo
(Carroll et al., ICML 2025) tem `7. Conclusion` depois de `6. Limitations`.

**Correção concreta:** `\section{Conclusion}` de um parágrafo, no idioma de [E §4.9] (Sokar §7):
verbo no **passado** para o que foi medido, "provided evidence that" para o que foi inferido.
Rascunho:
> "We derived an entry--exit law for the death and resurrection of a smoothed unit and verified each
> of its constants on a two-neuron instance. We measured the law in a small MLP under a
> pre-registered protocol: resurrection occurs, the residual area opens before the return in 13 of
> 16 cases, hard ReLU gives a structural zero, and the return probability is monotone in the
> activation stiffness. We provided evidence that the mechanism does not transfer unchanged to a
> GELU FFN, where the derivative changes sign; that transfer is stated as a protocol and left open."

## F9. Não existe Acknowledgments / financiamento / conflito de interesse

**Padrão violado:** [V §11] — financiamento no `\thanks` do título (SIAM) ou na seção de
agradecimentos (IOP: "acknowledgment of funding should be included in the acknowledgments section
rather than here"); Elsevier: "Authors must disclose any funding sources" e a *declarations tool*
obrigatória.

**Correção concreta:** `\section*{Acknowledgments}` não numerada, com uma frase — "This work
received no funding and was carried out independently. The author declares no competing interests."
Ausência de financiamento **é** uma declaração de financiamento.

## F10. Não existe Data availability statement

**Padrão violado:** [V §11] — **IOP: obrigatória**, "The journal **requires** authors to include a
data availability statement"; a verificação é **condição de publicação**; "stating that your data
will be made available upon request **is not sufficient**". Nonlinearity adotou a política padrão.

**Correção concreta:** `\section*{Data availability}` depois de Acknowledgments:
> "All code, per-run manifests, raw episode files and aggregate JSON needed to reproduce every
> number and figure in this note are in the repository `<URL>`, at the commits cited in
> Section~7. No external data were used; all data are generated by the scripts listed there."

Isto depende de F11.

## F11. Reprodutibilidade sem URL de repositório, sem licença, sem dependências

**Trecho atual (§7, linhas 496–523):** caminhos internos (`06-notas/t2-prova/`,
`etapa4-mlp-populacao/pilotos-4a2/pressao/`) e comandos exatos (`python run_4a3.py --workers 4`),
mas **nenhum endereço público**, nenhuma versão de Python/numpy, nenhuma licença.

**Padrão violado:** [M §6.4], leitura literal do dossiê: citar commit hash por experimento é
**acima** do padrão da área, "o que falta para bater o padrão é o par (URL pública do repositório,
comando exato de execução)"; e [M §4.2] item 5 do checklist NeurIPS: "If a subset of experiments are
reproducible, **you should state which ones are**". [M §4.6] ML Code Completeness Checklist item 1:
"Specification of dependencies".

**Correção concreta:** abrir o primeiro parágrafo de §7 com:
> "Code and results are at `<URL>` (MIT licence), Python 3.13 with `numpy` only for the MLP ladder;
> all commit hashes below resolve in that repository. Everything in this note is reproducible from
> it; nothing depends on data or hardware not described here."
Se o repositório **não** for público na data de circulação, a convenção manda dizer isso na mesma
frase, e não deixar o leitor descobrir clicando.

## F12. Não há informação de compute

**Padrão violado:** [M §8.4] item 8 do checklist NeurIPS: tipo de CPU/GPU, memória, e **tempo de
cada run**, incluindo experimentos preliminares não reportados; checklist v2.0: "The average runtime
for each result, or estimated energy cost" e "A description of the computing infrastructure used".
O dado existe no projeto (≈12,2 h para 800 runs).

**Correção concreta:** uma frase no fim de §7:
> "All runs are single-threaded CPU (AMD, 4 workers via `multiprocessing`, no GPU). The 4a.3
> confirmatory (400 runs, $H=5\cdot10^5$) took $\approx$X h wall-clock; the transformer protocol
> (800 runs, $H=2\cdot10^5$) is budgeted at $\approx$12.2 h. Preliminary and discarded runs
> (etapa-4a, 4a.2, pilots) total approximately Y h."

## F13. Não há seção de desvios do protocolo pré-registrado

**Padrão violado:** [M §9.2], duas fontes lidas: PMLR v181 §5.3 **"Documented Modifications"** e
PMLR v148 §5.1 **"Modifications to the Original Proposal"** — em ambos, lista enumerada, cada desvio
com sua razão. [M §11 item 5]: "Mesmo vazia, ela é o que sinaliza pré-registro real ao leitor."
O paper hoje diz "hypotheses that fell were archived, not rewritten" (linha 257) — afirmação sobre
processo, sem a lista que a torna auditável.

**Correção concreta:** `\subsection{Deviations from the pre-registered protocol}` no fim de §5:
> "(i) None for 4a.3: the analysis frozen at `a8f8f8c` was run unchanged at `5a961cb`.
> (ii) The eligibility window $[0,307{,}536]$ and the $\ge6$-return floor for C2 were fixed in the
> pre-registration, not after seeing the data. (iii) `<qualquer outro>`."
Se realmente não houve desvio, a seção diz isso em uma linha — e vale mais do que a frase genérica.

## F14. As predições de matar não estão escritas junto das hipóteses

**Trecho atual:** os limiares aparecem dispersos — "well above the kill threshold $0.076$"
(linha 381), "lower Clopper--Pearson bound $0.544>0.5$, $\ge6$ returns" (linha 384),
"C1's kill is intentionally near-unreachable at this $N$" (linha 411).

**Padrão violado:** [M §9.2] "Hipóteses numeradas e nomeadas como itens de primeira classe"
(PMLR v181: "Hypothesis 1: ...", "Hypothesis 2: ..."); [M §11 item 6]. E o próprio `CLAUDE.md`:
"toda hipótese entra com a predição falsificável (a condição que a MATA) escrita ANTES".

**Correção concreta:** uma tabela pequena no início de §5.4 — coluna 1 rótulo (C1/C2/C4), coluna 2
estimando, coluna 3 **condição que mata**, coluna 4 veredito. Tira três números de dentro da prosa e
torna o pré-registro legível em cinco segundos.

## F15. Falta C3 — a lista de veredictos pula de C2 para C4

**Trecho atual (linhas 377–395):** os itens são **C1**, **C2**, **C4**, "Validity". Não há uma
palavra sobre C3.

**Padrão violado:** [M §5.4] item 1 (claims vs. escopo) e [M §9.3], COS: uma pré-registração
especifica hipóteses e **ao menos um teste confirmatório**; se C3 foi pré-registrado e não é
reportado, o leitor não pode saber se caiu ou se foi omitido. Buraco numa numeração é lido como
resultado escondido — exatamente o oposto do que o paper quer transmitir.

**Correção concreta:** ou reportar C3 com o mesmo destaque dos outros, ou uma linha explícita:
"C3 (`<nome>`) was pre-registered as `<...>`; it is not decidable at this $N$ and we report it in
the repository analysis, not here." Não deixar o buraco.

## F16. "Non-commutation" é afirmada como resultado e nunca formalizada

**Trecho atual:** aparece no abstract ("The two limit orders do not commute", linha 43), como bullet
de contribuição (linha 75), como §4 inteira, dentro do teorema (linha 156) e na Tabela 1
(linha 393: "the non-commutation of Thm. 1 measured with $N=100$"). Mas §4 é **prosa**: nem
proposição, nem enunciado, nem prova.

**Padrão violado:** [M §5.4] item 3 (teoremas com enunciado e prova completos); [E §5.1] convenção
#5, "cada afirmação forte vem com sua condição colada".

**Correção concreta:** transformar §4 em uma `\begin{proposition}` de três linhas ("Let $\theta^\infty$
be as in Definition 1 and let $\theta^{\rm FF}$ denote the solution selected by the pointwise
generalized-gradient flow of [EJRW23, Theorem 3.3]. Under (H1)–(H2), $\theta^{\rm FF}$ freezes
$\theta_{k^\ast}$ on $[t_{\rm hit},T]$ while $\theta^\infty$ releases at $t_{\rm exit}$; hence the
two limits do not commute.") — ou rebaixar explicitamente para "we read this as", que é o que a
prosa hoje faz sem dizer.

## F17. Citações sem localizador quando se usa um resultado específico

**Trechos atuais:** "the quadratic normal form of De Maesschalck--Schecter~\cite{DMS16}" (linha 178);
"this is the rigorous branch of Eberle, Jentzen, Riekert and Weiss~\cite{EJRW23}, whose selected
solution has a non-decreasing degenerate set" (linha 237); "generic piecewise-smooth exit
theory~\cite{Jef16,GH}" (linha 191); "the canonical Sotomayor--Teixeira
regularization~\cite{ST98,LST07} requires both one-sided normal components nonzero" (linha 193).

**Padrão violado:** [E §2.7] Kristiansen: "nunca citar '[33]' seco quando se usa uma proposição de
[33]; escrever `[33, Proposition 6.3]`". [E §3.6] Eberle et al.: "[24, Proposition 2.2]",
"[25, Setting 2.1 and Proposition 2.3]". O dossiê nota que isso "é exatamente o que o nosso
protocolo de `verificacao-fontes` exige".

**Correção concreta:** `\cite[Theorem 1.1]{DMS16}`, `\cite[Theorem 3.3]{EJRW23}`,
`\cite[Prop.~X]{ST98}` — cada um com o número **conferido na fonte aberta**, não de memória.

## F18. Falta o parágrafo de roteiro no fim da introdução

**Padrão violado:** [V §10] idioma do `ex_article.tex` da SIAM, literal: "The paper is organized as
follows. Our main results are in §…, our new algorithm is in §…, experimental results are in §…, and
the conclusions follow in §…". [E §2.6 e §3.5] os dois espelhos de math.DS/ML-teórico têm o roteiro
e ele **declara onde a parte chata foi jogada** ("but delay the somewhat technical part ... to
Section 3"). Hoje o leitor cai da lista de contribuições direto no setup.

**Correção concreta:** duas frases no fim de §1, incluindo a declaração de escopo negativo (que
substitui parte do que o parágrafo "deliberately honest" faz hoje de forma mais ruidosa — ver G1).

## F19. Faltam MSC 2020 e keywords

**Padrão violado:** [V §5] — **SIADS: "Keywords and MSC codes must accompany all articles."**;
Physica D exige keywords (máx. 6); Nonlinearity recomenda `\ams{}`. Prática medida: num preprint
puro a maioria omite (`\subjclass` em 6/25 das notas curtas), então **isto é FORTE e não BLOQUEIA
enquanto o alvo for só arXiv** — vira bloqueio no dia da submissão.

**Correção concreta:**
```latex
\subjclass[2020]{Primary 34E15; Secondary 37N99, 68T07}
\keywords{entry--exit function, geometric singular perturbation, dead neurons,
smoothed ReLU, gradient flow, plasticity}
```
**Os três códigos MSC acima estão `[NÃO VERIFICADO]`** — [V §5] avisa literalmente "a **confirmar na
tabela oficial da MSC2020 antes de usar** ... **Não copiar sem checar**". Abrir
`mathscinet.ams.org/mathscinet/msc/msc2020.html` e conferir 34E15 (singular perturbations,
turning points) e 68T07 antes de colocar no arquivo.

## F20. Sem declaração de uso de LLM

**Padrão violado:** [M §3.5] — ICLR 2026: "If LLMs played a significant role in research ideation
and/or writing … authors should describe the precise role of the LLM in a separate section";
"**Not disclosing significant LLM usage can lead to desk rejection**". NeurIPS: é o **item 16** do
checklist. Dado o modo de trabalho deste projeto, omitir é o risco mais evitável do pacote.

**Correção concreta:** `\section*{Declaration of LLM usage}` antes das referências, dizendo o que é
verdade e nada além: quais partes (redação, revisão adversarial de desenho experimental, geração de
código de experimento) e o que permaneceu humano (as hipóteses, a decisão de matar, a leitura das
fontes). Uma declaração precisa é neutra; uma vaga é suspeita.

## F21. Checklist NeurIPS de 16 perguntas nunca foi respondido

**Padrão violado:** [M §5], que recomenda explicitamente: "rodar as 16 perguntas contra o draft
**antes de postar**, e arquivar as respostas em `00-admin/` mesmo sem submeter. É exatamente o
'patch test' editorial."

**Correção concreta:** `00-admin/checklist-neurips-paper-v1.md` com o tripé
`Question / Answer: [Yes|No|NA] / Justification` e o ponteiro de seção em cada uma
([M §5.2], formato literal). Barato e verificável.

## F22. Geometria da página fora do limite SIADS

**Trecho atual (linha 3):** `\usepackage[margin=1.05in]{geometry}` com `11pt` → em papel carta,
área de texto ≈ **6,4 in × 8,9 in**.

**Padrão violado:** [V §11] SIADS, para quem não usa as macros SIAM: "use at least 10pt font and set
the margins so that the **text width and height are no more than 6in and 8in**".

**Correção concreta:** `\usepackage[text={6in,8in},centering]{geometry}`. Custa ~1 página a mais e
resolve o item para sempre.

## F23. Subtítulos experimentais só metade no padrão de frase-resultado

**Trechos atuais:** §5.1 "The deterministic schedule is falsified under SGLD" (✓ frase completa);
§5.2 "Two instrument designs recorded as negatives" (fragmento nominal); §5.3 "The mechanism,
observed" (fragmento); §5.4 "Confirmatory: the area law in a real MLP" (dois-pontos + fragmento);
§5.5 "Transformer: two new mechanisms, confirmatory in progress" (status).

**Padrão violado:** [E §4.6], classificada no dossiê como "convenção fortíssima e diretamente
copiável": cada bloco experimental tem por título **a conclusão que ele sustenta**, com sujeito e
verbo, e o hedge embutido no título quando o resultado é fraco ("Input non-stationarity **does not
appear to be** a major factor.").

**Correção concreta:**
- §5.2 → "Two instrument designs failed their own null control."
- §5.3 → "A softplus unit dies and returns in a real MLP, and the area opens first."
- §5.4 → "The area law reproduces under a pre-registered confirmatory."
- §5.5 → "The mechanism does not transfer unchanged to a GELU feed-forward block."

---

# COSMÉTICO

## C1. Título em três linhas e três orações

**Trecho atual (linhas 17–19):** `An entry--exit law for dead-neuron resurrection:\\ the smoothed-ReLU
limit of training dynamics at an\\ activation boundary, and its confirmation in a small MLP`

**Padrão:** [V §2] "Títulos de 3 linhas **não aparecem na amostra**"; subtítulo com dois-pontos é
14% em math.DS. [M §10.3] "nenhum dos 40 [títulos] tem três orações"; padrão de teoria é
`<lei/objeto novo>: <o que se prova> in <onde>` em uma oração. [E §5.1] "**Nenhum título tem
dois-pontos**, subtítulo ou adjetivo de valor" nos quatro espelhos; [E §2.2] Kristiansen
**encurtou** o título na revisão e registrou a mudança em Comments — aplicações saem do título e
ficam no abstract.

**Correção concreta (3 opções, todas em uma linha, sentence case, terminando na restrição real —
molde de [E §3.2]):**
1. `An entry--exit law for the resurrection of smoothed dead neurons`
2. `Death and resurrection of neurons in the sharp-activation limit of gradient flow`
3. `An area law for neuron resurrection under smoothed ReLU activation`

E remover os `\\`: se ficar longo, a classe quebra sozinha.

## C2. 33 travessões `---` em 11 páginas

`grep -o '\-\-\-' | wc -l = 33` — média de 3 por página, vários por parágrafo (linhas 53, 66, 84,
90, 152, 189, 271, 274, 296, 299, 341, 385, 393, 413, 423, ...). É o conectivo default do texto.
Nenhum dos quatro espelhos escreve assim ([E §1.3, §3.3, §4.3]: períodos curtos, ponto final).
**Correção:** substituir 2/3 por ponto, ponto-e-vírgula ou dois-pontos. É a mudança isolada que mais
muda a "assinatura" do texto.

## C3. 42 `\emph` em 11 páginas

**Padrão:** [V §7] `\emph` é usado com parcimônia — **mediana de 2 ocorrências por paper** (quartis
0 e 16; máximo 71) na amostra de 47 fontes; o uso canônico é marcar o *definiendum* dentro de uma
`definition`. Aqui há `\emph{with memory}`, `\emph{falsified}`, `\emph{negatives}`, `\emph{invalid}`,
`\emph{universal}`, `\emph{area}`, `\emph{not}`, `\emph{breaks}`, `\emph{hypothesis}`… Ênfase que
aparece 42 vezes deixa de enfatizar.
**Correção:** manter ≤8: os termos que estão sendo *definidos* (activation stiffness, margin,
residual excluding the unit, hybrid reference, sticking depth) e nada mais. Nunca `\emph` em
`\emph{not}`.

## C4. Abreviações "Thm.", "Lem.", "Lems.", "Figs.", "Fig."

12× `Lem.`, 4× `Thm.`, 3× `Fig.`, 1× `Figs.`, 1× `Lems.`
**Padrão:** [V §7] IOP §4.4, literal: "**Do not include 'eq.' 'equation' etc before an equation
number or 'ref.' 'reference' etc before a reference number**"; e "The words **table and figure
should be written in full and not abbreviated** to tab. and fig."
**Correção:** carregar `cleveref` (recomendado e carregado pela classe SIAM, [V §7]) e usar
`\cref{thm:main}` → "Theorem 1", `\cref{lem:gauge}` → "Lemma 3", `\cref{fig:escada}` → "Figure 3".
Resolve os 21 casos de uma vez e sobrevive à renumeração de B13.

## C5. Autoelogio de honestidade — 9 ocorrências

**Trechos:** "The paper is **deliberately honest** about scope" (94); "the failed designs are shown
**rather than hidden**" (256); "An erratum records the next step **honestly**" (277);
"an object of study, **not hidden**" (386); "**The honest limits** of the confirmatory" (409);
"a **genuine test, not a stamp**" (461); "C2 (the area law, **the heart**...)" (382);
"the **most alive** arm" (250, 390).
**Padrão:** [E §2.8, §3.8, §4.9] — nos quatro espelhos o hedge e a limitação são ditos em voz ativa
e **curta**, sem adjetivo de virtude: "We do not aim to provide a general geometric framework. This
must be part of future research."; "ReDo reduces dormant neurons significantly but it doesn't
completely eliminate them."; "we hypothesize that". Nenhum narra a própria integridade.
**Correção:** apagar o adjetivo, manter o fato. "The paper is deliberately honest about scope: the
theorem is one two-neuron instance" → "**The theorem covers one two-neuron instance**; the
confirmatory covers one small architecture at zero noise; the stochastic regime is open and, where
we probed it, it breaks the deterministic schedule (§5.1)." Diz a mesma coisa e soa como um
pesquisador em vez de um comunicado.

## C6. Construção "X, not Y" 13 vezes

`grep -o ', not [a-z]' = 13`: "not a fit" (204), "not by number", "not the sign of $R$" (182, 562),
"not of the instrument" (275), "not the theorem's physical death" (296), "not of the zero-gradient
theorem" (311), "not hidden" (386), "not a technicality" (230, 430), "not discovered after" (414),
"not a stamp" (461)... É um tique retórico: a mesma figura de linguagem como espinha de 13 frases.
**Correção:** manter 3–4 onde a oposição é informativa (ex.: "the layer is held by the area, not the
sign of $R$" — essa é conteúdo), reescrever o resto em afirmativa.

## C7. "Grade:" — falso cognato

**Trecho atual (linha 349):** "Grade: $\{$sp-$\beta3$ (primary), ...$\}\times m\in\{4,6\}\times50$"
Em inglês *grade* é nota/inclinação; o que se quer dizer é **grade** (PT) = *grid*.
**Correção:** "**Design grid:**" ou "**Factorial design:**". Um revisor nativo marca isso na
primeira leitura, e é o tipo de coisa que faz o resto do texto perder crédito.

## C8. "virgin seeds"

**Trecho atual (linha 350):** "$\times50$ **virgin** seeds ($1900$--$1949$)".
Não é o termo da área. **Correção:** "50 **held-out** seeds (1900--1949, not used in any pilot)".

## C9. "An erratum records the next step honestly"

**Trecho atual (linha 277).** *Erratum* é a correção de um erro publicado; não é o lugar onde se
registra um próximo passo. **Correção:** "A follow-up note records what happened next:" — e o
conteúdo (o modelo OU refutado no estágio de preditor e **não rodado**) já é forte o bastante
sozinho.

## C10. "$z=+131$" sem nomear o teste

**Trecho atual (linha 270):** "the bin $[0,0.1)$ saw $150$ captures against $1.4$ predicted
($z=+131$)". Um z de 131 sem o teste nomeado parece erro de digitação.
**Padrão:** [M §8.1] "The method for calculating the error bars should be explained. **The
assumptions made should be given.**"
**Correção:** "(Poisson $z=+131$ against the model's expected count, under the model's own
independence assumption)".

## C11. Labels internos em português num paper em inglês

`sec:falsif`, `sec:negativos`, `sec:mecanismo`, `sec:confirm`. Invisíveis no PDF, visíveis no fonte —
e o fonte vai junto na submissão do arXiv. **Correção:** `sec:falsification`, `sec:negatives`,
`sec:mechanism`, `sec:confirmatory`.

## C12. Caminhos e nomes de script em português, sem glosa, dentro do PDF

§7 imprime `etapa4-mlp-populacao/pilotos-4a2/pressao/taxa/`, `figuras_draft.py`, `analise_4a3.py`,
`NEGATIVOS.md`, `p5-relu-morte-v2/`. Um leitor não-lusófono não sabe o que é "pressao" (o regime
sub-parametrizado), "taxa" (o piloto de taxa de eventos) nem "NEGATIVOS".
**Padrão:** [M §4.5] checklist v2.0, "README file includes table of results accompanied by **precise
command to run**" — o comando precisa ser *seguível*.
**Correção:** não renomear nada (quebraria os commits citados); acrescentar uma glosa de uma linha
por item: "`.../pressao/` (the *pressure* regime: sub-parametrized, plateau $\sim0.15$)",
"`.../taxa/` (the event-rate pilot)", "`NEGATIVOS.md` (register of negative results)".

## C13. Larguras de figura inconsistentes

`0.60`, `0.95`, `0.56`, `0.56` `\linewidth`. **Correção:** duas larguras só — `0.6` para figura de
uma coluna e `0.95` para a de duas — e a mesma altura de fonte nos eixos das três.

## C14. Macro `\R` definida e nunca usada

**Trecho atual (linha 12):** `\newcommand{\R}{\mathbb{R}}` — zero usos no corpo. Sinal de arquivo em
obra. **Correção:** remover, ou usar (§2 nunca declara em que espaço vivem $x_i$; ver F2).

## C15. `\emph` no abstract

**Trecho atual:** `\emph{with memory}` (30), `\emph{falsified}` (39), `\emph{confirmatory}` (não),
`\emph{breaks}`.
**Padrão:** [V §4] SIAM: "Because of online formatting, abstracts must appear **as plain as
possible**"; arXiv pede evitar comandos de fonte no campo abstract. O `\emph` no PDF é tolerável; o
texto colado no formulário do arXiv tem de ir limpo.
**Correção:** tirar do abstract e lembrar de colar a versão sem markup no formulário.

## C16. Abstract com 203 palavras / 1318 caracteres

Dentro de SIADS (≤250) e do arXiv (≤1920 chars), **3 palavras acima** do limite de 200 do guia LaTeX
da IOP [V §4]. (A política web atual da IOP diz ≤300; os dois números conflitam e o dossiê registra
ambos.) **Correção:** cortar três palavras — a frase de status do transformer já sai por B8, o que
resolve com folga.

## C17. Planejar o campo `Comments` do arXiv

**Padrão:** [V §11] "N pages, M figures"; 137/282 declaram páginas, 15/282 acrescentam "comments
welcome". [M §1.7] arXiv: "This is the proper field for 'to be published in' or 'submitted to'".
**Correção:** `11 pages, 3 figures, 1 table. Comments welcome.` E escolher a categoria primária
(math.DS com cross-list cs.LG, ou o inverso) — [M §1.7]: "Please carefully review the descriptions
of categories to find the best fit".

---

# O que soa "gerado" e não "escrito por um pesquisador da área"

Consolidando os itens de forma acima com o que eles produzem juntos. Nenhum destes é fatal
isoladamente; a **assinatura** é a co-ocorrência dos seis.

| Marca | Medida no arquivo | O que os espelhos fazem |
|---|---|---|
| **G1. Travessão como conectivo default** | 33 `---` em 11 pp (C2) | períodos curtos, ponto final [E §1.3, §3.3] |
| **G2. Ênfase inflacionada** | 42 `\emph` + 30 `\textbf` (C3) | mediana 2 `\emph`/paper em 47 fontes [V §7] |
| **G3. Narrar a própria virtude** | 9 ocorrências de honest/hidden/genuine/heart (C5) | limitação em voz ativa e curta, sem adjetivo [E §2.8, §4.9] |
| **G4. Aposto avaliativo no fim da frase** | "the heart", "the most alive", "not a stamp", "an object of study, not hidden", "The scoping verdict is itself a result" | conclusão vem no *resultado*, não no aposto [E §1.10: hedge sobre matemática é proibido, e elogio também não existe] |
| **G5. Paralelismo "X, not Y"** | 13 vezes (C6) | usado, mas raro e informativo |
| **G6. Rótulos internos vazados** | `T2'`-min (B14), `[E5-PENDING]` (B8), `etapa-4a`, `piloto`, labels em PT (C11), `pressao`/`taxa`/`NEGATIVOS` sem glosa (C12) | vocabulário público e estável [E §5.1] |

Diagnóstico honesto: **o conteúdo não soa gerado — a forma sim.** O texto tem coisas que nenhum
gerador produz (um negativo de instrumento reportado com o mesmo destaque de um positivo; um
"predicted-zero window" violado e mantido no paper; um modelo refutado no estágio de preditor e
declarado não-rodado). O que o denuncia é o acabamento: pontuação uniforme, ênfase inflacionada,
e a insistência em *dizer* que é honesto, quando a honestidade já está nos dados mostrados.
Cortar G1–G5 é uma passada de edição de meio dia e muda a percepção mais do que qualquer
reestruturação.

---

# O que já está conforme (não mexer)

Registrado para não haver regressão numa próxima passada:

1. **Legendas de figura embaixo** nas três figuras — [V §8]: 37 abaixo, 0 acima em 47 fontes, zero
   exceções.
2. **`booktabs`, sem réguas verticais** na Tabela 1 — [M §1.5]: "Publication-quality tables do not
   contain vertical rules. We strongly suggest the use of the `booktabs` package."
3. **Abstract sem `\cite`, sem referência a equação/figura numerada, um parágrafo** — [V §4]:
   citação no abstract em apenas 1/22 dos publicados; a regra escrita é de fato seguida.
4. **Números no abstract** (400 runs, $p$, $\beta$) — [V §4]: 12/21 dos abstracts de math.DS curto
   contêm dígitos; é aceito.
5. **Bullets de contribuição na introdução** — [M §2.1]: modo dominante em papers
   empírico-teóricos.
6. **Seção `Limitations` com itens (i)–(v)** — [M §3.1]: é exigência formal do NeurIPS e o formato
   bate (ainda que [M §2.3] mostre o exemplar mais próximo usando *subseções* numeradas, uma por
   limitação).
7. **Commit hash por experimento** — [M §6.4], literal: "citar commit hash curto por experimento é
   **mais forte** do que o padrão da área, não mais fraco". Falta só a URL (F11).
8. **Verificação executável por lema** — [M §7.3]: "não tem paralelo que eu tenha localizado na
   amostra; é acima do padrão e vale **dizer no paper** que é isso que se está fazendo, porque o
   leitor não vai assumir". O paper já diz (linhas 70–71, 175–176). Manter e amarrar em F11.
9. **Negativos no corpo, com o mesmo destaque dos positivos** — [M §9.1] é literalmente a motivação
   declarada do formato pré-registrado (PMLR v148, Preface).
10. **A citação `DMS16` está correta.** [E, Ressalva 1] alertava que 1511.01815 é
    **De Maesschalck–Schecter**, não "Dumortier–De Maesschalck". Conferido: linhas 178, 191, 532 e o
    `\bibitem` (601) usam a autoria certa. **Nada a corrigir aqui** — o alerta do dossiê não se
    aplica a esta versão.
11. **Bibliografia à mão (`thebibliography`)** — [V §9]: 15/25 das notas curtas de math.DS fazem
    assim; não penaliza.
12. **Aspas TeX corretas** (`` ``dies'' ``), `microtype`, `hyperref` — [V §7]: 17/22 dos publicados
    carregam hyperref/cleveref.

---

# Fora do escopo desta auditoria

- **Correção matemática** do teorema, dos seis lemas e das constantes. Nada aqui foi reconferido;
  esta é auditoria de *forma*.
- **Correção dos números experimentais** contra os `results/*.jsonl`. As inconsistências apontadas
  (B7, F15) são de **apresentação** — dois condicionamentos com o mesmo nome, e um rótulo ausente —
  não uma acusação de erro no cálculo.
- **Como pesquisadores independentes se apresentam** (B3): os três dossiês não mediram isso. A
  obrigação de afiliação+e-mail é verificada; a forma "Independent researcher" não é.
- **Códigos MSC** (F19): sugeridos e explicitamente `[NÃO VERIFICADO]`; o próprio [V §5] manda
  conferir na tabela oficial antes de usar.
- **Título exato da referência Guglielmi–Hairer 2015** (B9): citado de memória e marcado como não
  verificado; abrir o DOI.
