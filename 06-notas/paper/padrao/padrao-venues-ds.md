# Padrão de forma para nota curta (teorema + experimentos) em sistemas dinâmicos / análise aplicada

Alvo: arXiv **math.DS / math.CA** (+ cross-list) e, como venue de submissão, **Physica D**,
**Nonlinearity** ou **SIAM J. Appl. Dyn. Syst. (SIADS)**.
Formato-alvo: nota curta, 8–14 pp, um teorema principal + escada experimental.

Data do levantamento: **2026-07-25**. Autor do dossiê: subagente de pesquisa (j-space).

## 0. Como ler as etiquetas

- `[VERIFICADO em <fonte>]` — texto lido diretamente na fonte citada (guia de autor, arquivo de
  classe, documentação oficial, ou medição feita sobre fontes LaTeX baixadas). Onde a fonte tem
  data, a data está junto: guias de editora mudam.
- `[PRATICA COMUM observada]` — não é regra escrita em lugar nenhum; é frequência medida em
  amostra de papers reais, com o n e a proporção explícitos. Amostras descritas no §12.
- Nada aqui é inventado. Onde não achei fonte, está escrito **"não verificado"** e o item fica
  como pergunta em aberto (§13).

Três guias de editora estavam bloqueados a acesso direto (Cloudflare/captcha em `epubs.siam.org`,
`sciencedirect.com`, `siam.org`); foram lidos por *snapshot* do Internet Archive, com a data do
snapshot registrada. O SIAM Style Manual (o manual de copidesque, mais detalhado que o guia de
autor) **não foi acessível** — nenhuma afirmação abaixo depende dele.

---

## 1. Quadro comparativo: o que cada venue **obriga**

| Item | SIADS (SIAM) | Nonlinearity (IOP) | Physica D (Elsevier) | arXiv |
|---|---|---|---|---|
| Abstract | 1 parágrafo, **≤250 palavras**, refs por extenso e minimizadas | 1 parágrafo, **≤200 palavras** (guia LaTeX 2005) / **≤300** (política web atual); sem refs, sem eqs, sem nº de figura | conciso, *stand-alone*, refs evitadas (se essenciais: autor+ano) | **≤1920 caracteres** |
| Keywords | **obrigatório** | não exigido em Nonlinearity | **obrigatório, máx. 6** | opcional |
| MSC | **obrigatório** | recomendado (`\ams{}`; PACS ou MSC) | não exigido | opcional (campo MSC-class) |
| Estilo de citação | numérico, alfanumérico `[T]` ou autor-ano — livre, mas consistente | Harvard **ou** Vancouver, livre | **numérico `[1]`**, ordem de aparição | — |
| Highlights | não | não | **obrigatório** (3–5 bullets, ≤85 caracteres cada, arquivo separado) | — |
| Declaração de dados | não exigida no guia | **obrigatória** (condição de publicação) | encorajada (*Data statement*) | — |
| Conflito de interesse | — | política IOP | **obrigatória** (declarations tool) | — |
| Limite de tamanho | **≤40 pp** e ≤10 MB | sem limite publicado | sem limite publicado | — |
| Numeração de teoremas | **sequência única, independente da seção** (Teorema 1, 2, 3…) | livre | livre | — |

Fontes de cada célula: §§2–11 abaixo.

**Consequência prática para nós:** a interseção que satisfaz os três de uma vez é —
abstract de ~200 palavras num parágrafo, sem citações e sem referência a equações/figuras;
keywords (≤6) + MSC 2020; citação numérica `[1]` em ordem de aparição; figuras com legenda
embaixo; seção *Acknowledgments* não numerada antes das referências; e uma declaração de
disponibilidade de dados/código. Escrevendo assim, a nota serve aos três sem retrabalho.

---

## 2. Título

- **IOP/Nonlinearity:** "The first letter of the title should be capitalized with the rest in
  lower case. Mathematical expressions within the title may be left in light-face type." Título
  curto opcional para cabeçalho: `\title[Short title]{Full title}`.
  `[VERIFICADO em IOP, "How to prepare and submit an article ... using LaTeX2e" (G. Douglas, IOPP), §3.1 — PDF datado 2005-06-27, espelho li.mit.edu]`
- **SIAM/SIADS:** "Titles should be brief and appropriate for indexing and should specifically
  describe the content of the paper." A classe converte título e autores para **maiúsculas
  automaticamente** no display — logo, capitalização do fonte é irrelevante lá.
  `[VERIFICADO em SIADS Instructions for Authors, snapshot Wayback 2023-04-05]` e
  `[VERIFICADO em docsiamart.tex §"Front matter"]`
- **Elsevier/Physica D:** "Title. Concise and informative. Titles are often used in
  information-retrieval systems. Avoid abbreviations and formulae where possible."
  `[VERIFICADO em Physica D Guide for Authors, snapshot Wayback 2024-02-18]`
- **arXiv:** "Do not use all uppercase letters" no campo Title.
  `[VERIFICADO em info.arxiv.org/help/prep.html]`
- Capitalização de fato em math.DS: **44% Title Case, 44% sentence case, 12% misto**
  (123 / 125 / 34 de 282 títulos). Em math.CA: 44 Title Case, 59 sentence case de 136.
  Ou seja: **não há convenção dominante**; sentence case é a escolha segura porque é a única
  exigida por alguém (IOP).
  `[PRATICA COMUM observada — amostras A e B, §12]`
- Subtítulo com dois-pontos: **14% em math.DS (40/282), 10% em math.CA (14/136)**. É aceito, mas
  é minoria; título de uma linha é o padrão.
  `[PRATICA COMUM observada — amostras A e B]`
- Fórmulas no título são comuns em math.DS (`$R_0$`, `$C^1$`, `$p$-adic`) e não causam problema
  no arXiv; a Elsevier pede para evitar.
  `[PRATICA COMUM observada — amostra A]` / `[VERIFICADO em Physica D GfA 2024-02-18]`

> **Nota sobre o draft atual:** `paper-v1.tex` tem título em 3 linhas com `\\`. Títulos de 3
> linhas não aparecem na amostra; considerar encurtar para uma frase + dois-pontos no máximo.

## 3. Bloco autor / afiliação / e-mail / data / financiamento

Os três venues divergem no **mecanismo**, não no conteúdo. O conteúdo pedido é sempre: nome,
afiliação onde o trabalho foi feito, e-mail de pelo menos um autor, e o financiamento.

- **AMS/amsart** (o mais próximo do que uma nota solo usa no arXiv): ordem do template oficial é
  `\title` → `\author` → `\address` → `\curraddr` → `\email` → `\thanks` → `\subjclass[…]` →
  `\keywords` → `\date` → `\dedicatory` → `\begin{abstract}…\end{abstract}` → `\maketitle`.
  **O abstract vem ANTES do `\maketitle` em amsart** (ao contrário de `article`).
  `[VERIFICADO em /usr/share/texlive/texmf-dist/doc/latex/amscls/amsart-template.tex, v1.1 2014-07-24]`
- **SIAM:** endereço + e-mail de cada autor vão em `\thanks` do próprio autor; **o agradecimento
  de financiamento vai no `\thanks` do título**, não numa seção. `\headers{título curto}{autores}`
  gera o cabeçalho.
  `[VERIFICADO em docsiamart.tex §"Front matter"]`
- **IOP:** `\author{P J Smith$^1$, T M Collins$^2$ ... and Janet Williams$^3$}` — iniciais antes
  do sobrenome, **sem ponto depois das iniciais**, vírgula entre todos menos os dois últimos, que
  são separados por "and"; `\address{}` por afiliação; `\ead{}` com o e-mail de pelo menos um
  autor. Financiamento **não** vai em nota de título: "acknowledgment of funding should be
  included in the acknowledgments section rather than here."
  `[VERIFICADO em IOP LaTeX guidelines §3.2 e §3.1, PDF 2005]`
- **Elsevier:** afiliações marcadas com **letra minúscula sobrescrita**; autor correspondente
  explicitamente indicado com e-mail; "Present address" como nota de rodapé com **algarismo
  arábico sobrescrito**.
  `[VERIFICADO em Physica D GfA, snapshot 2024-02-18]`
- **Data (`\date`)**: em amsart o padrão é vazio (não imprime); preprints em arXiv normalmente
  deixam a data do arXiv falar. Colocar "DRAFT — not for circulation" na linha de data é uso
  interno, não é convenção de venue — nenhuma amostra tinha isso. `[PRATICA COMUM observada]`

## 4. Abstract

Regras escritas:

- **SIADS:** "A one-paragraph abstract, **not exceeding 250 words**, that summarizes the principal
  techniques and conclusions of the manuscript **in relation to known results** must accompany each
  manuscript. Because the abstract must be able to stand independently, **mathematical formulas and
  bibliographic references should be kept to a minimum**; bibliographic references must be **written
  out in full, not given by number**. LaTeX commands specific to the article should be avoided."
  `[VERIFICADO em SIADS Instructions for Authors, snapshot Wayback 2023-04-05]`
- **SIAM template:** "Abstracts must be able to stand alone and so **cannot contain citations to the
  paper's references, equations, etc.** An abstract must consist of a **single paragraph** and be
  concise. Because of online formatting, abstracts must appear as plain as possible. **Any equations
  should be inline.**"
  `[VERIFICADO em ex_article.tex do pacote SIAM LaTeX]`
- **Nonlinearity/IOP:** "It should be **self-contained with no reference to figures, tables,
  equations, bibliographic references** etc included and should **not normally exceed 200 words**
  ... normally restricted to a single paragraph."
  `[VERIFICADO em IOP LaTeX guidelines §3.3, PDF 2005]`
  A política web atual da IOP diz **≤300 palavras** e repete a proibição de acrônimos não definidos,
  números de tabela/figura, equações e referências.
  `[VERIFICADO em publishingsupport.iopscience.iop.org, "Article structure", acesso 2026-07-25]`
  (Os dois números conflitam; 200 satisfaz ambos.)
- **Physica D:** "A concise and factual abstract is required ... it must be able to stand alone.
  For this reason, **References should be avoided, but if essential, then cite the author(s) and
  year(s)**. Also, non-standard or uncommon abbreviations should be avoided."
  `[VERIFICADO em Physica D GfA, snapshot 2024-02-18]`
- **arXiv:** "abstracts longer than **1920 characters** will not be accepted"; evitar `\em`, `\it`
  e comandos de fonte.
  `[VERIFICADO em info.arxiv.org/help/prep.html]`

Prática medida:

- Comprimento do abstract em papers publicados nos três venues: **mediana 127 palavras**, média
  154, faixa 62–447; **17/21 ≤200 palavras, 18/21 ≤250**.
  `[PRATICA COMUM observada — amostra D, §12]`
- Em math.DS curto (≤14 pp): **mediana 109 palavras**, faixa 31–280.
  `[PRATICA COMUM observada — amostra C]`
- **"we" aparece em 20/22 abstracts** e verbo de resultado no **presente** ("we prove / show /
  study / establish / derive") em 18/22. Passado e voz passiva ("it is shown that") são minoria.
  `[PRATICA COMUM observada — amostra D]`
- **Citação dentro do abstract: 1/22 (4,5%)**. Ou seja, a regra escrita é de fato seguida: não se
  cita no abstract.
  `[PRATICA COMUM observada — amostra D]`
- **Números no abstract são normais**: 12/21 dos abstracts de math.DS curto contêm dígitos, e
  6/22 dos publicados contêm matemática inline `$…$`. Reportar "$400$ runs, $p=0.50$" no abstract
  é aceito; o que não se faz é referenciar `(3.7)` ou `[12]`.
  `[PRATICA COMUM observada — amostras C e D]`

> **Nota sobre o draft atual:** o abstract de `paper-v1.tex` tem **201 palavras / 1318 caracteres**
> — dentro de 250 (SIADS), no limite dos 200 (IOP), e dentro dos 1920 caracteres do arXiv. Não cita
> nada. Está conforme. Contém `\emph{}` — SIAM pede "as plain as possible", e o arXiv pede para
> evitar comandos de fonte no campo abstract (o `\emph` do PDF é fine; o texto colado no formulário
> do arXiv é que deve ir limpo).

## 5. MSC 2020 e keywords

- **MSC2020** é a revisão vigente da Mathematics Subject Classification, feita em conjunto por
  Mathematical Reviews e zbMATH como revisão da MSC2010.
  `[VERIFICADO em mathscinet.ams.org/mathscinet/msc/msc2020.html]`
- **SIADS: "Keywords and MSC codes must accompany all articles."**
  `[VERIFICADO em SIADS Instructions for Authors, snapshot 2023-04-05]`
  Na classe SIAM atual os ambientes são `\begin{keywords}` e **`\begin{MSCcodes}`** — a classe
  `siamart220329.cls` (v1.4.4, 2022-03-29) registra explicitamente no cabeçalho:
  "Changover from using 'AMS subject classifications' to 'MSC codes'". O ambiente antigo
  `\begin{AMS}` ainda existe por compatibilidade. Rótulo impresso das keywords: "Key words".
  `[VERIFICADO em siamart220329.cls, linhas 52, 741–745, 1116–1120]`
- **Nonlinearity/IOP:** "For Inverse Problems and Nonlinearity authors may use either PACS or MSC
  codes"; entram **logo depois do abstract** via `\ams{...}`. **Keywords não são exigidas em
  Nonlinearity** (a lista de journals que as exigem não a inclui); se usadas, entram como
  parágrafo `\noindent{\it Keywords\/}: ...` depois do abstract.
  `[VERIFICADO em IOP LaTeX guidelines §§3.4–3.5, PDF 2005]`
  Confirmado na prática: submissão real a Nonlinearity usa `\documentclass[a4paper]{iopart}` com
  `\ams{34C14, 34C23, ...}`, `\ead{}`, `\submitto{\NL}` e um parágrafo "Keywords:".
  `[VERIFICADO em arXiv:2510.11911 (journal-ref Nonlinearity), fonte LaTeX]`
- **Physica D:** "Immediately after the abstract, provide a **maximum of 6 keywords**, using
  American spelling and avoiding general and plural terms and multiple concepts (avoid, for
  example, 'and', 'of')." **Nenhuma exigência de MSC** aparece no guia.
  `[VERIFICADO em Physica D GfA, snapshot 2024-02-18]`
  Em `elsarticle`, o markup é `\begin{keyword} ... \sep ... \MSC[2020] ... \end{keyword}`;
  `\MSC` aceita o ano como argumento opcional (default 2000).
  `[VERIFICADO em elsdoc.pdf, elsarticle v3.5, doc datado 2026-01-09]`
- **amsart:** `\subjclass[2020]{Primary 37C10; Secondary 34E15, 68T07}` — o **default do ano no
  código-fonte da classe já é 2020** (`\newcommand*\subjclass[2][2020]`), e a divisão
  Primary/Secondary é a forma canônica.
  `[VERIFICADO em amsclass.pdf (TeX Live), linha 579]`
- **arXiv:** campo MSC-class é **opcional**, só para o arquivo math, e usa as designações
  "Primary"/"Secondary".
  `[VERIFICADO em info.arxiv.org/help/prep.html]`
- Prática medida: `\subjclass` aparece em **6/25** notas curtas de math.DS e keywords em 9/25 —
  ou seja, **num preprint puro a maioria omite os dois**; quem coloca usa `\subjclass[2020]{...}`
  com Primary/Secondary. Nos papers já publicados nos três venues, MSC aparece em 11/22.
  `[PRATICA COMUM observada — amostras C e D]`

## 6. Ambientes de teorema, numeração, prova

Aqui há uma divergência **dura** entre AMS e SIAM, e ela precisa de decisão explícita.

- **AMS (amsart), padrão do template oficial:**
  ```latex
  \newtheorem{theorem}{Theorem}[section]     % numera por seção: Theorem 2.1
  \newtheorem{lemma}[theorem]{Lemma}         % contador COMPARTILHADO
  \theoremstyle{definition}
  \newtheorem{definition}[theorem]{Definition}
  \newtheorem{example}[theorem]{Example}
  \theoremstyle{remark}
  \newtheorem{remark}[theorem]{Remark}
  \numberwithin{equation}{section}
  ```
  `[VERIFICADO em amsart-template.tex, AMS, v1.1 2014-07-24]`
  Efeito: Theorem 2.1, Lemma 2.2, Remark 2.3 — **um único contador**, numerado por seção; nunca
  "Theorem 1" e "Lemma 1" convivendo.
- **SIAM:** "Note that SIAM now numbers theorems **in sequence, independent of the section
  number**." Ambientes já definidos pela classe: `theorem`, `lemma`, `corollary`, `definition`,
  `proposition`, mais `proof`. Para criar novos: `\newsiamthm` (cabeçalho versalete, corpo
  itálico) e `\newsiamremark` (cabeçalho itálico, corpo romano) — **em vez de** `\newtheorem`.
  Argumento opcional nomeia o teorema: `\begin{theorem}[Mean Value Theorem]`.
  `[VERIFICADO em docsiamart.tex §"Theorem-like environments"]`
  Na classe: `\newtheorem{theorem}{Theorem}` (sem `[section]`) e todos os demais com
  `\newtheorem{#1}[theorem]{#2}` — contador compartilhado, sequência única.
  `[VERIFICADO em siamart220329.cls, linhas 1448 e 1458]`
- **Elsevier:** `\newtheorem` (corpo itálico), `\newdefinition` (corpo romano) e `\newproof`
  (corpo romano, **sem contador**); `\newproof{pot}{Proof of Theorem \ref{thm2}}` é o idioma para
  provas nomeadas. `amsthm` pode ser carregado e sobrescreve tudo isso.
  `[VERIFICADO em elsdoc.pdf, elsarticle v3.5]`
- **Prova / QED:**
  - `amsthm` "defines a proof environment that automatically adds a QED symbol at the end";
    o ambiente é "primarily intended for **short proofs, no more than a page or two** in length;
    **longer proofs are usually better done as a separate `\section` or `\subsection`**".
    Nome alternativo via argumento opcional: `\begin{proof}[Proof of Theorem 1.2]`. Linha em
    branco antes de `\end{proof}` **desliga** a colocação do símbolo — bug comum.
    `[VERIFICADO em amsthdoc.pdf (TeX Live), §5]`
  - SIAM insere `\proofbox` automaticamente "even if it ends in an equation environment", mas
    "the document may need to be compiled twice for the mark to appear".
    `[VERIFICADO em docsiamart.tex]`
- Prática medida: **14/25** das notas curtas de math.DS usam contador compartilhado numerado por
  seção; **17/25** usam o ambiente `proof`; **apenas 2/25** escrevem `\qed`/`\square` à mão (o
  resto deixa a classe fazer). Ambiente `Remark` definido em **18/22** dos publicados.
  `[PRATICA COMUM observada — amostras C e D]`

> **Nota sobre o draft atual:** `paper-v1.tex` declara
> `\newtheorem{theorem}{Theorem}` + `\newtheorem{lemma}{Lemma}` + `\newtheorem{definition}{Definition}`
> com **contadores independentes** — produz "Theorem 1 / Lemma 1 / Definition 1" simultâneos.
> Isso não corresponde nem ao padrão AMS nem ao SIAM. Correção mínima (mantendo `article` e o
> estilo SIAM de sequência única):
> `\newtheorem{theorem}{Theorem}` ; `\newtheorem{lemma}[theorem]{Lemma}` ;
> `\theoremstyle{definition}\newtheorem{definition}[theorem]{Definition}`.

## 7. Notação: ênfase, displays, cross-references

- **Ênfase:** `\emph` é usado com parcimônia — **mediana de 2 ocorrências por paper** (quartis
  0 e 16; máximo 71) na amostra combinada de 47 fontes; `\textit` para ênfase é praticamente
  inexistente (mediana 0). O uso canônico de itálico é *para o termo sendo definido* dentro de
  uma `definition` (que em `\theoremstyle{definition}` já tem corpo romano, por isso o itálico
  marca o definiendum).
  `[PRATICA COMUM observada — amostras C, D e E]`
- **Displays numerados:** mediana de **21 equações rotuladas por paper**; e **~20–25% dos rótulos
  nunca são referenciados** (mediana 0,20 da fração não-referenciada). Ou seja, a regra "só numere
  o que você cita" é seguida de forma frouxa; numerar um display de destaque sem citá-lo depois
  não é violação de norma.
  `[PRATICA COMUM observada — amostra E]`
- **Referência a equações:** IOP proíbe explicitamente escrever a palavra: "**Do not include
  'eq.' 'equation' etc before an equation number or 'ref.' 'reference' etc before a reference
  number**"; e "The words table and figure should be written in full and not abbreviated to tab.
  and fig."
  `[VERIFICADO em IOP LaTeX guidelines §4.4]`
  (Elsevier faz o oposto em apêndices: "Eq. (A.1)". Regra é por venue.)
- **cleveref/hyperref:** SIAM recomenda `cleveref` e o carrega; usa `\cref`/`\Cref` e mostra o
  idioma `\cref{cl:constant,thm:mvt,hyp1,hyp2}` para referência múltipla mista.
  `[VERIFICADO em docsiamart.tex §"Cross references and hyperlinks"]`
  17/22 dos publicados nos três venues carregam hyperref e/ou cleveref.
  `[PRATICA COMUM observada — amostra D]`
- **Numeração de seções:** IOP e Elsevier numeram automaticamente 1, 1.1, 1.1.1; Elsevier:
  "Divide your article into clearly defined, logically organized, and **numbered** sections ...
  (the abstract is not included in section numbering)" e "Use this numbering also for internal
  cross-referencing: **do not just refer to 'the text'**".
  `[VERIFICADO em Physica D GfA 2024-02-18; IOP LaTeX guidelines §4.1]`

## 8. Figuras e tabelas

- **Legenda de figura vai EMBAIXO.** Medição direta no fonte LaTeX (posição do `\caption`
  relativa ao `\includegraphics` dentro de cada ambiente `figure`): **37 abaixo, 0 acima** na
  amostra de math.DS curto, e o mesmo padrão (100% abaixo) na amostra dos três venues. Zero
  exceções em 47 fontes.
  `[PRATICA COMUM observada — amostras C e D]` (é também o comportamento default do LaTeX quando
  se escreve `\caption` depois do gráfico)
- **Legenda de tabela vai EM CIMA:** "Table captions should go above the tables."
  `[VERIFICADO em docsiamart.tex §"Tables"]`. Elsevier: notas de tabela **abaixo** do corpo da
  tabela; "avoid using vertical rules and shading in table cells".
  `[VERIFICADO em Physica D GfA 2024-02-18]`
- **Conteúdo da legenda:** "A caption should comprise **a brief title (not on the figure itself)
  and a description of the illustration**. Keep text in the illustrations themselves to a minimum
  but explain all symbols and abbreviations used." (Elsevier)
  `[VERIFICADO em Physica D GfA 2024-02-18]`
  IOP: "As much lettering as possible should be removed from the figure itself and included in
  the caption. **If a figure has parts, these should be labelled (a), (b), (c), etc and all parts
  should be described within a single caption.**"
  `[VERIFICADO em IOP LaTeX guidelines §9.3]`
- **Posição na página:** SIADS exige figuras **inline**: "Non-supplemental images/figures, if any,
  must be embedded 'inline' in the manuscript" (nada de figuras no fim).
  `[VERIFICADO em SIADS Instructions for Authors, snapshot 2023-04-05]`
  Na prática, `\begin{figure}[tbhp]` (SIAM) ou `[htbp]` são os floats usados.
  `[VERIFICADO em docsiamart.tex / ex_article.tex]`
- Cor: SIADS "welcomes interesting, appealing color images and authors are encouraged to include
  color figures". `[VERIFICADO em SIADS IfA, snapshot 2023-04-05]`

## 9. Citações e bibliografia

- **Physica D (regra mais rígida dos três):** "Text: Indicate references by **number(s) in square
  brackets** in line with the text. The actual authors can be referred to, but the reference
  number(s) must always be given. ... List: **Number the references in the order in which they
  appear in the text**." Formato de artigo:
  `[1] J. van der Geer, J.A.J. Hanraads, R.A. Lupton, The art of writing a scientific article,
  J. Sci. Commun. 163 (2010) 51–59. https://doi.org/10.1016/j.Sc.2010.00372.`
  Nomes de periódicos abreviados pela *List of Title Word Abbreviations*.
  **Mas:** "There are **no strict requirements on reference formatting at submission**. References
  can be in any style or format as long as the style is consistent." (*Your Paper Your Way*)
  `[VERIFICADO em Physica D GfA, snapshot 2024-02-18]`
- **Nonlinearity/IOP:** Harvard (autor-ano) **ou** Vancouver (numérico) — "authors should use the
  style they are comfortable with". No estilo numérico: `\section*{References}` +
  `\begin{thebibliography}{<num>}`. Formato: autores em romano (sobrenome + iniciais sem pontos),
  ano **sem parênteses**, periódico em itálico e abreviado, **volume em negrito**, páginas em
  romano: `Cisneros A 1971 Astrophys. Space Sci. 10 87`.
  `[VERIFICADO em IOP LaTeX guidelines §§6.1–6.5]`
- **SIADS:** "References should be listed in **either alphabetical order or order of citation**."
  Formato canônico:
  `[16] L. N. Trefethen, Cubature, approximation, and isotropy in the hypercube, SIAM Rev., 59
  (2017), pp. 469–491.` — títulos de periódicos abreviados **conforme Mathematical Reviews**.
  Variantes aceitas: rótulo alfanumérico `[T]` ou autor-ano. Citação com página:
  `[2, p. 51]` ou `Mandelbrot [2, p. 51]`. BibTeX: `\bibliographystyle{siamplain}`.
  **"Authors are required to provide DOIs."**
  `[VERIFICADO em SIADS IfA (snapshot 2023-04-05) e docsiamart.tex §"Bibliography"]`
- **arXiv IDs na bibliografia:**
  - SIAM: campos BibTeX `eprint` e `eprintclass` (`archive` default `http://arXiv.org/abs`,
    `archiveprefix` default "arXiv") — produz `arXiv:1403.2226` com hyperlink.
    `[VERIFICADO em docsiamart.tex §"Preprint servers such as arXiv" + references.bib]`
  - IOP: `[2] Harrison M 1999 Diffeomorphism-invariant manifolds Preprint hep-th/9909196`,
    e "**Preprints may be referenced but if the article concerned has been published in a
    peer-reviewed journal, that reference should take precedence.** If only a preprint reference
    can be given, it is helpful to include the article title."
    `[VERIFICADO em IOP LaTeX guidelines §6.7]`
  - Elsevier: "Preprints should be clearly marked as such, for example by including the word
    preprint, or the name of the preprint server, as part of the reference. **The preprint DOI
    should also be provided.**"
    `[VERIFICADO em Physica D GfA 2024-02-18]`
  - Prática medida na amostra combinada: a forma **`arXiv:2301.01234`** (nua, eventualmente
    seguida de ponto) domina — 91 ocorrências contra 15 da forma `arXiv preprint arXiv:...`
    (esta última é artefato do BibTeX do Google Scholar e é minoria).
    `[PRATICA COMUM observada — amostras C e D]`
- **BibTeX vs bibliografia à mão:** nas notas curtas de math.DS, **15/25 escrevem
  `thebibliography` à mão** e 9/25 usam BibTeX. Entre os publicados, os `.bst` observados foram
  `siamplain`, `elsarticle-num`, `abbrv`, `amsplain`, `amsref`, `plain`, `unsrt`.
  `[PRATICA COMUM observada — amostras C e D]`
  Para uma nota de ~30 referências, bibliografia à mão é prática majoritária e não penaliza.

## 10. Seções canônicas e ordem

Ordem verificada em guia:

- **Elsevier/Physica D:** Introduction ("State the objectives of the work and provide adequate
  background and motivation") → corpo ("Organize your article into sections that correspond to
  essential steps in the development of the argument ... Lengthy calculations should preferably
  be placed in appendices") → Discussion ("should explore the significance of the work, **not
  repeat the results**") → Conclusions ("short Conclusions section, which may stand alone or be
  part of the Discussion") → **Acknowledgements** ("in a separate section at the end of the
  article **before the references**") → Appendices (A, B; Eq. (A.1), Fig. A.1, Table A.1).
  `[VERIFICADO em Physica D GfA, snapshot 2024-02-18]`
- **IOP:** "The first section is normally an introduction, which should state clearly the object
  of the work, its scope and the main advances reported"; Acknowledgments = "unnumbered
  'Acknowledgments' section **immediately following the last numbered section**" (`\ack`);
  apêndices "**after the acknowledgments section (if any) but before the reference list**".
  `[VERIFICADO em IOP LaTeX guidelines §§4.1–4.3]`
- **SIAM:** esqueleto do `ex_article.tex`: Introduction → Main results → Algorithm → Experimental
  results → Discussion → Conclusions → `\appendix` → `\section*{Acknowledgments}` → bibliografia.
  Ou seja, no SIAM os **agradecimentos vêm DEPOIS dos apêndices**, ao contrário da IOP.
  A introdução tem exigência própria: "All SIADS papers must have clearly written introductions in
  which the authors outline their new results and explain why their work is of interest. The
  introduction should help the reader **decide whether to read the details**."
  `[VERIFICADO em ex_article.tex + SIADS IfA, snapshot 2023-04-05]`
  O `ex_article.tex` traz também o idioma do parágrafo de roteiro: "The paper is organized as
  follows. Our main results are in §…, our new algorithm is in §…, experimental results are in §…,
  and the conclusions follow in §…". `[VERIFICADO em ex_article.tex]`

Prática medida (amostra C, 25 notas curtas de math.DS): a **primeira seção é "Introduction" em
20/25** (variantes: "Introduction and main results", "Introduction and background"). Seções
recorrentes: "Preliminaries" (5), "Proof of the main theorem" (3), "Conclusion(s)" (3),
"Acknowledgements" (3).
`[PRATICA COMUM observada — amostra C]`

Amostra D (22 publicados em Nonlinearity/Physica D/SIADS): seção de conclusão/discussão/outlook
em **13/22**; seção de numérica/exemplos ("Numerical Results", "Computational Results",
"Examples", "Numerical simulations", "Examples and simulations", "Validation by numerical
simulations", "Numerical investigation") em **13/22**; apêndice em 12/22; seção de agradecimentos
em 8/22 (nos demais o agradecimento é `\thanks` ou parágrafo final).
`[PRATICA COMUM observada — amostra D]`

**Esqueleto observado para o gênero "teorema + experimentos"** (união dos casos da amostra D):

```
1. Introduction              (problema, o que é novo, por que interessa, roteiro)
2. Setup and notation        (às vezes "Preliminaries")
3. Main result               (enunciado do teorema; prova aqui se curta)
4. Proof                     (seção própria se a prova passa de ~2 páginas)
5. Numerical experiments     (protocolo → resultado → figura/tabela)
6. Discussion / Limitations
7. Conclusions               (curta)
Acknowledgments (não numerada)
Data availability / Declarations
References
Appendix A, B ...            (posição depende do venue: SIAM depois; IOP antes dos refs)
```

## 11. Agradecimentos, financiamento, dados e código

- **Financiamento:**
  - SIAM: no `\thanks` **do título**. `[VERIFICADO em docsiamart.tex §"Front matter"]`
  - IOP: **na seção de agradecimentos**, explicitamente *não* em nota de título.
    `[VERIFICADO em IOP LaTeX guidelines §3.1]`
  - Elsevier: seção separada no fim, antes das referências; "Authors must disclose any funding
    sources". `[VERIFICADO em Physica D GfA, snapshot 2024-02-18]`
- **Declaração de dados:**
  - **IOP: obrigatória.** "The journal **requires** authors to include a data availability
    statement in their article"; a verificação é **condição de publicação**; e "stating that your
    data will be made available upon request **is not sufficient** to comply".
    `[VERIFICADO em publishingsupport.iopscience.iop.org — IOP Publishing research data
    availability policy, acesso 2026-07-25]`. Nonlinearity adotou a política padrão de dados da
    IOP. `[VERIFICADO em publishingsupport.iopscience.iop.org/journals/nonlinearity/]`
  - **Elsevier/Physica D:** *Data statement* encorajada ("we encourage you to state the
    availability of your data in your submission"), com opção de justificar indisponibilidade;
    aparece publicada junto ao artigo. Software/código deve ser **citado como referência**
    (criador, título, repositório, versão, ano, identificador persistente), e datasets levam
    `[dataset]` antes da entrada.
    `[VERIFICADO em Physica D GfA, snapshot 2024-02-18]`
  - **SIADS:** o guia não exige declaração de dados; o que existe é o canal de **Supplementary
    Materials** revisado por pares junto com o manuscrito, com um índice obrigatório descrevendo
    e justificando cada item.
    `[VERIFICADO em SIADS IfA, snapshot 2023-04-05]`
  - Prática medida: seção explícita de "Data availability" aparece em **2/25** das notas curtas de
    math.DS e em 2/22 dos publicados (a de Physica D tinha `\section*{Data availability}` depois
    de Acknowledgments). Ou seja, **ainda é minoria em DS** — mas é obrigatória se o alvo for
    Nonlinearity.
    `[PRATICA COMUM observada — amostras C e D]`
- **Conflito de interesse:** Elsevier exige a *declarations tool* preenchida sempre; se houver uso
  de IA generativa na escrita, exige seção intitulada "Declaration of Generative AI and AI-assisted
  technologies in the writing process" **antes da lista de referências**.
  `[VERIFICADO em Physica D GfA, snapshot 2024-02-18]`

**Comprimento:**
- SIADS: "Generally, manuscripts should not exceed **40 pages** in length and/or ten megabytes";
  se não usar as macros SIAM, "use at least 10pt font and set the margins so that the text width
  and height are no more than 6in and 8in".
  `[VERIFICADO em SIADS IfA, snapshot 2023-04-05]`
- IOP: não publica limite numérico; "some of our journals have guidelines for the maximum
  recommended length" e remete à página do periódico. A página do periódico lista dois tipos de
  contribuição: **research papers** (relatos de pesquisa original) e **invited articles**
  (encomendados pelo corpo editorial) — nenhuma categoria de *Letter* é anunciada ali. (A classe
  `iopart` define `\letter{}` entre seus tipos de artigo, mas o guia não diz a quais periódicos
  se aplica: **não verificado** se Nonlinearity aceita Letters.)
  `[VERIFICADO em publishingsupport.iopscience.iop.org — Article length; About Nonlinearity, acesso 2026-07-25]`
- Physica D: sem limite publicado.
- **Distribuição real em math.DS (julho/2026, 137 papers com contagem declarada):** mediana
  **27 páginas**; ≤8 pp: 6%; ≤12 pp: 13%; ≤20 pp: 31%; ≤30 pp: 58%.
  Em math.CA (76 papers): mediana **21 páginas**; ≤12 pp: 14%.
  Uma nota de 8–14 páginas está no **decil–quintil inferior**: é curta para o campo, mas nada
  anômalo (25 papers do mês em math.DS tinham ≤14 pp).
  `[PRATICA COMUM observada — amostras A e B]`
- **Campo `Comments` do arXiv:** o costume é "N pages, M figures"; 137/282 declaram páginas e
  76/282 declaram figuras; 15/282 acrescentam "comments welcome". O arXiv confirma que o campo
  serve para "page count, number of figures" e para "to be published in ...".
  `[VERIFICADO em info.arxiv.org/help/prep.html]` + `[PRATICA COMUM observada — amostra A]`

## 12. Amostras usadas (para auditoria/reprodução)

Toda estatística acima é reproduzível a partir destes conjuntos (baixados em 2026-07-25):

- **Amostra A — math.DS, mês corrente:** `https://arxiv.org/list/math.DS/2026-07?skip=0&show=500`
  → 282 entradas; 137 com contagem de páginas no campo Comments.
- **Amostra B — math.CA, mês corrente:** `https://arxiv.org/list/math.CA/2026-07?skip=0&show=500`
  → 136 entradas; 76 com contagem de páginas.
- **Amostra C — 25 notas curtas (≤14 pp) de math.DS, fonte LaTeX baixado de `arxiv.org/e-print/`:**
  2607.01386, 2607.03244, 2607.03257, 2607.03979, 2607.04174, 2607.04453, 2607.04967, 2607.05703,
  2607.08155, 2607.08676, 2607.10536, 2607.10920, 2607.11330, 2607.11549, 2607.11748, 2607.11804,
  2607.14693, 2607.14885, 2607.15832, 2607.17180, 2607.18690, 2607.19203, 2607.20273, 2607.21376,
  2607.21398.
  (Classes: amsart 11, article 10, e 4 de outras comunidades — IEEEtran/ieeeconf/pnas/ejpecp.)
- **Amostra D — 22 papers com `journal-ref` para os três venues-alvo** (busca avançada do arXiv
  por campo journal_ref, restrita a matemática), fonte LaTeX baixado:
  - Nonlinearity: 2405.11709, 2507.07918, 2507.12051, 2509.16435, 2510.11911, 2512.10460, 2601.15134
  - Physica D: 2409.16166, 2501.07181, 2502.19972, 2504.10513, 2505.03239, 2508.05927, 2510.21937, 2511.07609
  - SIADS: 2308.14167, 2402.03026, 2403.04625, 2403.11591, 2403.16765, 2411.11402, 2504.02776
- **Amostra E** = C ∪ D (47 fontes) para as métricas de `\emph` e de equações rotuladas.

Fontes normativas lidas (com data):

| Fonte | Como foi lida | Data |
|---|---|---|
| SIADS *Instructions for Authors* | Wayback (`epubs.siam.org` bloqueia acesso direto) | snapshot 2023-04-05 |
| SIAM `docsiamart.tex` (Guide to Using SIAM's LaTeX Style) | GitHub `tgkolda/siam-latex` | lido 2026-07-25 |
| SIAM `ex_article.tex`, `references.bib` | GitHub `tgkolda/siam-latex` | lido 2026-07-25 |
| SIAM `siamart220329.cls` | cópia pública no GitHub | classe v1.4.4, 2022-03-29 |
| IOP *How to prepare and submit an article ... using LaTeX2e* | espelho `li.mit.edu` | PDF datado 2005-06-27 |
| IOP *Article structure* / *Article length* / *About Nonlinearity* / *Data availability policy* | publishingsupport.iopscience.iop.org | acesso 2026-07-25 |
| Physica D *Guide for Authors* | Wayback (ScienceDirect exige captcha) | snapshot 2024-02-18 |
| `elsdoc.pdf` (elsarticle.cls) | CTAN (`ctan.math.illinois.edu`) | v3.5, doc 2026-01-09 |
| `amsart-template.tex`, `amsclass.pdf`, `amsthdoc.pdf` | TeX Live local | template v1.1 2014-07-24 |
| arXiv *Preparing your submission* | info.arxiv.org/help/prep.html | acesso 2026-07-25 |
| MSC2020 | mathscinet.ams.org/mathscinet/msc/msc2020.html | acesso 2026-07-25 |

## 13. Não verificado / em aberto

- **SIAM Style Manual** (o manual de copidesque, `stylemanual.pdf`): inacessível — `siam.org`,
  `archive.siam.org` e `epubs.siam.org` retornaram 403, e não há snapshot no Wayback. Regras
  finas de estilo SIAM (uso de "we", tempo verbal, hifenização, itálico) **não foram verificadas
  na fonte**; o que está neste dossiê sobre voz/tempo é medição de prática, não norma.
- **Limite de páginas específico de Nonlinearity:** a IOP remete à página do periódico e a página
  não publica número. Não verificado.
- **Guia LaTeX atual da IOP:** o único documento LaTeX completo que consegui abrir é de 2005.
  A classe `iopart` continua em uso (confirmado por submissão real de 2025 a Nonlinearity), mas
  se houver revisão posterior do guia, ela não foi lida. Onde a política web atual conflita com o
  PDF de 2005 (200 vs 300 palavras de abstract), os dois números estão registrados.
- **Política de RCR (Reproducibility of Computational Results) em SIADS:** o guia lido menciona
  *Supplementary Materials* revisados por pares, mas não um selo/política RCR. Não verificado.

## 14. Checklist para a nossa nota

Cada item abaixo é uma regra verificada acima; a coluna "porquê" aponta a norma que a obriga.

1. Título em sentence case, uma linha, sem `\\`. — IOP §3.1.
2. Autor + afiliação + e-mail + (se houver) financiamento no bloco de título; financiamento em
   `\thanks` se SIAM, em Acknowledgments se IOP/Elsevier.
3. Abstract: **um parágrafo, ≤200 palavras, ≤1920 caracteres, zero `\cite`, zero referência a
   equação/figura/tabela numerada, matemática só inline**. Números de resultado (n, p, β) podem
   e devem estar lá.
4. Keywords: ≤6, logo depois do abstract.
5. MSC 2020 com Primary/Secondary — obrigatório se SIADS, recomendado se Nonlinearity.
   Candidatos plausíveis para esta nota (a **confirmar na tabela oficial da MSC2020 antes de
   usar** — não verificados aqui): família 34E (perturbação singular/métodos assintóticos para
   EDOs), 37 (sistemas dinâmicos), 68T07 (redes neurais). **Não copiar sem checar.**
6. Contador de teoremas **único e compartilhado** (Theorem 1, Lemma 2, Definition 3), ou por
   seção no estilo AMS — nunca contadores paralelos.
7. `\begin{proof}` com QED automático; prova longa (>2 pp) vira seção própria.
8. Legenda de figura **embaixo**; legenda de tabela **em cima**; partes da figura rotuladas
   (a), (b), (c) e descritas numa **única** legenda; figuras inline, não no fim.
9. Citação numérica `[1]` em ordem de aparição (satisfaz Physica D e é aceita por IOP e SIADS);
   DOI em todas as entradas que tiverem; preprints marcados como `arXiv:XXXX.XXXXX` com título.
10. Seções: Introduction → Setup → Main result → Proof → Experiments → Discussion/Limitations →
    Conclusions → Acknowledgments (não numerada) → Data availability → References → Appendices.
11. Parágrafo de roteiro no fim da introdução ("The paper is organized as follows…") — idioma do
    template SIAM.
12. Declaração de disponibilidade de dados/código com **link real** (não "upon request") —
    obrigatória para Nonlinearity, e barata para nós, já que os `results/*.jsonl` estão no repo.
13. Comments do arXiv: "N pages, M figures" (+ "comments welcome" se quisermos).
14. Não abreviar "figure"/"table"/"equation" no corpo, nem escrever "eq. (3)" — IOP §4.4.
