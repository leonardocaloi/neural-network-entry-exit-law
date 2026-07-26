# Padrão real de papers de teoria de deep learning (arXiv cs.LG / stat.ML, estilo NeurIPS/ICML/JMLR)

Data: 2026-07-25. Autor do levantamento: agente de pesquisa (j-space).
Escopo: convenções aplicáveis a uma nota/paper **postado no arXiv sem submissão a venue**,
escrito no estilo das conferências. Alvo direto: `06-notas/paper/paper-v1.tex`.

## Como ler as etiquetas

Este documento usa as duas etiquetas pedidas na tarefa, que se mapeiam assim na
etiquetagem canônica do projeto (CLAUDE.md):

- **`[VERIFICADO em <fonte>]`** — a fonte foi **aberta e lida** neste levantamento
  (página oficial, PDF extraído, ou HTML do próprio paper). Equivale a
  `[ESTABELECIDO]`/`[RESULTADO 2024-26]` com fonte nomeada. Citações entre aspas são
  transcrições literais da fonte.
- **`[PRATICA COMUM]`** — **não existe norma escrita** que eu tenha localizado; é padrão
  observado em papers concretos que abri. Sempre digo **quantos** e **quais**. Equivale a
  `[NÃO VERIFICADO]` promovido a evidência amostral: é indução sobre uma amostra pequena,
  não é regra.
- Onde não achei evidência, está escrito **"sem evidência localizada"**. Nada foi inventado.

Aviso de amostra: as generalizações de `[PRATICA COMUM]` vêm de **6 papers abertos**
(arXiv 2501.17745, 2308.12108, 2402.02364, 2410.17592, 1806.07572; PMLR v148/chandra21a,
v181/narayanan22a) mais **40 títulos** da listagem `stat.ML/recent`. Amostra pequena e
enviesada para SLT/NTK. Tratar como orientação, não como lei.

---

## 1. Formato e estilo do arquivo (arXiv sem submissão)

**1.1 Style file com opção `preprint`.**
`[VERIFICADO em arXiv:2506.15953v1, "Formatting Instructions For NeurIPS 2025"]`
O `.sty` do NeurIPS tem três argumentos opcionais: `final` (câmera-ready), `preprint`
(preprint para, e.g., arXiv) e `nonatbib`. A opção `preprint` produz **versão
não-anonimizada com o rodapé "Preprint. Work in progress."**. Na submissão real omitem-se
`final` e `preprint` (anonimiza + numera linhas). Uso: `\usepackage[preprint]{neurips_2025}`.

**1.2 ICML: opção `accepted`.**
`[VERIFICADO em arXiv:2412.10032v2, "Submission and Formatting Instructions for ICML 2025"]`
`\usepackage{icml2025}` gera a nota de rodapé de "preliminary work"; `\usepackage[accepted]{icml2025}`
troca pelo boilerplate de proceedings. Para preprint sem aceite, mantém-se **sem** `accepted`.

**1.3 Limites de página (referência de escala, não obrigação para arXiv).**
- NeurIPS 2025 `[VERIFICADO em arXiv:2506.15953v1]`: corpo com figuras ≤ **9 páginas**;
  "References, checklist, and optional technical appendices do not count as content pages";
  apêndices técnicos **sem limite de página**.
- ICML 2025 `[VERIFICADO em icml.cc/Conferences/2025/AuthorInstructions]`: submissão com
  corpo de até **8 páginas**, câmera-ready **9**; "followed by any number of pages for
  references and appendices"; "Unlimited appendices are allowed".
- ICLR 2026 `[VERIFICADO em iclr.cc/Conferences/2026/AuthorGuide]`: submissão **9 páginas**,
  câmera-ready **10**; "The list of references does not count towards the page limit";
  apêndices **depois** da bibliografia, ilimitados, "reviewers are not required to read the appendix".
- JMLR `[VERIFICADO em jmlr.org/author-info.html]`: até **35 páginas incluindo apêndices**;
  acima de 50 exige justificativa; abstract **≤ 200 palavras**; **5 keywords**; running title
  de ≤ 50 caracteres. JMLR aceita explicitamente trabalho já no arXiv.

*Leitura para o j-space:* 11 páginas de corpo + apêndices é coerente com o registro
"conference-style preprint". Nada obriga 9 páginas fora de submissão, mas **corpo enxuto +
apêndice gordo** é o formato que o leitor da área espera.

**1.4 Abstract.**
`[VERIFICADO em arXiv:2506.15953v1]` NeurIPS: "must be limited to one paragraph".
`[VERIFICADO em arXiv:2412.10032v2]` ICML: parágrafo único auto-contido, idealmente **4–6 frases**.
`[VERIFICADO em info.arxiv.org/help/prep.html]` arXiv: limite **duro de 1920 caracteres**;
expandir macros TeX opacas; não escrever a palavra "Abstract" dentro do campo.

**1.5 Tabelas e figuras.**
`[VERIFICADO em arXiv:2506.15953v1]`
- Tabelas: "The table number and title always appear **before** the table";
  "Publication-quality tables do **not** contain vertical rules. We strongly suggest the use
  of the `booktabs` package."
- Figuras: "The figure number and caption always appear **after** the figure"; caption em
  minúsculas exceto primeira palavra e nomes próprios.
- Notas de rodapé "should be used sparingly".

**1.6 Citações.**
`[VERIFICADO em arXiv:2506.15953v1]` natbib carregado por padrão; autor/ano ou numérico,
"as long as you maintain internal consistency"; `\citet` para citação inline.
`[VERIFICADO em icml.cc/Conferences/2025/AuthorInstructions]` ICML pede: "replace arXiv
citations with peer-reviewed papers where possible (arXiv is generally not peer-reviewed)".

**1.7 Metadados arXiv (campo Comments, categorias).**
`[VERIFICADO em info.arxiv.org/help/prep.html]`
- Comments: "Indicate number of pages and number of figures"; e "This is the proper field for
  'to be published in' or 'submitted to' information". Formato observado
  `[PRATICA COMUM — arXiv:2501.17745]`: `37 pages, 27 figures`.
- Classes opcionais: `MSC-class` (formato `14J60 (Primary) 14F05, 14J26 (Secondary)`) e
  `ACM-class` (múltiplos separados por ponto-e-vírgula, e.g. `F.2.2; I.2.7`).
- Cross-list exige ler as descrições das categorias ("Please carefully review the descriptions
  of categories to find the best fit").

---

## 2. Estrutura macro

**2.1 Contribuições em bullets na introdução: comum, não universal.**
`[PRATICA COMUM]` — dos 4 papers cujo corpo li:
- arXiv:2501.17745 (Carroll, Hoogland, Farrugia-Roberts, Murfet, ICML 2025): **lista com
  bullets** na intro, aberta por "…contributing the following."
  `[VERIFICADO em ar5iv/2501.17745]`
- arXiv:2410.17592 ("A Kernel Perspective on Distillation-based Collaborative Learning"):
  **lista numerada** de 3 contribuições. `[VERIFICADO em arxiv.org/html/2410.17592v2]`
- arXiv:2308.12108 (Lau, Murfet, Wei — LLC): contribuições **em prosa**, sem lista.
  `[VERIFICADO em ar5iv/2308.12108]`
- arXiv:2402.02364 (Hoogland et al.): contribuições em parágrafos, na **conclusão**.
  `[VERIFICADO em ar5iv/2402.02364]`

Conclusão honesta: bullets são o modo dominante em papers empírico-teóricos recentes; papers
de teoria pura frequentemente usam prosa. **Não é norma.**

**2.2 Onde vai Related Work: varia, e a variação é significativa.**
`[PRATICA COMUM]` — quatro pontos de dado, dois de cada lado:
- **§2, logo após a introdução**: arXiv:2501.17745; arXiv:2410.17592.
- **Tarde, imediatamente antes ou junto dos experimentos/discussão**: arXiv:2308.12108
  (Related Work é **§9**, entre a teoria e os experimentos); arXiv:2402.02364
  (**§6 "Discussion and Related Work"**, depois de todos os resultados).

Regra observável, não escrita: quando o paper tem um **teorema como espinha**, a literatura
tende a ir depois da teoria; quando é empírico, vai no §2. Sem fonte normativa localizada.

**2.3 Esqueleto observado num paper teoria+experimento da área (exemplar mais próximo do j-space).**
`[VERIFICADO em ar5iv/2501.17745]` — Carroll et al., ICML 2025:

```
1. Introduction              (com bullets de contribuição)
2. Related Work
3. <setup formal>            (3.1 distribuições, 3.2 objetivo, 3.3 preditores idealizados)
4. <fenômeno>                (4.1 treino, 4.2 método de análise, 4.3 resultados)
5. <explicação teórica>      (5.1-5.4, terminando em "Empirical validation of the explanation")
6. Limitations and future work   (6.1-6.5, uma subseção por limitação)
7. Conclusion
8. Impact statement
9. Acknowledgments
References
Appendix A..I  (detalhes de treino, de avaliação, de método, de estimação, figuras extras)
```

Note: **Limitations com subseções numeradas**, uma por limitação — não um parágrafo. E
"Impact statement" como seção não-numerada depois da conclusão (exigência ICML, §3.3 abaixo).

**2.4 Apêndices: um por bloco instrumental, nomeados pela função.**
`[VERIFICADO em ar5iv/2501.17745]`: "Transformer training details", "Transformer evaluation
details", "Trajectory PCA details", "Smoothing details", "LLC estimation details", "Additional
Figures and Code". `[VERIFICADO em ar5iv/2402.02364]`: A–F com "Experimental Details" e
"Additional Figures and Code" separados.
`[PRATICA COMUM]` O apêndice é organizado por **instrumento/protocolo**, não por ordem de
aparição no texto.

---

## 3. Limitations, Impact e Ethics (a norma pós-2021)

**3.1 Limitations — NeurIPS.**
`[VERIFICADO em neurips.cc/public/guides/PaperChecklist, acessado 2026-07-25]` — item 2 do
checklist. Texto literal das guidelines:
- "The authors are encouraged to create a separate 'Limitations' section in their paper."
- "The paper should point out any strong assumptions and how robust the results are to
  violations of these assumptions"
- "The authors should reflect on the scope of the claims made, e.g., if the approach was only
  tested on a few datasets or **with a few runs**."
- "In general, empirical results often depend on **implicit assumptions**, which should be
  articulated."
- "The authors should reflect on the factors that influence the performance of the approach."

`[VERIFICADO em neurips.cc/Conferences/2022/PaperInformation/PaperChecklist]` A formulação de
2022 já pedia seção de limitações cobrindo "strong assumptions and how robust your results are
to violations" — ou seja, **a norma é pós-2021 e estável desde então**.

**3.2 Limitations em ICML/ICLR.**
`[VERIFICADO em icml.cc/Conferences/2025/AuthorInstructions]`: as instruções ICML **não**
exigem explicitamente uma seção "Limitations" (o que exigem é o Impact Statement).
`[VERIFICADO em iclr.cc/Conferences/2026/AuthorGuide]`: idem — ICLR normatiza Reproducibility
Statement e Ethics Statement, não Limitations.
Conclusão: **a exigência formal de "Limitations" é do NeurIPS**; nos outros é
`[PRATICA COMUM]` importada.

**3.3 Impact statement (ICML) — obrigatório, com boilerplate oficial.**
`[VERIFICADO em icml.cc/Conferences/2025/AuthorInstructions e arXiv:2412.10032v2]`
"Your camera-ready pdf should include a mandatory impact statement just before the
bibliography", fora do limite de páginas. Boilerplate que a própria ICML declara suficiente
quando não há impacto específico a destacar:

> "This paper presents work whose goal is to advance the field of Machine Learning. There are
> many potential societal consequences of our work, none which we feel must be specifically
> highlighted here."

**3.4 Ethics statement (ICLR) — opcional.**
`[VERIFICADO em iclr.cc/Conferences/2026/AuthorGuide]` "Authors are encouraged to include a
paragraph of Ethics Statement (at the end of the main text before references)"; não conta no
limite de páginas; ≤ 1 página.

**3.5 Declaração de uso de LLM.**
`[VERIFICADO em iclr.cc/Conferences/2026/AuthorGuide]` "If LLMs played a significant role in
research ideation and/or writing … authors should describe the precise role of the LLM in a
separate section", que pode ir no apêndice e não conta no limite; "Not disclosing significant
LLM usage can lead to desk rejection".
`[VERIFICADO em neurips.cc/public/guides/PaperChecklist]` É o **item 16** do checklist:
"Declaration of LLM usage".

---

## 4. Reproducibility statement

**4.1 ICLR: a formulação normativa mais explícita da área.**
`[VERIFICADO em iclr.cc/Conferences/2026/AuthorGuide]`
"Authors are strongly encouraged to include a paragraph-long **Reproducibility Statement at the
end of the main text (before references)**" discutindo os esforços de reprodutibilidade;
"will not count toward the page limit"; deve **apontar para** as partes do corpo, apêndice e
material suplementar que ajudam a reproduzir (é um **índice**, não uma repetição).

**4.2 NeurIPS: sem seção obrigatória, mas duas perguntas do checklist.**
`[VERIFICADO em neurips.cc/public/guides/PaperChecklist]`
- Item 4 (Experimental Result Reproducibility): "what steps did you take to make your results
  reproducible or verifiable?"
- Item 5 (Open Access to Data and Code), guidelines literais:
  - "The instructions should contain the **exact command and environment** needed to run to
    reproduce the results."
  - "If a subset of experiments are reproducible, **you should state which ones are**."
  - "'no because the code is proprietary' is an acceptable answer."
  - "At submission time, to preserve anonymity, remember to release anonymized versions."

**4.3 ICML.**
`[VERIFICADO em icml.cc/Conferences/2025/AuthorInstructions]` "Authors are encouraged to submit
code to foster reproducibility. **Reproducibility of results and easy availability of code will
be taken into account in the decision-making process.**"

**4.4 JMLR.**
`[VERIFICADO em jmlr.org/author-info.html]` recomenda explicitamente **consultar o checklist de
reprodutibilidade de Joelle Pineau**; exige que afirmações experimentais sejam "clearly
articulated and supported", com experimentos reprodutíveis.

**4.5 Machine Learning Reproducibility Checklist v2.0 — texto literal.**
`[VERIFICADO em cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf, extraído com pdftotext]`
Cabeçalho: "The Machine Learning Reproducibility Checklist (v2.0, Apr.7 2020)".

*For all models and algorithms presented:* descrição clara do setting matemático/algoritmo/modelo;
explicação clara de **quaisquer suposições**; análise de complexidade (tempo, espaço, tamanho de amostra).

*For any theoretical claim:* "A clear statement of the claim"; "A complete proof of the claim".

*For all datasets used:* estatísticas relevantes; splits train/val/test; explicação de **dados
excluídos** e de todo pré-processamento; link para versão baixável do dataset ou ambiente de
simulação; para dados novos, descrição completa da coleta.

*For all shared code:* "Specification of dependencies"; "Training code"; "Evaluation code";
"(Pre-)trained model(s)"; "README file includes table of results accompanied by **precise
command to run** to produce those results".

*For all reported experimental results:*
- "The range of hyper-parameters considered, method to select the best hyper-parameter
  configuration, and specification of all hyper-parameters used to generate results."
- "**The exact number of training and evaluation runs.**"
- "A clear definition of the specific measure or statistics used to report results."
- "A description of results with **central tendency (e.g. mean) & variation (e.g. error bars)**."
- "The average runtime for each result, or estimated energy cost."
- "A description of the computing infrastructure used."

Contexto: `[VERIFICADO em jmlr.org/papers/v22/20-303.html]` Pineau, Vincent-Lamarre, Sinha,
Larivière, Beygelzimer, d'Alché-Buc, Fox, Larochelle, "Improving Reproducibility in Machine
Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)", JMLR 2021 — é o
paper que documenta a adoção do checklist no NeurIPS 2019.

**4.6 ML Code Completeness Checklist (5 itens) — o que o repositório deve conter.**
`[VERIFICADO em github.com/paperswithcode/releasing-research-code]` (título do repo: "Tips for
releasing research code in Machine Learning (with official NeurIPS 2020 recommendations)"):
1. Specification of dependencies (`requirements.txt` / `environment.yml` / `setup.py`)
2. Training code
3. Evaluation code
4. Pre-trained models
5. README com **tabela de resultados acompanhada do comando preciso** que os reproduz.

---

## 5. O checklist NeurIPS como padrão de auditoria interna

**5.1 As 16 perguntas.**
`[VERIFICADO em neurips.cc/public/guides/PaperChecklist, acessado 2026-07-25]`
1. Claims — 2. Limitations — 3. Theory, Assumptions and Proofs — 4. Experimental Result
Reproducibility — 5. Open Access to Data and Code — 6. Experimental Setting/Details —
7. Experiment Statistical Significance — 8. Experiments Compute Resource — 9. Code of Ethics —
10. Broader Impacts — 11. Safeguards — 12. Licenses — 13. Assets — 14. Crowdsourcing and
Research with Human Subjects — 15. IRB Approvals — 16. Declaration of LLM usage.

**5.2 Formato da resposta.**
`[VERIFICADO em neurips.cc/public/guides/PaperChecklist]` "You should answer yes, no, or n/a.
You should **reference the section(s) of the paper that provide support for your answer**. You
can also optionally write a short (1–2 sentence) justification."
`[PRATICA COMUM]` No LaTeX dos papers o formato impresso é o tripé
`Question: … / Answer: [Yes] / Justification: …`.

**5.3 Onde o checklist vai fisicamente.**
`[VERIFICADO em arXiv:2506.15953v1]` "The checklist should follow the references and follow the
(optional) supplemental material. The checklist does NOT count towards the page limit."

**5.4 Perguntas 1 e 3, texto literal (as que mordem um paper de teoria).**
`[VERIFICADO em neurips.cc/public/guides/PaperChecklist]`
- **1 (Claims):** "Do the main claims made in the abstract and introduction accurately reflect
  the paper's contributions and scope?"
- **3 (Theory):** "did you state the **full set of assumptions** of all theoretical results, and
  did you include **complete proofs**?" — suposições devem estar **no enunciado do teorema**;
  provas no corpo ou no suplemento; recomenda-se **proof sketch** para intuição.

*Uso no j-space:* rodar as 16 perguntas contra o draft **antes de postar**, e arquivar as
respostas em `00-admin/` mesmo sem submeter. É exatamente o "patch test" editorial: o
checklist é auditoria interna barata e verificável.

---

## 6. Código e commits: como se referencia repositório num paper

**6.1 Nota de rodapé com URL, no corpo.**
`[VERIFICADO em PMLR v148, Chandra et al., "On Initial Pools for Deep Active Learning"]` —
nota de rodapé literal: "3. Our code is available at https://github.com/acl21/init-pools-dal".
`[PRATICA COMUM]` As duas posições dominantes são: (a) nota de rodapé na primeira página ou na
seção de experimentos; (b) frase "Code is available at <URL>" no fim do abstract.

**6.2 Ponteiro no apêndice.**
`[VERIFICADO em ar5iv/2402.02364]` (Hoogland et al.): o apêndice F, "Additional Figures and
Code", é que "contains a link to additional figures and the codebase used to run the
experiments in this paper" — o corpo só aponta para o apêndice.

**6.3 Anonimização sob double-blind.**
`[VERIFICADO em icml.cc/Conferences/2025/AuthorInstructions]` "Authors should anonymize code by
removing author names and licenses, or use anonymous GitHub branches"; aceita zip, PDF ou link
de repositório anônimo.
`[VERIFICADO em neurips.cc/public/guides/PaperChecklist, item 5]` "At submission time, to
preserve anonymity, remember to release anonymized versions."
`[VERIFICADO em 4open.science]` O serviço **anonymous.4open.science** (Anonymous GitHub)
existe para isso: autentica com a conta GitHub, recebe a URL do repo e uma lista de termos a
mascarar (nome, usuário, e-mail, instituição, nomes internos de projeto).
**Irrelevante para preprint sem submissão** — publicação no arXiv com autoria não é
double-blind e o repositório entra com nome real.

**6.4 Pinar versão: commit hash e/ou DOI de arquivo.**
`[PRATICA COMUM]` — não localizei norma escrita de nenhuma das quatro venues exigindo commit
hash. O que existe é:
- `[VERIFICADO em github.com/paperswithcode/releasing-research-code]` recomendação de usar
  **Zenodo** para versionar o release do código (DOI permanente por release/tag, integração com
  GitHub, 50 GB).
- `[VERIFICADO em cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf]` o checklist exige
  "Specification of dependencies" e comandos precisos — o espírito é pinagem, sem nomear commit.
- `[PRATICA COMUM]` papers de 2025–26 que citam "código arquivado no Zenodo sob DOI, pinado a
  commits específicos, com seeds e configs fixados" aparecem em busca, mas **não abri nenhum
  desses PDFs**; registro como prática relatada, não verificada.

*Leitura para o j-space:* citar **commit hash curto por experimento** (como `paper-v1.tex` já
faz: `a8f8f8c`, `5a961cb`, `1b45498`) é **mais forte** do que o padrão da área, não mais fraco.
O que falta para bater o padrão é o par (URL pública do repositório, comando exato de execução)
— a seção `\section{Reproducibility}` do draft já tem os comandos (`python run_4a3.py --workers 4`
etc.) mas os commits só são verificáveis se o repositório for público. Sem URL pública, a
convenção manda **dizer isso explicitamente** ("If a subset of experiments are reproducible, you
should state which ones are").

---

## 7. Apêndices de provas

**7.1 Provas fora do corpo, teoremas reenunciados.**
`[VERIFICADO em ar5iv/1806.07572, Jacot, Gabriel, Hongler — NTK]` Enunciados e argumento de alto
nível no corpo ("proven in the appendix", "proven in Appendix A.4"); no apêndice os resultados
são **reenunciados com a mesma numeração** antes da prova completa, na ordem Proposição 1,
Teorema 1, Teorema 2.

**7.2 Norma escrita que sustenta isso.**
`[VERIFICADO em neurips.cc/public/guides/PaperChecklist, item 3]` "All assumptions should be
clearly stated or referenced in the statement of any theorems"; provas podem ir "in the main
paper or supplemental material"; "**proof sketches** for intuition" são recomendados.
`[VERIFICADO em cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf]` "For any theoretical claim:
a clear statement of the claim; a complete proof of the claim."

**7.3 Provas com verificação executável.**
Sem norma localizada. `[PRATICA COMUM]` em SLT/devinterp: apêndices de "estimation details"
com hiperparâmetros de SGLD e calibração (`[VERIFICADO em ar5iv/2501.17745]`, Apêndice F "LLC
estimation details"). A prática do j-space de **um script executável por lema** (`prova_L1--L6.py`)
não tem paralelo que eu tenha localizado na amostra; é acima do padrão e vale **dizer no paper
que é isso que se está fazendo**, porque o leitor não vai assumir.

---

## 8. Tabelas de resultados com intervalos de confiança

**8.1 A norma escrita (a mais citável da área).**
`[VERIFICADO em neurips.cc/public/guides/PaperChecklist, item 7 "Experiment Statistical
Significance"]` — guidelines literais:
- "The authors should answer 'Yes' if the results are accompanied by error bars, confidence
  intervals, or statistical significance tests"
- "**The factors of variability that the error bars are capturing should be clearly stated**"
- "The method for calculating the error bars should be explained"
- "The assumptions made should be given"
- "It should be clear whether the error bar is the **standard deviation** or the **standard error
  of the mean**."
- "It is OK to report 1-sigma error bars, but **one should state it**."
- "For asymmetric distributions, the authors should be careful not to show in tables or figures
  symmetric error bars"
E, no corpo do checklist: "if error bars are reported in tables or plots, the authors should
explain **in the text** how they were calculated and reference the corresponding figures or
tables in the text."

`[VERIFICADO em neurips.cc/Conferences/2022/PaperInformation/PaperChecklist]` A formulação de
2022 era mais crua e ainda é boa régua: reportar "error bars (e.g., with respect to the random
seed after running experiments multiple times)".

**8.2 Como aparece de fato nas tabelas.**
`[VERIFICADO em ar5iv/2308.12108, Lau/Murfet/Wei]` Tabela 1(a) reporta "**mean and standard
deviation**" com entradas do tipo `0.17 (0.03)` — o desvio entre parênteses, não com `±`; o
texto diz que os experimentos sintéticos usaram **10 repetições** com seeds diferentes.
`[VERIFICADO em ar5iv/2402.02364, Hoogland et al.]` texto literal: "(a) natural language data
(**1 seed**) and (b) synthetic in-context regression data (**error bars over 5 seeds**)".
`[VERIFICADO em PMLR v181, Narayanan et al.]` "experiments across **three seeds** and report the mean".

`[PRATICA COMUM]` Formatos que coexistem: `média (desvio)`, `média ± desvio`, `média [lo, hi]`.
**O que a norma exige não é o formato e sim a legenda dizer o que a barra é e sobre o que ela
varia.** Erro mais comum penalizado: `±` sem dizer se é sd, sem, ou IC.

**8.3 Poucas seeds: a referência canônica.**
`[VERIFICADO em arXiv:2108.13264, Agarwal, Schwarzer, Castro, Courville, Bellemare, "Deep
Reinforcement Learning at the Edge of the Statistical Precipice", NeurIPS 2021 Outstanding Paper]`
Recomendam: (i) reportar **estimativas por intervalo** em vez de ponto; (ii) **intervalos de
confiança por bootstrap estratificado**, aplicáveis a amostras pequenas e "better justified than
sample standard deviations"; (iii) métricas agregadas robustas — **IQM (interquartile mean)**,
menos sensível a outliers que a média e mais eficiente que a mediana; (iv) **performance
profiles**. Biblioteca: `google-research/rliable`, descrita como "Library for reliable evaluation
on RL and ML benchmarks, even with only a handful of seeds".

*Leitura para o j-space:* com N=50 seeds e estimandos binomiais (p(volta|elegível)), o padrão da
área é **CI explícito + N na legenda + método do CI nomeado**. O draft já reporta
`0.500 [0.313, 0.687]` — falta a legenda dizer **qual** intervalo é (bootstrap? Wilson?
Clopper-Pearson?) e **sobre o que** ele varia (seeds? episódios?). Sem norma da área nomeando
o estimador de CI para proporções — isso é escolha do autor, mas **nomeá-la é obrigatório**
pela guideline do item 7.

**8.4 Compute.**
`[VERIFICADO em neurips.cc/public/guides/PaperChecklist, item 8]` Especificar tipo de
CPU/GPU, memória, armazenamento e **tempo de cada run**; e declarar o compute total, incluindo
"undisclosed preliminary experiments".
`[VERIFICADO em cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf]` "The average runtime for
each result, or estimated energy cost" e "A description of the computing infrastructure used".

---

## 9. Linguagem para claims experimentais pré-registrados

**9.1 Existe convenção em ML, e é jovem e institucionalizada.**
`[VERIFICADO em PMLR v148, "Preface", Bertinetto, Henriques, Albanie, Paganini, Varol, 2021]` —
citação literal, é a melhor definição operacional que achei:

> "A preregistered paper is a regular paper that is submitted for peer-review **without any
> experimental results**, describing instead an experimental protocol to be followed after the
> paper is accepted. This implies that it is important for the authors to make compelling
> arguments from theory or past published evidence. As for reviewers, they must assess these
> arguments together with the quality of the experimental design, rather than comparing numeric
> results. Experiments are conducted after reviewers have accepted the preregistered paper and
> are included as an extension to the preregistered paper. Finally, a second '**confirmatory**'
> round of peer review is conducted to validate that the experiments **faithfully followed the
> proposed experimental protocol**…"

Motivação declarada, também literal: "The status quo incentivises researchers to 'beat the state
of the art', potentially at the expense of deep scientific understanding and rigorous
experimental design. Since typically only positive results are rewarded, the negative results
inevitably encountered during research are often omitted…"

Números do piloto `[VERIFICADO na mesma fonte]`: 73 propostas, 38 aceitas (53%), 25 submeteram
results paper (66%), 23 aceitas na confirmatória (92%). Houve segunda edição em 2021
(`[VERIFICADO em proceedings.mlr.press/v181/]`).

**9.2 Convenções de escrita observadas nos papers pré-registrados reais.**
Amostra: PMLR v148/chandra21a e v181/narayanan22a — os dois papers pré-registrados que abri e
extraí integralmente.

- **Hipóteses numeradas e nomeadas como itens de primeira classe.**
  `[VERIFICADO em PMLR v181/narayanan22a]`: "Hypothesis 1: Can existing methods to handle domain
  shift in DG be translated and adapted…", "Hypothesis 2: Can a curriculum learning strategy
  address cluster imbalance in UDG?"
- **Seção de protocolo em tempo futuro**, distinta da seção de resultados.
  `[VERIFICADO em PMLR v181/narayanan22a, §3 "Experimental Protocol & Planned Implementation" e
  §4 "Additional Experiments"]`: "We plan to conduct additional experiments…", "we will train and
  test on a single domain so that domain shift is controlled for", "We also plan to test whether…".
  O tempo verbal **marca tipograficamente o que foi escrito antes dos dados**.
- **Seção explícita de desvios**, com nomes literais:
  - `[VERIFICADO em PMLR v181/narayanan22a]` **§5.3 "Documented Modifications"**, escrita como
    lista enumerada romana: "(i) We discarded the ImageNet baseline as it is the only method that
    has access to ImageNet weights which made it an unfair comparison; (ii) We subsumed the
    curriculum learning additional experiment in Table 2; (iii) Given the sub-par performance of
    our proposed curriculum learning method, we didn't expect to gain additional insights and
    hence didn't perform the novel class discovery and the labeled domain additional experiments."
  - `[VERIFICADO em PMLR v148/chandra21a]` **§5.1 "Modifications to the Original Proposal"**,
    aberta por: "In this section, we first document the modifications to the original experimental
    protocol." Cada desvio ganha um subtítulo em negrito com a razão ("Switching to Better
    Performing Tasks: We initially proposed to use … we switched to …").
- **Resultado negativo declarado sem amaciamento.** `[VERIFICADO em PMLR v148/chandra21a]`
  "Surprisingly, our unsupervised initial pool sampling methods did not help the three AL…";
  conclusão: "not conclusively prove our hypothesis that AL methods can benefit from more
  intelligently…".

**9.3 Convenção importada da ciência experimental (fora de ML).**
- `[VERIFICADO em cos.io/initiatives/prereg]` Definição: "When you preregister your research,
  you're simply specifying your research plan in advance of your study and submitting it to a
  registry." Distinção normativa: **confirmatório** = teste de efeito predito, hipótese
  especificada e método de teste único predeterminado, minimiza erro tipo I; **exploratório** =
  busca de relações nos dados, minimiza erro tipo II. Motivo: p-valores "lose diagnostic value"
  em análise exploratória. Uma pré-registração deve especificar no mínimo hipóteses, plano de
  análise, desenho, variáveis e **ao menos um teste confirmatório**. **Registered Report** =
  o plano passa por revisão *antes* dos dados e recebe "in principle acceptance".
- `[VERIFICADO em Simmons, Nelson & Simonsohn, "A 21 Word Solution", SSRN 2160588, 2012]`
  Fórmula de divulgação mínima, literal:
  > "We report how we determined our sample size, all data exclusions (if any), all
  > manipulations, and all measures in the study."

**9.4 Como declarar pré-registro sem OSF.**
Sem norma localizada em nenhuma das quatro venues sobre pré-registro em papers submetidos
normalmente (o pré-registro em ML é institucional **só** nos workshops PMLR v148/v181).
`[PRATICA COMUM]` — em papers de 2025–26 aparece o padrão "pré-registrado no OSF, arquivo travado
em <data>, com timestamp confiável, garantindo a ordenação temporal verificável entre desenho
registrado e dados observados"; localizei essas frases em busca mas **não abri os PDFs**, então
registro como relatado, não verificado.

*Leitura para o j-space:* o equivalente honesto e verificável, dado que o repositório é a
única testemunha, é **exatamente o que o draft já faz** — "Pre-registration commits precede
every confirmatory run in the repository history" + hash por experimento. Para bater a
convenção da §9.2, faltam duas coisas concretas: (a) as **hipóteses numeradas** aparecerem no
texto como `H1/H2/...` com a condição que as MATA escrita ao lado; (b) uma seção ou parágrafo
com nome explícito no padrão da literatura — **"Documented Modifications"** ou **"Deviations
from the pre-registered protocol"** — mesmo que o conteúdo seja "nenhum desvio".

---

## 10. Estilo dos títulos

**10.1 Amostra medida.**
`[VERIFICADO em arxiv.org/list/stat.ML/recent, acessado 2026-07-25, primeiros 40 títulos]`
Contagem feita sobre a lista lida:
- **13 de 40 (33%)** usam **dois-pontos** no padrão `Nome curto: expansão descritiva`.
  Exemplos literais: "Fisher Widths: Local Learning Geometry and Anisotropic Recovery";
  "High Minima of Gaussian Processes: Overshoots and Minimizer Locations";
  "Self-Balancing Sequential Sampling: Fast Convergence with Controlled Predictability".
- **~7 de 13** dos títulos com dois-pontos são **acrônimo + expansão**: "SPECTRA: State-Space
  Exogenous Context…", "CLOE: Christoffel Loss Autoencoder for Anomaly Detection",
  "RELTA-SGLD: …", "ConfidenceBench: Evaluating…", "Hilbert Operator for Progressive Encoding
  (HOPE): A Mathematical Framework for…".
- **1 de 40** é **pergunta**: "When Does Recurrence Become an Algorithm? Convergence Selection
  in Weight-Tied Looped Transformers" (nota: pergunta **+** subtítulo declarativo).
- **Maioria (26/40) é sintagma nominal descritivo sem dois-pontos**, frequentemente começando
  por adjetivo do resultado: "Fundamental limits of distributed multiclass classification from
  simple binary decisions"; "Asymptotically Optimal Regret for Reinforcement Learning without
  Horizon Dependence"; "Optimal use of a black-box learner in semiparametric estimation".

**10.2 Padrões de teoria pura (papers que abri).**
`[VERIFICADO nos abs/HTML citados]`
- "Neural Tangent Kernel: Convergence and Generalization in Neural Networks" (1806.07572) —
  objeto novo : o que se prova sobre ele.
- "The local learning coefficient: A singularity-aware complexity measure" (2308.12108) —
  quantidade : qualificação.
- "Dynamics of Transient Structure in In-Context Linear Regression Transformers" (2501.17745) —
  sintagma nominal puro, sem dois-pontos, nomeando **fenômeno + arquitetura**.
- "Quantifying degeneracy in singular models via the learning coefficient" — gerúndio + "via" +
  instrumento.

**10.3 Regras negativas.**
`[VERIFICADO em info.arxiv.org/help/prep.html]` arXiv: evitar CAIXA ALTA; expandir macros TeX
mistificadoras ("'Nonlinear Sigma Models' instead of '\nlsm'"); passar corretor ortográfico.
`[PRATICA COMUM]` Não observei nenhum título na amostra com ponto de exclamação, nem com
declaração de superioridade ("state-of-the-art", "best") no título.

*Leitura para o j-space:* o título atual — "An entry–exit law for dead-neuron resurrection: the
smoothed-ReLU limit of training dynamics at an activation boundary, and its confirmation in a
small MLP" — está no padrão dominante (`objeto/lei : expansão`), mas é longo para a amostra
(nenhum dos 40 tem três orações). O padrão observado nos títulos de teoria é
**`<lei/objeto novo>: <o que se prova> in <onde>`** em uma oração só.

---

## 11. Checklist de aplicação ao `paper-v1.tex` (diagnóstico, não prescrição)

O que o draft **já** cumpre, contra o padrão medido acima:
- Bullets de contribuição na intro (§2.1) — sim.
- Seção `Limitations` explícita, com itens (i)–(v) (§3.1) — sim, e no formato NeurIPS.
- Seção `Reproducibility` com comandos exatos e commits por experimento (§4.1, §6.4) — sim, e
  **acima** do padrão da amostra.
- Negativos registrados e citados no corpo (§9.1 é literalmente a motivação do formato) — sim.
- Tabela com CI (§8) — sim, com CI reportado.
- Apêndice de lemas (§7) — sim.

Lacunas concretas contra as fontes lidas (cada uma amarrada a uma citação acima):
1. **Legenda do CI não nomeia o estimador nem o fator de variabilidade** — viola a guideline
   literal do item 7 ("The method for calculating the error bars should be explained"; "The
   factors of variability that the error bars are capturing should be clearly stated").
2. **N por célula e número exato de runs** precisam estar na legenda da tabela, não só no texto —
   checklist v2.0: "The exact number of training and evaluation runs".
3. **Compute** — não vi seção de compute; itens 8 do checklist NeurIPS e "computing
   infrastructure"/"average runtime" do v2.0 pedem tipo de máquina e tempo por run (o projeto
   tem esse dado: ~12,2 h para 800 runs).
4. **Related Work** — não há seção com esse nome; o padrão da amostra tem uma, em §2 ou tarde
   (§2.2). Hoje a literatura entra por citações soltas na intro.
5. **Sem seção de desvios com o nome da literatura** — "Documented Modifications" /
   "Modifications to the Original Proposal" (§9.2). Mesmo vazia, ela é o que sinaliza
   pré-registro real ao leitor.
6. **Hipóteses não estão numeradas como H1/H2** no corpo (§9.2).
7. **URL pública do repositório ausente** — os commits só viram evidência se o repo for
   acessível; se não for, a convenção manda dizer isso (§4.2, item 5: "If a subset of experiments
   are reproducible, you should state which ones are").
8. **Checklist NeurIPS não foi respondido** — §5. Rodar as 16 e arquivar.
9. **Sem declaração de uso de LLM** (§3.5, item 16) — relevante dado o modo de trabalho do projeto.
10. **Título com três orações** (§10.3).

---

## 12. Fontes (todas abertas e lidas em 2026-07-25)

Normativas:
- NeurIPS Paper Checklist Guidelines — https://neurips.cc/public/guides/PaperChecklist
- NeurIPS 2022 Paper Checklist — https://neurips.cc/Conferences/2022/PaperInformation/PaperChecklist
- Formatting Instructions For NeurIPS 2025 — arXiv:2506.15953v1
- ICML 2025 Author Instructions — https://icml.cc/Conferences/2025/AuthorInstructions
- Submission and Formatting Instructions for ICML 2025 — arXiv:2412.10032v2
- ICLR 2026 Author Guide — https://iclr.cc/Conferences/2026/AuthorGuide
- JMLR Author Information — https://www.jmlr.org/author-info.html
- arXiv submission preparation help — https://info.arxiv.org/help/prep.html

Reprodutibilidade:
- Machine Learning Reproducibility Checklist v2.0 (Pineau, 2020-04-07) —
  https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf
- Pineau et al., "Improving Reproducibility in Machine Learning Research", JMLR 22, 2021 —
  https://www.jmlr.org/papers/v22/20-303.html
- ML Code Completeness Checklist — https://github.com/paperswithcode/releasing-research-code
- Anonymous GitHub — https://4open.science/

Pré-registro:
- Preface, NeurIPS 2020 Workshop on Pre-registration in ML, PMLR v148 (Bertinetto, Henriques,
  Albanie, Paganini, Varol) — https://proceedings.mlr.press/v148/bertinetto21a/bertinetto21a.pdf
- Chandra, Desai, Devaguptapu, Balasubramanian, "On Initial Pools for Deep Active Learning",
  PMLR v148 — https://proceedings.mlr.press/v148/chandra21a/chandra21a.pdf
- Narayanan, Deshmukh, Dogan, Balasubramanian, "On Challenges in Unsupervised Domain
  Generalization", PMLR v181 — https://proceedings.mlr.press/v181/narayanan22a/narayanan22a.pdf
- NeurIPS 2021 Pre-registration Workshop proceedings — https://proceedings.mlr.press/v181/
- Center for Open Science, Preregistration — https://www.cos.io/initiatives/prereg
- Simmons, Nelson & Simonsohn, "A 21 Word Solution", 2012 —
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2160588

Estatística de resultados:
- Agarwal, Schwarzer, Castro, Courville, Bellemare, "Deep Reinforcement Learning at the Edge of
  the Statistical Precipice", NeurIPS 2021 — arXiv:2108.13264; https://github.com/google-research/rliable

Papers usados como amostra de estrutura:
- arXiv:2501.17745 (Carroll, Hoogland, Farrugia-Roberts, Murfet)
- arXiv:2308.12108 (Lau, Murfet, Wei)
- arXiv:2402.02364 (Hoogland et al.)
- arXiv:2410.17592
- arXiv:1806.07572 (Jacot, Gabriel, Hongler)
- Listagem https://arxiv.org/list/stat.ML/recent (40 títulos)
