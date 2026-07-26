# Checklist de submissão — paper-v3

Regras duras, verificadas nas fontes abertas em 2026-07-26 (links no fim).
Cada linha é uma regra do venue, não uma opinião nossa.

`paper-v3.tex` é a **versão canônica arXiv**: autor visível, `\thanks` com nome
e e-mail, Acknowledgments presentes. A submissão TMLR é uma **conversão
anonimizada** desse mesmo texto para `tmlr.sty` — nunca o mesmo arquivo.

## TMLR (Transactions on Machine Learning Research)

| # | Regra dura | Estado em v3 |
|---|---|---|
| T1 | Template obrigatório: `tmlr.sty` + `tmlr.bst` do repositório `JmlrOrg/tmlr-style-file`; `\documentclass[10pt]{article}` + `\usepackage{tmlr}` **sem opção** na submissão. `[preprint]` só para postar em preprint server; `[accepted]` só depois do aceite. | pendente (v3 está em `article` 11pt + `geometry`) |
| T2 | Anonimato total. O template esconde os autores enquanto `tmlr` for usado sem `[accepted]`/`[preprint]`: "Non-anonymous submissions will be rejected without review". Remover nome, `\thanks`, cidade, e-mail, Acknowledgments e qualquer link para repositório próprio. | a fazer na conversão |
| T3 | Revisão é double blind; o próprio guia lista preprint servers (arXiv) como **não arquivais**, então o arXiv não impede a submissão — mas o texto submetido não pode apontar para a versão identificada. | ok por construção |
| T4 | Citações via `natbib` author–year: `\citet{}` no texto corrente, `\citep{}` entre parênteses; `\bibliographystyle{tmlr}`. | a fazer (v3 usa `thebibliography` numérico) |
| T5 | Se houver DOIs no `.bib`, definir `\doi` no preâmbulo (o `tmlr.bst` emite `\doi{...}`), senão o build quebra. | a fazer |
| T6 | Suplementar: PDF ou ZIP, ≤ 100 MB, também anonimizado. | n/a por ora (nada além do PDF) |
| T7 | Sem limite de páginas; comprimento tem de ser justificado pelo conteúdo — papers longos atrasam a revisão. 38 pp com apêndice de provas é defensável, mas o corpo deve ficar autossuficiente. | ok |
| T8 | Broader Impact só é exigido se o trabalho tiver risco significativo de dano. **Decisão consciente: omitir** — o objeto é um teorema de EDO e experimentos em redes de ≤6 unidades. | decidido |
| T9 | Sem dual submission: nada de versão expandida de paper de conferência, nem texto/figuras/resultados publicados, aceitos ou **submetidos em paralelo** a outro venue arquival. | ok |
| T10 | LLM é ferramenta assistiva permitida, **nunca autor**; a responsabilidade por todo o conteúdo é do autor. A seção "Declaration of LLM usage" do v3 cumpre isso e deve ser mantida. | ok |
| T11 | Revelar identidade após rejeição impede submeter versão revista à TMLR. | anotado |

## arXiv

| # | Regra dura | Estado em v3 |
|---|---|---|
| A1 | O `.bbl` deve ter o **mesmo nome-base** do `.tex` principal (`paper-v3.tex` → `paper-v3.bbl`). O v3 usa `thebibliography` embutido, então nem é preciso `.bbl`; o arquivo vazio existe e é inofensivo — **ou** remove-se, ou mantém-se casado. | ok |
| A2 | Processamento em TeX Live 2025 (default) ou 2023. Compilar localmente contra algo próximo antes de subir. | verificar no envio |
| A3 | Figuras em pdfLaTeX: apenas `.pdf`, `.png` ou `.jpg`. As 5 figuras são `.pdf`. | ok |
| A4 | Não incluir arquivos auxiliares nem de saída (`.aux`, `.log`, `.out`, `.pdf` do próprio paper), arquivos ocultos, templates de journal, figuras não usadas, backups. Empacotar só `paper-v3.tex` + os 5 PDFs de figura. | a fazer no empacotamento |
| A5 | Abstract-metadado ≤ 1920 caracteres, sem `~`, `\,`, `\ ` nem comandos de fonte; MathJax simples é aceito. O abstract do v3 tem **1357 caracteres** e nenhum `\,` ou `~`. | ok |
| A6 | Título sem caixa alta integral e sem Unicode; macros TeX crípticas proibidas. O título do v3 é texto puro. | ok |
| A7 | Comments: `38 pages, 5 figures, 4 tables. Comments welcome.` (contagem de páginas e figuras é convenção do campo). | header do `.tex` já traz a string |
| A8 | Enquadramento: research article. Desde out/2025 o cs do arXiv só aceita review/position articles já revisados por pares em venue — este paper **não** é review nem position, então a regra não morde; declarar categoria primária math.DS com cross-list cs.LG evita a ambiguidade. | ok |
| A9 | Política de IA generativa: LLM **não pode ser autor**; uso significativo de ferramentas sofisticadas deve ser reportado; a responsabilidade é integralmente do autor que assina. O paper **já traz** a declaração — **não removê-la**. | ok |

## Ordem de operações sugerida

1. arXiv primeiro (versão canônica, autor visível, `paper-v3`).
2. Conversão TMLR: cópia nova (`paper-tmlr.tex`), `tmlr.sty` sem opção,
   `natbib`+`\bibliographystyle{tmlr}`, `\doi` definido, autor/thanks/
   Acknowledgments/links de repositório removidos.
3. Nunca editar os dois arquivos em paralelo: o TMLR é gerado do canônico.

## Fontes lidas (2026-07-26)

- TMLR author guide — <https://jmlr.org/tmlr/author-guide.html>
- TMLR editorial policies — <https://jmlr.org/tmlr/editorial-policies.html>
- Template TMLR (`main.tex`, `tmlr.sty`) — <https://github.com/JmlrOrg/tmlr-style-file>
- arXiv, submissões TeX — <https://info.arxiv.org/help/submit_tex.html>
- arXiv, metadados (título/abstract/comments) — <https://info.arxiv.org/help/prep.html>
- arXiv, política de IA generativa — <https://blog.arxiv.org/2023/01/31/arxiv-announces-new-policy-on-chatgpt-and-similar-tools/>
