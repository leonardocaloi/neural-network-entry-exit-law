# 9º parecer adversarial — desenho etapa-4a.3 (CORRIGIR-DESENHO) — 2026-07-25

Revisor independente (Opus); rodou o piloto + 6 runs β=3 próprias a H=5e5.
Texto integral, sem edição. Resposta da casa: piloto-de-taxa em execução;
correções de C1/C2/relu/janela aplicadas no desenho final.

---

## VEREDITO: CORRIGIR-DESENHO. O instrumento está são (T=0 mata o chatter que reprovou a 4a.2; relu dá zero estrutural limpo, 0/5 em todos os braços; ground-truth régua-independente). O mecanismo existe. Mas o sizing S=60 não é "frágil", é CONTRADITO pelos próprios dados do piloto: a nota assume P(volta|morte)=0.70; o piloto mostra 11/15 = 73% das mortes dinâmicas NUNCA abrem área (park fundo, σ(βz)≈0 — terminais), então P(volta|morte) ≤ 4/15 = 0.267 e o observado é 2/15 = 0.133. Com isso o MATA de C1 dispara por azar de taxa/censura em até ~78% dos cenários compatíveis. Corrigir o estimando (condicionar à contagem de mortes), erguer um piloto-de-taxa intermediário, e reescrever C2.

## Achados principais

(a) C1: apagar "pareado"/"excesso Poisson sobre relu" — relu e softplus divergem do passo 1 (sem matched-pairs); o zero da relu é TEOREMA, não amostra. relu = CONTROLE DE VALIDADE do instrumento (qualquer volta ⟹ bug), usado certo. Estimando primário = p(volta | morte dinâmica elegível) da própria softplus, Clopper-Pearson.

(b) Sizing: mortes dinâmicas são FIXADAS PELA INIT (mesmo padrão de seeds em todos os braços — superdispersão; não é Poisson iid). Poisson-CI sobre n=5: taxa/run [0.101, 0.729]. p(volta): obs 2/15=0.133, teto 4/15=0.267; a nota usou 0.70 (o teto-CI de 1/5). Tabela de E[eventos] honesta: 1.6–10 (não 24); P(MATA por azar) 12–78%. EXIGIR piloto-de-taxa (~30-40 seeds, 4 braços, H=5e5) antes do freeze; MATA condicional: N_mortes < n* ⟹ INCONCLUSIVO, não negativo.

(c) C2: denominador correto = TODAS as voltas dinâmicas (sem-área/terminal conta como NÃO-ordenada); desigualdades ESTRITAS t_morte<t_área<t_res (empate = não-ordenado); born-dead excluídos; sizing guiado por C2 (o coração), não por C1. Inconsistência README(≤)-vs-código(<) a congelar.

(d) β=3 VIÁVEL (rodado): morre dinâmico (t=1483) e VOLTA (t=59732, ~2.3× mais rápido que β10) — física do canal. Escada C4 como p(volta|morte), não contagem bruta (taxa de morte é β-independente; a volta é β-dependente).

(e) Exposição: mortes dinâmicas são TRANSIENTES (t_morte ≤ 45k) ⟹ censura de mortes tardias desprezível; o "não-volta" dominante é TERMINAL (park fundo), não censura. Janela-de-morte-elegível [0, H−τ] pré-registrada; reportar p(volta|morte) E p(volta|morte, área-abriu); person-time/KM.

Novos: contador de volta tem de ser GROUND-TRUTH (pipeline do piloto usava v3); CI com clustering por seed (eventos concentrados em ~5 seeds); config/preditor/runner 4a3 não existem (gap N4).

Runtime validado: ~4.5 min/run a H=5e5; 480 runs ≈ 9h Pool 4 — mas o gargalo é potência, não relógio.

## Lista de freeze (10 itens)
1. Piloto-de-taxa (pré-requisito). 2. Estimando C1 = p(volta|morte elegível) CP. 3. MATA condicional (n*, piso, α; N<n* ⟹ inconclusivo). 4. Janela [0,H−τ] + terminal-vs-censurado + dois p + person-time. 5. C2: denominador todas-as-voltas, estritas, born-out, sizing por C2. 6. Contador ground-truth + regra dyn/born. 7. Grade β∈{3,10,300}; C4 sem MATA com piso. 8. relu = validade. 9. RNGs virgens + clustering por seed. 10. config_4a3 + preditor_4a3 + runner (rebind v2+GT, CPU).
