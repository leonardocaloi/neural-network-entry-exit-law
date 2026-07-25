# Etapa-5 (transformer FFN-GELU) — RELATÓRIO de SCOPING exploratório

**Pergunta:** o mecanismo morte→ressurreição-por-área, observado em MLP raso
(etapa-4a: canal suave `e^{βz}` reabre unidades mortas; ReLU-dura = 0 por teorema
grad≡0), generaliza para a **FFN-GELU de um transformer pequeno**?

**Status epistêmico: [EVIDÊNCIA NUMÉRICA exploratória]** — piloto de 18 runs, seeds
42–44, SEM pré-registro. Nada aqui é confirmatório. As hipóteses da casa (H-a/H-b/H-c)
são [ESPECULAÇÃO]; o que segue é o que os números dizem, sem amaciar. Serve para
desenhar (ou reprovar) o confirmatório da etapa-5.

Reprodução: `motor_e5.py` (motor + instrumento + `--testes`), `piloto_e5.py`
(`--run --workers 2` / `--analise`), `results-piloto/*.json`. Toda a dinâmica é
GD puro full-batch T=0 determinístico em CPU (`torch.use_deterministic_algorithms`).

---

## 1. Desenho mínimo honesto

**Arquitetura** (1 camada; `motor_e5.TinyTransformer`), sem LayerNorm (para não
mascarar a pré-ativação):

```
e_{n,s}   = x_{n,s}·w_val + P[s]                 embedding valor + posição
h_{n,s}   = e_{n,s} + MHSA_causal(e)_{n,s}       self-attn causal (H=2), residual
z_{n,s,k} = W1[k]·h_{n,s} + b1[k]                pré-ativação FFN, unidade k
a_{n,s,k} = GELU(z)  (ReLU no controle)
h2_{n,s}  = h_{n,s} + (a·W2ᵀ + b2)_{n,s}         residual em torno da FFN
ŷ_{n,s}   = ⟨w_out, h2_{n,s}⟩ + b_out            leitura escalar por posição
```

**Tarefa (routing + transformada rica):** x_{n,s}~U[0,2]; alvo
`y_{n,s} = rich_f(x_{n,s-1})`, `rich_f(u)=sin(2πu)+0.3 sin(5πu)` (o alvo rico do
etapa-4). O modelo precisa de **atenção** (copiar o token anterior — s→s-1, causal)
**e** da **FFN** (a transformada não-linear rica). Config do piloto: d=16, H=2, S=3,
N=24 (⟹ 72 tokens), η=1e-3, H=1.5e5 passos.

**Por que a pressão cai na FFN.** Embedding, atenção e cabeça são **lineares**; a
ÚNICA capacidade não-linear é a FFN. A FFN pos-wise mapeia o escalar atendido
(~x_{n,s-1}) por `m_ffn` ridges GELU — **estrutura idêntica ao MLP de m unidades**.
`m_ffn` pequeno vs riqueza de `rich_f` ⟹ platô (subparametrização). Controle H-c:
`m_ffn` grande deve baixar a loss.

**Peso de saída efetivo (a chave da generalização do instrumento).** A contribuição
ESCALAR da unidade k à predição em (n,s) é `c_k·a_{n,s,k}` com
> **c_k := ⟨w_out, W2[:,k]⟩**

(a FFN é pos-wise e a cabeça é linear ⟹ a_{n,s,k} só afeta ŷ_{n,s}). c_k faz o papel
EXATO de v_k do MLP. **Líder** = (n*,s*) = argmax_{n,s} z_{n,s,k}. Verificado por
teste sintético (perturbação de a_k reproduz c_k a 1e-8).

**Derivação de θ (limiar de morte GELU).** GELU(z)=z·Φ(z), GELU'(z)=Φ(z)+z·φ(z).
GELU' TROCA DE SINAL em z≈−0.752; para z<−0.752 é NEGATIVA (mín ≈−0.126 em z≈−1.3,
→0⁻ na cauda). Em z=−3: |GELU(−3)|≈0.004, |GELU'(−3)|≈0.012 (ambos desprezíveis).
⟹ **morte-GT = zmax_k = max_{n,s} z_{n,s,k} < −θ, θ=3** (cauda onde saída e canal
somem). ReLU: morte = zmax<0 (saída≡0, o zero exato do MLP).

## 2. Instrumento GT-ÁREA generalizado (H-b)

Todos os sinais no relógio ground-truth (zmax por unidade, sobre TODOS os (n,s)),
régua-independente. Durante o episódio morto de k acumula por passo:

- **porte direto do MLP** (sem g'): `R̂_mlp = c_k·(−res_lider + c_k·a_lider)`
- **drive faithful** (com a inclinação local): `D_k = c_k·(−res_lider)·GELU'(z_lider)`
- `A_mlp += R̂_mlp·η`, `A_drive += D_k·η`; `t_area_* = 1º passo com área>0`;
  `z_park = min zmax`; `zmax_hi = max zmax pós-morte`.

O motor é read-only sobre a dinâmica (aditivo). `motor_e5.py --testes` passa 8
testes sintéticos: determinismo; gradiente=diferenças-finitas; c_k=∂ŷ/∂a_k; líder
(n*,s*); **S1** (abaixo); **S2/drive** (abaixo); ReLU-morto grad-próprio≡0;
consistência contador↔episódios.

## 3. Duas descobertas ESTRUTURAIS (transformer ≠ MLP) — [EVIDÊNCIA, testes sintéticos]

Antes de qualquer estatística, o desenho já expôs duas diferenças que o frame da
casa não previa e que MUDAM o instrumento e a interpretação:

**S1 — o sinal da inclinação inverte a área.** softplus tem g'(z)=σ(βz)>0 sempre;
por isso o MLP pôde usar R̂ SEM g'. **GELU tem g'(z)<0 na região morta** (troca de
sinal em −0.752). O drive real sobre a margem-líder é ∝ −res·c_k·GELU'(z); portar a
área do MLP (sem g') **INVERTE o sinal na região morta**. Teste sintético: em z=−2,
sign(R̂_mlp) = OPOSTO a sign(D_drive). ⟹ o instrumento GT-ÁREA primário para GELU
DEVE ser o `A_drive` (com inclinação); o porte-MLP `A_mlp` só serve de diagnóstico.
Confirmado empiricamente (§piloto): unidades presas em z≈−3 têm A_mlp>0 (falso
"área abriu") mas A_drive<0 (correto: sem drive líquido para cima) e NÃO revivem.

**S1b — obstáculo de reversão de sinal.** Para reviver, a unidade sobe de z<0 a z>0
e CRUZA z=−0.752, onde g' troca de sinal. O drive self que a empurrava para cima
abaixo de −0.752 passa a empurrá-la para baixo acima de −0.752 (com res·c_k de sinal
fixo). ⟹ o **self-canal GELU não conduz monotonicamente** do fundo à vida, ao
contrário do softplus — um obstáculo estrutural à auto-ressurreição.

**S2 — a entrada da FFN é NÃO-ESTACIONÁRIA.** No MLP a entrada x é fixa ⟹ unidade
ReLU morta tem grad≡0 ⟹ CONGELADA ⟹ morte TERMINAL (teorema, 0 voltas). No
transformer `h = e + attn(e)` é saída da atenção TREINÁVEL ⟹ mesmo uma unidade com
grad-próprio≡0 pode reviver porque **h se move** (drive de MONTANTE). ⟹ o teorema
"ReLU = terminal" do MLP **NÃO transfere**: o controle de canal ReLU não é garantido
silencioso. Medimos `own_move` (deslocamento dos pesos próprios no episódio) para
atribuir self-canal vs montante; e comparamos ReLU-congelada (self só ⟹ terminal
pelo teorema) vs ReLU-treinável (montante pode reviver).

## 4. Narrativa de calibração (EXPLORATÓRIO, pré-piloto)

Sondagens (documentadas, results em transcript) fixaram o regime:
- **Morte profunda (<−3) é RARA por construção no transformer.** A entrada da FFN é
  d=16-dimensional e varia sobre 72 tokens; exigir zmax<−3 em TODOS os tokens é muito
  mais difícil que o zmax<0 do MLP (entrada 1-D). Nos braços naturais, 0 mortes em
  1.5e5 passos (§piloto). Esta é, em si, uma descoberta: **unidades de FFN são mais
  difíceis de matar que unidades de MLP** (entrada de alta dimensão).
- Poucos tokens ⟹ a FFN RESOLVE a tarefa (sem pressão); muitos tokens ⟹ pressão mas
  morte profunda ainda mais rara. N=24 dá platô ~0.13 (pressão) sem mortes naturais.
- ⟹ para exercitar/validar o mecanismo e o instrumento com N>0 eventos, o piloto usa
  **injeção de nascidas-mortas** (`n_kill=2` unidades rebaixadas a z≈−1.5; z_dead=−1):
  alavanca EXPLORATÓRIA declarada, análoga ao "probe" que a etapa-4 usou para flagrar
  o mecanismo. NÃO é o regime natural; serve para testar SE o canal reabre e SE a
  área prediz, dado que a unidade está morta.

## 5. Resultados do piloto (18 runs, seeds 42–44, H=1.5e5, 31 min em 2 workers)

Tabela por braço (agregado sobre 3 seeds). `n(<-3)` = episódios que atingiram a
cauda profunda; `zpark_med` só nos braços com injeção.

| braço | m_ffn | attn | platô | loss_fim | born | dyn | eps | **rev** | zpark_med | n(<-3) |
|---|---|---|---|---|---|---|---|---|---|---|
| gelu-press   | 4  | treina  | 0.119 | 0.116 | 0 | 0 | 0 | **0** | — | 0 |
| gelu-slack   | 32 | treina  | **0.015** | 0.015 | 0 | 0 | 0 | **0** | — | 0 |
| gelu-kill-tr | 4  | treina  | 0.218 | 0.214 | 6 | 0 | 6 | **0** | −1.51 | 0 |
| gelu-kill-fz | 4  | congela | 0.250 | 0.250 | 6 | 0 | 6 | **0** | −1.53 | 0 |
| relu-kill-tr | 4  | treina  | 0.204 | 0.203 | 6 | 0 | 6 | **0** | −1.57 | 0 |
| relu-kill-fz | 4  | congela | 0.240 | 0.240 | 6 | 0 | 6 | **0** | −1.50 | 0 |

Fatos crus (sem amaciar):
- **0 mortes dinâmicas** e **0 mortes profundas naturais** (zmax<−3) em TODOS os
  braços, mesmo em 1.5e5 passos. Unidades de FFN não morrem sozinhas neste regime.
- **0 ressurreições em 24 episódios injetados** (6 por braço × 4 braços de kill).
  Nenhuma unidade morta voltou a z>0 em nenhum braço (GELU/ReLU, attn treina/congela).
- Climb GELU: as unidades injetadas SOBEM parcialmente (zmax_hi: kill-tr até **−0.56**
  — passou o ponto de reversão −0.752 — kill-fz até −0.85) mas ESTAGNAM antes de z=0.

## 6. Veredito por hipótese — SEM AMACIAR

**H-c (pressão necessária) — SUSTENTADA.** platô m4=0.119 vs m32=0.015 (~8×); a FFM
folgada essencialmente RESOLVE a tarefa (loss_fim 0.010–0.023) com **0 mortes**. A
pressão é de capacidade de FFN e a folga a alivia. (A tarefa de routing é solúvel:
prova que o gargalo do braço m4 é a FFN, não a atenção.) Único H confirmado.

**H-a (unidades GELU morrem e ressuscitam) — NÃO SUSTENTADA neste regime.** Dois
motivos MECANÍSTICOS, não de amostragem:
1. **Morte é rara.** A entrada da FFN é d=16-D sobre 72 tokens; exigir zmax<−θ em
   TODOS ⟹ 0 mortes naturais. Unidades de FFN são MAIS difíceis de matar que as de
   MLP (entrada 1-D). [O frame previa morte dinâmica; o piloto não a produz sem injeção.]
2. **Ressurreição não ocorre**, mesmo com unidades injetadas mortas + pressão +
   1.5e5 passos: 0/24. Diagnóstico: **obstáculo de reversão de sinal do GELU** (S1b).
   O drive self só empurra a margem para cima abaixo de z=−0.752 (g'<0); ao cruzar
   −0.752 (g'>0) o sinal exigido inverte e a subida estanca — as unidades GELU
   chegaram a −0.56 e pararam. O canal GELU **não conduz monotonicamente** do fundo
   à vida, ao contrário do softplus (g'>0 sempre) que reabriu no MLP.

**H-b (instrumento GT-ÁREA generaliza) — PARCIAL.** Generaliza ESTRUTURALMENTE
(c_k=⟨w_out,W2[:,k]⟩ como v_k; líder=(n*,s*); 8 testes sintéticos passam) e a **área
com inclinação (A_drive) é a correta e necessária**: em **12/12 episódios GELU** o
A_drive e o porte-MLP A_mlp têm **sinais OPOSTOS** — o instrumento do MLP portado às
cegas erraria o sinal em todos. O sinal de A_drive concorda com a direção do climb
(A_drive>0 ⟹ zmax_hi sobe). PORÉM, **sem voltas (0/24), não há como validar a
ordenação temporal** (t_morte≤t_área<t_res) — o coração de C2 do MLP fica sem teste.

**S2 (entrada não-estacionária ⟹ ReLU não é terminal) — CONFIRMADA e quantificada.**
- `relu-kill-fz` (attn congelada, entrada fixa): Δmargem (zmax_hi−z_park) = **0.000
  exato** em 6/6, own_move=0 ⟹ unidade CONGELADA ⟹ **terminal** (teorema do MLP vale
  quando a entrada é fixa).
- `relu-kill-tr` (attn treina): **own_move=0** (grad-próprio≡0, como o teorema exige)
  MAS Δmargem = **+0.69 em média, até +1.32** ⟹ a margem de uma unidade de gradiente
  congelado SOBE por **drive de MONTANTE** (h se move). O teorema "ReLU morto =
  terminal" **NÃO transfere** ao transformer. (A subida foi insuficiente p/ z>0 em
  1.5e5 ⟹ 0 voltas, mas o mecanismo de montante é real e medido.)

**Resumo:** o mecanismo do MLP **não generaliza direto** à FFN-GELU do transformer
neste desenho mínimo. O que generaliza: a álgebra do instrumento (c_k, líder) e a
necessidade da área-com-inclinação. O que QUEBRA: (i) morte de FFN é rara (entrada
alta-D); (ii) o canal GELU tem reversão de sinal em −0.752 que trava a auto-volta;
(iii) surge um segundo canal de volta (montante, via atenção) que o MLP não tinha.

## 7. Recomendação para o desenho confirmatório da etapa-5

O confirmatório NÃO está pronto — o piloto REPROVA o desenho ingênuo (portar o
mecanismo do MLP) e aponta o que consertar antes de gastar N seeds:

1. **Escolher o regime de morte revivível, não a cauda θ=3.** A física interessante
   do GELU está em z∈(−0.75, 0) (g'>0, como o softplus). θ=3 põe a unidade na cauda
   morta onde o canal some E abaixo do ponto de reversão. Duas saídas:
   (a) redefinir morte-GT em z_dead≈−0.75 (o zero de g'), medindo z_park e o climb; ou
   (b) usar ativação da CLASSE-DO-TEOREMA (softplus-β / erf-β, g'=Φ(βz)>0) na FFN —
   aí o canal é monotônico e o mecanismo do MLP deve transferir. GELU vira o braço
   FORA-DE-CLASSE (o interessante: mostrar que o sign-flip o mata).
2. **Isolar os dois canais.** Pré-registrar `attn-congelada` (self-canal, reduz ao
   MLP) vs `attn-treina` (self+montante) como fator, com Δmargem e own_move como
   observáveis de atribuição (já instrumentados). A volta de montante é um FENÔMENO
   NOVO — merece seu próprio estimando, não ser confundido com o canal de ativação.
3. **Fazer a morte acontecer sem injeção, ou assumir a injeção como parte do desenho.**
   Se o alvo é morte dinâmica natural: subir a pressão (mais tokens/alvo mais rico,
   d_model menor) e medir a taxa — provavelmente ainda baixa (entrada alta-D). Se
   injeção: pré-registrá-la como condição inicial controlada (profundidade, nº),
   não como "probe" pós-hoc.
4. **Horizonte e janela.** Com climb ~0.9 unidade de margem em 1.5e5 e alvo z=0 a
   partir de −1.5, ou o H é muito maior (≥5e5, como o 4a.3) ou a profundidade de
   injeção é rasa (z_park≈−0.5). Calibrar o return-time num piloto de taxa ANTES,
   como fez a etapa-4 (herdar a disciplina: n*, janela [0,H−τ], censura por KM).
5. **Instrumento:** manter A_drive (com inclinação) como primário; A_mlp só como
   sonda do sign-flip. c_k e líder=(n*,s*) já validados por teste sintético.

**Veredito de scoping:** sinal NEGATIVO informativo. O transformer-GELU **não**
reproduz morte→ressurreição-por-área como o MLP; o canal GELU tem um obstáculo
estrutural (reversão de sinal em −0.752) e a FFN é difícil de matar. Antes de
qualquer confirmatório: trocar GELU→classe-do-teorema OU mover o limiar para a região
revivível, e tratar a volta-de-montante como mecanismo próprio. Etiqueta:
**[EVIDÊNCIA NUMÉRICA exploratória]** — 18 runs, sem pré-registro, N=3 seeds/braço.
