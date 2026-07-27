# Degrau 2 — o teorema de forcing com readout treinável (scoping)

**Data:** 2026-07-27
**Escopo:** análise estrutural + numérica leve do item (O6) de `06-notas/paper/paper-v4.tex`
(`\label{app:forcing-open}`). Nada aqui entrou no paper. Nada foi commitado.
**Script:** `05-experimentos/t2-n-neuronios/degrau2_readout.py` → `degrau2_readout.jsonl`

---

## TL;DR

**Veredito: (a) — tratável com a maquinaria atual, ao preço de UM lema novo elementar.**

Três achados, em ordem de importância:

1. **A exponencial que volta não é a que o (O6) teme.** Com readout treinável o passo 1
   do L2′ deixa de ser identidade com matriz *constante*, mas continua identidade com o
   **Hessiano médio** $\bar H$ (teorema do valor médio para gradientes — o campo vivo
   continua sendo $-\nabla U$). A parte Gauss–Newton de $\nabla^2U$ é PSD; a parte de
   curvatura residual é **bloco-diagonal por neurônio**, cada bloco $3\times3$ com
   autovalores $\pm\|c_k\|,0$. Logo
   $$\lambda_{\min}(\nabla^2_{\theta_B}L)\ \ge\ -\Lambda,\qquad
     \Lambda:=\max_k\|c_k\|=\max_k\frac{\|\nabla_{(w_k,b_k)}L\|}{|v_k|}\ \le\ X_{\max}C_{\mathrm{res}} .$$
   O fator de Grönwall é $e^{\Lambda T_{\rm ep}}$ com $\Lambda$ **independente de $m$**
   (máximo sobre blocos, não soma) e limitado pela magnitude do resíduo sobre o guarda
   $\min|v_k|\ge\varphi_v$. O $e^{L_B T_{\rm ep}}$ do (O6) com $L_B=\lambda_{\max}(G_B)\sim m$
   *não* é o expoente certo. [ESPECULAÇÃO com prova esboçada + verificação por FD]

2. **`v_k² − ‖(w_k,b_k)‖²` é EXATAMENTE conservado** por unidade sob fluxo gradiente ReLU
   (balanceamento de Du–Hu–Lee), e conservado *beyond all orders* sob softplus dentro de
   (H4′). Consequência operacional: se $B_k(0)=v_k(0)^2-\|(w_k,b_k)(0)\|^2=\varphi^2>0$,
   então $|v_k(t)|\ge\varphi$ para sempre e $v_k$ nunca troca de sinal — **o guarda do
   Algoritmo 1 deixa de ser filtro de instância e vira condição de inicialização.**
   [ESTABELECIDO para o invariante; [RESULTADO 2026] para o corolário do guarda]

3. **O coeficiente sobrevive, com um fator novo e calculável.** Com $v_0$ (readout do
   morto) treinável, a lei vira
   $$\beta\,(t_{\rm release}-t_{\rm exit})\ \longrightarrow\ -\frac{1}{c\,v_0(t_{\rm hit})\,R^{\rm hyb}(t_{\rm exit})},$$
   e $v_0$ é quase-constante no episódio ($\dot v_0=O(\beta^{-2})$, porque o gradiente do
   readout é proporcional à *saída* da unidade, que está beyond all orders enquanto ela
   está morta — o mesmo mecanismo do platô). O cancelamento do L5 é aritmética em $R$ e
   em $c\,v_0$ e não é tocado. **A lei não quebra.**

---

## Status epistêmico, honesto

| item | etiqueta |
|---|---|
| identidade $\dot\Delta=-\bar H\Delta+G_A+G_S$ | [ESTABELECIDO] (TVM para gradientes; campo é $-\nabla U$) |
| estrutura bloco-diagonal da curvatura residual e $\lambda_{\min}\ge-\Lambda$ | [ESTABELECIDO] (conta fechada abaixo; conferida por diferenças finitas a $4\cdot10^{-7}$) |
| invariante $v_k^2-\|(w_k,b_k)\|^2$ sob ReLU | [ESTABELECIDO] Du, Hu, Lee, arXiv:1806.00900, NeurIPS 2018 — **fonte aberta e lida em 2026-07-27** (título/autores/abstract confirmados) |
| defeito softplus do invariante é beyond all orders sob (H4′) | [ESTABELECIDO] (conta + verificação em $\beta=10,10^2,10^3$) |
| $\dot v_0=O(\beta^{-2})$ no episódio | [ESPECULAÇÃO com derivação] — a conta está feita, a prova rigorosa exige o bootstrap do L4 |
| Teorema Degrau 2 (enunciado abaixo) | [ESPECULAÇÃO] — esboço de prova, nenhuma etapa auditada |
| números da §Numérica | [RESULTADO 2026-07-27], $N$ instâncias, $m=4$, um único $\beta$ |

O que **não** foi feito: a prova completa (os seis lemas na forma Degrau 2), o controle de
discretização, o sweep em $\beta$, $m>4$, e qualquer teste da cota fina
$\Lambda\le\|\nabla L\|/\varphi_v$ fora do episódio.

---

## 1. Análise estrutural

### 1.1 O que exatamente quebra

Com readout treinável, $\theta_B=(v_k,w_k,b_k)_{k=1}^m\in\mathbb R^{3m}$ e, com padrão de
ativação congelado,
$$f_{-A}(x_i;\theta_B)=\sum_k v_k\,\mathbf 1_{k,i}\,(w_kx_i+b_k),$$
**bilinear**, não linear. A classe afim da §3.1 do paper morre: não existe $G_B$ fixa com
$\dot\theta_B=-G_B(\theta_B-\theta_B^\ast)$.

Mas o que o passo 1 do L2′ realmente usa não é a afinidade — é o fato de **os dois fluxos
terem o mesmo campo**, diferindo só pelas fontes ($A$ e os defeitos softplus). Isso
continua valendo: ambos são $-\nabla_{\theta_B}U$ com o mesmo $U$. Pelo teorema do valor
médio aplicado ao gradiente,
$$\boxed{\ \dot\Delta=-\bar H(t)\,\Delta+G_A(t)+G_S(t),\qquad
\bar H(t)=\int_0^1\nabla^2U\bigl(\theta_B^{\rm hyb}+s\Delta\bigr)\,ds\ }$$
com $\bar H$ **simétrica**. Ainda é uma identidade, não uma estimativa. O que se perdeu é
apenas $\bar H\succeq0$.

### 1.2 Quanto do PSD sobra (o lema novo)

$$\nabla^2 U=\underbrace{\tfrac1n J^\top J}_{\succeq\,0\ \text{(Gauss–Newton, o }G_B\text{ generalizado)}}
\;+\;\underbrace{\tfrac1n\sum_i r_i\,\nabla^2 f_i}_{\text{curvatura residual}} .$$

Para rede de uma camada oculta, $f_i$ é bilinear e $\nabla^2f_i$ só tem cruzados
$\partial^2 f_i/\partial v_k\partial(w_k,b_k)=\mathbf 1_{k,i}(x_i,1)$ — **dentro** de cada
neurônio, nunca entre neurônios. Logo a curvatura residual é bloco-diagonal, com bloco $k$
igual a
$$\begin{pmatrix}0&c_k\\ c_k^\top&0\end{pmatrix},\qquad
c_k=\frac1n\sum_i r_i\,\mathbf 1_{k,i}\,(x_i,1)\in\mathbb R^2,$$
de autovalores $\pm\|c_k\|$ e $0$. Portanto

> **Lema (curvatura negativa bloco-diagonal).** Para rede de uma camada oculta com padrão
> de ativação congelado e perda quadrática,
> $$\lambda_{\min}\bigl(\nabla^2_{\theta_B}L\bigr)\ \ge\ -\max_k\|c_k\|=:-\Lambda,
> \qquad \Lambda=\max_k\frac{\|\nabla_{(w_k,b_k)}L\|}{|v_k|}\ \le\ X_{\max}C_{\mathrm{res}} .$$

Duas leituras, ambas úteis:

* **crua:** $\Lambda\le X_{\max}C_{\mathrm{res}}$ — só a magnitude dos resíduos e dos dados;
  **nada de $m$**, porque é máximo sobre blocos e não traço/soma;
* **fina:** $\Lambda\le\|\nabla_{\theta_B}L\|/\varphi_v$ — decai junto com a convergência, e
  é **por aqui** que o guarda $\min_k|v_k|\ge\varphi_v$ do Algoritmo 1 entra na conta. O
  guarda que os experimentos já impõem é exatamente o objeto certo.

*Verificação independente (diferenças finitas, $n=5$, $m=3$):* Hessiano montado bate com FD
a $3.7\cdot10^{-7}$; $\lambda_{\min}=-0.0913$ contra a cota $-\max_k\|c_k\|=-0.1333$ — cota
válida e apertada dentro de $\sim30\%$.

**Comparar com o (O6).** O (O6) prevê "$e^{L_B T_{\rm ep}}$ de volta". Correto quanto ao
$T_{\rm ep}$; errado quanto ao $L_B$. O expoente não é $\lambda_{\max}(G_B)$ (que cresce
$\sim m$ com readouts unitários e é o que tornava L2 vacuoso para $m$ grande); é $\Lambda$,
independente de $m$ e proporcional ao resíduo. A degradação com readout treinável é
**qualitativamente mais branda** do que o paper admite.

### 1.3 Balanceamento: o candidato (iii) do escopo, confirmado

Sob fluxo gradiente ReLU, para cada unidade $k$:
$$\tfrac{d}{dt}\Bigl(v_k^2-\|(w_k,b_k)\|^2\Bigr)
=-2v_k\tfrac1n\!\sum_i r_i\mathbf 1_{k,i}z_{k,i}
 +2(w_k,b_k)\!\cdot\!v_k\tfrac1n\!\sum_i r_i\mathbf 1_{k,i}(x_i,1)=0,$$
exatamente, pela homogeneidade de grau 1 do ReLU ($v\,\partial_v f=(w,b)\cdot\partial_{(w,b)}f$).
É o invariante de balanceamento de Du–Hu–Lee (arXiv:1806.00900). Verificado
numericamente a $10^{-17}$.

Sob **softplus** o invariante não é exato: $g_\beta$ não é homogênea e
$$\tfrac{d}{dt}B_k=-2v_k\tfrac1n\sum_i r_i\bigl[g_\beta(z_{k,i})-z_{k,i}\sigma(\beta z_{k,i})\bigr],$$
com $0\le g_\beta(z)-z\sigma(\beta z)\le\beta^{-1}\log 2$ em geral e
$\le e^{-\beta|z|}(|z|+\beta^{-1})$ para $|z|\ge\delta_B$. Sob (H4′) o defeito é **beyond all
orders**, exatamente da mesma família dos termos de step 2b do L2′ — cabe no orçamento
existente sem termo novo. Verificado: defeito $2.8\cdot10^{-5}$ ($\beta=10$),
$1.7\cdot10^{-7}$ ($\beta=10^2$), $\sim10^{-13}$ ($\beta=10^3$).

**Três corolários.**

* **(C1) O guarda vira condição de init.** $|v_k(t)|^2=\|(w_k,b_k)(t)\|^2+B_k(0)$. Se
  $B_k(0)=\varphi^2>0$ então $|v_k|\ge\varphi$ e $\operatorname{sign}v_k$ é constante,
  para todo $t$, sem hipótese sobre a trajetória. A hipótese (H6′) abaixo é *derivada*.
* **(C2) Com init escala-NTK o corolário é vazio.** $v_k=1/\sqrt m$ e $\|w_k\|=O(\sqrt m)$
  dão $B_k(0)<0$; o que fica limitado por baixo é $\|(w_k,b_k)\|\ge\sqrt{-B_k(0)}$, e $v_k$
  *pode* cruzar zero. É o que acontece: ver §3, item 5.
* **(C3) Dimensão efetiva $2m$.** Sobre a folha invariante $\{B_k=\text{const}\}$ o
  subsistema vivo tem $3m-m=2m$ graus de liberdade — a mesma contagem da classe afim do
  Teorema 6. A folha é *invariante*, não uma aproximação; é isso que faz a mudança de
  variável do escopo funcionar.

### 1.4 O readout do morto muda o coeficiente — e de forma calculável

Com $v_0$ treinável, a equação da margem na face $j$ é
$$\dot z=-v_0\tfrac1n\sum_i r_i\,\sigma(\beta z_i)(x_ix_j+1)
      = v_0\,c\,\sigma(\beta z)\,R+O(e^{-\beta\delta_1}),\qquad c=\tfrac{x_j^2+1}{n}.$$
Ou seja **$c\mapsto c\,v_0$ em toda parte**: velocidade de chegada $c|v_0||R(t_{\rm hit})|$,
velocidade de saída $c\,v_0R(t_{\rm exit})$, e coeficiente de 1ª ordem
$-1/(c\,v_0R^{\rm hyb}(t_{\rm exit}))$. **A área e $t_{\rm exit}$ não mudam**, porque $v_0$
sai como fator constante do integrando.

E $v_0$ é quase-constante no episódio:
$\dot v_0=-\tfrac1n\sum_i r_i\,g_\beta(z_{0,i})$, com
$g_\beta(z_{\rm park})=g_\beta(-\beta^{-1}\log\beta)=O(\beta^{-2})$ na face e
$O(\beta^{-1}e^{-\beta\delta_1})$ nas outras. Logo
$|v_0(t)-v_0(t_{\rm hit})|=O(\beta^{-2}T_{\rm ep})$ no episódio inteiro.

Isso é um mecanismo, não um acidente: **o gradiente do readout de uma unidade é
proporcional à saída dessa unidade**, que está beyond all orders enquanto ela está morta.
Uma unidade morta congela o próprio readout. É o mesmo fato que produz o platô, visto na
outra coordenada.

---

## 2. Enunciado-candidato: Teorema Degrau 2 [ESPECULAÇÃO]

> **Teorema (lei entrada–saída, forma forcing, readout treinável).**
> Rede $f_\theta(x)=v_0g_\beta(w_0x+b_0)+\sum_{k=1}^m v_kg_\beta(w_kx+b_k)$ com **todos** os
> parâmetros treinados. Suponha (H0′)–(H5′) do Teorema 6, com (H4′) inalterada (padrão
> congelado, margens vivas $\ge\delta_B$), e adicionalmente
>
> **(H6′) guarda de readout.** $|v_k(t)|\ge\varphi_v>0$ e $\operatorname{sign}v_k$ constante
> em $[0,H]$, para $k=0,\dots,m$. *(Automática, com $\varphi_v=\min_k\sqrt{B_k(0)}$, se
> $B_k(0)=v_k(0)^2-\|(w_k,b_k)(0)\|^2>0$ — §1.3, C1.)*
>
> Ponha $\Lambda:=\sup_{[t_{\rm hit},t_{\rm exit}]}\max_k\|c_k(t)\|\le X_{\max}C_{\mathrm{res}}$
> e $T_{\rm ep}=t_{\rm exit}-t_{\rm hit}$. Então existe $\beta_0<\infty$ tal que para
> $\beta\ge\beta_0$ valem (i), (ii), (iii), (ivA), (ivB) do Teorema 6 com **duas** trocas:
>
> **(a)** o coeficiente de primeira ordem é
> $$t^\beta_{\rm release}=t_{\rm exit}-\frac{1}{c\,v_0(t_{\rm hit})\,R^{\rm hyb}(t_{\rm exit})}\cdot\frac1\beta+o(\beta^{-1});$$
> **(b)** as constantes ganham um único fator: $M_2\mapsto M_2e^{\Lambda T_{\rm ep}}$,
> $M_3\mapsto M_3e^{\Lambda T_{\rm ep}}$, $\beta_0\mapsto\beta_0e^{\Lambda T_{\rm ep}}$.
>
> Em particular $\beta_0\lesssim O(m)\,e^{\Lambda T_{\rm ep}}$: **o expoente não depende de
> $m$**, e a Proposição 7 (limiar polinomial) sobrevive a $\Lambda,T_{\rm ep}$ fixos.

**Esboço de prova, lema a lema.**

| lema | o que muda |
|---|---|
| L1 (forma normal) | fator $v_0$ na equação da margem; quadratura aditiva intacta |
| L2′ passo 1 | identidade com $\bar H$ no lugar de $G_B$ (§1.1) |
| L2′ passos 2a/2b | **intactos** — são sobre $A$ e sobre defeitos softplus, não sobre a classe |
| L2′ passo 3 | $\frac{d}{dt}\frac12\rho^2\le\Lambda\rho^2+\rho(\|G_A\|+\|G_S\|)$; Grönwall dá $\rho(t)\le e^{\Lambda(t-t_{\rm hit})}\bigl[\rho(t_{\rm hit})+\int e^{-\Lambda(s-t_{\rm hit})}(\|G_A\|+\|G_S\|)ds\bigr]$ |
| L2′ passo 4 (pré-hit) | idem, com $\bar H$ cheio; a separação de uma potência ((O5)) não usa PSD, só o sinal do defeito |
| L2′ passo 5 (pós-release) | idem, mesmo $\bar H$ para os dois fluxos |
| L3/L4 (camada, bootstrap) | intactos em forma; as constantes herdam $e^{\Lambda T_{\rm ep}}$ |
| L5 (cantos/cancelamento) | **intacto** — é aritmética em $R$ e em $c\,v_0$. É por isso que o coeficiente sobrevive |
| L6 | intacto |
| **lema novo** | **um só**: curvatura negativa bloco-diagonal (§1.2). Elementar |

**O que o Degrau 2 exige de matemática nova: nada além desse lema.** Não é preciso PL, não
é preciso mudança de variável, não é preciso teoria de fluxos bilineares. O balanceamento
(§1.3) é bônus: converte (H6′) de hipótese em condição de inicialização e dá a contagem
$2m$.

---

## 3. Numérica leve

**Desenho.** Mesmo motor de `verifica_forcing.py` (Euler explícito, $\beta=10^4$,
$dt_0=10^{-5}$ no híbrido, $dt=2\cdot10^{-6}$ no acoplado, $H=1.8$, $n=4$ dados, $m=4$
vivas + 1 morta), mesma família de sorteio, **com os readouts $v_k$ de todas as unidades
(vivas E morta) treinados junto com $(w_k,b_k)$**. Peneira grosseira em 500 sementes → 50
avaliações completas; 43 passaram (H1′)–(H4′) + guarda de sinal. 9,2 min de CPU, 2 workers,
`nice -19`. Rejeições: 5 sem release no horizonte, 1 sem morte, 1 morte antes de $t_{\min}$.

*Desvio declarado:* a família de `verifica_forcing.py` produz **muito menos episódios** com
readout treinável (rendimento da peneira 11% contra ~4% lá, mas com 92% de "sem morte") e os
episódios são **3–10× mais curtos**: $T_{\rm ep}$ mediana $0{,}058$ contra $0{,}2$–$0{,}8$ na
mesma família com readout fixo. O guarda de sinal foi tornado *local à janela*
$[t_{\rm hit},t_{\rm exit}]$ (rejeitar por troca de sinal fora do episódio é estrito demais).

### (1) A lei sobrevive; a constante piora ~2,5× na mediana e muito mais na cauda

| | mediana | p75 | p90 | máx |
|---|---|---|---|---|
| readout **treinável** ($n=43$) | **1,96 %** | 7,83 % | 20,2 % | 69,7 % |
| readout **fixo**, $m=4$ ($n=23$, baseline) | 0,80 % | — | 5,4 % | 16,2 % |

A lei $\beta(t_{\rm rel}-t_{\rm exit})\to-1/(cR^{\rm hyb}(t_{\rm exit}))$ continua valendo,
sem parâmetro livre e sem ajuste. **Ressalva:** as duas linhas não são a mesma amostra
(instâncias diferentes, $T_{\rm ep}$ 3–10× menor), então o fator 2,5 é indicativo, não um
contraste pareado.

### (2) O readout do morto não se move — o teste do fator $v_0$ deu NULO DE INSTRUMENTO

$$\max_{\text{43 instâncias}}|v_0(t_{\rm hit})-1|=8{,}7\cdot10^{-6}.$$

Os dois preditores ($-1/(cR_{\rm exit})$ e $-1/(c\,v_0R_{\rm exit})$) coincidem em **todos**
os dígitos reportados; `err_naive` $\equiv$ `err_corr`. Isto **confirma o mecanismo** da
§1.4 (uma unidade quase-morta congela o próprio readout — e aqui ela nasce com $z_A(0)\le
0{,}05$, então $v_0$ mal se move nem antes da morte), mas **não testa a correção**: nesta
família a correção é invisível por construção. Registrado como negativo de instrumento.
*Teste que discriminaria:* inicializar $v_0(0)\in\{0{,}5;\,2\}$; a lei corrigida prevê o
coeficiente escalando por $1/v_0$, a do paper prevê nada. Não foi rodado.

### (3) A cota do lema novo é SATURADA, não só válida

$\Lambda=\max_k\|c_k\|$ e $L_B=\max(0,-\lambda_{\min}(\nabla^2_{\theta_B}U))$ medido por
autovalores exatos do Hessiano $12\times12$ na trajetória híbrida:
$$\max_{\text{instâncias}}|L_B-\Lambda|=4{,}4\cdot10^{-15}\quad(\text{relativo }9\cdot10^{-16}).$$
Igualdade em máquina, em todas as 43. **Mecanismo plausível** [ESPECULAÇÃO]:
$\operatorname{rank}(J^\top J)\le n=4<3m=12$, e o auto-vetor negativo do bloco residual cai
no núcleo da parte Gauss–Newton, fechando a cota. Não é consequência automática do posto:
no teste de diferenças finitas ($n=5$, $m=3$, ponto aleatório fora de trajetória) sobrou
folga de 30% com o mesmo déficit de posto. Portanto: **saturação observada nesta família,
não teorema**. O que é teorema é a desigualdade.

Faixas: $\Lambda\in[1{,}9;\,6{,}9]$ (mediana 4,0), $T_{\rm ep}\in[0{,}020;\,0{,}39]$
(mediana 0,058).

### (4) O fator de Grönwall é pequeno E NÃO EXPLICA a degradação

$$e^{\Lambda T_{\rm ep}}\in[1{,}07;\ 2{,}70],\qquad \text{mediana }1{,}31 .$$

O preço exponencial que o (O6) teme vale, nestas instâncias, **um fator entre 1,1 e 2,7** —
e com $\Lambda$ independente de $m$ ele não cresce com a largura.

Mais forte: a degradação **anti-correlaciona** com ele.

| regressor | corr(log erro, ·) | $R^2$ sozinho |
|---|---|---|
| $\log e^{\Lambda T_{\rm ep}}$ | **−0,633** | 0,401 (coef **−6,0**) |
| $\log(1/R^{\rm hyb}(t_{\rm exit}))$ | **+0,604** | 0,365 |
| $\log T_{\rm ep}$ | −0,650 | — |

Regressão conjunta:
$$\log\text{erro}=-6{,}93+1{,}83\,\log\tfrac1{R_{\rm exit}}-4{,}19\,\log e^{\Lambda T_{\rm ep}},
\qquad R^2=0{,}51 .$$

O coeficiente do Grönwall é **negativo**: instâncias com fator de Grönwall maior erram
*menos*. Um mecanismo causal não pode ter esse sinal. O que sobra com sinal certo é
$1/R^{\rm hyb}(t_{\rm exit})$, expoente $+1{,}8$ — **exatamente o mesmo driver que o (O4)
identifica para o readout fixo**: perda de transversalidade na saída faz $\beta_0$ explodir.
*Ressalva honesta:* os dois regressores são colineares (corr $=-0{,}50$) e $R^2=0{,}51$
não é uma decomposição limpa; a amostra ($N=43$, um único $\beta$) não separa os dois
canais, só exclui que o Grönwall seja o canal dominante pelo sinal.

**Resposta direta à pergunta do escopo:** a degradação **não** é Grönwall real. É o mesmo
$\beta_0(\text{forcing})$ do Degrau 1. Há estrutura melhor disponível do que a exponencial
— e a estrutura é a do §1.2, que já torna a exponencial inofensiva.

### (5) Balanceamento: conservado; guarda automático NÃO se ativa neste init

Deriva máxima de $B_k=v_k^2-\|(w_k,b_k)\|^2$ ao longo do episódio:

| | mediana | máx |
|---|---|---|
| híbrido (ReLU, $dt=10^{-5}$) | $2{,}2\cdot10^{-5}$ | $6{,}8\cdot10^{-5}$ |
| $\beta$-run (softplus, $dt=2\cdot10^{-6}$) | $4{,}2\cdot10^{-6}$ | $1{,}1\cdot10^{-5}$ |

Razão mediana híbrido/$\beta$ $=5{,}56$, contra razão de passos $=5$. **A deriva é erro de
Euler, não defeito de softplus** — a conservação é exata para o fluxo e o defeito softplus
está abaixo do ruído de discretização, como prevê a §1.3. Confirmado à parte por
integração direta: defeito $2{,}8\cdot10^{-5}\to1{,}7\cdot10^{-7}\to\sim10^{-13}$ para
$\beta=10,10^2,10^3$.

$B_k(0)<0$ em **100%** das unidades vivas (faixa $[-134;\,-0{,}32]$), como prevê o corolário
C2 para o init escala-NTK. Consequência prática: o guarda **não** é automático aqui —
$\min_k|v_k|$ chega a $9{,}5\cdot10^{-4}$ em 4 instâncias (que, curiosamente, erram pouco:
0,0–2,5%). Em varredura separada com $dt$ grosseiro, 19/300 instâncias perdem o guarda por
troca de sinal de $v_k$ dentro da janela; o reescalamento balanceado
$(v,w,b)\to(sv,w/s,b/s)$ — que **não muda $f$** — elimina 100% dessas trocas (0/150), ao
preço de gerar falhas de (H4′) por encolher as margens. O init do paper é fortemente
desbalanceado; o guarda de (H6′) é hipótese genuína nesta família.

---

## 4. Veredito

**(a) — tratável com a maquinaria atual, mais UM lema novo elementar.**

Não é (b) e não é (c).

* **Não é (c) — a lei não quebra.** 43 instâncias com readout treinável, erro mediano 1,96%
  contra 0,80% do readout fixo, sem parâmetro livre. O cancelamento do L5 é aritmética em
  $R$ e não vê a classe do subsistema vivo; a numérica é consistente com isso.
* **Não é (b) — não exige matemática nova.** O passo 1 do L2′ continua sendo uma
  *identidade*, com o Hessiano médio $\bar H$ no lugar de $G_B$. O único objeto novo é a
  cota $\lambda_{\min}(\nabla^2_{\theta_B}L)\ge-\max_k\|\nabla_{w_k}L\|/|v_k|$, que sai da
  estrutura bloco-diagonal da curvatura residual em três linhas e é *saturada* no regime
  sobreparametrizado. Não é preciso PL local, não é preciso teoria de fluxos bilineares.
* **O lema novo, nomeado:** *curvatura negativa bloco-diagonal* (§1.2). Mais um lema de
  apoio, opcional mas barato: *balanceamento por unidade* (§1.3), que converte o guarda
  $\min|v_k|\ge\varphi_v$ do Algoritmo 1 de hipótese sobre a trajetória em condição sobre a
  inicialização.

**Correção ao (O6), a registrar.** O (O6) diz "espera-se $e^{L_B T_{\rm ep}}$ de volta". O
$T_{\rm ep}$ está certo; o $L_B$ não. O expoente não é $\lambda_{\max}(G_B)\sim m$ — é
$\Lambda\le X_{\max}C_{\mathrm{res}}$, **independente de $m$**, e medido vale
$e^{\Lambda T_{\rm ep}}\in[1{,}07;\,2{,}70]$. Ou seja: o readout treinável custa um fator
$\lesssim3$ nas constantes e **não** reintroduz a explosão em $m$ que o passo PSD havia
removido. O (O6), como está redigido, é mais pessimista do que a matemática exige.

**O que a numérica mediu e o que não mediu.** Mediu: a lei, $\Lambda$, $L_B$, $T_{\rm ep}$,
o fator de Grönwall, a conservação do balanceamento e o driver da degradação. **Não** mediu
o fator $v_0$ (nulo de instrumento — §3, item 2), nem $m>4$, nem sweep em $\beta$, nem
controle de discretização.

**Próximo passo mínimo, se o Degrau 2 for perseguido**, em ordem de retorno:
1. rodar a família com $v_0(0)\ne1$ (custo: idêntico ao deste run) — é o único teste que
   separa a lei corrigida da lei do paper, e é falsificável;
2. sweep em $\beta$ nas 5 piores instâncias, para checar se a cauda de 70% colapsa como
   colapsou no Degrau 1 (o que confirmaria que é $\beta_0$ e não a lei);
3. só então escrever as provas na forma Degrau 2 — a alteração é local aos passos 3, 4 e 5
   do L2′, mais o fator $v_0$ no L1.
