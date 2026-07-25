# 8º parecer adversarial — desenho etapa-4a.2 (REPROVADO) — 2026-07-25

Revisor independente (Opus), com runs próprios de verificação no regime de
deploy. Texto integral, sem edição. Resposta da casa: pivô de física
(pressão/subparametrização) — ver README-4a3 quando existir.

---

## VEREDITO: **REPROVADO.**

**Primeira linha, regra da casa:** rodei a régua v3 na temperatura de deploy (T=1e-5, m=64, H=2e5) que a calibração nunca tocou, e ela **não separa relu-dura de softplus** — relu-comp = 7, 5 vs softplus-comp = 9, 6 (razão ~1.2-1.3×). relu-dura tem gradiente ≡0 no morto: **não pode ressuscitar por mecanismo**, só por ruído. Que ela produza quase tantos episódios v3-confirmados quanto a softplus é *exatamente* a doença que invalidou a 4a (NEGATIVOS.md: "ReLU-dura ressuscita igual a softplus"), agora reproduzida sob a v3 no T real. O rascunho esconde isso ao escolher **semgrad** (0, 2 episódios — difusão pura, loss diverge para 0.68-1.0) como nulo do Q1: um nulo que *qualquer* dinâmica vence. Q1 contra semgrad é uma máquina de falso-positivo; Q1 contra o controle certo (relu-dura) já morre. Não se congela um preditor sobre um instrumento com zero evidência de que funciona no T em que vai rodar, e com o precedente da 4a dizendo que provavelmente não funciona.

Isto não reprova a lei de área nem a P5. Reprova este desenho: instrumento não-calibrado no regime de deploy + nulo que mascara a falha.

## Evidência que rodei (nice -19, ≤2 proc, ParRegua = mesma trajetória vista por v2 e v3)

**m=64, T=1e-5, H=2e5, n_σ=10 (dwell D=100), prof OFF:**

| braço | seed | loss | v2≥50 (proxy do sizing) | v3-comp | v3-cens |
|---|---|---|---|---|---|
| softplus | 45 | 0.000 | 47 | **9** | 22 |
| softplus | 46 | 0.000 | 13 | **6** | 23 |
| relu-dura | 45 | 0.000 | 49 | **7** | 22 |
| relu-dura | 46 | 0.000 | 11 | **5** | 23 |
| semgrad | 45 | 0.679 | 0 | 0 | 20 |
| semgrad | 46 | 0.999 | 4 | 2 | 23 |

Três leituras, todas fatais:

1. **relu ≈ softplus** (7≈9, 5≈6) ⇒ os episódios v3 da softplus são majoritariamente cruzamentos por ruído, não a ressurreição-por-área. O "chatter residual 3-5%" que o rascunho supõe é na verdade ~80-100% do sinal quando medido contra o nulo certo.
2. **v3 colapsa a contagem ~5×** vs o proxy v2≥50 que dimensionou tudo (47→9). O sizing (§0.4, §2, §4) é de um instrumento que não vai rodar.
3. **Censura ~70%** (9 comp / 31 físicos em s45), não os "27-48%" do §0.4; semgrad ~100%. A comparação de *completos* entre braços é confundida por censura diferencial.

## Respostas item a item às 5 perguntas do §5

**(a) Q1 é decidível a priori? O semgrad pareado garante excesso ≥0? O pareamento é real ou cosmético?**
- **Pareamento é cosmético.** semgrad tem `sem_grad=True`: zero termo `−η∇L`. A softplus minimiza a loss (→0), a semgrad passeia (loss 0.68-1.0). Depois do burn-in as duas trajetórias compartilham só o *incremento de ruído* da mesma seed, não a dinâmica — variância-redução de par ≈ nula. "Pareado por seed/init/ruído" entre dinâmicas que divergem não é matched-pairs; é enfeite.
- **Excesso ≥0 não é garantido por construção** (back-reaction pode censurar completos da softplus — e censura, de fato: 22 censurados vs 9 completos), **mas a direção softplus ≫ semgrad é quase-tautológica**: semgrad não tem dinâmica, então difusão do vetor inteiro raramente sustenta um dwell duplo de 100 passos. Q1-vs-semgrad testa "gradiente cria mais mortes-sustentadas que ausência-de-dinâmica" — quase definicional. É o padrão de degeneração do P-D da 4a (rev_freeze=rev_controle=0 ⇒ "sem back-reaction" tirado de zeros).
- **O nulo certo é relu-dura**, não semgrad: dinâmica completa (loss→0, campo residual realista), mas grad≡0 no morto ⇒ sem mecanismo de ressurreição. É o discriminador que a própria 4a usou. Com ele, Q1 Desfecho A **falha** (razão ~1.2×). Trocar o nulo por semgrad é uma regressão que apaga a lição da 4a.

**(b) O gate de 8 ep/neurônio no Q2 é honesto ou é escape?**
- **É escape — sob o instrumento que vai rodar.** Analisei ep/neurônio na célula primária. Sob v2 (proxy) parecia farto: s45 = 151 comp em 5 neurônios (med 34/neurônio). **Mas sob v3 são 2-4 neurônios com med 2-3 ep/neurônio** (dwell colapsa 151→9). A própria lição TPR (0.16 com ~2 ep/neurônio) diz que 2-3 é potência nula. O gate de 8 **nunca dispara** na célula primária com a v3 ⇒ Q2 reporta "sem potência" para sempre ⇒ falsificação impossível por construção. Confirmado o medo (b).
- Pior: **o denominador do gate não está definido** — "mediana de episódios/neurônio" sobre *neurônios ativos* (dá 2-3) ou sobre os m=64 da camada (dá 0)? A resposta ao gate inverte conforme a escolha não-congelada.

**(c) Q1 precisa de piso de efeito? Qual estatística e que vieses de contagem?**
- **Sim, precisa de piso** — e o piso tem de ser em unidades de contagem de episódios v3-limpos-completos calibrada ao instrumento v3, não ao proxy v2.
- **Vieses de contagem presentes:** (i) **censura diferencial** — softplus ~70% censurada, semgrad ~100%; comparar *completos* mistura mecanismo com censura. n_fis (inclui censurados) vs n_comp dão respostas diferentes e o rascunho não fixa qual nem trata a assimetria. (ii) **superdispersão/não-independência** — episódios concentram em 1-4 neurônios (semgrad s45 = 1 neurônio faz os 27; softplus 2-5); não são contagens independentes, o CI bootstrap-por-seed subestima a variância. (iii) **"chatter residual"** não é 3-5%: relu-dura mostra que é ~80-100% do sinal.

**(d) β=30 vale sem recalibração? (calibração foi β=300)**
- O limiar `2√(nT)/‖x‖`, σ_z e D=⌈n_σ²⌉ são **β-independentes** ⇒ a régua *mecanicamente* aplica a β=30. **Mas toda a evidência de aceitação (toy-preservado, chatter-rejeitado, relu-vs-softplus) foi só a β=300**, e a β=300/T=1e-5 o instrumento já reprova no controle relu. β=30 herda um instrumento não-calibrado *e falho*. É o pecado B1 da série (instrumento ≠ calibrado), duplo: nem no T nem no β de deploy há calibração.
- **Buraco extra:** `semgrad-β30` é dinamicamente idêntico a `semgrad-β300` (sem gradiente, β só entra na `a=act` que a régua usa para R̂, não na trajetória). O braço "semgrad-β30" da grade (§2) é mal-posto.
- Q4 como **estimação** está bem rotulado *desde que* nenhum veredito MATA se prenda à direção "β=30 > β=300"; como está (CI, informa 4a.3) é aceitável — não é MATA disfarçado. É o único dos cinco que passa.

**(e) v3 nunca calibrada no toy a T=1e-5 (calibração usou T=1e-3). O limiar e n_σ=10 fazem sentido a T=1e-5? σ_z cai 10×; D=100 continua certo?**
- D=n_σ² é **adimensional/escala-invariante por construção** (Δ e σ_z escalam juntos) ⇒ formalmente D=100 "continua certo" em qualquer T. **Mas escala-invariância é necessária, não suficiente**, e a calibração ausente teria pego o que meus runs pegaram: **a T=1e-5 o debounce empiricamente NÃO rejeita ressurreição por ruído** (relu≈softplus). Motivo físico: a banda de histerese é c_h≈1.04-1.11 (n_σ·σ_z ≪ lim_morte), e em 3e5 passos o passeio da margem-líder atravessa uma banda tão fina por deriva-de-ruído com folga; os episódios v3 a T=1e-5 têm dur~16000 (gigantes), então o dwell de 100 é desprezível frente ao episódio. A banda deveria escalar com a razão deriva/difusão sobre o horizonte, não com um n_σ fixo. **Nenhuma das três aceitações da calibração (toy-preservado, chatter, relu-vs-softplus) existe a T=1e-5** — e a única T calibrada (1e-3) deu veredito inconclusivo ("inutilizável").

## Buracos novos (além das 5)

- **N1 (o que reprova):** controle relu-dura ausente a T=1e-5; quando rodado, relu≈softplus ⇒ instrumento re-invalidado. semgrad mascara.
- **N2:** sizing 100% sobre proxy v2+dur≥50. Sob v3: contagem ÷5, censura ~70%, episódios ~16000 passos (objeto diferente). Todo número de §0.4/§2/§4 (3.76 = 188/50 v2-comp; 17-20 comp/seed; censura 27-48%; p99 ressurreição ~1.6e5) é do instrumento errado.
- **N3 — "população" oca:** 1-4 neurônios ativos por seed fazem todos os episódios (n_k=1 em semgrad s45). O nulo estratificado foi calibrado a n_k=4 (nulo-smoke); as células reais têm n_k∈{0,1,2,3,4}. FPR/TPR a n_k≤2 e com autocorrelação forte de um único neurônio ciclando são **não-validados**. O experimento mal exercita o corpo-de-m-corpos que é a tese da etapa-4.
- **N4 — deploy da v3 inexistente:** `run.py` liga a régua **v2** (`harness_mlp.Regua`), não a v3. Não há `config_etapa4.2.json`, `preditor_4a2`, nem runner que faça o rebind v3 na confirmatória. O caminho de deploy do instrumento congelado não existe nem foi verificado.
- **Runtime/memória (OK):** m=256/2e5 = 147s ⇒ ~220s a 3e5; 350 runs ≈ 5-6 h em Pool 4 (o "~4h" do §2 é otimista, não bloqueante). Memória trivial: episódios v3 são poucos; mesmo v2 (~1700 ep/run a m=256) é <1 MB. Este é o único eixo sem problema.

## O que precisa existir ANTES de qualquer preditor-freeze (lista numerada)

1. **Trocar/adicionar o nulo relu-dura** como controle primário do Q1 (semgrad é fraco demais) e **demonstrar que a v3 separa relu-dura de softplus a T=1e-5** com muitas seeds e contagem com FPR controlada. Enquanto relu≈softplus, a etapa-4 encerra pelo mesmo negativo-de-instrumento da 4a — e isso deveria ser escrito como o desfecho mais provável.
2. **Calibrar a v3 no regime de deploy**: toy-preservado, chatter-rejeitado e relu-vs-softplus **a T=1e-5** (não 1e-3), e um n_σ (banda/dwell) que rejeite ruído sobre H=3e5 — provavelmente n_σ tem de crescer com o horizonte, ou a banda tem de escalar com deriva/difusão. Sem isto o instrumento é não-calibrado no T que roda.
3. **Re-dimensionar tudo com a régua v3** (não v2+dur≥50): contagem/seed, censura, horizonte para censura<50% (meu run dá ~70% a 2e5 — 3e5 pode não bastar para episódios de ~16000), razão softplus/nulo, potência. Reescrever §0.4/§2/§4.
4. **Construir e verificar o runner v3 confirmatório**: `config_etapa4.2.json`, `preditor_4a2.py` (congela grade, régua v3, critérios), `run.py` que faz rebind da v3, device-pin cpu restabelecido, seeds virgens.
5. **Congelar os números que faltam:** k* (limiar de seeds positivas do Desfecho A); piso de efeito do Q1 em unidades de contagem v3; α do teste de sinais Q1; nível do CI bootstrap agregado; níveis de CI Q3/Q4; **denominador do gate Q2** (por-neurônio-ativo vs por-camada) e o "8" recalibrado para S=40 e n_k real sob v3; definição de "físico" sob v3 (piso de dur? ou v3-confirmado ≡ físico?).
6. **Fixar a regra de contagem sob censura diferencial** (n_fis vs n_comp) e justificar contra o viés — a comparação de completos entre softplus (70% cens) e nulo (100% cens) não é neutra.
7. **Validar o nulo estratificado a n_k∈{1,2,3}** com autocorrelação de um único neurônio ciclando (o smoke só cobriu n_k=4); reportar FPR nesse extremo antes de confiar no P-NULL de Q2.
8. **RNGs virgens e separados** para o bootstrap do Q1 e a permutação do Q2 no dado REAL (as chaves 5100/5200/5300 são do sintético; 777/779 são da 4a) — congelar em config.
9. **Corrigir o braço semgrad-β30** (dinamicamente idêntico ao semgrad-β300) ou removê-lo.

**Resumo:** a espinha (existência via excesso sobre um nulo, mapa de washout, β como mecanismo) é legítima e os desfechos-nos-dois-sentidos são honestos. O que reprova é uma coisa só, e é a coluna vertebral repetida da 4a: **o instrumento congelado (régua v3) não foi calibrado no regime que vai rodar (T=1e-5, β=30), e quando eu o rodo lá ele não distingue ressurreição-física (softplus) de ressurreição-por-ruído (relu-dura, grad≡0) — a mesma falha que matou a 4a.** O desenho a esconde trocando o controle discriminante (relu-dura) por um controle fraco (semgrad). Calibrar a v3 a T=1e-5, provar separação contra relu-dura, re-dimensionar e erguer o runner v3 — aí se volta a falar de freeze.
