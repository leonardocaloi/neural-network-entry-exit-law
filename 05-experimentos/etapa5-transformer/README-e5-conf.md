# etapa-5 — CONFIRMATORIO: ressurreicao-por-canal na FFN de um transformer (T=0)

**STATUS: CANDIDATO — aguardando 11o parecer.** NENHUMA run confirmatoria foi
executada. Trilha: scoping (`RELATORIO-scoping.md`, sinal negativo informativo) →
piloto-de-taxa (`TAXA-e5.md`, 160 runs EXPLORATORIO, de-chatter) → este pacote. Herda
a disciplina do confirmatorio **4a.3** (contador ground-truth, CP/bootstrap exatos sem
scipy, exploratorio-separado-de-confirmatorio, pre-registro com predicao-que-mata
escrita ANTES da run).

Etiqueta: **[ESPECULACAO⟶EVIDENCIA]**. Regime **DETERMINISTICO (GD puro T=0)**; o
estocastico e fora de escopo.

---

## 0. O que o piloto ensinou (e o que este pacote conserta)

O piloto (`TAXA-e5.md`) fixou tres fatos que MOLDAM o desenho:

1. **Morte natural ≈ 0.** λ_dyn/run inclui 0 em todos os bracos (unidades de FFN d=16-D
   nao morrem sozinhas). ⟹ o confirmatorio e OBRIGATORIAMENTE injecao-condicional.
2. **CHATTER do canal nao-estacionario.** Com o contador do piloto (z_dead=z_alive=0),
   uma unidade parada em zmax≈0 cruza o limiar milhares de vezes (relu-train-s103: 1a
   travessia genuina em t=66846 + 159 micro-oscilacoes). O sinal cru fica inflado.
   **Conserto: HISTERESE** (morte confirmada zmax<−δ_h, volta confirmada zmax>+δ_h) —
   a licao explicita do piloto (TAXA-e5.md §0/§7).
3. **Return-time lento (τ≈60–66k).** A H=8e4 a janela [0,H−τ] fica apertadissima.
   **Conserto: H=2e5 (≈3τ)**, janela [0,H−τ]≈[0,1.36e5].

Este pacote implementa a casa pedida: motor com histerese + volta SUSTENTADA, injecao
em dois niveis como FATOR, cinco estimandos (E-A…E-E) com MATA e decidibilidade
declaradas, e numeros 100% derivados do piloto/calibracao.

## 1. Instrumento: histerese + volta sustentada

- **δ_h DERIVADO (nao magico).** `preditor_e5.calibra_deltah`: quantil **q0.99** da
  amplitude de flutuacao (meia-p2p de zmax em janelas de L=200 passos) de unidades
  **VIVAS** (zmax>0 o run todo) em runs **FROZEN** de calibracao (seeds 7–9, 15000
  passos). Resultado **δ_h ≈ 0.0476** — que **valida a posteriori** a escolha δ≈0.05
  do piloto. Por-braco: sp-b3 0.048, sp-b10 0.048, gelu 0.042, relu 0.032.
- **Maquina de estado (por unidade FFN):** viva —(zmax<−δ_h)→ morta (morte confirmada);
  morta —(zmax>+δ_h)→ viva (VOLTA confirmada). A banda [−δ_h,+δ_h] segura o estado ⟹
  mata o chatter na origem.
- **Volta SUSTENTADA** = unidade injetada que voltou E terminou H **viva**. **Hover** =
  voltou mas reabriu (terminou morta). Regra de contagem: **1 sustentada por unidade
  injetada** (colapso). Verificado: no seed-chatter s103, 160 travessias cruas (δ_h=0)
  colapsam para **0 voltas confirmadas** (δ_h=0.05) — o instrumento faz o que promete.
- **Paridade byte-a-byte** da DINAMICA com `motor_e5b`: o instrumento e aditivo e
  read-only (`motor_e5_conf.py --testes`, caso (1): loss_hist identico em 4 casos).

## 2. Grade (800 runs)

`{sp-b3, sp-b10, gelu, relu} × {attn-treina, attn-frozen} × injecao {−0.3, −2.0} × 50
seeds VIRGENS 2000–2049` = **800 runs**. m_ffn=4, N=24×S=3 tokens, d=16, H_heads=2,
η=1e-3, **H=2e5**, T=0. **1 unidade injetada por run** na profundidade da grade (a
profundidade e FATOR ⟹ isola o canal por profundidade; o piloto injetava 2/run). Isso
deixa 3 vivas de 4 (pressao mais branda que as 2-de-4 do piloto) — se algo, **favorece**
a volta, logo conservador para o MATA.

## 3. Numeros CONGELADOS (de `config_e5.json`, origem no piloto/calibracao)

| numero | valor | origem |
|---|---|---|
| δ_h (histerese) | **≈0.0476** | q0.99 meia-p2p zmax de unidades vivas, frozen (calibracao propria) |
| τ q0.9 sp-b3 | **64 240** | return-time colapso de-chatter, pool train+frozen |
| τ q0.9 sp-b10 | **62 517** | idem |
| τ q0.9 gelu | **59 680** | idem |
| τ q0.9 relu | **67 724** | idem |
| janela [0,H−τ] sp-b3 | **[0, 135 760]** | H − τ, H=2e5 |
| E-C mata_thresh | **0.00617** | ½ × CP-inf(piloto sp-b3-frozen rasa 2/20) |
| E-C n* | **6** | piso de voltas sustentadas (TAXA-e5.md §7) |
| E-B Δ piloto (b3/b10/gelu/relu) | **+6/+6/+6/+8** | frozen vs train voltas (colapso) |
| E-D piloto gelu-frozen | rasa 4/20 = prof(−1.0) 4/20 | EMPATE ⟹ injeta −2.0 |
| RNGs bootstrap (novos) | prop=161803, montante=141421 | ≠ 777/779/5xxx/9090/314159/271828 |

## 4. Os cinco estimandos, MATA e decidibilidade

- **E-A VALIDADE (obrigatoria, DECIDIVEL).** Qualquer volta sustentada em **relu-frozen**
  (rasa OU profunda) ⟹ EXPERIMENTO INVALIDO. Base: teorema — grad proprio≡0 (relu morto)
  **E** entrada fixa (frozen) ⟹ terminal. Piloto: relu-frozen 0/40. Verificado no motor
  (`--testes`, caso 4): 0 sustentadas, own_move==0 em ambas profundidades.
- **E-B MONTANTE (reproducao, DECIDIVEL a S=50).** Δ = p_sust(train) − p_sust(frozen)
  por braco, CI bootstrap por seed. **REPRODUZ** se CI-inf>0; **MATA** (montante falso)
  se CI-sup≤0. *Disclosure honesta:* o piloto mostrou Δ>0 decisivo (+6/+6/+6/+8; relu
  0→8 = montante puro, own_move=0), MAS sob o contador z_alive=0 (chatter-propenso). A
  histerese e ESTRITA: exige zmax>+δ_h SUSTENTADO. A H=2e5 (≥3τ) os climbs de montante
  tem tempo de exceder δ_h e sustentar ⟹ reproducao ESPERADA mas **nao garantida** —
  se o montante for so hover, E-B MATA. Isto torna E-B um teste GENUINO, nao um carimbo.
- **E-C TRANSFERENCIA (o genuino nº1).** sp-b3-frozen inj −0.3: p(volta sustentada|inj),
  Clopper-Pearson. **MATA** se CP-sup < mata_thresh (=½ CI-inf do piloto); **TRANSFERE**
  se sustentadas≥n*=6; senao **INCONCLUSIVO**. *Decidibilidade (disclosure):*
  mata_thresh=0.00617 e minusculo ⟹ CP-sup(0,50)=0.071 > thresh ⟹ **a N=50 a MATA e
  quase-inatingivel** (paridade honesta com o C1 do 4a.3; MATA so a N≥800). E-C decide na
  pratica entre TRANSFERE (≥6 sustentadas) e INCONCLUSIVO (<6). O poder REAL e o gate
  positivo: piloto 2/20 a H=8e4 ⟹ projetado ~0.28 a H=2e5 ⟹ ≥6/50 esperado. Falsificavel
  em principio (N grande), como o sintetico demonstra.
- **E-D BARREIRA-GELU (o genuino nº2).** gelu-frozen: p_sust(profunda −2.0) vs
  p_sust(rasa −0.3). A barreira preve **rasa≫profunda** no canal PROPRIO. **MATA-barreira**
  se CP-inf(prof) ≥ CP-sup(rasa) (barreira nao existe); **SUSTENTA-barreira** se
  CP-inf(rasa) > CP-sup(prof) e rasa≥n*. *Decidibilidade:* o piloto injetou −1.0 (so um
  pouco abaixo de z*=−0.752) e deu EMPATE (rasa 4/20 = prof 4/20) — **nao decidiu**. O
  confirmatorio injeta **−2.0** (bem abaixo de z*): se a barreira for terminal, a profunda
  cai e a separacao aparece. Estimando extra: inversao-de-sinal de A_drive (S1), fracao de
  trios com sinal(A_drive)≠sinal(A_mlp), deep vs raso (piloto: 80–95% opostos nos profundos).
- **E-E ESCADA (estimacao, SEM MATA).** p(volta sustentada) por β∈{3,10}, frozen e train.

**Regra de contagem congelada:** volta SUSTENTADA unica por unidade injetada; trios
GT-area (t_area_drive ≤ t_res) com A_drive; born-dead por injecao ⟹ t_morte=0.

## 5. Pipeline e verificacoes ja rodadas (pre-registro)

- `motor_e5_conf.py --testes`: **7/7 PASSAM** — (1) PARIDADE loss_hist byte-identico a
  motor_e5b (4 casos); (2) determinismo; (3) injecao 1-nivel (rasa/profunda, deep bate a
  z*); (4) relu-frozen 0 sustentadas + own_move==0 (teorema, ambas profundidades); (5)
  histerese mata chatter (s103: 160 cruas → 0 confirmadas); (6) sustained⟺(revived&alive),
  hover⟺(revived&~alive); (7) consistencia (1 injetada/run, sust≤rev≤inj).
- `preditor_e5.py`: calibra δ_h + le o piloto (colapso de-chatter) ⟹ `config_e5.json`
  (todos os numeros com origem; nada digitado a mao).
- `analise_e5.py --sintetico`: **passa TUDO** — 6 mundos com asserts nos dois sentidos
  (tudo-reproduz; montante-zero→E-B MATA; transferencia-zero→E-C INCONCLUSIVO a N=50 +
  MATA atingivel a N grande; barreira-inexistente→E-D MATA-BARREIRA;
  relu-frozen-com-volta→E-A INVALIDO; poucos-eventos→INCONCLUSIVOS) + CP/Poisson exatos
  batendo com valores conhecidos (CP(0/10)=0.3085, CP(6/6)=0.5407). alpha/B lidos do config.
- `run_e5.py --estimativa`: 2 runs de teste (seeds 42-43) → tempo real + projecao (§7).
  **Nenhuma run confirmatoria executada.**

## 6. Como rodar (SO apos pre-registro + 11o parecer)

```
cd 05-experimentos/etapa5-transformer
setsid taskset -c 0-7 nice -n 19 ../../venv/bin/python run_e5.py --workers 4 \
       > run_e5.log 2>&1 < /dev/null &      # 800 runs, retomavel, destacado
../../venv/bin/python analise_e5.py         # le results-e5/ + config
```

## 7. Pendencias para o 11o parecer

- **E-C MATA quase-inatingivel a N=50** (mata_thresh=0.00617; MATA so a N≥800). E
  consequencia honesta da regra "½ CI-inf do piloto" com o piloto frozen-rasa em 2/20 —
  igual ao C1 do 4a.3. O poder de E-C e o gate positivo (≥6 sustentadas ⟹ TRANSFERE).
  Sinalizado, nao escondido. Alternativa (se o parecer quiser MATA util): usar o
  CI-inf do braco pooled ou do canal train — decisao do parecer.
- **E-B sob histerese e mais ESTRITO que o piloto.** O Δ>0 do piloto veio do contador
  chatter-propenso; a reproducao a H=2e5 depende de os climbs de montante excederem δ_h
  e sustentarem. Bem-posto e decidivel, mas NAO e carimbo — pode MATAR.
- **Pressao 1-injetada (3 vivas de 4) ≠ 2-injetadas do piloto.** Os p(volta|inj) do
  piloto (usados p/ thresholds) vieram de 2-de-4. A pressao mais branda tende a facilitar
  a volta (conservador p/ o MATA). Disclosure no config.
- **τ de n pequeno** (piloto: 8–22 voltas/braco). A janela e robusta (injetadas nascem
  em t=0; a janela liga no tempo de VOLTA), mas τ em si tem incerteza larga.
- **Runtime ~3.7 min/run** (H=2e5, celula frozen-injetada; a mais cara). 800 runs / 4
  workers ≈ 12 h — rodar em lote noturno, comecando pelo par relu (E-A/E-B, decisivos e
  baratos de interpretar) e pelo primario sp-b3.
