# Revisão adversarial do desenho ETAPA 4 (etapa4-mlp-populacao) — 2026-07-24 (6º parecer)

Revisor: agente adversarial independente (rodou smoke + calibração + checagens
mecânicas próprias; reproduziu 18/18 da calibração). Texto integral abaixo,
sem edição. Respostas/correções da casa: ver `respostas-parecer.md`.

---

**VEREDITO: NÃO APTO para preditor-freeze / pré-registro.** 4 blockers + 3 correções-de-desenho + menores. Não é reprovação da hipótese: a parte cara e genuína (P-A sob m-corpos, P-D populacional, formas P-E) é sólida. Mas **o instrumento que o harness embarca não é o que a calibração validou**, a camada inteira de pré-registro não existe, o pareamento por seed está fisicamente quebrado sem device-pin, e dois MATAs (P-C, P-B) estão decididos a priori / fora do teorema.

Rodei: `harness_mlp.py --smoke` (ok), `calibracao_regua.py` (ok, reproduz 18/18), e checagens mecânicas de device/v_k/ordenação/chatter. Nenhuma run confirmatória.

## 1. RÉGUA GENERALIZADA — a conta e os bugs

**A régua está certa em 1ª ordem, e v=1 recupera o toy.** Refiz a dinâmica da margem morta com v_k treinável e multi-dado: para o neurônio morto k, `dz_{k,j*}/dt ≈ +(1/n)·R_{−k,j*}·v_k·act'(z_{k,j*})·‖(x_{j*},1)‖²`. Como `act'>0`, **o sinal do drift da margem = sinal de v_k·R_{−k,j*} = sinal de R̂_k**. A álgebra do código confere: `v*(−res + v·act_j)` colapsa em `v_k·R_{−k,j*}` (o termo `v·act_j` cancela a auto-contribuição de k dentro de res), **linear** em v_k. Correto.

**MAS toda a dependência em v é NÃO CALIBRADA** — o toy tem v≡1. Três problemas concretos:

- **v_k cruza zero (o "morto de outra forma"):** quando v_k→0, `R̂_k=v_k·R_{−k}→0` e o zero de Â_k deixa de marcar ressurreição — o drift da margem `∝v_k` também some, então (v_k≈0, morto) é quase-ponto-fixo (neurônio duplamente preso). No smoke de 2000 passos v_k não cruzou sinal (params mortos ~congelam, coerente com o teorema), mas o horizonte confirmatório é 10⁴–2·10⁵ passos com T>0: v_k vai difundir e cruzar. A régua não detecta cruzamento de v_k nem o flag do episódio; Â_k acumula com troca de sinal sem que nada acuse.

- **Debounce `vivo = zmax > 2√T` é BUG (linha 65).** `lim = self.lim_fator*np.sqrt(n)/normas[jstar]*0` — o `*0` é código morto e apaga **exatamente** o fator `√n/‖x_{j*}‖`. Medido: no MLP (n=32) o threshold `2√T` do harness é **2.6–5.7× menor** que o calibrado `2√(nT)/‖x_{j*}‖`, e ignora a norma do dado líder que o toy usa. Removendo o `*0` e usando `lim` na linha seguinte, o harness reproduz a régua calibrada **exatamente** (2√(nT)/‖x‖ ≡ o toy com N↔n). Conserto de uma linha, mas hoje o harness mede com instrumento mais frouxo que o validado.

## 2. DECIDIBILIDADE (P-A..P-E) — com o que já se sabe

- **P-C é confirmação quase-garantida, não teste.** A calibração JÁ mostra suave≠hard: relu-duro prev(área)=13760 vs ressurreição real **difusiva** em 392361 (28× depois); softplus por área ~15k. O próprio §5 admite que o lado-hard é "quase construção" (EJRW23 field-first). Pior: o desfecho binário "suave≈hard" fica **decidido pelo horizonte** escolhido a priori — qualquer horizonte entre a escala-área (~15k) e a escala-difusão (~392k), ambas conhecidas, GARANTE "hard permanente, soft não". Isto é o pecado que os 4 pareceres mataram. **Rebaixar P-C de MATA binário a âncora/estimação;** o genuíno é se o mecanismo sobrevive ao acoplamento m-corpos, com censura modelada — não "≈".

- **P-B usa GELU FORA da classe do teorema.** O draft §reg é explícito: *"GELU itself, zΦ(z), has non-monotone derivative and is not a member of the family for any β"* — o teorema cobre erf-**smoothing** (g'=Φ(βz)), monótona. O harness roda GELU **real** `z*Normal.cdf(√β·z)`. Então "predição forte: DEVEM coincidir" não é sustentada pelo teorema para a ativação rodada. Ou trocar por erf-smoothing (classe provada), ou re-etiquetar P-B como teste ALÉM do teorema (sem "devem coincidir"). Bônus: `√β` no GELU vs `β` no softplus → nitidez não-comparável, o par não é "mesma dureza".

- **P-A é genuinamente incerto — mas tem ordenação estrutural embutida.** Por construção `t_area ≥ t_sinal` (Â_k só cruza 0 depois de R̂_k virar sinal e sustentar; confirmei na lógica da Regua). O rival sem-memória (sinal) sempre dispara ≤ área; qualquer atraso de memória faz a área vencer mecanicamente (é o que dá 18/18 no toy). Não invalida P-A, mas o nulo do MATA deve ser **assimétrico consciente disso** — senão a potência declarada é inflada.

- **P-D / P-E:** genuinamente incertos, bem posicionados (P-E já rebaixado a estimação). OK.

## 3. CONFUNDIDORES DE POPULAÇÃO

- **v_k≈0 (morto-por-saída) — confundidor de 1ª ordem, não exótico, e PIORA com m (o eixo do teste).** Init v~N(0,1/m): fração |v_k|<0.1 medida = **0.38 / 0.67 / 0.87** para m=16/64/256. Em m=256, 87% dos neurônios contribuem ~0 para f seja qual for a margem; contam como "vivos" pela régua zmax>lim. Contamina fração-permanente (P-C/P-E) e escala R̂_k (∝v_k, minúsculo para a maioria). Precisa de definição de morte-por-saída (|v_k|<φ_v) na régua.
- **Líder j* oscilando:** medido no regime 1D (m=32, 4000 passos) → **0 trocas** (params mortos congelam, argmax 1D estável). Confundidor menor em 1D; pode reaparecer em 2D/MNIST ou com v-drift. Honesto: não é blocker para 1D, mas o filtro de estabilidade de líder deve existir antes de 2D/MNIST.
- **Episódios sobrepostos/correlacionados:** a régua per-k usa R_{−k} (exclui só k); com k,l morrendo/ressuscitando juntos, R_{−k} salta a cada switch alheio (H4 quebra por construção). O §8 promete filtro de "episódio limpo (≤q switchings)" mas **não existe no código** nem há medição de fração-limpa. É o risco científico central — precisa ser código pré-registrado + fração-limpa reportada vs m.
- **Snapshot a cada 5 passos:** quantiza t_morte/t_area/t_sinal/t_res em múltiplos de 5 — exatamente as diferenças que P-A compara. O toy calibrou **passo-a-passo**. Perde episódios curtos e enviesa P-A. Escolher Δsnapshot e recalibrar com ele.

## 4. GPU vs CPU — pareamento quebrado (blocker)

Confirmei: **mesma seed, mesmos 500 passos, CPU loss 0.091770 vs CUDA 0.093354 (Δ=1.6e-3).** Streams RNG do torch diferem por device; o init (`torch.manual_seed` global) também. Todo P-A/B/C/D é pareado por seed/init → se braços/células caírem em devices diferentes (ou fallback CPU), o pareamento é **inválido**. Não declarado. Precisa: pinar device por par (idealmente toda confirmatória num device) + flags de determinismo (`use_deterministic_algorithms`, cudnn) declarados no config.

## 5. O QUE FALTA CONGELAR (ordem)

Estado atual: **existe só calibracao_regua.py + harness_mlp.py.** Não há preditor.py, config.json, analise() nem run.py.

1. **Consertar a régua do harness:** remover `*0` (usar threshold norm-scaled = o calibrado); escolher Δsnapshot; adicionar morte-por-saída φ_v e flag de v_k-cruza-zero; usar `self.act` (não relu) na reconstrução de R_{−k}.
2. **RE-CALIBRAR a régua EXATA do harness** (v treinável, Δsnapshot escolhido, threshold consertado) contra o toy — a calibração atual valida outro instrumento (v≡1, passo-a-passo, threshold norm-scaled).
3. **Filtro de episódio-limpo como código** + medição de fração-limpa vs m (risco-1).
4. **Decisões de desenho:** GELU→erf ou reframe P-B; P-C→âncora/estimação; nulo assimétrico P-A.
5. **preditor.py:** congela P-A..P-E, φ, φ_v, piso |Δ|, q, erro-MC, seeds virgens, grade (T,η,m), Δsnapshot, horizonte, **orçamento βη** (default β=300/η=1e-3 → βη=0.30 → ~7% erro intrínseco da régua no toy; regime 3.5% exige η≤3e-4), potência/MDE por célula.
6. **analise() como código + smoke sintético** (regra do 5º parecer: análise sempre testada no smoke-gate; foi o blocker do v4).
7. **run.py + smoke completo:** o smoke atual NÃO tem grad-vs-diferenças-finitas nem overfit-1-batch (o protocolo pede; o v4 tinha). Tem: init fração-morta (0.335 ✓), finitude ✓, determinismo-CPU ✓, caminho-GPU ✓. Adicionar device-pin (B4).
8. Censura ligada a P-C (permanente-por-mecanismo vs censurado).
9. Numeração vN vs expNN + OSF.

**Menores:** README diz 1D n≈256 mas `dados_1dreg` default n=32 (8×); 2D e MNIST não implementados. Régua lê margens/v pós-update mas res pré-update (off-by-one em `observa()`).

**Resumo:** a hipótese e a espinha do experimento estão certas; o que reprova é (B1) régua shippada ≠ régua calibrada — o `*0` literalmente apaga a reconciliação; (B2) v_k≈0/cruzamento não modelado e dominante em m grande; (B3) preditor+análise inexistentes; (B4) device não pinado. P-C e P-B, como MATA binário, estão decididos a priori / fora do teorema. Consertar régua → recalibrar → erguer preditor+análise → então pré-registrar.
