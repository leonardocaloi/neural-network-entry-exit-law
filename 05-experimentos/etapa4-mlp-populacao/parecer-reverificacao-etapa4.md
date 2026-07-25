# Re-revisão focada (7º parecer) — etapa-4a, 2026-07-24

Revisor: agente de verificação independente sobre o commit 1cba68b (rodou
smoke, calibração v2, análise sintética e simulações próprias de nível; script
de verificação preservado em `verificacao-placebo-vies.py`). Texto integral,
sem edição. Respostas: seção 7º parecer de `respostas-parecer.md`.

---

## VEREDITO: NÃO APTO — 1 blocker novo (B-novo: placebo de atraso-permutado estruturalmente viesado; violação de nível demonstrada numericamente). B1–B4 do 6º parecer estão sanados; run.py está são; o resto é lista de correções menores.

A primeira linha, como manda a regra da casa: o pacote conserta tudo que o 6º parecer pediu, mas o conserto central de P-A — o placebo — não segura o nível que declara, e a validação sintética que deveria pegar isso valida contra um nulo que o instrumento real não pode produzir.

### 1. B1 sanado? SIM.

- Limiar calibrado ativo, sem código morto: `harness_mlp.py:80` (`lim_base = 2√(max(T,1e-7))`) e `:104` (`lim = lim_base·√n/normas[jstar]`) — o `*0` sumiu e `lim` é usado em `:105`.
- Observação pré-update, a cada passo: `harness_mlp.py:198-205` — `observa(t,…)` recebe z, a, res, v do MESMO estado, antes de `backward()` e do update. Off-by-one sanado.
- R̂ com a act da rede: `:107` usa a matriz `a = net.act(z)` passada, não relu hardcoded.
- Smoke rodado: **passa limpo** (fração nascida-morta 0.335, grad≡dif.finitas em float64, overfit-1-batch 5.25e-07, determinismo CPU, RNG do freeze alinhado Δloss=0.0, erf com g′=Φ(βz) exato <1e-9). Os dois checks que o parecer cobrava (grad-check, overfit) existem agora.
- Working tree ≡ commit 1cba68b para o diretório (git diff vazio) — o que revisei é o que está commitado.

### 2. Recalibração válida? SIM, com uma complacência de rótulo (correção menor).

Rodei `calibracao_regua_v2.py`: reproduz exatamente os números declarados — (a1) Δ3.4% em τ_res; (a2) 4.54%→1.91%, razão 0.42; (b) área vence 34/34, rel-err 0.534 vs 0.851; unit-tests (c) passam. O princípio "calibra-se o instrumento shippado" está cumprido, e o regime da calibração (β=300, βη≈0.10) é o regime confirmatório (βη=0.09) — isso é honesto.

O teste (a2) é **parcialmente** honesto: razão 0.42 vs 0.30 previsto. Decompondo err = A/β + C com os dois pontos: C ≈ 0.8pp de piso residual NÃO explicado por 1/β (~17% do erro em β=300). Dois pontos de β não identificam o expoente, e o assert aceita até 0.6. A frase "erro **identificado** como o atraso O(1/β) do teorema" (`calibracao_regua_v2.py:115-116`, README §2) deveria ser "**consistente com**; piso residual ~0.8pp não explicado". Não invalida nada: nenhum MATA consome a atribuição 1/β, e o erro total no regime de deploy (4.54%) está medido diretamente.

### 3. B2 sanado? SIM.

Flags `v_cruzou`/`v_min_abs`/`j_trocas` no `Regua` (`harness_mlp.py:85-87, 128-134`); `episodio_limpo()` usa todos os quatro critérios (`:155-163`); `analise.py:46-48` (`eh_limpo`) aplica com `alheios` calculado por `run.py:21-34`. Verifiquei a equivalência semântica das duas implementações de "alheios" (intervalo (t_morte, fim], próprio-k subtraído, inclusive múltiplos episódios do mesmo k) — batem. Menor: não há teste que asserte as duas implementações iguais no mesmo dado.

### 4. B3 sanado? Quase — nada disfarçado, três números na fronteira.

`preditor.py` congela grade, régua, critérios, RNG (777/778), k_crítico=16 (conferi a conta binomial), potência. `analise.py --sintetico` **passa nos dois sentidos** em todos os P's. Achados da varredura por números fora do config:
- `MIN_EP_SEED = 3` (`analise.py:24`) está no config só como prosa dentro de `P_A.teste` — promover a campo explícito.
- Piso do P-D usa episódios **limpos** do controle (`analise.py:234-236`); config diz só "do controle".
- O "mínimo 30 limpos/célula para MATA" do README §4 está implementado só para P-A (`valida`); P-B não tem gate (auto-protegido pelo teste de sinais exigir S≥10 seeds não-empatados, mas README↔código divergem).

Nenhum é escolha pós-hoc disfarçada; são inconsistências de documentação. **Porém o `--sintetico` valida contra um nulo impossível — ver item 6.**

### 5. B4 sanado? SIM.

`config` `device.pin="cpu"` + `torch_deterministic` + `threads_por_worker=1`; `harness_mlp.py:29` `use_deterministic_algorithms(True)` no import; `run.py:38-39,50` respeita (`set_num_threads` do config, `device=CFG["device"]["pin"]`). Determinismo CPU confirmado no smoke.

### 6. Decidibilidade dos MATAs — AQUI ESTÁ O BLOCKER.

**Fato mecânico confirmado empiricamente:** t_sinal ≤ t_area ≤ t_res sempre (0 violações em 513 episódios reais que gerei com o harness, m=64). O instrumento **para de medir na ressurreição** — t_area nunca ultrapassa t_res. Consequência 1: o leg "área vence sinal" é quase-tautológico (área vence sempre que t_area > t_sinal estritamente); só mata o competidor-1. Todo o peso probatório de P-A está no placebo — como a própria casa desenhou.

**O placebo tem viés estrutural de truncamento.** O atraso da área é censurado na duração do próprio episódio (episódios onde a área não disparou caem de `com_area`); o atraso permutado de outro episódio NÃO herda esse truncamento. Sob um nulo puro — atraso independente da duração, truncado exatamente como o Regua trunca — medi, usando as funções reais de `analise.py` (`fracao_vitorias_placebo`, `teste_sinais`, α e mínimos do config):

- P(área vence placebo) por episódio sob o nulo: 0.52–0.57 (deveria ser ≤ 0.5);
- FPR do teste-placebo (α declarado 0.01): **0.017–0.033** com 8 limpos/seed, **0.05–0.23** com 20/seed, **0.12–0.58** com 40/seed. O teste de sinais entre seeds protege contra ruído, não contra viés sistemático pequeno: com muitos episódios/seed a fração-de-vitórias concentra acima de 0.5 e o teste dispara sob o nulo.
- E o regime real é de MUITOS episódios: medi ~170 episódios completos/seed já em 20k passos (m=64); o horizonte é 100k. O α é violado por uma a duas ordens de magnitude exatamente no regime esperado.

**Por que a validação sintética não pegou:** `_fabrica` modo "nulo" (`analise.py:332-334`) fabrica t_area/t_sinal sem truncamento — pode exceder t_res, coisa que o instrumento real nunca produz. O "mundo-nulo dispara MATA" foi validado contra um nulo estruturalmente mais fácil que o nulo real. É uma violação do espírito da regra do 5º parecer: a análise foi testada, mas contra o processo de medição errado.

**P-NULL fecha o buraco do artefato-de-régua?** Em princípio sim, e sob este viés o desfecho mais provável é P-NULL disparar "instrumento viciado → tudo inválido" (honesto, mas run noturna desperdiçada e experimento indecidível). O caso perigoso: o braço semgrad tem estatística de episódios diferente (difusão pura: durações/atrasos/contagens distintas das do softplus) — o viés pode ser fraco no semgrad e forte no softplus, e aí sai falsa "MEMORIA DEMONSTRADA". P-NULL não garante o nível do placebo no regime que importa.

**Testei um conserto ingênuo** (elegibilidade simétrica: j vale para i sse atraso_j≤dur_i ∧ atraso_i≤dur_j): reduz mas não zera o vazamento (FPR 0.157 em 40 ep/seed) e **colapsa a potência** (0.01 sob memória forte — o filtro vaza informação de duração para o placebo). O conserto não é uma linha; é desenho. Caminhos possíveis (decisão da casa, não minha): calibração empírica do placebo pelo braço semgrad (compara a estatística entre braços, herdando o truncamento medido — exige reportar comparabilidade das distribuições de episódio); ou estatística de tracking com censura modelada (território de análise de sobrevivência); ou placebo com rejeição cujo nível seja demonstrado por simulação. **Condição de saída do blocker:** o `--sintetico` passa a fabricar o nulo COM truncamento (t_area=-1 quando o atraso excede a duração; jamais t_area>t_res) e demonstra FPR≈α para n_ep/seed ∈ {8, 40, 200}, antes do freeze.

**P-B (erf in-class):** teste genuíno, mas fraco na direção "sustentada". Nota técnica: no mesmo β, g′=Φ(βz) é ~1.6× mais nítida que sigmoid(βz) (Φ′(0)=0.399 vs 0.25) — ainda dentro-da-classe, então testa L5.2 de verdade; a direção MATA (Δ>5% significativo) é falsificável. Testes de universalidade são inerentemente "prever não-diferença"; com o rótulo honesto que está, aceito.

### 7. Buracos novos das correções — um blocker (acima) e quatro menores.

- **"≥2 de 3 incluindo m=16": OK.** Exige coerência com a teoria (m=16 é onde a lei mais deve valer); torna a demonstração mais difícil, não mais fácil; a semântica 3-vias é bem definida a priori.
- **Exclusão de seeds <3 limpos: OK.** Não vicia o nível (a exclusão independe da direção da vitória sob o nulo); k_crítico é recalculado para o S efetivo (`analise.py:95-100`).
- **mediana_dur: OK.** Justificada (t_res cru dominado pelos instantes de morte) e congelada antes da run.
- **Menor — P-D com S=0:** se `med_seed` fica vazio, ci=(0,0), mediana=0 → `contem0` e |0|<piso → classe "sem" (`analise.py:231-243`) — pode declarar "MATA: sem back-reaction" com zero dados. Improvável (33% nascidos-mortos), mas é análise pré-registrada: guardar classe "sem" com S mínimo.
- **Menor — episódios sem t_area invisíveis:** limpos-completos onde a área nunca disparou caem silenciosamente da fração-de-vitórias e de todos os reports — medi **17%** (87/513) em m=64/20k. É evidência contra a lei da área que fica invisível. Pré-registrar o report dessa fração (P-E) e a regra de contagem.
- **Menor — potência declarada assume 30 limpos/SEED** (`preditor.py:19`, "n_ep=30"), mas o gate de validade exige 30/CÉLULA (1.5/seed). O MDE 0.62 vale só no cenário otimista (que os meus números empíricos sugerem que se realiza, mas a condição deveria estar escrita no README §5).
- **Freeze metade par: OK.** Determinístico, índices permutáveis no init, RNG alinhado validado no smoke (Δloss=0.0).

### 8. run.py — SÃO. Nenhum bug que estrague a run noturna.

- Retomada por arquivo-existente com escrita atômica tmp+`os.replace` (`run.py:60-63`) — interrupção não deixa .json.gz corrupto que seria pulado depois. Correto.
- `anota_alheios` via bisect (`run.py:21-34`): verifiquei contra a semântica de `episodio_limpo` — intervalos e subtração do próprio-k batem, inclusive com múltiplos episódios do mesmo neurônio.
- Python 3.14.6 + ProcessPoolExecutor + torch nos filhos: o mesmo padrão rodou na calibração (6 workers) sem incidente.
- Timing medido: 44s (m=16) a 60s (m=256) por run de 100k passos → 600 runs / 6 workers ≈ 1.7–1.9h. O dimensionamento declarado bate.
- Menores: falha de um job aborta o pool (retomável, aceitável); manifest não registra commit/hash do config; o "vigia de load" do README §7 não existe no código.

### Resumo executivo

B1, B2, B3, B4: sanados e verificados por execução. P-C rebaixado e P-B reenquadrado: corretos. O que reprova é uma coisa só, mas é a coluna vertebral: **o placebo — a resposta da casa à crítica de decidibilidade de P-A — não segura o α que declara** (FPR até 58× o nominal no regime realista de episódios/seed), por um viés de truncamento que a validação sintética não podia pegar porque fabrica um nulo que o instrumento real não produz. Consertar o placebo (ou trocar a estatística), refazer o `--sintetico` com nulo truncado e demonstrar nível — aí o pacote está pronto para o commit de pré-registro. As sete correções menores (rótulo do (a2), MIN_EP_SEED como campo, guarda S>0 no P-D, report de sem-t_area, condição da potência, prosa do piso P-D, gate P-B) cabem no mesmo passe.

---

# 2ª rodada (mesmo revisor, contexto preservado) — VEREDITO: APTO (commit 8788b83)

Condição de saída verificada e reproduzida pelo revisor: `_fabrica` respeita o
invariante do instrumento; FPR = 0.000/0.000/0.010 (n_ep 8/40/200, α=0.01) e
TPR = 1.000 reproduzidos; `_km`/`_amostra_km` verificados contra casos à mão
(ECDF exata; massa censurada preservada, 0.375 exato). **Ataque próprio do
revisor:** 21 configurações de nulo truncado — 7 famílias de F (cauda pesada
σ=1.5, bimodal, quantizada com empates, átomo 30%, truncamento pesado,
uniforme, durações largas σ=1.2) × n_ep ∈ {3, 40, 200}, S=20 seeds reais.
Pior FPR: 0.040 (compatível com α=0.01 dado R=100 e 21 comparações); resto
≤ 0.02. Nível robusto. Potência sob tracking fraco (atraso=0.5·dur,
σ_rel=0.5): TPR 0.07/0.42/1.00 para n_ep 10/40/200 — com ~170 episódios/seed
do piloto, adequada; escassez de limpos (m=256) erra na direção honesta
(MATA/parcial, não falso positivo). `preditor.py` regenera o config
byte-idêntico; as 7 menores conferidas no diff.

Pontos de ceticismo da casa respondidos: (a) KM dos mesmos dados = bootstrap
paramétrico clássico, consistente sob H0, custo de amostra finita limitado
pelas simulações (inclusive canto n_ep=3); (b) má-especificação de F̂ sob H1
afeta só potência; (c) FPR=0.000 é conservadorismo leve (B=199 na demo;
análise real usa B=1000); (d) atacado com 7 famílias — segurou.

Não-bloqueantes registradas: (1) caveat de tendência temporal comum
(rejeição 10% só sob tendência ~fator e) → desarmado com o descritivo
pré-registrado corr(dur,t_morte)/corr(atraso,t_morte) no P-E (implementado
antes do freeze); (2) edge T_b=None em p_calibrado — anti-conservador em
princípio, nunca disparou em ~2100 mundos; mantido.
