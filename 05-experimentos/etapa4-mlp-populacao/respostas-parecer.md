# Respostas ao 6º parecer (etapa-4) — 2026-07-24

Cada item do parecer → o que foi feito, onde está a evidência.

| Item do parecer | Resposta | Evidência |
|---|---|---|
| **B1** régua shippada ≠ calibrada (`*0` apaga o limiar norm-scaled) | `*0` removido; limiar `2√(nT)/‖x_j*‖` ativo; observação PRÉ-update (off-by-one sanado) a CADA passo (Δsnapshot=1); R̂ reconstruído com a act da rede (matriz `a` passada) | `harness_mlp.py` (Regua.observa); smoke OK |
| **B1-recalibração** calibração v1 validava outro instrumento | Recalibrado o instrumento EXATO (classe importada, v treinável, toy-em-harness). Achado: o erro residual é o atraso O(1/β) do PRÓPRIO teorema — 4.54%→1.91% com β 300→1000; área vence sinal 34/34 | `calibracao_regua_v2.py` (roda limpo) |
| **B2** v_k≈0 / cruzamento não modelados | Flags por episódio: `v_cruzou`, `v_min_abs`; morte-por-saída excluída por φ_v=1/(4√m) (escala do init ⟹ fração excluída comparável entre m; sensibilidade ×2/÷2 exploratória); frações reportadas por célula (P-E) | harness + README §2-§3; unit-tests (c) |
| **B3** preditor/config/análise/run inexistentes | `preditor.py` → `config_etapa4.json` (freeze); `analise.py` com smoke sintético; `run.py` Pool nice-19 | arquivos no diretório |
| **B4** pareamento CPU/GPU quebrado | Device ÚNICO pinado: toda a confirmatória em CPU (modelos minúsculos; GPU não paga overhead e quebraria pareamento); `use_deterministic_algorithms(True)`; 1 thread/worker; declarado no config. GPU fica para etapa-4b/5 | harness + config |
| **P-C decidido a priori** | Rebaixado a ÂNCORA/ESTIMAÇÃO com censura modelada (permanente-por-mecanismo vs censurado). Sem MATA binário suave≈hard | README §3 P-C |
| **P-B fora da classe (GELU real)** | Dividido: braço **erf-smoothing** g′=Φ(βz) (classe provada, mesmo β ⟹ mesma nitidez) carrega o MATA; **GELU padrão** vira P-B′ exploratório além-do-teorema, sem "devem coincidir" | README §3; harness act "erf" (g′=Φ(βz) verificado no smoke, erro <1e-9) |
| **P-A ordenação estrutural (t_area ≥ t_sinal)** | Nulo assimétrico: **placebo de atraso-permutado** (mesma distribuição marginal de atrasos, zero informação por-episódio). Área tem de vencer sinal E placebo; P-NULL exige que no braço difusão-pura ela NÃO vença o placebo | README §3 P-A/P-NULL |
| Confundidor líder j* | q_j=0 no filtro-limpo (1D: 0 trocas medidas pelo parecer); recalibração do filtro é pré-condição declarada da etapa-4b (2D/MNIST) | README §2, escopo 4a |
| Episódios correlacionados / filtro-limpo só prometido | `episodio_limpo()` é código compartilhado (calibração, análise, smoke); fração-limpa × m é entregável P-E; mínimo 30 limpos/célula para MATA | harness; README §3-§4 |
| Snapshot a cada 5 | Δsnapshot=1 (igual à calibração); custo medido (~0.5 ms/passo) e absorvido no horizonte | dimensionamento no commit bd09a6b |
| Menores: n≈256 vs n=32; off-by-one | n=32 congelado (documentado como desvio do rascunho); off-by-one sanado (observação pré-update, estado consistente) | README §4; harness |
| Numeração vN/expNN + OSF | Alias **exp05** no config; pasta mantém o nome. OSF/circulação adiados por decisão do pesquisador (memória do projeto) | README §7 |
| Smoke sem grad-check/overfit | Adicionados: gradiente vs diferenças finitas (float64, <1e-6) e overfit-1-batch (loss 5e-7); âncora nascida-morta 0.335 mantida; alinhamento RNG do freeze testado (Δloss=0) | smoke do harness v2 |
