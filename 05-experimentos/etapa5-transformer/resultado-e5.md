carregados 800 runs de /home/caloi/Projects/j-space/05-experimentos/etapa5-transformer/results-e5
# Confirmatorio etapa-5 (transformer FFN) — resultado

Runs: 800. delta_h=0.0476. H=200000.

## E-A VALIDADE relu-frozen
VALIDO (relu-frozen 0 voltas sustentadas).

## Tabela por celula x profundidade (p_sust = sustained/inj)
cell | depth | inj | sust | rev | hover | p_sust CP | p_sust boot | area_ok/trios
---|---|---|---|---|---|---|---|---
gelu-frozen | -2.0 | 50 | 0 | 0 | 0 | 0.000 [0.000, 0.071] | 0.000 [0.000, 0.000] | 0/0
gelu-frozen | -0.3 | 50 | 29 | 29 | 0 | 0.580 [0.432, 0.718] | 0.580 [0.440, 0.720] | 24/29
gelu-train | -2.0 | 50 | 8 | 8 | 0 | 0.160 [0.072, 0.291] | 0.160 [0.060, 0.260] | 8/8
gelu-train | -0.3 | 50 | 43 | 44 | 1 | 0.860 [0.733, 0.942] | 0.860 [0.760, 0.940] | 36/43
relu-frozen | -2.0 | 50 | 0 | 0 | 0 | 0.000 [0.000, 0.071] | 0.000 [0.000, 0.000] | 0/0
relu-frozen | -0.3 | 50 | 0 | 0 | 0 | 0.000 [0.000, 0.071] | 0.000 [0.000, 0.000] | 0/0
relu-train | -2.0 | 50 | 2 | 2 | 0 | 0.040 [0.005, 0.137] | 0.040 [0.000, 0.100] | 1/2
relu-train | -0.3 | 50 | 18 | 19 | 1 | 0.360 [0.229, 0.508] | 0.360 [0.240, 0.500] | 12/18
sp-b10-frozen | -2.0 | 50 | 0 | 0 | 0 | 0.000 [0.000, 0.071] | 0.000 [0.000, 0.000] | 0/0
sp-b10-frozen | -0.3 | 50 | 9 | 9 | 0 | 0.180 [0.086, 0.314] | 0.180 [0.080, 0.300] | 7/9
sp-b10-train | -2.0 | 50 | 2 | 2 | 0 | 0.040 [0.005, 0.137] | 0.040 [0.000, 0.100] | 2/2
sp-b10-train | -0.3 | 50 | 27 | 30 | 3 | 0.540 [0.393, 0.682] | 0.540 [0.400, 0.680] | 26/27
sp-b3-frozen | -2.0 | 50 | 0 | 0 | 0 | 0.000 [0.000, 0.071] | 0.000 [0.000, 0.000] | 0/0
sp-b3-frozen | -0.3 | 50 | 22 | 22 | 0 | 0.440 [0.300, 0.587] | 0.440 [0.300, 0.580] | 21/22
sp-b3-train | -2.0 | 50 | 4 | 4 | 0 | 0.080 [0.022, 0.192] | 0.080 [0.020, 0.160] | 4/4
sp-b3-train | -0.3 | 50 | 39 | 39 | 0 | 0.780 [0.640, 0.885] | 0.780 [0.660, 0.880] | 34/39

## E-B MONTANTE (Delta = p_sust train - frozen; boot por seed)
- sp-b3: **REPRODUZ** Delta=0.210 CI[0.120,0.300] (train 43/100 vs frozen 22/100)
- sp-b10: **REPRODUZ** Delta=0.200 CI[0.110,0.280] (train 29/100 vs frozen 9/100)
- gelu: **REPRODUZ** Delta=0.220 CI[0.120,0.320] (train 51/100 vs frozen 29/100)
- relu: **REPRODUZ** Delta=0.200 CI[0.120,0.280] (train 20/100 vs frozen 0/100)

## E-C TRANSFERENCIA (sp-b3-frozen -0.3): **TRANSFERE**
sust=22/50 CP=0.440 [0.300, 0.587] n*=6 mata_thresh=0.0062

## E-D BARREIRA-GELU (frozen): **CONSISTENTE-COM-BARREIRA (confundido com profundidade)**
rasa(-0.3) 29/50 CP=0.580 [0.432, 0.718]; profunda(-2.0) 0/50 CP=0.000 [0.000, 0.071]
  sinal A_drive!=A_mlp rasa: 2/50 (frac=0.04)
  sinal A_drive!=A_mlp profunda: 46/50 (frac=0.92)

## E-E ESCADA p_sust por beta (sem MATA)
- sp-b10-frozen: 0.090 [0.042, 0.164] (9/100)
- sp-b10-train: 0.290 [0.204, 0.389] (29/100)
- sp-b3-frozen: 0.220 [0.143, 0.314] (22/100)
- sp-b3-train: 0.430 [0.331, 0.533] (43/100)
