carregados 400 runs de /home/caloi/Projects/j-space/05-experimentos/etapa4-mlp-populacao/pilotos-4a2/pressao/taxa/results-4a3
# Confirmatório etapa-4a.3 — resultado

Grade: 400 runs. Janela elegível [0, 307536] (H−τ, τ do piloto β3).

## VALIDADE relu
VÁLIDO (relu 0 voltas).

## Por braço
braço | runs | mortes-din | λ/run (boot) | voltas | p(volta|morte) CP | p(volta|morte elegível) CP | ordenadas/voltas
---|---|---|---|---|---|---|---
sp-b3 | 100 | 35 | 0.350 [0.220, 0.500] | 16 | 0.457 [0.288, 0.634] | 0.500 [0.313, 0.687] | 13/16
sp-b10 | 100 | 27 | 0.270 [0.160, 0.390] | 10 | 0.370 [0.194, 0.576] | 0.385 [0.202, 0.594] | 6/10
sp-b300 | 100 | 19 | 0.190 [0.100, 0.290] | 4 | 0.211 [0.061, 0.456] | 0.222 [0.064, 0.476] | 3/4
relu | 100 | 17 | 0.170 [0.090, 0.260] | 0 | 0.000 [0.000, 0.195] | 0.000 [0.000, 0.195] | 0/0

## C1 (primário sp-b3): **REPRODUZ**
N_elig=30 (n*=10), voltas_elig=15, CP p(volta|elegível)=0.500 [0.313, 0.687], mata_thresh=0.0760
## C2 (coração, sp-b3): **SUSTENTA**
ordenadas=13/16 (piso 0.5, ≥6 voltas), CP-inf=0.544
## C4 (escada p(volta|morte) por β; SEM MATA)
- sp-b3: 0.457 [0.288, 0.634] (16/35)
- sp-b10: 0.370 [0.194, 0.576] (10/27)
- sp-b300: 0.211 [0.061, 0.456] (4/19)
- relu: 0.000 [0.000, 0.195] (0/17)
