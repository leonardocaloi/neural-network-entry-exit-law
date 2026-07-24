"""Parametros congelados do p5-relu-morte. Data do rascunho: 2026-07-23.
Pre-registro passa a valer no commit deste arquivo + README.md, ANTES da run confirmatoria.
Mudancas posteriores: somente via secao 'Desvios' do README, nunca editando aqui em silencio.
"""

import numpy as np

# --- dados (fixos, sinteticos) ---
X = np.array([0.5, 1.5])
Y = np.array([0.5, 1.0])

# --- dinamica ---
ETA = 1e-3
T_PRIMARIA = 1e-3                  # braco confirmatorio de P-a/b
TEMPERATURAS_EXPLORATORIAS = [3e-3, 1e-2]
N_PASSOS = 500_000
SEEDS = list(range(20))
P0 = np.array([0.0, -0.5])         # init deterministica, interior do cone de morte

# --- rival suave (softplus_beta) e limite de rigidez (P-c) ---
BETAS = [2.0, 8.0, 32.0]

# --- estimadores e janelas (a priori) ---
K_EVENTO = 200                     # passos de janela antes/depois do evento p/ deriva
HORIZONTE_POS = 100_000            # janela pos-ressurreicao para P-b
MIN_PASSOS_VALIDOS_JANELA = 100    # K/2: evento com menos passos validos e descartado

# --- criterios pre-registrados ---
TOL_SALTO_REL = 0.25               # P-a: |IQM(salto) - Delta_j|/Delta_j <= 0.25
RATIO_SALTO_MIN = 3.0              # P-a: salto_relu / salto_softplus(beta=2) >= 3
FATOR_RATCHET = 5.0                # P-b: f_SGLD < f_nulo / 5
TOL_RIGIDEZ_REL = 0.25             # P-c: |t_med(32) - t_med(relu)|/t_med(relu) <= 0.25
MAX_FRAC_CENSURA = 0.30            # braco com >30% de runs sem ressurreicao = inconclusivo

# --- analise ---
N_BOOTSTRAP = 10_000               # bootstrap percentil do IQM
SEED_BOOTSTRAP = 12345

# --- alvo de P-a NO NIVEL DO PROTOCOLO (derivado a priori; ver README) ---
# O estimador de salto em janela K nao mede o Delta_j idealizado: (i) a forca decai ao
# longo da janela pos-evento (ODE linear na coordenada normal: fator kappa_j), e
# (ii) a norma de vetor com ruido infla: E||v+eps||^2 = ||v||^2 + 4*sigma_c^2.
# Alvo pre-registrado = SALTO_PROT[j]; comparar o medido com ELE, nao com DELTA_TEO.
N_DADOS = len(X)
DELTA_TEO = (Y / N_DADOS) * np.sqrt(X ** 2 + 1)          # salto idealizado por face (L2)
RHO = ETA * (X ** 2 + 1) / N_DADOS                       # taxa discreta por passo na normal
KAPPA = (1.0 - (1.0 - RHO) ** K_EVENTO) / (K_EVENTO * RHO)   # media do decaimento na janela
SIGMA_C = np.sqrt(2.0 * T_PRIMARIA / (ETA * K_EVENTO))   # sd por componente do v-hat
SALTO_PROT = np.sqrt((DELTA_TEO * KAPPA) ** 2 + 4.0 * SIGMA_C ** 2)

# instrumento P-b(i): deriva na fase morta deve ficar abaixo de 3x o erro-padrao teorico
FATOR_INSTRUMENTO = 3.0
