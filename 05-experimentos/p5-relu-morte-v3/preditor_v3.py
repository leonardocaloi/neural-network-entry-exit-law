"""Preditor A PRIORI do v3: modelo reduzido 2.0 = BM no cone (exato) x excursao 1D
com cronograma ESTOCASTICO (OU exato de B, AR(1) por automodo — sem parametro livre).
NAO roda a dinamica 4D. Grade e reguas herdadas de ../p5-relu-morte-v2/config_v2.json.
"""
import json

import numpy as np

X = np.array([0.5, 1.5]); Y = np.array([0.5, 1.0]); N = 2
ETA = 1e-3; ETA_B = 1e-4
B_INIT = np.array([0.2, 0.05])
N_PASSOS = 500_000
NORMA = np.sqrt(X ** 2 + 1)
RHO = ETA * NORMA ** 2 / N
DEPTH2APEX = np.sqrt(1 + X.max() ** 2)
FRAC_CAPTURA, SIG_MIN_CAPTURA = 0.5, 4.0
W_LOCAL = 70_000
T_FREEZE = 25_000
COARSE = 100                                   # passo grosso do cronograma OU
RNG_CAN = 20260724

with open('../p5-relu-morte-v2/config_v2.json') as fh:
    CFG2 = json.load(fh)
CELULAS = [(c["d0"], c["T"], c["alvo"]) for c in CFG2["celulas"]]
TAU_ZERO_DET = {t: {"0.0003": 10650, "0.001": 6997, "0.003": 3972}[t] for t in
                ("0.0003", "0.001", "0.003")}  # janelas deterministicas (parecer v2)

# ---------- OU exato de B por automodo (AR(1) com passo COARSE) ------------------
G = np.mean([np.outer([x, 1.0], [x, 1.0]) for x in X], axis=0)
lam, V = np.linalg.eigh(G)                     # lam asc; V colunas ortonormais
P_STAR = np.linalg.solve(np.array([[X[0], 1], [X[1], 1]]), Y)
A_MODE = (1.0 - ETA_B * lam) ** COARSE         # decaimento por bloco
def var_mode(T):                                # variancia acumulada por bloco
    a2 = (1.0 - ETA_B * lam) ** 2
    return 2 * ETA_B * T * (1 - a2 ** COARSE) / (1 - a2)

def cronogramas(T, m, n_passos, seed, freeze=None):
    """m amostras do residuo R_j(t) = y_j - z_Bj(t) (linear, sem clipping — declarado)
    em grade grossa; retorna (n_blocos, m, N_faces)."""
    rng = np.random.default_rng(seed)
    nb = n_passos // COARSE
    c0 = (B_INIT - P_STAR) @ V                  # coords modais iniciais (deterministicas)
    C = np.tile(c0, (m, 1))
    sd = np.sqrt(var_mode(T))
    out = np.empty((nb, m, N))
    XB = np.vstack([X, np.ones(N)])             # (2, N): z_B = p_B @ XB
    kf = None if freeze is None else freeze // COARSE
    for k in range(nb):
        if kf is None or k < kf:
            C = C * A_MODE + rng.standard_normal((m, 2)) * sd
        pB = C @ V.T + P_STAR
        out[k] = Y[None, :] - pB @ XB
    return out

# ---------- MC canonico do cone (reuso da logica do v2, compacto) ----------------
def mc_cone(c, sigma, m, horizonte, seed, bloco=2000):
    rng = np.random.default_rng(seed)
    pos = np.tile([0.0, -c], (m, 1))
    tau = np.full(m, horizonte, np.int64); face = np.full(m, -1, np.int64)
    vivos = np.arange(m); t = 0
    while len(vivos) and t < horizonte:
        k = min(bloco, horizonte - t)
        cam = pos[vivos, None, :] + np.cumsum(
            rng.standard_normal((len(vivos), k, 2)) * sigma, axis=1)
        z = cam @ np.vstack([X, np.ones(N)])
        saiu = (z > 0).any(axis=2); algum = saiu.any(axis=1)
        idx = np.argmax(saiu, axis=1); quem = np.nonzero(algum)[0]
        gl = vivos[quem]
        tau[gl] = t + idx[quem] + 1
        face[gl] = np.argmax(z[quem, idx[quem]], axis=1)
        pos[vivos] = cam[:, -1]; vivos = vivos[~algum]; t += k
    return tau, face

class Canon:
    S, H, M = 0.02, 80_000, 30_000
    def __init__(self):
        self.tau, self.face = mc_cone(1.0, self.S, self.M, self.H, RNG_CAN)
    def amostra(self, d0, T, m, seed):
        c = d0 * DEPTH2APEX; sig = np.sqrt(2 * ETA * T)
        f = (self.S * c / sig) ** 2
        if f < 5.0:
            return mc_cone(c, sig, m, N_PASSOS, seed)
        rng = np.random.default_rng(seed)
        i = rng.integers(0, self.M, m)
        tau = np.minimum((self.tau[i] * f).astype(np.int64), N_PASSOS)
        face = np.where((self.tau[i] * f) < N_PASSOS, self.face[i], -1)
        return np.where(face == -1, N_PASSOS, tau), face

# ---------- excursao 1D com cronograma amostrado ---------------------------------
def excursao(tau, face, R, T, seed):
    """R: (n_blocos, m, N). Cada walker usa SEU cronograma. Retorna capturas."""
    rng = np.random.default_rng(seed)
    m = len(tau); sigma = np.sqrt(2 * ETA * T)
    sigx = np.sqrt(N * T) / NORMA
    x = np.zeros(m); cap = np.zeros(m, bool)
    ativo = face >= 0
    t_loc = np.zeros(m, np.int64)
    nb = R.shape[0]
    while ativo.any():
        ids = np.nonzero(ativo)[0]
        ruido = rng.standard_normal((len(ids), 1000)) * sigma
        for s in range(1000):
            kb = np.minimum((tau[ids] + t_loc[ids]) // COARSE, nb - 1)
            rj = R[kb, ids, face[ids]]
            xst = np.maximum(rj, 0.0) / NORMA[face[ids]]   # R>0 = empuxo p/ fora
            drift = -RHO[face[ids]] * (x[ids] - xst) * (x[ids] > 0)
            x[ids] = x[ids] + drift + ruido[:, s]
            lim = sigx[face[ids]]
            pega = (x[ids] >= FRAC_CAPTURA * xst) & (xst >= SIG_MIN_CAPTURA * lim)
            cap[ids] |= pega
            t_loc[ids] += 1
        fim = cap[ids] | (tau[ids] + t_loc[ids] >= N_PASSOS) | (t_loc[ids] >= W_LOCAL)
        ativo[ids[fim]] = False
    return cap

# ---------- main -----------------------------------------------------------------
if __name__ == "__main__":
    M = 12_000
    cone = Canon()
    print(f"canonico: censura={np.mean(cone.face==-1):.3f}, face1={np.mean(cone.face==1):.3f}")
    out = {"constantes": {**CFG2["constantes"], "SEEDS_V3": [1000, 1149],
                          "T_FREEZE": T_FREEZE, "COARSE": COARSE,
                          "TAU_ZERO_DET": TAU_ZERO_DET, "M_PREDITOR": M},
           "celulas": [], "congeladas": []}
    for d0, T, alvo in CELULAS:
        tau, face = cone.amostra(d0, T, M, int(d0 * 1e4) + int(T * 1e7))
        R = cronogramas(T, M, N_PASSOS, 7 + int(T * 1e6))
        cap = excursao(tau, face, R, T, 11 + int(d0 * 1e5))
        p = float(cap.mean()); se = float(np.sqrt(p * (1 - p) / M))
        tard = (face == 1) & (tau > TAU_ZERO_DET[f"{T:g}"]) & cap
        n_f1 = max(int(((face == 1)).sum()), 1)
        taxa_tardia = float(tard.sum() / n_f1)
        out["celulas"].append({"d0": d0, "T": T, "alvo": alvo, "p_captura": p,
                               "se": se, "censura": float(np.mean(face == -1)),
                               "taxa_tardia_face1": taxa_tardia})
        print(f"  T={T} d0={d0:.3f}: p={p:.3f}±{se:.3f}  tardias/face1={taxa_tardia:.3f}")
        if abs(alvo - 0.5) < 1e-9:               # braco B-congelado
            Rf = cronogramas(T, M, N_PASSOS, 13 + int(T * 1e6), freeze=T_FREEZE)
            capf = excursao(tau, face, Rf, T, 17 + int(d0 * 1e5))
            pf = float(capf.mean())
            out["congeladas"].append({"d0": d0, "T": T, "p_captura": pf,
                                      "se": float(np.sqrt(pf * (1 - pf) / M))})
            print(f"    congelada(t_freeze={T_FREEZE}): p={pf:.3f}")
    with open("config_v3.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("congelado em config_v3.json")
