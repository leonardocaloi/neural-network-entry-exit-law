"""P7 — varredura de Lambda em m (aberto do parecer 18).

Mede, por instancia aceita (mesmo filtro (H1')-(H4') + guarda de sinal do
degrau2_readout.hibrido), na janela [t_hit, t_exit]:
  Lambda_ck   = sup_t max_k ||c_k(t)||          (a cota do lema)
  Lambda_meas = sup_t max(0, -lambda_min(Hess)) (curvatura negativa realizada)
  lam_max_GB  = sup_t lambda_max(J^T J / n)     (o L_B do paper-v4)
Predicao pre-registrada (parecer 18, C3/P7): o primeiro par e PLANO em m, o
terceiro e LINEAR em m.

Motor: copia fiel do laco de degrau2_readout.hibrido (mesma familia sorteia(),
mesmo Euler explicito), com a unica adicao de lambda_max(G_B).
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verifica_forcing import (FACE, T_FIM, DELTA1, T_MIN_MORTE, sorteia)

AQUI = os.path.dirname(os.path.abspath(__file__))
DT0 = 2e-4                 # mesmo dt do piloto parecer18_m2
HESS_EVERY = 25
SEED0 = 77000001


def init_P(m, seed):
    Xs, Ys, P0 = sorteia(seed, m)
    P = np.zeros((m + 1, 3))
    P[:, :2] = P0
    P[0, 2] = 1.0
    P[1:, 2] = 1.0 / np.sqrt(m)
    return Xs, Ys, P


def medir(X, Y, P0, dt0=DT0):
    n = len(X)
    P = P0.copy()
    t = 0.0
    if P[0, 0] * X[FACE] + P[0, 1] <= 0:
        return None
    sinalB0 = np.sign(P[1:, 0][:, None] * X[None, :] + P[1:, 1][:, None])
    sgn_v0 = np.sign(P[:, 2])
    while True:
        Z = (P[:, 0] * X[:, None]).T + P[:, 1][:, None]
        if Z[0, FACE] <= 0:
            break
        act = (Z > 0).astype(float)
        r = (P[:, 2:3] * np.maximum(Z, 0.0)).sum(axis=0) - Y
        gw = (P[:, 2:3] * act * (r * X)).mean(axis=1)
        gb = (P[:, 2:3] * act * r).mean(axis=1)
        gv = (act * Z * r).mean(axis=1)
        P -= dt0 * np.stack([gw, gb, gv], axis=1)
        t += dt0
        if t > T_FIM:
            return None
    t_hit = t
    if t_hit <= T_MIN_MORTE:
        return None
    zA_hit = P[0, 0] * X + P[0, 1]
    if not np.all(np.delete(zA_hit, FACE) <= -DELTA1):
        return None

    PB = P[1:].copy()
    m = len(PB)
    nt = int((T_FIM - t_hit) / dt0) + 1
    R_hyb = np.empty(nt)
    troca_pat, k_troca, k_troca_v = False, -1, -1
    nb = nt // HESS_EVERY + 2
    lam_c = np.zeros(nb); lam_h = np.zeros(nb); lam_g = np.zeros(nb)
    t_lam = np.zeros(nb, dtype=int)
    nlam = 0
    for k in range(nt):
        Z = (PB[:, 0] * X[:, None]).T + PB[:, 1][:, None]
        act = (Z > 0).astype(float)
        fB = (PB[:, 2:3] * np.maximum(Z, 0.0)).sum(axis=0)
        R_hyb[k] = Y[FACE] - fB[FACE]
        r = fB - Y
        if not troca_pat and not np.all(np.sign(Z) == sinalB0):
            troca_pat, k_troca = True, k
        if k_troca_v < 0 and np.any(np.sign(PB[:, 2]) != sgn_v0[1:]):
            k_troca_v = k
        if k % HESS_EVERY == 0:
            ck = np.stack([(act * (r * X)).mean(axis=1),
                           (act * r).mean(axis=1)], axis=1)
            J = np.zeros((n, 3 * m))
            for kk in range(m):
                J[:, 3 * kk] = PB[kk, 2] * act[kk] * X
                J[:, 3 * kk + 1] = PB[kk, 2] * act[kk]
                J[:, 3 * kk + 2] = act[kk] * Z[kk]
            G = J.T @ J / n
            Hs = G.copy()
            for kk in range(m):
                Hs[3 * kk + 2, 3 * kk] += ck[kk, 0]
                Hs[3 * kk, 3 * kk + 2] += ck[kk, 0]
                Hs[3 * kk + 2, 3 * kk + 1] += ck[kk, 1]
                Hs[3 * kk + 1, 3 * kk + 2] += ck[kk, 1]
            lam_c[nlam] = float(np.linalg.norm(ck, axis=1).max())
            lam_h[nlam] = max(0.0, -float(np.linalg.eigvalsh(Hs)[0]))
            lam_g[nlam] = float(np.linalg.eigvalsh(G)[-1])
            t_lam[nlam] = k
            nlam += 1
        gw = (PB[:, 2:3] * act * (r * X)).mean(axis=1)
        gb = (PB[:, 2:3] * act * r).mean(axis=1)
        gv = (act * Z * r).mean(axis=1)
        PB -= dt0 * np.stack([gw, gb, gv], axis=1)
    A_hyb = np.cumsum(R_hyb) * dt0
    if R_hyb[0] >= 0:
        return None
    pos = np.nonzero(A_hyb > 0)[0]
    if len(pos) == 0:
        return None
    k_exit = int(pos[0])
    t_exit = t_hit + k_exit * dt0
    if t_exit >= T_FIM or not np.all(A_hyb[:k_exit] < 0):
        return None
    R_exit = float(R_hyb[k_exit])
    if R_exit <= 0:
        return None
    if int(np.count_nonzero(np.diff(np.sign(R_hyb[:k_exit + 1])) != 0)) != 1:
        return None
    if troca_pat and k_troca <= k_exit:
        return None
    if 0 <= k_troca_v <= k_exit:
        return None
    sel = t_lam[:nlam] <= k_exit
    if not sel.any():
        return None
    return {"m": m, "t_hit": t_hit, "t_exit": t_exit, "T_ep": t_exit - t_hit,
            "R_exit": R_exit, "n_amostras": int(sel.sum()),
            "Lambda_ck": float(lam_c[:nlam][sel].max()),
            "Lambda_meas": float(lam_h[:nlam][sel].max()),
            "lam_max_GB": float(lam_g[:nlam][sel].max())}


def one(a):
    m, seed = a
    try:
        X, Y, P0 = init_P(m, seed)
    except RuntimeError:
        return None
    r = medir(X, Y, P0)
    if r is not None:
        r["seed"] = int(seed)
    return r


def main():
    ms = [int(x) for x in sys.argv[1].split(",")]
    n_try = int(sys.argv[2])
    alvo = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    saida = os.path.join(AQUI, "p7_lambda_m.jsonl")
    t0 = time.time()
    todos = []
    with Pool(2) as p:
        for m in ms:
            res = []
            for lote in range(0, n_try, 60):
                args = [(m, SEED0 + 100000 * m + i) for i in range(lote, min(lote + 60, n_try))]
                res += [r for r in p.map(one, args, chunksize=2) if r]
                print(f"m={m}: {len(res)} aceitas de {min(lote+60,n_try)} tentadas "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
                if len(res) >= alvo:
                    break
            todos += res[:alvo]
    with open(saida, "w") as fh:
        for r in todos:
            fh.write(json.dumps(r) + "\n")
    print("jsonl:", saida)


if __name__ == "__main__":
    main()
