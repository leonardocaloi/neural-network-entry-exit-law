"""Preditor v4: modelo 4.0 — p_A em 2D (BM no cone; fora, empurroes das faces
ativas) acoplado aos modos de B. Sem parametro livre. NAO roda o 4D softplus.
Diferencas p/ o sistema completo (declaradas): B linear-modal (exato sob H4),
gate relu (beta->inf) e captura pela regua operacional congelada do v2/v3."""
import json

import numpy as np

X = np.array([0.5, 1.5]); Y = np.array([0.5, 1.0]); N = 2
ETA = 1e-3; ETA_B = 1e-4
B_INIT = np.array([0.2, 0.05]); N_PASSOS = 500_000
NORMA = np.sqrt(X ** 2 + 1); DEPTH2APEX = np.sqrt(1 + X.max() ** 2)
FRAC_CAPTURA, SIG_MIN_CAPTURA, W_LOCAL = 0.5, 4.0, 70_000
XB = np.vstack([X, np.ones(N)])                    # (2,N)
G = np.mean([np.outer([x, 1.0], [x, 1.0]) for x in X], axis=0)
lam, V = np.linalg.eigh(G)
P_STAR = np.linalg.solve(np.array([[X[0], 1], [X[1], 1]]), Y)

with open('../relu-death-v2/config_v2.json') as fh:
    CFG2 = json.load(fh)
CELULAS = [(c["d0"], c["T"], c["alvo"]) for c in CFG2["celulas"]]

def celula_40(d0, T, m, seed, coarse=100):
    rng = np.random.default_rng(seed)
    sig = np.sqrt(2 * ETA * T)
    sigx = np.sqrt(N * T) / NORMA
    a_mode = (1.0 - ETA_B * lam) ** coarse
    a2 = (1.0 - ETA_B * lam) ** 2
    sd_blk = np.sqrt(2 * ETA_B * T * (1 - a2 ** coarse) / (1 - a2))
    pA = np.tile([0.0, -d0 * DEPTH2APEX], (m, 1))
    C = np.tile((B_INIT - P_STAR) @ V, (m, 1))
    cap = np.zeros(m, bool); ativo = np.ones(m, bool)
    saiu_uma_vez = np.zeros(m, bool); t_saida = np.full(m, N_PASSOS, np.int64)
    t = 0
    while ativo.any() and t < N_PASSOS:
        ids = np.nonzero(ativo)[0]
        zB = (C[ids] @ V.T + P_STAR) @ XB          # (n,N) margens de B
        res = Y[None, :] - zB                       # residuo excluindo A
        acc_A = np.zeros((len(ids), 2))
        for s in range(coarse):
            zA = pA[ids] @ XB                       # (n,N)
            atv = zA > 0
            # empurrao das faces ativas: -(1/n)(relu(zA)-res)*1[atv] em cada dado
            resid_tot = np.where(atv, zA - res, 0.0)
            gA = (resid_tot[:, None, :] * XB[None, :, :]).sum(2) / N
            pA[ids] = pA[ids] - ETA * gA + rng.standard_normal((len(ids), 2)) * sig
            acc_A += np.maximum(zA, 0.0) @ XB.T / N # contribuicao p/ arrasto de B
            # captura operacional (mesma regua): face = argmax zA
            j = np.argmax(zA, axis=1)
            x_n = np.take_along_axis(zA, j[:, None], 1)[:, 0] / NORMA[j]
            xst = np.maximum(np.take_along_axis(res, j[:, None], 1)[:, 0], 0) / NORMA[j]
            pega = (x_n >= FRAC_CAPTURA * xst) & (xst >= SIG_MIN_CAPTURA * sigx[j]) \
                   & atv[np.arange(len(ids)), j]
            cap[ids] |= pega
            fora = atv.any(1)
            novo = fora & ~saiu_uma_vez[ids]
            t_saida[ids[novo]] = t + s
            saiu_uma_vez[ids] |= fora
        # B: OU exato por bloco + arrasto medio da contribuicao de A
        C[ids] = (C[ids] * a_mode[None, :]
                  + rng.standard_normal((len(ids), 2)) * sd_blk[None, :]
                  - ETA_B * (acc_A @ V))
        t += coarse
        lim = np.where(saiu_uma_vez[ids], t - t_saida[ids], 0)
        ativo[ids] = ~(cap[ids] | (lim >= W_LOCAL))
    cens = ~saiu_uma_vez
    return float(cap.mean()), float(cens.mean())

def _tarefa(args):
    kind, d0, T, alvo = args
    if kind == "celula":
        p, cz = celula_40(d0, T, 6000, int(d0 * 1e4) + int(T * 1e7))
        return (kind, d0, T, alvo, p, cz)
    p20, _ = celula_40(d0, T, 6000, 99 + int(d0 * 1e4), coarse=20)
    return (kind, d0, T, alvo, p20, None)

if __name__ == "__main__":
    from multiprocessing import Pool
    M = 6000
    out = {"modelo": "4.0-2D-acoplado", "constantes": {**CFG2["constantes"],
           "SEEDS_V4": [1500, 1649], "M_PREDITOR": M}, "celulas": [],
           "controle_coarse": []}
    tarefas = [("celula", d0, T, a) for d0, T, a in CELULAS] + \
              [("controle", d0, T, a) for d0, T, a in CELULAS if T == 0.003]
    with Pool(6) as pool:                        # 6 de 8 nucleos, nice herdado
        for kind, d0, T, alvo, p, cz in pool.imap_unordered(_tarefa, tarefas):
            if kind == "celula":
                out["celulas"].append({"d0": d0, "T": T, "alvo": alvo,
                                       "p_captura": p,
                                       "se": float(np.sqrt(p * (1 - p) / M)),
                                       "censura": cz})
                print(f"  T={T} d0={d0:.3f}: p={p:.3f} cens={cz:.3f}", flush=True)
            else:
                out["controle_coarse"].append({"d0": d0, "T": T, "p_coarse20": p})
                print(f"  controle T={T} d0={d0:.3f}: p(COARSE=20)={p:.3f}", flush=True)
    out["celulas"].sort(key=lambda c: (c["T"], -c["d0"]))
    with open("config_v4.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("congelado em config_v4.json")
