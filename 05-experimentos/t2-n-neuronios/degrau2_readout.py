"""DEGRAU 2 — o teorema de forcing com READOUT TREINAVEL (scoping numerico).

Difere de verifica_forcing.py em UMA coisa: os readouts v_k (das m vivas E do
morto A) sao TREINADOS junto com (w_k, b_k).  Isso tira o subsistema vivo da
classe AFIM do Teorema 6 (o campo vivo passa a ser bilinear em (v, (w,b))), que
e exatamente o item (O6) do apendice de abertos.

O que este script mede, por instancia:
  1. a lei: beta*(t_release - t_exit) contra o predito.  DOIS preditos:
       (a) NAIVE   = -1/(c * R_exit)              [a formula do paper, v_0 == 1]
       (b) CORRIGIDO = -1/(c * v0_hit * R_exit)   [derivacao com v_0 treinavel]
     Se (b) bate e (a) nao, a lei sobrevive com o readout do MORTO entrando como
     fator; c -> c*v_0.
  2. Lambda := sup_t max_k ||c_k(t)||, com c_k = (1/n) sum_i r_i 1_{k,i} (x_i,1):
     a cota analitica para a curvatura NEGATIVA do bloco vivo (ver nota).
  3. LmaxJ := sup_t max(0, -lambda_min(Hess_B U)) medido por autovalores exatos
     do Hessiano 3m x 3m do campo vivo na trajetoria hibrida — o L_B do (O6).
  4. T_ep = t_exit - t_hit, e o fator de Gronwall exp(LmaxJ * T_ep).
  5. conservacao de B_k = v_k^2 - (w_k^2 + b_k^2) (balancedness), no hibrido
     (relu, deve ser EXATA) e no beta-run (softplus, deve driftar O(e^-beta d)).
  6. guarda de sinal: rejeita instancia em que algum v_k (vivo ou morto) troca
     de sinal na janela.

Motor: mesmo integrador Euler explicito de verifica_forcing.py; mesma familia de
sorteio (sorteia() importado), mesmos dt/beta/H.  MAXIMO 2 workers.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from verifica_forcing import (FACE, T_FIM, DT0, BETA, DELTA1, T_MIN_MORTE,
                              sorteia, softplus, sig)

M = 4
N_INST = 400
N_MAX_FULL = 50
SEED0 = 20260727
DT_SCREEN = 4e-4
AQUI = os.path.dirname(os.path.abspath(__file__))
HESS_EVERY = 500          # amostragem do Hessiano na fase morta do hibrido


def init_P(X, Y, m, seed):
    """(m+1) x 3: colunas (w, b, v).  Mesma familia de verifica_forcing, com os
    readouts agora sendo ESTADO: v_0 = 1 (morto A), v_k = 1/sqrt(m) (vivas)."""
    Xs, Ys, P0 = sorteia(seed, m)
    P = np.zeros((m + 1, 3))
    P[:, :2] = P0
    P[0, 2] = 1.0
    P[1:, 2] = 1.0 / np.sqrt(m)
    return Xs, Ys, P


def bal(P):
    return P[:, 2] ** 2 - (P[:, 0] ** 2 + P[:, 1] ** 2)


def hibrido(X, Y, P0, dt0=DT0, medir=False):
    """Referencia hibrida com readout treinavel: fase A-viva (relu, tudo junto),
    depois A CONGELADO (w,b,v) e as vivas seguem sozinhas."""
    n = len(X)
    P = P0.copy()
    t = 0.0
    if P[0, 0] * X[FACE] + P[0, 1] <= 0:
        return {"ok": False, "motivo": "A ja nasce morto na face"}
    sinalB0 = np.sign(P[1:, 0][:, None] * X[None, :] + P[1:, 1][:, None])
    sgn_v0 = np.sign(P[:, 2])
    B0 = bal(P)
    while True:
        Z = (P[:, 0] * X[:, None]).T + P[:, 1][:, None]      # (K,n)
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
            return {"ok": False, "motivo": "sem morte no horizonte"}
    t_hit = t
    if t_hit <= T_MIN_MORTE:
        return {"ok": False, "motivo": "morte antes de t_min (fora da janela)"}
    zA_hit = P[0, 0] * X + P[0, 1]
    if not np.all(np.delete(zA_hit, FACE) <= -DELTA1):
        return {"ok": False, "motivo": "H3 falha: outra margem de A > -delta1"}
    v0_hit = float(P[0, 2])

    PB = P[1:].copy()
    m = len(PB)
    nt = int((T_FIM - t_hit) / dt0) + 1
    R_hyb = np.empty(nt)
    troca_pat = False
    k_troca = -1
    lam_c = np.zeros(nt // HESS_EVERY + 2)      # max_k ||c_k||
    lam_h = np.zeros(nt // HESS_EVERY + 2)      # max(0, -lambda_min(Hess))
    t_lam = np.zeros(nt // HESS_EVERY + 2, dtype=int)
    nlam = 0
    v_min = np.inf
    k_troca_v = -1
    vmin_hist = np.empty(nt)
    for k in range(nt):
        Z = (PB[:, 0] * X[:, None]).T + PB[:, 1][:, None]    # (m,n)
        act = (Z > 0).astype(float)
        fB = (PB[:, 2:3] * np.maximum(Z, 0.0)).sum(axis=0)
        R_hyb[k] = Y[FACE] - fB[FACE]
        r = fB - Y
        if not troca_pat and not np.all(np.sign(Z) == sinalB0):
            troca_pat, k_troca = True, k
        vmin_hist[k] = np.abs(PB[:, 2]).min()
        if k_troca_v < 0 and np.any(np.sign(PB[:, 2]) != sgn_v0[1:]):
            k_troca_v = k
        if medir and k % HESS_EVERY == 0:
            # c_k = (1/n) sum_i r_i 1_{k,i} (x_i, 1)
            ck = np.stack([(act * (r * X)).mean(axis=1),
                           (act * r).mean(axis=1)], axis=1)          # (m,2)
            lam_c[nlam] = float(np.linalg.norm(ck, axis=1).max())
            # Hessiano exato do bloco vivo (padrao congelado), ordem (w,b,v) por unidade
            J = np.zeros((n, 3 * m))
            for kk in range(m):
                J[:, 3 * kk] = PB[kk, 2] * act[kk] * X
                J[:, 3 * kk + 1] = PB[kk, 2] * act[kk]
                J[:, 3 * kk + 2] = act[kk] * Z[kk]
            Hs = J.T @ J / n
            for kk in range(m):
                Hs[3 * kk + 2, 3 * kk] += ck[kk, 0]
                Hs[3 * kk, 3 * kk + 2] += ck[kk, 0]
                Hs[3 * kk + 2, 3 * kk + 1] += ck[kk, 1]
                Hs[3 * kk + 1, 3 * kk + 2] += ck[kk, 1]
                Hs[3 * kk + 2, 3 * kk + 2] += 0.0
            lam_h[nlam] = max(0.0, -float(np.linalg.eigvalsh(Hs)[0]))
            t_lam[nlam] = k
            nlam += 1
        gw = (PB[:, 2:3] * act * (r * X)).mean(axis=1)
        gb = (PB[:, 2:3] * act * r).mean(axis=1)
        gv = (act * Z * r).mean(axis=1)
        PB -= dt0 * np.stack([gw, gb, gv], axis=1)
    A_hyb = np.cumsum(R_hyb) * dt0

    if R_hyb[0] >= 0:
        return {"ok": False, "motivo": "H1 falha: R_hyb(t_hit) >= 0"}
    pos = np.nonzero(A_hyb > 0)[0]
    if len(pos) == 0:
        return {"ok": False, "motivo": "H2 falha: area nao volta a zero em [t_hit,H]"}
    k_exit = int(pos[0])
    t_exit = t_hit + k_exit * dt0
    if t_exit >= T_FIM:
        return {"ok": False, "motivo": "H2 falha: t_exit >= H"}
    if not np.all(A_hyb[:k_exit] < 0):
        return {"ok": False, "motivo": "H2 falha: area nao e < 0 em (t_hit,t_exit)"}
    R_exit = float(R_hyb[k_exit])
    if R_exit <= 0:
        return {"ok": False, "motivo": "H2 falha: R_hyb(t_exit) <= 0"}
    trocas = int(np.count_nonzero(np.diff(np.sign(R_hyb[:k_exit + 1])) != 0))
    if trocas != 1:
        return {"ok": False, "motivo": f"H2 falha: R troca de sinal {trocas}x na janela"}
    if troca_pat and k_troca <= k_exit:
        return {"ok": False, "motivo": "H4 falha: unidade viva chaveia na janela"}
    if 0 <= k_troca_v <= k_exit:
        return {"ok": False, "motivo": "guarda: readout vivo troca de sinal na janela"}
    v_min = float(vmin_hist[:k_exit + 1].min())
    out = {"ok": True, "t_hit": t_hit, "t_exit": t_exit, "v0_hit": v0_hit,
           "R_hit": float(R_hyb[0]), "R_exit": R_exit,
           "T_ep": t_exit - t_hit, "v_min_live": v_min,
           "bal0": B0.tolist(), "bal_hyb_drift": float(np.abs(bal(PB) - B0[1:]).max())}
    if medir:
        sel = t_lam[:nlam] <= k_exit
        out["Lambda_ck"] = float(lam_c[:nlam][sel].max()) if sel.any() else 0.0
        out["L_B"] = float(lam_h[:nlam][sel].max()) if sel.any() else 0.0
        out["Lambda_ck_H"] = float(lam_c[:nlam].max())
        out["L_B_H"] = float(lam_h[:nlam].max())
    return out


def roda_acoplado(X, Y, P0, b):
    """beta-run acoplado, readouts treinados, com contagem de mortes, guarda de
    sinal dos readouts e drift de balancedness."""
    dt = min(1e-5, 0.02 / b)
    P = P0.copy()
    B0 = bal(P)
    sgn0 = np.sign(P[:, 2])
    t = 0.0
    morto = False
    n_mortes = 0
    troca_v = False
    drift = 0.0
    it = 0
    while t < T_FIM:
        Z = (P[:, 0] * X[:, None]).T + P[:, 1][:, None]
        s = sig(Z, b)
        r = (P[:, 2:3] * softplus(Z, b)).sum(axis=0) - Y
        gw = (P[:, 2:3] * s * (r * X)).mean(axis=1)
        gb = (P[:, 2:3] * s * r).mean(axis=1)
        gv = (softplus(Z, b) * r).mean(axis=1)
        P -= dt * np.stack([gw, gb, gv], axis=1)
        t += dt
        it += 1
        if it % 2000 == 0:
            drift = max(drift, float(np.abs(bal(P) - B0).max()))
            if np.any(np.sign(P[:, 2]) != sgn0):
                troca_v = True
        z0 = P[0, 0] * X[FACE] + P[0, 1]
        if not morto and z0 <= 0 and t > T_MIN_MORTE:
            morto = True
            n_mortes += 1
        elif morto and z0 >= 0:
            return t, n_mortes, troca_v, drift, float(P[0, 2])
    return None, n_mortes, troca_v, drift, float(P[0, 2])


def uma(arg):
    seed, m, dt0 = arg
    X, Y, P0 = init_P(None, None, m, seed)
    c = float((X[FACE] ** 2 + 1) / len(X))
    reg = {"seed": int(seed), "m": m, "c": c}
    h = hibrido(X, Y, P0, dt0, medir=True)
    if not h["ok"]:
        reg.update({"passou": False, "motivo": h["motivo"]})
        return reg
    reg.update({k: v for k, v in h.items() if k != "ok"})
    reg["pred_naive"] = -1.0 / (c * h["R_exit"])
    reg["pred_corr"] = -1.0 / (c * h["v0_hit"] * h["R_exit"])
    reg["gronwall"] = float(np.exp(h["L_B"] * h["T_ep"]))
    tr, n_mortes, troca_v, drift, v0_fim = roda_acoplado(X, Y, P0, BETA)
    reg.update({"n_mortes": n_mortes, "bal_beta_drift": drift, "v0_fim": v0_fim})
    if tr is None:
        reg.update({"passou": False, "motivo": "beta-run sem release no horizonte"})
        return reg
    if n_mortes != 1:
        reg.update({"passou": False, "motivo": f"{n_mortes} mortes no beta-run"})
        return reg
    if troca_v:
        reg.update({"passou": False, "motivo": "guarda: readout troca de sinal (beta-run)"})
        return reg
    medido = BETA * (tr - h["t_exit"])
    reg.update({"passou": True, "motivo": "", "t_release": tr, "beta": BETA,
                "medido": float(medido),
                "err_naive": float(abs(medido - reg["pred_naive"]) / abs(reg["pred_naive"])),
                "err_corr": float(abs(medido - reg["pred_corr"]) / abs(reg["pred_corr"]))})
    return reg


def peneira(arg):
    seed, m = arg
    try:
        X, Y, P0 = init_P(None, None, m, seed)
    except RuntimeError:
        return None
    return seed if hibrido(X, Y, P0, DT_SCREEN)["ok"] else None


def main():
    n_inst = int([a.split("=")[1] for a in sys.argv if a.startswith("--n=")][0]) \
        if any(a.startswith("--n=") for a in sys.argv) else N_INST
    t0 = time.time()
    with Pool(2) as pool:
        cand = pool.map(peneira, [(SEED0 + i, M) for i in range(n_inst)], chunksize=8)
        tar = [(s, M, DT0) for s in cand if s is not None][:N_MAX_FULL]
        print(f"peneira: {len(tar)} candidatas de {n_inst} em "
              f"{(time.time()-t0)/60:.1f} min", flush=True)
        regs = pool.map(uma, tar, chunksize=1)
    saida = os.path.join(AQUI, "degrau2_readout.jsonl")
    with open(saida, "w") as fh:
        for r in regs:
            fh.write(json.dumps(r) + "\n")

    ok = [r for r in regs if r.get("passou")]
    print(f"\n=== DEGRAU 2 (readout treinavel), m={M}, beta={BETA:.0e}, "
          f"{(time.time()-t0)/60:.1f} min ===")
    print(f"{len(tar)} avaliadas, {len(ok)} passaram (H1')-(H4') + guarda de sinal")
    mot = {}
    for r in regs:
        if not r.get("passou"):
            mot[r["motivo"]] = mot.get(r["motivo"], 0) + 1
    for k, v in sorted(mot.items(), key=lambda kv: -kv[1]):
        print(f"    {v:3d}  {k}")
    if not ok:
        return
    for nome in ("err_naive", "err_corr"):
        e = np.array([r[nome] for r in ok])
        print(f"  {nome}: mediana {np.median(e)*100:.2f}% | p90 "
              f"{np.percentile(e,90)*100:.2f}% | max {e.max()*100:.2f}%")
    print("  baseline readout FIXO (verifica_forcing m=4): mediana 0.80%")
    for nome in ("v0_hit", "T_ep", "L_B", "Lambda_ck", "gronwall",
                 "bal_hyb_drift", "bal_beta_drift", "R_exit", "v_min_live"):
        a = np.array([r[nome] for r in ok], dtype=float)
        print(f"  {nome:15s} min {a.min():.4g} | mediana {np.median(a):.4g} | max {a.max():.4g}")
    e = np.array([r["err_corr"] for r in ok])
    for nome in ("gronwall", "L_B", "T_ep", "R_exit"):
        a = np.array([r[nome] for r in ok], dtype=float)
        cc = np.corrcoef(np.log(np.abs(a) + 1e-300), np.log(e + 1e-300))[0, 1]
        print(f"  corr(log err_corr, log {nome}) = {cc:+.3f}")
    a = np.array([1.0 / r["R_exit"] for r in ok])
    print(f"  corr(log err_corr, log 1/R_exit) = "
          f"{np.corrcoef(np.log(a), np.log(e))[0,1]:+.3f}")
    print(f"\njsonl: {saida}")


if __name__ == "__main__":
    main()
