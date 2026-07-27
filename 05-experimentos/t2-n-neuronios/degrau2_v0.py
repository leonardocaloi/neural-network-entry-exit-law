"""DEGRAU 2 — TESTE DISCRIMINANTE do fator v_0 (readout do MORTO).

O run de scoping (degrau2_readout.py) deu NULO DE INSTRUMENTO: com v_0(0)=1 os
dois preditores
      naive = -1/(c R_exit)          e      corr = -1/(c v0_hit R_exit)
coincidem em todos os digitos, porque v_0 nao se move (|v0_hit-1| < 1e-5).

Este script muda UMA coisa: v_0(0) in {0.5, 2.0}.  Se v_0 continua congelado no
valor inicial, os dois preditores separam por fator 2 e o teste DECIDE qual
esta certo.  Falsificavel: se err_corr nao for sistematicamente menor que
err_naive, a correcao v_0 do item (a) do Teorema Degrau 2 esta ERRADA.

MAXIMO 2 workers, nice -19.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from verifica_forcing import (FACE, T_FIM, DT0, BETA, sorteia)
from degrau2_readout import hibrido, roda_acoplado, bal

M = 4
N_INST = 500
N_MAX_FULL = 30
SEED0 = 20260728
DT_SCREEN = 4e-4
V0_NIVEIS = (0.5, 2.0)
AQUI = os.path.dirname(os.path.abspath(__file__))


def init_P(m, seed, v0):
    Xs, Ys, P0 = sorteia(seed, m)
    P = np.zeros((m + 1, 3))
    P[:, :2] = P0
    P[0, 2] = v0
    P[1:, 2] = 1.0 / np.sqrt(m)
    return Xs, Ys, P


def uma(arg):
    seed, v0, dt0 = arg
    X, Y, P0 = init_P(M, seed, v0)
    c = float((X[FACE] ** 2 + 1) / len(X))
    reg = {"seed": int(seed), "m": M, "c": c, "v0_init": v0}
    h = hibrido(X, Y, P0, dt0, medir=True)
    if not h["ok"]:
        reg.update({"passou": False, "motivo": h["motivo"]})
        return reg
    reg.update({k: v for k, v in h.items() if k != "ok"})
    reg["pred_naive"] = -1.0 / (c * h["R_exit"])
    reg["pred_corr"] = -1.0 / (c * h["v0_hit"] * h["R_exit"])
    reg["gronwall"] = float(np.exp(h["Lambda_meas"] * h["T_ep"]))
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
    seed, v0 = arg
    try:
        X, Y, P0 = init_P(M, seed, v0)
    except RuntimeError:
        return None
    return seed if hibrido(X, Y, P0, DT_SCREEN)["ok"] else None


def main():
    n_inst = int([a.split("=")[1] for a in sys.argv if a.startswith("--n=")][0]) \
        if any(a.startswith("--n=") for a in sys.argv) else N_INST
    t0 = time.time()
    todos = []
    with Pool(2) as pool:
        for v0 in V0_NIVEIS:
            cand = pool.map(peneira, [(SEED0 + i, v0) for i in range(n_inst)],
                            chunksize=8)
            tar = [(s, v0, DT0) for s in cand if s is not None][:N_MAX_FULL]
            print(f"[v0={v0}] peneira: {len(tar)} candidatas de {n_inst} em "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
            todos += pool.map(uma, tar, chunksize=1)
            print(f"[v0={v0}] pronto em {(time.time()-t0)/60:.1f} min", flush=True)

    saida = os.path.join(AQUI, "degrau2_v0.jsonl")
    with open(saida, "w") as fh:
        for r in todos:
            fh.write(json.dumps(r) + "\n")

    print(f"\n=== TESTE DISCRIMINANTE v_0, m={M}, beta={BETA:.0e}, "
          f"{(time.time()-t0)/60:.1f} min ===")
    for v0 in V0_NIVEIS:
        ok = [r for r in todos if r.get("passou") and r["v0_init"] == v0]
        tot = [r for r in todos if r["v0_init"] == v0]
        print(f"\n--- v0(0) = {v0}: {len(ok)}/{len(tot)} passaram ---")
        mot = {}
        for r in tot:
            if not r.get("passou"):
                mot[r["motivo"]] = mot.get(r["motivo"], 0) + 1
        for k, v in sorted(mot.items(), key=lambda kv: -kv[1]):
            print(f"    {v:3d}  {k}")
        if not ok:
            continue
        vh = np.array([r["v0_hit"] for r in ok])
        print(f"  v0_hit: min {vh.min():.6f} | mediana {np.median(vh):.6f} | "
              f"max {vh.max():.6f}   (|v0_hit-v0(0)| max = "
              f"{np.abs(vh-v0).max():.3e})")
        en = np.array([r["err_naive"] for r in ok])
        ec = np.array([r["err_corr"] for r in ok])
        for nome, e in (("err_naive", en), ("err_corr", ec)):
            print(f"  {nome}: mediana {np.median(e)*100:.2f}% | p90 "
                  f"{np.percentile(e,90)*100:.2f}% | max {e.max()*100:.2f}%")
        gan = int(np.count_nonzero(ec < en))
        print(f"  VEREDITO PAREADO: err_corr < err_naive em {gan}/{len(ok)} "
              f"instancias; razao mediana err_corr/err_naive = "
              f"{np.median(ec/np.maximum(en,1e-300)):.4f}")
        # 'Lambda_meas' = max(0,-lambda_min(Hess)); NAO e o L_B=lambda_max(G_B)
        # do paper-v4 (parecer 18, B3).
        for nome in ("T_ep", "Lambda_meas", "Lambda_ck", "gronwall", "R_exit",
                     "v_min_live"):
            a = np.array([r[nome] for r in ok], dtype=float)
            print(f"  {nome:12s} min {a.min():.4g} | mediana {np.median(a):.4g} "
                  f"| max {a.max():.4g}")
    print(f"\njsonl: {saida}")


if __name__ == "__main__":
    main()
