"""Wrapper de ARTEFATOS: reexecuta as tres campanhas de sec:forcing-numerics que
so tinham saida em stdout e grava jsonl versionavel. NAO altera a logica de
parecer14.py / beta.py / parecer15_o5.py -- apenas as invoca e serializa.

  python gera_artefatos.py unit    -> unit_readout_m.jsonl
  python gera_artefatos.py beta    -> beta_sweep.jsonl
  python gera_artefatos.py prehit  -> o5_prehit.jsonl
"""
from pathlib import Path
import json, os, sys
import numpy as np

AQUI = str(Path(__file__).resolve().parent / '.')
sys.path.insert(0, AQUI)
import verifica_forcing as V
import parecer14 as P14


def _dump(nome, linhas):
    p = os.path.join(AQUI, nome)
    with open(p, "w") as f:
        for r in linhas:
            f.write(json.dumps(r) + "\n")
    print(f"[artefato] {p}: {len(linhas)} linhas", flush=True)


# ------------------------------------------------ (a) familia readout unitario
def unit(n_seeds=900, n_full=25, H=1.8, ms=(4, 8, 16)):
    from multiprocessing import Pool
    P14.patch_readout_um()
    V.T_FIM = H
    linhas = []
    for m in ms:
        args = [(V.SEED0 + 900000 + i, m) for i in range(n_seeds)]
        with Pool(6) as pool:
            out = pool.map(P14._screen, args, chunksize=16)
            cand = [a[0] for a, o in zip(args, out) if o == "OK"][:n_full]
            n_screen_ok = sum(1 for o in out if o == "OK")
            print(f"m={m}: peneira {n_screen_ok}/{n_seeds} OK; "
                  f"{len(cand)} avaliadas integralmente", flush=True)
            regs = pool.map(V.uma, [(s, m, V.DT0) for s in cand], chunksize=1)
        for r in regs:
            X = np.array(r["X"]); P0 = np.array(r["P0"])
            r = dict(r)
            r["m"] = m
            r["H"] = H
            r["familia"] = "unit-readout"
            r["n_screen_ok"] = n_screen_ok
            r["n_screen_total"] = n_seeds
            if r.get("passou"):
                r["lam_max_GB"] = P14.lam_max_GB(X, P0, V.readout(m))
            linhas.append(r)
        ok = [r for r in regs if r["passou"]]
        if ok:
            e = np.array([r["erro_rel"] for r in ok])
            print(f"  m={m}: {len(ok)}/{len(regs)} passam | mediana "
                  f"{np.median(e)*100:.2f}% | max {e.max()*100:.2f}%", flush=True)
    _dump("unit_readout_m.jsonl", linhas)


# ------------------------------------------------------------- (b) sweep em beta
def beta():
    from multiprocessing import Pool
    rs = [json.loads(l) for l in open(os.path.join(AQUI, "forcing_m.jsonl"))]
    rs = [r for r in rs if r["passou"] and r["m"] == 8]
    rs = sorted(rs, key=lambda r: -r["erro_rel"])[:6]

    linhas = []
    with Pool(6) as p:
        for b in (1e4, 3e4):
            out = p.map(_job_top, [(r, b) for r in rs])
            for r, med in zip(rs, out):
                pr = r["predito"]
                linhas.append({
                    "seed": r["seed"], "m": r["m"], "beta": b,
                    "R_exit": r["R_exit"], "predito": pr, "medido": med,
                    "erro_rel": (abs(med - pr) / abs(pr)) if med is not None else None,
                })
                if med is None:
                    print(f"  beta={b:.0e} seed {r['seed']}: sem release", flush=True)
                else:
                    print(f"  beta={b:.0e} seed {r['seed']} pred={pr:.2f} "
                          f"med={med:.2f} erro={abs(med-pr)/abs(pr)*100:.2f}%", flush=True)
    _dump("beta_sweep.jsonl", linhas)


def _job_top(a):
    r, b = a
    X = np.array(r["X"]); Y = np.array(r["Y"]); P0 = np.array(r["P0"])
    tr, nm = V.roda_acoplado(X, Y, P0, b)
    if tr is None or nm != 1:
        return None
    return float(b * (tr - r["t_exit"]))


# -------------------------------------------------------- (c) gap pre-hit, RK4
def prehit():
    from parecer15_o5 import run
    recs = [json.loads(l) for l in open(os.path.join(AQUI, "forcing_m.jsonl"))]
    ok = [r for r in recs if r["passou"]]
    betas = [1e3, 2e3, 4e3, 8e3]
    linhas = []
    for m in (4, 8, 16):
        for r in [x for x in ok if x["m"] == m][:4]:
            o = run(r["seed"], m, betas)
            if o is None:
                continue
            c = r["c"]; v = c * abs(r["R_hit"])
            for b, rB, rA in o["rows"]:
                linhas.append({
                    "seed": r["seed"], "m": m, "beta": b, "t_hit": o["t_hit"],
                    "R_hit": r["R_hit"], "c": c, "v": v,
                    "rho_B": rB, "rho_A": rA,
                    "b2_rhoB": b * b * rB, "b_rhoA": b * rA,
                    "b2_v_rhoB": b * b * rB * v, "b_v_rhoA": b * rA * v,
                })
            b, rB, rA = o["rows"][-1]
            print(f"m={m} seed {r['seed']} v={v:.4f} b2*rhoB={b*b*rB:.3f} "
                  f"b2*v*rhoB={b*b*rB*v:.4f} b*rhoA={b*rA:.4f}", flush=True)
    _dump("o5_prehit.jsonl", linhas)


if __name__ == "__main__":
    for a in sys.argv[1:]:
        {"unit": unit, "beta": beta, "prehit": prehit}[a]()
