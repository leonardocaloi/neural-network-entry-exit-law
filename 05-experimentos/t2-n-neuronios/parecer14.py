"""Parecer 14 - testes adversariais sobre verifica_forcing.py (NAO edita nada do projeto)."""
from pathlib import Path
import json, os, sys, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent / '.'))
import verifica_forcing as V

AQUI = "/tmp/claude-1000/-home-caloi-Projects-j-space/552bb7fa-f3f0-42f8-8086-cb79faf68c7a/scratchpad"

# ---------------------------------------------------------------- L_B por instancia
def lam_max_GB(X, P0, a):
    """Gram do subsistema vivo em espaco de parametros (2m x 2m), padrao ativo em t=0."""
    PB = P0[1:]
    Z = (PB[:, 0] * X[:, None]).T + PB[:, 1][:, None]   # (m, n)
    act = (Z > 0).astype(float)
    n = len(X)
    feats = []
    for k in range(PB.shape[0]):
        feats.append(a[k+1] * act[k] * X)
        feats.append(a[k+1] * act[k])
    F = np.array(feats)          # (2m, n)
    G = F @ F.T / n
    return float(np.linalg.eigvalsh(G).max())


def carrega():
    regs = [json.loads(l) for l in open(
        str(Path(__file__).resolve().parent / 'forcing_m.jsonl'))]
    return [r for r in regs if r["passou"]]


# ------------------------------------------------- TESTE A: replica exata (clone)
def replica(X, Y, P0, p):
    """Cada viva k -> p copias com readout a_k/sqrt(p) e params (w,b)/sqrt(p).
    f_B identico, NTK identico, fluxo identico => R(t) identico. m' = p*m."""
    m = len(P0) - 1
    a_old = V.readout(m)
    mp = p * m
    a_new = V.readout(mp)          # a[0]=1, a[k]=1/sqrt(mp)
    P = np.empty((mp + 1, 2))
    P[0] = P0[0]
    idx = 1
    for k in range(m):
        # queremos a_new * ReLU(w' x + b') somado em p copias == a_old[k+1]*ReLU(w x+b)
        # com w' = s*w, b' = s*b:  p * a_new * s = a_old  =>  s = a_old/(p*a_new)
        s = a_old[k+1] / (p * a_new[1])
        for _ in range(p):
            P[idx] = s * P0[k+1]
            idx += 1
    return P


def teste_A():
    ok = carrega()
    base = [r for r in ok if r["m"] == 4]
    print(f"\n=== TESTE A - replica exata (R e NTK congelados, so m muda): "
          f"{len(base)} instancias m=4 -> m=8 e m=16")
    linhas = []
    for r in base:
        X = np.array(r["X"]); Y = np.array(r["Y"]); P0 = np.array(r["P0"])
        row = {"seed": r["seed"], "R_exit": r["R_exit"], "e4": r["erro_rel"],
               "lam4": lam_max_GB(X, P0, V.readout(4))}
        for p, tag in ((2, 8), (4, 16)):
            Pp = replica(X, Y, P0, p)
            h = V.hibrido(X, Y, Pp, V.DT0)
            if not h["ok"]:
                row[f"e{tag}"] = None; row[f"mot{tag}"] = h["motivo"]; continue
            c = float((X[V.FACE]**2 + 1) / len(X))
            pred = -1.0 / (c * h["R_exit"])
            tr, nm = V.roda_acoplado(X, Y, Pp, V.BETA)
            if tr is None or nm != 1:
                row[f"e{tag}"] = None; row[f"mot{tag}"] = f"beta-run nm={nm}"; continue
            med = V.BETA * (tr - h["t_exit"])
            row[f"e{tag}"] = abs(med - pred) / abs(pred)
            row[f"Rex{tag}"] = h["R_exit"]
            row[f"lam{tag}"] = lam_max_GB(X, Pp, V.readout(tag))
        linhas.append(row)
        print(f"  seed {row['seed']}  R_exit={row['R_exit']:.4f}  "
              f"m=4: {row['e4']*100:.2f}%  "
              f"m=8: {(row.get('e8') or float('nan'))*100:.2f}%  "
              f"m=16: {(row.get('e16') or float('nan'))*100:.2f}%  "
              f"lam_max(G_B) {row['lam4']:.3f}/{row.get('lam8',float('nan')):.3f}/"
              f"{row.get('lam16',float('nan')):.3f}", flush=True)
    json.dump(linhas, open(os.path.join(AQUI, "testeA.json"), "w"), indent=1)
    return linhas


# --------------------------------------- TESTE B: readout +1, horizonte x4
def patch_readout_um():
    V.readout = lambda m: np.ones(m + 1)


def teste_B(n_seeds=900, Hs=(1.8, 7.2)):
    print(f"\n=== TESTE B - familia readout +1 (a paper eq.1), m=4, peneira dt={V.DT_SCREEN}")
    patch_readout_um()
    from multiprocessing import Pool
    for H in Hs:
        V.T_FIM = H
        args = [(V.SEED0 + 900000 + i, 4) for i in range(n_seeds)]
        with Pool(6) as pool:
            out = pool.map(_screen, args, chunksize=16)
        mot = {}
        for m_ in out:
            mot[m_] = mot.get(m_, 0) + 1
        nok = mot.pop("OK", 0)
        print(f"  H={H}: {nok}/{n_seeds} passam (H1')-(H4') na peneira")
        for k, v in sorted(mot.items(), key=lambda kv: -kv[1])[:6]:
            print(f"      {v:4d}  {k}")
    return


def _screen(arg):
    seed, m = arg
    try:
        X, Y, P0 = V.sorteia(seed, m)
    except RuntimeError:
        return "rejeicao falhou"
    h = V.hibrido(X, Y, P0, V.DT_SCREEN)
    return "OK" if h["ok"] else h["motivo"]




def teste_B_full(n_seeds=900, n_full=25, H=1.8, m=4):
    """Avaliacao COMPLETA (dt0=1e-5 + beta-run) da familia readout +1, m=4."""
    from multiprocessing import Pool
    patch_readout_um()
    V.T_FIM = H
    args = [(V.SEED0 + 900000 + i, m) for i in range(n_seeds)]
    with Pool(6) as pool:
        out = pool.map(_screen, args, chunksize=16)
        cand = [a[0] for a, o in zip(args, out) if o == "OK"][:n_full]
        print(f"\n=== TESTE B-full readout+1 m={m} H={H}: {len(cand)} avaliadas integralmente")
        regs = pool.map(V.uma, [(s, m, V.DT0) for s in cand], chunksize=1)
    ok = [r for r in regs if r["passou"]]
    mot = {}
    for r in regs:
        if not r["passou"]:
            mot[r["motivo"]] = mot.get(r["motivo"], 0) + 1
    print(f"  {len(ok)}/{len(regs)} passaram (H1')-(H4') em dt0")
    for k, v in sorted(mot.items(), key=lambda kv: -kv[1]):
        print(f"      {v:3d}  {k}")
    if ok:
        e = np.array([r["erro_rel"] for r in ok])
        print(f"  erro rel: mediana {np.median(e)*100:.2f}% | p90 "
              f"{np.percentile(e,90)*100:.2f}% | max {e.max()*100:.2f}%  (n={len(ok)})")
        for r in ok:
            X = np.array(r["X"]); P0 = np.array(r["P0"])
            lam = lam_max_GB(X, P0, V.readout(m))
            print(f"    seed {r['seed']}: R_exit={r['R_exit']:.4f} "
                  f"Tep*beta={(r['t_exit']-r['t_hit'])*V.BETA:.0f} "
                  f"lam_max(G_B)={lam:.3f} erro={r['erro_rel']*100:.2f}%")
    json.dump(regs, open(os.path.join(AQUI, f"testeB_full_m{m}_H{H}.json"), "w"))
    return regs


if __name__ == "__main__":
    if "A" in sys.argv:
        teste_A()
    if "BF" in sys.argv:
        teste_B_full()
    if "BF8" in sys.argv:
        teste_B_full(m=8); teste_B_full(m=16)
    if "B" in sys.argv:
        teste_B()
