"""p5-relu-morte: morte/ressurreicao de neuronio ReLU como contato unilateral.

--smoke : SOMENTE mecanica (gradiente vs dif. finitas, determinismo, calibracao do
          estimador de deriva em processo com deriva conhecida). Nao toca em P-a/b/c.
(sem flag): run confirmatoria completa — rodar apenas APOS commit do pre-registro.

Saida: results/runs.jsonl (uma linha por run), results/analise.json (IQM, CI, vereditos),
figuras PNG se matplotlib disponivel.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

import config as C

NORMAIS = np.vstack([C.X, np.ones_like(C.X)]).T          # linha j = (x_j, 1)


# ---------------------------------------------------------------- dinamica
def grad_relu(p):
    z = p[0] * C.X + p[1]
    res = (np.maximum(z, 0.0) - C.Y) * (z > 0)           # relu'(0)=0
    return np.array([np.mean(res * C.X), np.mean(res)])


def grad_softplus(p, beta):
    z = p[0] * C.X + p[1]
    f = np.logaddexp(0.0, beta * z) / beta
    bz = beta * z                                        # sigmoide estavel nos dois ramos
    sig = np.where(bz >= 0, 1.0 / (1.0 + np.exp(-np.abs(bz))),
                   np.exp(-np.abs(bz)) / (1.0 + np.exp(-np.abs(bz))))
    res = (f - C.Y) * sig                                # (f-y)*sigma(beta z)
    return np.array([np.mean(res * C.X), np.mean(res)])


def grad_nulo(p):
    return np.zeros(2)


def morto(p):
    return np.all(p[0] * C.X + p[1] <= 0.0)


def roda(seed, T, gradf, n_passos):
    """Trajetoria SGLD. Ruido depende SO de (seed, T): braços ficam pareados por seed."""
    rng = np.random.default_rng(seed)
    ruido = rng.standard_normal((n_passos, 2)) * np.sqrt(2 * C.ETA * T)
    traj = np.empty((n_passos + 1, 2))
    traj[0] = C.P0
    p = C.P0.copy()
    for t in range(n_passos):
        p = p - C.ETA * gradf(p) + ruido[t]
        traj[t + 1] = p
    return traj


# ---------------------------------------------------------------- metricas
def metricas(traj):
    dead = np.array([morto(p) for p in traj])
    fora = np.nonzero(~dead)[0]
    out = {"censurada": len(fora) == 0}
    if out["censurada"]:
        return out
    t_res = int(fora[0])
    z_res = traj[t_res][0] * C.X + traj[t_res][1]
    face = int(np.argmax(z_res))
    out.update(t_res=t_res, face=face,
               delta_teo=float(C.DELTA_TEO[face]), salto_prot=float(C.SALTO_PROT[face]))

    passos = np.diff(traj, axis=0)                        # passo t: traj[t+1]-traj[t]
    # deriva antes (janela toda dentro de D por definicao de primeira saida)
    a0 = max(0, t_res - C.K_EVENTO)
    v_ant = passos[a0:t_res].mean(axis=0) / C.ETA if t_res > 0 else np.zeros(2)
    # deriva depois: so passos dados de estados com active set EXATAMENTE {face}
    j1 = min(len(passos), t_res + C.K_EVENTO)
    def ativo_so_face(t):
        z = traj[t][0] * C.X + traj[t][1]
        return z[face] > 0 and all(z[i] <= 0 for i in range(len(C.X)) if i != face)
    validos = [t for t in range(t_res, j1) if ativo_so_face(t)]
    out["janela_valida"] = len(validos) >= C.MIN_PASSOS_VALIDOS_JANELA
    if out["janela_valida"]:
        v_dep = passos[validos].mean(axis=0) / C.ETA
        out["salto"] = float(np.linalg.norm(v_dep - v_ant))
        out["v_ant_norm"] = float(np.linalg.norm(v_ant))
    # catraca: fracao de tempo morto no horizonte pos-ressurreicao
    h1 = min(len(dead), t_res + C.HORIZONTE_POS)
    out["f_pos"] = float(dead[t_res:h1].mean())
    # instrumento P-b(i): deriva media na fase morta (verdade exata = 0)
    idx_dead = np.nonzero(dead[:-1])[0]
    if len(idx_dead) > 10:
        out["deriva_morta"] = float(np.linalg.norm(passos[idx_dead].mean(axis=0) / C.ETA))
        out["n_passos_mortos"] = int(len(idx_dead))
    return out


# ---------------------------------------------------------------- estatistica
def iqm(v):
    v = np.sort(np.asarray(v, float))
    k = len(v) // 4
    return float(v[k:len(v) - k].mean()) if len(v) > 3 else float(v.mean())


def boot_ci(v, stat=iqm):
    rng = np.random.default_rng(C.SEED_BOOTSTRAP)
    v = np.asarray(v, float)
    reps = [stat(v[rng.integers(0, len(v), len(v))]) for _ in range(C.N_BOOTSTRAP)]
    return [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]


# ---------------------------------------------------------------- smoke
def smoke():
    print("== SMOKE: mecanica e instrumento, sem estatistica de P-a/b/c ==")
    rng = np.random.default_rng(7)
    # 1) gradientes vs diferencas finitas (fora das quinas)
    def loss_relu(p):
        z = p[0] * C.X + p[1]
        return 0.5 * np.mean((np.maximum(z, 0) - C.Y) ** 2)
    def loss_sp(p, b):
        z = p[0] * C.X + p[1]
        return 0.5 * np.mean((np.logaddexp(0, b * z) / b - C.Y) ** 2)
    h, err = 1e-6, 0.0
    for _ in range(500):
        p = rng.uniform(-2, 2, 2)
        if np.any(np.abs(p[0] * C.X + p[1]) < 1e-3):
            continue
        fd = np.array([(loss_relu(p + h * e) - loss_relu(p - h * e)) / (2 * h) for e in np.eye(2)])
        err = max(err, np.abs(fd - grad_relu(p)).max())
        for b in C.BETAS:
            fd = np.array([(loss_sp(p + h * e, b) - loss_sp(p - h * e, b)) / (2 * h) for e in np.eye(2)])
            err = max(err, np.abs(fd - grad_softplus(p, b)).max())
    assert err < 1e-5, err
    print(f"[OK] gradientes relu+softplus vs dif. finitas: err_max={err:.1e}")
    # 2) determinismo
    h1 = hashlib.sha256(roda(0, C.T_PRIMARIA, grad_relu, 2000).tobytes()).hexdigest()
    h2 = hashlib.sha256(roda(0, C.T_PRIMARIA, grad_relu, 2000).tobytes()).hexdigest()
    assert h1 == h2
    print(f"[OK] determinismo (seed 0, 2000 passos): {h1[:16]}...")
    # 3) calibracao do estimador de deriva: processo com deriva constante conhecida
    c_true = np.array([0.3, -0.7])
    ests = []
    for s in range(200):
        r = np.random.default_rng(1000 + s)
        passos = C.ETA * c_true + r.standard_normal((C.K_EVENTO, 2)) * np.sqrt(2 * C.ETA * C.T_PRIMARIA)
        ests.append(passos.mean(axis=0) / C.ETA)
    ests = np.array(ests)
    vies = np.abs(ests.mean(axis=0) - c_true).max()
    sd_teo = np.sqrt(2 * C.T_PRIMARIA / (C.ETA * C.K_EVENTO))
    sd_obs = ests.std(axis=0).mean()
    assert vies < 3 * sd_teo / np.sqrt(200)
    print(f"[OK] estimador de deriva calibrado: vies={vies:.2e}, sd_obs={sd_obs:.3f} vs sd_teo={sd_teo:.3f}")
    # 4) trajetoria curta sem NaN, comeca morta
    tr = roda(1, C.T_PRIMARIA, grad_relu, 2000)
    assert np.isfinite(tr).all() and morto(tr[0])
    print("[OK] trajetoria 2000 passos finita; init dentro do cone de morte")
    print("== SMOKE OK — nada de P-a/b/c foi computado ==")


# ---------------------------------------------------------------- confirmatoria
def confirmatoria():
    os.makedirs("results", exist_ok=True)
    runs = []
    bracos = [("relu", None, C.T_PRIMARIA, "confirmatorio"),
              ("nulo", None, C.T_PRIMARIA, "confirmatorio")]
    bracos += [("softplus", b, C.T_PRIMARIA, "confirmatorio") for b in C.BETAS]
    bracos += [("relu", None, T, "exploratorio") for T in C.TEMPERATURAS_EXPLORATORIAS]
    with open("results/runs.jsonl", "w") as fh:
        for nome, beta, T, papel in bracos:
            gf = {"relu": grad_relu, "nulo": grad_nulo}.get(nome) or (lambda p: grad_softplus(p, beta))
            for seed in C.SEEDS:
                m = metricas(roda(seed, T, gf, C.N_PASSOS))
                rec = {"braco": nome, "beta": beta, "T": T, "papel": papel, "seed": seed, **m}
                runs.append(rec)
                fh.write(json.dumps(rec) + "\n")
            print(f"braco {nome}{'' if beta is None else f'(b={beta})'} T={T}: ok")

    def sel(**kw):
        return [r for r in runs if all(r.get(k) == v for k, v in kw.items())]

    A = {}
    relu = sel(braco="relu", T=C.T_PRIMARIA)
    nulo = sel(braco="nulo", T=C.T_PRIMARIA)
    cens = np.mean([r["censurada"] for r in relu])
    A["frac_censura_relu"] = cens
    A["braco_primario_valido"] = bool(cens <= C.MAX_FRAC_CENSURA)

    ok = [r for r in relu if not r["censurada"] and r.get("janela_valida")]
    # P-b(i) instrumento: por seed, ||deriva_morta|| <= 3*sqrt(2T/(eta*n_mortos))
    viol = [r for r in relu if "deriva_morta" in r and
            r["deriva_morta"] > C.FATOR_INSTRUMENTO *
            np.sqrt(2 * C.T_PRIMARIA / (C.ETA * r["n_passos_mortos"]))]
    A["instrumento"] = {"seeds_violando": len(viol),
                        "valido": len(viol) <= 2}
    # P-a: razao medido/alvo-protocolar por evento
    salto_rel = [r["salto"] / r["salto_prot"] for r in ok]
    sp2 = [r["salto"] for r in sel(braco="softplus", beta=C.BETAS[0], T=C.T_PRIMARIA)
           if not r["censurada"] and r.get("janela_valida")]
    if salto_rel:
        A["Pa"] = {"salto_rel_iqm": iqm(salto_rel), "ci": boot_ci(salto_rel),
                   "n_eventos": len(salto_rel),
                   "ratio_vs_softplus2": (iqm([r["salto"] for r in ok]) / iqm(sp2)) if sp2 else None}
        dentro_tol = abs(A["Pa"]["salto_rel_iqm"] - 1.0) <= C.TOL_SALTO_REL
        razao_ok = A["Pa"]["ratio_vs_softplus2"] is not None and \
            A["Pa"]["ratio_vs_softplus2"] >= C.RATIO_SALTO_MIN
        A["Pa"]["veredito"] = "PASSA" if (dentro_tol and razao_ok) else "FALHA"
    # P-b
    f_s = [r["f_pos"] for r in relu if not r["censurada"]]
    f_n = [r["f_pos"] for r in nulo if not r["censurada"]]
    if f_s and f_n:
        A["Pb"] = {"f_sgld_iqm": iqm(f_s), "ci_sgld": boot_ci(f_s),
                   "f_nulo_iqm": iqm(f_n), "ci_nulo": boot_ci(f_n)}
        A["Pb"]["veredito"] = "PASSA" if iqm(f_s) < iqm(f_n) / C.FATOR_RATCHET else "FALHA"
    # P-c
    t_relu = iqm([r["t_res"] for r in relu if not r["censurada"]])
    curva = {}
    for b in C.BETAS:
        ts = [r["t_res"] for r in sel(braco="softplus", beta=b, T=C.T_PRIMARIA) if not r["censurada"]]
        curva[b] = iqm(ts) if ts else None
    A["Pc"] = {"t_relu": t_relu, "t_softplus": curva}
    vals = [curva[b] for b in C.BETAS if curva[b] is not None]
    mono = all(a < b for a, b in zip(vals, vals[1:]))
    conv = curva[C.BETAS[-1]] is not None and \
        abs(curva[C.BETAS[-1]] - t_relu) / t_relu <= C.TOL_RIGIDEZ_REL
    A["Pc"]["veredito"] = "PASSA" if (mono and conv) else "FALHA"

    with open("results/analise.json", "w") as fh:
        json.dump(A, fh, indent=1)
    print(json.dumps(A, indent=1))
    figuras(runs)
    return A


def figuras(runs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib ausente — figuras puladas)")
        return
    tr = roda(C.SEEDS[0], C.T_PRIMARIA, grad_relu, C.N_PASSOS)
    fig, ax = plt.subplots(figsize=(6, 5))
    ww = np.linspace(-1.2, 1.2, 2)
    for xj in C.X:
        ax.plot(ww, -xj * ww, "k--", lw=0.8)
    sub = tr[:: max(1, len(tr) // 20000)]
    ax.plot(sub[:, 0], sub[:, 1], lw=0.4, alpha=0.8)
    ax.plot(*C.P0, "ro", ms=5)
    ax.set_xlabel("w"); ax.set_ylabel("b")
    ax.set_title("p5-relu-morte: trajetoria (seed 0) e cone de morte")
    fig.savefig("results/trajetoria-seed0.png", dpi=150, bbox_inches="tight")
    print("figura: results/trajetoria-seed0.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    else:
        confirmatoria()
