"""P8 — quociente de Rayleigh q(t) = -Delta^T Hess U(theta_mid) Delta / ||Delta||^2.

O passo 3 da prova paga exp(int q dt); o Gronwall cota q <= Lambda_ep com
igualdade so se Delta se alinha ao autovetor minimizante. Aqui medimos q ao
longo do episodio [t_hit, t_exit] e comparamos int q dt com Lambda_ep * T_ep.

Desenho: o beta-run (softplus, tudo treinavel) e a referencia hibrida (relu, A
congelado depois de t_hit) sao integrados EM LOCKSTEP, mesmo dt, mesma
condicao inicial, de t=0. Delta(t) = theta_B^beta(t) - theta_B^hyb(t) no bloco
vivo (3m coordenadas, ordem (w,b,v) por unidade). Hess e avaliado no ponto
medio theta_hyb + Delta/2 com o padrao de ativacao congelado do hibrido (H4'),
que e a aproximacao de 1 ponto da media integrada Hbar = int_0^1 Hess(theta_hyb
+ tau Delta) dtau (exata se Hess for afim em tau; o erro e O(||Delta||^2) na
terceira derivada, e ||Delta|| ~ 1e-4 aqui).

Sementes: as aceitas de trainable_readout.jsonl (m=4).
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_forcing import (FACE, T_FIM, BETA, DELTA1, T_MIN_MORTE,
                              sorteia, softplus, sig)

AQUI = os.path.dirname(os.path.abspath(__file__))
DT = 2e-6
EVERY = 100
M = 4


def init_P(m, seed):
    Xs, Ys, P0 = sorteia(seed, m)
    P = np.zeros((m + 1, 3))
    P[:, :2] = P0
    P[0, 2] = 1.0
    P[1:, 2] = 1.0 / np.sqrt(m)
    return Xs, Ys, P


def hess_liv(PB, X, Y, act, n):
    """Hessiano 3m x 3m de U_B (padrao act congelado), e a parte Gauss-Newton."""
    m = len(PB)
    Z = (PB[:, 0] * X[:, None]).T + PB[:, 1][:, None]
    fB = (PB[:, 2:3] * (act * Z)).sum(axis=0)
    r = fB - Y
    ck = np.stack([(act * (r * X)).mean(axis=1), (act * r).mean(axis=1)], axis=1)
    J = np.zeros((n, 3 * m))
    for kk in range(m):
        J[:, 3 * kk] = PB[kk, 2] * act[kk] * X
        J[:, 3 * kk + 1] = PB[kk, 2] * act[kk]
        J[:, 3 * kk + 2] = act[kk] * Z[kk]
    H = J.T @ J / n
    for kk in range(m):
        H[3 * kk + 2, 3 * kk] += ck[kk, 0]
        H[3 * kk, 3 * kk + 2] += ck[kk, 0]
        H[3 * kk + 2, 3 * kk + 1] += ck[kk, 1]
        H[3 * kk + 1, 3 * kk + 2] += ck[kk, 1]
    return H, ck


def one(seed):
    m = M
    try:
        X, Y, P0 = init_P(m, seed)
    except RuntimeError:
        return None
    n = len(X)
    Ph = P0.copy()          # hibrido (relu)
    Pb = P0.copy()          # beta-run (softplus)
    if Ph[0, 0] * X[FACE] + Ph[0, 1] <= 0:
        return None
    sinalB0 = np.sign(Ph[1:, 0][:, None] * X[None, :] + Ph[1:, 1][:, None])
    t = 0.0
    morto = False
    t_hit = None
    A = 0.0
    Rprev = None
    n_troca_R = 0
    qs, ts, lams, dns = [], [], [], []
    it = 0
    out = {"seed": int(seed), "m": m, "dt": DT}
    while t < T_FIM:
        # ---- hibrido
        Zh = (Ph[:, 0] * X[:, None]).T + Ph[:, 1][:, None]
        acth = (Zh > 0).astype(float)
        if not morto:
            if Zh[0, FACE] <= 0:
                if t <= T_MIN_MORTE:
                    return None
                zA = Ph[0, 0] * X + Ph[0, 1]
                if not np.all(np.delete(zA, FACE) <= -DELTA1):
                    return None
                morto = True
                t_hit = t
                out["t_hit"] = t
                out["v0_hit"] = float(Ph[0, 2])
        fh = (Ph[:, 2:3] * np.maximum(Zh, 0.0)).sum(axis=0)
        rh = fh - Y
        if morto:
            fB = (Ph[1:, 2:3] * np.maximum(Zh[1:], 0.0)).sum(axis=0)
            R = Y[FACE] - fB[FACE]
            A += R * DT
            if Rprev is not None and np.sign(R) != np.sign(Rprev):
                n_troca_R += 1
            Rprev = R
            if not np.all(np.sign(Zh[1:]) == sinalB0):
                return None            # H4' falha na janela
            if A > 0:
                out.update({"t_exit": t, "T_ep": t - t_hit, "R_exit": float(R),
                            "n_troca_R": n_troca_R})
                break
        gw = (Ph[:, 2:3] * acth * (rh * X)).mean(axis=1)
        gb = (Ph[:, 2:3] * acth * rh).mean(axis=1)
        gv = (acth * Zh * rh).mean(axis=1)
        step = np.stack([gw, gb, gv], axis=1)
        if morto:
            step[0] = 0.0                 # A congelado
        Ph -= DT * step
        # ---- beta-run
        Zb = (Pb[:, 0] * X[:, None]).T + Pb[:, 1][:, None]
        sb = sig(Zb, BETA)
        spb = softplus(Zb, BETA)
        rb = (Pb[:, 2:3] * spb).sum(axis=0) - Y
        Pb -= DT * np.stack([(Pb[:, 2:3] * sb * (rb * X)).mean(axis=1),
                             (Pb[:, 2:3] * sb * rb).mean(axis=1),
                             (spb * rb).mean(axis=1)], axis=1)
        # ---- medida
        if morto and it % EVERY == 0:
            D = (Pb[1:] - Ph[1:]).reshape(-1)
            nd = float(np.linalg.norm(D))
            Pmid = Ph[1:] + 0.5 * (Pb[1:] - Ph[1:])
            H, _ = hess_liv(Pmid, X, Y, acth[1:], n)
            Hh, _ = hess_liv(Ph[1:], X, Y, acth[1:], n)
            if nd > 0:
                qs.append(-float(D @ (H @ D)) / nd ** 2)
                ts.append(t)
                dns.append(nd)
                lams.append(max(0.0, -float(np.linalg.eigvalsh(Hh)[0])))
        t += DT
        it += 1
    if "t_exit" not in out or not qs:
        return None
    if n_troca_R != 1:
        return None
    q = np.array(qs); tt = np.array(ts); lam = np.array(lams); dn = np.array(dns)
    out.update({
        "n_amostras": len(q),
        "Lambda_ep": float(lam.max()),
        "q_sup": float(q.max()), "q_min": float(q.min()),
        "q_med": float(np.median(q)),
        "int_q": float(np.trapezoid(q, tt)),
        "int_lam": float(np.trapezoid(lam, tt)),
        "Lam_ep_T_ep": float(lam.max() * (tt[-1] - tt[0])),
        "T_amostrado": float(tt[-1] - tt[0]),
        "dn_ini": float(dn[0]), "dn_fim": float(dn[-1]),
        "log_growth": float(np.log(dn[-1] / dn[0])),
        "q_series": [float(x) for x in q[::max(1, len(q) // 40)]],
    })
    return out


def main():
    seeds = [json.loads(l)["seed"] for l in
             open(os.path.join(AQUI, "trainable_readout.jsonl"))
             if json.loads(l).get("passou")]
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    seeds = seeds[:n]
    t0 = time.time()
    with Pool(2) as p:
        res = [r for r in p.map(one, seeds, chunksize=1) if r]
    with open(os.path.join(AQUI, "p8_rayleigh.jsonl"), "w") as fh:
        for r in res:
            fh.write(json.dumps(r) + "\n")
    print(f"{len(res)}/{len(seeds)} instancias em {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
