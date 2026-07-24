"""p5-relu-morte-v2: dinamica 4D real vs modelo reduzido de corrida.

--smoke : mecanica apenas (gradiente vs dif. finitas, determinismo, init morto,
          leitura da config congelada). NAO produz estatistica de P-1/2/3.
(sem flag): run confirmatoria — SOMENTE apos commit do pre-registro.

Grade, predicoes e constantes vem CONGELADAS de config_v2.json (gerado do
preditor-output.json no fechamento do pre-registro). Nada de parametro novo aqui.
"""
import argparse
import hashlib
import json
import os

import numpy as np

X = np.array([0.5, 1.5])
Y = np.array([0.5, 1.0])
N = len(X)
ETA = 1e-3               # eta_A
ETA_B = 1e-4             # eta_B (ver README: Langevin pre-condicionado diagonal)
ETAS = np.array([1e-3, 1e-3, 1e-4, 1e-4])
B_INIT = np.array([0.2, 0.05])
NORMA = np.sqrt(X ** 2 + 1)
DEPTH2APEX = np.sqrt(1 + X.max() ** 2)


def carrega_config():
    with open(os.path.join(os.path.dirname(__file__), "config_v2.json")) as fh:
        return json.load(fh)


# ------------------------------------------------------------------ dinamica 4D
def grad_relu(p):
    zA, zB = p[0] * X + p[1], p[2] * X + p[3]
    r = np.maximum(zA, 0) + np.maximum(zB, 0) - Y
    aA, aB = (zA > 0), (zB > 0)
    return np.array([np.mean(r * aA * X), np.mean(r * aA),
                     np.mean(r * aB * X), np.mean(r * aB)])


def grad_softplus(p, beta):
    zA, zB = p[0] * X + p[1], p[2] * X + p[3]
    def sp(z):
        return np.logaddexp(0.0, beta * z) / beta
    def sig(z):
        bz = beta * z
        return np.where(bz >= 0, 1 / (1 + np.exp(-np.abs(bz))),
                        np.exp(-np.abs(bz)) / (1 + np.exp(-np.abs(bz))))
    r = sp(zA) + sp(zB) - Y
    return np.array([np.mean(r * sig(zA) * X), np.mean(r * sig(zA)),
                     np.mean(r * sig(zB) * X), np.mean(r * sig(zB))])


def grad_nulo(p):
    return np.zeros(4)


def roda(seed, T, d0, gradf, n_passos):
    """Ruido depende so de (seed): bracos pareados por seed. Trajetoria 4D completa."""
    rng = np.random.default_rng(seed)
    ruido = rng.standard_normal((n_passos, 4)) * np.sqrt(2 * ETAS * T)
    p = np.array([0.0, -d0 * DEPTH2APEX, *B_INIT])
    traj = np.empty((n_passos + 1, 4))
    traj[0] = p
    for t in range(n_passos):
        p = p - ETAS * gradf(p) + ruido[t]
        traj[t + 1] = p
    return traj


def grad_relu_lote(P):
    """Gradiente vetorizado sobre lote de estados P (S,4)."""
    zA = P[:, 0:1] * X[None, :] + P[:, 1:2]
    zB = P[:, 2:3] * X[None, :] + P[:, 3:4]
    r = np.maximum(zA, 0) + np.maximum(zB, 0) - Y[None, :]
    aA, aB = zA > 0, zB > 0
    return np.stack([np.mean(r * aA * X, axis=1), np.mean(r * aA, axis=1),
                     np.mean(r * aB * X, axis=1), np.mean(r * aB, axis=1)], axis=1)


def grad_nulo_lote(P):
    return np.zeros_like(P)


def roda_lote(seeds, T, d0, gradf_lote, n_passos, bloco=20_000):
    """Lote de seeds em lockstep. Streams de ruido por seed identicos aos de roda()
    (Generator PCG64 e sequencial: draws em blocos concatenam igual a um draw unico).
    Retorna trajetorias (S, n+1, 4)."""
    gens = [np.random.default_rng(s) for s in seeds]
    S = len(seeds)
    P = np.tile([0.0, -d0 * DEPTH2APEX, *B_INIT], (S, 1))
    traj = np.empty((S, n_passos + 1, 4))
    traj[:, 0] = P
    esc = np.sqrt(2 * ETAS * T)
    t = 0
    while t < n_passos:
        k = min(bloco, n_passos - t)
        ruido = np.stack([g.standard_normal((k, 4)) for g in gens]) * esc
        for i in range(k):
            P = P - ETAS * gradf_lote(P) + ruido[:, i]
            traj[:, t + i + 1] = P
        t += k
    return traj


# ------------------------------------------------------------------ medicoes
def metricas(traj, T, cfg):
    """Espelha as definicoes operacionais do README/preditor (mesma regua)."""
    zA = traj[:, 0:1] * X[None, :] + traj[:, 1:2]          # (t, N)
    dead = np.all(zA <= 0, axis=1)
    fora = np.nonzero(~dead)[0]
    out = {"censurada": len(fora) == 0}
    if out["censurada"]:
        return out
    t1 = int(fora[0])
    out["tau1"] = t1
    out["face1"] = int(np.argmax(zA[t1]))

    zB = traj[:, 2:3] * X[None, :] + traj[:, 3:4]
    sigx = np.sqrt(N * T) / NORMA                          # sd estacionaria por face
    W_P3 = cfg["constantes"]["W_P3"]

    capturado, t_cap = False, None
    excedeu, re_morte_p3 = False, False
    face_corrente = out["face1"]
    # horizonte de busca IDENTICO ao preditor (w_local): captura tardia alem de
    # t1+W_LOCAL nao conta em nenhum dos lados (correcao M1 do parecer)
    fim_busca = min(len(traj), t1 + cfg["constantes"]["W_LOCAL"])
    for t in range(t1, fim_busca):
        if dead[t]:
            if excedeu and (t - t1) < W_P3 and not capturado:
                re_morte_p3 = True
            excedeu = False
            continue
        j = int(np.argmax(zA[t]))
        face_corrente = j
        x = zA[t, j] / NORMA[j]                            # coordenada normal (fora)
        res_menos_A = max(zB[t, j], 0) - Y[j]              # residuo excluindo A
        xstar = max(-res_menos_A, 0.0) / NORMA[j]
        if x > cfg["constantes"]["SIG_REDEATH"] * sigx[j]:
            excedeu = True
        if (not capturado and x >= cfg["constantes"]["FRAC_CAPTURA"] * xstar
                and xstar >= cfg["constantes"]["SIG_MIN_CAPTURA"] * sigx[j]):
            capturado, t_cap = True, t
            break                                          # capturado: desfecho final
    out.update(capturada=capturado, t_captura=t_cap,
               re_morte_p3=bool(re_morte_p3), face_final=face_corrente)
    return out


# ------------------------------------------------------------------ smoke
def smoke():
    print("== SMOKE v2: mecanica apenas ==")
    cfg = carrega_config()
    print(f"[OK] config congelada: {len(cfg['celulas'])} celulas, "
          f"constantes {cfg['constantes']}")
    rng = np.random.default_rng(11)
    def loss(p):
        return 0.5 * np.mean((np.maximum(p[0] * X + p[1], 0)
                              + np.maximum(p[2] * X + p[3], 0) - Y) ** 2)
    h, err = 1e-6, 0.0
    for _ in range(500):
        p = rng.uniform(-2, 2, 4)
        zs = np.concatenate([p[0] * X + p[1], p[2] * X + p[3]])
        if np.any(np.abs(zs) < 1e-3):
            continue
        fd = np.array([(loss(p + h * e) - loss(p - h * e)) / (2 * h) for e in np.eye(4)])
        err = max(err, np.abs(fd - grad_relu(p)).max())
    assert err < 1e-6, err
    print(f"[OK] gradiente 4D vs dif. finitas: err_max={err:.1e}")
    c0 = cfg["celulas"][0]
    t1 = roda(0, c0["T"], c0["d0"], grad_relu, 2000)
    t2 = roda(0, c0["T"], c0["d0"], grad_relu, 2000)
    assert hashlib.sha256(t1.tobytes()).hexdigest() == hashlib.sha256(t2.tobytes()).hexdigest()
    zA0 = t1[0, 0] * X + t1[0, 1]
    assert np.all(zA0 < 0), zA0
    print("[OK] determinismo; init de A dentro do cone de morte; B init "
          f"{t1[0, 2:]} (vivo: {np.all(t1[0, 2] * X + t1[0, 3] > 0)})")
    print("== SMOKE v2 OK — nada de P-1/2/3 computado ==")


# ------------------------------------------------------------------ confirmatoria
def confirmatoria():
    cfg = carrega_config()
    n_passos = cfg["constantes"]["N_PASSOS"]
    seeds = list(range(cfg["constantes"]["N_SEEDS_CELULA"]))
    os.makedirs("results", exist_ok=True)
    runs = []
    with open("results/runs.jsonl", "w") as fh:
        for cel in cfg["celulas"]:
            for braco, gfl in (("relu", grad_relu_lote), ("nulo", grad_nulo_lote)):
                trajs = roda_lote(seeds, cel["T"], cel["d0"], gfl, n_passos)
                for seed, tr in zip(seeds, trajs):
                    m = metricas(tr, cel["T"], cfg)
                    rec = {"braco": braco, "T": cel["T"], "d0": cel["d0"],
                           "alvo": cel["alvo"], "seed": seed, **m}
                    runs.append(rec)
                    fh.write(json.dumps(rec) + "\n")
                del trajs
            print(f"celula T={cel['T']} d0={cel['d0']:.3f}: ok", flush=True)
    analise(runs, cfg)


def analise(runs, cfg):
    A = {"celulas": [], "n_rejeicoes_P1": 0}
    desvios = []
    for cel in cfg["celulas"]:
        # REGUA IGUAL ao preditor (avalia(): p = mean(cap) sobre TODOS os walkers):
        # censurada conta como NAO capturada no denominador; a validade da censura
        # e checada a parte contra MAX_FRAC_CENSURA.
        rs = [r for r in runs if r["braco"] == "relu" and r["T"] == cel["T"]
              and r["d0"] == cel["d0"]]
        n = len(rs)
        if n == 0:
            A["celulas"].append({**cel, "veredito": "sem-dados"})
            continue
        cens = float(np.mean([r["censurada"] for r in rs]))
        k = sum(r.get("capturada", False) for r in rs)
        p_obs, p_pred = k / n, cel["p_captura"]
        se = np.sqrt(p_pred * (1 - p_pred) / n)
        z = (p_obs - p_pred) / se if se > 0 else 0.0
        valida = cens <= cfg["constantes"]["MAX_FRAC_CENSURA"]
        # censura e predicao EXATA (L1-v2: dentro de D a dinamica E BM) — estouro de
        # censura = instrumento invalido, celula fora de P-1 (nao conta rejeicao)
        rejeita = bool(abs(z) > 1.96) and valida
        A["n_rejeicoes_P1"] += int(rejeita)
        if valida:
            desvios.append(np.sign(p_obs - p_pred))
        A["celulas"].append({"T": cel["T"], "d0": cel["d0"], "n": n,
                             "p_obs": p_obs, "p_pred": p_pred, "z": float(z),
                             "rejeita": rejeita, "censura_obs": cens,
                             "celula_valida": valida})
    mesmo_sinal = max(np.sum(np.array(desvios) > 0), np.sum(np.array(desvios) < 0)) \
        if desvios else 0
    n_validas = sum(c.get("celula_valida", False) for c in A["celulas"])
    if n_validas < 7:
        A["P1_veredito"] = "INCONCLUSIVO-INSTRUMENTO"     # <7 de 9 celulas validas
    else:
        A["P1_veredito"] = ("FALHA"
                            if (A["n_rejeicoes_P1"] >= cfg["constantes"]["P1_MIN_REJEICOES"]
                                or mesmo_sinal >= cfg["constantes"]["P1_MIN_MESMO_SINAL"])
                            else "PASSA")
    A["P1_mesmo_sinal"] = int(mesmo_sinal)
    A["P1_celulas_validas"] = int(n_validas)

    # P-2 (redesenho pos-parecer de 2026-07-24): calibracao da CURVA INTEIRA, eventos
    # face-1 de TODAS as celulas e T pooled, binados por q PREDITO (Hosmer-Lemeshow
    # com bins fixos em q), Poisson-binomial por bin + falsificador EXATO de captura
    # tardia (o modelo reduzido preve ZERO capturas com tau1 > tau10(T)).
    q_bins = cfg["constantes"]["P2_Q_BINS"]
    eventos = []                                           # (q_pred, cap, tardia)
    for T_str, pool in cfg["pooled_por_T"].items():
        T = float(T_str)
        rs = [r for r in runs if r["braco"] == "relu" and r["T"] == T
              and not r["censurada"] and r["face1"] == 1]
        edges = np.array(pool["q_edges"])
        qv = np.array(pool["q_vals"])
        for r in rs:
            idx = int(np.clip(np.searchsorted(edges, r["tau1"], side="right") - 1,
                              0, len(qv) - 1))
            # tardia = q_pred EXATAMENTE zero (janela fechada no modelo), NAO tau1>tau10
            # (re-check do parecer: tau10 e cruzamento de q=0.1; havia fatia com
            # q=0.089 acima dele -> E[tardias|modelo]=0.48, falso-MATA ~9%. Com
            # q_pred==0, E=0 exato.) Nota registrada: uma tardia capturada tambem cai
            # no bin [0,0.1) — ramos de OR correlacionados, inofensivo (bin quase
            # sempre pulado por var<1), mas o falso-MATA de P-2 nao e soma limpa.
            eventos.append((float(qv[idx]), bool(r["capturada"]),
                            qv[idx] == 0.0))
    qp = np.array([e[0] for e in eventos])
    cp = np.array([e[1] for e in eventos], float)
    tard = np.array([e[2] for e in eventos])
    det2, rej2 = [], 0
    for a, b in zip(q_bins[:-1], q_bins[1:]):
        m = (qp >= a) & (qp < b) if b < 1 else (qp >= a)
        n = int(m.sum())
        var = float(np.sum(qp[m] * (1 - qp[m])))
        if n >= 10 and var > 1:
            z2 = (cp[m].sum() - qp[m].sum()) / np.sqrt(var)
            r2 = abs(z2) > 1.96
            rej2 += int(r2)
            det2.append({"q_bin": [a, b], "n": n, "cap_obs": float(cp[m].sum()),
                         "cap_pred": float(qp[m].sum()), "z": float(z2),
                         "rejeita": bool(r2)})
        else:
            det2.append({"q_bin": [a, b], "n": n, "var": var, "insuficiente": True})
    n_tardias = int(np.sum(tard & (cp > 0)))
    A["P2"] = {"bins": det2, "n_rejeicoes": rej2, "n_eventos_pooled": len(eventos),
               "capturas_tardias": n_tardias}
    falha_bins = rej2 >= cfg["constantes"]["P2_MIN_BINS_REJEITADOS"]
    falha_tardia = n_tardias >= cfg["constantes"]["P2_MIN_CAPTURAS_TARDIAS"]
    A["P2_veredito"] = "FALHA" if (falha_bins or falha_tardia) else "PASSA"

    # P-3 (rebaixada a EXPLORATORIA pelo parecer: faixa discriminante faminta e o
    # modelo reduzido tem drift espurio em res~0; reportar, sem MATA)
    A["P3_exploratorio"] = []
    for T_str, pool in cfg["pooled_por_T"].items():
        T = float(T_str)
        rs = [r for r in runs if r["braco"] == "relu" and r["T"] == T
              and not r["censurada"] and r["face1"] == 1]
        det3 = []
        for (a, b), fpred3 in zip(pool["faixas"], pool["P3_freq_re_morte"]):
            grupo = [r for r in rs if a <= r["tau1"] < b]
            det3.append({"faixa": [a, b], "n": len(grupo),
                         "f_obs": float(np.mean([r["re_morte_p3"] for r in grupo]))
                         if grupo else None, "f_pred": fpred3})
        A["P3_exploratorio"].append({"T": T, "faixas": det3})

    with open("results/analise.json", "w") as fh:
        json.dump(A, fh, indent=1)
    print(json.dumps(A, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    smoke() if args.smoke else confirmatoria()
