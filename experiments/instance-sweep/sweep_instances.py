"""SWEEP DE INSTANCIAS do teorema T2' (lei de entrada-saida).

Pergunta: a verificacao de uma instancia (a de referencia do paper,
X=[0.5,1.5], Y=[0.5,1.0], PA0=(-0.5,0.27), PB0=(1.144,0.261)) e uma
coincidencia ou a lei vale como LEI numa familia de instancias?

Predicao testada (eq. release do paper, coeficiente de primeira ordem):

    beta * (t_release - t_exit^hyb)  ->  -1 / (c * R^hyb(t_exit))

com c = (x_face^2 + 1)/n. R_exit VARIA por instancia, logo a predicao varia
por instancia: nao ha nenhum ajuste livre.

Motor: copiado VERBATIM de theory/verify_step4.py
(referencia hibrida + roda_acoplado), so parametrizado pela instancia.
Nenhum parametro numerico e ajustado por instancia: dt0=1e-5,
dt = min(1e-5, 0.02/beta), beta=1e4, T_FIM=1.8, FACE=0 (os do passo 4).

Filtro (H1)-(H4) testado programaticamente na trajetoria hibrida:
  (H1) morte transversal: z_A(x_face) cruza 0 em t_hit in (0.05, T_FIM) e
       R^hyb(t_hit) < 0.
  (H2) A^hyb tem primeiro zero em t_exit < H, com R^hyb(t_exit) > 0 e
       A^hyb < 0 em (t_hit, t_exit); R^hyb troca de sinal UMA vez na janela.
  (H3) as outras margens de A ficam <= -delta1 (delta1 = 0.05) na janela.
  (H4) nenhum outro evento de chaveamento em [0,H]: os sinais de z_B(x_i)
       nao mudam na janela e z_A(x_face) nao cruza 0 antes de t_hit.
Instancia que falha qualquer item conta como NAO-passou, com o motivo gravado.
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np

# ------------------------------------------------------------------ constantes
FACE = 0
T_FIM = 1.8          # = H, horizonte
DT0 = 1e-5           # passo da referencia hibrida (passo 4)
BETA = 1e4           # beta do paper
DELTA1 = 0.05        # delta_1 de (H3)
N_INST = 600
SEED0 = 20260726
AQUI = os.path.dirname(os.path.abspath(__file__))


def softplus(z, b):
    return np.logaddexp(0.0, b * z) / b


def sig(z, b):
    bz = np.clip(b * z, -500, 500)
    return 1.0 / (1.0 + np.exp(-bz))


# ------------------------------------------------------- sorteio das instancias
def sorteia(seed):
    """(X, Y, theta(0)) aleatorios. Faixas escolhidas UMA vez, iguais p/ todos."""
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        cand = _sorteia1(rng)
        X, _, PA0, _ = cand
        zA0 = PA0[0] * X + PA0[1]
        # rejeicao a priori (mesma regra p/ todas as seeds): o evento de morte
        # so e definivel se A nasce VIVO na face e ja morto nas outras (H3).
        if zA0[FACE] > 0 and np.all(np.delete(zA0, FACE) <= -DELTA1):
            return cand
    raise RuntimeError("rejeicao nao convergiu")


def _sorteia1(rng):
    x0 = rng.uniform(0.2, 1.0)
    x1 = x0 + rng.uniform(0.4, 1.4)
    X = np.array([x0, x1])
    Y = np.array([rng.uniform(0.2, 1.2), rng.uniform(0.4, 1.6)])
    # A e parametrizado pela margem inicial na face (z0 pequena e positiva: a
    # morte tem de caber no horizonte H) e pela slope negativa.
    wA = rng.uniform(-1.2, -0.2)
    z0 = rng.uniform(0.01, 0.15)
    bA = z0 - wA * x0
    wB = rng.uniform(0.4, 1.8)
    bB = rng.uniform(0.05, 0.6)
    return X, Y, np.array([wA, bA]), np.array([wB, bB])


# ---------------------------------------- referencia hibrida (VERBATIM passo 4)
def hibrido(X, Y, PA0, PB0):
    """relu acoplado ate t_hit, depois B-sozinho. Devolve tudo p/ o filtro."""
    dt0 = DT0
    p = np.concatenate([PA0, PB0])
    t = 0.0
    zA0 = PA0[0] * X + PA0[1]
    if zA0[FACE] <= 0:
        return {"ok": False, "motivo": "A ja nasce morto na face"}
    sinalB0 = np.sign(PB0[0] * X + PB0[1])
    while True:                                   # fase A-ativa (relu)
        zA, zB = p[0] * X + p[1], p[2] * X + p[3]
        if zA[FACE] <= 0:
            break
        r = np.maximum(zA, 0) + np.maximum(zB, 0) - Y
        aA, aB = (zA > 0), (zB > 0)
        g = np.concatenate([[np.mean(r * aA * X), np.mean(r * aA)],
                            [np.mean(r * aB * X), np.mean(r * aB)]])
        p -= dt0 * g
        t += dt0
        if t > T_FIM:
            return {"ok": False, "motivo": "sem morte no horizonte"}
    t_hit = t
    if t_hit <= 0.05:
        return {"ok": False, "motivo": "morte antes de t=0.05 (fora da janela)"}
    zA_hit = p[0] * X + p[1]
    # (H3): as outras margens de A (congeladas) tem de ficar <= -delta1
    outras = np.delete(zA_hit, FACE)
    if not np.all(outras <= -DELTA1):
        return {"ok": False, "motivo": "H3 falha: outra margem de A > -delta1"}

    pB = p[2:].copy()
    nt = int((T_FIM - t_hit) / dt0) + 1
    R_hyb = np.empty(nt)
    sinalB = np.empty((nt, len(X)))
    for k in range(nt):                           # fase morta: B-sozinho
        zB = pB[0] * X + pB[1]
        R_hyb[k] = Y[FACE] - max(zB[FACE], 0.0)
        sinalB[k] = np.sign(zB)
        r = np.maximum(zB, 0) - Y
        a = (zB > 0)
        pB -= dt0 * np.array([np.mean(r * a * X), np.mean(r * a)])
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
    # troca de sinal UNICA de R na janela (t_hit, t_exit)
    trocas = int(np.count_nonzero(np.diff(np.sign(R_hyb[:k_exit + 1])) != 0))
    if trocas != 1:
        return {"ok": False, "motivo": f"H2 falha: R troca de sinal {trocas}x na janela"}
    # (H4): sinais de z_B constantes ate o exit
    if not np.all(sinalB[:k_exit + 1] == sinalB0):
        return {"ok": False, "motivo": "H4 falha: z_B troca de sinal na janela"}
    return {"ok": True, "t_hit": t_hit, "t_exit": t_exit, "R_hit": float(R_hyb[0]),
            "R_exit": R_exit}


# -------------------------------------- 4D acoplado softplus (VERBATIM passo 4)
def roda_acoplado(X, Y, PA0, PB0, b):
    dt = min(1e-5, 0.02 / b)
    p = np.concatenate([PA0, PB0])
    t = 0.0
    morto = False
    n_mortes = 0
    while t < T_FIM:
        zA, zB = p[0] * X + p[1], p[2] * X + p[3]
        r = softplus(zA, b) + softplus(zB, b) - Y
        sA, sB = sig(zA, b), sig(zB, b)
        g = np.concatenate([[np.mean(r * sA * X), np.mean(r * sA)],
                            [np.mean(r * sB * X), np.mean(r * sB)]])
        p -= dt * g
        t += dt
        z0 = p[0] * X[FACE] + p[1]
        if not morto and z0 <= 0 and t > 0.05:
            morto = True
            n_mortes += 1
        elif morto and z0 >= 0:
            return t, n_mortes
    return None, n_mortes


# ------------------------------------------------------------------ uma seed
def uma(seed):
    X, Y, PA0, PB0 = sorteia(seed)
    reg = {"seed": int(seed), "X": X.tolist(), "Y": Y.tolist(),
           "PA0": PA0.tolist(), "PB0": PB0.tolist(),
           "c": float((X[FACE] ** 2 + 1) / len(X))}
    h = hibrido(X, Y, PA0, PB0)
    if not h["ok"]:
        reg.update({"passou_H1H2": False, "motivo": h["motivo"],
                    "R_exit": None, "predito": None, "medido": None,
                    "erro_rel": None})
        return reg
    reg.update({"t_hit": h["t_hit"], "t_exit": h["t_exit"],
                "R_hit": h["R_hit"], "R_exit": h["R_exit"]})
    tr, n_mortes = roda_acoplado(X, Y, PA0, PB0, BETA)
    predito = -1.0 / (reg["c"] * h["R_exit"])
    if tr is None:
        reg.update({"passou_H1H2": False,
                    "motivo": "beta-run sem release no horizonte (H1H2 valem no hibrido)",
                    "predito": predito, "medido": None, "erro_rel": None,
                    "n_mortes": n_mortes})
        return reg
    if n_mortes != 1:
        reg.update({"passou_H1H2": False, "motivo": f"{n_mortes} mortes no beta-run",
                    "predito": predito, "medido": None, "erro_rel": None,
                    "n_mortes": n_mortes})
        return reg
    medido = BETA * (tr - h["t_exit"])
    reg.update({"passou_H1H2": True, "motivo": "", "n_mortes": n_mortes,
                "t_release": tr, "beta": BETA,
                "predito": predito, "medido": float(medido),
                "erro_rel": float(abs(medido - predito) / abs(predito))})
    return reg


def main():
    seeds = list(range(SEED0, SEED0 + N_INST))
    t0 = time.time()
    with Pool(12) as pool:
        regs = pool.map(uma, seeds, chunksize=1)
    dur = time.time() - t0

    saida = os.path.join(AQUI, "sweep_instances.jsonl")
    with open(saida, "w") as fh:
        for r in regs:
            fh.write(json.dumps(r) + "\n")

    ok = [r for r in regs if r["passou_H1H2"]]
    err = np.array([r["erro_rel"] for r in ok]) if ok else np.array([])
    print(f"\n=== SWEEP T2' — {N_INST} instancias, beta={BETA:.0e}, "
          f"dt0={DT0}, H={T_FIM}, {dur/60:.1f} min ===")
    print(f"sorteadas .................. {len(regs)}")
    print(f"passaram (H1)-(H4) ......... {len(ok)}  "
          f"({100*len(ok)/len(regs):.1f}%)")
    motivos = {}
    for r in regs:
        if not r["passou_H1H2"]:
            motivos[r["motivo"]] = motivos.get(r["motivo"], 0) + 1
    print("motivos de exclusao:")
    for m, k in sorted(motivos.items(), key=lambda kv: -kv[1]):
        print(f"   {k:4d}  {m}")
    if len(err):
        print(f"\nerro relativo |medido - predito|/|predito|  (predito = -1/(c*R_exit)):")
        print(f"   mediana .... {np.median(err)*100:.2f}%")
        print(f"   p90 ........ {np.percentile(err, 90)*100:.2f}%")
        print(f"   max ........ {err.max()*100:.2f}%")
        print(f"   R_exit range .. [{min(r['R_exit'] for r in ok):.4f}, "
              f"{max(r['R_exit'] for r in ok):.4f}]")
        print(f"   predito range . [{min(r['predito'] for r in ok):.2f}, "
              f"{max(r['predito'] for r in ok):.2f}]")
    print(f"\njsonl: {saida}")


if __name__ == "__main__":
    main()
