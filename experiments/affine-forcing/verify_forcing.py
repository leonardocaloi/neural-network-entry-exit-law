"""DEGRAU 1 — verificacao numerica do TEOREMA DE FORCING (T2' com m vivos + 1 morto).

Pergunta: a lei de area / release do T2' depende de "o resto da rede e UM neuronio",
ou so do residuo agregado R(t)?  Se so de R, entao trocar o subsistema vivo por
m in {4,8,16} unidades softplus nao pode mudar a predicao

    beta * (t_release - t_exit^hyb)  ->  -1 / (c * R^hyb(t_exit)),
    c = (x_face^2 + 1)/n,

que NAO tem parametro livre: R_exit varia por instancia e e medido na referencia
hibrida (nao no beta-run).

Baseline a bater: sweep 2-neuronios (experiments/instance-sweep),
600 instancias, 57 passaram (H1)-(H4), erro relativo mediana 2.68% / p90 7.26% /
max 12.01%.

Motor: mesmo integrador canonico do passo4/sweep — Euler explicito,
dt0 = 1e-5 na referencia hibrida (relu), dt = min(1e-5, 0.02/beta) = 2e-6 no
acoplado, beta = 1e4, H = 1.8, FACE = 0, readout fixo em +1, dados 1-D.
Nada e ajustado por instancia nem por m.

Arquitetura: f(x) = g_beta(w_0 x + b_0) + (1/sqrt(m)) sum_{k=1..m} g_beta(w_k x + b_k).
A unidade k=0 e a MORTA (A) e mantem readout +1 — e so o readout de A que entra
em c e na equacao da margem, logo a lei predita NAO muda.  As m VIVAS levam
readout FIXO 1/sqrt(m) (escala NTK).  Motivo declarado: com readout +1 nas
vivas, a velocidade do subsistema vivo no espaco de funcoes cresce ~m, o
episodio encolhe ~1/m e a comparacao entre m mistura dois efeitos; com 1/sqrt(m)
a escala de tempo e independente de m e o unico fator que varia e o numero de
unidades.  Dar as vivas um readout diferente do de A e, alias, exatamente o tipo
de generalizacao que o teorema de forcing afirma cobrir ("qualquer subsistema").
Com readout fixo e padrao de ativacao congelado por (H4), o campo de B e AFIM em
theta_B para QUALQUER m — e por isso que L2 (Gronwall) atravessa sem mudanca
nesta classe; a troca afim -> Lipschitz-local so e necessaria com readout
treinado ou 2a camada.

Filtro (H1')-(H4') medido na trajetoria HIBRIDA (a mesma logica do sweep 2-neuronios):
  (H1') morte transversal: z_A(x_face) cruza 0 em t_hit in (0.01, H), R^hyb(t_hit) < 0.
  (H2') Area^hyb tem primeiro zero em t_exit < H, com R^hyb(t_exit) > 0,
        Area^hyb < 0 em (t_hit, t_exit) e UMA troca de sinal de R na janela.
  (H3') as outras margens de A ficam <= -delta1 = 0.05.
  (H4') nenhum chaveamento das VIVAS na janela; A nao cruza 0 antes de t_hit.
Instancia que falha qualquer item conta como NAO-passou, com o motivo gravado.

Controle de discretizacao: --controle roda um subconjunto com dt0 refinado 5x
para separar erro do modelo de erro do integrador.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

FACE = 0
T_FIM = 1.8
DT0 = 1e-5
BETA = 1e4
DELTA1 = 0.05
T_MIN_MORTE = 0.005
N_DADOS = 4
MS = (4, 8, 16)
N_INST = 600          # sorteios por m
N_MAX_FULL = 25       # avaliacoes completas por m (as primeiras por seed)
SEED0 = 20260726
AQUI = os.path.dirname(os.path.abspath(__file__))


def softplus(z, b):
    return np.logaddexp(0.0, b * z) / b


def sig(z, b):
    return 1.0 / (1.0 + np.exp(-np.clip(b * z, -500, 500)))


def readout(m):
    """a[0] = 1 (unidade morta A); a[k] = 1/sqrt(m) (vivas, escala NTK)."""
    return np.concatenate([[1.0], np.full(m, 1.0 / np.sqrt(m))])


# ------------------------------------------------------------------- sorteio
def sorteia(seed, m):
    """(X, Y, P0) com P0[0] = unidade MORTA A, P0[1:] = m unidades VIVAS.

    Familia desenhada para que o episodio de entrada-saida seja POSSIVEL — a
    mesma escolha estrutural da instancia de referencia do paper, generalizada:
      * o subsistema vivo comeca SOBRE-ESTIMANDO em todos os dados (e_i>0), com
        excesso de inclinacao; e o mecanismo que faz o residuo da face cruzar
        zero por acoplamento de kernel (K_{jl}>0) e devolver a area;
      * as vivas se dividem em 3 classes de padrao de ativacao (sufixos
        {x_0..}, {x_1..}, {x_2..}), de modo que f_B e linear-por-partes com 2
        quinas e o subsistema NAO e uma copia reescalada de 1 neuronio;
      * o init das vivas leva escala 1/(tamanho da classe) para manter
        f_B = O(1) com readout fixo em +1. Os gradientes NAO sao reescalados,
        logo L_B ~ m cresce com m: e esse o canal de degradacao testado.
      * z_A(0) e sorteada LOG-UNIFORME em [10^-4, 10^-1.3] (rng.uniform(-4.0,
        -1.3) no expoente). Motivo declarado:
        a morte exige z_A(0) < c * int_0^{t_c} e_j dt, e essa integral encolhe
        como ~1/m porque o subsistema vivo converge ~m vezes mais rapido; uma
        faixa log-uniforme fixa cobre os tres m sem sintonia por m.
    """
    rng = np.random.default_rng(seed)
    a = readout(m)
    for _ in range(20000):
        X = np.cumsum(np.concatenate([[rng.uniform(0.2, 0.6)],
                                      rng.uniform(0.3, 0.8, N_DADOS - 1)]))
        aa, sl = rng.uniform(0.2, 0.6), rng.uniform(0.0, 0.4)
        Y = aa + sl * X + rng.uniform(-0.05, 0.05, N_DADOS)
        wA = -rng.uniform(0.2, 1.2)
        z0 = 10.0 ** rng.uniform(-4.0, -1.3)
        bA = z0 - wA * X[FACE]
        if not np.all(np.delete(wA * X + bA, FACE) <= -DELTA1):
            continue
        d1 = rng.uniform(0.05, 0.5)        # sobre-estimacao inicial na face
        W0 = sl + rng.uniform(0.5, 1.5)    # excesso de inclinacao
        cls = [k % 3 for k in range(m)]
        cnt = [cls.count(ccc) for ccc in range(3)]
        P = np.empty((m + 1, 2))
        P[0] = (wA, bA)
        ok = True
        for k, cc in enumerate(cls):
            esc = 1.0 / (a[k + 1] * cnt[cc])          # p/ que a soma pesada seja O(1)
            if cc == 0:
                w = W0 * esc
                b = (Y[FACE] + d1) * esc - w * X[FACE]
            else:
                w = rng.uniform(1.0, 3.0) * esc
                u = rng.uniform(0.3, 0.8) * esc
                if w * (X[cc] - X[cc - 1]) <= u + 0.05 * esc:
                    ok = False
                    break
                b = u - w * X[cc]
            P[k + 1] = (w, b)
        if ok:
            return X, Y, P
    raise RuntimeError("rejeicao nao convergiu")


# ------------------------------------------------- referencia hibrida (relu)
def hibrido(X, Y, P0, dt0=DT0):
    a = readout(len(P0) - 1)
    P = P0.copy()
    t = 0.0
    zA = P[0, 0] * X + P[0, 1]
    if zA[FACE] <= 0:
        return {"ok": False, "motivo": "A ja nasce morto na face"}
    sinalB0 = np.sign(P[1:, 0][:, None] * X[None, :] + P[1:, 1][:, None])
    while True:                                    # fase A-ativa (relu, tudo junto)
        Z = P[:, 0] * X[:, None]                   # (n, K)
        Z = Z.T + P[:, 1][:, None]                 # (K, n)
        if Z[0, FACE] <= 0:
            break
        act = Z > 0
        r = (a[:, None] * np.maximum(Z, 0.0)).sum(axis=0) - Y   # (n,)
        G = a[:, None] * np.stack([(act * (r * X)).mean(axis=1),
                                   (act * r).mean(axis=1)], axis=1)
        P -= dt0 * G
        t += dt0
        if t > T_FIM:
            return {"ok": False, "motivo": "sem morte no horizonte"}
    t_hit = t
    if t_hit <= T_MIN_MORTE:
        return {"ok": False, "motivo": "morte antes de t_min (fora da janela)"}
    zA_hit = P[0, 0] * X + P[0, 1]
    if not np.all(np.delete(zA_hit, FACE) <= -DELTA1):
        return {"ok": False, "motivo": "H3 falha: outra margem de A > -delta1"}

    PB = P[1:].copy()
    nt = int((T_FIM - t_hit) / dt0) + 1
    R_hyb = np.empty(nt)
    troca_pat = False
    for k in range(nt):                            # fase morta: B-sozinho (relu)
        Z = (PB[:, 0] * X[:, None]).T + PB[:, 1][:, None]
        fB = (a[1:, None] * np.maximum(Z, 0.0)).sum(axis=0)
        R_hyb[k] = Y[FACE] - fB[FACE]
        if not troca_pat and not np.all(np.sign(Z) == sinalB0):
            troca_pat = True
            k_troca = k
        act = Z > 0
        r = fB - Y
        PB -= dt0 * a[1:, None] * np.stack([(act * (r * X)).mean(axis=1),
                                            (act * r).mean(axis=1)], axis=1)
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
    return {"ok": True, "t_hit": t_hit, "t_exit": t_exit,
            "R_hit": float(R_hyb[0]), "R_exit": R_exit,
            "dRdt_exit": float((R_hyb[min(k_exit + 1, nt - 1)] - R_hyb[k_exit - 1])
                               / (2 * dt0)) if k_exit >= 1 else float("nan")}


# --------------------------------------------------- acoplado softplus-beta
def roda_acoplado(X, Y, P0, b):
    a = readout(len(P0) - 1)
    dt = min(1e-5, 0.02 / b)
    P = P0.copy()
    t = 0.0
    morto = False
    n_mortes = 0
    while t < T_FIM:
        Z = (P[:, 0] * X[:, None]).T + P[:, 1][:, None]
        r = (a[:, None] * softplus(Z, b)).sum(axis=0) - Y
        s = sig(Z, b)
        P -= dt * a[:, None] * np.stack([(s * (r * X)).mean(axis=1),
                                         (s * r).mean(axis=1)], axis=1)
        t += dt
        z0 = P[0, 0] * X[FACE] + P[0, 1]
        if not morto and z0 <= 0 and t > T_MIN_MORTE:
            morto = True
            n_mortes += 1
        elif morto and z0 >= 0:
            return t, n_mortes
    return None, n_mortes


# ------------------------------------------------------------------ uma seed
def uma(arg):
    seed, m, dt0 = arg
    X, Y, P0 = sorteia(seed, m)
    c = float((X[FACE] ** 2 + 1) / len(X))
    reg = {"seed": int(seed), "m": m, "dt0": dt0, "c": c,
           "X": X.tolist(), "Y": Y.tolist(), "P0": P0.tolist()}
    h = hibrido(X, Y, P0, dt0)
    if not h["ok"]:
        reg.update({"passou": False, "motivo": h["motivo"],
                    "predito": None, "medido": None, "erro_rel": None})
        return reg
    reg.update({k: h[k] for k in ("t_hit", "t_exit", "R_hit", "R_exit", "dRdt_exit")})
    predito = -1.0 / (c * h["R_exit"])
    reg["predito"] = predito
    tr, n_mortes = roda_acoplado(X, Y, P0, BETA)
    reg["n_mortes"] = n_mortes
    if tr is None:
        reg.update({"passou": False, "medido": None, "erro_rel": None,
                    "motivo": "beta-run sem release no horizonte"})
        return reg
    if n_mortes != 1:
        reg.update({"passou": False, "medido": None, "erro_rel": None,
                    "motivo": f"{n_mortes} mortes no beta-run"})
        return reg
    medido = BETA * (tr - h["t_exit"])
    reg.update({"passou": True, "motivo": "", "t_release": tr, "beta": BETA,
                "medido": float(medido),
                "erro_rel": float(abs(medido - predito) / abs(predito))})
    return reg


def resume(regs, titulo, n_sorteadas):
    ok = [r for r in regs if r["passou"]]
    print(f"\n--- {titulo}: {n_sorteadas} sorteadas, {len(regs)} passaram a peneira, "
          f"{len(ok)} passaram (H1')-(H4') em dt0  "
          f"({100*len(ok)/max(n_sorteadas,1):.1f}% do total)")
    motivos = {"peneira: sem episodio (dt grosseiro)": n_sorteadas - len(regs)}
    for r in regs:
        if not r["passou"]:
            motivos[r["motivo"]] = motivos.get(r["motivo"], 0) + 1
    for mo, k in sorted(motivos.items(), key=lambda kv: -kv[1]):
        print(f"      {k:4d}  {mo}")
    if ok:
        e = np.array([r["erro_rel"] for r in ok])
        print(f"    erro rel |medido-predito|/|predito|: mediana {np.median(e)*100:.2f}% "
              f"| p90 {np.percentile(e,90)*100:.2f}% | max {e.max()*100:.2f}%")
        print(f"    R_exit in [{min(r['R_exit'] for r in ok):.4f}, "
              f"{max(r['R_exit'] for r in ok):.4f}]  "
              f"predito in [{min(r['predito'] for r in ok):.2f}, "
              f"{max(r['predito'] for r in ok):.2f}]")
    return ok


DT_SCREEN = 4e-4


def peneira(arg):
    """Pre-filtro BARATO (dt grosseiro, so relu, sem beta-run): descarta as
    instancias sem episodio. Nao decide nada — toda instancia que passa aqui e
    reavaliada integralmente em dt0=1e-5, e reprovada la se (H1')-(H4') falharem."""
    seed, m = arg
    try:
        X, Y, P0 = sorteia(seed, m)
    except RuntimeError:
        return None
    return seed if hibrido(X, Y, P0, DT_SCREEN)["ok"] else None


def main():
    controle = "--controle" in sys.argv
    n_inst = int([a.split("=")[1] for a in sys.argv if a.startswith("--n=")][0]) \
        if any(a.startswith("--n=") for a in sys.argv) else N_INST
    t0 = time.time()
    with Pool(6) as pool:
        cand = pool.map(peneira, [(SEED0 + 100000 * m + i, m)
                                  for m in MS for i in range(n_inst)], chunksize=8)
        tarefas = []
        for m in MS:
            base = SEED0 + 100000 * m
            tarefas += [(s, m, DT0) for s in cand
                        if s is not None and base <= s < base + n_inst][:N_MAX_FULL]
        print(f"peneira (dt={DT_SCREEN}): {len(tarefas)}/{n_inst*len(MS)} candidatas "
              f"em {(time.time()-t0)/60:.1f} min", flush=True)
        regs = pool.map(uma, tarefas, chunksize=1)
    dur = time.time() - t0

    saida = os.path.join(AQUI, "forcing_m.jsonl")
    with open(saida, "w") as fh:
        for r in regs:
            fh.write(json.dumps(r) + "\n")

    print(f"=== DEGRAU 1 — forcing com m vivos + 1 morto, beta={BETA:.0e}, "
          f"dt0={DT0}, H={T_FIM}, {dur/60:.1f} min ===")
    print("baseline 2-neuronios (m=1): mediana 2.68% | p90 7.26% | max 12.01% (n=57)")
    todos_ok = []
    for m in MS:
        todos_ok += resume([r for r in regs if r["m"] == m], f"m = {m} vivos", n_inst)

    if controle and todos_ok:
        alvo = sorted([r for r in todos_ok if r["m"] == MS[-1]],
                      key=lambda r: r["seed"])[:5]
        print(f"\n--- controle de discretizacao (m={MS[-1]}, dt0/5, {len(alvo)} inst.)")
        with Pool(5) as pool:
            fino = pool.map(uma, [(r["seed"], r["m"], DT0 / 5) for r in alvo])
        with open(os.path.join(AQUI, "step_size_control.jsonl"), "w") as fh:
            for r in fino:
                fh.write(json.dumps(r) + "\n")
        for a, f in zip(alvo, fino):
            if f["passou"]:
                print(f"    seed {a['seed']}: erro {a['erro_rel']*100:.2f}% (dt0) -> "
                      f"{f['erro_rel']*100:.2f}% (dt0/5)")
            else:
                print(f"    seed {a['seed']}: dt0/5 nao passou ({f['motivo']})")
    print(f"\njsonl: {saida}")


if __name__ == "__main__":
    main()
