"""DEGRAU 3a — verificacao numerica dos TEOREMAS A e B (multi-dado).

Objeto: a unidade morta A com TODAS as n faces ativas, campo EXATO (D1) do
scoping 2026-07-27:

    zdot_i = sum_k C_ik (R_k(t) - g_beta(z_k)) sigma(beta z_k),   C = X X^T / n.

Como o campo so ve C (e nao X), sorteamos C diretamente: C = diag(c_1..c_n)
no caso C-ortogonal, e C = D + acoplamento no caso violado (mantendo PSD).
Forcing PRESCRITO (classe thm:forcing do paper): R_i(t) = -a_i + b_i t.

TRES BLOCOS (registrados ANTES da execucao):

 (a) TEOREMA A — lider estavel.  beta*delta_gap >= 10 por construcao, C
     diagonal.  PREDICAO: a lei ESCALAR do lider (area do lider cruza zero)
     acerta o tempo de release com erro relativo O(1/beta), e acerta a face.
     MATA: se a face predita errar, ou se o erro nao decair com beta.

 (b) TEOREMA B — lei diagonal por face.  C diagonal (ortogonal), limiares
     DISTINTOS tau_i = [Psi(u_i^0) - 1]/(beta C_ii); o gap NAO e grande (a
     competicao entre faces e o fenomeno).  PREDICAO: t_rel = min_i t_i e a
     face vencedora = argmin.  MATA: face errada, ou erro que nao decai.

 (c) VIOLACAO CONTROLADA — C_ik grande.  PREDICAO (delimita o teorema): a lei
     diagonal passa a errar a face e/ou o tempo, e o erro NAO decai com beta.
     Se ela continuasse acertando, a hipotese de C-ortogonalidade seria
     desnecessaria e o Teorema B estaria mal enunciado.

Saida: degrau3a_verifica.jsonl (uma linha por instancia).
MAXIMO 2 workers, nice -19.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
SEED0 = 20260729
N_A = 18   # bloco (a)
N_B = 18   # bloco (b)
N_C = 12   # bloco (c)
T_FIM = 8.0


# ----------------------------------------------------------------- dinamica
def g_beta(z, beta):
    # softplus estavel
    return np.logaddexp(0.0, beta * z) / beta


def sigmoide(z, beta):
    return 1.0 / (1.0 + np.exp(-beta * z))


def campo(z, t, C, A, B, beta):
    R = A + B * t
    return C @ ((R - g_beta(z, beta)) * sigmoide(z, beta))


def integra(z0, C, A, B, beta, dt, t_fim=T_FIM):
    """RK4 no campo exato (D1).  Retorna (t_rel, face, ok)."""
    z = z0.copy()
    t = 0.0
    n_pass = int(np.ceil(t_fim / dt))
    for _ in range(n_pass):
        k1 = campo(z, t, C, A, B, beta)
        k2 = campo(z + 0.5 * dt * k1, t + 0.5 * dt, C, A, B, beta)
        k3 = campo(z + 0.5 * dt * k2, t + 0.5 * dt, C, A, B, beta)
        k4 = campo(z + dt * k3, t + dt, C, A, B, beta)
        zn = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if t > dt and zn.max() >= 0.0:
            i = int(np.argmax(zn))
            # cruzamento interpolado na face i
            if zn[i] != z[i]:
                frac = -z[i] / (zn[i] - z[i])
            else:
                frac = 1.0
            return t + frac * dt, i, True
        z = zn
        t += dt
    return None, None, False


# --------------------------------------------------------------- preditores
def area(i, t, A, B):
    return A[i] * t + 0.5 * B[i] * t * t


def limiar(i, z0, C, beta):
    """tau_i = [Psi(u_i^0) - 1] / (beta C_ii),  Psi(u) = log u + u."""
    u0 = np.exp(-beta * z0[i])
    Psi = -beta * z0[i] + u0
    return (Psi - 1.0) / (beta * C[i, i])


def primeiro_cruz(i, tau, A, B, t_fim=T_FIM, n_grid=200001):
    """primeiro t com area_i(t) >= tau (grid + bissecao)."""
    ts = np.linspace(0.0, t_fim, n_grid)
    f = A[i] * ts + 0.5 * B[i] * ts * ts - tau
    idx = np.where(f >= 0.0)[0]
    idx = idx[idx > 0]
    if idx.size == 0:
        return None
    j = int(idx[0])
    lo, hi = ts[j - 1], ts[j]
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if A[i] * mid + 0.5 * B[i] * mid * mid - tau >= 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def lei_por_face(z0, C, A, B, beta):
    """Teorema B: t_i = primeiro cruzamento do limiar tau_i; vence o min."""
    cand = []
    for i in range(len(z0)):
        tau = limiar(i, z0, C, beta)
        t_i = primeiro_cruz(i, tau, A, B)
        if t_i is not None:
            cand.append((t_i, i, tau))
    if not cand:
        return None, None, None
    t_i, i, tau = min(cand)
    corr = 1.0 / (beta * C[i, i] * (A[i] + B[i] * t_i))
    return t_i, i, t_i - corr


def lei_escalar_lider(z0, C, A, B, beta):
    """Teorema A: so o lider (z0 = 0), limiar 0, com a correcao -1/(beta c R)."""
    i = int(np.argmax(z0))
    t_i = primeiro_cruz(i, limiar(i, z0, C, beta), A, B)
    if t_i is None:
        return None, i, None
    corr = 1.0 / (beta * C[i, i] * (A[i] + B[i] * t_i))
    return t_i, i, t_i - corr


# ---------------------------------------------------------------- instancias
def sorteia(bloco, k):
    rng = np.random.default_rng(SEED0 + 1000 * "abc".index(bloco) + k)
    n = int(rng.integers(2, 4))              # n = 2 ou 3 dados
    beta = float(rng.choice([200.0, 500.0, 1000.0, 2000.0]))
    diag = rng.uniform(0.5, 1.5, size=n)
    C = np.diag(diag)
    # forcing: R_i = -a_i + b_i t, a_i>0 (entrada transversal), b_i>0
    a = rng.uniform(0.4, 1.5, size=n)
    b = rng.uniform(0.6, 3.0, size=n)
    z0 = np.zeros(n)
    lider = 0
    if bloco == "a":
        # gap GRANDE por construcao: beta*delta >= 10
        bd = rng.uniform(10.0, 16.0, size=n)
        z0[1:] = -bd[1:] / beta
    elif bloco == "b":
        # gap MODERADO: limiares por face comparaveis -> competicao real
        bd = rng.uniform(0.5, 4.0, size=n)
        z0[1:] = -bd[1:] / beta
        # e o lider da margem tem area lenta: forca a competicao
        a[0] = float(rng.uniform(0.9, 1.5))
        b[0] = float(rng.uniform(0.6, 1.1))
    else:  # "c": violacao — acoplamento forte, C ainda PSD
        bd = rng.uniform(0.5, 4.0, size=n)
        z0[1:] = -bd[1:] / beta
        a[0] = float(rng.uniform(0.9, 1.5))
        b[0] = float(rng.uniform(0.6, 1.1))
        # X_i = (x_i, 1) reais -> C = X X^T / n, fortemente acoplada
        x = rng.uniform(0.6, 1.6, size=n)
        X = np.stack([x, np.ones(n)], axis=1)
        C = X @ X.T / n
    a = -a  # A_i = -a_i  (R_i(0) < 0)
    return dict(n=n, beta=beta, C=C, A=a, B=b, z0=z0, lider=lider)


def uma(arg):
    bloco, k = arg
    ins = sorteia(bloco, k)
    C, A, B, beta, z0 = ins["C"], ins["A"], ins["B"], ins["beta"], ins["z0"]
    dt = min(1e-4, 1.0 / (20.0 * beta))
    t_num, face_num, ok = integra(z0, C, A, B, beta, dt)
    reg = dict(bloco=bloco, k=k, n=ins["n"], beta=beta, dt=dt,
               C=[[float(v) for v in row] for row in C],
               A=[float(v) for v in A], B=[float(v) for v in B],
               z0=[float(v) for v in z0],
               beta_delta=[float(-beta * v) for v in z0],
               ok=bool(ok))
    if not ok:
        return reg
    reg["t_num"] = float(t_num)
    reg["face_num"] = int(face_num)

    tS0, faceS, tS = lei_escalar_lider(z0, C, A, B, beta)
    tB0, faceB, tB = lei_por_face(z0, C, A, B, beta)
    reg["S_t_exit"] = None if tS0 is None else float(tS0)
    reg["S_face"] = int(faceS)
    reg["S_pred"] = None if tS is None else float(tS)
    reg["B_t_exit"] = None if tB0 is None else float(tB0)
    reg["B_face"] = None if faceB is None else int(faceB)
    reg["B_pred"] = None if tB is None else float(tB)
    reg["tau"] = [float(limiar(i, z0, C, beta)) for i in range(ins["n"])]

    for nome, pred, t0, face in (("S", tS, tS0, faceS), ("B", tB, tB0, faceB)):
        if pred is None:
            continue
        reg[f"err_{nome}"] = float(abs(pred - t_num) / t_num)
        reg[f"err_{nome}_ord0"] = float(abs(t0 - t_num) / t_num)
        reg[f"face_ok_{nome}"] = bool(face == face_num)
        reg[f"sinal_{nome}"] = float((pred - t_num) * beta)  # beta*(pred-num)
    # controle de dt: metade do passo
    t2, f2, ok2 = integra(z0, C, A, B, beta, dt / 2.0)
    reg["dt_ctrl"] = None if not ok2 else float(abs(t2 - t_num) / t_num)
    reg["dt_ctrl_face"] = None if not ok2 else int(f2)
    return reg


def controle_n1():
    """CONTROLE n=1: o cenario ESCALAR do paper (Thm 6(ii)).  Serve para decidir
    se a discrepancia do coeficiente de 1a ordem -1/(c R) beta^-1 e um artefato
    multi-face ou ja esta no caso escalar."""
    regs = []
    for beta in (200.0, 1000.0, 5000.0, 20000.0):
        C = np.array([[1.0]])
        A = np.array([-1.0])
        B = np.array([1.2])
        z0 = np.array([0.0])
        dt = min(1e-4, 1.0 / (20.0 * beta))
        t, f, ok = integra(z0, C, A, B, beta, dt)
        te = primeiro_cruz(0, 0.0, A, B)
        cR = float(C[0, 0] * (A[0] + B[0] * te))
        regs.append(dict(bloco="n1", beta=beta, ok=bool(ok), dt=dt,
                         t_num=float(t), t_exit=float(te), cR=cR,
                         beta_d=float(beta * (t - te)),
                         beta2_d_sobre_log=float(beta * beta * (t - te)
                                                 / np.log(beta)),
                         coef_paper=float(-1.0 / cR)))
        print("  n1 beta=%7.0f  beta*(t_rel-t_exit) = %+.4f   "
              "coef do paper -1/(cR) = %+.3f   beta^2 d/log beta = %.2f"
              % (beta, regs[-1]["beta_d"], regs[-1]["coef_paper"],
                 regs[-1]["beta2_d_sobre_log"]))
    return regs


def main():
    tarefas = ([("a", k) for k in range(N_A)]
               + [("b", k) for k in range(N_B)]
               + [("c", k) for k in range(N_C)])
    with Pool(2) as p:
        regs = p.map(uma, tarefas)
    print("\n=== CONTROLE n=1 (cenario escalar do paper) ===")
    regs = regs + controle_n1()
    out = os.path.join(AQUI, "degrau3a_verifica.jsonl")
    with open(out, "w") as fh:
        for r in regs:
            fh.write(json.dumps(r) + "\n")
    # ---- sumario
    for bloco in "abc":
        R = [r for r in regs if r["bloco"] == bloco and r.get("ok")]
        print(f"\n=== BLOCO {bloco}  (N={len(R)}) ===")
        for nome in ("S", "B"):
            e = [r[f"err_{nome}"] for r in R if f"err_{nome}" in r]
            fo = [r[f"face_ok_{nome}"] for r in R if f"face_ok_{nome}" in r]
            if not e:
                continue
            print(f"  {nome}: face ok {sum(fo)}/{len(fo)}   "
                  f"err med {np.median(e):.3e}  max {max(e):.3e}")
            for bt in sorted(set(r["beta"] for r in R)):
                ee = [r[f"err_{nome}"] for r in R
                      if r["beta"] == bt and f"err_{nome}" in r]
                if ee:
                    print(f"      beta={bt:6.0f}  N={len(ee):2d}  "
                          f"err med {np.median(ee):.3e}  "
                          f"beta*err med {np.median(ee) * bt:.3f}")
        d = [r["dt_ctrl"] for r in R if r.get("dt_ctrl") is not None]
        if d:
            print(f"  controle dt (dt/2): max {max(d):.2e}")
    print("\nescrito:", out)


if __name__ == "__main__":
    sys.exit(main())
