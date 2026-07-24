"""T2' Passo 1: verificacao — (H1)/(H2) na instancia do ataque + Gronwall empirico.
Convencao: R = y_j - f_{-A}(x_j)  (R>0 = empuxo pro-ativacao).
Instancia (do ataque de 2026-07-24): A=(w_A,b_A) softplus, B=(w_B,b_B) softplus,
X=(0.5,1.5), Y=(0.5,1.0), GF em tempo continuo (Euler fino), sem ruido.
"""
import numpy as np

X = np.array([0.5, 1.5])
Y = np.array([0.5, 1.0])
N = 2
PA0 = np.array([-0.5, 0.27])
PB0 = np.array([1.144, 0.261])
T_FIM = 2.0
FACE = 0                                    # face de morte da instancia (dado 0)
C_GRAM = (X[FACE] ** 2 + 1) / N


def softplus(z, beta):
    return np.logaddexp(0.0, beta * z) / beta


def sig(z, beta):
    bz = np.clip(beta * z, -500, 500)
    return 1.0 / (1.0 + np.exp(-bz))


def campo(p, beta):
    """GF completo 4D: p = (w_A,b_A,w_B,b_B)."""
    zA, zB = p[0] * X + p[1], p[2] * X + p[3]
    r = softplus(zA, beta) + softplus(zB, beta) - Y
    gA = np.array([np.mean(r * sig(zA, beta) * X), np.mean(r * sig(zA, beta))])
    gB = np.array([np.mean(r * sig(zB, beta) * X), np.mean(r * sig(zB, beta))])
    return -np.concatenate([gA, gB])


def campo_B_sozinho(pB, beta):
    zB = pB[0] * X + pB[1]
    r = softplus(zB, beta) - Y
    return -np.array([np.mean(r * sig(zB, beta) * X), np.mean(r * sig(zB, beta))])


def integra(f, p0, dt, T):
    n = int(T / dt)
    p = p0.copy()
    traj = np.empty((n + 1, len(p0)))
    traj[0] = p
    for k in range(n):
        p = p + dt * f(p)                    # Euler fino (dt << 1/beta na camada)
        traj[k + 1] = p
    return traj


# ---------- (H1)/(H2) no R^0 de B-sozinho (beta alto p/ B ~ regime relu ativo) ----
BETA_REF = 3000.0
dt0 = 2e-5
trB = integra(lambda q: campo_B_sozinho(q, BETA_REF), PB0, dt0, T_FIM)
tt = np.arange(len(trB)) * dt0
R0 = Y[FACE] - softplus(trB[:, 0] * X[FACE] + trB[:, 1], BETA_REF)

# t_hit: precisa da trajetoria COMPLETA (A vivo ate morrer) — detectar no sistema
# acoplado; usar beta alto como proxy do limite
tr4 = integra(lambda q: campo(q, BETA_REF), np.concatenate([PA0, PB0]), dt0, T_FIM)
zA0 = tr4[:, 0] * X[FACE] + tr4[:, 1]
i_hit = int(np.argmax(zA0 < 0))
t_hit = i_hit * dt0
assert 0 < t_hit < T_FIM, t_hit
R0_hit = R0[i_hit]
assert R0_hit < 0, f"(H1) falhou: R0(t_hit)={R0_hit}"
print(f"[OK] (H1): t_hit={t_hit:.3f}, R0(t_hit)={R0_hit:+.4f} < 0 (morte transversal)")

A_int = np.cumsum(C_GRAM * R0[i_hit:]) * dt0
pos = np.nonzero(A_int > 0)[0]
assert len(pos) > 0, "(H2) falhou: area nunca retorna a zero"
i_exit = i_hit + pos[0]
t_exit = i_exit * dt0
assert R0[i_exit] > 0, f"(H2) transversalidade falhou: R0(t_exit)={R0[i_exit]}"
assert np.all(A_int[:pos[0]] <= 1e-12), "(H2): area deveria ser <0 no episodio"
print(f"[OK] (H2): t_exit={t_exit:.3f}, R0(t_exit)={R0[i_exit]:+.4f} > 0, "
      f"area<0 em (t_hit,t_exit)")

# ---------- Gronwall empirico: beta * sup|theta_B^beta - theta_B^0| ~ const ------
print("Gronwall empirico (referencia: B-sozinho a partir do estado em t_hit^beta):")
consts = []
for beta in (300.0, 1000.0, 3000.0):
    dt = min(2e-5, 0.05 / beta)
    tr = integra(lambda q: campo(q, beta), np.concatenate([PA0, PB0]), dt, T_FIM)
    zA = tr[:, 0] * X[FACE] + tr[:, 1]
    ih = int(np.argmax(zA < 0))
    th = ih * dt
    # janela do episodio: [t_hit, t_exit] (usar t_exit da referencia, conservador)
    n_ep = int((t_exit - th) / dt)
    ref = integra(lambda q: campo_B_sozinho(q, beta), tr[ih, 2:].copy(), dt, n_ep * dt)
    m = min(len(ref), len(tr) - ih)
    dif = np.linalg.norm(tr[ih:ih + m, 2:] - ref[:m], axis=1)
    c_emp = beta * dif.max()
    consts.append(c_emp)
    print(f"  beta={beta:6.0f}: t_hit={th:.3f}, beta*sup|dTheta_B| = {c_emp:.4f}")
# L1.2 afirma cota SUPERIOR O(1/beta): verificar beta*sup|.| limitado; observar
# decaimento ainda mais rapido CONFIRMA a cota com folga (lema conservador).
assert max(consts) < 0.2, f"cota O(1/beta) violada: {consts}"
assert consts[-1] <= consts[0], f"nao-monotono: {consts}"
print(f"[OK] L1.2: beta*sup|dTheta_B| <= {max(consts):.3f} (cota O(1/beta) confirmada"
      f" com folga; empiricamente o feedback decai ate MAIS rapido — lema conservador)")

# ---------- L1.3: estabilidade de t_exit sob perturbacao do forcing --------------
for d0 in (0.01, 0.02):
    Rp = R0 + d0
    Ap = np.cumsum(C_GRAM * Rp[i_hit:]) * dt0
    ip = np.nonzero(Ap > 0)[0][0]
    dt_exit = abs((i_hit + ip) * dt0 - t_exit)
    cota = C_GRAM * d0 * (t_exit - t_hit) / abs(C_GRAM * R0[i_exit]) * 1.5  # folga 50%
    assert dt_exit <= cota, (dt_exit, cota)
    print(f"[OK] L1.3: delta0={d0}: |t_exit-shift| = {dt_exit:.4f} <= cota {cota:.4f}")

print("\nPASSO 1 VERIFICADO — (H1)/(H2) valem na instancia; Gronwall O(1/beta); "
      "estabilidade de t_exit conforme L1.3.")
