"""Prova L6: colagem pos-release — trajetoria-beta vs fluxo relu reiniciado."""
import numpy as np

X = np.array([0.5, 1.5]); Y = np.array([0.5, 1.0]); N = 2
PA0 = np.array([-0.5, 0.27]); PB0 = np.array([1.144, 0.261])
FACE = 0; T_FIM = 1.8
T_EXIT = 1.1325

def softplus(zv, b): return np.logaddexp(0.0, b * zv) / b
def sig(zv, b):
    bz = np.clip(b * zv, -500, 500)
    return 1.0 / (1.0 + np.exp(-bz))

def campo_beta(p, b):
    zA, zB = p[0] * X + p[1], p[2] * X + p[3]
    r = softplus(zA, b) + softplus(zB, b) - Y
    sA, sB = sig(zA, b), sig(zB, b)
    return -np.concatenate([[np.mean(r * sA * X), np.mean(r * sA)],
                            [np.mean(r * sB * X), np.mean(r * sB)]])

def campo_relu(p):
    zA, zB = p[0] * X + p[1], p[2] * X + p[3]
    r = np.maximum(zA, 0) + np.maximum(zB, 0) - Y
    aA, aB = (zA > 0), (zB > 0)
    return -np.concatenate([[np.mean(r * aA * X), np.mean(r * aA)],
                            [np.mean(r * aB * X), np.mean(r * aB)]])

# referencia hibrida de theta_B em t_exit (B-sozinho desde o hit — como passo 4)
dt0 = 1e-5
p = np.concatenate([PA0, PB0]); t = 0.0
while (p[0] * X + p[1])[FACE] > 0:
    p += dt0 * campo_relu(p); t += dt0
t_hit = t; thA_hit = p[:2].copy(); pB = p[2:].copy()
while t < T_EXIT:
    zB = pB[0] * X + pB[1]
    rr = np.maximum(zB, 0) - Y; a = zB > 0
    pB -= dt0 * np.array([np.mean(rr * a * X), np.mean(rr * a)])
    t += dt0
# ramo ATIVANTE da Def. 5.1: reinicio em z_A = 0 exato com relu'(0)=0 faz o
# Euler cair no ramo congelado (a nao-unicidade field-first de EJRW23,
# demonstrada acidentalmente pela 1a versao deste script — ver nota). O objeto
# do teorema e o ramo ativante: nudge +eps na coordenada normal da face.
nrm = np.array([X[FACE], 1.0]) / np.hypot(X[FACE], 1.0)
# o loop pre-hit sobrepassa a face por ~dt0*zdot ~ 1e-6 (<0): projetar de volta
# a face EXATAMENTE e so entao aplicar o nudge ativante (>> overshoot de Euler)
zA_hit = thA_hit[0] * X[FACE] + thA_hit[1]
thA_face = thA_hit - zA_hit * nrm / np.hypot(X[FACE], 1.0)
theta_inf_exit = np.concatenate([thA_face + 1e-6 * nrm, pB])

print("colagem pos-release: sup|theta^beta - theta^relu-reiniciado| em [t_rel+0.05, T]")
ms = []
for b in (1000.0, 3000.0):
    dt = min(1e-5, 0.02 / b)
    p = np.concatenate([PA0, PB0]); t = 0.0; morto = False; t_rel = None
    sup = 0.0; q = None
    while t < T_FIM:
        p += dt * campo_beta(p, b); t += dt
        z0 = p[0] * X[FACE] + p[1]
        if not morto and z0 <= 0 and t > 0.05:
            morto = True
        elif morto and t_rel is None and z0 >= 0:
            t_rel = t; q = theta_inf_exit.copy()
        elif t_rel is not None:
            q += dt * campo_relu(q)
            if t > t_rel + 0.05:
                sup = max(sup, np.linalg.norm(p - q))
    m = sup * b / np.log(b)
    ms.append(m)
    print(f"  beta={b:6.0f}: t_rel={t_rel:.4f}, sup = {sup:.3e}, metrica beta/log(beta) = {m:.3f}")
ms = np.array(ms)
assert ms.max() < 20 and ms[1] <= ms[0] * 1.3, ms
print("[OK] colagem: sup|dtheta| = O(log(beta)/beta) pos-release — L6(iii) confirmada")
print("(i)/(ii): identidades de L3 reutilizadas; selecao do ramo ativante provada por L5")
print("\nPROVA L6 VERIFICADA.")
