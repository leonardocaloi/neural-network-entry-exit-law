"""T2' Passo 4: bootstrap acoplado — release do 4D vs cronograma HIBRIDO."""
import numpy as np

X = np.array([0.5, 1.5]); Y = np.array([0.5, 1.0]); N = 2
PA0 = np.array([-0.5, 0.27]); PB0 = np.array([1.144, 0.261])
FACE = 0; C_G = (X[FACE] ** 2 + 1) / N; T_FIM = 1.8

def softplus(z, b): return np.logaddexp(0.0, b * z) / b
def sig(z, b):
    bz = np.clip(b * z, -500, 500)
    return 1.0 / (1.0 + np.exp(-bz))

# ---------------- referencia hibrida (limite): relu acoplado ate t_hit, depois B-so
dt0 = 1e-5
p = np.concatenate([PA0, PB0]); t = 0.0
while True:                                   # fase A-ativa (relu)
    zA, zB = p[0] * X + p[1], p[2] * X + p[3]
    if zA[FACE] <= 0: break
    r = np.maximum(zA, 0) + np.maximum(zB, 0) - Y
    aA, aB = (zA > 0), (zB > 0)
    g = np.concatenate([[np.mean(r * aA * X), np.mean(r * aA)],
                        [np.mean(r * aB * X), np.mean(r * aB)]])
    p -= dt0 * g; t += dt0
t_hit = t
pB = p[2:].copy()
nt = int((T_FIM - t_hit) / dt0) + 1
R_hyb = np.empty(nt); trB_hyb = np.empty((nt, 2))
for k in range(nt):                           # fase morta: B-sozinho
    zB = pB[0] * X + pB[1]
    R_hyb[k] = Y[FACE] - max(zB[FACE], 0.0)
    trB_hyb[k] = pB
    r = np.maximum(zB, 0) - Y; a = (zB > 0)
    pB -= dt0 * np.array([np.mean(r * a * X), np.mean(r * a)])
A_hyb = np.cumsum(R_hyb) * dt0
pos = np.nonzero(A_hyb > 0)[0]
t_exit = t_hit + pos[0] * dt0
assert R_hyb[0] < 0 and R_hyb[pos[0]] > 0    # (H1)/(H2) na referencia hibrida
print(f"referencia hibrida: t_hit={t_hit:.4f}, t_exit={t_exit:.4f}, "
      f"R_hyb(hit)={R_hyb[0]:+.4f}, R_hyb(exit)={R_hyb[pos[0]]:+.4f}")

# ---------------- 4D acoplado (softplus) por beta --------------------------------
def roda_acoplado(b):
    dt = min(1e-5, 0.02 / b)
    p = np.concatenate([PA0, PB0]); t = 0.0
    morto = False; sup_dB = 0.0
    while t < T_FIM:
        zA, zB = p[0] * X + p[1], p[2] * X + p[3]
        r = softplus(zA, b) + softplus(zB, b) - Y
        sA, sB = sig(zA, b), sig(zB, b)
        g = np.concatenate([[np.mean(r * sA * X), np.mean(r * sA)],
                            [np.mean(r * sB * X), np.mean(r * sB)]])
        p -= dt * g; t += dt
        z0 = p[0] * X[FACE] + p[1]
        if not morto and z0 <= 0 and t > 0.05:
            morto = True
        elif morto:
            k = min(int((t - t_hit) / dt0), nt - 1)
            if k >= 0:
                sup_dB = max(sup_dB, np.linalg.norm(p[2:] - trB_hyb[k]))
            if z0 >= 0:
                return t, sup_dB
    return None, sup_dB

print(f"{'beta':>7} {'t_release':>10} {'beta*(t_rel-t_exit)':>20} {'beta*sup|dTheta_B|':>18}")
res = []
for b in (300.0, 1000.0, 3000.0, 10000.0):
    tr, sdb = roda_acoplado(b)
    assert tr is not None, f"sem release beta={b}"
    res.append((b * (tr - t_exit), b * sdb))
    print(f"{b:7.0f} {tr:10.4f} {res[-1][0]:20.3f} {res[-1][1]:18.3f}")
res = np.array(res)
assert np.abs(res[:, 0]).max() < 60, "L4.2: t_release nao e t_exit_hyb + O(1/beta)"
assert res[:, 1].max() < 60, "bootstrap: theta_B nao fica O(1/beta) do hibrido"
print("\n[OK] L4.2: release ACOPLADO = t_exit^hyb + O(1/beta) — lei fechada verificada")
print("[OK] bootstrap: sup|theta_B - hibrido|*beta limitado de ponta a ponta")
print("\nPASSO 4 VERIFICADO.")
