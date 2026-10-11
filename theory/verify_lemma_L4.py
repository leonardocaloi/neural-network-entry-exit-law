"""Prova L4: monotonicidade (SymPy) + formula do release Dt = corr/(c beta R_exit)
testada na mesma run + corolario do -15.0."""
import numpy as np
import sympy as sp

# (1) monotonicidade do RHS em Phi (via u): d/du [c*log(1+1/u)] < 0
u_, c_ = sp.symbols('u c', positive=True)
rhs_corr = c_ * sp.log(1 + 1 / u_)
assert sp.simplify(sp.diff(rhs_corr, u_)) == -c_ / (u_ * (u_ + 1))
print("[OK] (1): termo de correcao decrescente em u (RHS decrescente em Phi) — comparacao monotona vale")

# cronograma hibrido (identico ao passo 4)
X = np.array([0.5, 1.5]); Y = np.array([0.5, 1.0]); N = 2
PA0 = np.array([-0.5, 0.27]); PB0 = np.array([1.144, 0.261])
FACE = 0; C_G = (X[FACE] ** 2 + 1) / N; T_FIM = 1.6
def softplus(zv, b): return np.logaddexp(0.0, b * zv) / b
def sig(zv, b):
    bz = np.clip(b * zv, -500, 500)
    return 1.0 / (1.0 + np.exp(-bz))
dt0 = 2e-6
p = np.concatenate([PA0, PB0]); t = 0.0
while True:
    zA, zB = p[0] * X + p[1], p[2] * X + p[3]
    if zA[FACE] <= 0: break
    r = np.maximum(zA, 0) + np.maximum(zB, 0) - Y
    aA, aB = (zA > 0), (zB > 0)
    p -= dt0 * np.concatenate([[np.mean(r * aA * X), np.mean(r * aA)],
                               [np.mean(r * aB * X), np.mean(r * aB)]])
    t += dt0
t_hit = t; pB = p[2:].copy()
nt = int((T_FIM - t_hit) / dt0) + 1
R_tab = np.empty(nt)
for k in range(nt):
    zB = pB[0] * X + pB[1]
    R_tab[k] = Y[FACE] - max(zB[FACE], 0.0)
    rr = np.maximum(zB, 0) - Y; a = zB > 0
    pB -= dt0 * np.array([np.mean(rr * a * X), np.mean(rr * a)])
A_tab = np.cumsum(R_tab) * dt0
pos = np.nonzero(A_tab > 0)[0]
t_exit = t_hit + pos[0] * dt0; R_exit = R_tab[pos[0]]
print(f"hibrido: t_hit={t_hit:.4f}, t_exit={t_exit:.4f}, R_exit={R_exit:+.4f}")

# margem escalar prescrita: mede t_rel e corr NA MESMA run
def episodio(b):
    dt = min(2e-6, 0.02 / b)
    z, t, corr = 0.0, t_hit, 0.0
    while t < T_FIM:
        Rt = R_tab[min(int((t - t_hit) / dt0), nt - 1)]
        z += dt * (-C_G * (softplus(z, b) - Rt) * sig(z, b))
        corr += dt * C_G * np.log1p(np.exp(np.clip(b * z, None, 50)))
        t += dt
        if t > t_hit + 0.05 and z >= 0:
            return t, corr
    return None, corr

print("formula do release: beta*Dt medido vs corr/(c*R_exit):")
for b in (300.0, 1000.0, 3000.0):
    tr, corr = episodio(b)
    dt_med = b * (tr - t_exit)
    dt_teo = corr / (C_G * R_exit)
    desv = abs(dt_med - dt_teo) / max(abs(dt_teo), 1e-9)
    print(f"  beta={b:6.0f}: beta*Dt = {dt_med:+.3f} | previsto = {dt_teo:+.3f} | desvio {desv*100:.1f}%")
    assert desv < 0.10, (b, dt_med, dt_teo)
print("[OK] (ii): o atraso do release E a integral de correcao de L1 (fecha a <10%)")

# corolario: constante acoplada -K*|R_hit|/R_exit vs -14.38 (passo 4, beta=1e4)
K_EMP = 7.127; R_HIT = 0.2272
c_teo = -K_EMP * R_HIT / R_exit
desv = abs(c_teo - (-14.38)) / 14.38
print(f"[.] corolario: -K*|R_hit|/R_exit = {c_teo:.2f} vs -14.38 medido (desvio {desv*100:.1f}%)")
assert desv < 0.08
print("[OK] corolario: a constante do passo 4 esta explicada (entrada corta area negativa)")
print("\nPROVA L4 VERIFICADA.")
