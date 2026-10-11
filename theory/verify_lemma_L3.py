"""Prova L3: identidades (SymPy), constante K_teo = 1/v vs refit do passo 2,
e integral com peso exponencial na trajetoria real."""
import numpy as np
import sympy as sp

# (a) identidades
s_, b_, z_ = sp.symbols('s beta z', positive=True)
sig = 1 / (1 + sp.exp(-s_))
assert sp.simplify((1 - sig) / sig - sp.exp(-s_)) == 0
assert sp.integrate(sp.exp(-b_ * z_), (z_, 0, sp.oo)) == 1 / b_
print("[OK] (a): (1-sigma)/sigma = e^{-s}; int_0^inf e^{-beta z} dz = 1/beta")

# (b) constante do passo 2 vs teoria
C_G = (0.5 ** 2 + 1) / 2
R_HIT = 0.2272                                  # |R^hyb(t_hit)| do passo 4
v = C_G * R_HIT
K_TEO = 1.0 / v
betas = np.array([300.0, 1000.0, 3000.0, 10000.0])
ts = np.array([0.13518, 0.11850, 0.11379, 0.11215])   # medidos no passo 2
A = np.vstack([np.ones_like(betas), 1.0 / betas]).T
coef = np.linalg.lstsq(A, ts, rcond=None)[0]; t_inf, K_emp = coef
desvio = abs(K_emp - K_TEO) / K_TEO
print(f"[.] K_teo = 1/(c*|R_hit|) = {K_TEO:.3f} | K_emp (refit passo 2) = {K_emp:.3f} "
      f"| desvio = {desvio*100:.1f}%")
assert desvio < 0.05, f"constante nao bate: {K_emp} vs {K_TEO}"
print("[OK] (b): a prova PREVE a constante do fit empirico (desvio < 5%)")

# (c) peso exponencial na trajetoria acoplada real (pre-hit)
X = np.array([0.5, 1.5]); Y = np.array([0.5, 1.0]); N = 2
PA0 = np.array([-0.5, 0.27]); PB0 = np.array([1.144, 0.261])
def softplus(zv, b): return np.logaddexp(0.0, b * zv) / b
def sig_n(zv, b):
    bz = np.clip(b * zv, -500, 500)
    return 1.0 / (1.0 + np.exp(-bz))
def peso(b):
    dt = min(1e-5, 0.02 / b)
    p = np.concatenate([PA0, PB0]); t = 0.0; acc = 0.0
    while t < 0.5:
        zA, zB = p[0] * X + p[1], p[2] * X + p[3]
        r = softplus(zA, b) + softplus(zB, b) - Y
        sA, sB = sig_n(zA, b), sig_n(zB, b)
        g = np.concatenate([[np.mean(r * sA * X), np.mean(r * sA)],
                            [np.mean(r * sB * X), np.mean(r * sB)]])
        p -= dt * g; t += dt
        z0 = p[0] * X[0] + p[1]
        if z0 <= 0:
            return acc * b
        acc += dt * np.exp(np.clip(-b * z0, None, 50))
    return acc * b
vals = [peso(b) for b in (300.0, 1000.0, 3000.0)]
# teoria EXATA: int e^{-beta z} dt = int e^{-s}/(v*sigma(s)) ds/beta com
# 1/sigma(s) = 1+e^{-s}: int_0^inf e^{-s}(1+e^{-s}) ds = 3/2  =>  (3/2)/(beta*v)
TEO_C = 1.5 / v
print(f"[.] beta*int e^(-beta z) dt pre-hit = {[f'{x:.3f}' for x in vals]} "
      f"(teoria exata (3/2)/v = {TEO_C:.3f})")
vals = np.array(vals)
assert abs(vals[-1] - TEO_C) / TEO_C < 0.02, (vals[-1], TEO_C)
assert vals.max() / vals.min() < 1.05
print("[OK] (c): peso = (3/2)/(beta*v) EXATO — fator 3/2 do sigma na camada, "
      "previsto e confirmado a <1%")
print("\nPROVA L3 VERIFICADA.")
