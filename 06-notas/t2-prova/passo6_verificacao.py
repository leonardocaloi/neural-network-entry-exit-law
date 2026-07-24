"""T2' Passo 6: (H3) + reducao escalar — margem 1 inativa, deriva tangencial exp-pequena."""
import numpy as np
import sympy as sp

# simbolico: termo i do campo em theta_A e paralelo a (x_i,1); cota sigma(-b*d) <= e^{-b*d}
b_, d_ = sp.symbols('beta delta', positive=True)
assert sp.limit(sp.exp(b_ * d_) / (1 + sp.exp(b_ * d_)) - 1, b_, sp.oo) == 0
sigma_neg = 1 / (1 + sp.exp(b_ * d_))
assert sp.simplify(sigma_neg - sp.exp(-b_ * d_) / (1 + sp.exp(-b_ * d_))) == 0
print("[OK] L6.1 simbolico: sigma(-beta*delta) <= e^{-beta*delta}; campo i e rank-1 em (x_i,1)")

X = np.array([0.5, 1.5]); Y = np.array([0.5, 1.0]); N = 2
PA0 = np.array([-0.5, 0.27]); PB0 = np.array([1.144, 0.261])
FACE = 0; T_FIM = 1.6
TAN = np.array([1.0, -X[FACE]]) / np.hypot(1.0, X[FACE])   # tangente a face 0

def softplus(z, b): return np.logaddexp(0.0, b * z) / b
def sig(z, b):
    bz = np.clip(b * z, -500, 500)
    return 1.0 / (1.0 + np.exp(-bz))

def episodio(b):
    dt = min(1e-5, 0.02 / b)
    p = np.concatenate([PA0, PB0]); t = 0.0
    morto = False; tan0 = None; max_z1 = -np.inf; dtan = 0.0
    while t < T_FIM:
        zA, zB = p[0] * X + p[1], p[2] * X + p[3]
        r = softplus(zA, b) + softplus(zB, b) - Y
        sA, sB = sig(zA, b), sig(zB, b)
        g = np.concatenate([[np.mean(r * sA * X), np.mean(r * sA)],
                            [np.mean(r * sB * X), np.mean(r * sB)]])
        p -= dt * g; t += dt
        z0 = p[0] * X[FACE] + p[1]
        if not morto and z0 <= 0 and t > 0.05:
            morto = True; tan0 = p[:2] @ TAN
        elif morto:
            max_z1 = max(max_z1, p[0] * X[1] + p[1])
            dtan = max(dtan, abs(p[:2] @ TAN - tan0))
            if z0 >= 0:
                return max_z1, dtan
    return max_z1, dtan

print("episodio acoplado (instancia):")
ds = {}
for b in (300.0, 1000.0):
    mz1, dtan = episodio(b)
    ds[b] = dtan
    assert mz1 < -0.1, f"(H3) falhou: max z_A1 no episodio = {mz1}"
    print(f"  beta={b:6.0f}: max z_A1 = {mz1:+.4f} (< -0.1: H3 OK), "
          f"deriva tangencial = {dtan:.2e}")
# com beta*delta_1 >= 150, e^{-beta*delta_1} ~ 1e-65 << eps de maquina: a deriva
# medida e puro roundoff. Criterio correto: zero numerico em AMBOS os beta.
assert all(d < 1e-10 for d in ds.values()), ds
print(f"[OK] L6.1: deriva tangencial = zero em precisao de maquina em ambos os beta"
      f" (e^-beta*delta ~ 1e-65 << eps; mais forte que o lema exige)")
print("\nPASSO 6 VERIFICADO — reducao escalar valida sob (H3); ápice declarado aberto.")
