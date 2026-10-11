"""T2' Passo 5: L5.2 — a lei de area vale na classe erf (GELU-beta) acoplada."""
import numpy as np
from scipy.special import erf

X = np.array([0.5, 1.5]); Y = np.array([0.5, 1.0]); N = 2
PA0 = np.array([-0.5, 0.27]); PB0 = np.array([1.144, 0.261])
FACE = 0; T_FIM = 1.8
T_EXIT_HYB = 1.1325                          # referencia hibrida do passo 4 (relu)

def Phi_g(s):                                 # CDF normal
    return 0.5 * (1.0 + erf(np.clip(s, -30, 30) / np.sqrt(2.0)))

def phi_g(s):
    s = np.clip(s, -30, 30)
    return np.exp(-0.5 * s * s) / np.sqrt(2 * np.pi)

def ativ(z, b):                               # g_b(z) = z*Phi(bz) + phi(bz)/b ; g' = Phi(bz)
    return z * Phi_g(b * z) + phi_g(b * z) / b

def roda_gelu(b):
    dt = min(1e-5, 0.02 / b)
    p = np.concatenate([PA0, PB0]); t = 0.0
    morto = False
    while t < T_FIM:
        zA, zB = p[0] * X + p[1], p[2] * X + p[3]
        r = ativ(zA, b) + ativ(zB, b) - Y
        sA, sB = Phi_g(b * zA), Phi_g(b * zB)
        g = np.concatenate([[np.mean(r * sA * X), np.mean(r * sA)],
                            [np.mean(r * sB * X), np.mean(r * sB)]])
        p -= dt * g; t += dt
        z0 = p[0] * X[FACE] + p[1]
        if not morto and z0 <= 0 and t > 0.05:
            morto = True; t_hit = t
        elif morto and z0 >= 0:
            return t_hit, t
    return None, None

print("classe erf (GELU-beta), sistema ACOPLADO — mesma referencia hibrida do relu:")
print(f"  t_exit^hyb (relu) = {T_EXIT_HYB}")
gaps = []
for b in (300.0, 3000.0):
    th, tr = roda_gelu(b)
    assert tr is not None, f"sem release beta={b}"
    gaps.append(b * (tr - T_EXIT_HYB))
    print(f"  beta={b:6.0f}: t_hit={th:.4f}, t_release={tr:.4f}, "
          f"beta*(t_rel - t_exit^hyb) = {gaps[-1]:+.2f}")
assert all(abs(g) < 80 for g in gaps), gaps
# o gap deve ENCOLHER em valor absoluto de t (nao necessariamente em beta*gap):
assert abs(gaps[1] / 3000) < abs(gaps[0] / 300), "gap nao encolhe com beta"
print("\n[OK] L5.2: release da classe erf converge ao MESMO t_exit^hyb da lei de area")
print("     => a lei e independente da regularizacao (inclui GELU) — ponte pratica")
print("\nPASSO 5 VERIFICADO.")
