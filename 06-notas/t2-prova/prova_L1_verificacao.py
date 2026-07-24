"""Prova L1: identidade aditiva exata (SymPy) + cota integral O(log beta / beta)."""
import numpy as np
import sympy as sp

# ---------- passos 2-3: identidade exata, simbolica, SEM aproximacao -------------
z, c, R, beta = sp.symbols('z c R beta', real=True)
bpos = sp.Symbol('beta', positive=True)
z_, c_, R_ = sp.symbols('z c R', real=True)
u = sp.exp(-bpos * z_)
w = sp.exp(bpos * z_)
softplus = sp.log(1 + sp.exp(bpos * z_)) / bpos
sigma = 1 / (1 + sp.exp(-bpos * z_))
zdot = -c_ * (softplus - R_) * sigma                      # eq. da margem (termo j)
Phi = sp.log(u) + u
dPhi_dt = sp.simplify(sp.diff(Phi, z_) * zdot)
alvo = -c_ * bpos * R_ + c_ * sp.log(1 + w)
assert sp.simplify(dPhi_dt - alvo) == 0
print("[OK] passos 2-3: dPhi/dt = -c*beta*R + c*log(1+w)  EXATA (sem divisao por R)")

# sigma = 1/(1+u) e beta*softplus = log(1+w) (identidades usadas)
assert sp.simplify(sigma - 1 / (1 + u)) == 0
assert sp.simplify(bpos * softplus - sp.log(1 + w)) == 0
print("[OK] identidades auxiliares: sigma = 1/(1+u), beta*softplus = log(1+w)")

# ---------- passo 4: cota integral, numerica, no episodio real -------------------
X = np.array([0.5, 1.5]); Y = np.array([0.5, 1.0]); N = 2
PA0 = np.array([-0.5, 0.27]); PB0 = np.array([1.144, 0.261])
FACE = 0; T_FIM = 1.6

def softplus_n(zv, b): return np.logaddexp(0.0, b * zv) / b
def sig_n(zv, b):
    bz = np.clip(b * zv, -500, 500)
    return 1.0 / (1.0 + np.exp(-bz))

def correcao_integral(b):
    """roda o 4D acoplado e acumula int c*log(1+w) dt do hit ao release."""
    dt = min(1e-5, 0.02 / b)
    p = np.concatenate([PA0, PB0]); t = 0.0
    morto = False; acc = 0.0
    cG = (X[FACE] ** 2 + 1) / N
    while t < T_FIM:
        zA, zB = p[0] * X + p[1], p[2] * X + p[3]
        r = softplus_n(zA, b) + softplus_n(zB, b) - Y
        sA, sB = sig_n(zA, b), sig_n(zB, b)
        g = np.concatenate([[np.mean(r * sA * X), np.mean(r * sA)],
                            [np.mean(r * sB * X), np.mean(r * sB)]])
        p -= dt * g; t += dt
        z0 = p[0] * X[FACE] + p[1]
        if not morto and z0 <= 0 and t > 0.05:
            morto = True
        elif morto:
            acc += dt * cG * np.log1p(np.exp(np.clip(b * z0, None, 50)))
            if z0 >= 0:
                return acc
    return acc

print("cota da correcao: int c*log(1+w) dt no episodio, normalizada por log(b)/b:")
vals = []
for b in (300.0, 1000.0, 3000.0, 10000.0):
    I = correcao_integral(b)
    vals.append(I * b / np.log(b))
    print(f"  beta={b:7.0f}: integral = {I:.3e}, *beta/log(beta) = {vals[-1]:.3f}")
vals = np.array(vals)
# criterio principiado: (a) metrica normalizada ESTAVEL (scaling certo);
# (b) abaixo da constante teorica C4 ~ 2/(c*min|R|) ~ 30 da prova (c=0.625,
#     min(|R_hit|,R_exit)=0.108). O valor ~10.6 esta confortavelmente dentro.
C4_TEO = 2.0 / (0.625 * 0.108)
assert vals.max() / vals.min() < 1.5, f"scaling instavel: {vals}"
assert vals.max() < C4_TEO, f"acima da constante teorica {C4_TEO:.1f}: {vals}"
print(f"[OK] passo 4: correcao = O(log(beta)/beta), metrica estavel "
      f"[{vals.min():.2f}, {vals.max():.2f}] < C4_teo = {C4_TEO:.1f}")
print("\nPROVA L1 VERIFICADA.")
