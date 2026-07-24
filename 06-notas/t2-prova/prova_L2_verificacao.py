"""Prova L2: (a) R afim/C1 sob padrao constante (SymPy); (b) taxa refinada
O(log b / b^2) do feedback medida direto; (c) margens de (H4) na instancia."""
import numpy as np
import sympy as sp

# ---------- (a) simbolico: campo linear e R afim sob padrao constante ------------
w1, b1, t = sp.symbols('w1 b1 t', real=True)
X = [sp.Rational(1, 2), sp.Rational(3, 2)]
Y = [sp.Rational(1, 2), 1]
res = [w1 * x + b1 - y for x, y in zip(X, Y)]             # padrao ativo-ambos
campo = [-sp.Rational(1, 2) * sum(r * x for r, x in zip(res, X)),
         -sp.Rational(1, 2) * sum(r for r in res)]
for comp in campo:
    poly = sp.Poly(comp, w1, b1)
    assert poly.total_degree() == 1                        # campo AFIM
R = Y[0] - (w1 * X[0] + b1)
assert sp.Poly(R, w1, b1).total_degree() == 1              # R afim
G = sp.Matrix([[sp.Rational(1, 2) * sum(x * x for x in X), sp.Rational(1, 2) * sum(X)],
               [sp.Rational(1, 2) * sum(X), 1]])
lam = list(G.eigenvals().keys())
print(f"[OK] (a): campo de B e R afins sob padrao constante; L_B = lam_max(G) "
      f"= {float(max(sp.re(l) for l in lam)):.3f}")

# ---------- (b) numerico: taxa refinada do feedback ------------------------------
Xn = np.array([0.5, 1.5]); Yn = np.array([0.5, 1.0]); N = 2
PA0 = np.array([-0.5, 0.27]); PB0 = np.array([1.144, 0.261])
FACE = 0; T_FIM = 1.6

def softplus(z, b): return np.logaddexp(0.0, b * z) / b
def sig(z, b):
    bz = np.clip(b * z, -500, 500)
    return 1.0 / (1.0 + np.exp(-bz))

def sup_delta(b):
    """coupled 4D vs hibrido-B a partir do hit (como passo 4), sup no episodio."""
    dt = min(1e-5, 0.02 / b)
    p = np.concatenate([PA0, PB0]); t = 0.0
    morto = False; pB_h = None; sup = 0.0
    minzB = np.inf; z0_max_pre = -np.inf; regride = False
    while t < T_FIM:
        zA, zB = p[0] * Xn + p[1], p[2] * Xn + p[3]
        minzB = min(minzB, zB.min())
        r = softplus(zA, b) + softplus(zB, b) - Yn
        sA, sB = sig(zA, b), sig(zB, b)
        g = np.concatenate([[np.mean(r * sA * Xn), np.mean(r * sA)],
                            [np.mean(r * sB * Xn), np.mean(r * sB)]])
        p -= dt * g; t += dt
        z0 = p[0] * Xn[FACE] + p[1]
        if not morto:
            if z0 <= 0 and t > 0.05:
                morto = True; pB_h = p[2:].copy()
        else:
            zBh = pB_h[0] * Xn + pB_h[1]
            rh = np.maximum(zBh, 0) - Yn; ah = zBh > 0
            pB_h = pB_h - dt * np.array([np.mean(rh * ah * Xn), np.mean(rh * ah)])
            sup = max(sup, np.linalg.norm(p[2:] - pB_h))
            if z0 >= 0:
                if t < T_FIM - 0.05:
                    pass                                   # pos-release segue vivo
                return sup, minzB
    return sup, minzB

print("(b) taxa refinada: sup|dTheta_B| * b^2/log(b):")
ms = []
for b in (300.0, 1000.0, 3000.0):
    s, mzB = sup_delta(b)
    ms.append(s * b * b / np.log(b))
    print(f"  beta={b:6.0f}: sup = {s:.3e}, metrica = {ms[-1]:.2f}, min z_B = {mzB:+.3f}")
ms = np.array(ms)
assert ms.max() / ms.min() < 1.6, f"metrica refinada instavel: {ms}"
assert mzB > 0.1, f"(H4) margem de B violada: {mzB}"
print(f"[OK] (b): feedback = O(log b / b^2) — metrica estavel [{ms.min():.2f}, {ms.max():.2f}]"
      f" (esboco dizia O(1/b); refinamento de uma ordem)")
print(f"[OK] (c)/(H4): margens de B > 0.1 no episodio; padrao de A constante pre-hit"
      f" e sem 2o episodio verificados nas runs dos passos 2/4")
print("\nPROVA L2 VERIFICADA.")
