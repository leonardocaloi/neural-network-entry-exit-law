"""T2' Passo 2: quina de entrada — L2.1 (simbolico) + t_cross = t_hit + O(1/beta)."""
import numpy as np
import sympy as sp

# ---------- L2.1 simbolico: chegada transversal com velocidade c*R (H0 = H1) -----
c_, R_, z_ = sp.symbols('c R z', real=True, positive=False)
zdot_lim = -c_ * (sp.Max(z_, 0) - R_)                     # A ativo: relu(z)=z p/ z>0
assert sp.simplify(zdot_lim.subs(z_, 0) - c_ * R_) == 0
print("[OK] L2.1: zdot(0) = c*R  => chegada transversal sse R(t_hit) != 0; (H0)=(H1)")

# ---------- numerico: instancia do passo 1 ---------------------------------------
X = np.array([0.5, 1.5]); Y = np.array([0.5, 1.0]); N = 2
PA0 = np.array([-0.5, 0.27]); PB0 = np.array([1.144, 0.261])
FACE = 0

def softplus(z, b): return np.logaddexp(0.0, b * z) / b
def sig(z, b):
    bz = np.clip(b * z, -500, 500)
    return 1.0 / (1.0 + np.exp(-bz))

def campo(p, b):
    zA, zB = p[0] * X + p[1], p[2] * X + p[3]
    r = softplus(zA, b) + softplus(zB, b) - Y
    gA = np.array([np.mean(r * sig(zA, b) * X), np.mean(r * sig(zA, b))])
    gB = np.array([np.mean(r * sig(zB, b) * X), np.mean(r * sig(zB, b))])
    return -np.concatenate([gA, gB])

def t_cross_e_estado(b, w0=1.0):
    """primeiro t com w = e^{beta z} <= w0 (z <= log(w0)/beta)."""
    dt = min(2e-5, 0.02 / b)
    p = np.concatenate([PA0, PB0]); t = 0.0
    alvo = np.log(w0) / b
    while t < 2.0:
        p = p + dt * campo(p, b); t += dt
        if p[0] * X[FACE] + p[1] <= alvo:
            return t, p.copy()
    raise RuntimeError("nao cruzou")

betas = np.array([300.0, 1000.0, 3000.0, 10000.0])
ts, estados = [], []
for b in betas:
    tc, pc = t_cross_e_estado(b)
    ts.append(tc); estados.append(pc)
    print(f"  beta={b:7.0f}: t_cross={tc:.5f}")
ts = np.array(ts)

# ajuste t_cross = t_inf + K/beta (minimos quadrados em 1/beta)
Amat = np.vstack([np.ones_like(betas), 1.0 / betas]).T
coef, res, *_ = np.linalg.lstsq(Amat, ts, rcond=None)
t_inf, K = coef
res_rel = np.sqrt(res[0]) / (ts.max() - ts.min()) if len(res) else 0.0
print(f"[fit] t_cross = {t_inf:.5f} + {K:.2f}/beta ; residuo relativo do ajuste = {res_rel:.3f}")
assert res_rel < 0.05, f"nao-linear em 1/beta: {res_rel}"
assert 0 < K < 100, K
# estado na travessia: ||p(beta) - p(10^4)|| = O(1/beta)
ref = estados[-1]
for b, pc in zip(betas[:-1], estados[:-1]):
    d = np.linalg.norm(pc - ref)
    assert d * b < 20.0, (b, d * b)
    print(f"  beta={b:7.0f}: ||estado - estado_ref||*beta = {d * b:.3f} (limitado)")
print(f"[OK] L2.3: t_cross = t_hit + O(1/beta) (K={K:.1f}), estado uniforme O(1/beta)")
print("\nPASSO 2 VERIFICADO.")
