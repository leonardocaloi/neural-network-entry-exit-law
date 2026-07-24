"""T2' Passo 3: sticking/release com R prescrito — L3.0-L3.2 numericos + simbolico."""
import numpy as np
import sympy as sp

# ---------- L3.0 simbolico: dPhi/dt = -c*beta*R (fator beta) ---------------------
u, c, R, beta_s = sp.symbols('u c R beta', positive=True)
du_dt = -beta_s * c * R * u / (1 + u)          # I3 em tempo lento: du/dt = beta*du/dtau
Phi = sp.log(u) + u
assert sp.simplify(sp.diff(Phi, u) * du_dt + c * beta_s * R) == 0
print("[OK] L3.0: dPhi/dt = -c*beta*R (fator beta explicito em tempo lento)")
# corolario feedback: softplus = log(1+w)/beta, w=1/u pequeno => ~ 1/(beta*u) = O(beta^-2) se u~beta
w = sp.symbols('w', positive=True)
assert sp.limit(sp.log(1 + w) / w, w, 0) == 1
print("[OK] corolario: softplus_beta(z) ~ w/beta = 1/(beta*u) = O(beta^-2) no miolo (u~beta)")

# ---------- numerico: R prescrito da instancia (B-sozinho, beta alto) ------------
X = np.array([0.5, 1.5]); Y = np.array([0.5, 1.0]); N = 2
PB0 = np.array([1.144, 0.261]); FACE = 0
C_G = (X[FACE] ** 2 + 1) / N
T_FIM = 1.6

def softplus(z, b): return np.logaddexp(0.0, b * z) / b
def sig(z, b):
    bz = np.clip(b * z, -500, 500)
    return 1.0 / (1.0 + np.exp(-bz))

# cronograma R(t): B-sozinho em regime relu (exato para B ativo-ambos)
dt0 = 1e-5
nt = int(T_FIM / dt0) + 1
pB = PB0.copy()
R_tab = np.empty(nt)
for k in range(nt):
    zB = pB[0] * X + pB[1]
    R_tab[k] = Y[FACE] - max(zB[FACE], 0.0)
    r = np.maximum(zB, 0) - Y
    a = (zB > 0)
    pB -= dt0 * np.array([np.mean(r * a * X), np.mean(r * a)])

t_hit = 0.11141                                  # limite do passo 2 (t_inf do fit)
i_hit = int(t_hit / dt0)
A_tab = np.concatenate([[0.0], np.cumsum(R_tab[i_hit:]) * dt0])   # A(t) desde t_hit
pos = np.nonzero(A_tab > 0)[0]
t_exit = t_hit + pos[0] * dt0
R_exit = R_tab[i_hit + pos[0]]
Amax = np.abs(A_tab[:pos[0]]).max()
print(f"cronograma: t_hit={t_hit}, t_exit={t_exit:.4f}, R0(t_exit)={R_exit:+.4f}, |A|max={Amax:.4f}")

def episodio(b):
    """margem escalar z' = -c*(softplus(z)-R(t))*sig(beta z), de z(t_hit)=0."""
    dt = min(1e-5, 0.02 / b)
    z, t = 0.0, t_hit
    umax, zmin, fb_meio = 1.0, 0.0, None
    t_meio = 0.5 * (t_hit + t_exit)
    while t < T_FIM:
        Rt = R_tab[min(int(t / dt0), nt - 1)]
        z += dt * (-C_G * (softplus(z, b) - Rt) * sig(z, b))
        t += dt
        zmin = min(zmin, z)
        umax = max(umax, np.exp(np.clip(-b * z, None, 500)))
        if fb_meio is None and t >= t_meio:
            fb_meio = softplus(z, b)
        if t > t_hit + 0.05 and z >= 0:          # release: retorno a z=0
            return t, umax, zmin, fb_meio
    return None, umax, zmin, fb_meio

print(f"{'beta':>7} {'t_rel':>8} {'beta*(t_rel-t_exit)':>20} {'umax/(c*b*|A|max)':>18} "
      f"{'b*|zmin|/log(b)':>16} {'fb_meio*b^2':>12}")
mets = []
for b in (300.0, 1000.0, 3000.0, 10000.0):
    tr, umax, zmin, fb = episodio(b)
    assert tr is not None, f"sem release em beta={b}"
    m = (b * (tr - t_exit), umax / (C_G * b * Amax), b * abs(zmin) / np.log(b), fb * b * b)
    mets.append(m)
    print(f"{b:7.0f} {tr:8.4f} {m[0]:20.3f} {m[1]:18.3f} {m[2]:16.3f} {m[3]:12.3f}")

mets = np.array(mets)
assert np.abs(mets[:, 0]).max() < 50, "t_release nao e t_exit + O(1/beta)"
assert mets[2:, 1].std() / mets[2:, 1].mean() < 0.2, "umax nao escala c*beta*|A|"
assert mets[:, 2].max() / mets[:, 2].min() < 2.0, "zmin nao escala log(beta)/beta"
assert mets[2:, 3].max() / mets[2:, 3].min() < 3.0, "feedback nao e O(beta^-2)"
print("\n[OK] L3.1: u_max ~ c*beta*|A|max e z_min ~ -log(beta)/beta (cola exponencial)")
print("[OK] L3.2: t_release = t_exit + O(1/beta), SEM fator log(beta)")
print("[OK] corolario: feedback no miolo = O(beta^-2)")
print("\nPASSO 3 VERIFICADO.")
