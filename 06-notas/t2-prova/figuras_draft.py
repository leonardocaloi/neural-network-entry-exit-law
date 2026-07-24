"""Figuras do draft T2' (padrao da casa: scienceplots, PDF+PNG, por script).
F1: mecanismo (z(t) + area A(t), beta=1000). F2: constantes previsto-vs-medido.
Valores medidos com proveniencia: saidas dos scripts passo2/prova_L5 (commits)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import scienceplots  # noqa
    plt.style.use(["science", "no-latex"])
except Exception:
    pass

X = np.array([0.5, 1.5]); Y = np.array([0.5, 1.0]); N = 2
PA0 = np.array([-0.5, 0.27]); PB0 = np.array([1.144, 0.261])
FACE = 0; C_G = (X[FACE]**2 + 1) / N; T_FIM = 1.6
T_HIT, T_EXIT = 0.1114, 1.1325

def sp_(z, b): return np.logaddexp(0.0, b * z) / b
def sg_(z, b): return 1.0 / (1.0 + np.exp(-np.clip(b * z, -500, 500)))

# trajetoria acoplada beta=1000 + cronograma hibrido de A(t)
b = 1000.0; dt = 1e-5
p = np.concatenate([PA0, PB0]); ts, zs = [], []
t = 0.0
while t < T_FIM:
    zA, zB = p[0]*X + p[1], p[2]*X + p[3]
    r = sp_(zA, b) + sp_(zB, b) - Y
    sA, sB = sg_(zA, b), sg_(zB, b)
    g = np.concatenate([[np.mean(r*sA*X), np.mean(r*sA)],
                        [np.mean(r*sB*X), np.mean(r*sB)]])
    p -= dt * g; t += dt
    ts.append(t); zs.append(p[0]*X[FACE] + p[1])
ts, zs = np.array(ts), np.array(zs)
# hibrido p/ A(t)
p2 = np.concatenate([PA0, PB0]); t = 0.0
while (p2[0]*X + p2[1])[FACE] > 0:
    zA, zB = p2[0]*X + p2[1], p2[2]*X + p2[3]
    r = np.maximum(zA, 0) + np.maximum(zB, 0) - Y
    aA, aB = zA > 0, zB > 0
    p2 -= dt*np.concatenate([[np.mean(r*aA*X), np.mean(r*aA)],
                             [np.mean(r*aB*X), np.mean(r*aB)]]); t += dt
pB = p2[2:].copy(); tsh, Rh = [], []
while t < T_FIM:
    zB = pB[0]*X + pB[1]
    Rh.append(Y[FACE] - max(zB[FACE], 0)); tsh.append(t)
    rr = np.maximum(zB, 0) - Y; a = zB > 0
    pB -= dt*np.array([np.mean(rr*a*X), np.mean(rr*a)]); t += dt
tsh = np.array(tsh); Ah = np.cumsum(Rh)*dt

fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 3.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1]})
a1.plot(ts, zs, lw=0.9, color="#26547C")
a1.axhline(0, color="k", lw=0.5, alpha=0.5)
a1.axvline(T_HIT, color="#EF476F", lw=0.7, ls=":")
a1.axvline(T_EXIT, color="#06D6A0", lw=0.7, ls=":")
a1.set_ylabel(r"margem $z(t)$")
a1.annotate("morte", (T_HIT, 0.02), fontsize=7, color="#EF476F")
a1.annotate("release", (T_EXIT, 0.02), fontsize=7, color="#06D6A0")
a1.annotate(r"camada $z\sim-\beta^{-1}\log\beta$", (0.55, -0.009), fontsize=7)
a2.plot(tsh, Ah, lw=0.9, color="#26547C")
a2.axhline(0, color="k", lw=0.5, alpha=0.5)
a2.axvline(T_EXIT, color="#06D6A0", lw=0.7, ls=":")
a2.set_ylabel(r"$A(t)=\int_{t_{hit}}^{t} R^{hyb}$")
a2.set_xlabel(r"tempo $t$")
a2.annotate(r"$A(t_{exit})=0$", (T_EXIT + 0.03, 0.001), fontsize=7, color="#06D6A0")
fig.suptitle(r"Portão com memória: release quando $\int R\,dt=0$  ($\beta=10^3$)", fontsize=8)
for ext in ("pdf", "png"):
    fig.savefig(f"fig1-mecanismo.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)

# F2: constantes (medidos com proveniencia dos scripts committed)
betas = np.array([300., 1000., 3000., 10000.])
t_cross = np.array([0.13518, 0.11850, 0.11379, 0.11215])      # passo2
sweep = np.array([-12.74, -13.93, -14.38, -14.68])            # prova_L5 (dt=1e-5)
corr = np.array([0.1561, 0.0659, 0.0271, 0.0098])             # prova_L4/L1
R_EXIT = 0.1080; K_TEO = 1/(C_G*0.2272)
fig, (b1, b2) = plt.subplots(1, 2, figsize=(5.4, 2.2))
xb = np.linspace(0, 1/280, 100)
b1.plot(1/betas, t_cross, "o", ms=4, color="#26547C", label="medido")
b1.plot(xb, 0.11141 + K_TEO*xb, "-", lw=0.9, color="#EF476F",
        label=rf"teoria: $K=1/(c|R_{{hit}}|)={K_TEO:.2f}$")
b1.set_xlabel(r"$1/\beta$"); b1.set_ylabel(r"$t_{cross}$")
b1.legend(fontsize=6); b1.set_title("quina de entrada (L3)", fontsize=8)
b2.plot(betas, sweep, "o", ms=4, color="#26547C", label="medido (acoplado)")
b2.plot(betas, (-1+corr)/(C_G*R_EXIT), "s--", ms=3, lw=0.8, color="#EF476F",
        label=r"teoria: $(-1+\mathrm{corr})/(cR_{exit})$")
b2.axhline(-1/(C_G*R_EXIT), color="k", lw=0.5, ls=":",
           label=r"$-1/(cR_{exit})=-14.81$")
b2.set_xscale("log"); b2.set_xlabel(r"$\beta$")
b2.set_ylabel(r"$\beta\,(t_{rel}-t_{exit})$")
b2.legend(fontsize=6); b2.set_title("lei de dois termos (L5)", fontsize=8)
for ext in ("pdf", "png"):
    fig.savefig(f"fig2-constantes.{ext}", dpi=300, bbox_inches="tight")
print("figuras: fig1-mecanismo.{pdf,png}, fig2-constantes.{pdf,png}")
