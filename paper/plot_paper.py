"""Figura do paper relu-morte: escada do canal C4 (4a.3), p(volta|morte) por beta,
com CIs Clopper-Pearson. Numeros verbatim de
experiments/mlp/pilots/pressure/rates/results-4a3.md
(commit 5a961cb). PDF+PNG (padrao da casa). Roda em nice -19.

Tambem ACHATA para este diretorio as figuras geradas noutros lugares, com os
nomes que paper-v2.tex usa sem caminho (o tarball do arXiv tem raiz unica e nao
resolve '../'). Assim a figura no paper nunca diverge da figura gerada."""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots  # noqa
    plt.style.use(["science", "no-latex"])
except Exception:
    pass

# results-4a3.md, coluna p(volta|morte) CP [ponto, inf, sup]; 400 runs, 100/braco
ARMS = [
    (r"$\beta=3$",   0.457, 0.288, 0.634, "16/35"),
    (r"$\beta=10$",  0.370, 0.194, 0.576, "10/27"),
    (r"$\beta=300$", 0.211, 0.061, 0.456, "4/19"),
    (r"ReLU",        0.000, 0.000, 0.195, "0/17"),
]
x = np.arange(len(ARMS))
p = np.array([a[1] for a in ARMS])
lo = np.array([a[2] for a in ARMS])
hi = np.array([a[3] for a in ARMS])
labels = [a[0] for a in ARMS]
frac = [a[4] for a in ARMS]

fig, ax = plt.subplots(figsize=(3.6, 2.6))
err = np.vstack([p - lo, hi - p])
colors = ["#26547C", "#3D7EA6", "#F4A259", "#EF476F"]
for i in range(len(ARMS)):
    ax.errorbar(x[i], p[i], yerr=err[:, i:i + 1], fmt="o", ms=6, capsize=3,
                lw=1.2, color=colors[i])
    ax.annotate(frac[i], (x[i], hi[i] + 0.02), fontsize=6, ha="center",
                color="0.35")
ax.axhline(0, color="k", lw=0.5, alpha=0.4)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel(r"$p(\mathrm{return}\mid\mathrm{death})$  [CP 95%]")
ax.set_xlabel(r"activation stiffness (channel $e^{\beta z}$ closes as $\beta\to\infty$)")
ax.set_ylim(-0.05, 0.72)
ax.set_title(r"Channel staircase, MLP 4a.3 (400 runs, $T=0$)", fontsize=8)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"fig-4a3-activation-sweep.{ext}", dpi=300, bbox_inches="tight")
print("fig-4a3-activation-sweep.{pdf,png} escritas")

# --- achatamento das figuras externas (B1 da gap-list) -----------------------
AQUI = Path(__file__).resolve().parent
T2 = AQUI / ".." / "theory"
E4 = AQUI / ".." / "experiments" / "mlp"
EXTERNAS = {                                   # destino (sem extensao): origem
    "fig1-mechanism": T2 / "fig1-mechanism",
    "fig2-constants": T2 / "fig2-constants",
    "fig3-diagnostics": E4 / "fig-duration-diagnostics",
}
for destino, origem in EXTERNAS.items():
    for ext in ("pdf", "png"):
        o = Path(str(origem) + f".{ext}").resolve()
        if o.is_file():
            shutil.copyfile(o, AQUI / f"{destino}.{ext}")
            print(f"{destino}.{ext} <- {o}")
        else:
            print(f"AVISO: origem ausente, {destino}.{ext} nao atualizada ({o})")
