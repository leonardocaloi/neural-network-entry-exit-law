"""Figura do paper relu-morte: escada do canal C4 (4a.3), p(volta|morte) por beta,
com CIs Clopper-Pearson. Numeros verbatim de
05-experimentos/etapa4-mlp-populacao/pilotos-4a2/pressao/taxa/resultado-4a3.md
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

# resultado-4a3.md, coluna p(volta|morte) CP [ponto, inf, sup]; 400 runs, 100/braco
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
ax.set_ylabel(r"$p(\mathrm{return}\mid\mathrm{death})$  [CP 95\%]")
ax.set_xlabel(r"activation stiffness (channel $e^{\beta z}$ closes as $\beta\to\infty$)")
ax.set_ylim(-0.05, 0.72)
ax.set_title(r"Channel staircase, MLP 4a.3 (400 runs, $T=0$)", fontsize=8)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"fig-escada-4a3.{ext}", dpi=300, bbox_inches="tight")
print("fig-escada-4a3.{pdf,png} escritas")

# --- achatamento das figuras externas (B1 da gap-list) -----------------------
AQUI = Path(__file__).resolve().parent
EXTERNAS = {
    "fig1-mecanismo.pdf": AQUI / ".." / "t2-prova" / "fig1-mecanismo.pdf",
    "fig2-constantes.pdf": AQUI / ".." / "t2-prova" / "fig2-constantes.pdf",
    "fig3-diagnostico.pdf": (AQUI / ".." / ".." / "05-experimentos"
                             / "etapa4-mlp-populacao"
                             / "fig-diagnostico-duracoes.pdf"),
}
for destino, origem in EXTERNAS.items():
    origem = origem.resolve()
    if origem.is_file():
        shutil.copyfile(origem, AQUI / destino)
        print(f"{destino} <- {origem}")
    else:
        print(f"AVISO: origem ausente, {destino} nao atualizada ({origem})")
