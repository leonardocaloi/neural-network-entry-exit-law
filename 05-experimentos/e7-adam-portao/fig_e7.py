"""Figura exploratoria e7: z_A(t) e os dois relogios, GD vs Adam (instancia de referencia)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(AQUI, "results")

d = np.load(os.path.join(RES, "serie_ref.npz"))
gd, ad = d["gd"], d["adam"]
gde, ade = d["gd_ev"], d["ad_ev"]

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "figure.dpi": 140})
fig, ax = plt.subplots(2, 2, figsize=(9.2, 5.4), sharex="col")

for j, (s, ev, tit) in enumerate([(gd, gde, "GD (canonico)"),
                                  (ad, ade, "Adam ($\\epsilon=10^{-8}$)")]):
    t, z, R, aR, aS = s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4]
    a0, a1 = ax[0, j], ax[1, j]
    a0.plot(t, z * 1e4, color="C3", lw=1.0)
    a0.axhline(0, color="k", lw=0.6)
    a0.set_ylabel(r"$\beta\,z_A(x_{\rm face})$")
    a0.set_ylim(-30, 30)      # zoom na barreira: a fase viva sai da escala
    zp = np.nanmin(z * 1e4)
    a0.annotate(f"$\\beta z_{{\\rm park}}={zp:.1f}$", xy=(0.55, 0.12),
                xycoords="axes fraction", fontsize=8, color="C3")
    a0.set_title(tit)
    a1.plot(t, aR, color="C0", lw=1.2, label=r"$\int R\,dt$ (lei velha)")
    a1.plot(t, aS, color="C1", lw=1.2, label=r"$\int {\rm sign}(R)\,dt$ (H-SIGN)")
    a1.axhline(0, color="k", lw=0.6)
    a1.set_xlabel("t")
    a1.set_ylabel("relogios acumulados")
    for a in (a0, a1):
        a.axvline(ev[0], color="0.4", ls=":", lw=1.0)
        if np.isfinite(ev[1]):
            a.axvline(ev[1], color="C3", ls="--", lw=1.2)
        if np.isfinite(ev[2]):
            a.axvline(ev[2], color="C0", ls="-.", lw=1.0)
        if np.isfinite(ev[3]):
            a.axvline(ev[3], color="C1", ls="-.", lw=1.0)
    a1.legend(loc="lower right", fontsize=7.5, framealpha=0.9)

fig.suptitle("e7 [EXPLORATORIO]: portao de memoria sob GD vs Adam — instancia de "
             "referencia ($\\beta=10^4$)\npontilhado cinza: morte; tracejado "
             "vermelho: release observado; tracos-pontos: relogio zerando",
             fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.93))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(AQUI, f"fig-e7-relogios.{ext}"), bbox_inches="tight")
print("figura escrita")

# ------------------------------------------------- tabela de erros por eps
with open(os.path.join(RES, "e7_runs.jsonl")) as fh:
    regs = [json.loads(l) for l in fh]
print(f"\n{'eps':>8} {'N':>3} {'rel':>4} {'medErrArea':>11} {'medErrSign':>11} "
      f"{'nArea':>6} {'nSign':>6} {'bz_park':>9} {'rat_med':>9}")
for eps in sorted({r["eps"] for r in regs}):
    rs = [r for r in regs if r["eps"] == eps]
    ok = [r for r in rs if r["adam"]["t_release"] is not None
          and r["adam"]["n_mortes"] == 1]
    ea = [r["err_area"] for r in ok if r["err_area"] is not None]
    es = [r["err_sign"] for r in ok if r["err_sign"] is not None]
    bz = [r["adam"]["bz_park"] for r in ok]
    rt = [r["adam"]["rat_med"] for r in ok if r["adam"]["rat_med"] is not None]
    f = lambda a: f"{np.median(a)*100:10.2f}%" if a else "        --"
    print(f"{eps:8.0e} {len(rs):3d} {len(ok):4d} {f(ea)} {f(es)} "
          f"{len(ea):6d} {len(es):6d} "
          f"{np.median(bz) if bz else float('nan'):9.2f} "
          f"{np.median(rt) if rt else float('nan'):9.2e}")

ok = [r for r in regs if r["eps"] == 1e-8 and r["gd"]["t_release"] is not None]
ega = [r["err_area_gd"] for r in ok if r["err_area_gd"] is not None]
egs = [r["err_sign_gd"] for r in ok if r["err_sign_gd"] is not None]
print(f"\nBASELINE GD (mesmas instancias, N={len(ok)}): "
      f"medErrArea={np.median(ega)*100:.3f}% (n={len(ega)}), "
      f"medErrSign={np.median(egs)*100:.2f}% (n={len(egs)})")
