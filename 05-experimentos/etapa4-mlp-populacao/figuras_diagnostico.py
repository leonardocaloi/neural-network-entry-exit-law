"""Figura diagnóstica pós-hoc (rotulada exploratória) do veredito INVÁLIDO:
distribuição de durações de episódio por braço (m=64) — o chatter de fronteira
(T>0: mediana 4 passos; relu ≡ softplus) vs a física real (T=0: ~1e3 passos,
99.6% censurado). Gera PDF+PNG (mesmo basename), como manda a skill."""
import gzip
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BRACOS = [("softplus", "softplus T=1e-3", "C0"),
          ("relu", "ReLU-dura T=1e-3", "C3"),
          ("T0", "softplus T=0", "C2")]
fig, ax = plt.subplots(figsize=(3.6, 2.7), dpi=300)
bins = np.logspace(0, 5, 40)
for nome, rot, cor in BRACOS:
    durs, cens = [], 0
    for s in range(1700, 1720):
        d = json.load(gzip.open(f"results/{nome}-m64-s{s}.json.gz", "rt"))
        for ep in d["episodios"]:
            if ep["t_res"] >= 0:
                durs.append(ep["t_res"] - ep["t_morte"])
            else:
                cens += 1
    tot = len(durs) + cens
    ax.hist(durs, bins=bins, histtype="step", color=cor, lw=1.2,
            label=f"{rot} (censura {cens/tot:.0%})", density=True)
ax.axvspan(1, 5, color="0.85", zorder=0)
ax.text(2.1, ax.get_ylim()[1] * 0.5 if ax.get_ylim()[1] > 0 else 1,
        "chatter", fontsize=7, rotation=90, va="center", color="0.35")
ax.set_xscale("log")
ax.set_xlabel("duração do episódio (passos)")
ax.set_ylabel("densidade")
ax.legend(fontsize=6, frameon=False)
ax.set_title("m=64, 20 seeds — relu ≡ softplus em T>0: ruído", fontsize=7)
fig.tight_layout()
fig.savefig("fig-diagnostico-duracoes.pdf")
fig.savefig("fig-diagnostico-duracoes.png")
print("fig-diagnostico-duracoes.{pdf,png} escritas")
