"""Figura diagnóstica pós-hoc (rotulada exploratória) do veredito INVÁLIDO:
distribuição de durações de episódio por braço (m=64) — o chatter de fronteira
(T>0: mediana ~4 passos; relu ≡ softplus, impossível por gradiente) vs a física
real (T=0: quase tudo censurado no horizonte de 1e5).

RÓTULOS EM INGLÊS (a figura entra no paper, que é em inglês). Gera PDF+PNG com o
mesmo basename, como manda a skill.

Honestidade do braço T=0 (revisão 2026-07-26): em m=64 o braço T=0 tem SÓ 2
episódios completos (672 e 1548 passos; mediana 1110) contra 459 censurados —
uma "densidade" desenhada a partir de 2 pontos seria teatro. Os 2 completos
entram como marcas individuais e a massa censurada entra como seta no horizonte.
A incerteza dos braços T>0 é a dispersão entre as 20 seeds (±1 e.p.).
"""
import gzip
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except Exception:
    pass
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 6})

SEEDS = range(1700, 1720)
H = 100_000                       # horizonte dos runs da etapa-4a
# durações são INTEIRAS: arredondar as bordas log e deduplicar evita bins vazios
# (o pente artificial em durações pequenas) sem mexer nos dados.
BINS = np.unique(np.round(np.logspace(0, 5, 34)))
BINS[-1] = 1e5
CENTROS = np.sqrt(BINS[:-1] * BINS[1:])


def carrega(nome):
    """Por seed: durações completas e nº de censurados. Dados reais, sem filtro."""
    por_seed, censurados = [], 0
    for s in SEEDS:
        d = json.load(gzip.open(f"results/{nome}-m64-s{s}.json.gz", "rt"))
        durs = [e["t_res"] - e["t_morte"] for e in d["episodios"] if e["t_res"] >= 0]
        censurados += sum(1 for e in d["episodios"] if e["t_res"] < 0)
        por_seed.append(np.asarray(durs, float))
    return por_seed, censurados


def perfil(por_seed):
    """Fração de episódios por bin, média e erro-padrão ENTRE seeds."""
    P = np.array([np.histogram(d, bins=BINS)[0] / max(len(d), 1)
                  for d in por_seed if len(d)])
    return P.mean(0), P.std(0, ddof=1) / np.sqrt(len(P))


fig, ax = plt.subplots(figsize=(3.5, 2.6))

# softplus grosso, relu tracejado POR CIMA: as duas coincidirem é o resultado
for nome, rotulo, cor, lw, ls in [
        ("softplus", r"softplus, $T=10^{-3}$", "#26547C", 2.0, "-"),
        ("relu", r"hard ReLU, $T=10^{-3}$", "#EF476F", 1.0, (0, (2.6, 1.6)))]:
    por_seed, cens = carrega(nome)
    todas = np.concatenate(por_seed)
    med, ep = perfil(por_seed)
    frac_cens = cens / (len(todas) + cens)
    ax.fill_between(CENTROS, med - ep, med + ep, color=cor, alpha=0.30, lw=0,
                    step="mid")
    ax.step(CENTROS, med, where="mid", color=cor, lw=lw, ls=ls,
            label=f"{rotulo}: {len(todas)} episodes, "
                  f"median {np.median(todas):.0f}, {frac_cens:.1%} censored")

# braço T=0: 2 completos em 461 — marcas individuais, nunca uma "densidade"
por_seed0, cens0 = carrega("T0")
completas = np.concatenate(por_seed0) if any(len(d) for d in por_seed0) else np.array([])
frac0 = cens0 / (len(completas) + cens0)
y0 = 0.055
ax.plot(completas, np.full(len(completas), y0), "v", ms=4.5, color="#06D6A0",
        mec="k", mew=0.4, zorder=5,
        label=(f"softplus, $T=0$: {len(completas)} completed of "
               f"{len(completas) + cens0} ({frac0:.1%} censored)"))
ax.annotate("", xy=(H, y0), xytext=(2.2e3, y0),
            arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#06D6A0",
                            shrinkA=3, shrinkB=0))
ax.text(1.05e4, y0 + 0.008, f"{cens0} censored at $H=10^5$", fontsize=6,
        color="#06D6A0", ha="center")

ax.axvspan(1, 5, color="0.86", zorder=0, lw=0)
ax.text(3.6, 0.20, "chatter", fontsize=6.5, rotation=90, va="center",
        ha="center", color="0.35")
ax.set_xscale("log")
ax.set_xlim(1, 1.4e5)
ax.set_ylim(0, 0.30)
ax.set_xlabel("episode duration (steps)")
ax.set_ylabel("fraction of episodes per bin")
ax.legend(loc="upper right", frameon=False, handlelength=1.3, labelspacing=0.3)
ax.set_title(r"$m=64$, 20 seeds: at $T>0$ ReLU $\equiv$ softplus $\Rightarrow$ noise",
             fontsize=7.5)
fig.tight_layout()
fig.savefig("fig-diagnostico-duracoes.pdf", dpi=400)
fig.savefig("fig-diagnostico-duracoes.png", dpi=400)
print("fig-diagnostico-duracoes.{pdf,png} escritas (rótulos em inglês)")
print(f"  T=0: completas={completas.tolist()} censuradas={cens0} "
      f"frac={frac0:.4f}")
