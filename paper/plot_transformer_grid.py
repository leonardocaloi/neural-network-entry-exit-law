"""Figure 6 (paper j-space): confirmatory grid of etapa-5 (transformer FFN).

All numbers are verbatim from the frozen analysis
experiments/transformer/results-e5.md (800 runs, delta_h=0.0476,
H=2e5, 50 runs per cell = activation x depth x upstream-regime).

Panels (a),(b): per-cell sustained-return rate p_sust with 95% Clopper-Pearson
intervals, shallow injection (z=-0.3) and deep injection (z=-2.0).
Panel (c): the upstream channel, Delta = p_sust(training) - p_sust(frozen)
pooled over both depths (N=100 per arm), bootstrap-by-seed 95% CI.

Suggested caption
-----------------
Figure 6. ("sp beta3" / "sp beta10" = softplus with beta=3 / beta=10.)
Confirmatory grid on the transformer FFN (800 runs; 50 runs per
activation x injection-depth x upstream-regime cell; hysteresis delta_h=0.0476,
horizon H=2x10^5). (a),(b) Sustained-return rate p_sust with 95%
Clopper-Pearson intervals, for shallow (z=-0.3) and deep (z=-2.0) injection.
(c) Upstream channel: Delta = p_sust(training) - p_sust(frozen), pooled over
both depths (N=100 per arm), 95% bootstrap-by-seed interval. Three readings:
(i) Delta is uniform at ~0.20 across all four activations -- moving the
upstream weights adds the same ~20 percentage points of return probability
regardless of the nonlinearity, so the upstream channel is not an artefact of
any one activation; (ii) ReLU-frozen is exactly 0/50 at both depths, the
designed negative control (an exactly dead stratum with a frozen upstream has
no escape route); (iii) shallow GELU is the highest cell (0.58 frozen, 0.86
training), consistent with GELU having no dead stratum at all at z=-0.3
(see Figure 7). Deep injection (z=-2.0) suppresses every frozen cell to 0.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except Exception:
    pass

# --- results-e5.md, "Tabela por celula x profundidade": p_sust CP [p, lo, hi]
# key: (activation, depth, regime) -> (sust, n, p, lo, hi)
CELLS = {
    ("sp-b3",  -0.3, "frozen"):   (22, 50, 0.440, 0.300, 0.587),
    ("sp-b3",  -0.3, "training"): (39, 50, 0.780, 0.640, 0.885),
    ("sp-b10", -0.3, "frozen"):   (9,  50, 0.180, 0.086, 0.314),
    ("sp-b10", -0.3, "training"): (27, 50, 0.540, 0.393, 0.682),
    ("gelu",   -0.3, "frozen"):   (29, 50, 0.580, 0.432, 0.718),
    ("gelu",   -0.3, "training"): (43, 50, 0.860, 0.733, 0.942),
    ("relu",   -0.3, "frozen"):   (0,  50, 0.000, 0.000, 0.071),
    ("relu",   -0.3, "training"): (18, 50, 0.360, 0.229, 0.508),
    ("sp-b3",  -2.0, "frozen"):   (0,  50, 0.000, 0.000, 0.071),
    ("sp-b3",  -2.0, "training"): (4,  50, 0.080, 0.022, 0.192),
    ("sp-b10", -2.0, "frozen"):   (0,  50, 0.000, 0.000, 0.071),
    ("sp-b10", -2.0, "training"): (2,  50, 0.040, 0.005, 0.137),
    ("gelu",   -2.0, "frozen"):   (0,  50, 0.000, 0.000, 0.071),
    ("gelu",   -2.0, "training"): (8,  50, 0.160, 0.072, 0.291),
    ("relu",   -2.0, "frozen"):   (0,  50, 0.000, 0.000, 0.071),
    ("relu",   -2.0, "training"): (2,  50, 0.040, 0.005, 0.137),
}

# --- results-e5.md, "E-B MONTANTE": Delta, CI, (train k/100, frozen k/100)
DELTA = {
    "sp-b3":  (0.210, 0.120, 0.300, 43, 22),
    "sp-b10": (0.200, 0.110, 0.280, 29, 9),
    "gelu":   (0.220, 0.120, 0.320, 51, 29),
    "relu":   (0.200, 0.120, 0.280, 20, 0),
}

ACTS = ["sp-b3", "sp-b10", "gelu", "relu"]
TICK = [r"sp $\beta$3", r"sp $\beta$10", "GELU", "ReLU"]
C_FROZEN, C_TRAIN, C_DELTA = "#26547C", "#F4A259", "#118AB2"

fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.35),
                         gridspec_kw={"width_ratios": [1.0, 1.0, 0.85]})
x = np.arange(len(ACTS))
w = 0.36

for ax, depth, tag in zip(axes[:2], (-0.3, -2.0), ("(a)", "(b)")):
    for j, (reg, col, off) in enumerate([("frozen", C_FROZEN, -w / 2),
                                         ("training", C_TRAIN, +w / 2)]):
        p = np.array([CELLS[(a, depth, reg)][2] for a in ACTS])
        lo = np.array([CELLS[(a, depth, reg)][3] for a in ACTS])
        hi = np.array([CELLS[(a, depth, reg)][4] for a in ACTS])
        ax.bar(x + off, p, w, color=col, edgecolor="black", linewidth=0.5,
               label=f"upstream {reg}", zorder=2)
        ax.errorbar(x + off, p, yerr=np.vstack([p - lo, hi - p]), fmt="none",
                    ecolor="black", elinewidth=0.7, capsize=1.8, zorder=3)
        for xi, pi, hii in zip(x + off, p, hi):
            ax.text(xi, hii + 0.02, f"{pi:.2f}", ha="center", va="bottom",
                    fontsize=5.6, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(TICK, fontsize=7)
    ax.set_ylim(0, 1.30)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title(f"{tag} injection $z={depth}$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4, zorder=0)

axes[0].set_ylabel(r"sustained return $\hat{p}$ ($n=50$)", fontsize=7.5)
axes[1].legend(fontsize=6.2, loc="upper left", framealpha=0.9)
axes[1].annotate("ReLU frozen $=0/50$\nat both depths\n(exact control)",
                 xy=(3 - w / 2, 0.02), xytext=(2.35, 0.60), fontsize=6.0,
                 ha="center", va="bottom",
                 arrowprops=dict(arrowstyle="->", lw=0.6, color="black"))

# panel (c): pooled upstream step
ax = axes[2]
d = np.array([DELTA[a][0] for a in ACTS])
dlo = np.array([DELTA[a][1] for a in ACTS])
dhi = np.array([DELTA[a][2] for a in ACTS])
ax.axhline(0.20, color="0.45", ls="--", lw=0.7, zorder=1)
ax.text(1.5, 0.365, r"dashed line: $\Delta=0.20$", fontsize=6.4, ha="center",
        va="center", color="0.30")
ax.bar(x, d, 0.55, color=C_DELTA, edgecolor="black", linewidth=0.5, zorder=2)
ax.errorbar(x, d, yerr=np.vstack([d - dlo, dhi - d]), fmt="none",
            ecolor="black", elinewidth=0.7, capsize=2.0, zorder=3)
for xi, di, hii in zip(x, d, dhi):
    ax.text(xi, hii + 0.008, f"{di:.2f}", ha="center", va="bottom", fontsize=6.2)
ax.set_xticks(x)
ax.set_xticklabels(TICK, fontsize=7)
ax.set_ylim(0, 0.40)
ax.set_ylabel(r"$\Delta=\hat{p}_{\rm train}-\hat{p}_{\rm frozen}$", fontsize=7.5)
ax.set_title(r"(c) upstream channel ($N=100$)", fontsize=8)
ax.tick_params(labelsize=7)
ax.grid(axis="y", alpha=0.25, linewidth=0.4, zorder=0)

fig.tight_layout(pad=0.4, w_pad=1.0)
for ext, kw in (("pdf", dict(dpi=400)), ("png", dict(dpi=400))):
    fig.savefig(f"fig6-transformer-grid.{ext}", bbox_inches="tight", **kw)

print("fig6-transformer-grid: key numbers")
for depth in (-0.3, -2.0):
    for a in ACTS:
        f = CELLS[(a, depth, "frozen")]
        t = CELLS[(a, depth, "training")]
        print(f"  z={depth:+.1f} {a:7s} frozen {f[0]:2d}/{f[1]} p={f[2]:.3f} "
              f"[{f[3]:.3f},{f[4]:.3f}] | train {t[0]:2d}/{t[1]} p={t[2]:.3f} "
              f"[{t[3]:.3f},{t[4]:.3f}]")
for a in ACTS:
    v = DELTA[a]
    print(f"  Delta {a:7s} = {v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}] "
          f"(train {v[3]}/100 vs frozen {v[4]}/100)")
