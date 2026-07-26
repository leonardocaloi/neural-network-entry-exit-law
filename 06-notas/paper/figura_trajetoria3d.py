"""Fig. 5 — the 3D trajectory of ONE REAL resurrection in the MLP (etapa-4a.3).

Cell (the discovery cell of the fast grid, `pilotos-4a2/pressao/roda_rapida.py`):
    m=4, softplus beta=10, n=16 rich data, eta=1e-3, T=0 (pure GD), H=2e5, seed 42.
Unit k=1 dies at t=781 and returns at t=139179; the v2 ruler puts the area zero
at t=45098 (record: pressao/results/sp-b10-m4-s42-rapida.jsonl.gz, grid-rapida.log).

This script RE-RUNS that exact trajectory (same dynamics as
`treina_pressao.treina_pressao`, verified against the committed trio) and records
per step the leader margin z_k(t) and the ruler's accumulated area A_k(t), which
the production runs never stored as a time series. Nothing is fabricated: the run
aborts if the reproduced trio does not match the committed one.

Axes: t (training step), z_1(t) (leader margin), A_1(t) (accumulated hybrid area).
Outputs fig5-trajectory3d.{pdf,png} plus a cached .npz of the series.

Run:  nice -19 python figura_trajetoria3d.py [--force]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
ETAPA4 = RAIZ / "05-experimentos" / "etapa4-mlp-populacao"
PRESSAO = ETAPA4 / "pilotos-4a2" / "pressao"
CACHE = AQUI / "fig5-trajectory3d-serie.npz"

# célula da descoberta (idêntica a roda_rapida.py, tarefa sp-b10 / m=4 / s=42)
M, BETA, SEED, N_DADOS, ETA, T, PASSOS, K = 4, 10.0, 42, 16, 1e-3, 0.0, 200_000, 1
# trio COMMITADO (régua v2) que esta reprodução tem de bater exatamente
TRIO_REF = dict(t_morte=781, t_area=45098, t_sinal=33905, t_res=139179)


def roda_e_grava():
    """Reproduz a run e grava a série por passo de (z_k, A_k, R_k) do neurônio K."""
    sys.path.insert(0, str(ETAPA4))
    sys.path.insert(0, str(PRESSAO))
    import torch
    torch.set_num_threads(1)
    try:
        os.nice(19)
    except Exception:
        pass
    from harness_mlp import MLP, Regua
    from regua_v3 import ReguaV3
    from treina_pressao import dados_rico

    x, y = dados_rico(n=N_DADOS, seed=SEED)
    net = MLP(M, "softplus", BETA, SEED)
    reg2, reg3 = Regua(M, ETA, T), ReguaV3(M, ETA, T, n_sigma=10.0)
    x_np = x.numpy()
    g = torch.Generator().manual_seed(SEED + 1)
    sig = float(np.sqrt(2 * ETA * T))                       # = 0 em T=0

    z_ser = np.empty(PASSOS, np.float64)                    # margem-líder de K
    A_ser = np.empty(PASSOS, np.float64)                    # área acumulada Â_K
    R_ser = np.empty(PASSOS, np.float64)                    # R̂_K (integrando)
    loss_ser = np.empty(PASSOS, np.float64)
    idx = np.arange(M)
    for t in range(PASSOS):
        z = net.margens(x)
        a = net.act(z)
        res = (a @ net.v) - y[:, 0]
        loss = 0.5 * (res ** 2).mean()
        z_np = z.detach().numpy()
        a_np = a.detach().numpy()
        res_np = res.detach().numpy()
        v_np = net.v.detach().numpy()
        # régua idêntica à da produção, no MESMO estado pré-update
        reg2.observa(t, z_np, a_np, res_np, v_np, x_np)
        reg3.observa(t, z_np, a_np, res_np, v_np, x_np)
        jstar = z_np.argmax(0)
        R_hat = v_np * (-res_np[jstar] + v_np * a_np[jstar, idx])
        z_ser[t] = z_np.max(0)[K]
        A_ser[t] = reg2.A_hat[K]
        R_ser[t] = R_hat[K]
        loss_ser[t] = float(loss.detach())
        net.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in net.parameters():
                ruido = torch.randn(p.shape, generator=g)
                p += sig * ruido - ETA * p.grad
        if t % 20000 == 0:
            print(f"  t={t} z={z_ser[t]:+.4f} A={A_ser[t]:+.3e} loss={loss_ser[t]:.4f}",
                  flush=True)

    eps = list(reg2.episodios) + reg2.censurados(PASSOS)
    ep = [e for e in eps if e["k"] == K and e["t_morte"] > 0]
    assert len(ep) == 1, f"esperado 1 episódio dinâmico de k={K}, achei {len(ep)}"
    ep = ep[0]
    obtido = {c: int(ep[c]) for c in ("t_morte", "t_area", "t_sinal", "t_res")}
    print(f"trio reproduzido (v2, k={K}): {obtido}")
    print(f"trio commitado             : {TRIO_REF}")
    if obtido != TRIO_REF:
        raise SystemExit("ABORTA: a reprodução NÃO bate o trio commitado — "
                         "não se plota dado que não reproduz.")
    np.savez_compressed(CACHE, z=z_ser.astype(np.float32),
                        A=A_ser.astype(np.float32), R=R_ser.astype(np.float32),
                        loss=loss_ser.astype(np.float32),
                        trio=np.array([obtido["t_morte"], obtido["t_area"],
                                       obtido["t_sinal"], obtido["t_res"]]))
    print(f"série gravada em {CACHE}")


def plota():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "no-latex"])
    except Exception:
        pass
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "legend.fontsize": 6})

    d = np.load(CACHE)
    z, A = d["z"].astype(float), d["A"].astype(float)
    t_morte, t_area, t_sinal, t_res = (int(v) for v in d["trio"])
    n = len(z)
    t = np.arange(n)

    # subamostragem para o PDF (curva de 2e5 pontos é ilegível e pesada);
    # os MARCOS são plotados no passo exato, sem subamostrar.
    passo = 40
    sel = np.unique(np.concatenate([np.arange(0, n, passo),
                                    [t_morte, t_sinal, t_area, t_res, n - 1]]))
    ts, zs, As = t[sel] / 1e5, z[sel], A[sel]

    C_ALIVE, C_DEBT, C_CRED, C_BACK = "#26547C", "#EF476F", "#F4A259", "#06D6A0"
    fig = plt.figure(figsize=(4.3, 3.5))
    ax = fig.add_subplot(111, projection="3d")

    tramos = [(0, t_morte, C_ALIVE, "alive (dive)"),
              (t_morte, t_area, C_DEBT, r"dead, debt $\mathcal{A}<0$"),
              (t_area, t_res, C_CRED, r"dead, credit $\mathcal{A}>0$"),
              (t_res, n - 1, C_BACK, r"returned ($\mathcal{A}$ frozen)")]

    z_lo, z_hi = z.min() - 0.10, z.max() + 0.06
    A_span = A.max() - A.min()
    A_lo, A_hi = A.min() - 0.16 * A_span, A.max() + 0.06 * A_span

    for t0, t1, cor, rot in tramos:
        m = (sel >= t0) & (sel <= t1)
        # paredes: projeções 2D (z×t no piso, A×t na parede de fundo)
        ax.plot(ts[m], zs[m], np.full(m.sum(), A_lo), lw=0.8, color=cor,
                alpha=0.40, zorder=1)
        ax.plot(ts[m], np.full(m.sum(), z_hi), As[m], lw=0.8, color=cor,
                alpha=0.40, zorder=1)
        ax.plot(ts[m], zs[m], As[m], lw=1.3, color=cor, label=rot, zorder=4)
    # planos de referência z=0 (morte) e A=0 (portão), nas respectivas paredes
    tt = np.array([0.0, (n - 1) / 1e5])
    ax.plot(tt, [0, 0], [A_lo, A_lo], color="k", lw=0.5, ls=":", alpha=0.55)
    ax.plot(tt, [z_hi, z_hi], [0, 0], color="k", lw=0.5, ls=":", alpha=0.55)

    # marcos: ponto no passo EXATO + rótulo deslocado à mão (3D não tem offset px)
    marcos = [
        (t_morte, r"$t_{\rm death}$", C_DEBT, "o", 20, (0.055, 0.10, 1.7)),
        (t_sinal, "", "0.35", "v", 14, (0.0, 0.0, 0.0)),   # rotulado no zoom
        (t_area, r"$\mathcal{A}_1=0$", C_CRED, "o", 20, (0.02, 0.15, -2.6)),
        (t_res, r"$t_{\rm return}$", C_BACK, "o", 20, (-0.30, 0.05, 2.0)),
    ]
    for tm, rot, cor, mk, s, (dx, dy, dA) in marcos:
        ax.scatter([tm / 1e5], [z[tm]], [A[tm]], s=s, color=cor, marker=mk,
                   edgecolor="k", linewidth=0.4, depthshade=False, zorder=6)
        if rot:
            ax.text(tm / 1e5 + dx, z[tm] + dy, A[tm] + dA, rot, color=cor,
                    fontsize=6.5, zorder=7)
    # a distância entre portão e volta é o conteúdo de C2 (área abre ANTES)
    ax.plot([t_area / 1e5, t_res / 1e5], [z_hi, z_hi], [0, 0], color="0.35",
            lw=0.8, ls="-", alpha=0.8, zorder=2)
    ax.text((t_area + t_res) / 2e5, z_hi, -3.1,
            "gate opens $%s$ steps\nbefore the return" % f"{t_res - t_area:,}".replace(",", r"\,"),
            color="0.35", fontsize=6, ha="center", zorder=7)

    ax.set_xlim(0, (n - 1) / 1e5)
    ax.set_ylim(z_lo, z_hi)
    ax.set_zlim(A_lo, A_hi)
    ax.set_xlabel(r"training step $t$ ($\times10^{5}$)", labelpad=-2)
    ax.set_ylabel(r"leader margin $z_1(t)$", labelpad=-2)
    ax.set_zlabel(r"accumulated area $\mathcal{A}_1(t)$", labelpad=-3)
    ax.tick_params(axis="x", pad=-2.5)
    ax.tick_params(axis="y", pad=-1.5)
    ax.tick_params(axis="z", pad=0.5)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_yticks([-0.2, 0.0, 0.4, 0.8])
    ax.set_zticks([0, 5, 10, 15])
    ax.view_init(elev=19, azim=-56)
    try:
        ax.set_box_aspect((1.55, 1.0, 0.92), zoom=1.03)
    except TypeError:
        pass
    for eixo in (ax.xaxis, ax.yaxis, ax.zaxis):
        eixo.pane.set_alpha(0.04)
        eixo._axinfo["grid"]["linewidth"] = 0.25
    ax.legend(loc="upper left", bbox_to_anchor=(-0.03, 0.615), frameon=False,
              handlelength=1.3, borderaxespad=0.0, labelspacing=0.22,
              fontsize=6)

    # zoom 2D da fase de dívida: a lei vive perto de A=0, o crédito depois é grande
    ins = fig.add_axes([0.055, 0.665, 0.25, 0.215])
    lim = int(t_area * 1.22)
    tz = np.arange(0, lim, 20)
    ins.plot(tz / 1e4, A[tz], lw=0.9, color=C_DEBT)
    mcred = tz >= t_area
    ins.plot(tz[mcred] / 1e4, A[tz][mcred], lw=0.9, color=C_CRED)
    ins.axhline(0, color="k", lw=0.5, ls=":", alpha=0.7)
    ins.scatter([t_sinal / 1e4], [A[t_sinal]], s=8, marker="v", color="0.35",
                zorder=5)
    ins.annotate(r"$\hat R_1$ flips", (t_sinal / 1e4, A[t_sinal]),
                 textcoords="offset points", xytext=(-2, 5), fontsize=5,
                 color="0.35", ha="right")
    ins.scatter([t_area / 1e4], [0], s=10, color=C_CRED, ec="k", lw=0.35,
                zorder=5)
    ins.set_ylabel(r"$\mathcal{A}_1$", fontsize=5.5, labelpad=0.5)
    ins.tick_params(labelsize=5, length=1.8, pad=1)
    ins.set_title(r"zoom: debt repaid  ($t/10^{4}$)", fontsize=5.5, pad=1.5)

    ax.set_title(r"One real resurrection: MLP $m=4$, softplus $\beta=10$, "
                 r"$T=0$, seed 42", fontsize=7.5, y=0.985)
    fig.subplots_adjust(left=0.0, right=0.995, bottom=0.015, top=0.995)
    for ext in ("pdf", "png"):
        fig.savefig(AQUI / f"fig5-trajectory3d.{ext}", dpi=400)
    print("fig5-trajectory3d.{pdf,png} escritas")
    print(f"  t_death={t_morte} z={z[t_morte]:+.4f} | "
          f"R flips t={t_sinal} A_min={A[t_sinal]:+.4e} | "
          f"A=0 t={t_area} | t_return={t_res} z={z[t_res]:+.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-roda a simulação")
    ap.add_argument("--so-plota", action="store_true")
    a = ap.parse_args()
    if not a.so_plota and (a.force or not CACHE.exists()):
        roda_e_grava()
    plota()
