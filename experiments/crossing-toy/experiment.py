"""crossing-toy: SGLD em L(w1,w2)=w1²w2² vs baseline nulo (L=0).

Mede as três predições do README (P-a pouso/deslize, P-b complementaridade
ocupacional, P-c deriva entrópica). Saída: results/*.jsonl + figuras PNG.
"""
import json
import os

import numpy as np

import config as C


def grad(w):
    w1, w2 = w[..., 0], w[..., 1]
    return np.stack([2 * w1 * w2**2, 2 * w1**2 * w2], axis=-1)


def roda(seed, T, nulo=False):
    rng = np.random.default_rng(seed)
    # init longe da cruz
    while True:
        w = rng.uniform(-C.INIT_BOX, C.INIT_BOX, 2)
        if np.all(np.abs(w) >= C.INIT_MIN):
            break
    traj = np.empty((C.N_PASSOS, 2), dtype=np.float32)
    ruido = rng.standard_normal((C.N_PASSOS, 2)) * np.sqrt(2 * C.ETA * T)
    for t in range(C.N_PASSOS):
        g = 0.0 if nulo else grad(w)
        w = w - C.ETA * g + ruido[t]
        traj[t] = w
    return traj


def metricas(traj):
    """d = distância ao estrato mais próximo; r_N = |componente do gradiente normal a ele|."""
    a = np.abs(traj)
    perto_w2 = a[:, 1] <= a[:, 0]        # estrato {w2=0} é o mais próximo
    d = np.where(perto_w2, a[:, 1], a[:, 0])
    g = np.abs(grad(traj))
    fn = np.where(perto_w2, g[:, 1], g[:, 0])
    # coordenada tangencial (posição ao longo do estrato ativo)
    tang = np.where(perto_w2, traj[:, 0], traj[:, 1])
    return d, fn, tang


def main():
    os.makedirs("results", exist_ok=True)
    out = open("results/metricas.jsonl", "w")
    heat = {}
    for T in C.TEMPERATURAS:
        for nulo in (False, True):
            key = f"T={T}_nulo={nulo}"
            occ = np.zeros((C.BINS_OCUPACAO, C.BINS_OCUPACAO))
            for seed in C.SEEDS:
                traj = roda(seed, T, nulo)
                d, fn, tang = metricas(traj)
                contato = d < C.DELTA_CONTATO
                # P-a: primeiro pouso e fração de tempo em contato depois dele
                idx = np.flatnonzero(contato[: C.JANELA_POUSO[1]])
                t_pouso = int(idx[0]) if idx.size else None
                frac_pos = float(contato[t_pouso:].mean()) if t_pouso is not None else 0.0
                # P-c: deriva de <tang²> na janela, só em contato
                j0, j1 = C.JANELA_DERIVA
                sel = contato[j0:j1]
                tt = tang[j0:j1][sel] ** 2
                ts = np.flatnonzero(sel)
                deriva = float(np.polyfit(ts, tt, 1)[0]) if tt.size > 100 else None
                out.write(json.dumps({
                    "T": T, "nulo": nulo, "seed": seed, "t_pouso": t_pouso,
                    "frac_contato_pos_pouso": frac_pos, "deriva_tang2_por_passo": deriva,
                    "tang2_inicio": float(tt[:1000].mean()) if tt.size > 1000 else None,
                    "tang2_fim": float(tt[-1000:].mean()) if tt.size > 1000 else None,
                }) + "\n")
                # P-b: ocupação (log d, log r_N)
                ld = np.log10(np.clip(d, 1e-8, None))
                lf = np.log10(np.clip(fn, 1e-10, None))
                h, _, _ = np.histogram2d(ld, lf, bins=C.BINS_OCUPACAO,
                                         range=[[-8, 1], [-10, 1]])
                occ += h
            heat[key] = occ
            print(f"[ok] {key}")
    out.close()
    np.savez("results/ocupacao.npz", **heat)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(heat)
        fig, axs = plt.subplots(2, n // 2, figsize=(4 * n // 2, 8), squeeze=False)
        for ax, (k, h) in zip(axs.flat, heat.items()):
            ax.imshow(np.log1p(h).T, origin="lower", extent=[-8, 1, -10, 1], aspect="auto")
            ax.set_title(k, fontsize=9)
            ax.set_xlabel("log10 d")
            ax.set_ylabel("log10 r_N")
        fig.tight_layout()
        fig.savefig("results/ocupacao.png", dpi=150)
        print("[ok] results/ocupacao.png")
    except ImportError:
        print("[aviso] matplotlib ausente; heatmaps só em ocupacao.npz")


if __name__ == "__main__":
    main()
