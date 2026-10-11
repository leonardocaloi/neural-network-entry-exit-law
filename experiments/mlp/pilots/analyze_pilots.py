"""Analisa pilot-results/. Por (braço, T): n_ep, físicos (>=50), completos,
censura, med/p90 duração, e físicos-por-seed (base p/ projetar a célula de 20
seeds da 4a.2). Separação de instrumento: razão físicos softplus / semgrad no
mesmo T — o debounce só serve se o nulo (difusão pura) produzir MUITO menos.
"""
import glob
import gzip
import json
import os

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(AQUI, "pilot-results")
PISO = 50


def dur(ep):
    if ep["t_res"] >= 0:
        return ep["t_res"] - ep["t_morte"], False
    return ep.get("censurado_em", ep.get("passos", 500000)) - ep["t_morte"], True


def carrega():
    por = {}
    for arq in sorted(glob.glob(os.path.join(RES, "*.json.gz"))):
        d = json.load(gzip.open(arq, "rt"))
        por.setdefault((d["braco"], d["T"]), []).append(d)
    return por


def resume(runs):
    fis_por_seed, comp_por_seed = [], []
    durs_fis, cens_fis, n_ep_tot = [], 0, 0
    for d in runs:
        nfis = ncomp = 0
        for e in d["episodios"]:
            n_ep_tot += 1
            dd, cens = dur(e)
            if dd >= PISO:
                nfis += 1
                durs_fis.append(dd)
                cens_fis += int(cens)
                ncomp += int(not cens)
        fis_por_seed.append(nfis)
        comp_por_seed.append(ncomp)
    df = np.array(durs_fis, float)
    n_fis = int(df.size)
    return dict(
        n_runs=len(runs), n_ep=n_ep_tot, n_fis=n_fis,
        n_comp=int(n_fis - cens_fis), n_cens=int(cens_fis),
        censura_fis=round(cens_fis / n_fis, 3) if n_fis else None,
        med_dur=round(float(np.median(df)), 0) if n_fis else None,
        p90_dur=round(float(np.percentile(df, 90)), 0) if n_fis else None,
        fis_por_seed=fis_por_seed, comp_por_seed=comp_por_seed,
        fis_por_seed_med=round(float(np.median(fis_por_seed)), 1),
        comp_por_seed_med=round(float(np.median(comp_por_seed)), 1),
    )


def main():
    por = carrega()
    tab = {f"{b}-T{T:.0e}": resume(runs) for (b, T), runs in sorted(por.items())}
    print(f"{'braco-T':18s} {'runs':>4s} {'n_ep':>6s} {'n>=50':>6s} {'comp':>5s} "
          f"{'cens':>5s} {'cens%':>6s} {'med_dur':>8s} {'p90_dur':>8s} "
          f"{'fis/seed':>9s} {'comp/seed':>9s}")
    print("-" * 108)
    for k, r in tab.items():
        cp = "" if r["censura_fis"] is None else f"{100*r['censura_fis']:5.1f}"
        print(f"{k:18s} {r['n_runs']:4d} {r['n_ep']:6d} {r['n_fis']:6d} "
              f"{r['n_comp']:5d} {r['n_cens']:5d} {cp:>6s} "
              f"{str(r['med_dur']):>8s} {str(r['p90_dur']):>8s} "
              f"{str(r['fis_por_seed']):>9s} {str(r['comp_por_seed']):>9s}")
    # separação de instrumento: softplus vs semgrad no mesmo T
    print("\n-- separação de instrumento (físicos>=50 softplus / semgrad, mesmo T) --")
    for T in (1e-5, 1e-4):
        sp = tab.get(f"softplus-T{T:.0e}"); sg = tab.get(f"semgrad-T{T:.0e}")
        if sp and sg:
            r_ep = sp["n_fis"] / max(sg["n_fis"], 1)
            r_comp = sp["n_comp"] / max(sg["n_comp"], 1)
            print(f"  T={T:.0e}: softplus {sp['n_fis']} vs semgrad {sg['n_fis']} "
                  f"físicos (razão {r_ep:.2f}); completos {sp['n_comp']} vs "
                  f"{sg['n_comp']} (razão {r_comp:.2f}) "
                  f"{'-> SEPARA' if r_comp>=3 else '-> NAO separa (ruido domina)'}")
    # projeção 4a.2: físicos/seed * 20 seeds
    print("\n-- projeção p/ célula de 20 seeds (físicos>=50 e completos) --")
    for k, r in tab.items():
        if k.startswith("softplus") or k.startswith("T0"):
            proj_fis = r["fis_por_seed_med"] * 20
            proj_comp = r["comp_por_seed_med"] * 20
            ok = proj_comp >= 30 and (r["censura_fis"] or 1) < 0.5
            print(f"  {k}: ~{proj_fis:.0f} físicos, ~{proj_comp:.0f} completos "
                  f"/ célula×20 (censura {100*(r['censura_fis'] or 0):.0f}%) "
                  f"{'OK' if ok else 'insuf.'}")
    json.dump(tab, open(os.path.join(AQUI, "pilot_analysis.json"), "w"), indent=1)
    print("\nescrito pilot_analysis.json")


if __name__ == "__main__":
    main()
