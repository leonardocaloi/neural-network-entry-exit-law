"""Trilho-3 (GRÁTIS): minera os results/*.json.gz EXISTENTES da 4a para
dimensionar o regime T/horizonte da 4a.2. NÃO toca em nada pré-registrado.

Para cada (braço, m) agrega sobre as 20 seeds (1700-1719):
  - n_ep total
  - n_fis = episódios com DURAÇÃO >= 50 passos (proxy de episódio físico
    pós-debounce; chatter = <=5 passos). Duração:
        completo:  t_res - t_morte
        censurado: censurado_em - t_morte  (lower bound, right-censored)
  - dos físicos: quantos completos (dão t_res) vs censurados
  - censura = n_censurado_fis / n_fis  (o que importa p/ a confirmatória)
  - mediana/p90 da duração dos físicos (censurado usa o lower bound)
  - fração de episódios >=50 passos sobre o total (chatter vs físico)
"""
import glob
import gzip
import json
import os

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(AQUI, "..", "results")
PISO_FIS = 50

BRACOS_T = [("T0", 0.0), ("Tmenor", 3e-4), ("softplus", 1e-3), ("Tmaior", 3e-3)]
# nulos p/ testar se o debounce >=50 separa fisica de ruido (todos T=1e-3):
BRACOS_NULO = [("relu", 1e-3), ("semgrad", 1e-3), ("erf", 1e-3), ("gelu", 1e-3)]
MS = [16, 64, 256]


def dur(ep):
    if ep["t_res"] >= 0:
        return ep["t_res"] - ep["t_morte"], False       # completo
    return ep.get("censurado_em", 100000) - ep["t_morte"], True  # censurado


def carrega_celula(braco, m):
    eps = []
    for arq in glob.glob(os.path.join(RES, f"{braco}-m{m}-s*.json.gz")):
        d = json.load(gzip.open(arq, "rt"))
        eps.extend(d["episodios"])
    return eps


def resume(eps):
    durs = np.array([dur(e)[0] for e in eps], float)
    cens = np.array([dur(e)[1] for e in eps], bool)
    fis = durs >= PISO_FIS
    n = len(eps)
    n_fis = int(fis.sum())
    n_fis_cens = int((fis & cens).sum())
    n_fis_comp = int((fis & ~cens).sum())
    df = durs[fis]
    return dict(
        n_ep=n,
        n_fis=n_fis,
        n_fis_comp=n_fis_comp,
        n_fis_cens=n_fis_cens,
        censura_fis=(n_fis_cens / n_fis) if n_fis else None,
        frac_fis=(n_fis / n) if n else None,
        med_dur_fis=float(np.median(df)) if n_fis else None,
        p90_dur_fis=float(np.percentile(df, 90)) if n_fis else None,
        med_dur_todos=float(np.median(durs)) if n else None,
        censura_total=float(cens.mean()) if n else None,
    )


def main():
    print(f"{'braco':9s} {'T':>7s} {'m':>4s} | {'n_ep':>6s} {'n>=50':>6s} "
          f"{'comp':>5s} {'cens':>5s} {'cens%':>6s} | {'med_dur':>8s} "
          f"{'p90_dur':>9s} | {'frac>=50':>8s} {'med_all':>8s} {'cens_all%':>9s}")
    print("-" * 118)
    tabela = {}
    for braco, T in BRACOS_T + BRACOS_NULO:
        for m in MS:
            eps = carrega_celula(braco, m)
            if not eps:
                print(f"{braco:9s} {T:7.0e} {m:4d} |  SEM DADOS")
                continue
            r = resume(eps)
            tabela[(braco, m)] = r
            cp = "" if r["censura_fis"] is None else f"{100*r['censura_fis']:5.1f}"
            md = "" if r["med_dur_fis"] is None else f"{r['med_dur_fis']:8.0f}"
            p9 = "" if r["p90_dur_fis"] is None else f"{r['p90_dur_fis']:9.0f}"
            print(f"{braco:9s} {T:7.0e} {m:4d} | {r['n_ep']:6d} {r['n_fis']:6d} "
                  f"{r['n_fis_comp']:5d} {r['n_fis_cens']:5d} {cp:>6s} | {md:>8s} "
                  f"{p9:>9s} | {100*r['frac_fis']:7.1f}% {r['med_dur_todos']:8.0f} "
                  f"{100*r['censura_total']:8.1f}%")
        if braco == "Tmaior":
            print("-" * 118 + "  <<< abaixo: NULOS (debounce >=50 deve MATAR o ruido)")
    json.dump({f"{b}-m{m}": tabela[(b, m)] for (b, m) in tabela},
              open(os.path.join(AQUI, "existing_run_analysis.json"), "w"), indent=1)
    print("\nescrito existing_run_analysis.json")


if __name__ == "__main__":
    main()
