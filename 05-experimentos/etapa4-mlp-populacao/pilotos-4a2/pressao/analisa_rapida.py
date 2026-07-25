"""Análise da grade RÁPIDA (etapa-4a.3), spec expandida (64 runs).

Tabelas: (1) por (m,ativ) — platô, born-dead vs dinâmicas, ressurreições em cada
classe, rev_real, v3 z_min<0; (2) LEI DE ÁREA — trio (t_morte, t_area[v2], t_res)
episódio a episódio p/ ressurreições, testando se o zero-de-área ANTECEDE o
retorno; (3) deriva prevista vs dur observada. Veredito (ii')(iii)(iv'). Zero
teatro; separa SEMPRE born-dead de dinâmico.
"""
import glob
import gzip
import json
import os

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(AQUI, "results")
ROTS = ["sp-b10", "sp-b30", "sp-b300", "relu"]
MS = [4, 6]


def carrega():
    runs = []
    for p in sorted(glob.glob(os.path.join(OUT, "*-rapida.jsonl.gz"))):
        runs.append(json.load(gzip.open(p, "rt")))
    return runs


def cel(runs, rot, m):
    return [r for r in runs if r["rot"] == rot and r["m"] == m]


def _S(rs, k):
    return sum(r[k] for r in rs)


def _med_dur(rs, classe):
    ds = [e["dur_obs"] for r in rs for e in r["episodios"]
          if e["classe"] == classe and e["revived"]]
    return (float(np.median(ds)) if ds else None), len(ds)


def main():
    runs = carrega()
    if not runs:
        print("sem resultados ainda.")
        return
    L = []
    L.append("# Grade rápida (etapa-4a.3) — tabela e veredito\n")
    L.append(f"_Runs: {len(runs)}/64. m∈{{4,6}} × {{β10,β30,β300,relu}} × seeds 42-49. "
             f"n=16, η=1e-3, T=0, H=2e5, alvo rico._\n")

    # ---- Tabela 1: por célula ----
    L.append("## Tabela 1 — por (m × ativ), agregado nos seeds\n")
    L.append("| ativ | m | seeds | loss_plato (min–max) | born(rev) | dyn(rev) | "
             "Σrev_real | Σv3 z_min<0 | med dur din |")
    L.append("|---|--:|--:|---|--:|--:|--:|--:|--:|")
    tab = {}
    for rot in ROTS:
        for m in MS:
            rs = cel(runs, rot, m)
            if not rs:
                continue
            lp = [r["loss_plato"] for r in rs]
            md, _ = _med_dur(rs, "dyn")
            tab[(rot, m)] = dict(
                rs=rs, ns=len(rs), lpmin=min(lp), lpmax=max(lp),
                born=_S(rs, "n_born_dead"), bornrev=_S(rs, "n_born_revividos"),
                dyn=_S(rs, "n_dinamicas"), dynrev=_S(rs, "n_dinamicas_revividas"),
                rev_real=_S(rs, "n_revive_real"),
                v3zn=_S(rs, "v3_n_completos_zmin_neg"), meddur=md)
            c = tab[(rot, m)]
            L.append(f"| {rot} | {m} | {len(rs)} | {min(lp):.3f}–{max(lp):.3f} | "
                     f"{c['born']}({c['bornrev']}) | {c['dyn']}({c['dynrev']}) | "
                     f"{c['rev_real']} | {c['v3zn']} | "
                     f"{md:.0f} |" if md is not None else
                     f"| {rot} | {m} | {len(rs)} | {min(lp):.3f}–{max(lp):.3f} | "
                     f"{c['born']}({c['bornrev']}) | {c['dyn']}({c['dynrev']}) | "
                     f"{c['rev_real']} | {c['v3zn']} | — |")
    L.append("")

    # ---- Tabela 2: LEI DE ÁREA (trio) p/ ressurreições ----
    L.append("## Tabela 2 — LEI DE ÁREA: trio (t_morte, t_area[v2], t_res) das "
             "ressurreições\n")
    L.append("_Testa se o zero-de-área ANTECEDE o retorno (t_area < t_res). "
             "gap = t_res − t_area. Se t_area=−1 (não abriu) mas voltou, ou "
             "t_area>t_res, é dado contra a leitura ingênua da lei._\n")
    L.append("| ativ | m | seed | classe | k | t_morte | t_area(v2) | t_res | "
             "gap=t_res−t_area | z_park(z_min) | dur/pred |")
    L.append("|---|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|")
    antecede = total_rev = area_abriu = 0
    for rot in ("sp-b10", "sp-b30", "sp-b300"):
        for m in MS:
            for r in cel(runs, rot, m):
                for e in r["episodios"]:
                    if not e["revived"]:
                        continue
                    total_rev += 1
                    v2 = e["v2"] or {}
                    ta = v2.get("t_area", -1)
                    tr = e["v3_t_res"]
                    gap = (tr - ta) if ta >= 0 else None
                    if ta >= 0:
                        area_abriu += 1
                        if ta < tr:
                            antecede += 1
                    pred = e.get("t_ret_pred")
                    ratio = (e["dur_obs"] / pred) if pred else None
                    L.append(
                        f"| {rot} | {m} | {r['seed']} | {e['classe']} | {e['k']} | "
                        f"{e['v3_t_morte']} | {ta} | {tr} | "
                        f"{gap if gap is not None else '—'} | {e['z_min']:.4f} | "
                        f"{ratio:.2f} |" if ratio is not None else
                        f"| {rot} | {m} | {r['seed']} | {e['classe']} | {e['k']} | "
                        f"{e['v3_t_morte']} | {ta} | {tr} | "
                        f"{gap if gap is not None else '—'} | {e['z_min']:.4f} | — |")
    L.append(f"\n_Ressurreições totais (softplus)={total_rev}; com t_area aberto="
             f"{area_abriu}; t_area ANTECEDE t_res em {antecede}/{area_abriu}._")
    L.append("")

    # ---- Veredito ----
    L.append("## Veredito (ii')(iii)(iv') — sem amaciar\n")

    # (ii') dynamic β10 revivals > 0 in >= half the seeds
    L.append("**(ii') ressurreições DINÂMICAS softplus-β10 > 0 em ≥ metade das seeds:**")
    for m in MS:
        rs = cel(runs, "sp-b10", m)
        seeds_com = sum(1 for r in rs if r["n_dinamicas_revividas"] > 0)
        L.append(f"- β10 m={m}: {seeds_com}/{len(rs)} seeds com ≥1 ressurreição "
                 f"dinâmica {'(≥metade OK)' if seeds_com*2 >= len(rs) and rs else '(< metade)'}.")
    b10dyn = _S([r for r in runs if r["rot"] == "sp-b10"], "n_dinamicas_revividas")
    L.append(f"- β10 total ressurreições dinâmicas={b10dyn}.")

    # (iii) relu zero everywhere
    relu_rev = _S([r for r in runs if r["rot"] == "relu"], "n_revive_real")
    relu_zn = _S([r for r in runs if r["rot"] == "relu"], "v3_n_completos_zmin_neg")
    relu_dr = _S([r for r in runs if r["rot"] == "relu"], "n_dinamicas_revividas")
    relu_br = _S([r for r in runs if r["rot"] == "relu"], "n_born_revividos")
    okiii = (relu_rev == 0 and relu_zn == 0 and relu_dr == 0 and relu_br == 0)
    L.append("\n**(iii) relu: ZERO ressurreição de morto-real em todas:**")
    L.append(f"- Σrev_real={relu_rev}, Σv3 z_min<0={relu_zn}, Σdin.rev={relu_dr}, "
             f"Σborn.rev={relu_br}. "
             f"{'CONFIRMA (limpo)' if okiii else '**RED FLAG — investigar ANTES de reportar**'}.")

    # (iv') rate ladder
    L.append("\n**(iv') taxa de ressurreição dinâmica: β10 ≥ β30 ≥ β300:**")
    counts = []
    for rot in ("sp-b10", "sp-b30", "sp-b300"):
        rr = [r for r in runs if r["rot"] == rot]
        dv = _S(rr, "n_dinamicas_revividas")
        dt = _S(rr, "n_dinamicas")
        rate = (dv / dt) if dt else None
        counts.append(dv)
        L.append(f"- {rot}: {dv} ress.dinâmicas / {dt} mortes dinâmicas = "
                 f"taxa {rate:.2f}" if rate is not None else
                 f"- {rot}: {dv} ress.dinâmicas / {dt} mortes dinâmicas = taxa —")
    mono = counts == sorted(counts, reverse=True)
    L.append(f"- contagens (β10,β30,β300)={counts}: "
             f"{'monotônica ≥ CONFIRMA' if mono else 'NÃO monotônica'} "
             "(exploratório).")

    txt = "\n".join(L) + "\n"
    with open(os.path.join(AQUI, "tabela-rapida.md"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
