"""Analise DE-CHATTER do piloto-de-taxa e5 (le rate-results/, nao re-roda).

MOTIVO: o motor usou z_dead=z_alive=0 (regua do etapa-4). No transformer a entrada
da FFN e NAO-ESTACIONARIA (h se move); uma unidade empurrada para zmax~=0 CRUZA o
limiar milhares de vezes (chatter), cada cruzamento contado como morte+volta. Isto
NAO e sinal — e artefato de contagem (só 8/20 seeds relu-train; demais bracos ~0).
Prova (relu-train-s103, unidade k=2): 1a travessia GENUINA em t=66846 (upstream,
own_move=0 exato) e depois 159 micro-oscilacoes em zmax=0.0000, z_park=-0.0007.

CORRECAO (uniforme a TODOS os bracos): COLAPSA por (cell,seed,unidade):
  - injetada?  -> flag do 1o episodio
  - classe-1a  -> 'born'/'dyn' do 1o episodio (dyn = morte DINAMICA t>0 genuina)
  - revived    -> qualquer episodio cruzou z_alive=0
  - t_res      -> MENOR t_res (1a travessia = return-time genuino)
  - z_park     -> min ; zmax_hi -> max (robustez da volta)
  - A_drive/t_area_drive/own_move/canal -> do 1o episodio revivido (ou do 1o episodio)
Isto conta cada UNIDADE uma vez => de-chattered. lambda_dyn = unidades nao-injetadas
com 1a-classe 'dyn' por run. p(volta|inj) = unidades injetadas que cruzaram 0.

Recomendacao herdada p/ confirmatorio: usar HISTERESE (z_dead=-d, z_alive=+d, d~0.05)
no instrumento p/ eliminar o chatter na origem.
"""
import json
import os
import glob
import collections
import numpy as np
from scipy.stats import beta

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rate-results")
Z_STAR = -0.7518
EPS_ROBUST = 0.05    # zmax_hi acima disto = volta ROBUSTA (vs hover-na-barreira)
ARMS = ["sp-b3", "sp-b10", "gelu", "relu"]
FACS = ["train", "frozen"]
SEEDS = list(range(100, 120))


def cp(k, n, a=0.05):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return k / n, lo, hi


def boot_rate(counts, B=5000, seed=0):
    c = np.asarray(counts, float)
    rng = np.random.default_rng(seed)
    pt = c.mean()
    out = [c[rng.integers(0, len(c), len(c))].mean() for _ in range(B)]
    return pt, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def collapse(cell):
    """Retorna lista de UNIDADES colapsadas de todas as seeds do cell."""
    units = []
    per_seed_dyn = []
    per_seed_born = []
    for s in SEEDS:
        fp = os.path.join(RES, f"{cell}-s{s}.json")
        if not os.path.exists(fp):
            continue
        d = json.load(open(fp))
        byk = collections.defaultdict(list)
        for e in d["episodios"]:
            byk[e["k"]].append(e)
        ndyn = nborn = 0
        for k, eps in byk.items():
            eps = sorted(eps, key=lambda e: (e["t_morte"] if e["t_morte"] >= 0 else 0))
            first = eps[0]
            revs = [e for e in eps if e["revived"]]
            rev = len(revs) > 0
            # 1o episodio revivido (return-time genuino) ou 1o episodio
            src = min(revs, key=lambda e: e["t_res"]) if rev else first
            t_res = min((e["t_res"] for e in revs), default=None)
            u = dict(
                cell=cell, seed=s, k=k, injetada=first["injetada"],
                injdepth=first["injdepth"], classe1=first["classe"],
                deep=(first["injdepth"] is not None and first["injdepth"] < Z_STAR)
                     if first["injetada"] else (min(e["z_park"] for e in eps) < Z_STAR),
                revived=rev, t_res=t_res,
                z_park=min(e["z_park"] for e in eps),
                zmax_hi=max(e["zmax_hi"] for e in eps),
                robust=(max(e["zmax_hi"] for e in eps) >= EPS_ROBUST),
                A_drive=src["A_drive_fim"], A_mlp=src["A_mlp_fim"],
                t_area_drive=src["t_area_drive"], own_move=src["own_move"],
                canal=src["canal"], n_eps=len(eps))
            units.append(u)
            if not first["injetada"]:
                if first["classe"] == "dyn":
                    ndyn += 1
                elif first["classe"] == "born":
                    nborn += 1
        per_seed_dyn.append(ndyn)
        per_seed_born.append(nborn)
    return units, per_seed_dyn, per_seed_born


def main():
    ALL = {}
    for arm in ARMS:
        for fac in FACS:
            ALL[(arm, fac)] = collapse(f"{arm}-{fac}")

    print("# Piloto-de-taxa e5 — analise DE-CHATTER (colapso por unidade)")
    print(f"# 160 runs, H=80000, kill_depths=(0.3,1.0) rasa/profunda, seeds 100-119, "
          f"barreira z*={Z_STAR}, EPS_ROBUST={EPS_ROBUST}\n")

    print("## 1. Tabela mestra colapsada (por braco x fator)\n")
    hdr = ("cell", "plato", "lam_dyn/run[CI]", "bornN/run", "injU", "voltas",
           "robustas", "cens", "p(volta|inj)[CP]", "own_med(volta)")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join("---" for _ in hdr) + "|")
    for arm in ARMS:
        for fac in FACS:
            units, dyn, born = ALL[(arm, fac)]
            inj = [u for u in units if u["injetada"]]
            rev = [u for u in inj if u["revived"]]
            rob = [u for u in rev if u["robust"]]
            lam, llo, lhi = boot_rate(dyn)
            bpt = np.mean(born)
            p, lo, hi = cp(len(rev), len(inj))
            own = np.median([u["own_move"] for u in rev]) if rev else float("nan")
            plato = np.mean([json.load(open(os.path.join(RES, f"{arm}-{fac}-s{s}.json")))["plato"]
                             for s in SEEDS])
            print(f"| {arm}-{fac} | {plato:.3f} | {lam:.2f}[{llo:.2f},{lhi:.2f}] | {bpt:.2f} | "
                  f"{len(inj)} | {len(rev)} | {len(rob)} | {len(inj)-len(rev)} | "
                  f"{p:.2f}[{lo:.2f},{hi:.2f}] | {own:.4f} |")
    print("\n_voltas = unidade injetada cruzou z_alive=0 (GT). robustas = zmax_hi>=0.05 "
          "(climb real, nao hover-na-barreira). cens = nao cruzou 0 em H. lam_dyn/bornN "
          "contam unidades NAO-injetadas (natural)._")

    print("\n## 2. p(volta|inj) e return-time por PROFUNDIDADE (rasa z=-0.3>z*, profunda z=-1.0<z*)\n")
    print("| cell | rasa volta/n [CP] | rasa robustas | rasa τ=q0.9 (med) | prof volta/n [CP] | prof robustas | prof τ=q0.9 |")
    print("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        for fac in FACS:
            units = [u for u in ALL[(arm, fac)][0] if u["injetada"]]
            def blk(deep):
                sub = [u for u in units if u["deep"] == deep]
                rv = [u for u in sub if u["revived"]]
                rob = sum(1 for u in rv if u["robust"])
                p, lo, hi = cp(len(rv), len(sub))
                ts = [u["t_res"] for u in rv if u["t_res"] is not None]
                tau = f"{int(np.quantile(ts,0.9))}(med={int(np.median(ts))})" if ts else "-"
                return f"{len(rv)}/{len(sub)} [{lo:.2f},{hi:.2f}]", str(rob), tau
            rp, rrob, rtau = blk(False)
            pp, prob, ptau = blk(True)
            print(f"| {arm}-{fac} | {rp} | {rrob} | {rtau} | {pp} | {prob} | {ptau} |")

    print("\n## 3. Return-time t_res (1a travessia GT) por braco — pool train+frozen, so voltas\n")
    print("| braco | n voltas | min | mediana | τ=q0.9 | max | n robustas |")
    print("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        units = [u for fac in FACS for u in ALL[(arm, fac)][0]
                 if u["injetada"] and u["revived"] and u["t_res"] is not None]
        ts = sorted(u["t_res"] for u in units)
        nrob = sum(1 for u in units if u["robust"])
        if ts:
            print(f"| {arm} | {len(ts)} | {ts[0]} | {int(np.median(ts))} | "
                  f"{int(np.quantile(ts,0.9))} | {ts[-1]} | {nrob} |")
        else:
            print(f"| {arm} | 0 | - | - | - | - | 0 |")

    print("\n## 4. Atribuicao de canal: drive-proprio (frozen) vs montante (train). "
          "P-iii/P-iv\n")
    print("| braco | frozen voltas (proprio) | frozen robustas | train voltas | train robustas | "
          "train own_med | Δ(train-frozen)=montante |")
    print("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        fz = [u for u in ALL[(arm, "frozen")][0] if u["injetada"] and u["revived"]]
        tr = [u for u in ALL[(arm, "train")][0] if u["injetada"] and u["revived"]]
        fzrob = sum(u["robust"] for u in fz); trrob = sum(u["robust"] for u in tr)
        ownt = np.median([u["own_move"] for u in tr]) if tr else float("nan")
        print(f"| {arm} | {len(fz)} | {fzrob} | {len(tr)} | {trrob} | {ownt:.4f} | "
              f"{len(tr)-len(fz):+d} |")

    print("\n## 5. GT-AREA: trios A_drive das VOLTAS (t_area_drive<t_res) + sinal drive vs mlp\n")
    print("| cell | seed | k | depth | z_park | zmax_hi | t_area_drive | t_res | A_drive | A_mlp | ordenado | robusta |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    nord = ndisc = ntot = nrob_area = 0
    for arm in ARMS:
        for fac in FACS:
            for u in ALL[(arm, fac)][0]:
                if u["injetada"] and u["revived"]:
                    ntot += 1
                    ordd = (u["t_area_drive"] >= 0 and u["t_area_drive"] <= (u["t_res"] or 1 << 30))
                    nord += int(ordd)
                    ndisc += int(u["A_drive"] * u["A_mlp"] < 0)
                    if u["robust"]:
                        nrob_area += int(u["t_area_drive"] >= 0)
                    print(f"| {arm}-{fac} | {u['seed']} | {u['k']} | {u['injdepth']} | "
                          f"{u['z_park']:.2f} | {u['zmax_hi']:.3f} | {u['t_area_drive']} | "
                          f"{u['t_res']} | {u['A_drive']:+.2e} | {u['A_mlp']:+.2e} | {ordd} | {u['robust']} |")
    print(f"\n  voltas totais={ntot}; A_drive aberta antes/na travessia (t_area_drive<=t_res)={nord}; "
          f"sinal(A_drive)!=sinal(A_mlp)={ndisc}")

    print("\n## 6. S1 sign-flip GELU: sinal A_drive vs A_mlp em TODOS episodios GELU (nao-colapsado)\n")
    for fac in FACS:
        eps = [e for s in SEEDS
               for e in json.load(open(os.path.join(RES, f"gelu-{fac}-s{s}.json")))["episodios"]]
        deep = [e for e in eps if (e["injdepth"] is not None and e["injdepth"] < Z_STAR)]
        rasa = [e for e in eps if (e["injdepth"] is not None and e["injdepth"] >= Z_STAR)]
        dd = sum(1 for e in deep if e["A_drive_fim"] * e["A_mlp_fim"] < 0)
        rr = sum(1 for e in rasa if e["A_drive_fim"] * e["A_mlp_fim"] < 0)
        print(f"  gelu-{fac}: profundos={len(deep)} opostos={dd} ({dd/max(len(deep),1):.0%}) | "
              f"rasos={len(rasa)} opostos={rr} ({rr/max(len(rasa),1):.0%})")


if __name__ == "__main__":
    main()
