"""Célula recomendada m64 T=1e-5 a H=2e5: censura, med/p90, projeção 30 seeds,
e teste de robustez (a física sobrevive sem o seed dominante?).
Usa os dados m64 nativos (5e5) truncados corretamente a H=2e5 (só episódios com
t_morte<H; ressuscita só se t_res-t_morte<=H)."""
from pathlib import Path
import gzip, json, os
import numpy as np

RES = str(Path(__file__).resolve().parent / 'pilot-results')
H, L = 200000, 50


def cell(braco, suf, seeds=(45, 46, 47)):
    per_fis, per_comp, comp_durs = [], [], []
    for s in seeds:
        p = os.path.join(RES, f"{braco}-{suf}-s{s}.json.gz")
        if not os.path.exists(p):
            per_fis.append(0); per_comp.append(0); continue
        d = json.load(gzip.open(p, "rt"))
        nf = nc = 0
        for e in d["episodios"]:
            if e["t_morte"] >= H:
                continue
            if e["t_res"] >= 0 and (e["t_res"] - e["t_morte"]) <= H:
                dur, comp = e["t_res"] - e["t_morte"], True
            else:
                base = e["t_res"] if e["t_res"] >= 0 else e.get("censurado_em", d["passos"])
                dur, comp = min(base, H) - e["t_morte"], False
            if dur >= L:
                nf += 1
                if comp:
                    nc += 1; comp_durs.append(dur)
        per_fis.append(nf); per_comp.append(nc)
    return per_fis, per_comp, np.array(comp_durs, float)


for braco, suf in [("softplus", "m64-T1e-05"), ("semgrad", "m64-T1e-05")]:
    pf, pc, cd = cell(braco, suf)
    fis, comp = sum(pf), sum(pc)
    cens = (fis - comp) / fis if fis else None
    print(f"{braco:9s} m64 T1e-5 H2e5: fis={fis} comp={comp} "
          f"censura={100*cens:.0f}%  fis/seed={pf} comp/seed={pc}")
    if cd.size:
        print(f"            dur completos: med={np.median(cd):.0f} "
              f"p75={np.percentile(cd,75):.0f} p90={np.percentile(cd,90):.0f} "
              f"p99={np.percentile(cd,99):.0f}")

# robustez: física sem o seed dominante (45)
pf, pc, _ = cell("softplus", "m64-T1e-05")
pfg, pcg, _ = cell("semgrad", "m64-T1e-05")
print("\n-- robustez da separação por seed (softplus vs semgrad, completos) --")
for i, s in enumerate((45, 46, 47)):
    r = pc[i] / max(pcg[i], 1)
    print(f"  seed {s}: softplus {pc[i]} vs semgrad {pcg[i]} -> {r:.2f}x")
print(f"  SEM seed 45: softplus {sum(pc[1:])} vs semgrad {sum(pcg[1:])} "
      f"-> {sum(pc[1:])/max(sum(pcg[1:]),1):.2f}x")

# projeção 30 seeds
print("\n-- projeção célula m64 T1e-5 H2e5 x 30 seeds --")
print(f"  ~{sum(pc)/3*30:.0f} completos, ~{sum(pf)/3*30:.0f} físicos>=50, "
      f"censura ~{100*(sum(pf)-sum(pc))/max(sum(pf),1):.0f}%")
