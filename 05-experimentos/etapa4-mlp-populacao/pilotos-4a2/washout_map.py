"""Mapa do washout: separação softplus/semgrad (completos dur>=50, H=2e5 casado)
vs m em T=1e-5. m=16,128 (H2e5 nativo), m=64 (5e5 truncado a 2e5), m=256 (2e5)."""
from pathlib import Path
import gzip, json, os
import numpy as np

RES = str(Path(__file__).resolve().parent / 'results-piloto')
H = 200000
L = 50


def load(fn):
    p = os.path.join(RES, fn)
    return json.load(gzip.open(p, "rt")) if os.path.exists(p) else None


def counts(fn):
    d = load(fn)
    if d is None:
        return None
    nf = nc = 0
    comp_seed = 0
    for e in d["episodios"]:
        if e["t_res"] >= 0 and (e["t_res"] - e["t_morte"]) <= H:
            dur, comp = e["t_res"] - e["t_morte"], True
        else:
            base = e["t_res"] if e["t_res"] >= 0 else e.get("censurado_em", d["passos"])
            dur, comp = min(base, H) - e["t_morte"], False
        if e["t_morte"] >= H:
            continue
        if dur >= L:
            nf += 1
            nc += int(comp)
    return nf, nc


# fontes por m (nome do arquivo difere)
SRC = {
    16: "m16-T1e-05-H2e5",
    64: "m64-T1e-05",          # 5e5 nativo, truncado a 2e5
    128: "m128-T1e-05-H2e5",
    256: "m256-T1e-05-H2e5",
}

print("== MAPA DO WASHOUT: separação física/ruído vs m, T=1e-5, H=2e5 casado ==")
print(f"{'m':>5} {'sp_comp':>8} {'sg_comp':>8} {'razão':>7} {'sp/seed':>18} {'CV_sp':>6} {'veredito':>16}")
for m in (16, 64, 128, 256):
    suf = SRC[m]
    sp_seed, sg_seed = [], []
    for s in (45, 46, 47):
        rs = counts(f"softplus-{suf}-s{s}.json.gz")
        rg = counts(f"semgrad-{suf}-s{s}.json.gz")
        if rs:
            sp_seed.append(rs[1])
        if rg:
            sg_seed.append(rg[1])
    if not sp_seed or not sg_seed:
        print(f"{m:>5}  (sem dados)")
        continue
    sp, sg = sum(sp_seed), sum(sg_seed)
    ratio = sp / max(sg, 1)
    cv = np.std(sp_seed) / max(np.mean(sp_seed), 1e-9)
    vd = "FÍSICA" if ratio >= 2 else ("marginal" if ratio >= 1.5 else "RUÍDO (washout)")
    print(f"{m:>5} {sp:>8} {sg:>8} {ratio:>7.2f} {str(sp_seed):>18} {cv:>6.2f} {vd:>16}")
