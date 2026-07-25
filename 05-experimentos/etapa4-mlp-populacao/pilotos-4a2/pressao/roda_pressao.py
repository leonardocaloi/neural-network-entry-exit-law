"""Piloto EXPLORATÓRIO etapa-4a.3 "pressão" — runner.

HIPÓTESE [ESPECULAÇÃO da casa]: a ressurreição-por-área (T2') exige RESÍDUO
PERSISTENTE. Rede SUBPARAMETRIZADA (m pequeno vs alvo rico) a T=0 (zero ruído
⟹ zero chatter ⟹ qualquer ressurreição é MECANISMO puro):
  (i)   loss platôia em valor > 0 (pressão persiste);
  (ii)  softplus tem ressurreições genuínas (>0, duração física);
  (iii) relu tem EXATAMENTE 0 ressurreições de neurônio morto-em-todos-os-dados
        (grad≡0 ⟹ freeze determinístico; qualquer violação = bug, investigar);
  (iv)  β menor ⟹ mais ressurreições (fome de gradiente; β∈{30,300}).
O QUE MATA: subparam. a T=0, softplus com ressurreições ≈0 mesmo com loss-platô
alto e mortes ocorrendo ⟹ pressão não basta / mecanismo não opera em MLP.

Grade: m∈{4,8,16} × ativ∈{softplus-β300, softplus-β30, relu} × T=0 × η=3e-4
       × H=3e5 × seeds 42-45 = 36 runs. Pool 2, nice-19, torch 1 thread.
Saída: results/{ativ}-m{m}-s{seed}.jsonl.gz (episódios + ground-truth por run)
       e agregado results/pressao.jsonl (1 linha/run, sem episódios).
"""
import gzip
import json
import os
import time
from multiprocessing import Pool

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
ETAPA4 = os.path.dirname(os.path.dirname(AQUI))       # dir com harness_mlp/regua_v3
OUT = os.path.join(AQUI, "results")
os.makedirs(OUT, exist_ok=True)

PASSOS = 300_000
ETA = 3e-4
T = 0.0
SEEDS = [42, 43, 44, 45]
N_SIGMA = 10.0

# (rótulo, ativ, beta)
ATIVS = [
    ("softplus-b300", "softplus", 300.0),
    ("softplus-b30", "softplus", 30.0),
    ("relu", "relu", 300.0),           # beta ignorado no relu
]
MS = [4, 8, 16]

TAREFAS = []
for rot, ativ, beta in ATIVS:
    for m in MS:
        for s in SEEDS:
            TAREFAS.append(dict(rot=rot, ativ=ativ, beta=beta, m=m, seed=s))


def _nome(t):
    return f"{t['rot']}-m{t['m']}-s{t['seed']}.jsonl.gz"


def _dur(ep):
    return int(ep["t_res"] - ep["t_morte"]) if ep["t_res"] >= 0 else None


def _stats(durs):
    durs = [d for d in durs if d is not None]
    if not durs:
        return dict(n=0, med=None, p90=None)
    a = np.array(durs, float)
    return dict(n=len(durs), med=float(np.median(a)), p90=float(np.percentile(a, 90)))


def roda(t):
    import sys
    sys.path.insert(0, ETAPA4)
    sys.path.insert(0, AQUI)
    import torch
    torch.set_num_threads(1)
    try:
        os.nice(19)
    except Exception:
        pass
    import treina_pressao as tp
    caminho = os.path.join(OUT, _nome(t))
    if os.path.exists(caminho):
        return f"skip {_nome(t)}"
    t0 = time.time()
    net, reg2, reg3, metr = tp.treina_pressao(
        m=t["m"], ativ=t["ativ"], beta=t["beta"], eta=ETA, T=T,
        passos=PASSOS, seed=t["seed"], n_sigma=N_SIGMA)

    lh = metr["loss_hist"]
    n10 = max(1, len(lh) // 10)
    loss_plato = float(np.mean(lh[-n10:]))

    v2_comp = list(reg2.episodios)
    v2_cens = reg2.censurados(PASSOS)
    v3_comp = list(reg3.episodios)
    v3_cens = reg3.censurados(PASSOS)

    v3_zmin_comp = [ep["z_min"] for ep in v3_comp]
    v3_comp_zmin_neg = [ep for ep in v3_comp if ep["z_min"] < 0.0]
    v3_comp_profundo = [ep for ep in v3_comp if ep.get("profundo")]

    rec = dict(
        rot=t["rot"], ativ=t["ativ"], beta=t["beta"], m=t["m"], seed=t["seed"],
        T=T, eta=ETA, passos=PASSOS, n_sigma=N_SIGMA,
        loss_inicial=float(lh[0]), loss_final=float(lh[-1]), loss_plato=loss_plato,
        n_mortos_real=metr["n_mortos_real"],
        n_revive_real=metr["n_revive_real"],
        n_neuronios_revive_real=metr["n_neuronios_revive_real"],
        # v2
        v2_n_completos=len(v2_comp), v2_n_censurados=len(v2_cens),
        v2_dur=_stats([_dur(e) for e in v2_comp]),
        # v3
        v3_n_completos=len(v3_comp), v3_n_censurados=len(v3_cens),
        v3_dur=_stats([_dur(e) for e in v3_comp]),
        v3_n_completos_zmin_neg=len(v3_comp_zmin_neg),
        v3_n_completos_profundo=len(v3_comp_profundo),
        v3_zmin_min=(float(min(v3_zmin_comp)) if v3_zmin_comp else None),
        segundos=round(time.time() - t0, 1),
    )
    # detalhe por run (com episódios) em gz separado
    detalhe = dict(rec, loss_hist=lh,
                   revive_real_eventos=metr["revive_real_eventos"],
                   v2_completos=v2_comp, v2_censurados=v2_cens,
                   v3_completos=v3_comp, v3_censurados=v3_cens)
    with gzip.open(caminho, "wt") as f:
        json.dump(detalhe, f)
    return ("OK " + _nome(t) + f": {rec['segundos']:.0f}s "
            f"loss_plato={loss_plato:.4f} mortos_real={rec['n_mortos_real']} "
            f"rev_real={rec['n_revive_real']} v2c={rec['v2_n_completos']} "
            f"v3c={rec['v3_n_completos']} v3zneg={rec['v3_n_completos_zmin_neg']}")


def agrega():
    linhas = []
    for t in TAREFAS:
        p = os.path.join(OUT, _nome(t))
        if not os.path.exists(p):
            continue
        d = json.load(gzip.open(p, "rt"))
        for k in ("loss_hist", "revive_real_eventos", "v2_completos",
                  "v2_censurados", "v3_completos", "v3_censurados"):
            d.pop(k, None)
        linhas.append(d)
    with open(os.path.join(OUT, "pressao.jsonl"), "w") as f:
        for d in linhas:
            f.write(json.dumps(d) + "\n")
    return len(linhas)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=None,
                    help="filtra por rot (ex: relu) ou 'm4' etc; vazio = tudo")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--agrega-so", action="store_true")
    a = ap.parse_args()
    if a.agrega_so:
        print("agregado:", agrega(), "linhas")
        raise SystemExit
    tarefas = TAREFAS
    if a.subset:
        s = a.subset
        tarefas = [t for t in TAREFAS
                   if s == t["rot"] or s == f"m{t['m']}"
                   or s == f"s{t['seed']}" or s == f"{t['rot']}-m{t['m']}"]
    print(f"{len(tarefas)} runs, Pool {a.workers}, {PASSOS} passos, T={T}",
          flush=True)
    with Pool(a.workers) as p:
        for msg in p.imap_unordered(roda, tarefas):
            print(msg, flush=True)
    print("agregado:", agrega(), "linhas")
    print("PRESSAO OK")
