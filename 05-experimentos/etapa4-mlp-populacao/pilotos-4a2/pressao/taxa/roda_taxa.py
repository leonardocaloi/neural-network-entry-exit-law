"""Runner do PILOTO-DE-TAXA (etapa-4a.3). Grade congelada pelo coordenador:

  braços {softplus-β3, softplus-β10, softplus-β300, relu} × m∈{4,6}
  × seeds 100-129 (30 NOVAS) × n=16, η=1e-3, T=0, H=5e5, rec_neuronio=0.
  = 4 × 2 × 30 = 240 runs.

Motor: motor_taxa.treina_taxa (paridade bit-a-bit com treina_pressao verificada).
Saída régua-independente por ground-truth + episódios v2 crus (para o trio).
RESUMÁVEL: pula arquivo já existente. Resultados em taxa/results/*.jsonl.gz.

Rodar (de pilotos-4a2/pressao/taxa/):
  nice -19 taskset -c 0-7 ../../../../../venv/bin/python roda_taxa.py --workers 3
"""
import gzip
import json
import os
import time
from multiprocessing import Pool

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
PRESSAO = os.path.dirname(AQUI)
ETAPA4 = os.path.dirname(os.path.dirname(PRESSAO))
OUT = os.path.join(AQUI, "results")
os.makedirs(OUT, exist_ok=True)

PASSOS = 500_000
ETA = 1e-3
T = 0.0
N_DADOS = 16
SEEDS = list(range(100, 130))          # 30 seeds virgens 100..129

# (rótulo, ativ, beta)
ATIVS = [
    ("sp-b3", "softplus", 3.0),
    ("sp-b10", "softplus", 10.0),
    ("sp-b300", "softplus", 300.0),
    ("relu", "relu", 300.0),
]
MS = [4, 6]

TAREFAS = []
for rot, ativ, beta in ATIVS:
    for m in MS:
        for s in SEEDS:
            TAREFAS.append(dict(rot=rot, ativ=ativ, beta=beta, m=m, seed=s))


def _nome(t):
    return f"{t['rot']}-m{t['m']}-s{t['seed']}.jsonl.gz"


def _v2_lista(reg2, passos):
    """Episódios v2 crus (completos + censurados) com o estado explícito.
    Guarda o trio (t_morte, t_area, t_res) e flags — o parecer usa a v2."""
    out = []
    for e in reg2.episodios:
        d = dict(e); d["estado"] = "completo"
        out.append(d)
    for e in reg2.censurados(passos):
        d = dict(e); d["estado"] = "censurado"
        out.append(d)
    return out


def roda(t):
    import sys
    sys.path.insert(0, ETAPA4)
    sys.path.insert(0, PRESSAO)
    sys.path.insert(0, AQUI)
    import torch
    torch.set_num_threads(1)
    try:
        os.nice(19)
    except Exception:
        pass
    import motor_taxa as M
    caminho = os.path.join(OUT, _nome(t))
    if os.path.exists(caminho):
        return f"skip {_nome(t)}"
    t0 = time.time()
    reg2, metr = M.treina_taxa(
        m=t["m"], ativ=t["ativ"], beta=t["beta"], eta=ETA, T=T, passos=PASSOS,
        seed=t["seed"], n_dados=N_DADOS, rec_cada=100)

    lh = metr["loss_hist"]
    n10 = max(1, len(lh) // 10)
    loss_plato = float(np.mean(lh[-n10:]))

    rec = dict(
        rot=t["rot"], ativ=t["ativ"], beta=t["beta"], m=t["m"], seed=t["seed"],
        T=T, eta=ETA, n_dados=N_DADOS, passos=PASSOS,
        loss_inicial=float(lh[0]), loss_final=float(lh[-1]), loss_plato=loss_plato,
        # ground-truth régua-independente (contador-mestre do 9º parecer)
        nasce_morto=metr["nasce_morto"],
        n_nasce_morto=metr["n_nasce_morto"],
        death_events=metr["death_events"],       # (t,k) vivo->morto, t>0 = dinâmica
        revive_events=metr["revive_events"],     # (t,k) morto->vivo (zmax>0)
        n_mortos_real=metr["n_mortos_real"],
        n_revive_real=metr["n_revive_real"],
        n_neuronios_revive_real=metr["n_neuronios_revive_real"],
        # régua v2 crua (para o trio t_morte<t_area<t_res e a flag área-abriu)
        v2_episodios=_v2_lista(reg2, PASSOS),
        segundos=round(time.time() - t0, 1),
    )
    with gzip.open(caminho, "wt") as f:
        json.dump(rec, f)
    return (f"OK {_nome(t)}: {rec['segundos']:.0f}s plato={loss_plato:.3f} "
            f"born={metr['n_nasce_morto']} deaths_dyn={len(metr['death_events'])} "
            f"rev_gt={metr['n_revive_real']} v2eps={len(rec['v2_episodios'])}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--subset", default=None, help="rot|mN|sN substring; vazio=tudo")
    a = ap.parse_args()
    tarefas = TAREFAS
    if a.subset:
        s = a.subset
        tarefas = [t for t in TAREFAS
                   if s in (t["rot"], f"m{t['m']}", f"s{t['seed']}")]
    feitos = sum(1 for t in TAREFAS if os.path.exists(os.path.join(OUT, _nome(t))))
    print(f"{len(tarefas)} tarefas (de {len(TAREFAS)} na grade; {feitos} já feitos), "
          f"Pool {a.workers}, H={PASSOS}, n={N_DADOS}, eta={ETA}, T={T}, "
          f"seeds {SEEDS[0]}..{SEEDS[-1]}", flush=True)
    with Pool(a.workers) as p:
        for msg in p.imap_unordered(roda, tarefas):
            print(msg, flush=True)
    print("RODA_TAXA fim.", flush=True)
