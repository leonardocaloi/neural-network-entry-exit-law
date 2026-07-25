"""Trilho-3 PILOTOS 4a.2: dimensionar T pequeno + horizonte longo.

Importa treina()/Regua de harness_mlp SEM modificar (Regua v2 serve só p/
CONTAGEM aqui — episódio >=50 passos como proxy de físico pós-debounce).

Grade piloto (exploratória, seeds 45-47, fora de 1700-1799):
  - softplus β=300, η=3e-4, m=64, T ∈ {1e-5, 1e-4, 0.0}, horizonte 5e5  (9 runs)
  - semgrad NULO (difusão pura), mesmos T pequenos {1e-5, 1e-4}, m=64      (6 runs)
    -> controle de instrumento: se o nulo produz tantos episódios >=50 quanto
       o softplus, o debounce NÃO separa física de ruído nesse T (foi isso que
       matou a 4a em T=1e-3). semgrad@T=0 é imóvel (σ=0), omitido.

Cada run -> pilotos-4a2/results-piloto/{braco}-m{m}-T{T}-s{seed}.json.gz
Pool<=2, nice-19, 1 thread torch/worker. ~4.3 min/run 5e5 => ~34 min total.
"""
import gzip
import json
import os
import time
from multiprocessing import Pool

AQUI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(AQUI, "results-piloto")
os.makedirs(OUT, exist_ok=True)

PASSOS = 500_000
M = 64
SEEDS = [45, 46, 47]
ETA = 3e-4
BETA = 300.0

TAREFAS = []
for T in (1e-5, 1e-4, 0.0):
    for s in SEEDS:
        TAREFAS.append(dict(braco="softplus", ativ="softplus", T=T, seed=s,
                            sem_grad=False))
for T in (1e-5, 1e-4):                       # nulo só em T>0 (σ=0 em T=0)
    for s in SEEDS:
        TAREFAS.append(dict(braco="semgrad", ativ="softplus", T=T, seed=s,
                            sem_grad=True))


def _nome(t):
    return f"{t['braco']}-m{M}-T{t['T']:.0e}-s{t['seed']}.json.gz"


def roda(t):
    import sys
    sys.path.insert(0, os.path.dirname(AQUI))   # etapa4 dir tem harness_mlp
    import torch
    torch.set_num_threads(1)
    try:
        os.nice(19)
    except Exception:
        pass
    from harness_mlp import treina
    caminho = os.path.join(OUT, _nome(t))
    if os.path.exists(caminho):
        return f"skip {_nome(t)}"
    t0 = time.time()
    net, reg, loss, _ = treina(m=M, ativ=t["ativ"], beta=BETA, eta=ETA,
                               T=t["T"], passos=PASSOS, seed=t["seed"],
                               sem_grad=t["sem_grad"])
    eps = list(reg.episodios) + reg.censurados(PASSOS)
    d = dict(braco=t["braco"], m=M, seed=t["seed"], T=t["T"], eta=ETA,
             beta=BETA, passos=PASSOS, sem_grad=t["sem_grad"],
             loss=loss, n_switches=len(reg.switches), episodios=eps,
             segundos=round(time.time() - t0, 1))
    with gzip.open(caminho, "wt") as f:
        json.dump(d, f)
    return (f"OK {_nome(t)}: {d['segundos']:.0f}s loss={loss:.4f} "
            f"n_ep={len(eps)} completos={len(reg.episodios)}")


if __name__ == "__main__":
    print(f"{len(TAREFAS)} runs, Pool 2, {PASSOS} passos, m={M}")
    with Pool(2) as p:
        for msg in p.imap_unordered(roda, TAREFAS):
            print(msg, flush=True)
    print("PILOTOS OK")
