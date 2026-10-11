"""Cheque de heterogeneidade em m=256, T=1e-5, horizonte 2e5 (de-risca a escolha
de m para a 4a.2). softplus (física) + semgrad (null difusão pura), seeds 45-47.
Importa treina() de harness_mlp SEM modificar. Salva em results-piloto/ no mesmo
formato. Pool 2, nice-19."""
from pathlib import Path
import gzip, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

BASE = str(Path(__file__).resolve().parent / '..')
OUT = os.path.join(BASE, "pilotos-4a2", "results-piloto")
if BASE not in sys.path:
    sys.path.insert(0, BASE)

PASSOS = 200_000
M = 256
T = 1e-5


def job(spec):
    import torch
    torch.set_num_threads(1)
    from harness_mlp import treina
    braco, seed = spec
    tagT = f"{T:.0e}"
    caminho = os.path.join(OUT, f"{braco}-m{M}-T{tagT}-H2e5-s{seed}.json.gz")
    if os.path.exists(caminho):
        return caminho, "pulado"
    t0 = time.perf_counter()
    _, reg, loss, _ = treina(m=M, ativ="softplus", beta=300.0, eta=3e-4, T=T,
                             passos=PASSOS, seed=seed, device="cpu",
                             sem_grad=(braco == "semgrad"))
    eps = reg.episodios + reg.censurados(PASSOS)
    for ep in eps:
        ep.setdefault("alheios", 0)
    out = dict(braco=braco, m=M, seed=seed, loss=float(loss), passos=PASSOS,
               eta=3e-4, T=T, congelados=[], n_switches=len(reg.switches),
               episodios=eps, segundos=round(time.perf_counter() - t0, 1))
    tmp = caminho + ".tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(out, f)
    os.replace(tmp, caminho)
    return caminho, f"{out['segundos']}s ep={len(eps)}"


def worker_init():
    os.nice(19)


if __name__ == "__main__":
    # braco semgrad usa a mesma treina com sem_grad=True (mas ativ precisa ser válido)
    specs = ([("softplus", s) for s in (45, 46, 47)]
             + [("semgrad", s) for s in (45, 46, 47)])
    print(f"{len(specs)} runs m={M} T={T:.0e} passos={PASSOS}; Pool 2 nice-19", flush=True)
    with ProcessPoolExecutor(max_workers=2, initializer=worker_init) as ex:
        for cam, msg in ex.map(job, specs):
            print(os.path.basename(cam), "->", msg, flush=True)
    print("== check m256 OK ==", flush=True)
