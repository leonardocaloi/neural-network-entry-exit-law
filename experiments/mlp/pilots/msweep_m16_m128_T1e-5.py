"""Cheque da borda inferior do washout: m=16 (e m=128), T=1e-5, H=2e5.
softplus (física) + semgrad (null). Localiza onde a separação some. Pool 2 nice-19."""
from pathlib import Path
import gzip, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

BASE = str(Path(__file__).resolve().parent / '..')
OUT = os.path.join(BASE, "pilots", "pilot-results")
if BASE not in sys.path:
    sys.path.insert(0, BASE)

PASSOS = 200_000
T = 1e-5


def job(spec):
    import torch
    torch.set_num_threads(1)
    from harness_mlp import treina
    braco, M, seed = spec
    caminho = os.path.join(OUT, f"{braco}-m{M}-T{T:.0e}-H2e5-s{seed}.json.gz")
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
    specs = [(b, M, s) for M in (16, 128) for b in ("softplus", "semgrad")
             for s in (45, 46, 47)]
    print(f"{len(specs)} runs T={T:.0e} passos={PASSOS}; Pool 2 nice-19", flush=True)
    with ProcessPoolExecutor(max_workers=2, initializer=worker_init) as ex:
        for cam, msg in ex.map(job, specs):
            print(os.path.basename(cam), "->", msg, flush=True)
    print("== check m16/m128 OK ==", flush=True)
