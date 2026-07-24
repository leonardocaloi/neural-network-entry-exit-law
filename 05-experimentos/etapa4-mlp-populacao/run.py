"""Etapa 4a — runner da grade congelada. SÓ rodar após o commit de pré-registro.

Lê config_etapa4.json; executa 600 runs (braço × m × seed) em Pool nice-19,
CPU pinada, 1 thread/worker; grava results/{braco}-m{m}-s{seed}.json.gz com
episódios (+ contagem de switchings alheios por episódio, p/ filtro-limpo e
sensibilidade em q), censurados, lista de congelados (P-D) e loss final.
Retomável: pula arquivos já existentes. Ao final: manifest.json com SHA256.
"""
import gzip
import hashlib
import json
import os
import time
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor

CFG = json.load(open(os.path.join(os.path.dirname(__file__), "config_etapa4.json")))
RESDIR = os.path.join(os.path.dirname(__file__), "results")


def anota_alheios(eps, switches):
    ts = sorted(t for t, _, _ in switches)
    por_k = {}
    for t, k, _ in switches:
        por_k.setdefault(k, []).append(t)
    for v in por_k.values():
        v.sort()
    for ep in eps:
        fim = ep["t_res"] if ep["t_res"] >= 0 else ep["censurado_em"]
        tot = bisect_right(ts, fim) - bisect_right(ts, ep["t_morte"])
        prop = por_k.get(ep["k"], [])
        meus = bisect_right(prop, fim) - bisect_right(prop, ep["t_morte"])
        ep["alheios"] = tot - meus
    return eps


def job(spec):
    import torch
    torch.set_num_threads(int(CFG["device"]["threads_por_worker"]))
    from harness_mlp import treina
    braco, m, seed = spec
    caminho = os.path.join(RESDIR, f"{braco['nome']}-m{m}-s{seed}.json.gz")
    if os.path.exists(caminho):
        return caminho, "pulado"
    g = CFG["grade"]
    t0 = time.perf_counter()
    _, reg, loss, congela = treina(
        m=m, ativ=braco["ativ"], beta=braco.get("beta") or 300.0,
        eta=g["eta"], T=braco["T"], passos=g["passos"], seed=seed,
        device=CFG["device"]["pin"],
        embaralha=braco.get("embaralha", False),
        sem_grad=braco.get("sem_grad", False),
        freeze_meio_mortos=braco.get("freeze_meio_mortos", False))
    eps = anota_alheios(reg.episodios + reg.censurados(g["passos"]), reg.switches)
    out = dict(braco=braco["nome"], m=m, seed=seed, loss=float(loss),
               passos=g["passos"], eta=g["eta"], T=braco["T"],
               congelados=[int(i) for i in range(m) if congela[i]],
               n_switches=len(reg.switches), episodios=eps,
               segundos=round(time.perf_counter() - t0, 1))
    tmp = caminho + ".tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(out, f)
    os.replace(tmp, caminho)
    return caminho, f"{out['segundos']}s ep={len(eps)}"


def worker_init():
    os.nice(19)


if __name__ == "__main__":
    os.makedirs(RESDIR, exist_ok=True)
    specs = [(b, m, s) for b in CFG["grade"]["bracos"]
             for m in CFG["grade"]["m"] for s in CFG["grade"]["seeds"]]
    print(f"{len(specs)} runs; device={CFG['device']['pin']}; Pool 6 nice-19")
    feitos = 0
    with ProcessPoolExecutor(max_workers=6, initializer=worker_init) as ex:
        for caminho, info in ex.map(job, specs):
            feitos += 1
            if feitos % 20 == 0 or info != "pulado":
                print(f"[{feitos}/{len(specs)}] {os.path.basename(caminho)}: {info}",
                      flush=True)
    man = {}
    for nome in sorted(os.listdir(RESDIR)):
        if nome.endswith(".json.gz"):
            with open(os.path.join(RESDIR, nome), "rb") as f:
                man[nome] = hashlib.sha256(f.read()).hexdigest()
    with open(os.path.join(RESDIR, "manifest.json"), "w") as f:
        json.dump(man, f, indent=0, sort_keys=True)
    print(f"manifest.json: {len(man)} arquivos")
