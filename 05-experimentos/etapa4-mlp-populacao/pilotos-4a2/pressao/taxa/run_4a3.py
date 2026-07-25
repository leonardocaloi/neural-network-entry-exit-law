"""Runner do CONFIRMATÓRIO etapa-4a.3 (SÓ rodar após o commit de pré-registro e o
10º parecer). Lê config_4a3.json; motor = motor_4a3.treina_4a3 (paridade provada);
instrumento GT-ÁREA (episodios_gt). 400 runs (4 braços × m∈{4,6} × 50 seeds 1900-49).

Robustez (lição da ceifa: run confirmatória NÃO pode morrer com o terminal):
  - DESTACADO: ignora SIGHUP e best-effort os.setsid(); lançar com
      setsid nice -19 ../../../../../venv/bin/python run_4a3.py --workers 4 \
             > run_4a3.log 2>&1 < /dev/null &
  - RETOMÁVEL: pula results-4a3/*.jsonl.gz já existentes (escrita atômica .tmp→replace);
  - Pool 4, nice -19 por worker, 1 thread torch/worker, CPU;
  - MANIFEST ao fim: SHA256 por arquivo + SHA256 do config + git commit.

--smoke: 2 runs curtas (H=2e4, seeds 42-43, sp-b3 m4, FORA das virgens) em
results-4a3-smoke/ — mecânica apenas, NÃO confirmatório, NÃO toca a grade.
"""
import argparse
import gzip
import hashlib
import json
import os
import signal
import time
from multiprocessing import Pool

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
PRESSAO = os.path.dirname(AQUI)
ETAPA4 = os.path.dirname(os.path.dirname(PRESSAO))
CONFIG = os.path.join(AQUI, "config_4a3.json")
RESDIR = os.path.join(AQUI, "results-4a3")
SMOKEDIR = os.path.join(AQUI, "results-4a3-smoke")


def _nome(t):
    return f"{t['rot']}-m{t['m']}-s{t['seed']}.jsonl.gz"


def _tarefas_confirmatorio(cfg):
    g = cfg["grade"]
    tarefas = []
    for b in g["bracos"]:
        for m in g["m"]:
            for s in g["seeds"]:
                tarefas.append(dict(rot=b["nome"], ativ=b["ativ"], beta=b["beta"],
                                    m=m, seed=s, passos=g["H"], eta=g["eta"], T=g["T"],
                                    n_dados=g["n_dados"], rec_cada=g["rec_cada"]))
    return tarefas


def _roda(t, outdir):
    import sys
    for p in (ETAPA4, PRESSAO, AQUI):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    torch.set_num_threads(1)
    try:
        os.nice(19)
    except Exception:
        pass
    import motor_4a3 as M
    caminho = os.path.join(outdir, _nome(t))
    if os.path.exists(caminho):
        return f"skip {_nome(t)}"
    t0 = time.time()
    metr = M.treina_4a3(m=t["m"], ativ=t["ativ"], beta=t["beta"], eta=t["eta"],
                        T=t["T"], passos=t["passos"], seed=t["seed"],
                        n_dados=t["n_dados"], rec_cada=t["rec_cada"])
    lh = metr["loss_hist"]
    n10 = max(1, len(lh) // 10)
    rec = dict(
        rot=t["rot"], ativ=t["ativ"], beta=t["beta"], m=t["m"], seed=t["seed"],
        T=t["T"], eta=t["eta"], n_dados=t["n_dados"], passos=t["passos"],
        loss_inicial=float(lh[0]), loss_final=float(lh[-1]),
        loss_plato=float(np.mean(lh[-n10:])),
        nasce_morto=metr["nasce_morto"], n_nasce_morto=metr["n_nasce_morto"],
        death_events=metr["death_events"], revive_events=metr["revive_events"],
        n_mortos_real=metr["n_mortos_real"], n_revive_real=metr["n_revive_real"],
        n_neuronios_revive_real=metr["n_neuronios_revive_real"],
        episodios_gt=metr["episodios_gt"],          # instrumento GT-ÁREA
        segundos=round(time.time() - t0, 1),
    )
    tmp = caminho + ".tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(rec, f)
    os.replace(tmp, caminho)
    n_dyn = sum(1 for e in metr["episodios_gt"] if e["classe"] == "dyn")
    n_volt = sum(1 for e in metr["episodios_gt"] if e["classe"] == "dyn" and e["revived"])
    return (f"OK {_nome(t)}: {rec['segundos']:.0f}s plato={rec['loss_plato']:.3f} "
            f"born={metr['n_nasce_morto']} dyn={n_dyn} voltas={n_volt}")


def _job_conf(t):
    return _roda(t, RESDIR)


def _job_smoke(t):
    return _roda(t, SMOKEDIR)


def _manifest(outdir, nome_saida):
    man = {}
    for nome in sorted(os.listdir(outdir)):
        if nome.endswith(".jsonl.gz"):
            with open(os.path.join(outdir, nome), "rb") as f:
                man[nome] = hashlib.sha256(f.read()).hexdigest()
    with open(CONFIG, "rb") as f:
        man["_config_sha256"] = hashlib.sha256(f.read()).hexdigest()
    import subprocess
    man["_git_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        cwd=AQUI).stdout.strip()
    with open(os.path.join(outdir, nome_saida), "w") as f:
        json.dump(man, f, indent=0, sort_keys=True)
    print(f"[manifest] {os.path.join(outdir, nome_saida)}: {len(man)} entradas "
          f"(config {man['_config_sha256'][:12]}…, git {man['_git_commit'][:12]}…)",
          flush=True)


def _destaca():
    """Insula contra o hangup do terminal (a 'ceifa'). Best-effort."""
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    except Exception:
        pass
    try:
        os.setsid()
    except OSError:
        pass          # já é líder de sessão (lançado via `setsid`) — ok
    with open(os.path.join(AQUI, "run_4a3.pid"), "w") as f:
        f.write(str(os.getpid()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true",
                    help="2 runs curtas H=2e4 seeds 42-43 (mecânica; NÃO confirmatório)")
    a = ap.parse_args()
    _destaca()

    if a.smoke:
        os.makedirs(SMOKEDIR, exist_ok=True)
        tarefas = [dict(rot="sp-b3", ativ="softplus", beta=3.0, m=4, seed=s,
                        passos=20_000, eta=1e-3, T=0.0, n_dados=16, rec_cada=100)
                   for s in (42, 43)]
        print(f"SMOKE: {len(tarefas)} runs curtas (H=2e4, seeds 42-43) → {SMOKEDIR}",
              flush=True)
        with Pool(min(a.workers, 2)) as p:
            for msg in p.imap_unordered(_job_smoke, tarefas):
                print(msg, flush=True)
        _manifest(SMOKEDIR, "manifest-smoke.json")
        print("SMOKE fim (mecânica; nenhuma estatística confirmatória).", flush=True)
        raise SystemExit(0)

    cfg = json.load(open(CONFIG))
    os.makedirs(RESDIR, exist_ok=True)
    tarefas = _tarefas_confirmatorio(cfg)
    feitos = sum(1 for t in tarefas if os.path.exists(os.path.join(RESDIR, _nome(t))))
    print(f"CONFIRMATÓRIO: {len(tarefas)} runs ({feitos} já feitos), Pool {a.workers}, "
          f"nice-19, H={cfg['grade']['H']}, seeds {cfg['grade']['seeds'][0]}.."
          f"{cfg['grade']['seeds'][-1]}", flush=True)
    with Pool(a.workers) as p:
        for msg in p.imap_unordered(_job_conf, tarefas):
            print(msg, flush=True)
    _manifest(RESDIR, "manifest-4a3.json")
    print("RUN_4A3 fim.", flush=True)
