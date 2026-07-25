"""Runner do CONFIRMATORIO etapa-5 (SO rodar apos commit de pre-registro + 11o parecer).
Le config_e5.json; motor = motor_e5_conf.treina_e5_conf (paridade byte-a-byte com
motor_e5b provada em --testes); instrumento GT com HISTERESE + volta SUSTENTADA.
800 runs = 8 celulas (4 bracos x {train,frozen}) x 2 injecoes {-0.3,-2.0} x 50 seeds 2000-49.

Robustez (licao da ceifa: run confirmatoria NAO pode morrer com o terminal):
  - DESTACADO: ignora SIGHUP + os.setsid(); lancar com
      setsid taskset -c 0-7 nice -n 19 ../../venv/bin/python run_e5.py --workers 4 \
             > run_e5.log 2>&1 < /dev/null &
  - RETOMAVEL: pula results-e5/*.jsonl.gz ja existentes (escrita atomica .tmp->replace);
  - Pool 4 (cores 0-7), nice-19 por worker, 1 thread torch/worker, CPU;
  - MANIFEST ao fim: SHA256 por arquivo + SHA256 do config + git commit.

--smoke: 2 runs H=2e5, seeds 42-43 (FORA das virgens), sp-b3-frozen d-0.3, em
results-e5-smoke/ — mecanica/tempo apenas, NAO confirmatorio, NAO toca a grade.
--estimativa: mede o tempo de 2 runs de teste e projeta o custo total (nao grava grade).
"""
import argparse
import gzip
import hashlib
import json
import os
import signal
import time
from multiprocessing import Pool

AQUI = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(AQUI, "config_e5.json")
RESDIR = os.path.join(AQUI, "results-e5")
SMOKEDIR = os.path.join(AQUI, "results-e5-smoke")

ARM_MAP = {"sp-b3": ("softplus", 3.0), "sp-b10": ("softplus", 10.0),
           "gelu": ("gelu", 1.0), "relu": ("relu", 1.0)}


def _depth_tag(depth):
    return f"d{abs(depth):.1f}".replace(".", "_")   # -0.3 -> d0_3 ; -2.0 -> d2_0


def _nome(t):
    return f"{t['arm']}-{t['fac']}-{_depth_tag(t['depth'])}-s{t['seed']}.jsonl.gz"


def _tarefas(cfg):
    g = cfg["grade"]
    tarefas = []
    for a in g["bracos"]:
        for fac in g["factors"]:
            for depth in g["inj_depths"]:
                for s in g["seeds"]:
                    tarefas.append(dict(arm=a["nome"], ativ=a["ativ"], beta=a["beta"],
                                        fac=fac, depth=depth, seed=s))
    return tarefas


def _roda(t, cfg, outdir):
    import sys
    if AQUI not in sys.path:
        sys.path.insert(0, AQUI)
    import torch
    torch.set_num_threads(1)
    try:
        os.nice(19)
    except Exception:
        pass
    import motor_e5_conf as MC
    caminho = os.path.join(outdir, _nome(t))
    if os.path.exists(caminho):
        return f"skip {_nome(t)}"
    g = cfg["grade"]
    t0 = time.time()
    mt = MC.treina_e5_conf(
        m_ffn=g["m_ffn"], ativ=t["ativ"], beta=t["beta"], eta=g["eta"], passos=g["H"],
        seed=t["seed"], N=g["n_dados"], S=g["S_seq"], d=g["d_model"], H=g["H_heads"],
        delta_h=cfg["delta_h"], rec_cada=g["rec_cada"],
        freeze_attn=(t["fac"] == "frozen"), kill_depth=abs(t["depth"]))
    rec = dict(
        arm=t["arm"], fac=t["fac"], depth=t["depth"], seed=t["seed"], cfg=mt["cfg"],
        plato=mt["plato"], loss_fim=mt["loss_fim"],
        n_injetadas=mt["n_injetadas"], n_inj_revived=mt["n_inj_revived"],
        n_inj_sustained=mt["n_inj_sustained"], n_inj_hover=mt["n_inj_hover"],
        n_natural_dyn=mt["n_natural_dyn"], n_born_natural=mt["n_born_natural"],
        n_revive=mt["n_revive"], unidades=mt["unidades"],
        segundos=round(time.time() - t0, 1))
    os.makedirs(outdir, exist_ok=True)
    tmp = caminho + ".tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(rec, f)
    os.replace(tmp, caminho)
    return (f"OK {_nome(t)}: {rec['segundos']:.0f}s plato={rec['plato']:.3f} "
            f"inj_sust={mt['n_inj_sustained']} inj_rev={mt['n_inj_revived']} "
            f"hover={mt['n_inj_hover']}")


_CFG = None


def _init(cfg):
    global _CFG
    _CFG = cfg


def _job_conf(t):
    return _roda(t, _CFG, RESDIR)


def _job_smoke(t):
    return _roda(t, _CFG, SMOKEDIR)


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
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=AQUI).stdout.strip()
    with open(os.path.join(outdir, nome_saida), "w") as f:
        json.dump(man, f, indent=0, sort_keys=True)
    print(f"[manifest] {os.path.join(outdir, nome_saida)}: {len(man)} entradas "
          f"(config {man['_config_sha256'][:12]}, git {man['_git_commit'][:12]})", flush=True)


def _destaca(pidname):
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    except Exception:
        pass
    try:
        os.setsid()
    except OSError:
        pass
    with open(os.path.join(AQUI, pidname), "w") as f:
        f.write(str(os.getpid()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--estimativa", action="store_true",
                    help="tempo de 2 runs de teste (seeds 42-43) + projecao (nao grava grade)")
    a = ap.parse_args()
    cfg = json.load(open(CONFIG))
    _destaca("run_e5.pid")

    if a.estimativa or a.smoke:
        os.makedirs(SMOKEDIR, exist_ok=True)
        tarefas = [dict(arm="sp-b3", ativ="softplus", beta=3.0, fac="frozen",
                        depth=-0.3, seed=s) for s in (42, 43)]
        print(f"{'ESTIMATIVA' if a.estimativa else 'SMOKE'}: {len(tarefas)} runs H={cfg['grade']['H']} "
              f"(seeds 42-43, sp-b3-frozen d-0.3) -> {SMOKEDIR}", flush=True)
        t0 = time.time()
        _init(cfg)
        with Pool(min(a.workers, 2), initializer=_init, initargs=(cfg,)) as p:
            for msg in p.imap_unordered(_job_smoke, tarefas):
                print(msg, flush=True)
        dt = time.time() - t0
        _manifest(SMOKEDIR, "manifest-smoke.json")
        total = cfg["grade"]["total_runs"]
        seg_por_run = dt / len(tarefas) * min(a.workers, 2)   # tempo-CPU medio/run
        est_h = total * seg_por_run / a.workers / 3600
        print(f"[estimativa] 2 runs em {dt:.0f}s (>={min(a.workers,2)} workers) => "
              f"~{seg_por_run:.0f}s-CPU/run; {total} runs / {a.workers} workers ~= {est_h:.1f} h",
              flush=True)
        raise SystemExit(0)

    os.makedirs(RESDIR, exist_ok=True)
    tarefas = _tarefas(cfg)
    feitos = sum(1 for t in tarefas if os.path.exists(os.path.join(RESDIR, _nome(t))))
    print(f"CONFIRMATORIO e5: {len(tarefas)} runs ({feitos} ja feitos), Pool {a.workers}, "
          f"nice-19, H={cfg['grade']['H']}, delta_h={cfg['delta_h']:.4f}, "
          f"seeds {cfg['seeds'][0]}..{cfg['seeds'][-1]}", flush=True)
    with Pool(a.workers, initializer=_init, initargs=(cfg,)) as p:
        for msg in p.imap_unordered(_job_conf, tarefas):
            print(msg, flush=True)
    _manifest(RESDIR, "manifest-e5.json")
    print("RUN_E5 fim.", flush=True)
