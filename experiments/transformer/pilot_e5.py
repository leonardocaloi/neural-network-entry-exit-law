"""Piloto de SCOPING da etapa-5 (transformer FFN-GELU: morte->ressurreicao-por-area).
EXPLORATORIO — dimensiona/informa o confirmatorio; NADA aqui e pre-registrado.

Grade (6 bracos x seeds 42-44 = 18 runs). Contrastes:
  - PRESSAO vs FOLGA (H-c): gelu-press (m4) vs gelu-slack (m32).
  - RESSURREICAO GELU + instrumento GT-AREA (H-a,H-b): gelu-kill-tr / gelu-kill-fz
    (unidades NASCIDAS-MORTAS injetadas @ z=-1.5; z_dead=-1; attn treina vs congelada).
  - CANAL + entrada nao-estacionaria (S1/S2): relu-kill-tr / relu-kill-fz
    (relu-fz = teorema MLP: terminal; relu-tr = so drive de montante pode reviver).

Uso:
  setsid taskset -c 0-7 nice -n 19 PYTHON pilot_e5.py --run --workers 2 > pilot.log 2>&1 &
  PYTHON pilot_e5.py --analise
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pilot-results")
SEEDS = [42, 43, 44]

# base comum do sistema
BASE = dict(N=24, S=3, d=16, H=2, eta=1e-3, passos=150_000, rec_cada=300, z_alive=0.0)

ARMS = {
    # nome            ativ    m   kwargs extra
    "gelu-press":  dict(ativ="gelu", m_ffn=4,  n_kill=0, z_dead=-3.0, freeze_attn=False),
    "gelu-slack":  dict(ativ="gelu", m_ffn=32, n_kill=0, z_dead=-3.0, freeze_attn=False),
    "gelu-kill-tr": dict(ativ="gelu", m_ffn=4, n_kill=2, kill_depth=1.5, z_dead=-1.0, freeze_attn=False),
    "gelu-kill-fz": dict(ativ="gelu", m_ffn=4, n_kill=2, kill_depth=1.5, z_dead=-1.0, freeze_attn=True),
    "relu-kill-tr": dict(ativ="relu", m_ffn=4, n_kill=2, kill_depth=1.5, z_dead=0.0,  freeze_attn=False),
    "relu-kill-fz": dict(ativ="relu", m_ffn=4, n_kill=2, kill_depth=1.5, z_dead=0.0,  freeze_attn=True),
}


def _job_kwargs(arm, seed):
    kw = dict(BASE); kw.update(ARMS[arm]); kw["seed"] = seed
    return kw


def run_one(job):
    arm, seed = job
    import torch
    torch.set_num_threads(1)
    import engine_e5 as M
    kw = _job_kwargs(arm, seed)
    t0 = time.time()
    mt = M.treina_e5(**kw)
    dt = time.time() - t0
    out = dict(
        arm=arm, seed=seed, secs=round(dt, 1), cfg=mt["cfg"],
        plato=mt["plato"], loss_fim=mt["loss_fim"], loss_hist=mt["loss_hist"],
        n_nasce_morto=mt["n_nasce_morto"], n_death_dyn=len(mt["death_events"]),
        n_mortos=mt["n_mortos"], n_revive=mt["n_revive"],
        n_neuronios_revive=mt["n_neuronios_revive"],
        episodios=mt["episodios"])
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, f"{arm}-s{seed}.json"), "w") as f:
        json.dump(out, f)
    print(f"OK {arm}-s{seed}: {dt:.0f}s plato={out['plato']:.4f} lossf={out['loss_fim']:.4f} "
          f"born={out['n_nasce_morto']} dyn={out['n_death_dyn']} rev={out['n_revive']} "
          f"eps={len(out['episodios'])}", flush=True)
    return arm, seed


def main_run(workers):
    os.makedirs(RES, exist_ok=True)
    jobs = [(arm, s) for arm in ARMS for s in SEEDS]
    print(f"== piloto_e5: {len(jobs)} runs, {workers} workers, passos={BASE['passos']} ==", flush=True)
    t0 = time.time()
    with Pool(workers, maxtasksperchild=1) as p:
        for _ in p.imap_unordered(run_one, jobs):
            pass
    print(f"== FIM: {len(jobs)} runs em {time.time()-t0:.0f}s ==", flush=True)


# ============================== ANALISE ==============================
def _load():
    out = {}
    for arm in ARMS:
        rows = []
        for s in SEEDS:
            fp = os.path.join(RES, f"{arm}-s{s}.json")
            if os.path.exists(fp):
                with open(fp) as f:
                    rows.append(json.load(f))
        out[arm] = rows
    return out


def main_analise():
    import numpy as np
    D = _load()
    print("# Piloto etapa-5 (transformer FFN) — analise\n")
    # ---- tabela por braco ----
    print("## Tabela por braco (agregado sobre seeds)\n")
    hdr = ("braco", "runs", "plato", "lossf", "born", "dyn", "eps", "rev",
           "p(rev|ep)", "zpark_med", "n_<-3")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for arm, rows in D.items():
        if not rows:
            print(f"| {arm} | 0 | - | - | - | - | - | - | - | - | - |"); continue
        eps = [e for r in rows for e in r["episodios"]]
        nrev = sum(r["n_revive"] for r in rows)
        born = sum(r["n_nasce_morto"] for r in rows)
        dyn = sum(r["n_death_dyn"] for r in rows)
        plato = np.mean([r["plato"] for r in rows])
        lossf = np.mean([r["loss_fim"] for r in rows])
        prev = nrev / len(eps) if eps else float("nan")
        zparks = [e["z_park"] for e in eps]
        zmed = np.median(zparks) if zparks else float("nan")
        n_deep = sum(1 for e in eps if e["z_park"] < -3.0)
        print(f"| {arm} | {len(rows)} | {plato:.4f} | {lossf:.4f} | {born} | {dyn} | "
              f"{len(eps)} | {nrev} | {prev:.2f} | {zmed:.2f} | {n_deep} |")

    # ---- H-c: pressao vs folga ----
    print("\n## H-c (pressao necessaria): gelu-press vs gelu-slack")
    for arm in ("gelu-press", "gelu-slack"):
        rows = D.get(arm, [])
        if rows:
            eps = [e for r in rows for e in r["episodios"]]
            print(f"  {arm}: plato_med={np.mean([r['plato'] for r in rows]):.4f} "
                  f"born={sum(r['n_nasce_morto'] for r in rows)} "
                  f"dyn={sum(r['n_death_dyn'] for r in rows)} eps={len(eps)} "
                  f"rev={sum(r['n_revive'] for r in rows)}")

    # ---- H-b: instrumento GT-AREA nas voltas (drive vs porte-MLP) ----
    print("\n## H-b (instrumento GT-AREA): episodios REVIVIDOS (todos os bracos)")
    print("| arm | seed | k | classe | t_morte | z_park | zmax_hi | t_area_drive | t_area_mlp | t_res | own_move | ordenado_drive |")
    print("|" + "|".join(["---"] * 12) + "|")
    n_ord = n_rev_tot = 0
    for arm, rows in D.items():
        for r in rows:
            for e in r["episodios"]:
                if e["revived"]:
                    n_rev_tot += 1
                    ordd = (e["t_area_drive"] >= 0 and e["t_morte"] <= e["t_area_drive"] < (e["t_res"] or 1 << 30))
                    n_ord += int(ordd)
                    print(f"| {arm} | {r['seed']} | {e['k']} | {e['classe']} | {e['t_morte']} | "
                          f"{e['z_park']:.2f} | {e['zmax_hi']:.2f} | {e['t_area_drive']} | "
                          f"{e['t_area_mlp']} | {e['t_res']} | {e['own_move']:.3f} | {ordd} |")
    print(f"\n  voltas totais={n_rev_tot}; com area_drive ABERTA antes da volta (t_morte<=t_area_drive<t_res)={n_ord}")

    # ---- S1: discordancia de sinal area_drive vs area_mlp (episodios mortos) ----
    print("\n## S1 (sinal GELU): area_drive vs area_mlp em episodios GELU")
    for arm, rows in D.items():
        if not arm.startswith("gelu"):
            continue
        eps = [e for r in rows for e in r["episodios"]]
        disc = sum(1 for e in eps if e["A_drive_fim"] * e["A_mlp_fim"] < 0)
        print(f"  {arm}: eps={len(eps)} discordancia-de-sinal(drive vs mlp)={disc}")

    # ---- validade ReLU ----
    print("\n## Validade ReLU (S2)")
    for arm in ("relu-kill-fz", "relu-kill-tr"):
        rows = D.get(arm, [])
        if rows:
            eps = [e for r in rows for e in r["episodios"]]
            rev = sum(r["n_revive"] for r in rows)
            own = np.mean([e["own_move"] for e in eps]) if eps else float("nan")
            print(f"  {arm}: eps={len(eps)} rev={rev} own_move_med={own:.4f} "
                  f"(fz: rev>0 seria upstream, own~0 => teorema MLP; tr: rev = drive de montante)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analise", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if a.run:
        main_run(a.workers)
    elif a.analysis:
        main_analise()
    else:
        print("uso: --run --workers 2  |  --analise")
