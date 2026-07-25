"""PILOTO-DE-TAXA da etapa-5 (transformer FFN) — pre-requisito do confirmatorio.

Herda a disciplina do piloto-de-taxa 4a.3: mede TAXAS com CI para dimensionar o
confirmatorio (S seeds, n*, janela, bracos, MATAs). EXPLORATORIO — NADA pre-registrado.

GRADE (mission): bracos {sp-b3, sp-b10, gelu, relu} x {attn-treina, attn-frozen}
  x m_ffn=4 x seeds 100-119 (20) = 160 runs. H=8e4 (dimensionado por calibracao:
  climb da margem ~= revive em ~4-8e4 p/ injecao rasa; deep essencialmente terminal
  no horizonte). Injecao controlada em DUAS profundidades por run: rasa z_park=-0.3
  (ACIMA da barreira GELU z*=-0.7518 => revivivel) e profunda z_park=-1.0 (ABAIXO =>
  teste de terminalidade P-ii/P-iv). Motor: motor_e5b.treina_e5b (z_dead=0 unificado).

Uso (destacado, retomavel, ==2 workers pesados, cores 0-7, nice 19):
  export PYTHONPATH=<esta pasta>
  setsid taskset -c 0-7 nice -n 19 PYTHON piloto_taxa_e5.py --run --workers 2 > taxa.log 2>&1 &
  PYTHON piloto_taxa_e5.py --analise > TAXA-analise.txt
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results-taxa")
SEEDS = list(range(100, 120))            # 20 seeds
H = 80_000
KILL_DEPTHS = (0.3, 1.0)                  # rasa (>z*=-0.7518) ; profunda (<z*)
BASE = dict(N=24, S=3, d=16, H=2, eta=1e-3, passos=H, rec_cada=300,
            z_dead=0.0, z_alive=0.0, kill_depths=KILL_DEPTHS)

# braco -> (ativ, beta) ; fator -> freeze_attn
ARMS = {"sp-b3": ("softplus", 3.0), "sp-b10": ("softplus", 10.0),
        "gelu": ("gelu", 1.0), "relu": ("relu", 1.0)}
FACTORS = {"train": False, "frozen": True}


def _cell(arm, fac):
    return f"{arm}-{fac}"


def _kwargs(arm, fac, seed):
    ativ, beta = ARMS[arm]
    kw = dict(BASE)
    kw.update(m_ffn=4, ativ=ativ, beta=beta, freeze_attn=FACTORS[fac], seed=seed)
    return kw


def run_one(job):
    arm, fac, seed = job
    fp = os.path.join(RES, f"{_cell(arm, fac)}-s{seed}.json")
    if os.path.exists(fp):                # RETOMAVEL: pula o que ja existe
        return arm, fac, seed, "skip"
    import torch
    torch.set_num_threads(1)
    import motor_e5b as MB
    kw = _kwargs(arm, fac, seed)
    t0 = time.time()
    mt = MB.treina_e5b(**kw)
    dt = time.time() - t0
    out = dict(
        arm=arm, fac=fac, seed=seed, secs=round(dt, 1), cfg=mt["cfg"],
        plato=mt["plato"], loss_fim=mt["loss_fim"],
        n_nasce_morto=mt["n_nasce_morto"], n_born_natural=mt["n_born_natural"],
        n_injetadas=mt["n_injetadas"], n_death_dyn=len(mt["death_events"]),
        n_mortos=mt["n_mortos"], n_revive=mt["n_revive"],
        n_neuronios_revive=mt["n_neuronios_revive"], episodios=mt["episodios"])
    os.makedirs(RES, exist_ok=True)
    tmp = fp + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, fp)                   # escrita atomica
    print(f"OK {_cell(arm,fac)}-s{seed}: {dt:.0f}s plato={out['plato']:.3f} "
          f"bornN={out['n_born_natural']} dyn={out['n_death_dyn']} rev={out['n_revive']}",
          flush=True)
    return arm, fac, seed, "done"


def main_run(workers):
    os.makedirs(RES, exist_ok=True)
    jobs = [(arm, fac, s) for arm in ARMS for fac in FACTORS for s in SEEDS]
    todo = [j for j in jobs if not os.path.exists(
        os.path.join(RES, f"{_cell(j[0], j[1])}-s{j[2]}.json"))]
    print(f"== piloto_taxa_e5: {len(jobs)} runs total, {len(todo)} pendentes, "
          f"{workers} workers, H={H}, kill_depths={KILL_DEPTHS} ==", flush=True)
    t0 = time.time()
    with Pool(workers, maxtasksperchild=1) as p:
        for _ in p.imap_unordered(run_one, todo):
            pass
    print(f"== FIM: {len(todo)} runs em {time.time()-t0:.0f}s ==", flush=True)


# ============================== ANALISE ==============================
def _cp(k, n, alpha=0.05):
    """Clopper-Pearson exato (beta ppf). Retorna (p, lo, hi)."""
    from scipy.stats import beta
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return p, lo, hi


def _boot_rate(per_seed_counts, per_seed_exposure, B=4000, seed=0):
    """Bootstrap POR SEED da taxa sum(counts)/sum(exposure). Retorna (ponto, lo, hi)."""
    import numpy as np
    c = np.asarray(per_seed_counts, float); e = np.asarray(per_seed_exposure, float)
    pt = c.sum() / e.sum() if e.sum() > 0 else float("nan")
    rng = np.random.default_rng(seed); n = len(c); out = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        es = e[idx].sum()
        out.append(c[idx].sum() / es if es > 0 else np.nan)
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return pt, float(lo), float(hi)


def _load():
    out = {}
    for arm in ARMS:
        for fac in FACTORS:
            rows = []
            for s in SEEDS:
                fp = os.path.join(RES, f"{_cell(arm, fac)}-s{s}.json")
                if os.path.exists(fp):
                    with open(fp) as f:
                        rows.append(json.load(f))
            out[(arm, fac)] = rows
    return out


def main_analise():
    import numpy as np
    D = _load()
    n_have = sum(len(v) for v in D.values())
    print(f"# Piloto-de-taxa etapa-5 (transformer FFN) — analise")
    print(f"# runs carregados: {n_have}/160  | H={H} kill_depths={KILL_DEPTHS} "
          f"(rasa>{-0.7518:.4f}=barreira; profunda<)  | seeds {SEEDS[0]}-{SEEDS[-1]}\n")

    # ---------- Tabela mestra por braco x fator ----------
    print("## Tabela mestra (por braco x fator; 20 seeds/celula, 2 injetadas/run)\n")
    hdr = ("cell", "runs", "plato", "bornN", "lam_dyn/run[CI]", "inj", "rev",
           "term", "cens", "p(rev|inj)[CP]", "own_med")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for arm in ARMS:
        for fac in FACTORS:
            rows = D[(arm, fac)]
            if not rows:
                print(f"| {_cell(arm,fac)} | 0 |" + " - |" * (len(hdr) - 2)); continue
            eps = [e for r in rows for e in r["episodios"]]
            inj = [e for e in eps if e["injetada"]]
            rev = sum(1 for e in inj if e["revived"])
            term = sum(1 for e in inj if (not e["revived"]) and e["t_res"] is None
                       and e["injetada"])   # censurado no fim = nao voltou
            # aqui term==cens (mesmo objeto): injetada nao-revivida = censurada no fim de H
            cens = sum(1 for e in inj if not e["revived"])
            plato = np.mean([r["plato"] for r in rows])
            bornN = sum(r["n_born_natural"] for r in rows)
            dyn_counts = [r["n_death_dyn"] for r in rows]
            lam, llo, lhi = _boot_rate(dyn_counts, [1] * len(rows))
            p, lo, hi = _cp(rev, len(inj))
            own = np.median([e["own_move"] for e in inj]) if inj else float("nan")
            print(f"| {_cell(arm,fac)} | {len(rows)} | {plato:.3f} | {bornN} | "
                  f"{lam:.3f}[{llo:.2f},{lhi:.2f}] | {len(inj)} | {rev} | "
                  f"{cens} | {cens} | {p:.2f}[{lo:.2f},{hi:.2f}] | {own:.4f} |")
    print("\n_term=cens: injetada nao-revivida = censurada no fim de H (nao ha estado "
          "'terminal' absoluto; terminalidade e inferida do canal/own_move e da profundidade)._")

    # ---------- p(volta|inj) SEPARADO por profundidade (rasa vs profunda) ----------
    print("\n## p(volta|injecao) por PROFUNDIDADE  (rasa z_park=-0.3 >z* ; profunda -1.0 <z*)\n")
    print("| cell | rasa: rev/n [CP] | rasa τ=q0.9(t_res) | profunda: rev/n [CP] | prof τ=q0.9 |")
    print("|---|---|---|---|---|")
    for arm in ARMS:
        for fac in FACTORS:
            rows = D[(arm, fac)]
            if not rows:
                continue
            inj = [e for r in rows for e in r["episodios"] if e["injetada"]]
            def _blk(pred):
                sub = [e for e in inj if pred(e)]
                rv = [e for e in sub if e["revived"]]
                p, lo, hi = _cp(len(rv), len(sub))
                ts = [e["t_res"] for e in rv if e["t_res"] is not None]
                tau = (f"{int(np.quantile(ts,0.9))}" if ts else "-")
                med = (f" med={int(np.median(ts))}" if ts else "")
                return f"{len(rv)}/{len(sub)} [{lo:.2f},{hi:.2f}]", f"{tau}{med}"
            rasa_p, rasa_tau = _blk(lambda e: not e["deep"])
            prof_p, prof_tau = _blk(lambda e: e["deep"])
            print(f"| {_cell(arm,fac)} | {rasa_p} | {rasa_tau} | {prof_p} | {prof_tau} |")

    # ---------- Return-times agregados por braco (pool train+frozen) ----------
    print("\n## Return-time t_res (ground-truth) por braco — todos os injetados revividos\n")
    print("| braco | n voltas | min | mediana | τ=q0.9 | max |")
    print("|---|---|---|---|---|---|")
    for arm in ARMS:
        ts = [e["t_res"] for fac in FACTORS for r in D[(arm, fac)]
              for e in r["episodios"] if e["injetada"] and e["revived"] and e["t_res"]]
        if ts:
            ts = sorted(ts)
            print(f"| {arm} | {len(ts)} | {ts[0]} | {int(np.median(ts))} | "
                  f"{int(np.quantile(ts,0.9))} | {ts[-1]} |")
        else:
            print(f"| {arm} | 0 | - | - | - | - |")

    # ---------- Canal (drive-proprio vs montante): train vs frozen ----------
    print("\n## Atribuicao de canal (P-iii/P-iv): voltas por fator e canal\n")
    print("| braco | frozen: rev(proprio) | train: rev | train canais | own_move(train rev) |")
    print("|---|---|---|---|---|")
    for arm in ARMS:
        fz = [e for r in D[(arm, "frozen")] for e in r["episodios"]
              if e["injetada"] and e["revived"]]
        tr = [e for r in D[(arm, "train")] for e in r["episodios"]
              if e["injetada"] and e["revived"]]
        canais = {}
        for e in tr:
            canais[e["canal"]] = canais.get(e["canal"], 0) + 1
        own_tr = (f"{np.median([e['own_move'] for e in tr]):.4f}" if tr else "-")
        print(f"| {arm} | {len(fz)} | {len(tr)} | "
              f"{canais if canais else '-'} | {own_tr} |")

    # ---------- Instrumento GT-AREA: trios A_drive nas voltas ----------
    print("\n## GT-AREA (A_drive): trios das voltas (t_morte <= t_area_drive < t_res)\n")
    print("| cell | seed | depth | z_park | zhi | t_area_drive | t_res | A_drive | A_mlp | ordenado | sinal(drive,mlp) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    n_ord = n_rev = 0; disc = 0
    for arm in ARMS:
        for fac in FACTORS:
            for r in D[(arm, fac)]:
                for e in r["episodios"]:
                    if e["injetada"] and e["revived"]:
                        n_rev += 1
                        ordd = (e["t_area_drive"] >= 0 and
                                e["t_morte"] <= e["t_area_drive"] < (e["t_res"] or 1 << 30))
                        n_ord += int(ordd)
                        sd = "opostos" if e["A_drive_fim"] * e["A_mlp_fim"] < 0 else "iguais"
                        disc += int(e["A_drive_fim"] * e["A_mlp_fim"] < 0)
                        print(f"| {_cell(arm,fac)} | {r['seed']} | {e['injdepth']} | "
                              f"{e['z_park']:.2f} | {e['zmax_hi']:.2f} | {e['t_area_drive']} | "
                              f"{e['t_res']} | {e['A_drive_fim']:+.2e} | {e['A_mlp_fim']:+.2e} | "
                              f"{ordd} | {sd} |")
    print(f"\n  voltas={n_rev}; area_drive aberta antes da volta (t_morte<=t_area_drive<t_res)={n_ord}; "
          f"sinal(drive)!=sinal(mlp)={disc}")

    # ---------- S1: discordancia de sinal drive vs mlp em TODOS os episodios GELU ----------
    print("\n## S1 (sign-flip GELU): sinal A_drive vs A_mlp em episodios GELU (todos)\n")
    for fac in FACTORS:
        eps = [e for r in D[("gelu", fac)] for e in r["episodios"]]
        dd = sum(1 for e in eps if e["A_drive_fim"] * e["A_mlp_fim"] < 0)
        deep = [e for e in eps if e["deep"]]
        dd_deep = sum(1 for e in deep if e["A_drive_fim"] * e["A_mlp_fim"] < 0)
        print(f"  gelu-{fac}: eps={len(eps)} sinais-opostos(drive,mlp)={dd} "
              f"| profundos={len(deep)} opostos-profundos={dd_deep}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analise", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if a.run:
        main_run(a.workers)
    elif a.analise:
        main_analise()
    else:
        print("uso: --run --workers 2  |  --analise")
