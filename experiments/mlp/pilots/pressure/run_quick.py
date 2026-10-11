"""Grade RÁPIDA etapa-4a.3 (pivô do coordenador) — spec expandida.

Física (sonda m4/β10/s42 confirmada por mim E pelo coordenador): a ressurreição-
por-área do T2' OPERA na MLP subparametrizada a T=0; a v1 (n=64, η=3e-4) só não a
via porque o tempo de retorno previsto (~|z_park|/deriva, deriva ~
η·v·R·σ(βz_park)·‖x*‖²/n) era ~1e7 passos, 100× além do horizonte. Rápido (n=16,
η=1e-3) encurta ~100× ⟹ retorno dentro de H. Sonda: previsto 1.49e5 vs obs 1.38e5.

Grade: m∈{4,6} × ativ∈{softplus-β10, β30, β300, relu} × T=0 × η=1e-3 × n=16 ×
H=2e5 × seeds 42-49 (8) = 64 runs. rec_neuronio=100 nas softplus (diagnóstico de
deriva); relu não precisa (grad≡0 no morto). Saída: results/{rot}-m{m}-s{seed}-
rapida.jsonl.gz (sufixo -rapida; convive com os 11 arquivos n=64 antigos).

TRIO por episódio DINÂMICO (t_morte>0), lido da v2 (o coordenador quer a v2):
(t_morte, t_area, t_res). A T=0 (determinístico) isto testa DIRETAMENTE se o
zero-de-área (t_area) ANTECEDE/prediz a ressurreição (t_res) — a lei do teorema
medida em rede real. Registrado também para born-dead que ressuscitam (β soft
revive born-dead RASO — mesmo mecanismo, refinando 'canal fechado').

Separação SEMPRE: born-dead (t_morte=0) vs dinâmica (t_morte>0). rev_real
(ground-truth zmax<0->zmax>0) é o contador-mestre, régua-independente.
Critérios: (ii') ress. dinâmica β10>0 em ≥metade das seeds; (iii) relu ZERO em
todas; (iv') taxa β10 ≥ β30 ≥ β300.
"""
import gzip
import json
import os
import time
from multiprocessing import Pool

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
ETAPA4 = os.path.dirname(os.path.dirname(AQUI))
OUT = os.path.join(AQUI, "results")
os.makedirs(OUT, exist_ok=True)

PASSOS = 200_000
ETA = 1e-3
T = 0.0
N_DADOS = 16
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]
N_SIGMA = 10.0

# (rótulo, ativ, beta, rec_neuronio)
ATIVS = [
    ("sp-b10", "softplus", 10.0, 100),
    ("sp-b30", "softplus", 30.0, 100),
    ("sp-b300", "softplus", 300.0, 100),
    ("relu", "relu", 300.0, 0),
]
MS = [4, 6]

TAREFAS = []
for rot, ativ, beta, recn in ATIVS:
    for m in MS:
        for s in SEEDS:
            TAREFAS.append(dict(rot=rot, ativ=ativ, beta=beta, m=m, seed=s,
                                recn=recn))


def _nome(t):
    return f"{t['rot']}-m{t['m']}-s{t['seed']}-rapida.jsonl.gz"


def _indexa_v2(reg2, passos):
    """{k: [ep_dict...]} de episódios v2 (completos+censurados), p/ o trio."""
    idx = {}
    for e in list(reg2.episodios) + reg2.censurados(passos):
        idx.setdefault(e["k"], []).append(e)
    return idx


def _trio_v2(idx_v2, k, t_morte_v3):
    """v2 (t_morte, t_area, t_sinal, t_res) do episódio de k mais próximo em t_morte."""
    cand = idx_v2.get(k, [])
    if not cand:
        return None
    e = min(cand, key=lambda ee: abs(ee["t_morte"] - t_morte_v3))
    return dict(t_morte=int(e["t_morte"]), t_area=int(e["t_area"]),
               t_sinal=int(e["t_sinal"]), t_res=int(e["t_res"]))


def _episodios(reg2, reg3, metr, beta, eta, n_dados, passos):
    idx_v2 = _indexa_v2(reg2, passos)
    serie = metr.get("serie")
    if serie is not None:
        st, szmax, sR, sv, sn2 = (serie["t"], serie["zmax"], serie["R"],
                                  serie["v"], serie["n2"])
    eps_v3 = [("completo", e) for e in reg3.episodios] + \
             [("censurado", e) for e in reg3.censurados(passos)]
    out = []
    for tp_, e in eps_v3:
        k, t0 = e["k"], e["t_morte"]
        revived = (tp_ == "completo")
        t1 = e["t_res"] if revived else passos
        z_min = float(e["z_min"])                    # z_park do coordenador
        sig_min = float(1.0 / (1.0 + np.exp(-beta * z_min)))
        row = dict(
            classe=("born" if t0 == 0 else "dyn"), revived=bool(revived), k=int(k),
            v3_t_morte=int(t0), v3_t_area=int(e["t_area"]),
            v3_t_sinal=int(e["t_sinal"]),
            v3_t_res=int(e["t_res"]) if revived else -1,
            dur_obs=int(e["t_res"] - t0) if revived else None,
            z_min=z_min, sigma_min=sig_min, profundo=bool(e.get("profundo", False)),
            v2=_trio_v2(idx_v2, k, t0))
        # deriva (estimador validado): margem MEDIANA enquanto morto
        if serie is not None:
            win = (st > t0) & (st <= t1)
            if win.sum() == 0:
                win = (st >= t0)
            if win.sum() > 0:
                z_med = float(np.median(szmax[win, k]))
                R = float(np.median(sR[win, k]))
                v = float(np.median(sv[win, k]))
                n2 = float(np.median(sn2[win, k]))
                sig_med = float(1.0 / (1.0 + np.exp(-beta * z_med)))
                drift = eta * v * R * sig_med * n2 / n_dados
                t_ret = (abs(z_med) / drift) if drift > 0 else None
                row.update(z_park_med=z_med, R=R, v=v, n2=n2, sigma_med=sig_med,
                           drift=drift, t_ret_pred=t_ret)
        out.append(row)
    return out


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
    import train_pressure as tp
    caminho = os.path.join(OUT, _nome(t))
    if os.path.exists(caminho):
        return f"skip {_nome(t)}"
    t0 = time.time()
    net, reg2, reg3, metr = tp.treina_pressao(
        m=t["m"], ativ=t["ativ"], beta=t["beta"], eta=ETA, T=T, passos=PASSOS,
        seed=t["seed"], n_dados=N_DADOS, n_sigma=N_SIGMA,
        rec_neuronio=t["recn"], rec_cada=100)

    lh = metr["loss_hist"]
    n10 = max(1, len(lh) // 10)
    loss_plato = float(np.mean(lh[-n10:]))
    eps = _episodios(reg2, reg3, metr, t["beta"], ETA, N_DADOS, PASSOS)
    born = [e for e in eps if e["classe"] == "born"]
    dyn = [e for e in eps if e["classe"] == "dyn"]
    born_rev = [e for e in born if e["revived"]]
    dyn_rev = [e for e in dyn if e["revived"]]
    v3_comp = list(reg3.episodios)
    v3_zmin_neg = [e for e in v3_comp if e["z_min"] < 0.0]

    rec = dict(
        rot=t["rot"], ativ=t["ativ"], beta=t["beta"], m=t["m"], seed=t["seed"],
        T=T, eta=ETA, n_dados=N_DADOS, passos=PASSOS, n_sigma=N_SIGMA,
        loss_inicial=float(lh[0]), loss_final=float(lh[-1]), loss_plato=loss_plato,
        n_mortos_real=metr["n_mortos_real"],
        n_revive_real=metr["n_revive_real"],
        n_neuronios_revive_real=metr["n_neuronios_revive_real"],
        n_born_dead=len(born), n_born_revividos=len(born_rev),
        n_dinamicas=len(dyn), n_dinamicas_revividas=len(dyn_rev),
        v2_n_completos=len(reg2.episodios),
        v2_n_censurados=len(reg2.censurados(PASSOS)),
        v3_n_completos=len(v3_comp),
        v3_n_censurados=len(reg3.censurados(PASSOS)),
        v3_n_completos_zmin_neg=len(v3_zmin_neg),
        episodios=eps,
        segundos=round(time.time() - t0, 1),
    )
    with gzip.open(caminho, "wt") as f:
        json.dump(dict(rec, loss_hist=lh), f)
    trio = "; ".join(
        f"k{e['k']}({e['v2']['t_morte'] if e['v2'] else '?'}→"
        f"a{e['v2']['t_area'] if e['v2'] else '?'}→{e['v3_t_res']})"
        for e in dyn_rev) or "-"
    return ("OK " + _nome(t) + f": {rec['segundos']:.0f}s plato={loss_plato:.3f} "
            f"born={len(born)}(rev{len(born_rev)}) dyn={len(dyn)}(rev{len(dyn_rev)}) "
            f"rev_real={rec['n_revive_real']} v3zneg={rec['v3_n_completos_zmin_neg']} "
            f"trio[{trio}]")


def agrega():
    linhas = []
    for t in TAREFAS:
        p = os.path.join(OUT, _nome(t))
        if not os.path.exists(p):
            continue
        d = json.load(gzip.open(p, "rt"))
        d.pop("loss_hist", None)
        linhas.append(d)
    with open(os.path.join(OUT, "pressao-rapida.jsonl"), "w") as f:
        for d in linhas:
            f.write(json.dumps(d) + "\n")
    return len(linhas)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=None, help="rot (sp-b10) | mN | sN | vazio=tudo")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--agrega-so", action="store_true")
    a = ap.parse_args()
    if a.agrega_so:
        print("agregado:", agrega(), "linhas")
        raise SystemExit
    tarefas = TAREFAS
    if a.subset:
        s = a.subset
        tarefas = [t for t in TAREFAS if s in (t["rot"], f"m{t['m']}", f"s{t['seed']}")]
    print(f"{len(tarefas)} runs, Pool {a.workers}, {PASSOS} passos, n={N_DADOS}, "
          f"η={ETA}, T={T}, seeds={SEEDS}", flush=True)
    with Pool(a.workers) as p:
        for msg in p.imap_unordered(roda, tarefas):
            print(msg, flush=True)
    print("agregado:", agrega(), "linhas")
    print("RAPIDA OK")
