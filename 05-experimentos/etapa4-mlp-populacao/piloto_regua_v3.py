"""Etapa 4a.2 — PILOTO da régua v3 (ARQUIVO NOVO; não toca no 4a pré-registrado).

Três medidas exploratórias (seeds fora de 1700-1799 — 42-99), rotuladas
[EXPLORATÓRIO], sobre a MESMA dinâmica do harness, trocando SÓ a régua por
rebind de harness_mlp.Regua (o arquivo do harness não é modificado):

  1. toy_preservado — a calibração do toy (episódio longo, predição-área)
     sobrevive ao debounce? erros de t_area antes (v2) e depois (v3).
  2. frac_chatter_rejeitada — no braço semgrad (difusão pura, grad≡0 ⟹ TODO
     episódio é chatter por construção): 1 − N_v3/N_v2.
  3. relu_quieto — relu-dura vs softplus em T>0: v2 os iguala (o diagnóstico
     que invalidou o 4a); v3 deve calar a relu (sem gradiente não há
     ressurreição física) e reter mais em softplus se houver física.

Uso:  ../../venv/bin/python piloto_regua_v3.py [--toy] [--mlp] [--nsigma 10]
      (sem flags roda os dois). nice -19 recomendado.
"""
import argparse
import time

import numpy as np
import torch

import harness_mlp
from harness_mlp import treina
from regua_v3 import ReguaV3

torch.set_num_threads(1)
ReguaV2 = harness_mlp.Regua          # referência ANTES de qualquer rebind


class ParRegua:
    """Observa a MESMA trajetória com v2 e v3 de uma vez (pareamento exato;
    a régua é passiva ⟹ não altera a dinâmica). Entra no harness por rebind."""

    def __init__(self, m, eta, T, n_sigma=10.0, c_p=1.0):
        self.v2 = ReguaV2(m, eta, T)
        self.v3 = ReguaV3(m, eta, T, n_sigma=n_sigma, c_p=c_p)
        self.episodios = []          # lido por treina só sob para_cedo (MLP: off)

    def observa(self, *a):
        self.v2.observa(*a)
        self.v3.observa(*a)


def _completos(eps):
    return [e for e in eps if e["t_res"] >= 0]


def _med(xs):
    return float(np.median(xs)) if xs else None


# --------------------------- 1) TOY PRESERVADO ------------------------------
def toy_preservado(n_sigma):
    from calibracao_regua_v2 import roda_toy   # usa treina->Regua (rebindável)
    print(f"\n=== 1) TOY PRESERVADO (n_σ={n_sigma}) ===")

    def erro(ep):
        return abs(ep["t_area"] - ep["t_res"]) / ep["t_res"]

    # (a) quase-determinístico: predição-área. casos do calibracao_regua_v2 (a).
    casos = [("β=300 η=1e-3/3", 300.0, 1e-3 / 3), ("β=1000 η=1e-4", 1000.0, 1e-4)]
    print("(a) quase-det (T=1e-8) — erro relativo da predição-área |t_area−t_res|/t_res")
    lin_a = []
    for rot, beta, eta in casos:
        harness_mlp.Regua = ReguaV2
        _, r2, _, _ = treina(m=2, ativ="softplus", beta=beta, eta=eta, T=1e-8,
                             passos=1_000_000, seed=1, params0=_P0(),
                             dados=_TOY(), para_cedo=True)
        e2 = [p for p in r2.episodios if p["k"] == 0][0]
        harness_mlp.Regua = lambda m, et, T, ns=n_sigma: ReguaV3(m, et, T, n_sigma=ns)
        _, r3, _, _ = treina(m=2, ativ="softplus", beta=beta, eta=eta, T=1e-8,
                             passos=1_000_000, seed=1, params0=_P0(),
                             dados=_TOY(), para_cedo=True)
        e3 = [p for p in r3.episodios if p["k"] == 0][0]
        print(f"   {rot:18s}: v2 t_area={e2['t_area']:6d} t_res={e2['t_res']:6d} "
              f"err={erro(e2)*100:5.2f}%  |  v3 t_area={e3['t_area']:6d} "
              f"t_res={e3['t_res']:6d} err={erro(e3)*100:5.2f}%  "
              f"profundo={e3.get('profundo')} D={e3['dwell']}")
        lin_a.append((rot, erro(e2), erro(e3)))

    # (b) sob ruído T=1e-3, 40 seeds: área-vs-sinal, pareado por seed.
    print("(b) sob ruído (T=1e-3, 40 seeds) — área vence sinal e erro rel. mediano")
    out_b = {}
    for tag, fabrica in (("v2", lambda: setattr(harness_mlp, "Regua", ReguaV2)),
                         ("v3", lambda: setattr(
                             harness_mlp, "Regua",
                             lambda m, et, T, ns=n_sigma: ReguaV3(m, et, T, n_sigma=ns)))):
        fabrica()
        ganha = tot = ncens = nprof = 0
        relA, relS = [], []
        for sd in range(3000, 3040):
            ep, _, _ = roda_toy(1e-3, sd, 1e-3)
            if ep is None:
                ncens += 1; continue
            if ep["t_area"] < 0 or ep["t_sinal"] < 0:
                continue
            tot += 1
            dur = max(ep["t_res"] - ep["t_morte"], 1)
            nprof += int(ep.get("profundo", True))
            eA = abs(ep["t_res"] - ep["t_area"]); eS = abs(ep["t_res"] - ep["t_sinal"])
            relA.append(eA / dur); relS.append(eS / dur)
            ganha += int(eA < eS)
        out_b[tag] = dict(tot=tot, ncens=ncens, nprof=nprof,
                          ganha=ganha, relA=_med(relA), relS=_med(relS))
        print(f"   {tag}: completos={tot} censurados={ncens} profundos={nprof} "
              f"| área vence sinal {ganha}/{tot} | "
              f"err.rel mediano ÁREA={out_b[tag]['relA']:.3f} SINAL={out_b[tag]['relS']:.3f}")
    harness_mlp.Regua = ReguaV2
    return dict(quase_det=lin_a, ruido=out_b)


def _TOY():
    return (torch.tensor([[0.5], [1.5]]), torch.tensor([[0.5], [1.0]]))


def _P0():
    return dict(w=[[-0.5], [1.144]], b=[0.27, 0.261], v=[1.0, 1.0])


# ----------------------- 2+3) PILOTO MLP (chatter/relu) ---------------------
def piloto_mlp(n_sigma, m=64, seeds=(42, 43, 44), passos=100_000, T=1e-3, eta=3e-4):
    print(f"\n=== 2+3) PILOTO MLP (m={m}, T={T}, {len(seeds)} seeds, "
          f"{passos} passos, n_σ={n_sigma}, dwell D={int(np.ceil(n_sigma**2))}) ===")
    bracos = [("softplus", "softplus", dict()),
              ("relu", "relu", dict()),
              ("semgrad", "softplus", dict(sem_grad=True))]
    agg = {b[0]: dict(v2=[], v3=[], v3p=[], dur_v2=[], dur_v3=[], cens_v3=0)
           for b in bracos}
    for nome, ativ, kw in bracos:
        for sd in seeds:
            harness_mlp.Regua = lambda mm, et, TT, ns=n_sigma: ParRegua(
                mm, et, TT, n_sigma=ns)
            t0 = time.perf_counter()
            _, reg, _, _ = treina(m=m, ativ=ativ, eta=eta, T=T, passos=passos,
                                  seed=sd, **kw)
            c2 = _completos(reg.v2.episodios)
            c3 = reg.v3.episodios                      # já só completos
            c3p = [e for e in c3 if e["profundo"]]
            agg[nome]["v2"].append(len(c2))
            agg[nome]["v3"].append(len(c3))
            agg[nome]["v3p"].append(len(c3p))
            agg[nome]["dur_v2"] += [e["t_res"] - e["t_morte"] for e in c2]
            agg[nome]["dur_v3"] += [e["t_res"] - e["t_morte"] for e in c3]
            agg[nome]["cens_v3"] += len(reg.v3.censurados(passos))
            print(f"   {nome:8s} s{sd}: v2={len(c2):4d} v3={len(c3):3d} "
                  f"v3-prof={len(c3p):3d}  ({round(time.perf_counter()-t0,1)}s)",
                  flush=True)
    harness_mlp.Regua = ReguaV2

    print("\n   --- resumo (soma sobre seeds) ---")
    res = {}
    for nome, _, _ in bracos:
        a = agg[nome]
        v2, v3, v3p = sum(a["v2"]), sum(a["v3"]), sum(a["v3p"])
        res[nome] = dict(v2=v2, v3=v3, v3p=v3p, cens_v3=a["cens_v3"],
                         med_dur_v2=_med(a["dur_v2"]), med_dur_v3=_med(a["dur_v3"]),
                         rej=(1 - v3 / v2) if v2 else None,
                         rej_prof=(1 - v3p / v2) if v2 else None)
        print(f"   {nome:8s}: v2={v2:5d} (med.dur {res[nome]['med_dur_v2']}) | "
              f"v3={v3:4d} (med.dur {res[nome]['med_dur_v3']}) prof={v3p:4d} "
              f"cens={a['cens_v3']:3d} | rejeitado={res[nome]['rej']:.3f} "
              f"(prof {res[nome]['rej_prof']:.3f})")

    # métrica 2: chatter rejeitado no semgrad (difusão pura = 100% chatter)
    sg = res["semgrad"]
    print(f"\n   [2] frac_chatter_rejeitada (semgrad, difusão pura):")
    print(f"       completos: {sg['rej']:.3f}  |  exigindo profundidade: "
          f"{sg['rej_prof']:.3f}")
    # métrica 3: relu deve calar vs softplus (v2 os igualava)
    sp, rl = res["softplus"], res["relu"]
    print(f"   [3] relu_quieto (v2 igualava relu≈softplus — diagnóstico do 4a):")
    print(f"       v2: softplus={sp['v2']} relu={rl['v2']} (razão relu/softplus "
          f"{rl['v2']/max(sp['v2'],1):.2f})")
    print(f"       v3: softplus={sp['v3']} relu={rl['v3']} (razão "
          f"{rl['v3']/max(sp['v3'],1):.2f})")
    print(f"       v3-prof: softplus={sp['v3p']} relu={rl['v3p']} (razão "
          f"{rl['v3p']/max(sp['v3p'],1):.2f})")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--mlp", action="store_true")
    ap.add_argument("--nsigma", type=float, default=10.0)
    ap.add_argument("--passos", type=int, default=100_000)
    a = ap.parse_args()
    fazer_toy = a.toy or not (a.toy or a.mlp)
    fazer_mlp = a.mlp or not (a.toy or a.mlp)
    if fazer_toy:
        toy_preservado(a.nsigma)
    if fazer_mlp:
        piloto_mlp(a.nsigma, passos=a.passos)
