"""Etapa 4a.2 — CALIBRAÇÃO da régua v3 (trilho 1b; ARQUIVO NOVO, não toca no 4a).

Contexto [RESULTADO 2026-07-24]: a run confirmatória do 4a foi INVALIDADA pelo
controle P-NULL. Diagnóstico (06-notas/2026-07-24-...): em T>0 a `Regua` v2 mede
CHATTER de fronteira (59% dos episódios ≤5 passos); ReLU-dura "ressuscita" na
MESMA taxa que softplus (impossível por gradiente: grad≡0 no morto ⟹ ruído). A
régua v3 (regua_v3.py, trilho 1a) adiciona um DEBOUNCE de base física (histerese
de dois limiares + dwell D=⌈n_σ²⌉ + profundidade c_p·lim_morte), tudo derivado de
UMA escala: σ_z=√(2ηT)·‖(x_j*,1)‖.

Este script CALIBRA a v3 com três testes de aceitação/rejeição, cada um com
números reais e comando reproduzível:

  (1) TOY PRESERVADO  — o toy-em-harness (m=2, init do toy, T=1e-3, 40 seeds
      3100-3139) tem UM episódio longo físico por seed (dur ~1.5-3k passos, 2
      switches). O debounce NÃO pode matá-lo nem estragar a predição-área
      (erro relativo mediano ~0.53 na v2). Reporta v2 (antes) vs v3 (depois).

  (2) CHATTER REJEITADO — toy-data com margem PAIRANDO no limiar (init em
      lim_morte) sob difusão pura (sem_grad, T=1e-3): a v2 conta dezenas de
      episódios curtos; a v3 deve rejeitar a grande maioria (dwell+profundidade
      barram a difusão). Reporta frações.

  (3) FUMAÇA relu-vs-softplus (MLP m=64, T=1e-3, 2e4 passos, seeds 42-44): ESTE
      é o critério que reprovou a v2 (relu≈softplus em T>0). Pós-debounce a relu
      deve calar (grad=0 ⟹ só difusão ⟹ barrada) e a softplus reter episódios
      físicos. Se a v3 NÃO separar relu de softplus, ela está REPROVADA também —
      e o script diz isso sem dó, com os números do fracasso. Um probe T=0
      (difusão nula) acompanha como separação-limpa de controle (relu grad=0 ⟹
      0 ressurreições; softplus revive por gradiente).

Etiqueta global: [EXPLORATÓRIO] (seeds fora de 1700-1799). A régua entra no
harness por rebind de harness_mlp.Regua (o arquivo pré-registrado do 4a NÃO é
tocado). Uso:
    ../../venv/bin/python calibracao_regua_v3.py [--toy] [--chatter] [--relu]
        [--nsigma 10] [--cp 1.0] [--t0]        (sem flags de teste: roda os 3)
"""
import argparse
import time

import numpy as np
import torch

import harness_mlp
from harness_mlp import treina, episodio_limpo
from regua_v3 import ReguaV3

torch.set_num_threads(1)
ReguaV2 = harness_mlp.Regua           # referência ANTES de qualquer rebind

TOY_X = torch.tensor([[0.5], [1.5]])
TOY_Y = torch.tensor([[0.5], [1.0]])
P0 = dict(w=[[-0.5], [1.144]], b=[0.27, 0.261], v=[1.0, 1.0])


def _med(xs):
    return float(np.median(xs)) if len(xs) else None


# ============================================================================
# régua-par: observa a MESMA trajetória com v2 e v3 (a régua é passiva ⟹ não
# altera a dinâmica; pareamento exato). Entra no harness por rebind.
# ============================================================================
class Par:
    def __init__(self, m, eta, T, n_sigma, c_p, para_por_v3=False):
        self.v2 = ReguaV2(m, eta, T)
        self.v3 = ReguaV3(m, eta, T, n_sigma=n_sigma, c_p=c_p)
        self._para_por_v3 = para_por_v3

    def observa(self, *a):
        self.v2.observa(*a)
        self.v3.observa(*a)

    @property
    def episodios(self):
        # treina só lê isto sob para_cedo. Ao parar por v3, para_cedo dispara
        # quando a v3 FECHA seu 1º episódio de k=0 (o físico; chatter é rejeitado
        # e nunca entra em v3.episodios). Sob para_por_v3=False devolve [] ⟹
        # para_cedo nunca dispara (rodada completa).
        return self.v3.episodios if self._para_por_v3 else []


def _bind(m, eta, T, n_sigma, c_p, para_por_v3=False):
    harness_mlp.Regua = lambda mm, et, TT: Par(mm, et, TT, n_sigma, c_p,
                                               para_por_v3=para_por_v3)


# ============================================================================
# (1) TOY PRESERVADO
# ============================================================================
def _ep_k0(eps):
    e = [p for p in eps if p["k"] == 0]
    return e[0] if e else None


def _rel(ep, chave):
    dur = max(ep["t_res"] - ep["t_morte"], 1)
    return abs(ep[chave] - ep["t_res"]) / dur


def toy_preservado(n_sigma, c_p, seeds=range(3100, 3140), T=1e-3, eta=1e-3,
                   passos=15_000):
    print(f"\n{'='*74}\n(1) TOY PRESERVADO  — m=2 toy-em-harness, T={T}, "
          f"{len(list(seeds))} seeds, n_σ={n_sigma} (D={int(np.ceil(n_sigma**2))}) "
          f"c_p={c_p}\n{'='*74}")
    print("    a lei do toy: 1 episódio LONGO físico/seed (born-dead, 2 switches).")
    print("    o debounce deve PRESERVAR dur e a predição-área |t_area−t_res|/dur.")
    v2 = dict(dur=[], relA=[], relS=[], win=0, tot=0, cens=0)
    v3 = dict(dur=[], relA=[], relS=[], win=0, tot=0, cens=0, prof=0,
              tm_match=0, dres=[])
    t0 = time.perf_counter()
    for sd in seeds:
        _bind(2, eta, T, n_sigma, c_p, para_por_v3=True)
        _, reg, _, _ = treina(m=2, ativ="softplus", beta=300.0, eta=eta, T=T,
                              passos=passos, seed=sd, params0=P0,
                              dados=(TOY_X, TOY_Y), para_cedo=True)
        e2, e3 = _ep_k0(reg.v2.episodios), _ep_k0(reg.v3.episodios)
        tr2 = e2["t_res"] if e2 else -9
        tr3 = e3["t_res"] if e3 else -9
        print(f"      s{sd}: v2 t_res={tr2:6d} | v3 t_res={tr3:6d} "
              f"prof={e3.get('profundo') if e3 else None}  "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
        # v2
        if e2 is None or e2["t_res"] < 0 or e2["t_area"] < 0 or e2["t_sinal"] < 0:
            v2["cens"] += 1
        else:
            v2["tot"] += 1
            v2["dur"].append(e2["t_res"] - e2["t_morte"])
            v2["relA"].append(_rel(e2, "t_area")); v2["relS"].append(_rel(e2, "t_sinal"))
            v2["win"] += int(_rel(e2, "t_area") < _rel(e2, "t_sinal"))
        # v3
        if e3 is None or e3["t_res"] < 0 or e3["t_area"] < 0 or e3["t_sinal"] < 0:
            v3["cens"] += 1
        else:
            v3["tot"] += 1
            v3["dur"].append(e3["t_res"] - e3["t_morte"])
            v3["relA"].append(_rel(e3, "t_area")); v3["relS"].append(_rel(e3, "t_sinal"))
            v3["win"] += int(_rel(e3, "t_area") < _rel(e3, "t_sinal"))
            v3["prof"] += int(e3.get("profundo", False))
            if e2 is not None:
                v3["tm_match"] += int(e3["t_morte"] == e2["t_morte"])
                if e2["t_res"] >= 0:
                    v3["dres"].append(e3["t_res"] - e2["t_res"])
    harness_mlp.Regua = ReguaV2
    dt = time.perf_counter() - t0

    def linha(tag, d, extra=""):
        print(f"    {tag}: completos={d['tot']:2d} censurados={d['cens']:2d} | "
              f"med.dur={_med(d['dur'])} | área vence sinal {d['win']}/{d['tot']} | "
              f"err.rel mediano ÁREA={_med(d['relA']):.3f} SINAL={_med(d['relS']):.3f}"
              f"{extra}")
    linha("v2 (antes)", v2)
    linha("v3 (depois)", v3, f" | profundos={v3['prof']}/{v3['tot']} "
          f"t_morte==v2: {v3['tm_match']}/{v3['tot']} "
          f"Δt_res(v3−v2) med={_med(v3['dres'])}")
    # veredito de preservação
    ok_dur = (v2["tot"] and v3["tot"] and _med(v3["dur"]) is not None
              and _med(v3["dur"]) >= 0.8 * _med(v2["dur"]))
    ok_cens = v3["cens"] <= v2["cens"] + 2
    ok_area = (_med(v3["relA"]) is not None and _med(v3["relA"]) <= _med(v2["relA"]) + 0.10)
    ver = "PRESERVADO" if (ok_dur and ok_cens and ok_area) else "DEGRADOU"
    print(f"    → {ver}  (dur≥80%v2:{ok_dur} censura≈:{ok_cens} "
          f"área≤v2+0.10:{ok_area})  [{dt:.0f}s]")
    return dict(v2=v2, v3=v3, veredito=ver)


# ============================================================================
# (2) CHATTER REJEITADO — difusão pura na fronteira (toy 1-ponto)
# ============================================================================
def chatter_rejeitado(n_sigma, c_p, m=32, T=1e-3, eta=1e-3, passos=20_000):
    # 1 ponto ⟹ sem ambiguidade de líder: jstar=0 sempre, ‖(1,1)‖=√2.
    xj = 1.0
    norm = np.sqrt(xj ** 2 + 1.0)
    lim = 2.0 * np.sqrt(1.0 * T) / norm       # n=1
    sig = np.sqrt(2.0 * eta * T) * norm
    X = torch.tensor([[xj]]); Y = torch.tensor([[0.5]])
    print(f"\n{'='*74}\n(2) CHATTER REJEITADO — toy 1-ponto, margem PAIRANDO no "
          f"limiar, difusão pura\n{'='*74}")
    print(f"    m={m} neurônios init em z=lim_morte={lim:.4f} (w=0,b=lim); "
          f"sem_grad (difusão pura); T={T} η={eta} {passos} passos.")
    print(f"    σ_z={sig:.5f}  lim/σ_z={lim/sig:.1f}  banda={n_sigma*sig:.4f} "
          f"D={int(np.ceil(n_sigma**2))}  profundidade exige z_min≤{-c_p*lim:.4f} "
          f"(={c_p*lim/sig:.0f}σ_z abaixo do init ⟹ difusão não alcança).")
    p0 = dict(w=np.zeros((m, 1)), b=np.full(m, lim), v=np.full(m, 1.0 / np.sqrt(m)))
    _bind(m, eta, T, n_sigma, c_p, para_por_v3=False)
    _, reg, _, _ = treina(m=m, ativ="softplus", eta=eta, T=T, passos=passos,
                          seed=4242, params0=p0, dados=(X, Y), sem_grad=True)
    harness_mlp.Regua = ReguaV2
    n2 = len(reg.v2.episodios)            # v2 completos (down→up cycles)
    c2 = len(reg.v2.censurados(passos))
    n3 = len(reg.v3.episodios)            # v3 completos (dwell+histerese)
    n3p = sum(1 for e in reg.v3.episodios if e["profundo"])
    c3 = len(reg.v3.censurados(passos))
    print(f"    v2: {n2} episódios completos (+{c2} censurados)  — o CHATTER.")
    print(f"    v3: {n3} completos (+{c3} censurados); profundos={n3p}.")
    rej = (1 - n3 / n2) if n2 else None
    rej_p = (1 - n3p / n2) if n2 else None
    print(f"    → frac_chatter_rejeitada = 1 − N_v3/N_v2 = "
          f"{rej:.3f} (completos)  |  {rej_p:.3f} (exigindo profundidade)")
    ver = "REJEITA CHATTER" if (rej is not None and rej >= 0.8) else "NÃO REJEITA"
    print(f"    → {ver}  (limiar de aceitação: ≥0.80)")
    return dict(v2=n2, v3=n3, v3_prof=n3p, rej=rej, rej_prof=rej_p, veredito=ver,
                lim=lim, sig=sig)


# ============================================================================
# (3) FUMAÇA relu-vs-softplus (MLP)
# ============================================================================
def relu_vs_softplus(n_sigma, c_p, m=64, T=1e-3, eta=3e-4, passos=20_000,
                     seeds=(42, 43, 44), t0_probe=False):
    print(f"\n{'='*74}\n(3) FUMAÇA relu-vs-softplus — MLP m={m}, T={T}, η={eta}, "
          f"{passos} passos, seeds {seeds}\n{'='*74}")
    print("    critério que REPROVOU a v2: relu≈softplus em T>0 (chatter). v3 deve")
    print("    calar a relu (grad=0 ⟹ só difusão ⟹ barrada) e reter softplus físico.")
    D = int(np.ceil(n_sigma ** 2))
    print(f"    n_σ={n_sigma} D={D} c_p={c_p}")

    def roda(ativ, T_, kw):
        agg = dict(v2=0, v3=0, v3p=0, cens2=0, cens3=0, dur2=[], dur3=[])
        for sd in seeds:
            _bind(m, eta, T_, n_sigma, c_p, para_por_v3=False)
            t0 = time.perf_counter()
            _, reg, _, _ = treina(m=m, ativ=ativ, eta=eta, T=T_, passos=passos,
                                  seed=sd, **kw)
            harness_mlp.Regua = ReguaV2
            c2 = reg.v2.episodios
            c3 = reg.v3.episodios
            agg["v2"] += len(c2); agg["v3"] += len(c3)
            agg["v3p"] += sum(1 for e in c3 if e["profundo"])
            agg["cens2"] += len(reg.v2.censurados(passos))
            agg["cens3"] += len(reg.v3.censurados(passos))
            agg["dur2"] += [e["t_res"] - e["t_morte"] for e in c2]
            agg["dur3"] += [e["t_res"] - e["t_morte"] for e in c3]
            print(f"      {ativ:8s} T={T_:<5g} s{sd}: v2={len(c2):4d} v3={len(c3):3d} "
                  f"v3p={sum(1 for e in c3 if e['profundo']):3d} "
                  f"({time.perf_counter()-t0:.1f}s)", flush=True)
        return agg

    def linha(nome, a):
        print(f"    {nome:16s}: v2={a['v2']:5d} (med.dur {_med(a['dur2'])}) "
              f"| v3={a['v3']:4d} (med.dur {_med(a['dur3'])}) prof={a['v3p']:4d} "
              f"| cens v2={a['cens2']} v3={a['cens3']}")

    print("\n    -- T>0 (o teste; difusão presente) --")
    sp = roda("softplus", T, dict())
    rl = roda("relu", T, dict())
    print()
    linha("softplus", sp); linha("relu", rl)

    def razao(a, b, ch):
        return (a[ch] / b[ch]) if b[ch] else float("inf")
    print(f"\n    v2  razão relu/softplus = {razao(rl, sp, 'v2'):.2f}   "
          f"(≈1 ⟹ chatter iguala os dois: o diagnóstico que matou o 4a)")
    print(f"    v3  razão relu/softplus = {razao(rl, sp, 'v3'):.2f}")
    print(f"    v3-prof razão relu/softplus = {razao(rl, sp, 'v3p'):.2f}")

    res = dict(sp=sp, rl=rl)
    if t0_probe:
        print("\n    -- T=0 (difusão NULA; separação-limpa de controle) --")
        print("       relu grad=0 no morto ⟹ 0 ressurreições; softplus revive por")
        print("       gradiente (grad=sigmoid(βz)>0). v3 deve reter softplus, calar relu.")
        sp0 = roda("softplus", 0.0, dict())
        rl0 = roda("relu", 0.0, dict())
        print()
        linha("softplus T=0", sp0); linha("relu T=0", rl0)
        res["sp0"] = sp0; res["rl0"] = rl0

    # veredito: v3 separa relu de softplus?  exige relu ~mudo E softplus retém.
    relu_mudo = rl["v3"] <= max(2, 0.25 * sp["v3"])
    sp_retem = sp["v3"] >= 3
    if relu_mudo and sp_retem:
        ver = "SEPARA (v3 aprovada neste critério)"
    elif relu_mudo and not sp_retem:
        ver = ("INCONCLUSIVO em T>0: v3 CALA a relu (debounce funciona) mas "
               "softplus TAMBÉM colapsa ⟹ sem sinal físico em T=1e-3/2e4 passos "
               "(física vive em T→0, cf. diagnóstico). Ver probe T=0.")
    else:
        ver = ("REPROVADA: v3 NÃO cala a relu em T>0 (chatter passa o debounce). "
               "mesma doença da v2.")
    print(f"\n    → {ver}")
    res["veredito"] = ver
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--chatter", action="store_true")
    ap.add_argument("--relu", action="store_true")
    ap.add_argument("--t0", action="store_true", help="inclui probe T=0 no teste 3")
    ap.add_argument("--nsigma", type=float, default=10.0)
    ap.add_argument("--cp", type=float, default=1.0)
    a = ap.parse_args()
    todos = not (a.toy or a.chatter or a.relu)
    print(f"CALIBRAÇÃO RÉGUA v3 — n_σ={a.nsigma} c_p={a.cp}  [EXPLORATÓRIO]")
    if a.toy or todos:
        toy_preservado(a.nsigma, a.cp)
    if a.chatter or todos:
        chatter_rejeitado(a.nsigma, a.cp)
    if a.relu or todos:
        relu_vs_softplus(a.nsigma, a.cp, t0_probe=a.t0)
