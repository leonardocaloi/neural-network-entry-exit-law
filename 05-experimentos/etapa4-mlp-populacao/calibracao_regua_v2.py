"""Etapa 4 — RE-CALIBRAÇÃO da régua EXATA do harness (item 2 do 6º parecer).

A calibração v1 (calibracao_regua.py) validou OUTRO instrumento: numpy puro,
v≡1, limiar implementado à parte. Esta valida o instrumento SHIPPADO — a classe
`Regua` de harness_mlp.py, com v TREINÁVEL, Δsnapshot=1, limiar calibrado
2√(nT)/‖x_j*‖ e R̂ reconstruído com a act da rede — rodando o TOY DENTRO do
harness (m=2, init do toy, dados do toy). [Regra da casa: calibra-se o
instrumento que vai medir, não um primo dele.]

(a) quase-det (T=1e-8, softplus β=300): erro relativo da predição-área cai
    quando βη cai (fronteira de discretização do 1º ciclo de calibração);
(b) sob ruído (T=1e-3, 40 seeds de ruído, MESMO init): pareado área-vs-sinal
    no MESMO episódio — área vence ≥60%. Reporta erros medianos absolutos e
    RELATIVOS à duração do episódio (⟹ δ_A do preditor);
(c) unit-tests mecânicos dos flags novos: v_cruzou, morte-por-saída φ_v,
    trocas de líder, switchings alheios, filtro `episodio_limpo`.
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from harness_mlp import Regua, episodio_limpo, treina

TOY_X = torch.tensor([[0.5], [1.5]])
TOY_Y = torch.tensor([[0.5], [1.0]])
P0 = dict(w=[[-0.5], [1.144]], b=[0.27, 0.261], v=[1.0, 1.0])


def roda_toy(T, seed, eta, passos=500_000):
    torch.set_num_threads(1)
    _, reg, _, _ = treina(m=2, ativ="softplus", beta=300.0, eta=eta, T=T,
                          passos=passos, seed=seed, params0=P0,
                          dados=(TOY_X, TOY_Y), para_cedo=True)
    eps = [ep for ep in reg.episodios if ep["k"] == 0]
    ep = eps[0] if eps else None
    limpo = (ep is not None and
             episodio_limpo(ep, reg.switches, q=0, phi_v=0.05))
    return ep, limpo, len(reg.switches)


def _job(args):
    return roda_toy(*args)


if __name__ == "__main__":
    # ---------- (c) unit-tests mecânicos (primeiro: são instantâneos) ----------
    x1 = np.array([[1.0]])
    zM = np.array([[-1.0]]); zV = np.array([[1.0]]); a0 = np.array([[0.0]])
    # c1: v cruza sinal durante o episódio ⟹ flag + filtro reprova
    r = Regua(1, 1.0, 1e-3)
    r.observa(0, zM, a0, np.array([0.5]), np.array([+0.5]), x1)
    r.observa(1, zM, a0, np.array([0.5]), np.array([-0.5]), x1)
    r.observa(2, zV, zV, np.array([0.5]), np.array([-0.5]), x1)
    ep = r.episodios[0]
    assert ep["v_cruzou"] and not episodio_limpo(ep, r.switches, 5, 0.05)
    # c2: morte-por-saída |v|<φ_v ⟹ filtro reprova; v estável ⟹ aprova
    r = Regua(1, 1.0, 1e-3)
    r.observa(0, zM, a0, np.array([0.5]), np.array([0.01]), x1)
    r.observa(1, zV, zV, np.array([0.5]), np.array([0.01]), x1)
    assert not episodio_limpo(r.episodios[0], r.switches, 5, 0.05)
    r = Regua(1, 1.0, 1e-3)
    r.observa(0, zM, a0, np.array([0.5]), np.array([0.5]), x1)
    r.observa(1, zV, zV, np.array([0.5]), np.array([0.5]), x1)
    assert episodio_limpo(r.episodios[0], r.switches, 0, 0.05)
    # c3: switching alheio dentro do episódio ⟹ q=0 reprova, q=1 aprova
    x2 = np.array([[1.0], [0.5]])
    r = Regua(2, 1.0, 1e-3)
    zz = lambda z0, z1: np.array([[z0, z1], [z0 - 0.1, z1 - 0.1]])
    aa = np.zeros((2, 2))
    r.observa(0, zz(-1., 1.), aa, np.zeros(2), np.array([.5, .5]), x2)  # k0 morre
    r.observa(1, zz(-1., -1.), aa, np.zeros(2), np.array([.5, .5]), x2) # k1 morre
    r.observa(2, zz(1., -1.), np.abs(zz(1., -1.)), np.zeros(2),
              np.array([.5, .5]), x2)                                   # k0 volta
    ep0 = [e for e in r.episodios if e["k"] == 0][0]
    assert not episodio_limpo(ep0, r.switches, 0, 0.05)
    assert episodio_limpo(ep0, r.switches, 1, 0.05)
    # c4: troca de líder contada
    r = Regua(1, 1.0, 1e-3)
    r.observa(0, np.array([[-1.0], [-2.0]]), np.zeros((2, 1)), np.zeros(2),
              np.array([0.5]), x2)
    r.observa(1, np.array([[-2.0], [-1.0]]), np.zeros((2, 1)), np.zeros(2),
              np.array([0.5]), x2)
    assert int(r.j_trocas[0]) == 1
    print("[OK] (c) unit-tests: v_cruzou, φ_v, switching alheio, troca de líder")

    # ---------- (a) quase-det: identificação do erro do instrumento ----------
    # DESCOBERTA da 1ª rodada desta calibração: no toy-em-harness o episódio é
    # ~20× mais curto que no toy v1 (η único ⟹ ambiente rápido), e o erro da
    # predição-área NÃO vai a zero com βη→0 — converge para o piso ≈ atraso de
    # liberação O(1/β) do PRÓPRIO teorema (t_release = t_exit − 1/(c·R·β)).
    # O 1.12% observado em βη=0.30 era cancelamento acidental (discretização
    # empurra no sentido oposto). Testes corretos:
    # (a1) o tempo CONTÍNUO τ_res = η·t_res converge quando η cai;
    # (a2) com βη fixo, o erro cai ~1/β (o piso é a correção do teorema,
    #      não bug do instrumento).
    def caso(beta_, eta_):
        _, reg, _, _ = treina(m=2, ativ="softplus", beta=beta_, eta=eta_,
                              T=1e-8, passos=1_000_000, seed=1, params0=P0,
                              dados=(TOY_X, TOY_Y), para_cedo=True)
        e = [p for p in reg.episodios if p["k"] == 0][0]
        assert e["t_area"] >= 0, e
        return e, abs(e["t_area"] - e["t_res"]) / e["t_res"]

    e1, r1 = caso(300.0, 1e-3)        # βη=0.30
    e2, r2 = caso(300.0, 1e-3 / 3)    # βη=0.10
    e3, r3 = caso(1000.0, 1e-4)       # βη=0.10, β maior
    tau1, tau2 = 1e-3 * e1["t_res"], (1e-3 / 3) * e2["t_res"]
    print(f"[a1] τ_res: η=1e-3 → {tau1:.3f}; η=1e-3/3 → {tau2:.3f} "
          f"(Δ={abs(tau1-tau2)/tau2*100:.1f}%)")
    assert abs(tau1 - tau2) / tau2 < 0.10
    print(f"[a2] erro relativo da área com βη=0.10: β=300 → {r2*100:.2f}%, "
          f"β=1000 → {r3*100:.2f}% (razão {r3/r2:.2f}; 1/β puro prevê ~0.30)")
    assert r3 < 0.6 * r2 and r3 < 0.025, (r2, r3)
    print("[OK] (a) erro do instrumento CONSISTENTE com o atraso O(1/β) do "
          "teorema (cai com β; piso residual ~0.8pp não explicado por 1/β — "
          "2 pontos de β não identificam o expoente); dinâmica converge em "
          "tempo contínuo. Rótulo por exigência do 7º parecer.")

    # ---------- (b) sob ruído: pareado área-vs-sinal, 40 seeds ----------
    with ProcessPoolExecutor(max_workers=6) as ex:
        outs = list(ex.map(_job, [(1e-3, sd, 1e-3) for sd in range(3000, 3040)]))
    ganha, tot, n_cens, n_sujo = 0, 0, 0, 0
    errA, errS, relA, relS = [], [], [], []
    for ep, limpo, _ in outs:
        if ep is None:
            n_cens += 1
            continue
        if not limpo:
            n_sujo += 1
        if ep["t_area"] < 0 or ep["t_sinal"] < 0:
            continue
        tot += 1
        dur = max(ep["t_res"] - ep["t_morte"], 1)
        eA = abs(ep["t_res"] - ep["t_area"]); eS = abs(ep["t_res"] - ep["t_sinal"])
        errA.append(eA); errS.append(eS)
        relA.append(eA / dur); relS.append(eS / dur)
        ganha += int(eA < eS)
    print(f"[T=1e-3, 40 seeds] episódios completos={tot}, censurados={n_cens}, "
          f"não-limpos={n_sujo}")
    print(f"  erro mediano: ÁREA={np.median(errA):.0f} vs SINAL={np.median(errS):.0f} "
          f"passos; área vence {ganha}/{tot}")
    print(f"  erro RELATIVO mediano (÷ duração): ÁREA={np.median(relA):.3f} "
          f"vs SINAL={np.median(relS):.3f}  ⟹ insumo p/ δ_A do preditor")
    assert tot >= 12 and ganha / tot >= 0.6, "área não supera sinal no toy-em-harness"
    print("[OK] (b) zero-de-área vence sinal com o instrumento SHIPPADO")
    print("\nRÉGUA v2 RECALIBRADA — o instrumento do harness (v treinável, "
          "Δsnapshot=1, limiar calibrado) está aprovado para o MLP.")
