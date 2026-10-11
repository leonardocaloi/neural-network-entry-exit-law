"""Diagnóstico da CÉLULA-SONDA do desenho rápido (etapa-4a.3, pivô do coordenador).

Célula: m=4, softplus β=10, seed 42, n=16 dados, η=1e-3, T=0, H=2e5.
Objetivo: com a grade rápida (η 3.3× maior, n 4× menor ⟹ deriva restauradora
~100× maior que a v1), a ressurreição DINÂMICA prevista deve cair para ≲3e4
passos, DENTRO do horizonte. Se a sonda mostrar ressurreição dinâmica com o
portão de área aberto, libera a grade de 24. Se NÃO, o cálculo de deriva está
errado em algo — este script imprime, por morte dinâmica: z_park, σ(β·z_park),
R no líder, deriva prevista/passo e tempo de retorno previsto vs observado.

Separação obrigatória (regra do coordenador):
  - NASCIDO-MORTO (t_morte=0): estaciona na profundidade do INIT (canal fechado,
    nunca volta; profundidade idêntica entre braços pois é a init). NÃO é T2'.
  - MORTE DINÂMICA (t_morte>0): neurônio VIVO que morre e estaciona RASO
    (~ camada log(β)/β). ESTE é o objeto de T2'.
"""
import argparse
import sys

import numpy as np

AQUI = __file__.rsplit("/", 1)[0]
sys.path.insert(0, AQUI)
sys.path.insert(0, AQUI.rsplit("/", 2)[0])   # etapa4 dir (harness/regua)

import torch  # noqa: E402

torch.set_num_threads(1)
import train_pressure as tp  # noqa: E402


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def diagnostica(m=4, beta=10.0, seed=42, n_dados=16, eta=1e-3, passos=200_000,
                rec=200):
    net, reg2, reg3, metr = tp.treina_pressao(
        m=m, ativ="softplus", beta=beta, eta=eta, T=0.0, passos=passos,
        seed=seed, n_dados=n_dados, n_sigma=10.0, rec_neuronio=rec,
        rec_cada=100)
    lh = metr["loss_hist"]
    n10 = max(1, len(lh) // 10)
    plato = float(np.mean(lh[-n10:]))
    S = metr["serie"]
    st, szmax, sR, sv, sn2 = S["t"], S["zmax"], S["R"], S["v"], S["n2"]

    print(f"=== SONDA m={m} softplus β={beta:g} seed={seed} n={n_dados} "
          f"η={eta:g} T=0 H={passos} ===")
    print(f"loss: inicial={lh[0]:.4f} final={lh[-1]:.4f} platô(últ10%)={plato:.4f} "
          f"{'(>=0.2 OK)' if plato >= 0.2 else '(<0.2 — platô baixo, ver nota)'}")
    print(f"ground-truth: n_mortos_real={metr['n_mortos_real']} "
          f"n_revive_real={metr['n_revive_real']} "
          f"(neurônios que voltaram: {metr['n_neuronios_revive_real']})")

    eps = [("completo", e) for e in reg3.episodios] + \
          [("censurado", e) for e in reg3.censurados(passos)]
    born = [e for tp_, e in eps if e["t_morte"] == 0]
    dyn = [(tp_, e) for tp_, e in eps if e["t_morte"] > 0]
    print(f"\nEpisódios v3: {len(eps)} total | nascido-morto(t_morte=0)={len(born)} "
          f"| morte-DINÂMICA(t_morte>0)={len(dyn)}")

    print("\n-- NASCIDOS-MORTOS (canal fechado; profundidade = init) --")
    for e in born:
        print(f"  k={e['k']} z_min={e['z_min']:.4f} profundo={e['profundo']} "
              f"t_res={e['t_res']} (>=0 ⟹ voltou)")

    print("\n-- MORTES DINÂMICAS (objeto de T2'): previsão de deriva vs observado --")
    if not dyn:
        print("  NENHUMA morte dinâmica nesta sonda.")
    hdr = ("  k | t_morte | t_res(obs) | dur_obs | z_park | σ(βz_park) | R_lider "
           "| v | ‖x*‖² | deriva/passo | t_retorno_prev")
    print(hdr)
    achou_revive_din = False
    for tp_, e in dyn:
        k = e["k"]
        t0, t1 = e["t_morte"], (e["t_res"] if e["t_res"] >= 0 else passos)
        win = (st > t0) & (st <= t1)
        if win.sum() == 0:
            win = (st >= t0)
        zk = szmax[win, k]
        # z_park = mediana da margem-líder enquanto MORTO (parada rasa)
        z_park = float(np.median(zk))
        R_park = float(np.median(sR[win, k]))
        v_park = float(np.median(sv[win, k]))
        n2_park = float(np.median(sn2[win, k]))
        sig = sigmoid(beta * z_park)
        deriva = eta * v_park * R_park * sig * n2_park / n_dados   # Δz previsto/passo
        if deriva > 0:
            t_ret = abs(z_park) / deriva
            t_ret_s = f"{t_ret:.3e}"
        else:
            t_ret_s = "∞ (deriva<=0: pressão empurra p/ baixo)"
        revive = tp_ == "completo"
        dur_obs = (e["t_res"] - e["t_morte"]) if revive else None
        if revive and e["t_morte"] > 0:
            achou_revive_din = True
        print(f"  {k} | {t0} | {e['t_res'] if revive else '—(cens)'} | "
              f"{dur_obs if dur_obs is not None else '—'} | {z_park:.4f} | "
              f"{sig:.3e} | {R_park:+.4f} | {v_park:+.4f} | {n2_park:.2f} | "
              f"{deriva:+.3e} | {t_ret_s}")

    print("\n-- portão de área (t_area>=0 ⟹ Â_k cruzou 0; 'gate aberto') --")
    for tp_, e in dyn:
        print(f"  k={e['k']} t_morte={e['t_morte']} t_area={e['t_area']} "
              f"t_sinal={e['t_sinal']} t_res={e['t_res'] if tp_=='completo' else '—'}")

    print("\n=== VEREDITO DA SONDA ===")
    if achou_revive_din:
        print("OK: HÁ ressurreição DINÂMICA (t_morte>0 e voltou). Grade de 24 liberada.")
    else:
        print("SEM ressurreição dinâmica nesta sonda. Comparar t_retorno_prev com H "
              f"={passos}: se prev << H mas não voltou ⟹ a conta de deriva está "
              "errada (sinal de R? leader? gate?); se prev >> H ou deriva<=0 ⟹ "
              "célula ainda lenta/pressão adversa — ajustar β/η/n antes da grade.")
    return dict(plato=plato, n_dyn=len(dyn), n_born=len(born),
                revive_din=achou_revive_din,
                n_revive_real=metr["n_revive_real"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--eta", type=float, default=1e-3)
    ap.add_argument("--passos", type=int, default=200_000)
    a = ap.parse_args()
    diagnostica(m=a.m, beta=a.beta, seed=a.seed, n_dados=a.n, eta=a.eta,
                passos=a.passos)
