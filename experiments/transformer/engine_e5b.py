"""Motor-B da etapa-5 (PILOTO-DE-TAXA): estende engine_e5.py sem tocá-lo.

Acrescenta o BRACO DA CLASSE-DO-TEOREMA (softplus-beta, g'=sigma(beta z)>0 sempre)
alem de GELU e ReLU, e reaparelha o contador-mestre GT para MEDIR TAXA:

  1) z_dead = z_alive = 0  para TODOS os bracos (morte-GT = unidade INATIVA: zmax<0;
     volta-GT = zmax>0). Unifica a regua com o etapa-4 (zmax<0 -> >0) e torna os bracos
     comparaveis. [Difere do scoping, que usava z_dead=-3 p/ GELU; aqui a morte e a
     INATIVIDADE e a PROFUNDIDADE e um atributo do episodio, nao o gatilho.]
  2) CLASSIFICACAO DE PROFUNDIDADE por z_park (min zmax no episodio), com a BARREIRA
     do GELU z*=-0.7518 (verificada: g'(z)<0 p/ z<z*): morte-profunda (z_park<z*) vs
     morte-rasa (z*<z_park<0), SEPARADAS. Para softplus/relu z* nao tem sentido fisico
     (canal monotono / degrau); reporta-se por PROFUNDIDADE DE INJECAO.
  3) INJECAO em DUAS PROFUNDIDADES (kill_depths): uma rasa (acima da barreira) e uma
     profunda (abaixo), p/ POPULAR ambas as classes e testar P-ii (GELU: profunda
     TERMINAL por barreira, rasa REVIVIVEL) num mesmo run. Alavanca EXPLORATORIA
     declarada (condicao inicial controlada, nao probe pos-hoc).
  4) drive faithful A_drive com a inclinacao CORRETA por ativacao:
        gelu:     g'(z) = Phi(z)+z phi(z)   (troca de sinal em z*)
        softplus: g'(z) = sigma(beta z)      (>0 sempre — classe do teorema)
        relu:     g'(z) = 1[z>0]             (0 no morto — sem self-canal)
  5) ATRIBUICAO DE CANAL: own_move=||dW1[k],db1[k]|| no episodio. attn-congelada
     (freeze_attn) => entrada da FFN FIXA => TODA volta e drive-proprio; attn-treina
     => own+montante (relu morto tem own_move==0 exato => volta = PURO montante).
  6) distingue INJETADA (born-dead por injecao) de NATURAL (born-dead de init OU
     dinamica t>0) => lambda_morte-dinamica sai das NATURAIS classe=='dyn'.

Toda a instrumentacao e ADITIVA (read-only sobre a dinamica GD pura T=0). Reusa
TinyTransformer, dados_seq, rich_f_torch, gelu_prime_np de motor_e5.
"""
import argparse
import math
import numpy as np
import torch

import engine_e5 as M
from engine_e5 import gelu_prime_np, rich_f_torch, dados_seq, RAIZ2

torch.use_deterministic_algorithms(True)

Z_STAR = -0.7518   # barreira do GELU: g'(z)<0 para z<z* (verificada na nota 2026-07-25)


def sigmoid_np(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def gprime_np(z, ativ, beta=1.0):
    """Inclinacao local g'(z) por ativacao (para o drive faithful A_drive)."""
    z = np.asarray(z, dtype=np.float64)
    if ativ == "gelu":
        return gelu_prime_np(z)
    if ativ == "softplus":
        return sigmoid_np(beta * z)
    if ativ == "relu":
        return (z > 0.0).astype(np.float64)
    raise ValueError(ativ)


class TinyTransformerB(M.TinyTransformer):
    """Igual ao TinyTransformer mas com softplus-beta na FFN (classe do teorema)."""

    def __init__(self, *a, beta=1.0, **kw):
        super().__init__(*a, **kw)
        self.beta = float(beta)

    def act(self, z):
        if self.ativ == "softplus":
            return torch.nn.functional.softplus(z, beta=self.beta)
        return super().act(z)


def treina_e5b(m_ffn=4, ativ="gelu", beta=3.0, eta=1e-3, passos=60_000, seed=0,
               device="cpu", N=24, S=3, d=16, H=2, z_dead=0.0, z_alive=0.0,
               rec_cada=300, freeze_attn=False, s_attn=None, s_ffn=None, s_head=None,
               kill_depths=(0.4, 1.2)):
    """GD puro T=0 determinista + contador-mestre GT-de-taxa + GT-AREA (A_drive).

    z_dead=z_alive=0: morte=unidade inativa (zmax<0), volta=zmax>0 (regua do etapa-4).
    kill_depths: injeta len(kill_depths) unidades NASCIDAS-MORTAS nas de menor zmax
      inicial, cada uma a zmax=-profundidade (rasa acima da barreira, profunda abaixo).
      kill_depths=() => sem injecao (mede so o regime natural).
    freeze_attn: congela embedding+attn (h=e+attn(e) ESTACIONARIO) => self-canal isolado.
    """
    x, y = dados_seq(N=N, S=S, seed=seed)
    x, y = x.to(device), y.to(device)
    net = TinyTransformerB(d=d, H=H, m_ffn=m_ffn, S=S, ativ=ativ, seed=seed, beta=beta,
                           s_attn=s_attn, s_ffn=s_ffn, s_head=s_head).to(device)
    m = m_ffn
    idx = np.arange(m)

    # ---- injecao de nascidas-mortas em multiplas profundidades ----
    injetada = np.zeros(m, bool)
    injdepth = np.full(m, np.nan)
    kd = list(kill_depths)
    if len(kd) > 0:
        with torch.no_grad():
            z0 = (net.h_pre_ffn(x) @ net.W1.T + net.b1).detach().cpu().numpy().reshape(N * S, m)
            zmax0 = z0.max(0)
            alvo = np.argsort(zmax0)[:len(kd)]
            for i, k in enumerate(alvo):
                k = int(k)
                net.b1[k] += float(-kd[i] - zmax0[k])
                injetada[k] = True
                injdepth[k] = -kd[i]

    frozen_names = {"w_val", "P", "WQ", "WK", "WV", "WO"}
    params_treina = [p for n_, p in net.named_parameters()
                     if (not freeze_attn) or n_ not in frozen_names]

    loss_hist = []
    esteve_morto = np.zeros(m, bool)
    morto_prev = np.zeros(m, bool)
    nasce_morto = None
    death_events = []            # (t,k) vivo->morto com t>0 = DINAMICA
    revive_events = []           # (t,k) morto->vivo
    n_revive = 0
    revive_neuronios = set()

    A_mlp = np.zeros(m); A_drive = np.zeros(m)
    t_area_mlp = np.full(m, -1); t_area_drive = np.full(m, -1)
    t_morte_ep = np.full(m, -1)
    zpark_ep = np.full(m, np.inf); zhi_ep = np.full(m, -np.inf)
    classe_ep = [None] * m
    W1_morte = np.zeros((m, d)); b1_morte = np.zeros(m)
    episodios = []

    def _snap_own():
        return (net.W1.detach().cpu().numpy(), net.b1.detach().cpu().numpy())

    def _abre(k, t_morte, classe, zmax_k, W1_np, b1_np):
        A_mlp[k] = 0.0; A_drive[k] = 0.0
        t_area_mlp[k] = -1; t_area_drive[k] = -1
        t_morte_ep[k] = t_morte; zpark_ep[k] = zmax_k; zhi_ep[k] = zmax_k
        classe_ep[k] = classe
        W1_morte[k] = W1_np[k]; b1_morte[k] = b1_np[k]

    def _fecha(k, t_res, revived, W1_np, b1_np):
        zp = float(zpark_ep[k])
        own = float(np.sqrt(((W1_np[k] - W1_morte[k]) ** 2).sum() +
                            (b1_np[k] - b1_morte[k]) ** 2))
        deep = bool(zp < Z_STAR)
        # canal: frozen => proprio (entrada fixa); train => proprio se moveu peso, senao montante
        if revived:
            if freeze_attn:
                canal = "proprio"
            else:
                canal = "montante" if own < 1e-9 else "proprio+montante"
        else:
            canal = None
        episodios.append(dict(
            k=int(k), classe=classe_ep[k], injetada=bool(injetada[k]),
            injdepth=(float(injdepth[k]) if injetada[k] else None),
            deep=deep, t_morte=int(t_morte_ep[k]),
            t_area_mlp=int(t_area_mlp[k]), t_area_drive=int(t_area_drive[k]),
            t_res=(int(t_res) if t_res is not None else None), revived=bool(revived),
            z_park=zp, zmax_hi=float(zhi_ep[k]),
            gprime_park=float(gprime_np(zp, ativ, beta)),
            A_mlp_fim=float(A_mlp[k]), A_drive_fim=float(A_drive[k]),
            own_move=own, canal=canal))
        t_morte_ep[k] = -1

    loss = None
    for t in range(passos):
        yhat, z, a, h = net.forward_parts(x)
        res = yhat - y
        loss = 0.5 * (res ** 2).mean()
        z_det = z.detach().reshape(N * S, m)
        zmv, jsv = z_det.max(0)
        zmax = zmv.cpu().numpy(); jstar = jsv.cpu().numpy()
        morto = zmax < z_dead
        esteve_morto |= morto
        # aberturas
        if t == 0:
            nasce_morto = morto.copy()
            if morto.any():
                W1_np, b1_np = _snap_own()
                for k in np.where(morto)[0]:
                    _abre(k, 0, "born", zmax[k], W1_np, b1_np)
            morto_prev = morto.copy()
        else:
            morre = (~morto_prev) & morto
            if morre.any():
                W1_np, b1_np = _snap_own()
                for k in np.where(morre)[0]:
                    death_events.append((int(t), int(k)))
                    _abre(k, int(t), "dyn", zmax[k], W1_np, b1_np)
        # acumulacao GT-AREA (so p/ episodios abertos)
        aberto = t_morte_ep >= 0
        if aberto.any():
            a_np = a.detach().reshape(N * S, m).cpu().numpy()
            res_np = res.detach().reshape(N * S).cpu().numpy()
            c_k = (net.w_out @ net.W2).detach().cpu().numpy()
            res_lider = res_np[jstar]; a_lider = a_np[jstar, idx]
            gp_lider = gprime_np(zmax, ativ, beta)
            R_mlp = c_k * (-res_lider + c_k * a_lider)
            D_k = c_k * (-res_lider) * gp_lider
            A_mlp[aberto] += R_mlp[aberto] * eta
            A_drive[aberto] += D_k[aberto] * eta
            cm = aberto & (t_area_mlp < 0) & (A_mlp > 0.0)
            cd = aberto & (t_area_drive < 0) & (A_drive > 0.0)
            t_area_mlp[cm] = t; t_area_drive[cd] = t
            zpark_ep[aberto] = np.minimum(zpark_ep[aberto], zmax[aberto])
            zhi_ep[aberto] = np.maximum(zhi_ep[aberto], zmax[aberto])
        # voltas
        if t > 0:
            revive = morto_prev & (zmax > z_alive)
            if revive.any():
                W1_np, b1_np = _snap_own()
                for k in np.where(revive)[0]:
                    n_revive += 1
                    revive_neuronios.add(int(k))
                    revive_events.append((int(t), int(k)))
                    _fecha(k, int(t), True, W1_np, b1_np)
            morto_prev = morto
        if t % rec_cada == 0:
            loss_hist.append(float(loss.detach()))
        net.zero_grad(); loss.backward()
        with torch.no_grad():
            for p in params_treina:
                if p.grad is not None:
                    p += -eta * p.grad
    loss_hist.append(float(loss.detach()))
    W1_np, b1_np = _snap_own()
    for k in np.where(t_morte_ep >= 0)[0]:
        _fecha(k, None, False, W1_np, b1_np)   # CENSURADO

    metr = dict(
        loss_hist=loss_hist,
        plato=float(np.median(loss_hist[-max(3, len(loss_hist)//10):])),
        loss_fim=float(loss_hist[-1]),
        nasce_morto=[int(k) for k in np.where(nasce_morto)[0]],
        n_nasce_morto=int(nasce_morto.sum()),
        n_born_natural=int((nasce_morto & ~injetada).sum()),
        n_injetadas=int(injetada.sum()),
        death_events=death_events, revive_events=revive_events,
        n_mortos=int(esteve_morto.sum()), n_revive=int(n_revive),
        n_neuronios_revive=len(revive_neuronios),
        episodios=episodios,
        cfg=dict(m_ffn=m_ffn, ativ=ativ, beta=beta, eta=eta, passos=passos, seed=seed,
                 N=N, S=S, d=d, H=H, z_dead=z_dead, z_alive=z_alive,
                 freeze_attn=freeze_attn, kill_depths=kd))
    return metr


# ============================ TESTES SINTETICOS (braco novo) ============================
def _testes():
    print("== TESTES motor_e5b (braco softplus-beta + contador-de-taxa) ==")
    torch.set_num_threads(1)

    # (0) regressao: os 8 testes de motor_e5 continuam passando (nao quebrei nada)
    M._testes()
    print("[OK] regressao: motor_e5._testes() passou (8 testes)")

    # (1) softplus g'(z)=sigma(beta z) > 0 SEMPRE (monotono; contraste ao sign-flip GELU)
    zs = np.linspace(-6, 6, 25)
    for b in (3.0, 10.0):
        gp = gprime_np(zs, "softplus", b)
        assert np.all(gp > 0), (b, gp.min())
    # e o GELU TROCA de sinal (garante o contraste que motiva o braco)
    assert gprime_np(-2.0, "gelu") < 0 < gprime_np(0.5, "gelu")
    print("[OK] softplus g'=sigma(beta z)>0 em [-6,6] (b=3,10); GELU troca sinal (barreira)")

    # (2) act softplus == formula estavel e derivada autograd == sigma(beta z)
    net = TinyTransformerB(d=8, H=2, m_ffn=4, S=3, ativ="softplus", beta=3.0, seed=7).double()
    zt = torch.linspace(-5, 5, 11, dtype=torch.float64, requires_grad=True)
    at = net.act(zt); at.sum().backward()
    ref = 1.0 / (1.0 + torch.exp(-3.0 * zt.detach()))
    assert torch.allclose(zt.grad, ref, atol=1e-10), (zt.grad - ref).abs().max()
    manual = torch.nn.functional.softplus(zt.detach(), beta=3.0)
    assert torch.allclose(at.detach(), manual, atol=1e-12)
    print("[OK] softplus act e d/dz = sigma(3 z) (autograd == analitico, float64)")

    # (3) determinismo com softplus
    m1 = treina_e5b(m_ffn=6, ativ="softplus", beta=3.0, passos=300, seed=3, kill_depths=())
    m2 = treina_e5b(m_ffn=6, ativ="softplus", beta=3.0, passos=300, seed=3, kill_depths=())
    assert abs(m1["loss_fim"] - m2["loss_fim"]) < 1e-12
    print(f"[OK] determinismo softplus: loss_fim={m1['loss_fim']:.6f}")

    # (4) softplus MORTO (zmax<0) tem gradiente PROPRIO != 0 (canal revivivel; vs relu==0)
    net = TinyTransformerB(d=12, H=2, m_ffn=8, S=4, ativ="softplus", beta=3.0, seed=9)
    x, y = dados_seq(N=8, S=4, seed=9)
    with torch.no_grad():
        net.b1 -= 6.0                     # empurra p/ zmax<0
    yhat, z, a, h = net.forward_parts(x)
    zmax = z.detach().numpy().reshape(-1, net.m_ffn).max(0)
    mortos = np.where(zmax < 0.0)[0]; assert len(mortos) > 0
    loss = 0.5 * ((yhat - y) ** 2).mean(); net.zero_grad(); loss.backward()
    gW1 = net.W1.grad.detach().numpy()
    assert np.all(np.abs(gW1[mortos]).sum(1) > 1e-12)
    # relu no MESMO estado tem grad proprio == 0 (contraste)
    net2 = TinyTransformerB(d=12, H=2, m_ffn=8, S=4, ativ="relu", seed=9)
    with torch.no_grad():
        net2.b1 -= 6.0
    yhat2, z2, a2, _ = net2.forward_parts(x)
    zmax2 = z2.detach().numpy().reshape(-1, net2.m_ffn).max(0)
    mortos2 = np.where(zmax2 < 0.0)[0]
    loss2 = 0.5 * ((yhat2 - y) ** 2).mean(); net2.zero_grad(); loss2.backward()
    assert np.allclose(net2.W1.grad.detach().numpy()[mortos2], 0.0)
    print(f"[OK] softplus morto: grad proprio !=0 ({len(mortos)} un.); relu morto: ==0 "
          f"({len(mortos2)} un.) => softplus e canal revivivel, relu nao")

    # (5) drive faithful softplus concorda com o auto-movimento da margem (attn congelada)
    net = TinyTransformerB(d=12, H=2, m_ffn=8, S=4, ativ="softplus", beta=3.0, seed=5)
    x, y = dados_seq(N=8, S=4, seed=5)
    with torch.no_grad():
        net.b1 -= 3.0
    concordam = tot = 0
    for _ in range(30):
        yhat, z, a, h = net.forward_parts(x)
        res = yhat - y; loss = 0.5 * (res ** 2).mean()
        N_, S_ = x.shape; m_ = net.m_ffn
        z_np = z.detach().numpy().reshape(N_ * S_, m_)
        res_np = res.detach().numpy().reshape(N_ * S_); c_k = net.c_eff()
        zmax = z_np.max(0); jstar = z_np.argmax(0)
        D = c_k * (-res_np[jstar]) * gprime_np(zmax, "softplus", 3.0)
        mortos = zmax < 0.0
        net.zero_grad(); loss.backward()
        with torch.no_grad():
            for nm, p in net.named_parameters():
                if nm in {"W1", "b1", "W2", "b2"} and p.grad is not None:
                    p += -1e-2 * p.grad
        z2 = (net.h_pre_ffn(x) @ net.W1.T + net.b1).detach().numpy().reshape(N_ * S_, m_)
        dzmax = z2.max(0) - zmax
        for k in np.where(mortos)[0]:
            if abs(D[k]) > 1e-9 and abs(dzmax[k]) > 1e-9:
                tot += 1; concordam += int(np.sign(D[k]) == np.sign(dzmax[k]))
    frac = concordam / max(tot, 1)
    assert frac >= 0.9, (concordam, tot, frac)
    print(f"[OK] drive softplus: sign(D)==sign(dzmax) em {concordam}/{tot} ({frac:.2f}) passos-un.")

    # (6) injecao em duas profundidades: 2 unidades nascem-mortas em zmax=-depth exatos,
    #     uma rasa (>z*) e uma profunda (<z*); classificacao deep bate.
    mt = treina_e5b(m_ffn=4, ativ="gelu", passos=200, seed=11, kill_depths=(0.4, 1.2))
    assert mt["n_injetadas"] == 2
    inj = [e for e in mt["episodios"] if e["injetada"]]
    depths = sorted(e["injdepth"] for e in inj)
    assert abs(depths[0] - (-1.2)) < 1e-6 and abs(depths[1] - (-0.4)) < 1e-6, depths
    rasa = [e for e in inj if e["injdepth"] == -0.4][0]
    prof = [e for e in inj if e["injdepth"] == -1.2][0]
    assert rasa["deep"] is False and prof["deep"] is True
    print(f"[OK] injecao 2-profundidades: zmax0={depths} exatos; rasa deep={rasa['deep']} "
          f"prof deep={prof['deep']} (barreira z*={Z_STAR})")

    # (7) atribuicao de canal ReLU: attn-CONGELADA + morto => own_move==0 exato (terminal)
    mt = treina_e5b(m_ffn=4, ativ="relu", passos=500, seed=2, kill_depths=(0.4, 1.2),
                    freeze_attn=True)
    for e in mt["episodios"]:
        if e["injetada"]:
            assert e["own_move"] < 1e-12, (e["k"], e["own_move"])
    assert mt["n_revive"] == 0
    print(f"[OK] relu attn-congelada: own_move==0 exato nas injetadas, 0 voltas (teorema MLP)")

    # (8) consistencia contador<->episodios (born+dyn == episodios; revividos == n_revive)
    mt = treina_e5b(m_ffn=6, ativ="softplus", beta=3.0, passos=800, seed=4, kill_depths=(0.5,))
    n_abertos = mt["n_nasce_morto"] + len(mt["death_events"])
    assert len(mt["episodios"]) == n_abertos, (len(mt["episodios"]), n_abertos)
    assert sum(1 for e in mt["episodios"] if e["revived"]) == mt["n_revive"]
    for e in mt["episodios"]:
        if e["t_area_drive"] >= 0:
            assert e["t_morte"] <= e["t_area_drive"]
            if e["t_res"] is not None:
                assert e["t_area_drive"] < e["t_res"]
    print(f"[OK] contador<->episodios: {n_abertos} eps, revive={mt['n_revive']}, trios bem-formados")
    print("== TODOS OS TESTES (motor_e5b) PASSARAM ==")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--testes", action="store_true")
    a = ap.parse_args()
    if a.testes:
        _testes()
    else:
        print("uso: python engine_e5b.py --testes")
