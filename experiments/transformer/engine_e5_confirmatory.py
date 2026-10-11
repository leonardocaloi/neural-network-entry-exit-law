"""Motor CONFIRMATORIO da etapa-5 (transformer FFN). Estende motor_e5b SEM tocar a
DINAMICA: acrescenta ao contador-mestre GT (i) HISTERESE (morte confirmada zmax<-d_h,
volta confirmada zmax>+d_h) para matar o CHATTER do canal nao-estacionario que
contaminou o piloto (rate-pilot.md 0), e (ii) a nocao de VOLTA SUSTENTADA (unidade viva
no fim de H) vs HOVER (voltou mas reabriu). A injecao passa a ser UM nivel por run
(kill_depths=(depth,)): rasa -0.3 (revivivel, >z*) ou profunda -2.0 (barreira, <z*).

PARIDADE OBRIGATORIA: com os MESMOS kwargs de dinamica (mesmos kill_depths, seed, etc.)
o loss_hist e byte-identico ao de motor_e5b.treina_e5b — o instrumento e ADITIVO e
read-only (so muda a CONTAGEM, nunca o passo de GD). Testado em `--testes`.

d_h (meia-banda da histerese) NAO e numero magico: e DERIVADO do piloto pelo
preditor_e5 (quantil alto da amplitude de flutuacao de zmax de unidades VIVAS em runs
frozen). A funcao `zmax_trace_frozen` aqui e a fonte dessa calibracao (mesma dinamica
GD frozen, sem injecao) — o preditor a importa.

Reusa TinyTransformerB, gprime_np, Z_STAR, dados_seq de motor_e5b/motor_e5.
"""
import argparse
import numpy as np
import torch

import engine_e5b as MB
from engine_e5b import TinyTransformerB, gprime_np, Z_STAR
from engine_e5 import dados_seq

torch.use_deterministic_algorithms(True)

FROZEN_NAMES = {"w_val", "P", "WQ", "WK", "WV", "WO"}


def _params_treina(net, freeze_attn):
    return [p for n_, p in net.named_parameters()
            if (not freeze_attn) or n_ not in FROZEN_NAMES]


def zmax_trace_frozen(ativ, beta, seed, passos, m_ffn=4, N=24, S=3, d=16, H=2,
                      eta=1e-3):
    """Roda GD FROZEN (attn/embedding congelados) SEM injecao e devolve o traco
    zmax[t,k] de cada unidade. Fonte de calibracao de d_h (amplitude de flutuacao
    de zmax das unidades VIVAS). Mesma dinamica de treina_e5b(freeze_attn=True,
    kill_depths=()), so que registrando o traco completo. NAO instrumenta area."""
    x, y = dados_seq(N=N, S=S, seed=seed)
    net = TinyTransformerB(d=d, H=H, m_ffn=m_ffn, S=S, ativ=ativ, seed=seed, beta=beta)
    params = _params_treina(net, True)
    m = m_ffn
    trace = np.empty((passos, m), dtype=np.float64)
    for t in range(passos):
        yhat, z, a, h = net.forward_parts(x)
        res = yhat - y
        loss = 0.5 * (res ** 2).mean()
        trace[t] = z.detach().reshape(N * S, m).max(0).values.cpu().numpy()
        net.zero_grad(); loss.backward()
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    p += -eta * p.grad
    return trace


def treina_e5_conf(m_ffn=4, ativ="gelu", beta=3.0, eta=1e-3, passos=200_000, seed=0,
                   device="cpu", N=24, S=3, d=16, H=2, delta_h=0.05, rec_cada=300,
                   freeze_attn=False, s_attn=None, s_ffn=None, s_head=None,
                   kill_depth=None, kill_depths=None):
    """GD puro T=0 determinista (IDENTICO a treina_e5b) + contador-mestre GT com
    HISTERESE d_h + colapso por unidade (voltas SUSTENTADAS).

    Injecao: passe kill_depth=0.3 (rasa) ou 2.0 (profunda) -> injeta 1 unidade
    nascida-morta em zmax=-kill_depth (a de menor zmax inicial). kill_depths=(...) e
    aceito p/ PARIDADE com motor_e5b (mesma assinatura); kill_depth e o atalho 1-nivel.

    Estado por unidade (maquina de histerese):
      viva  --(zmax<-d_h)-->  morta   (morte confirmada; classe born se t=0, dyn se t>0)
      morta --(zmax>+d_h)-->  viva    (VOLTA confirmada; grava t_res)
      banda [-d_h,+d_h]: segura o estado (nem morte nem volta) -> mata o chatter.
    VOLTA SUSTENTADA = unidade injetada que voltou E terminou H VIVA. HOVER = voltou
    mas reabriu (morreu de novo) e terminou MORTA.
    """
    if kill_depths is None:
        kill_depths = () if kill_depth is None else (float(kill_depth),)
    kd = list(kill_depths)

    x, y = dados_seq(N=N, S=S, seed=seed)
    x, y = x.to(device), y.to(device)
    net = TinyTransformerB(d=d, H=H, m_ffn=m_ffn, S=S, ativ=ativ, seed=seed, beta=beta,
                           s_attn=s_attn, s_ffn=s_ffn, s_head=s_head).to(device)
    m = m_ffn
    idx = np.arange(m)

    # ---- injecao de nascidas-mortas (IDENTICA a motor_e5b: mesma dinamica) ----
    injetada = np.zeros(m, bool)
    injdepth = np.full(m, np.nan)
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

    params_treina = _params_treina(net, freeze_attn)

    loss_hist = []

    # ---- estado da histerese e do episodio corrente, por unidade -------------
    viva = np.ones(m, bool)              # estado (True=viva); definido em t=0 abaixo
    aberto = np.zeros(m, bool)           # episodio morto aberto?
    esteve_morto = np.zeros(m, bool)
    A_mlp = np.zeros(m); A_drive = np.zeros(m)
    t_area_mlp = np.full(m, -1); t_area_drive = np.full(m, -1)
    t_morte_ep = np.full(m, -1)
    zpark_ep = np.full(m, np.inf); zhi_ep = np.full(m, -np.inf)
    classe_ep = [None] * m
    W1_morte = np.zeros((m, d)); b1_morte = np.zeros(m)

    # ---- acumuladores por unidade (colapso) ----------------------------------
    U_revived = np.zeros(m, bool)        # cruzou +d_h vindo de morta, alguma vez
    U_tres = np.full(m, np.inf)          # menor t_res (1a travessia genuina)
    U_zpark = np.full(m, np.inf)         # min zmax entre episodios
    U_zhi = np.full(m, -np.inf)          # max zmax entre episodios mortos
    U_neps = np.zeros(m, int)
    U_classe1 = [None] * m
    U_ndyn = np.zeros(m, int)            # nº de mortes DINAMICAS (t>0) da unidade
    # trio/canal do 1o episodio revivido (ou 1o episodio, se nunca reviveu)
    U_tarea = np.full(m, -1); U_Adrive = np.zeros(m); U_Amlp = np.zeros(m)
    U_own = np.zeros(m); U_canal = [None] * m; U_have_src = np.zeros(m, bool)

    n_revive = 0
    revive_neuronios = set()

    def _snap_own():
        return (net.W1.detach().cpu().numpy(), net.b1.detach().cpu().numpy())

    def _abre(k, t_morte, classe, zmax_k, W1_np, b1_np):
        A_mlp[k] = 0.0; A_drive[k] = 0.0
        t_area_mlp[k] = -1; t_area_drive[k] = -1
        t_morte_ep[k] = t_morte; zpark_ep[k] = zmax_k; zhi_ep[k] = zmax_k
        classe_ep[k] = classe; aberto[k] = True
        W1_morte[k] = W1_np[k]; b1_morte[k] = b1_np[k]
        U_neps[k] += 1
        if U_classe1[k] is None:
            U_classe1[k] = classe
        if classe == "dyn" and not injetada[k]:
            U_ndyn[k] += 1

    def _fecha(k, t_res, revived, W1_np, b1_np):
        zp = float(zpark_ep[k])
        own = float(np.sqrt(((W1_np[k] - W1_morte[k]) ** 2).sum() +
                            (b1_np[k] - b1_morte[k]) ** 2))
        if revived:
            if freeze_attn:
                canal = "proprio"
            else:
                canal = "montante" if own < 1e-9 else "proprio+montante"
        else:
            canal = None
        U_zpark[k] = min(U_zpark[k], zp)
        U_zhi[k] = max(U_zhi[k], float(zhi_ep[k]))
        # fonte do trio: 1o episodio REVIVIDO; se nenhum, o 1o episodio qualquer
        take = False
        if revived and (not U_have_src[k] or not U_revived[k]):
            take = True
        elif (not U_revived[k]) and (not U_have_src[k]):
            take = True
        if take:
            U_tarea[k] = int(t_area_drive[k]); U_Adrive[k] = float(A_drive[k])
            U_Amlp[k] = float(A_mlp[k]); U_own[k] = own; U_canal[k] = canal
            U_have_src[k] = True
        if revived:
            U_revived[k] = True
            if t_res is not None:
                U_tres[k] = min(U_tres[k], float(t_res))
        aberto[k] = False
        t_morte_ep[k] = -1

    for t in range(passos):
        yhat, z, a, h = net.forward_parts(x)
        res = yhat - y
        loss = 0.5 * (res ** 2).mean()
        z_det = z.detach().reshape(N * S, m)
        zmv, jsv = z_det.max(0)
        zmax = zmv.cpu().numpy(); jstar = jsv.cpu().numpy()

        if t == 0:
            # estado inicial pela histerese: morta se zmax<-d_h (injetadas caem aqui)
            morta0 = zmax < -delta_h
            viva = ~morta0
            if morta0.any():
                W1_np, b1_np = _snap_own()
                for k in np.where(morta0)[0]:
                    esteve_morto[k] = True
                    _abre(int(k), 0, "born", zmax[k], W1_np, b1_np)
        else:
            # mortes confirmadas (viva -> zmax<-d_h)
            morre = viva & (zmax < -delta_h)
            if morre.any():
                W1_np, b1_np = _snap_own()
                for k in np.where(morre)[0]:
                    esteve_morto[k] = True
                    viva[k] = False
                    _abre(int(k), int(t), "dyn", zmax[k], W1_np, b1_np)

        # acumulacao GT-AREA nos episodios ABERTOS (identica a e5b) ------------
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

        # voltas confirmadas (morta -> zmax>+d_h) ------------------------------
        if t > 0:
            revive = (~viva) & (zmax > delta_h)
            if revive.any():
                W1_np, b1_np = _snap_own()
                for k in np.where(revive)[0]:
                    n_revive += 1
                    revive_neuronios.add(int(k))
                    viva[k] = True
                    _fecha(int(k), int(t), True, W1_np, b1_np)

        if t % rec_cada == 0:
            loss_hist.append(float(loss.detach()))
        net.zero_grad(); loss.backward()
        with torch.no_grad():
            for p in params_treina:
                if p.grad is not None:
                    p += -eta * p.grad
    loss_hist.append(float(loss.detach()))

    # episodios ainda abertos no fim = CENSURADOS (unidade termina MORTA)
    W1_np, b1_np = _snap_own()
    for k in np.where(aberto)[0]:
        _fecha(int(k), None, False, W1_np, b1_np)

    # ---- colapso por unidade -------------------------------------------------
    unidades = []
    for k in range(m):
        rev = bool(U_revived[k])
        final_alive = bool(viva[k])
        sust = bool(rev and final_alive)
        deep = (bool(injdepth[k] < Z_STAR) if injetada[k]
                else (bool(U_zpark[k] < Z_STAR) if np.isfinite(U_zpark[k]) else False))
        unidades.append(dict(
            k=int(k), injetada=bool(injetada[k]),
            injdepth=(float(injdepth[k]) if injetada[k] else None),
            deep=deep, classe1=U_classe1[k],
            revived=rev, sustained=sust, final_alive=final_alive, hover=bool(rev and not final_alive),
            t_res=(int(U_tres[k]) if np.isfinite(U_tres[k]) else None),
            z_park=(float(U_zpark[k]) if np.isfinite(U_zpark[k]) else None),
            zmax_hi=(float(U_zhi[k]) if np.isfinite(U_zhi[k]) else None),
            n_eps=int(U_neps[k]), n_dyn=int(U_ndyn[k]),
            t_area_drive=int(U_tarea[k]), A_drive=float(U_Adrive[k]),
            A_mlp=float(U_Amlp[k]), own_move=float(U_own[k]), canal=U_canal[k]))

    n_inj = int(injetada.sum())
    inj_units = [u for u in unidades if u["injetada"]]
    metr = dict(
        loss_hist=loss_hist,
        plato=float(np.median(loss_hist[-max(3, len(loss_hist) // 10):])),
        loss_fim=float(loss_hist[-1]),
        n_injetadas=n_inj,
        n_born_natural=int(((U_neps > 0) & ~injetada & np.array(
            [c == "born" for c in U_classe1], bool)).sum()),
        n_natural_dyn=int(sum(1 for u in unidades
                              if (not u["injetada"]) and u["classe1"] == "dyn")),
        n_mortos=int(esteve_morto.sum()), n_revive=int(n_revive),
        n_neuronios_revive=len(revive_neuronios),
        n_inj_revived=sum(1 for u in inj_units if u["revived"]),
        n_inj_sustained=sum(1 for u in inj_units if u["sustained"]),
        n_inj_hover=sum(1 for u in inj_units if u["hover"]),
        unidades=unidades,
        cfg=dict(m_ffn=m_ffn, ativ=ativ, beta=beta, eta=eta, passos=passos, seed=seed,
                 N=N, S=S, d=d, H=H, delta_h=delta_h, freeze_attn=freeze_attn,
                 kill_depths=kd))
    return metr


# ============================ TESTES ============================
def _testes():
    print("== TESTES motor_e5_conf (histerese + volta sustentada; paridade e5b) ==")
    torch.set_num_threads(1)

    # (1) PARIDADE de DINAMICA: loss_hist byte-identico a motor_e5b (o instrumento
    #     e read-only). Varre bracos, fatores e injecoes.
    casos = [
        dict(ativ="softplus", beta=3.0, seed=2000, freeze_attn=True, kill_depths=(0.3,)),
        dict(ativ="softplus", beta=10.0, seed=2001, freeze_attn=False, kill_depths=(2.0,)),
        dict(ativ="gelu", beta=1.0, seed=2002, freeze_attn=True, kill_depths=(0.3, 1.0)),
        dict(ativ="relu", beta=1.0, seed=2003, freeze_attn=False, kill_depths=()),
    ]
    for c in casos:
        a = MB.treina_e5b(m_ffn=4, passos=600, N=24, S=3, d=16, H=2, eta=1e-3,
                          rec_cada=50, z_dead=0.0, z_alive=0.0, **c)
        b = treina_e5_conf(m_ffn=4, passos=600, N=24, S=3, d=16, H=2, eta=1e-3,
                           rec_cada=50, delta_h=0.05, **c)
        assert a["loss_hist"] == b["loss_hist"], (c, "loss_hist DIVERGE")
    print(f"[OK] PARIDADE: loss_hist byte-identico a motor_e5b em {len(casos)} casos "
          "(dinamica intacta; instrumento aditivo)")

    # (2) determinismo
    m1 = treina_e5_conf(ativ="softplus", beta=3.0, passos=400, seed=5, kill_depth=0.3)
    m2 = treina_e5_conf(ativ="softplus", beta=3.0, passos=400, seed=5, kill_depth=0.3)
    assert m1["loss_hist"] == m2["loss_hist"]
    print(f"[OK] determinismo: loss_fim={m1['loss_fim']:.6f}")

    # (3) injecao 1-nivel: kill_depth=0.3 -> 1 unidade rasa (deep=False);
    #     kill_depth=2.0 -> 1 unidade profunda (deep=True), em zmax=-depth exato.
    mr = treina_e5_conf(ativ="gelu", passos=100, seed=11, kill_depth=0.3)
    mp = treina_e5_conf(ativ="gelu", passos=100, seed=11, kill_depth=2.0)
    ir = [u for u in mr["unidades"] if u["injetada"]]
    ip = [u for u in mp["unidades"] if u["injetada"]]
    assert len(ir) == 1 and len(ip) == 1
    assert abs(ir[0]["injdepth"] + 0.3) < 1e-6 and ir[0]["deep"] is False
    assert abs(ip[0]["injdepth"] + 2.0) < 1e-6 and ip[0]["deep"] is True
    print(f"[OK] injecao 1-nivel: rasa z={ir[0]['injdepth']:.2f}(deep={ir[0]['deep']}) "
          f"profunda z={ip[0]['injdepth']:.2f}(deep={ip[0]['deep']}) barreira z*={Z_STAR}")

    # (4) VALIDADE relu-frozen: injetada morta => 0 voltas sustentadas (grad proprio==0
    #     E entrada fixa => terminal por teorema). Ambas profundidades.
    for dep in (0.3, 2.0):
        mt = treina_e5_conf(ativ="relu", passos=2000, seed=2, kill_depth=dep,
                            freeze_attn=True, delta_h=0.05)
        assert mt["n_inj_sustained"] == 0 and mt["n_inj_revived"] == 0, (dep, mt["n_inj_revived"])
        for u in mt["unidades"]:
            if u["injetada"]:
                assert u["own_move"] < 1e-12, (u["k"], u["own_move"])
    print("[OK] relu-frozen (rasa e profunda): 0 voltas sustentadas, own_move==0 (teorema)")

    # (5) HISTERESE mata chatter: no seed relu-train-103 (chatter PROVADO no piloto:
    #     rate-pilot.md 0 — 1a travessia genuina t=66846 + 159 micro-oscilacoes em
    #     zmax~=0), d_h=0 conta muitas voltas cruas; d_h=0.05 colapsa p/ a genuina.
    r0 = treina_e5_conf(ativ="relu", passos=80_000, seed=103, kill_depth=0.3,
                        freeze_attn=False, delta_h=0.0, rec_cada=300)
    rh = treina_e5_conf(ativ="relu", passos=80_000, seed=103, kill_depth=0.3,
                        freeze_attn=False, delta_h=0.05, rec_cada=300)
    assert r0["n_revive"] >= 2, ("esperava chatter cru em s103", r0["n_revive"])
    assert rh["n_revive"] < r0["n_revive"], (rh["n_revive"], r0["n_revive"])
    for u in rh["unidades"]:
        if u["sustained"]:
            assert u["zmax_hi"] > 0.05 - 1e-9, (u["k"], u["zmax_hi"])
    print(f"[OK] histerese: voltas cruas d_h=0 ({r0['n_revive']}) -> confirmadas d_h=0.05 "
          f"({rh['n_revive']}) no seed-chatter s103; sustentadas tem zmax_hi>d_h")

    # (6) sustained <=> (revived and final_alive); hover <=> (revived and not final)
    mt = treina_e5_conf(ativ="softplus", beta=3.0, passos=60_000, seed=100,
                        kill_depth=0.3, freeze_attn=False, delta_h=0.05)
    for u in mt["unidades"]:
        assert u["sustained"] == (u["revived"] and u["final_alive"])
        assert u["hover"] == (u["revived"] and not u["final_alive"])
        if u["sustained"]:
            assert u["t_res"] is not None
    print(f"[OK] sustained<=>(revived&final_alive); hover<=>(revived&~final): coerente "
          f"(inj_rev={mt['n_inj_revived']} sust={mt['n_inj_sustained']} hover={mt['n_inj_hover']})")

    # (7) consistencia: 1 injetada/run; contagens sust<=rev<=inj; z_park do injetado ~ injdepth
    mt = treina_e5_conf(ativ="softplus", beta=3.0, passos=3000, seed=7, kill_depth=0.3)
    assert mt["n_injetadas"] == 1
    assert mt["n_inj_sustained"] <= mt["n_inj_revived"] <= mt["n_injetadas"]
    inj = [u for u in mt["unidades"] if u["injetada"]][0]
    assert inj["z_park"] <= -0.3 + 1e-6
    print(f"[OK] consistencia: n_inj=1, sust<=rev<=inj, z_park({inj['z_park']:.3f})<=inj")

    print("== TODOS OS TESTES (motor_e5_conf) PASSARAM ==")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--testes", action="store_true")
    a = ap.parse_args()
    if a.testes:
        _testes()
    else:
        print("uso: python engine_e5_confirmatory.py --testes")
