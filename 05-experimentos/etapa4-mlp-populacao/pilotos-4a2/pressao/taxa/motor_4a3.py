"""Motor do CONFIRMATÓRIO etapa-4a.3, com o INSTRUMENTO GT-ÁREA. ARQUIVO NOVO.

Derivado de motor_taxa.treina_taxa (que já é bit-idêntico a treina_pressao a T=0).
A DINÂMICA é intocada; a única adição é um instrumento de área ancorado no
RELÓGIO ÚNICO ground-truth, substituindo a régua v2 (que era cega/atrasada e
introduzia empates de discretização — Diagnóstico C2 do piloto-de-taxa).

INSTRUMENTO GT-ÁREA (todos os sinais no relógio ground-truth, régua-independente):
  - morte GT  = zmax_k = max_j z_{j,k} < 0 em TODOS os dados (mesma convenção do
                contador-mestre de motor_taxa);
  - volta GT  = retorno a zmax_k > 0;
  - durante o episódio GT-morto de k acumula POR PASSO
        Â_k += R̂_k · η,   R̂_k = v_k·[ (y−f)_{j*} + v_k·a_{j*,k} ]
    (a MESMA projeção-líder da régua/série, mas no relógio GT — nunca a v2);
  - t_área = 1º passo do episódio com Â_k > 0 (relógio GT);
  - z_park = profundidade do estacionamento = min de zmax_k no episódio;
  - σ(β·z_park) = canal de reativação avaliado no fundo do park.

Cada episódio GT-morto vira um dict (k, classe ∈ {born,dyn}, t_morte, t_area,
t_res, revived, z_park, sigma_park). C2 (o coração) lê o trio (t_morte, t_area,
t_res) DESTE instrumento; nada de v2 no caminho crítico.

PARIDADE (teste _paridade abaixo, OBRIGATÓRIO): a dinâmica e o contador-mestre
(nasce_morto, death_events, revive_events, n_revive_real) são bit-idênticos aos
de motor_taxa.treina_taxa — o instrumento de área é aditivo e read-only.
"""
import numpy as np
import torch

from harness_mlp import MLP  # reuso; NÃO instancia Regua (sem v2 no caminho crítico)

torch.use_deterministic_algorithms(True)


def dados_rico(n=64, seed=0):
    """IDÊNTICO a motor_taxa/treina_pressao.dados_rico. x~U[0,2], y=sin(2πx)+0.3sin(5πx)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, generator=g) * 2.0
    y = torch.sin(2.0 * np.pi * x) + 0.3 * torch.sin(5.0 * np.pi * x)
    return x, y


def treina_4a3(m=6, ativ="softplus", beta=10.0, eta=1e-3, T=0.0,
               passos=500_000, seed=0, device="cpu", n_dados=16, rec_cada=100):
    """Retorna (metr). metr = contador-mestre ground-truth (idêntico a motor_taxa)
    + episodios_gt (instrumento GT-ÁREA). Sem reg2. Se T>0 cai no caminho com ruído
    (paridade só é reivindicada a T=0, o regime do confirmatório)."""
    x, y = dados_rico(n=n_dados, seed=seed)
    x, y = x.to(device), y.to(device)
    net = MLP(m, ativ, beta, seed).to(device)
    x_np = x.cpu().numpy()
    com_ruido = T > 0.0
    if com_ruido:
        g = torch.Generator(device=device).manual_seed(seed + 1)
        sig = float(np.sqrt(2 * eta * T))

    loss_hist = []
    idx = np.arange(m)

    # contador-mestre ground-truth (idêntico, bit-a-bit, a motor_taxa) -----------
    esteve_morto_real = np.zeros(m, bool)
    morto_real_prev = np.zeros(m, bool)
    nasce_morto = None
    death_events = []            # (t, k) vivo->morto, t>0 = DINÂMICA
    revive_events = []           # (t, k) morto->vivo (zmax>0)
    n_revive_real = 0
    revive_real_neuronios = set()

    # instrumento GT-ÁREA: estado do episódio GT-morto corrente por neurônio -----
    A_hat = np.zeros(m)                 # área acumulada no episódio corrente
    t_area_ep = np.full(m, -1)          # 1º passo com Â>0 no episódio corrente
    t_morte_ep = np.full(m, -1)         # t_morte do episódio aberto (-1 = fechado)
    zpark_ep = np.full(m, np.inf)       # min zmax no episódio corrente
    classe_ep = [None] * m             # 'born' | 'dyn'
    episodios_gt = []                   # episódios GT-mortos fechados (revived) + censurados no fim

    def _abre(k, t_morte, classe, zmax_k):
        A_hat[k] = 0.0
        t_area_ep[k] = -1
        t_morte_ep[k] = t_morte
        zpark_ep[k] = zmax_k
        classe_ep[k] = classe

    def _fecha(k, t_res, revived):
        zp = float(zpark_ep[k])
        sigma = 1.0 / (1.0 + np.exp(-float(beta) * zp))   # canal σ(β·z_park)
        episodios_gt.append(dict(
            k=int(k), classe=classe_ep[k], t_morte=int(t_morte_ep[k]),
            t_area=int(t_area_ep[k]), t_res=(int(t_res) if t_res is not None else None),
            revived=bool(revived), z_park=zp, sigma_park=float(sigma)))
        t_morte_ep[k] = -1

    loss = None
    for t in range(passos):
        z = net.margens(x)
        a = net.act(z)
        f = a @ net.v
        res = f - y[:, 0]
        loss = 0.5 * (res ** 2).mean()
        z_np = z.detach().cpu().numpy()
        a_np = a.detach().cpu().numpy()
        res_np = res.detach().cpu().numpy()
        v_np = net.v.detach().cpu().numpy()
        zmax = z_np.max(0)
        jstar = z_np.argmax(0)
        # projeção-líder R̂_k (res = f - y ⟹ (y-f) = -res), no relógio GT
        R_hat = v_np * (-res_np[jstar] + v_np * a_np[jstar, idx])
        morto_real = zmax < 0.0
        esteve_morto_real |= morto_real
        # 1) transições/aberturas de episódio (idênticas ao contador-mestre) -----
        if t == 0:
            nasce_morto = morto_real.copy()
            for k in np.where(morto_real)[0]:
                _abre(k, 0, "born", zmax[k])
            morto_real_prev = morto_real.copy()
        else:
            morre = (~morto_real_prev) & morto_real
            if morre.any():
                for k in np.where(morre)[0]:
                    death_events.append((int(t), int(k)))
                    _abre(k, int(t), "dyn", zmax[k])
        # 2) acumulação GT-ÁREA para episódios abertos que estão mortos AGORA -----
        acc = (t_morte_ep >= 0) & morto_real
        if acc.any():
            A_hat[acc] += R_hat[acc] * eta
            cruza = acc & (t_area_ep < 0) & (A_hat > 0.0)
            t_area_ep[cruza] = t
            zpark_ep[acc] = np.minimum(zpark_ep[acc], zmax[acc])
        # 3) voltas GT (idênticas ao contador-mestre) — fecham o episódio ---------
        if t > 0:
            revive = morto_real_prev & (zmax > 0.0)
            if revive.any():
                for k in np.where(revive)[0]:
                    n_revive_real += 1
                    revive_real_neuronios.add(int(k))
                    revive_events.append((int(t), int(k)))
                    _fecha(k, int(t), True)
            morto_real_prev = morto_real
        if t % rec_cada == 0:
            loss_hist.append(float(loss.detach()))
        # update — dinâmica IDÊNTICA a motor_taxa/treina_pressao (T=0 ⟹ GD puro) --
        net.zero_grad()
        loss.backward()
        with torch.no_grad():
            if com_ruido:
                for p in net.parameters():
                    ruido = torch.randn(p.shape, generator=g, device=device)
                    p += sig * ruido - eta * p.grad
            else:
                for p in net.parameters():
                    p += -eta * p.grad
    loss_hist.append(float(loss.detach()))
    # episódios ainda abertos no fim do horizonte = CENSURADOS (nunca 'permanente')
    for k in np.where(t_morte_ep >= 0)[0]:
        _fecha(k, None, False)

    metr = dict(
        loss_hist=loss_hist,
        nasce_morto=[int(k) for k in np.where(nasce_morto)[0]],
        n_nasce_morto=int(nasce_morto.sum()),
        death_events=death_events,
        revive_events=revive_events,
        n_mortos_real=int(esteve_morto_real.sum()),
        n_revive_real=int(n_revive_real),
        n_neuronios_revive_real=len(revive_real_neuronios),
        episodios_gt=episodios_gt,
    )
    return metr


def _paridade():
    """Prova que a DINÂMICA (loss_hist bit-a-bit) e o CONTADOR-MESTRE ground-truth
    (nasce_morto, death_events, revive_events, n_revive_real) de motor_4a3 são
    idênticos aos de motor_taxa.treina_taxa — o instrumento GT-ÁREA é aditivo e
    read-only. Também checa a CONSISTÊNCIA interna dos episodios_gt com os eventos."""
    import motor_taxa as MT
    P = 400
    casos = [
        dict(m=4, ativ="relu", beta=300.0, seed=100),
        dict(m=6, ativ="softplus", beta=10.0, seed=101),
        dict(m=4, ativ="softplus", beta=3.0, seed=129),
        dict(m=6, ativ="softplus", beta=300.0, seed=115),
        dict(m=6, ativ="softplus", beta=3.0, seed=106),   # tinha voltas no piloto
    ]
    for c in casos:
        metr_b = treina_4a3(passos=P, eta=1e-3, T=0.0, n_dados=16, rec_cada=100, **c)
        _reg2, metr_a = MT.treina_taxa(passos=P, eta=1e-3, T=0.0, n_dados=16,
                                       rec_cada=100, **c)
        # (1) dinâmica: loss_hist bit-a-bit
        assert len(metr_a["loss_hist"]) == len(metr_b["loss_hist"])
        dmax = max(abs(x - y) for x, y in zip(metr_a["loss_hist"], metr_b["loss_hist"]))
        assert dmax < 1e-12, (c, dmax)
        # (2) contador-mestre: eventos idênticos
        assert metr_a["nasce_morto"] == metr_b["nasce_morto"], c
        assert metr_a["death_events"] == metr_b["death_events"], c
        assert metr_a["revive_events"] == metr_b["revive_events"], c
        assert metr_a["n_revive_real"] == metr_b["n_revive_real"], c
        # (3) consistência interna do instrumento GT-ÁREA
        eps = metr_b["episodios_gt"]
        n_abertos = metr_b["n_nasce_morto"] + len(metr_b["death_events"])
        assert len(eps) == n_abertos, (c, len(eps), n_abertos)
        n_rev_ep = sum(1 for e in eps if e["revived"])
        assert n_rev_ep == metr_b["n_revive_real"], (c, n_rev_ep, metr_b["n_revive_real"])
        # trio bem-formado: t_area, quando aberta, cai no intervalo [t_morte, t_res)
        for e in eps:
            if e["t_area"] >= 0:
                assert e["t_morte"] <= e["t_area"], (c, e)
                if e["t_res"] is not None:
                    assert e["t_area"] < e["t_res"], (c, e)
        print(f"[OK] {c['ativ']}-b{c['beta']:g} m{c['m']} s{c['seed']}: "
              f"loss Δ={dmax:.1e} eps={len(eps)} rev={n_rev_ep} "
              f"born={metr_b['n_nasce_morto']} deaths_dyn={len(metr_b['death_events'])}")
    print("== PARIDADE OK: motor_4a3 dinâmica+contador bit-idênticos a motor_taxa (T=0) ==")


if __name__ == "__main__":
    _paridade()
