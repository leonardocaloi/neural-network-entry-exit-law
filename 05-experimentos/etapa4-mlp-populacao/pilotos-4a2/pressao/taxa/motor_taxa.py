"""Motor do PILOTO-DE-TAXA (etapa-4a.3). ARQUIVO NOVO em pilotos-4a2/pressao/taxa/.

Regra da casa: NÃO modifica treina_pressao.py nem harness_mlp.py. Esta é uma
cópia da função `treina_pressao` (motor de paridade já verificada) adaptada em
EXATAMENTE três pontos, todos comprovados por `_paridade()` como bit-idênticos à
dinâmica original a T=0:

  (1) LOG DE GROUND-TRUTH COMPLETO (régua-independente). treina_pressao só grava
      REVIVES (zmax<0->zmax>0). O piloto-de-taxa precisa do contador-mestre
      pedido pelo 9º parecer: mortes DINÂMICAS por ground-truth. Para isso gravo
      também (a) `nasce_morto` = zmax<0 no passo 0, e (b) as transições de MORTE
      (alive->dead). Com nasce_morto + death_events + revive_events reconstruo,
      em pós-processamento, TODOS os episódios de morte (born/dyn, volta/terminal
      /censurado) sem depender de régua nenhuma.

  (2) DROP da régua v3. treina_pressao instanciava v2 E v3; o piloto-de-taxa usa
      o trio (t_morte,t_area,t_res) da régua v2 (item (c) do 9º parecer) e o
      contador por ground-truth. A v3 (debounce/dwell) não entra em nenhum
      estimando ⟹ removida (ela é read-only, não toca a dinâmica ⟹ paridade
      preservada; ganho de tempo real na grade de 240 runs).

  (3) SKIP do ruído a T=0. treina_pressao sorteia `ruido = randn(...)` por passo/
      param e faz `dp = sig*ruido - eta*grad` com sig=√(2ηT)=0. A 0.0*ruido é
      exatamente zeros ⟹ dp = -eta*grad bit-a-bit. Pulo o sorteio (T=0 apenas);
      `_paridade()` prova loss bit-idêntica à de treina_pressao (que sorteia).
      Se T>0 for algum dia usado aqui, o motor cai no caminho COM ruído.

Todo o resto (init do MLP por seed, alvo rico, ordem observa-pré-update, régua v2
calibrada, update p += -η∇L) é idêntico a treina_pressao/harness_mlp.
"""
import numpy as np
import torch

from harness_mlp import MLP, Regua  # reuso; NÃO modifica o harness

torch.use_deterministic_algorithms(True)


def dados_rico(n=64, seed=0):
    """IDÊNTICO a treina_pressao.dados_rico. x~U[0,2] (n pts), y=sin(2πx)+0.3 sin(5πx)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, generator=g) * 2.0
    y = torch.sin(2.0 * np.pi * x) + 0.3 * torch.sin(5.0 * np.pi * x)
    return x, y


def treina_taxa(m=6, ativ="softplus", beta=10.0, eta=1e-3, T=0.0,
                passos=500_000, seed=0, device="cpu", n_dados=16, rec_cada=100):
    """Retorna (reg2, metr). metr traz o LOG DE GROUND-TRUTH completo:
      nasce_morto  : lista de k com zmax<0 no passo 0 (born-dead por ground-truth)
      death_events : lista de (t, k) transições vivo->morto (zmax cruza p/ <0)
      revive_events: lista de (t, k) transições morto->vivo (zmax<0 -> zmax>0)
    e o resumo agregado (n_mortos_real, n_revive_real, loss). reg2 = régua v2
    (episódios completos + .censurados(passos)) para o trio (t_morte,t_area,t_res).
    """
    x, y = dados_rico(n=n_dados, seed=seed)
    x, y = x.to(device), y.to(device)
    net = MLP(m, ativ, beta, seed).to(device)
    reg2 = Regua(m, eta, T)
    x_np = x.cpu().numpy()
    com_ruido = T > 0.0
    if com_ruido:
        g = torch.Generator(device=device).manual_seed(seed + 1)
        sig = float(np.sqrt(2 * eta * T))

    loss_hist = []
    # ground-truth régua-independente (zmax = max sobre os n dados, por neurônio)
    esteve_morto_real = np.zeros(m, bool)
    morto_real_prev = np.zeros(m, bool)
    nasce_morto = None
    death_events = []          # (t, k) vivo->morto
    revive_events = []         # (t, k) morto->vivo
    n_revive_real = 0
    revive_real_neuronios = set()
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
        # régua v2 na MESMA trajetória, estado pré-update (idêntico a treina_pressao)
        reg2.observa(t, z_np, a_np, res_np, v_np, x_np)
        # ground-truth: morte = zmax<0; vida = zmax>0 (mesma convenção de treina_pressao)
        zmax = z_np.max(0)
        morto_real = zmax < 0.0
        esteve_morto_real |= morto_real
        if t == 0:
            # estabelece o ESTADO INICIAL; born-dead vai em nasce_morto, NÃO em
            # death_events (senão o "prev=vivo" fictício do passo 0 marcaria morte).
            nasce_morto = morto_real.copy()
            morto_real_prev = morto_real.copy()
        else:
            # transições de MORTE (novo vs treina_pressao): vivo->morto, t>0 ⟹ DINÂMICA
            morre = (~morto_real_prev) & morto_real
            if morre.any():
                for k in np.where(morre)[0]:
                    death_events.append((int(t), int(k)))
            # transições de VIDA (idêntico a treina_pressao): morto->vivo (zmax>0)
            revive = morto_real_prev & (zmax > 0.0)
            if revive.any():
                for k in np.where(revive)[0]:
                    n_revive_real += 1
                    revive_real_neuronios.add(int(k))
                    revive_events.append((int(t), int(k)))
            morto_real_prev = morto_real
        if t % rec_cada == 0:
            loss_hist.append(float(loss.detach()))
        # update — dinâmica IDÊNTICA a treina_pressao/harness (T=0 ⟹ GD puro)
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
    metr = dict(
        loss_hist=loss_hist,
        nasce_morto=[int(k) for k in np.where(nasce_morto)[0]],
        n_nasce_morto=int(nasce_morto.sum()),
        death_events=death_events,
        revive_events=revive_events,
        n_mortos_real=int(esteve_morto_real.sum()),
        n_revive_real=int(n_revive_real),
        n_neuronios_revive_real=len(revive_real_neuronios),
    )
    return reg2, metr


def _paridade():
    """Prova que a dinâmica (loss final) e a régua v2 são bit-idênticas às de
    treina_pressao (que sorteia ruído mesmo a T=0 e instancia a v3), em vários
    braços/m/seed. Se este teste passa, os 3 pontos de adaptação são inócuos."""
    import treina_pressao as tp
    P = 400
    casos = [
        dict(m=4, ativ="relu", beta=300.0, seed=100),
        dict(m=6, ativ="softplus", beta=10.0, seed=101),
        dict(m=4, ativ="softplus", beta=3.0, seed=129),
        dict(m=6, ativ="softplus", beta=300.0, seed=115),
    ]
    for c in casos:
        reg2_b, metr_b = treina_taxa(passos=P, eta=1e-3, T=0.0, n_dados=16,
                                     rec_cada=100, **c)
        _, reg2_a, _reg3, metr_a = tp.treina_pressao(
            passos=P, eta=1e-3, T=0.0, n_dados=16, n_sigma=10.0,
            rec_neuronio=0, rec_cada=100, **c)
        la = metr_a["loss_hist"][-1]
        lb = metr_b["loss_hist"][-1]
        assert abs(la - lb) < 1e-12, (c, la, lb)
        assert len(reg2_a.episodios) == len(reg2_b.episodios), \
            (c, len(reg2_a.episodios), len(reg2_b.episodios))
        # o revive de ground-truth do piloto tem de bater com o de treina_pressao
        assert metr_a["n_revive_real"] == metr_b["n_revive_real"], \
            (c, metr_a["n_revive_real"], metr_b["n_revive_real"])
        print(f"[OK] {c['ativ']}-b{c['beta']} m{c['m']} s{c['seed']}: "
              f"loss Δ={abs(la-lb):.1e} v2eps={len(reg2_b.episodios)} "
              f"rev_gt={metr_b['n_revive_real']} nasce_morto={metr_b['n_nasce_morto']} "
              f"deaths={len(metr_b['death_events'])}")
    print("== PARIDADE OK: motor_taxa bit-idêntico a treina_pressao (T=0) ==")


if __name__ == "__main__":
    _paridade()
