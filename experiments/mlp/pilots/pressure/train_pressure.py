"""CÓPIA DERIVADA de harness_mlp.treina (piloto etapa-4a.3 "pressão").

Regra da casa: NÃO modifica harness_mlp.py. Esta é uma cópia da função `treina`
adaptada em EXATAMENTE três pontos, documentados aqui:

  (1) DADOS: injeta o alvo RICO subparametrizado `dados_rico` (x~U[0,2], n=64,
      y = sin(2πx) + 0.3·sin(5πx), ~8-12 kinks) em vez de dados_1dreg. O harness
      já aceitava `dados=`, mas fixo aqui por clareza do piloto.

  (2) RÉGUA DUPLA: instancia a v2 (harness_mlp.Regua) E a v3 (regua_v3.ReguaV3)
      e chama `.observa()` nas DUAS na MESMA trajetória, mesmo passo, mesmos
      z/a/res/v pré-update. (O harness só instancia a v2 internamente; era o
      único motivo de copiar a função.)

  (3) GROUND-TRUTH régua-independente: registra por passo o histórico de loss e
      a MORTE-REAL de neurônio = zmax_k = max_j z_{j,k} < 0 (ativação desligada
      em TODOS os dados ⟹ para relu grad≡0). Conta transições morte-real→vivo
      (`n_revive_real`), que para relu a T=0 DEVE ser 0 (freeze determinístico).

Toda a DINÂMICA (init do MLP, ordem observa-pré-update, sorteio de ruído para
todos os params, update p += (√(2ηT)·ξ − η∇L), máscara de freeze≡1) é idêntica
ao original harness_mlp.treina. Verificação de paridade em `_paridade()`.
"""
import numpy as np
import torch

from harness_mlp import MLP, Regua, dados_1dreg  # reuso; NÃO modifica o harness
from instrument_v3 import ReguaV3

torch.use_deterministic_algorithms(True)


def dados_rico(n=64, seed=0):
    """Alvo RICO subparametrizado. x~U[0,2] (n pontos), y=sin(2πx)+0.3 sin(5πx).
    ~8-12 kinks: um MLP raso de m pequeno NÃO fecha o resíduo (pressão persiste)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, generator=g) * 2.0
    y = torch.sin(2.0 * np.pi * x) + 0.3 * torch.sin(5.0 * np.pi * x)
    return x, y


def treina_pressao(m=8, ativ="softplus", beta=300.0, eta=3e-4, T=0.0,
                   passos=300_000, seed=0, device="cpu", n_sigma=10.0,
                   rec_cada=100, n_dados=64, rec_neuronio=0):
    """Retorna (net, reg2, reg3, metricas). metricas é dict com loss_hist,
    ground-truth de morte-real e contagens auxiliares. reg2/reg3 têm .episodios,
    .censurados(passos), .switches como no harness. `n_dados` = nº de pontos do
    alvo rico (design v1 n=64; design rápido n=16)."""
    x, y = dados_rico(n=n_dados, seed=seed)
    x, y = x.to(device), y.to(device)
    net = MLP(m, ativ, beta, seed).to(device)
    reg2 = Regua(m, eta, T)
    reg3 = ReguaV3(m, eta, T, n_sigma=n_sigma)
    x_np = x.cpu().numpy()
    g = torch.Generator(device=device).manual_seed(seed + 1)
    sig = float(np.sqrt(2 * eta * T))          # = 0 em T=0 ⟹ GD puro, sem chatter

    loss_hist = []
    esteve_morto_real = np.zeros(m, bool)       # zmax<0 em algum passo (grad≡0 relu)
    morto_real_prev = np.zeros(m, bool)
    n_revive_real = 0                           # transições morte-real -> vivo
    revive_real_neuronios = set()
    revive_real_eventos = []                    # (t, k) para diagnóstico
    # (opcional) série por neurônio p/ diagnóstico de deriva: t, zmax_k, R_leader_k, v_k
    serie_t, serie_zmax, serie_R, serie_v, serie_n2 = [], [], [], [], []
    normas_x = np.sqrt(x_np[:, 0] ** 2 + 1.0)   # ‖(x_j,1)‖ por dado
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
        # (2) régua dupla na MESMA trajetória, estado pré-update
        reg2.observa(t, z_np, a_np, res_np, v_np, x_np)
        reg3.observa(t, z_np, a_np, res_np, v_np, x_np)
        # (3) ground-truth régua-independente
        zmax = z_np.max(0)
        jstar = z_np.argmax(0)
        if rec_neuronio and (t % rec_neuronio == 0):
            serie_t.append(int(t))
            serie_zmax.append(zmax.copy())
            serie_R.append((-res_np[jstar]).copy())      # R = (y-f) no líder de cada k
            serie_v.append(v_np.copy())
            serie_n2.append((normas_x[jstar] ** 2).copy())  # ‖(x_j*,1)‖² do líder
        morto_real = zmax < 0.0
        esteve_morto_real |= morto_real
        revive = morto_real_prev & (zmax > 0.0)
        if revive.any():
            for k in np.where(revive)[0]:
                n_revive_real += 1
                revive_real_neuronios.add(int(k))
                if len(revive_real_eventos) < 200:
                    revive_real_eventos.append((int(t), int(k), float(zmax[k])))
        morto_real_prev = morto_real
        if t % rec_cada == 0:
            loss_hist.append(float(loss.detach()))
        # update — dinâmica IDÊNTICA ao harness (ruído p/ todos; sig=0 em T=0)
        net.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in net.parameters():
                ruido = torch.randn(p.shape, generator=g, device=device)
                dp = sig * ruido - eta * p.grad
                p += dp
    loss_hist.append(float(loss.detach()))
    metr = dict(
        loss_hist=loss_hist,
        n_mortos_real=int(esteve_morto_real.sum()),
        n_revive_real=int(n_revive_real),
        n_neuronios_revive_real=len(revive_real_neuronios),
        revive_real_eventos=revive_real_eventos,
        normas_x=normas_x,
    )
    if rec_neuronio:
        metr["serie"] = dict(
            t=np.array(serie_t),
            zmax=np.array(serie_zmax),   # (T_rec, m)
            R=np.array(serie_R),         # (T_rec, m)
            v=np.array(serie_v),         # (T_rec, m)
            n2=np.array(serie_n2),       # (T_rec, m) ‖(x_j*,1)‖²
        )
    return net, reg2, reg3, metr


def _paridade():
    """Confirma que a trajetória (loss final) é bit-idêntica à do harness original
    quando o harness recebe os MESMOS dados ricos e mesmo seed, T=0. Isto prova que
    a adaptação (régua dupla + ground-truth) não tocou a dinâmica."""
    from harness_mlp import treina
    xr, yr = dados_rico(n=64, seed=42)
    P = 300
    net_a, r2a, l_a, _ = treina(m=8, ativ="relu", beta=300.0, eta=3e-4, T=0.0,
                                passos=P, seed=42, dados=(xr, yr))
    net_b, r2b, r3b, metr = treina_pressao(m=8, ativ="relu", eta=3e-4, T=0.0,
                                           passos=P, seed=42)
    # harness devolve a loss da ÚLTIMA iteração (pré-update final); é exatamente
    # loss_hist[-1] do piloto (appendado após o loop). Comparação bit-a-bit.
    lb = metr["loss_hist"][-1]
    assert abs(l_a - lb) < 1e-12, (l_a, lb)
    # v2 do harness == v2 do piloto (mesma trajetória, mesma régua)
    assert len(r2a.episodios) == len(r2b.episodios), \
        (len(r2a.episodios), len(r2b.episodios))
    print(f"[OK] paridade dinâmica com harness: loss_harness={l_a:.6e} "
          f"loss_piloto={lb:.6e} Δ={abs(l_a-lb):.1e}; "
          f"v2 episódios harness={len(r2a.episodios)} piloto={len(r2b.episodios)}")
    print(f"[OK] régua dupla ativa: v2={len(r2b.episodios)} v3={len(r3b.episodios)} "
          f"episódios completos em {P} passos (m=8 relu seed42)")


if __name__ == "__main__":
    _paridade()
