"""Etapa 4 — harness de MLP com a régua populacional embarcada (v2, pós-6º parecer).

MLP raso: f(x) = sum_k v_k * act(w_k·x + b_k), treino SGD/SGLD full-batch,
régua O(m) por passo: morto (margem líder <= limiar calibrado), dado líder j*,
resíduo projetado R̂_k = v_k·[(y-f)_{j*} + v_k·act(z_k,j*)], área acumulada Â_k
por episódio, rival t_sinal (1º R̂>0), flags de v_k (morte-por-saída, cruzamento
de sinal) e log global de switchings para o filtro de episódio-limpo.

Correções do 6º parecer (review-06-stage4-design.md):
- B1: limiar calibrado 2·sqrt(n·T)/‖(x_j*,1)‖ (o `*0` removido); observação
  PRÉ-update (off-by-one corrigido) e A CADA passo (Δsnapshot=1, igual à
  calibração); R̂ reconstruído com a MESMA ativação da rede (matriz act passada).
- B2: morte-por-saída |v_k|<φ_v e flag v_cruzou por episódio; contagem de
  trocas do líder j*; log de switchings + `episodio_limpo()` compartilhado.
- B4: device único pinado (confirmatórias: CPU) + use_deterministic_algorithms.
- Braços: softplus-β / erf-smoothing-β (classe do teorema, g'=Φ(βz)) /
  gelu PADRÃO z·Φ(z) (fora-de-classe, exploratório) / relu-hard;
  freeze-parcial com RNG alinhado (ruído sorteado p/ todos, update mascarado).

--smoke: mecânica apenas (finitude, determinismo, gradiente vs dif. finitas,
overfit-1-batch, âncora da fração nascida-morta, alinhamento RNG do freeze) —
NENHUMA estatística de predição.
"""
import argparse

import numpy as np
import torch

torch.use_deterministic_algorithms(True)

RAIZ2 = float(np.sqrt(2.0))


def dados_1dreg(n=32, seed=0, embaralha=False):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, generator=g) * 2.0
    y = torch.sin(2.0 * x) + 0.3 * x
    if embaralha:
        perm = torch.randperm(n, generator=g)
        y = y[perm]
    return x, y


class MLP(torch.nn.Module):
    def __init__(self, m, ativ, beta=300.0, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.w = torch.nn.Parameter(torch.randn(m, 1) * 1.0)
        self.b = torch.nn.Parameter(torch.randn(m) * 1.0)
        self.v = torch.nn.Parameter(torch.randn(m) / np.sqrt(m))
        self.ativ, self.beta = ativ, beta

    def margens(self, x):                      # (n, m)
        return x @ self.w.T + self.b

    def act(self, z):
        if self.ativ == "relu":
            return torch.relu(z)
        if self.ativ == "softplus":
            return torch.nn.functional.softplus(z, beta=self.beta)
        if self.ativ == "erf":                 # classe do teorema: g'(z)=Φ(βz)
            bz = self.beta * z                 # g(z)=z·Φ(βz)+φ(βz)/β
            Phi = 0.5 * (1.0 + torch.erf(bz / RAIZ2))
            phi = torch.exp(-0.5 * bz ** 2) / np.sqrt(2 * np.pi)
            return z * Phi + phi / self.beta
        if self.ativ == "gelu":                # GELU PADRÃO (fora-de-classe)
            return z * 0.5 * (1.0 + torch.erf(z / RAIZ2))
        raise ValueError(self.ativ)

    def forward(self, x):
        return self.act(self.margens(x)) @ self.v


class Regua:
    """O(m)/passo, vetorizada; a MESMA lógica calibrada no toy, com limiar
    norm-scaled 2·sqrt(n·T)/‖(x_j*,1)‖ e flags de v_k por episódio."""

    def __init__(self, m, eta, T):
        self.m, self.eta = m, eta
        self.lim_base = 2.0 * np.sqrt(max(T, 1e-7))   # × sqrt(n)/‖x_j*‖ em observa
        self.morto = np.zeros(m, bool)
        self.A_hat = np.zeros(m)
        self.t_morte = np.full(m, -1); self.t_area = np.full(m, -1)
        self.t_sinal = np.full(m, -1)
        self.v_morte = np.zeros(m); self.v_min_abs = np.full(m, np.inf)
        self.v_cruzou = np.zeros(m, bool); self.j_prev = np.full(m, -1)
        self.j_trocas = np.zeros(m, int)
        self.switches = []                     # (t, k, +1 ressurreição | -1 morte)
        self.episodios = []

    def _fecha(self, k, t_res):
        self.episodios.append(dict(
            k=int(k), t_morte=int(self.t_morte[k]), t_area=int(self.t_area[k]),
            t_sinal=int(self.t_sinal[k]), t_res=int(t_res),
            v_morte=float(self.v_morte[k]), v_min_abs=float(self.v_min_abs[k]),
            v_cruzou=bool(self.v_cruzou[k]), j_trocas=int(self.j_trocas[k])))

    def observa(self, t, z, a, res, v, x):
        """Estado CONSISTENTE pré-update. z:(n,m) margens, a:(n,m)=act(z) da
        PRÓPRIA ativação da rede, res:(n,)=f-y, v:(m,), x:(n,1)."""
        n = z.shape[0]
        zmax, jstar = z.max(0), z.argmax(0)
        normas = np.sqrt(x[:, 0] ** 2 + 1.0)
        lim = self.lim_base * np.sqrt(n) / normas[jstar]   # limiar CALIBRADO
        vivo = zmax > lim
        idx = np.arange(self.m)
        R_hat = v * (-res[jstar] + v * a[jstar, idx])      # res=f-y ⟹ (y-f)=-res
        # 1) mortes novas (mesma ordem da calibração: marca, depois acumula)
        novos = (~vivo) & (~self.morto)
        if novos.any():
            self.morto[novos] = True
            self.t_morte[novos] = t; self.A_hat[novos] = 0.0
            self.t_area[novos] = -1; self.t_sinal[novos] = -1
            self.v_morte[novos] = v[novos]
            self.v_min_abs[novos] = np.abs(v[novos])
            self.v_cruzou[novos] = False
            self.j_prev[novos] = jstar[novos]; self.j_trocas[novos] = 0
            for k in np.where(novos)[0]:
                self.switches.append((int(t), int(k), -1))
        # 2) acumulação para todos os mortos (inclui os novos, como no toy)
        mortos = self.morto
        if mortos.any():
            self.A_hat[mortos] += R_hat[mortos] * self.eta
            cruza = mortos & (self.t_area < 0) & (self.A_hat > 0)
            self.t_area[cruza] = t
            sin = mortos & (self.t_sinal < 0) & (R_hat > 0)
            self.t_sinal[sin] = t
            self.v_min_abs[mortos] = np.minimum(self.v_min_abs[mortos],
                                                np.abs(v[mortos]))
            self.v_cruzou[mortos] |= (np.sign(v[mortos]) !=
                                      np.sign(self.v_morte[mortos])) & (v[mortos] != 0)
            troca_j = mortos & (jstar != self.j_prev)
            self.j_trocas[troca_j] += 1
            self.j_prev[mortos] = jstar[mortos]
            # 3) ressurreições
            rev = mortos & vivo
            for k in np.where(rev)[0]:
                self._fecha(k, t)
                self.switches.append((int(t), int(k), +1))
            self.morto[rev] = False

    def censurados(self, t_fim):
        """Episódios abertos no fim do horizonte (censura, nunca 'permanente')."""
        out = []
        for k in np.where(self.morto)[0]:
            out.append(dict(
                k=int(k), t_morte=int(self.t_morte[k]), t_area=int(self.t_area[k]),
                t_sinal=int(self.t_sinal[k]), t_res=-1,
                v_morte=float(self.v_morte[k]), v_min_abs=float(self.v_min_abs[k]),
                v_cruzou=bool(self.v_cruzou[k]), j_trocas=int(self.j_trocas[k]),
                censurado_em=int(t_fim)))
        return out


def episodio_limpo(ep, switches, q, phi_v, q_j=0):
    """Filtro pré-registrável: episódio de k é limpo se (a) ≤ q switchings de
    OUTROS neurônios dentro de (t_morte, t_fim]; (b) v_k não cruzou sinal;
    (c) min|v_k| ≥ φ_v (exclui morte-por-saída); (d) ≤ q_j trocas de líder."""
    t_fim = ep["t_res"] if ep["t_res"] >= 0 else ep.get("censurado_em", ep["t_res"])
    alheios = sum(1 for (t, k, _) in switches
                  if k != ep["k"] and ep["t_morte"] < t <= t_fim)
    return (alheios <= q and not ep["v_cruzou"]
            and ep["v_min_abs"] >= phi_v and ep["j_trocas"] <= q_j)


def treina(m=16, ativ="softplus", beta=300.0, eta=1e-3, T=1e-3, passos=2000,
           seed=0, device="cpu", embaralha=False, sem_grad=False,
           freeze_meio_mortos=False, params0=None, dados=None, para_cedo=False):
    """Device ÚNICO por experimento (confirmatórias: cpu — B4). Ruído é sorteado
    para TODOS os parâmetros sempre; freeze só mascara o update (RNG alinhado ⟹
    braço-freeze pareável com o controle da mesma seed). `dados`/`para_cedo`
    são conveniências de CALIBRAÇÃO (injetar o toy; parar no 1º episódio de k=0)."""
    x, y = dados if dados is not None else dados_1dreg(seed=seed, embaralha=embaralha)
    x, y = x.to(device), y.to(device)
    net = MLP(m, ativ, beta, seed).to(device)
    if params0 is not None:
        with torch.no_grad():
            net.w.copy_(torch.as_tensor(params0["w"], dtype=net.w.dtype))
            net.b.copy_(torch.as_tensor(params0["b"], dtype=net.b.dtype))
            net.v.copy_(torch.as_tensor(params0["v"], dtype=net.v.dtype))
    reg = Regua(m, eta, T)
    x_np = x.cpu().numpy()
    # freeze-parcial (P-D): metade determinística dos nascidos-mortos
    congela = np.zeros(m, bool)
    if freeze_meio_mortos:
        with torch.no_grad():
            z0 = net.margens(x).cpu().numpy()
        n0 = z0.shape[0]
        normas0 = np.sqrt(x_np[:, 0] ** 2 + 1.0)
        lim0 = reg.lim_base * np.sqrt(n0) / normas0[z0.argmax(0)]
        nascidos = np.where(z0.max(0) <= lim0)[0]
        congela[nascidos[::2]] = True          # metade par: determinístico, sem RNG
    masc = {id(net.w): torch.as_tensor(~congela, device=device)[:, None].float(),
            id(net.b): torch.as_tensor(~congela, device=device).float(),
            id(net.v): torch.as_tensor(~congela, device=device).float()}
    g = torch.Generator(device=device).manual_seed(seed + 1)
    sig = np.sqrt(2 * eta * T)
    for t in range(passos):
        z = net.margens(x)
        a = net.act(z)
        f = a @ net.v
        res = f - y[:, 0]
        loss = 0.5 * (res ** 2).mean()
        reg.observa(t, z.detach().cpu().numpy(), a.detach().cpu().numpy(),
                    res.detach().cpu().numpy(), net.v.detach().cpu().numpy(), x_np)
        if para_cedo and any(ep["k"] == 0 for ep in reg.episodios):
            break
        net.zero_grad()
        if not sem_grad:
            loss.backward()
        with torch.no_grad():
            for p in net.parameters():
                ruido = torch.randn(p.shape, generator=g, device=device)
                dp = sig * ruido
                if not sem_grad:
                    dp = dp - eta * p.grad
                p += dp * masc[id(p)]
    return net, reg, float(loss.detach()), congela


def smoke():
    print("== SMOKE etapa-4 (harness v2): mecânica apenas ==")
    # (1) âncora analítica: fração nascida-morta no init (MC do init, sem treino)
    torch.manual_seed(0)
    M = 20000
    w = torch.randn(M, 1); b = torch.randn(M)
    x, _ = dados_1dreg()
    z = x @ w.T + b
    frac_mc = float((z.max(0).values <= 0).float().mean())
    print(f"[OK] fração nascida-morta (init padrão, MC {M}): {frac_mc:.3f}")
    # (2) gradiente vs diferenças finitas (protocolo)
    net = MLP(4, "softplus", seed=7).double()
    xs, ys = dados_1dreg(n=8, seed=7)
    xs, ys = xs.double(), ys.double()
    loss = 0.5 * ((net(xs) - ys[:, 0]) ** 2).mean()
    loss.backward()
    eps = 1e-6
    for p in [net.w, net.b, net.v]:
        flat = p.detach().reshape(-1)
        i = 0
        orig = float(flat[i])
        with torch.no_grad():
            flat[i] = orig + eps
            lp = 0.5 * ((net(xs) - ys[:, 0]) ** 2).mean().item()
            flat[i] = orig - eps
            lm = 0.5 * ((net(xs) - ys[:, 0]) ** 2).mean().item()
            flat[i] = orig
        fd = (lp - lm) / (2 * eps)
        an = float(p.grad.reshape(-1)[i])
        assert abs(fd - an) < 1e-6 * max(1.0, abs(an)), (fd, an)
    print("[OK] gradiente = diferenças finitas (w, b, v; dtype float64)")
    # (3) overfit-1-batch (protocolo): n=4, loss deve afundar
    _, _, l4, _ = treina(m=32, eta=1e-2, T=0.0, passos=8000, seed=11)
    net4 = MLP(32, "softplus", seed=11)
    x4, y4 = dados_1dreg(n=4, seed=11)
    opt = torch.optim.SGD(net4.parameters(), lr=1e-2)
    for _ in range(8000):
        lo = 0.5 * ((net4(x4) - y4[:, 0]) ** 2).mean()
        opt.zero_grad(); lo.backward(); opt.step()
    assert float(lo.detach()) < 1e-3, float(lo.detach())
    print(f"[OK] overfit-1-batch (n=4, m=32): loss={float(lo.detach()):.2e}")
    # (4) treino + régua v2: finitude, episódios, censura
    net, reg, loss, _ = treina(passos=2000, seed=3)
    cens = reg.censurados(2000)
    assert np.isfinite(loss)
    print(f"[OK] treino m=16 softplus 2000 passos: loss={loss:.4f}, "
          f"mortos agora={int(reg.morto.sum())} (censurados={len(cens)}), "
          f"episódios completos={len(reg.episodios)}, switchings={len(reg.switches)}")
    # (5) determinismo (device único, CPU)
    _, r2, l2, _ = treina(passos=2000, seed=3)
    assert abs(l2 - loss) < 1e-12 and int(r2.morto.sum()) == int(reg.morto.sum())
    print("[OK] determinismo CPU (mesma seed ⟹ mesma loss e mesmos mortos)")
    # (6) alinhamento RNG do freeze: freeze com conjunto vazio == controle
    _, _, lc, cg = treina(passos=500, seed=3, freeze_meio_mortos=False)
    _, _, lf, cf = treina(passos=500, seed=99991, freeze_meio_mortos=True)
    _, _, lf0, _ = treina(passos=500, seed=3, freeze_meio_mortos=True,
                          params0=dict(w=np.full((16, 1), 5.0), b=np.full(16, 5.0),
                                       v=np.ones(16)))       # ninguém nasce morto
    _, _, lc0, _ = treina(passos=500, seed=3, freeze_meio_mortos=False,
                          params0=dict(w=np.full((16, 1), 5.0), b=np.full(16, 5.0),
                                       v=np.ones(16)))
    assert abs(lf0 - lc0) < 1e-12, (lf0, lc0)
    print(f"[OK] RNG alinhado: braço-freeze com conjunto vazio ≡ controle "
          f"(Δloss={abs(lf0-lc0):.1e}); freeze real congelou {int(cf.sum())} de 16")
    # (7) braços erf e gelu: finitude + erf'≈Φ(βz) numérico
    for ativ in ("erf", "gelu", "relu"):
        _, _, la, _ = treina(m=16, ativ=ativ, passos=500, seed=3)
        assert np.isfinite(la)
    netE = MLP(2, "erf", beta=300.0, seed=0).double()
    zt = torch.linspace(-0.05, 0.05, 41, dtype=torch.float64, requires_grad=True)
    netE.act(zt).sum().backward()
    Phi = 0.5 * (1 + torch.erf(300.0 * zt.detach() / RAIZ2))
    assert float((zt.grad - Phi).abs().max()) < 1e-9
    print("[OK] braços erf/gelu/relu finitos; erf-smoothing tem g'(z)=Φ(βz) exato")
    print("== SMOKE OK — nenhuma estatística de predição foi computada ==")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke: smoke()
    else: print("run confirmatória só após pré-registro (ver predict.py/config)")
