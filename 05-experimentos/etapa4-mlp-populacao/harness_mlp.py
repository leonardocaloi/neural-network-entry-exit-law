"""Etapa 4 — harness de MLP com a régua populacional embarcada.

MLP raso: f(x) = sum_k v_k * act(w_k·x + b_k), treino SGD/SGLD, régua O(m) por
snapshot: fração ativa a_k, morto (todas as margens <=0), dado líder j*_k,
resíduo projetado R̂_k = v_k*[(y-f)_{j*} + v_k*act(z_k,j*)] (o sinal de v_k
entra — derivado da dinâmica da margem; v=1 recupera a régua calibrada no toy),
área acumulada Â_k por episódio morto, e rival t_sinal (1º R̂>0).

--smoke: mecânica apenas (finitude, determinismo, âncora analítica da fração
nascida-morta, caminho GPU) — NENHUMA estatística de predição.
"""
import argparse

import numpy as np
import torch


def dados_1dreg(n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, generator=g) * 2.0
    y = torch.sin(2.0 * x) + 0.3 * x
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
        if self.ativ == "gelu":
            return z * torch.distributions.Normal(0., 1.).cdf(self.beta ** 0.5 * z)
        raise ValueError(self.ativ)

    def forward(self, x):
        return self.act(self.margens(x)) @ self.v


class Regua:
    """O(m) por snapshot; mesma lógica calibrada no toy (calibracao_regua.py)."""
    def __init__(self, m, eta, T):
        self.m, self.eta = m, eta
        self.lim_fator = 2.0 * np.sqrt(max(T, 1e-7))
        self.morto = np.zeros(m, bool)
        self.A_hat = np.zeros(m)
        self.t_morte = np.full(m, -1); self.t_area = np.full(m, -1)
        self.t_sinal = np.full(m, -1); self.t_res = np.full(m, -1)
        self.episodios = []

    def observa(self, t, z, res, v, x):        # z:(n,m) res:(n,) v:(m,) x:(n,1)
        n = z.shape[0]
        zmax, jstar = z.max(0), z.argmax(0)    # líder por neurônio
        normas = np.sqrt(x[:, 0] ** 2 + 1.0)
        lim = self.lim_fator * np.sqrt(n) / normas[jstar] * 0  # debounce simples:
        vivo = zmax > self.lim_fator            # limiar 2*sqrt(T) absoluto (barato)
        act_j = np.maximum(z[jstar, np.arange(self.m)], 0.0)
        R_hat = v * (-res[jstar] + v * act_j)   # res = f-y ⟹ (y-f) = -res
        for k in range(self.m):
            if not vivo[k] and not self.morto[k]:
                self.morto[k] = True
                self.t_morte[k] = t; self.A_hat[k] = 0.0
                self.t_area[k] = -1; self.t_sinal[k] = -1
            elif self.morto[k]:
                self.A_hat[k] += R_hat[k] * self.eta
                if self.t_area[k] < 0 and self.A_hat[k] > 0: self.t_area[k] = t
                if self.t_sinal[k] < 0 and R_hat[k] > 0: self.t_sinal[k] = t
                if vivo[k]:
                    self.morto[k] = False
                    self.episodios.append(dict(k=k, t_morte=int(self.t_morte[k]),
                                               t_area=int(self.t_area[k]),
                                               t_sinal=int(self.t_sinal[k]),
                                               t_res=int(t)))


def treina(m=16, ativ="softplus", beta=300.0, eta=1e-3, T=1e-3, passos=2000,
           seed=0, device="cpu", regua_cada=5):
    x, y = dados_1dreg(seed=seed)
    x, y = x.to(device), y.to(device)
    net = MLP(m, ativ, beta, seed).to(device)
    g = torch.Generator(device=device).manual_seed(seed + 1)
    reg = Regua(m, eta, T)
    for t in range(passos):
        f = net(x)
        res = f - y[:, 0]
        loss = 0.5 * (res ** 2).mean()
        net.zero_grad(); loss.backward()
        with torch.no_grad():
            for p in net.parameters():
                ruido = torch.randn(p.shape, generator=g, device=device)
                p -= eta * p.grad - np.sqrt(2 * eta * T) * ruido
            if t % regua_cada == 0:
                reg.observa(t, net.margens(x).detach().cpu().numpy(),
                            res.detach().cpu().numpy(),
                            net.v.detach().cpu().numpy(),
                            x.cpu().numpy())
    return net, reg, float(loss.detach())


def smoke():
    print("== SMOKE etapa-4: mecânica apenas ==")
    # (1) âncora analítica: fração nascida-morta no init (MC do init, sem treino)
    torch.manual_seed(0)
    M = 20000
    w = torch.randn(M, 1); b = torch.randn(M)
    x, _ = dados_1dreg()
    z = x @ w.T + b                            # (n, M)
    frac_mc = float((z.max(0).values <= 0).float().mean())
    print(f"[OK] fração nascida-morta (init padrão, MC {M}): {frac_mc:.3f} "
          f"(âncora p/ o instrumento; rascunho previa ~metade p/ init simétrica)")
    # (2) treino curto + régua: finitude e episódios
    net, reg, loss = treina(passos=2000, seed=3)
    assert np.isfinite(loss)
    print(f"[OK] treino m=16 softplus 2000 passos: loss={loss:.4f}, "
          f"mortos agora={int(reg.morto.sum())}, episódios completos={len(reg.episodios)}")
    # (3) determinismo
    _, r2, l2 = treina(passos=2000, seed=3)
    assert abs(l2 - loss) < 1e-10 and int(r2.morto.sum()) == int(reg.morto.sum())
    print("[OK] determinismo (mesma seed ⟹ mesma loss e mesmos mortos)")
    # (4) caminho GPU
    if torch.cuda.is_available():
        _, rg, lg = treina(passos=500, seed=3, device="cuda")
        assert np.isfinite(lg)
        print(f"[OK] caminho GPU ({torch.cuda.get_device_name(0)}): loss={lg:.4f}")
    else:
        print("[--] CUDA indisponível — runs grandes ficarão em CPU")
    print("== SMOKE OK — nenhuma estatística de predição foi computada ==")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke: smoke()
    else: print("run confirmatória só após pré-registro (config a congelar)")
