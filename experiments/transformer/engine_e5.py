"""Motor de SCOPING da etapa-5: morte->ressurreicao-por-area de unidades da FFN-GELU
num TRANSFORMER minimo (1 camada: self-attn causal + FFN pos-wise). ARQUIVO NOVO.

Generaliza o instrumento GT-AREA de engine_4a3.py (MLP) para a FFN de um transformer.
Toda a analise abaixo e [ESPECULACAO] ate o piloto dizer o contrario.

== Arquitetura (minima, honesta) ==
  e_{n,s}   = x_{n,s}*w_val + P[s]                         embedding valor+posicao
  h_{n,s}   = e_{n,s} + MHSA_causal(e)_{n,s}               residual em torno da attn
  z_{n,s,k} = W1[k]·h_{n,s} + b1[k]                        pre-ativacao FFN (unidade k)
  a_{n,s,k} = act(z_{n,s,k})                               GELU (ou ReLU no controle)
  ffn_{n,s} = a_{n,s}·W2^T + b2                            saida FFN
  h2_{n,s}  = h_{n,s} + ffn_{n,s}                          residual em torno da FFN
  yhat_{n,s}= <w_out, h2_{n,s}> + b_out                    leitura escalar por posicao
  y_{n,s}   = rich_f(x_{n, s-1})   (s=0 -> auto)           alvo: routing + transformada rica

A FFN pos-wise mapeia o escalar atendido ~x_{n,s-1} por m_ffn ridges GELU: MESMA
estrutura do MLP com m unidades. rich_f = sin(2pi u)+0.3 sin(5pi u) (o alvo rico do
etapa-4). PRESSAO = m_ffn pequeno vs riqueza de rich_f -> plato -> mortes.

== Peso de saida efetivo (a chave da generalizacao) ==
A contribuicao ESCALAR da unidade k a predicao em (n,s) e c_k * a_{n,s,k}, com
    c_k := <w_out, W2[:,k]>
(a FFN e pos-wise e a cabeca e linear => a_{n,s,k} so afeta yhat_{n,s}). c_k faz o
papel EXATO de v_k do MLP. Lider = (n*,s*) = argmax_{n,s} z_{n,s,k}.

== Duas sutilezas estruturais vs MLP (descobertas do scoping) ==
(S1) SINAL DA INCLINACAO. softplus tem g'(z)=sigma(bz)>0 sempre; GELU tem g'(z)<0 na
     regiao morta (g' troca de sinal em z~=-0.752). O drive real sobre a margem-lider
     e proporcional a -res_lider*c_k*g'(z_lider); portar a area do MLP (sem g') INVERTE
     o sinal na regiao morta. Instrumento primario = area COM inclinacao (D_k abaixo);
     mantemos a area-porte-MLP (R_mlp) so como diagnostico.
(S2) ENTRADA NAO-ESTACIONARIA. no MLP a entrada x e fixa => unidade ReLU morta tem
     grad==0 => congelada => morte TERMINAL (teorema). no transformer h e saida da attn
     TREINAVEL => mesmo unidade congelada (grad==0 nos pesos proprios) pode ressuscitar
     porque h se move (drive de MONTANTE). Logo o controle ReLU NAO e garantido silencioso;
     medimos o deslocamento dos pesos proprios no episodio p/ atribuir self-canal vs montante.

== Contador-mestre ground-truth (por unidade FFN k, sobre TODOS (n,s)) ==
  zmax_k = max_{n,s} z_{n,s,k};  morte-GT = zmax_k < z_dead;  volta-GT = zmax_k > z_alive.
  GELU: (z_dead,z_alive)=(-theta,0), theta=3 (cauda: |GELU(-3)|,|GELU'(-3)| < 0.012).
  ReLU: (0,0) (morte exata do MLP). Histerese z_dead<z_alive => episodios limpos.

== Instrumento GT-AREA (todos os sinais no relogio GT) ==
  durante o episodio morto de k acumula por passo:
    R_mlp_k = c_k*(-res_lider + c_k*a_lider)                    (porte direto do MLP)
    D_k     = c_k*(-res_lider)*gprime(z_lider)                  (drive faithful c/ inclinacao)
    A_mlp  += R_mlp_k*eta ;  A_drive += D_k*eta
  t_area_mlp/t_area_drive = 1o passo com area>0; z_park = min zmax; zmax_hi = max zmax pos-morte.
"""
import argparse
import math
import numpy as np
import torch

torch.use_deterministic_algorithms(True)
RAIZ2 = float(np.sqrt(2.0))
SQRT2PI = float(np.sqrt(2.0 * np.pi))


def _erf_arr(z):
    return torch.erf(torch.as_tensor(z, dtype=torch.float64) / RAIZ2).numpy()


def gelu_prime_np(z):
    """GELU'(z) = Phi(z) + z*phi(z)."""
    z = np.asarray(z, dtype=np.float64)
    Phi = 0.5 * (1.0 + _erf_arr(z))
    phi = np.exp(-0.5 * z * z) / SQRT2PI
    return Phi + z * phi


def rich_f_torch(u):
    """Alvo rico do etapa-4: sin(2pi u) + 0.3 sin(5pi u)."""
    return torch.sin(2.0 * np.pi * u) + 0.3 * torch.sin(5.0 * np.pi * u)


def dados_seq(N=16, S=4, seed=0):
    """x_{n,s} ~ U[0,2]. alvo y_{n,s} = rich_f(x_{n, s-1}); s=0 atende a si (x_{n,0}).
    Routing (attn: copiar token anterior) + transformada rica (FFN)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(N, S, generator=g) * 2.0
    x_prev = x.clone()
    x_prev[:, 1:] = x[:, :-1]          # posicao s recebe o valor de s-1; s=0 fica consigo
    y = rich_f_torch(x_prev)
    return x, y


class TinyTransformer(torch.nn.Module):
    """1 camada: MHSA causal (residual) + FFN pos-wise (residual) + cabeca escalar.
    Parametros explicitos (sem nn.Linear) p/ extracao read-only do instrumento."""

    def __init__(self, d=16, H=2, m_ffn=4, S=4, ativ="gelu", seed=0,
                 s_attn=None, s_ffn=None, s_head=None):
        super().__init__()
        assert d % H == 0
        self.d, self.H, self.m_ffn, self.S, self.ativ = d, H, m_ffn, S, ativ
        self.dh = d // H
        g = torch.Generator().manual_seed(seed)

        def randn(*shp, scale=1.0):
            return torch.randn(*shp, generator=g) * scale
        sa = (1.0 / np.sqrt(d)) if s_attn is None else s_attn
        sf = (1.0 / np.sqrt(d)) if s_ffn is None else s_ffn
        sh = (1.0 / np.sqrt(d)) if s_head is None else s_head
        # embedding valor + posicao
        self.w_val = torch.nn.Parameter(randn(d))
        self.P = torch.nn.Parameter(randn(S, d, scale=0.1))
        # attention (single matrizes d x d; heads por reshape)
        self.WQ = torch.nn.Parameter(randn(d, d, scale=sa))
        self.WK = torch.nn.Parameter(randn(d, d, scale=sa))
        self.WV = torch.nn.Parameter(randn(d, d, scale=sa))
        self.WO = torch.nn.Parameter(randn(d, d, scale=sa))
        # FFN
        self.W1 = torch.nn.Parameter(randn(m_ffn, d, scale=1.0 / np.sqrt(d)))
        self.b1 = torch.nn.Parameter(randn(m_ffn))
        self.W2 = torch.nn.Parameter(randn(d, m_ffn, scale=sf))
        self.b2 = torch.nn.Parameter(torch.zeros(d))
        # cabeca escalar
        self.w_out = torch.nn.Parameter(randn(d, scale=sh))
        self.b_out = torch.nn.Parameter(torch.zeros(1))
        mask = torch.triu(torch.ones(S, S), diagonal=1).bool()   # causal: proibe s'>s
        self.register_buffer("causal", mask)

    def act(self, z):
        if self.ativ == "gelu":
            return z * 0.5 * (1.0 + torch.erf(z / RAIZ2))
        if self.ativ == "relu":
            return torch.relu(z)
        raise ValueError(self.ativ)

    def _attn(self, e):
        N, S, d = e.shape
        Q = (e @ self.WQ).view(N, S, self.H, self.dh).transpose(1, 2)   # N,H,S,dh
        K = (e @ self.WK).view(N, S, self.H, self.dh).transpose(1, 2)
        V = (e @ self.WV).view(N, S, self.H, self.dh).transpose(1, 2)
        scores = (Q @ K.transpose(-1, -2)) / np.sqrt(self.dh)           # N,H,S,S
        scores = scores.masked_fill(self.causal[None, None], float("-inf"))
        alpha = torch.softmax(scores, dim=-1)
        ctx = (alpha @ V).transpose(1, 2).reshape(N, S, d)             # N,S,d
        return ctx @ self.WO

    def h_pre_ffn(self, x):
        """h = e + attn(e). x:(N,S) escalares."""
        e = x[..., None] * self.w_val + self.P[None]
        return e + self._attn(e)

    def forward_parts(self, x):
        """Retorna yhat:(N,S), z:(N,S,m), a:(N,S,m), h:(N,S,d). h2=h+ffn; yhat=<w_out,h2>+b."""
        h = self.h_pre_ffn(x)
        z = h @ self.W1.T + self.b1
        a = self.act(z)
        ffn = a @ self.W2.T + self.b2
        h2 = h + ffn
        yhat = h2 @ self.w_out + self.b_out
        return yhat, z, a, h

    def forward(self, x):
        return self.forward_parts(x)[0]

    def c_eff(self):
        """c_k = <w_out, W2[:,k]> = peso de saida escalar efetivo da unidade k. (m_ffn,)"""
        return (self.w_out @ self.W2).detach().cpu().numpy()


def treina_e5(m_ffn=4, ativ="gelu", eta=1e-3, passos=150_000, seed=0, device="cpu",
              N=16, S=4, d=16, H=2, z_dead=None, z_alive=0.0, rec_cada=200,
              freeze_attn=False, s_attn=None, s_ffn=None, s_head=None,
              n_kill=0, kill_depth=3.5):
    """Treino full-batch T=0 (GD puro determinista) + contador-mestre GT + GT-AREA.
    z_dead default: GELU -> -3.0 ; ReLU -> 0.0. freeze_attn: congela embedding+attn+cabeca
    (so a FFN treina) => reduz ao cenario MLP (entrada da FFN fixa) p/ isolar o self-canal.
    n_kill: injeta n_kill unidades NASCIDAS-MORTAS (rebaixa b1 das n_kill unidades de menor
    zmax inicial ate zmax=-kill_depth) — alavanca EXPLORATORIA p/ garantir episodios mortos e
    exercitar a ressurreicao/instrumento (documentado no relatorio; NAO e o regime natural)."""
    if z_dead is None:
        z_dead = -3.0 if ativ == "gelu" else 0.0
    x, y = dados_seq(N=N, S=S, seed=seed)
    x, y = x.to(device), y.to(device)
    net = TinyTransformer(d=d, H=H, m_ffn=m_ffn, S=S, ativ=ativ, seed=seed,
                          s_attn=s_attn, s_ffn=s_ffn, s_head=s_head).to(device)
    m = m_ffn
    idx = np.arange(m)
    # injecao de nascidas-mortas (rebaixa b1 das n_kill unidades de menor zmax inicial)
    if n_kill > 0:
        with torch.no_grad():
            z0 = (net.h_pre_ffn(x) @ net.W1.T + net.b1).detach().cpu().numpy().reshape(N * S, m)
            zmax0 = z0.max(0)
            alvo = np.argsort(zmax0)[:n_kill]
            for k in alvo:
                net.b1[int(k)] += float(-kill_depth - zmax0[k])

    # quais parametros treinam. freeze_attn congela SO o lado de entrada
    # (embedding + attn) => h = e + attn(e) fica ESTACIONARIO (entrada da FFN fixa) =>
    # reduz ao cenario MLP; FFN e cabeca continuam treinando (como v_k no MLP).
    frozen_names = {"w_val", "P", "WQ", "WK", "WV", "WO"}
    params_treina = [p for n_, p in net.named_parameters()
                     if (not freeze_attn) or n_ not in frozen_names]

    loss_hist = []
    # contador-mestre GT (por unidade FFN) --------------------------------------
    esteve_morto = np.zeros(m, bool)
    morto_prev = np.zeros(m, bool)
    nasce_morto = None
    death_events = []            # (t,k) vivo->morto, t>0 = DINAMICA
    revive_events = []           # (t,k) morto->vivo (zmax>z_alive)
    n_revive = 0
    revive_neuronios = set()
    # instrumento GT-AREA: estado do episodio corrente por unidade ---------------
    A_mlp = np.zeros(m)
    A_drive = np.zeros(m)
    t_area_mlp = np.full(m, -1)
    t_area_drive = np.full(m, -1)
    t_morte_ep = np.full(m, -1)
    zpark_ep = np.full(m, np.inf)
    zhi_ep = np.full(m, -np.inf)     # high-water mark de zmax pos-morte
    classe_ep = [None] * m
    W1_morte = np.zeros((m, d))       # snapshot dos pesos proprios no inicio do episodio
    b1_morte = np.zeros(m)
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
        episodios.append(dict(
            k=int(k), classe=classe_ep[k], t_morte=int(t_morte_ep[k]),
            t_area_mlp=int(t_area_mlp[k]), t_area_drive=int(t_area_drive[k]),
            t_res=(int(t_res) if t_res is not None else None), revived=bool(revived),
            z_park=zp, zmax_hi=float(zhi_ep[k]),
            gprime_park=float(gelu_prime_np(zp)) if ativ == "gelu" else 0.0,
            A_mlp_fim=float(A_mlp[k]), A_drive_fim=float(A_drive[k]),
            own_move=own))
        t_morte_ep[k] = -1

    loss = None
    for t in range(passos):
        yhat, z, a, h = net.forward_parts(x)
        res = yhat - y
        loss = 0.5 * (res ** 2).mean()
        # zmax/jstar por unidade (torch, barato) — detecta morte/volta ---------
        z_det = z.detach().reshape(N * S, m)
        zmv, jsv = z_det.max(0)
        zmax = zmv.cpu().numpy(); jstar = jsv.cpu().numpy()
        morto = zmax < z_dead
        esteve_morto |= morto
        # 1) aberturas / mortes ------------------------------------------------
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
        # 2) acumulacao GT-AREA p/ episodios abertos (extracao cara SO se aberto)
        aberto = t_morte_ep >= 0
        if aberto.any():
            a_np = a.detach().reshape(N * S, m).cpu().numpy()
            res_np = res.detach().reshape(N * S).cpu().numpy()
            c_k = (net.w_out @ net.W2).detach().cpu().numpy()
            res_lider = res_np[jstar]; a_lider = a_np[jstar, idx]
            gprime_lider = gelu_prime_np(zmax) if ativ == "gelu" else np.ones(m)
            R_mlp = c_k * (-res_lider + c_k * a_lider)             # porte MLP (sem g')
            D_k = c_k * (-res_lider) * gprime_lider                 # drive faithful (com g')
            A_mlp[aberto] += R_mlp[aberto] * eta
            A_drive[aberto] += D_k[aberto] * eta
            cm = aberto & (t_area_mlp < 0) & (A_mlp > 0.0)
            cd = aberto & (t_area_drive < 0) & (A_drive > 0.0)
            t_area_mlp[cm] = t; t_area_drive[cd] = t
            zpark_ep[aberto] = np.minimum(zpark_ep[aberto], zmax[aberto])
            zhi_ep[aberto] = np.maximum(zhi_ep[aberto], zmax[aberto])
        # 3) voltas GT (zmax > z_alive) ----------------------------------------
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
        # update — GD puro T=0 (dinamica intocada pelo instrumento) -------------
        net.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in params_treina:
                if p.grad is not None:
                    p += -eta * p.grad
    loss_hist.append(float(loss.detach()))
    # episodios ainda abertos no fim = CENSURADOS
    W1_np, b1_np = _snap_own()
    for k in np.where(t_morte_ep >= 0)[0]:
        _fecha(k, None, False, W1_np, b1_np)

    metr = dict(
        loss_hist=loss_hist, plato=float(np.median(loss_hist[-max(3, len(loss_hist)//10):])),
        loss_fim=float(loss_hist[-1]),
        nasce_morto=[int(k) for k in np.where(nasce_morto)[0]],
        n_nasce_morto=int(nasce_morto.sum()),
        death_events=death_events, revive_events=revive_events,
        n_mortos=int(esteve_morto.sum()), n_revive=int(n_revive),
        n_neuronios_revive=len(revive_neuronios),
        episodios=episodios,
        cfg=dict(m_ffn=m_ffn, ativ=ativ, eta=eta, passos=passos, seed=seed,
                 N=N, S=S, d=d, H=H, z_dead=z_dead, z_alive=z_alive,
                 freeze_attn=freeze_attn, n_kill=n_kill, kill_depth=kill_depth))
    return metr


# ============================ TESTES SINTETICOS ============================
def _testes():
    print("== TESTES do instrumento GT-AREA generalizado (transformer FFN) ==")
    torch.set_num_threads(1)
    # (1) determinismo: mesma seed => mesma loss
    m1 = treina_e5(m_ffn=6, ativ="gelu", passos=300, seed=3)
    m2 = treina_e5(m_ffn=6, ativ="gelu", passos=300, seed=3)
    assert abs(m1["loss_fim"] - m2["loss_fim"]) < 1e-12, (m1["loss_fim"], m2["loss_fim"])
    print(f"[OK] determinismo CPU: loss_fim={m1['loss_fim']:.6f} identico")

    # (2) gradiente autograd vs diferencas finitas (dtype float64)
    net = TinyTransformer(d=8, H=2, m_ffn=4, S=3, ativ="gelu", seed=7).double()
    x, y = dados_seq(N=5, S=3, seed=7); x, y = x.double(), y.double()
    loss = 0.5 * ((net(x) - y) ** 2).mean()
    loss.backward()
    eps = 1e-6
    for nome, p in [("W1", net.W1), ("b1", net.b1), ("W2", net.W2),
                    ("w_out", net.w_out), ("WQ", net.WQ)]:
        flat = p.detach().reshape(-1); i = min(1, flat.numel() - 1)
        orig = float(flat[i])
        with torch.no_grad():
            flat[i] = orig + eps; lp = float(0.5 * ((net(x) - y) ** 2).mean())
            flat[i] = orig - eps; lm = float(0.5 * ((net(x) - y) ** 2).mean())
            flat[i] = orig
        fd = (lp - lm) / (2 * eps); an = float(p.grad.reshape(-1)[i])
        assert abs(fd - an) < 1e-6 * max(1.0, abs(an)), (nome, fd, an)
    print("[OK] gradiente = diferencas finitas (W1,b1,W2,w_out,WQ; float64)")

    # (3) c_k = <w_out,W2[:,k]> reproduz d yhat / d a_k (perturbacao de a_k)
    net = TinyTransformer(d=8, H=2, m_ffn=5, S=3, ativ="gelu", seed=1).double()
    x, y = dados_seq(N=4, S=3, seed=1); x = x.double()
    c_k = net.c_eff()
    h = net.h_pre_ffn(x); z = h @ net.W1.T + net.b1; a = net.act(z)
    a2 = a.clone(); k = 2; da = 1e-4; a2[..., k] = a[..., k] + da
    y0 = (h + a @ net.W2.T + net.b2) @ net.w_out + net.b_out
    y1 = (h + a2 @ net.W2.T + net.b2) @ net.w_out + net.b_out
    num = ((y1 - y0) / da).detach().cpu().numpy()
    assert np.allclose(num, c_k[k], atol=1e-8), (num.ravel()[:3], c_k[k])
    print(f"[OK] c_k reproduz d yhat/d a_k (unidade {k}): num~{num.ravel()[0]:.4f} c_k={c_k[k]:.4f}")

    # (4) lider = argmax_{n,s} z; res_lider e a_lider batem com a extracao
    N, S, m = 4, 3, 5
    net = TinyTransformer(d=8, H=2, m_ffn=m, S=S, ativ="gelu", seed=2)
    x, y = dados_seq(N=N, S=S, seed=2)
    yhat, z, a, _ = net.forward_parts(x)
    z_np = z.detach().numpy().reshape(N * S, m); jstar = z_np.argmax(0)
    for k in range(m):
        ns = jstar[k]; nn, ss = ns // S, ns % S
        assert abs(z_np[ns, k] - float(z[nn, ss, k])) < 1e-6
        assert z_np[ns, k] >= z_np[:, k].max() - 1e-6
    print("[OK] lider (n*,s*)=argmax_{n,s} z identificado corretamente")

    # (5) SINAL (S1): na regiao morta GELU (g'<0) o drive faithful D_k tem sinal
    #     OPOSTO ao porte-MLP R_mlp. Construimos res_lider, c_k, z_lider e conferimos.
    z_lider = -2.0                      # regiao morta: g'(-2) < 0
    gp = float(gelu_prime_np(z_lider)); assert gp < 0
    c, res_l, a_l = 0.7, -0.5, float(0.5 * (1 + math.erf(z_lider / RAIZ2)) * z_lider)
    R_mlp = c * (-res_l + c * a_l)
    D = c * (-res_l) * gp
    assert np.sign(R_mlp) == -np.sign(D), (R_mlp, D)
    print(f"[OK] S1: regiao morta GELU g'({z_lider})={gp:.3f}<0 => sign(R_mlp)={np.sign(R_mlp):+.0f} "
          f"OPOSTO a sign(D_drive)={np.sign(D):+.0f} (area-porte-MLP inverte o sinal)")

    # (6) drive faithful prediz o sinal do auto-movimento da margem-lider.
    #     Attn CONGELADA (h fixo) => so a FFN move z_lider; comparamos sign(D_k) com
    #     sign(delta zmax) de um passo, na media sobre unidades mortas.
    net = TinyTransformer(d=12, H=2, m_ffn=8, S=4, ativ="gelu", seed=5)
    x, y = dados_seq(N=8, S=4, seed=5)
    with torch.no_grad():                      # empurra alguns b1 p/ regiao morta
        net.b1 -= 4.0
    concordam = tot = 0
    for _ in range(30):
        yhat, z, a, h = net.forward_parts(x)
        res = yhat - y; loss = 0.5 * (res ** 2).mean()
        N_, S_ = x.shape; m_ = net.m_ffn
        z_np = z.detach().numpy().reshape(N_ * S_, m_); a_np = a.detach().numpy().reshape(N_ * S_, m_)
        res_np = res.detach().numpy().reshape(N_ * S_); c_k = net.c_eff()
        zmax = z_np.max(0); jstar = z_np.argmax(0)
        D = c_k * (-res_np[jstar]) * gelu_prime_np(zmax)
        mortos = zmax < -3.0
        net.zero_grad(); loss.backward()
        with torch.no_grad():
            # so FFN (freeze attn/embedding/head) => entrada h fixa
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
    print(f"[OK] S2/drive: sign(D_k)==sign(delta zmax) em {concordam}/{tot} ({frac:.2f}) "
          f"passos-unidade mortos (attn congelada => self-canal isolado)")

    # (7) ReLU com entrada FIXA (attn congelada) => unidade morta tem grad proprio==0
    net = TinyTransformer(d=12, H=2, m_ffn=8, S=4, ativ="relu", seed=9)
    x, y = dados_seq(N=8, S=4, seed=9)
    with torch.no_grad():
        net.b1 -= 6.0                          # mata (zmax<0)
    yhat, z, a, h = net.forward_parts(x)
    zmax = z.detach().numpy().reshape(-1, net.m_ffn).max(0)
    mortos = np.where(zmax < 0.0)[0]; assert len(mortos) > 0
    loss = 0.5 * ((yhat - y) ** 2).mean(); net.zero_grad(); loss.backward()
    gW1 = net.W1.grad.detach().numpy(); gb1 = net.b1.grad.detach().numpy()
    assert np.allclose(gW1[mortos], 0.0) and np.allclose(gb1[mortos], 0.0)
    print(f"[OK] ReLU morto (entrada fixa): grad proprio==0 em {len(mortos)} unidades "
          f"(teorema MLP; no transformer h treina => nao terminal)")

    # (8) contador-mestre: born-dead e episodios consistentes com os eventos
    mt = treina_e5(m_ffn=6, ativ="gelu", passos=800, seed=4, z_dead=-1.0)
    n_abertos = mt["n_nasce_morto"] + len(mt["death_events"])
    assert len(mt["episodios"]) == n_abertos, (len(mt["episodios"]), n_abertos)
    n_rev_ep = sum(1 for e in mt["episodios"] if e["revived"])
    assert n_rev_ep == mt["n_revive"], (n_rev_ep, mt["n_revive"])
    for e in mt["episodios"]:                  # trio bem-formado no relogio GT
        if e["t_area_drive"] >= 0:
            assert e["t_morte"] <= e["t_area_drive"]
            if e["t_res"] is not None:
                assert e["t_area_drive"] < e["t_res"]
    print(f"[OK] contador-mestre: {n_abertos} episodios (born={mt['n_nasce_morto']} "
          f"dyn={len(mt['death_events'])}), revive={mt['n_revive']}, trios bem-formados")
    print("== TODOS OS TESTES PASSARAM ==")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--testes", action="store_true")
    a = ap.parse_args()
    if a.testes:
        _testes()
    else:
        print("uso: python engine_e5.py --testes")
