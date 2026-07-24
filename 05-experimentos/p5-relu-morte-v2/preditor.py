"""Preditor A PRIORI do p5-relu-morte-v2 (modelo reduzido de corrida).

REGRA DURA: este script NAO roda a dinamica 4D do modelo. Ele so usa:
  (a) MC de passeio browniano PURO no cone de morte (exato p/ fase morta, por L1-v2);
  (b) o cronograma deterministico do residuo de B-sozinho (exato enquanto A morto);
  (c) o MODELO REDUZIDO 1D de excursao na coordenada normal da face de saida
      (OU rumo ao equilibrio movel x*(t) = -res_j(t)/||(x_j,1)||) — APROXIMACAO
      declarada, que ignora: back-reaction de A no residuo, migracao entre faces,
      geometria 2D das excursoes. E exatamente isso que P-1/P-2/P-3 testam.

Saida: preditor-output.json com grade 3x3 congelada, p_captura previsto por celula,
censura, curva captura-vs-tau1 (tau50, largura 10-90) p/ P-2, faixas de tau1 e
frequencias de re-morte p/ P-3. Constantes do modelo reduzido fixadas AQUI, a priori.
"""
import json

import numpy as np

# ---------------- objeto (identico ao verificacao-lemas-v2.py) ----------------
X = np.array([0.5, 1.5])
Y = np.array([0.5, 1.0])
N = len(X)
ETA = 1e-3               # eta_A: neuronio morto A (e ruido das coords de A)
ETA_B = 1e-4             # eta_B = eta_A/10: separa as escalas da corrida (Langevin
                         # pre-condicionado diagonal — mesma Gibbs e^{-L/T})
B_INIT = np.array([0.2, 0.05])
N_PASSOS = 500_000

NORMA = np.sqrt(X ** 2 + 1)            # ||(x_j,1)|| por face
RHO = ETA * NORMA ** 2 / N             # taxa OU discreta por face
DEPTH2APEX = np.sqrt(1 + X.max() ** 2)  # c = d0 * DEPTH2APEX para init (0,-c)

# ---------------- constantes do modelo reduzido (congeladas a priori) ----------
FRAC_CAPTURA = 0.5      # capturado quando x >= x*(t) * FRAC_CAPTURA ...
SIG_MIN_CAPTURA = 4.0   # ... e x*(t) >= SIG_MIN_CAPTURA * sigma_x (basin > ruido)
SIG_REDEATH = 2.0       # re-morte: voltou a x<0 depois de exceder SIG_REDEATH*sigma_x
W_P3 = 50_000           # re-morte de P-3 contada DENTRO desta janela pos-saida
W_LOCAL = 70_000        # horizonte local da excursao: cobre a janela de captura
                        # (fecha quando x*(t) < SIG_MIN_CAPTURA*sigma_x, t<=~16k
                        # em todas as T da grade) + W_P3; truncamento consistente
                        # com o proprio criterio de captura do modelo reduzido
RNG_CANONICO = 20260724

# ---------------- cronograma do residuo (exato, discreto) ----------------------
def cronograma_residuo(n_passos=N_PASSOS):
    pB = B_INIT.astype(float).copy()
    res = np.empty((n_passos, N))
    for t in range(n_passos):
        z = pB[0] * X + pB[1]
        r = np.maximum(z, 0) - Y
        res[t] = r
        pB -= ETA_B * np.array([np.mean(r * (z > 0) * X), np.mean(r * (z > 0))])
    return res

# ---------------- MC de saida do cone (BM puro; exato por L1-v2) ---------------
def mc_saida_cone(c, sigma, m, horizonte, seed, bloco=2000):
    """Passeio 2D com passo sigma/componente desde (0,-c); retorna (tau, face),
    tau=horizonte p/ censurados (face=-1)."""
    rng = np.random.default_rng(seed)
    pos = np.tile([0.0, -c], (m, 1))
    tau = np.full(m, horizonte, dtype=np.int64)
    face = np.full(m, -1, dtype=np.int64)
    vivos = np.arange(m)
    t = 0
    while len(vivos) and t < horizonte:
        k = min(bloco, horizonte - t)
        ruido = rng.standard_normal((len(vivos), k, 2)) * sigma
        caminho = pos[vivos, None, :] + np.cumsum(ruido, axis=1)
        z = caminho @ np.vstack([X, np.ones(N)])          # (m_vivos, k, N)
        saiu = (z > 0).any(axis=2)                        # (m_vivos, k)
        algum = saiu.any(axis=1)
        idx1 = np.argmax(saiu, axis=1)
        quem = np.nonzero(algum)[0]
        gl = vivos[quem]
        tau[gl] = t + idx1[quem] + 1
        face[gl] = np.argmax(z[quem, idx1[quem], :], axis=1)
        pos[vivos] = caminho[:, -1, :]
        vivos = vivos[~algum]
        t += k
    return tau, face

class SaidasCanonicas:
    """MC canonico (c=1, sigma=SIGMA_CAN) reescalado por difusao: fator (SIGMA_CAN*c/sigma)^2."""
    SIGMA_CAN = 0.02
    HORIZONTE_CAN = 80_000
    M_CAN = 30_000

    def __init__(self):
        self.tau, self.face = mc_saida_cone(1.0, self.SIGMA_CAN, self.M_CAN,
                                            self.HORIZONTE_CAN, RNG_CANONICO)

    def amostra(self, d0, T, m, seed):
        c = d0 * DEPTH2APEX
        sigma = np.sqrt(2 * ETA * T)
        fator = (self.SIGMA_CAN * c / sigma) ** 2
        if fator < 5.0:                                   # regime grosseiro: simular direto
            return mc_saida_cone(c, sigma, m, N_PASSOS, seed)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, self.M_CAN, m)
        tau = np.minimum((self.tau[idx] * fator).astype(np.int64), N_PASSOS)
        face = np.where((self.tau[idx] * fator) < N_PASSOS, self.face[idx], -1)
        cens_can = self.face[idx] == -1                   # censura ja no canonico
        face = np.where(cens_can, -1, face)
        tau = np.where(face == -1, N_PASSOS, tau)
        return tau, face

# ---------------- excursao reduzida 1D (OU com equilibrio movel) ---------------
def mc_excursao(tau, face, res, T, seed, bloco=2000, w_local=W_LOCAL):
    """Para cada (tau1, face j): x_{t+1} = x + [-rho_j (x - x*(t))]*1[x>0] + sigma*xi.
    Retorna: capturado (bool), re_mortes (int), t_captura (ou -1)."""
    rng = np.random.default_rng(seed)
    m = len(tau)
    sigma = np.sqrt(2 * ETA * T)
    sigx = np.sqrt(N * T) / NORMA                          # sd estacionaria por face
    x = np.zeros(m)
    cap = np.zeros(m, bool)
    t_cap = np.full(m, -1, dtype=np.int64)
    excedeu = np.zeros(m, bool)
    re_mortes = np.zeros(m, dtype=np.int64)
    ativo = (face >= 0)
    t_local = np.zeros(m, dtype=np.int64)
    while ativo.any():
        ids = np.nonzero(ativo)[0]
        k = bloco
        ruido = rng.standard_normal((len(ids), k)) * sigma
        for step in range(k):
            tg = np.minimum(tau[ids] + t_local[ids], N_PASSOS - 1)
            rj = res[tg, face[ids]]
            xstar = np.maximum(-rj, 0.0) / NORMA[face[ids]]
            drift = -RHO[face[ids]] * (x[ids] - xstar) * (x[ids] > 0)
            x[ids] = x[ids] + drift + ruido[:, step]
            lim = sigx[face[ids]]
            novo_exc = x[ids] > SIG_REDEATH * lim
            rmorte = (excedeu[ids] & (x[ids] < 0) & (t_local[ids] < W_P3)
                      & ~cap[ids])                     # regua identica ao experimento:
            re_mortes[ids] += rmorte                   # re-morte so conta pre-captura
            excedeu[ids] = (excedeu[ids] | novo_exc) & ~rmorte
            pega = (x[ids] >= FRAC_CAPTURA * xstar) & (xstar >= SIG_MIN_CAPTURA * lim)
            recem = pega & ~cap[ids]
            cap[ids] |= pega
            t_cap[ids[recem]] = tau[ids[recem]] + t_local[ids[recem]]
            t_local[ids] += 1
        fim = cap[ids] | (tau[ids] + t_local[ids] >= N_PASSOS) | (t_local[ids] >= w_local)
        ativo[ids[fim]] = False
    return cap, re_mortes, t_cap

# ---------------- avaliacao de uma condicao ------------------------------------
def avalia(cone, res, d0, T, m, seed, w_local=W_LOCAL):
    tau, face = cone.amostra(d0, T, m, seed)
    cap, rmt, _ = mc_excursao(tau, face, res, T, seed + 1, w_local=w_local)
    cens = float(np.mean(face == -1))
    p = float(np.mean(cap))
    return {"d0": d0, "T": T, "m": m, "p_captura": p,
            "se": float(np.sqrt(p * (1 - p) / m)), "censura": cens,
            "frac_face1": float(np.mean(face == 1)),
            "tau": tau, "face": face, "cap": cap, "re_mortes": rmt}

def curva_q_pooled(taus, caps, re_mortes, nb=30):
    """Tabela q(tau1) pooled (face 1 ja filtrada), faixas [0,t90],[t90,t10],[t10,inf)
    e frequencias de re-morte por faixa — o objeto congelado por T."""
    ordem = np.argsort(taus)
    tt, cc, rr = taus[ordem], caps[ordem].astype(float), (re_mortes[ordem] > 0)
    bordas_idx = np.array_split(np.arange(len(tt)), nb)
    q_edges = [float(tt[b[0]]) for b in bordas_idx] + [float(tt[-1]) + 1]
    q_vals = [float(cc[b].mean()) for b in bordas_idx]
    tq = np.array([tt[b].mean() for b in bordas_idx])
    pq = np.array(q_vals)
    def cruza(nivel):
        s = np.nonzero(pq <= nivel)[0]
        return float(tq[s[0]]) if len(s) else float(tq[-1])
    t90, t10 = cruza(0.9), cruza(0.1)
    faixas = [[0.0, t90], [t90, t10], [t10, float(N_PASSOS)]]
    freq_rm = []
    for a, b in faixas:
        m = (tt >= a) & (tt < b)
        freq_rm.append(float(rr[m].mean()) if m.sum() > 50 else None)
    return {"q_edges": q_edges, "q_vals": q_vals, "tau50": cruza(0.5),
            "tau90": t90, "tau10": t10, "faixas": faixas,
            "P3_freq_re_morte": freq_rm, "n_pooled": int(len(tt))}

# ---------------- self-tests ----------------------------------------------------
def self_tests(cone, res):
    print("== SELF-TESTS do preditor ==")
    d0, T = 0.15, 1e-3
    c, sigma = d0 * DEPTH2APEX, np.sqrt(2 * ETA * T)
    tau_d, face_d = mc_saida_cone(c, sigma, 6000, N_PASSOS, 99)
    tau_r, face_r = cone.amostra(d0, T, 6000, 98)
    md, mr = np.median(tau_d), np.median(tau_r)
    assert abs(md - mr) / md < 0.15, (md, mr)
    fd, fr = np.mean(face_d == 1), np.mean(face_r == 1)
    assert abs(fd - fr) < 0.05, (fd, fr)
    print(f"[OK] reescala canonica vs direta (d0={d0}, T={T}): mediana {md:.0f} vs {mr:.0f}"
          f" (<15%), frac face1 {fd:.3f} vs {fr:.3f} (<0.05)")
    res_forte = np.tile(res[0], (N_PASSOS, 1))            # residuo congelado no maximo
    cap_f, _, _ = mc_excursao(np.zeros(3000, np.int64), np.ones(3000, np.int64),
                              res_forte, 1e-3, 7)
    res_zero = np.zeros((N_PASSOS, N))
    cap_z, _, _ = mc_excursao(np.zeros(3000, np.int64), np.ones(3000, np.int64),
                              res_zero, 1e-3, 8)
    assert cap_f.mean() > 0.95, cap_f.mean()
    assert cap_z.mean() < 0.02, cap_z.mean()
    print(f"[OK] limites da excursao: residuo forte -> captura {cap_f.mean():.3f} (>0.95);"
          f" residuo zero -> {cap_z.mean():.3f} (<0.02)")

# ---------------- main ----------------------------------------------------------
if __name__ == "__main__":
    print("cronograma do residuo (exato)...")
    res = cronograma_residuo()
    print(f"  res(0)={res[0].round(4)}, cruzamento face0 em t="
          f"{int(np.argmax(res[:, 0] > 0))}, res(50k)={res[50_000].round(5)}")
    print("MC canonico do cone (uma vez, reescalado depois)...")
    cone = SaidasCanonicas()
    cens_can = np.mean(cone.face == -1)
    print(f"  M={cone.M_CAN}, censura canonica={cens_can:.3f}, "
          f"frac face1={np.mean(cone.face == 1):.3f}")
    self_tests(cone, res)

    print("== BUSCA DE GRADE (M=4000 por ponto) ==")
    ALVOS = [0.25, 0.5, 0.75]
    TS = [3e-4, 1e-3, 3e-3]
    grade = {}
    for T in TS:
        d0s = np.geomspace(0.02, 0.45, 10)
        ps = [avalia(cone, res, d0, T, 4000, 1000 + i, w_local=20_000)["p_captura"]
              for i, d0 in enumerate(d0s)]
        print(f"  T={T}: p(captura) por d0 = "
              f"{[f'{d:.3f}:{p:.2f}' for d, p in zip(d0s, ps)]}")
        celulas = []
        for alvo in ALVOS:                                # p decresce com d0
            j = int(np.argmin(np.abs(np.array(ps) - alvo)))
            celulas.append(float(d0s[j]))
        grade[T] = celulas
    print("== AVALIACAO FINAL (M=20000 por celula) ==")
    saida = {"constantes": {"FRAC_CAPTURA": FRAC_CAPTURA,
                            "SIG_MIN_CAPTURA": SIG_MIN_CAPTURA,
                            "SIG_REDEATH": SIG_REDEATH,
                            "ETA": ETA, "N_PASSOS": N_PASSOS,
                            "B_INIT": B_INIT.tolist(), "X": X.tolist(), "Y": Y.tolist()},
             "celulas": []}
    saida["pooled_por_T"] = {}
    for T in TS:
        pool_t, pool_c, pool_r = [], [], []
        for alvo, d0 in zip(ALVOS, grade[T]):
            cond = avalia(cone, res, d0, T, 20000, int(T * 1e7) + int(d0 * 1e4))
            cel = {k: cond[k] for k in
                   ("d0", "T", "p_captura", "se", "censura", "frac_face1")}
            cel["alvo"] = alvo
            saida["celulas"].append(cel)
            sel1 = cond["face"] == 1
            pool_t.append(cond["tau"][sel1])
            pool_c.append(cond["cap"][sel1])
            pool_r.append(cond["re_mortes"][sel1])
            print(f"  T={T} d0={d0:.3f} (alvo {alvo}): p={cel['p_captura']:.3f}"
                  f"±{cel['se']:.3f}, censura={cel['censura']:.3f}, "
                  f"face1={cel['frac_face1']:.2f}")
        # lema de pooling: q(tau1) nao depende de d0 no modelo reduzido -> poolar
        pooled = curva_q_pooled(np.concatenate(pool_t), np.concatenate(pool_c),
                                np.concatenate(pool_r))
        saida["pooled_por_T"][str(T)] = pooled
        print(f"  T={T} POOLED: n={pooled['n_pooled']}, tau50={pooled['tau50']:.0f}, "
              f"faixas={[[round(a), round(b)] for a, b in pooled['faixas']]}, "
              f"P3_freq={pooled['P3_freq_re_morte']}")
    with open("preditor-output.json", "w") as fh:
        json.dump(saida, fh, indent=1)
    print("congelado em preditor-output.json")
