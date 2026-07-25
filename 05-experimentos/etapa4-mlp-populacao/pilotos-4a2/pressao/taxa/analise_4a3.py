"""Análise do CONFIRMATÓRIO etapa-4a.3. Lê config_4a3.json (números congelados
pelo preditor, todos derivados do piloto-de-taxa) + results/ (episodios_gt do
instrumento GT-ÁREA, relógio único ground-truth). Nenhuma régua v2 aqui.

Estimandos pré-registrados (config manda; nada aqui é decidido a posteriori):
  C1 (existência-com-taxa, primário β3): p(volta | morte dinâmica ELEGÍVEL),
      Clopper-Pearson. Elegível = t_morte ≤ H−τ. MATA: N_elig ≥ n* e CP-superior
      < metade do CI-inf do piloto; N_elig < n* ⟹ INCONCLUSIVO (nunca negativo).
  C2 (lei-de-área, o CORAÇÃO, primário β3): fração de voltas dinâmicas ORDENADAS
      no relógio GT — t_morte ≤ t_área < t_res (empate morte-área permitido;
      área-volta estrito; volta sem área = NÃO-ordenada). MATA: CP-inferior < 0.5
      com ≥6 voltas; <6 voltas ⟹ INCONCLUSIVO.
  C4 (canal-β): escada p(volta|morte) por β, com CIs, SEM MATA.
  VALIDADE: qualquer volta GT no relu ⟹ EXPERIMENTO INVÁLIDO (teorema grad≡0).

CIs agregados por BOOTSTRAP DE SEED (clustering; eventos concentram em poucas
seeds). Binomial (Clopper-Pearson) e Poisson (Garwood) EXATOS, SEM scipy —
por bisseção na cauda exata. --sintetico valida a lógica em 5 mundos com asserts
nos dois sentidos ANTES de qualquer run.

Uso:
  ../../../../../venv/bin/python analise_4a3.py --sintetico   # obrigatório: passa TUDO
  ../../../../../venv/bin/python analise_4a3.py               # roda sobre results/
"""
import glob
import gzip
import json
import math
import os

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(AQUI, "config_4a3.json")


# ============================ estatística EXATA (sem scipy) ===================
def _bisect(f, target, incr, lo=0.0, hi=1.0, it=200):
    """p em [lo,hi] com f(p)=target; f monótona (incr=True crescente)."""
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        val = f(mid)
        if (val < target) == incr:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _binom_sf_ge(k, n, p):
    """P(X>=k), X~Bin(n,p), soma exata."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def _binom_cdf_le(k, n, p):
    """P(X<=k), X~Bin(n,p), soma exata."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def clopper_pearson(k, n, alpha=0.05):
    """CI exato de Clopper-Pearson para k/n, por bisseção na cauda binomial exata.
    (Idêntico a beta.ppf(a/2,k,n-k+1) / beta.ppf(1-a/2,k+1,n-k), sem scipy.)"""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    a = alpha / 2.0
    lo = 0.0 if k == 0 else _bisect(lambda p: _binom_sf_ge(k, n, p), a, incr=True)
    hi = 1.0 if k == n else _bisect(lambda p: _binom_cdf_le(k, n, p), a, incr=False)
    return (k / n, float(lo), float(hi))


def _pois_cdf_le(k, mu):
    """P(X<=k), X~Pois(mu), soma exata iterativa (sem overflow de fatorial)."""
    if k < 0:
        return 0.0
    term = math.exp(-mu)
    s = term
    for i in range(1, k + 1):
        term *= mu / i
        s += term
    return s


def _pois_sf_ge(k, mu):
    """P(X>=k) = 1 - P(X<=k-1)."""
    return 1.0 - _pois_cdf_le(k - 1, mu)


def _bisect_mu(f, target, incr, hi, it=200):
    lo = 0.0
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        val = f(mid)
        if (val < target) == incr:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def poisson_ci(k, exposure, alpha=0.05):
    """CI exato (Garwood) para taxa Poisson k/exposure, sem scipy."""
    a = alpha / 2.0
    hi_mu = k + 12.0 * math.sqrt(k + 1.0) + 30.0
    mu_lo = 0.0 if k == 0 else _bisect_mu(lambda mu: _pois_sf_ge(k, mu), a, True, hi_mu)
    mu_hi = _bisect_mu(lambda mu: _pois_cdf_le(k, mu), a, False, hi_mu)
    e = max(exposure, 1e-12)
    return (k / e, mu_lo / e, mu_hi / e)


def poisson_lower_quantile(mu, q=0.05):
    """Menor inteiro x com P(X<=x) >= q, X~Pois(mu). (n* = gate de INCONCLUSIVO.)"""
    term = math.exp(-mu)
    cdf = term
    x = 0
    while cdf < q:
        x += 1
        term *= mu / x
        cdf += term
    return x


# ------------------------- bootstrap de seed (clustering) --------------------
def seed_bootstrap_rate(por_seed, B=20000, alpha=0.05, seed=0):
    """CI de taxa por CLUSTER DE SEED. por_seed: lista (n_runs_seed, n_mortes_seed)."""
    if not por_seed:
        return (float("nan"), float("nan"), float("nan"))
    runs = np.array([r for r, _ in por_seed], float)
    mortes = np.array([d for _, d in por_seed], float)
    rate = mortes.sum() / max(runs.sum(), 1e-12)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(por_seed), size=(B, len(por_seed)))
    est = mortes[idx].sum(1) / np.maximum(runs[idx].sum(1), 1e-12)
    lo, hi = np.percentile(est, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(rate), float(lo), float(hi))


def seed_bootstrap_prop(por_seed_kn, B=20000, alpha=0.05, seed=1):
    """CI de proporção por cluster de seed. por_seed_kn: lista (k_seed, n_seed)."""
    kn = [(k, n) for (k, n) in por_seed_kn]
    if not kn:
        return (float("nan"), float("nan"), float("nan"))
    ks = np.array([k for k, _ in kn], float)
    ns = np.array([n for _, n in kn], float)
    if ns.sum() == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = ks.sum() / ns.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(kn), size=(B, len(kn)))
    bn = ns[idx].sum(1)
    est = np.where(bn > 0, ks[idx].sum(1) / np.maximum(bn, 1e-12), np.nan)
    lo, hi = np.nanpercentile(est, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(p), float(lo), float(hi))


# ============================ classificação (GT-ÁREA) =========================
def classifica_run(run, window_fim):
    """Do episodios_gt (relógio único GT) de UM run devolve contagens + detalhes.
    dyn = morte dinâmica (t_morte>0); volta = revived; elegível = t_morte<=window.
    ordenado (C2) = t_area>=0 e t_morte <= t_area < t_res (empate morte-área OK)."""
    eps = run["episodios_gt"]
    dyn = [e for e in eps if e["classe"] == "dyn"]
    born = [e for e in eps if e["classe"] == "born"]
    voltas, terminais, censurados = [], [], []
    n_area_abriu = 0
    n_voltas_area = 0
    n_dyn_elig = 0
    n_voltas_elig = 0
    trios = []
    ret_times = []
    for e in dyn:
        area_abriu = e["t_area"] >= 0
        elegivel = e["t_morte"] <= window_fim
        if area_abriu:
            n_area_abriu += 1
        if elegivel:
            n_dyn_elig += 1
        if e["revived"]:
            voltas.append(e)
            if area_abriu:
                n_voltas_area += 1
            if elegivel:
                n_voltas_elig += 1
            tm, ta, tr = e["t_morte"], e["t_area"], e["t_res"]
            ordenado = (ta >= 0) and (tm <= ta < tr)
            ret_times.append(int(tr - tm))
            trios.append(dict(k=e["k"], t_morte=tm, t_area=ta, t_res=tr,
                              ordenado=bool(ordenado), area_abriu=bool(area_abriu),
                              z_park=e.get("z_park"), sigma_park=e.get("sigma_park")))
        else:
            (censurados if area_abriu else terminais).append(e)
    born_rev = [e for e in born if e["revived"]]
    n_ordenados = sum(1 for tr in trios if tr["ordenado"])
    return dict(
        rot=run["rot"], m=run["m"], seed=run["seed"], beta=run.get("beta"),
        n_born_dead=len(born), n_born_rev=len(born_rev),
        n_dyn=len(dyn), n_voltas=len(voltas),
        n_terminais=len(terminais), n_censurados=len(censurados),
        n_area_abriu=n_area_abriu, n_voltas_area=n_voltas_area,
        n_dyn_elig=n_dyn_elig, n_voltas_elig=n_voltas_elig,
        n_ordenados=n_ordenados, n_revive_real=run.get("n_revive_real",
                                                        len(voltas) + len(born_rev)),
        ret_times=ret_times, trios=trios,
    )


def agrega(cls_list, chave, boot_rate_seed=0, boot_prop_seed=1):
    """Agrega classificações por chave(cl)->grupo com CIs por bootstrap de seed."""
    grupos = {}
    for cl in cls_list:
        grupos.setdefault(chave(cl), []).append(cl)
    tabela = {}
    for g, items in grupos.items():
        por_seed = {}
        for cl in items:
            d = por_seed.setdefault(cl["seed"], dict(runs=0, dyn=0, voltas=0,
                                                     dyn_elig=0, voltas_elig=0,
                                                     ordenados=0))
            d["runs"] += 1
            d["dyn"] += cl["n_dyn"]
            d["voltas"] += cl["n_voltas"]
            d["dyn_elig"] += cl["n_dyn_elig"]
            d["voltas_elig"] += cl["n_voltas_elig"]
            d["ordenados"] += cl["n_ordenados"]
        S = lambda key: sum(cl[key] for cl in items)  # noqa: E731
        n_runs = len(items)
        n_dyn = S("n_dyn"); n_voltas = S("n_voltas")
        n_dyn_elig = S("n_dyn_elig"); n_voltas_elig = S("n_voltas_elig")
        n_ord = S("n_ordenados")
        tabela[g] = dict(
            n_runs=n_runs, n_seeds=len(por_seed),
            n_born_dead=S("n_born_dead"), n_born_rev=S("n_born_rev"),
            n_dyn=n_dyn, n_voltas=n_voltas,
            n_terminais=S("n_terminais"), n_censurados=S("n_censurados"),
            n_area_abriu=S("n_area_abriu"), n_voltas_area=S("n_voltas_area"),
            n_dyn_elig=n_dyn_elig, n_voltas_elig=n_voltas_elig, n_ordenados=n_ord,
            rate_boot=seed_bootstrap_rate([(d["runs"], d["dyn"])
                                           for d in por_seed.values()], seed=boot_rate_seed),
            rate_pois=poisson_ci(n_dyn, n_runs),
            p_volta_morte=clopper_pearson(n_voltas, n_dyn),
            p_volta_morte_boot=seed_bootstrap_prop([(d["voltas"], d["dyn"])
                                                    for d in por_seed.values()], seed=boot_prop_seed),
            p_volta_elig=clopper_pearson(n_voltas_elig, n_dyn_elig),
            p_volta_elig_boot=seed_bootstrap_prop([(d["voltas_elig"], d["dyn_elig"])
                                                   for d in por_seed.values()], seed=boot_prop_seed),
            cp_ordenados=clopper_pearson(n_ord, n_voltas),
            ordenados_boot=seed_bootstrap_prop([(d["ordenados"], d["voltas"])
                                                for d in por_seed.values()], seed=boot_prop_seed),
            ret_times=[rt for cl in items for rt in cl["ret_times"]],
            trios=[tr for cl in items for tr in cl["trios"]],
        )
    return tabela


# ============================ decisões pré-registradas =======================
def avalia_C1(agg_primario, cfg):
    N = agg_primario["n_dyn_elig"]
    k = agg_primario["n_voltas_elig"]
    cp = clopper_pearson(k, N)
    n_star = cfg["C1"]["n_star"]
    thr = cfg["C1"]["mata_thresh"]
    if N < n_star:
        status = "INCONCLUSIVO"
    elif cp[2] < thr:
        status = "MATA"
    else:
        status = "REPRODUZ"
    return dict(N_elig=N, voltas_elig=k, cp=cp, n_star=n_star, mata_thresh=thr,
                status=status)


def avalia_C2(agg_primario, cfg):
    n_v = agg_primario["n_voltas"]
    n_o = agg_primario["n_ordenados"]
    cp = clopper_pearson(n_o, n_v)
    minv = cfg["C2"]["min_voltas"]
    piso = cfg["C2"]["piso"]
    if n_v < minv:
        status = "INCONCLUSIVO"
    elif cp[1] < piso:
        status = "MATA"
    else:
        status = "SUSTENTA"
    return dict(n_voltas=n_v, n_ordenados=n_o, cp=cp, min_voltas=minv, piso=piso,
                status=status)


def avalia_validade(cls_list):
    """Qualquer volta GT no relu ⟹ INVÁLIDO (teorema grad≡0)."""
    bug = []
    for cl in cls_list:
        if cl["rot"] == "relu" and (cl["n_voltas"] > 0 or cl["n_born_rev"] > 0
                                    or cl["n_revive_real"] > 0):
            bug.append((cl["seed"], cl["m"], cl["n_voltas"], cl["n_born_rev"],
                        cl["n_revive_real"]))
    return dict(valido=(len(bug) == 0), bugs=bug)


def escada_C4(por_braco):
    """C4: p(volta|morte) por β, com CIs. SEM MATA."""
    out = {}
    for b in ["sp-b3", "sp-b10", "sp-b300", "relu"]:
        d = por_braco.get(b)
        if d:
            out[b] = dict(p_volta_morte=d["p_volta_morte"],
                          p_volta_morte_boot=d["p_volta_morte_boot"],
                          n_voltas=d["n_voltas"], n_dyn=d["n_dyn"])
    return out


def avaliacao_completa(cls_list, cfg):
    win = cfg["janela_elegivel"]["fim"]
    por_braco = agrega(cls_list, lambda cl: cl["rot"],
                       boot_rate_seed=cfg["bootstrap"]["seed_rate"],
                       boot_prop_seed=cfg["bootstrap"]["seed_prop"])
    prim = cfg["primario"]
    agg_prim = por_braco.get(prim, dict(n_dyn_elig=0, n_voltas_elig=0,
                                        n_voltas=0, n_ordenados=0))
    return dict(
        por_braco=por_braco,
        C1=avalia_C1(agg_prim, cfg),
        C2=avalia_C2(agg_prim, cfg),
        C4=escada_C4(por_braco),
        validade=avalia_validade(cls_list),
        window_fim=win,
    )


# ============================ self-test sintético ============================
def _ep(k, classe, t_morte, t_area, t_res, revived, z_park=-0.3):
    sigma = 1.0 / (1.0 + math.exp(-3.0 * z_park))
    return dict(k=k, classe=classe, t_morte=t_morte, t_area=t_area,
                t_res=t_res, revived=revived, z_park=z_park, sigma_park=sigma)


def _run(rot, ativ, beta, m, seed, eps, H=500000):
    n_rev = sum(1 for e in eps if e["revived"])
    return dict(rot=rot, ativ=ativ, beta=beta, m=m, seed=seed, passos=H,
                episodios_gt=eps, n_revive_real=n_rev)


def _cls(runs, win):
    return [classifica_run(r, win) for r in runs]


def sintetico(cfg):
    print("== SELF-TEST SINTÉTICO analise_4a3 (5 mundos, asserts nos dois sentidos) ==")
    win = cfg["janela_elegivel"]["fim"]
    n_star = cfg["C1"]["n_star"]
    falhas = 0

    def check(nome, cond):
        nonlocal falhas
        print(f"  [{'OK' if cond else 'FALHA'}] {nome}")
        falhas += (not cond)

    # ---- estatística exata: valores conhecidos (bate com scipy) ----
    _, lo, hi = clopper_pearson(0, 10)
    check(f"CP(0/10) hi={hi:.4f}~0.3085 & lo=0", abs(hi - 0.30850) < 1e-3 and lo == 0.0)
    _, lo, hi = clopper_pearson(6, 6)
    check(f"CP(6/6) lo={lo:.4f}~0.5407 (>0.5)", abs(lo - 0.54074) < 1e-3 and lo > 0.5)
    r, plo, phi = poisson_ci(13, 60)
    check(f"PoisCI(13,60)=[{plo:.4f},{phi:.4f}]~[0.1154,0.3705]",
          abs(plo - 0.11537) < 1e-3 and abs(phi - 0.37051) < 1e-3)
    q = poisson_lower_quantile(16.6667, 0.05)
    check(f"n*=PoisLowQ(16.67,0.05)={q} (=10, o congelado)", q == 10 == n_star)
    _, lo0, hi0 = clopper_pearson(0, 50)
    check(f"CP(0/50) hi={hi0:.4f} < mata_thresh {cfg['C1']['mata_thresh']:.4f}",
          hi0 < cfg["C1"]["mata_thresh"])

    # ================= MUNDO (i): taxa-piloto + lei-ordenada =================
    # sp-b3: 20 mortes dinâmicas elegíveis em 12 seeds; 8 voltas TODAS ordenadas.
    runs_i = []
    seeds12 = list(range(1900, 1912))
    v_feitas = 0
    d_feitas = 0
    for i, s in enumerate(seeds12):
        eps = []
        # ~1-2 mortes dinâmicas por seed; algumas voltam (ordenadas), outras terminais
        n_here = 2 if i < 8 else 1
        for j in range(n_here):
            d_feitas += 1
            if v_feitas < 8 and j == 0:
                v_feitas += 1
                eps.append(_ep(j, "dyn", 1000, 30000, 90000, True))    # ordenada
            else:
                eps.append(_ep(j, "dyn", 1500, -1, None, False))       # terminal
        runs_i.append(_run("sp-b3", "softplus", 3.0, 4, s, eps))
    # relu válido (só born-dead terminais)
    runs_i.append(_run("relu", "relu", 300.0, 4, 1900,
                       [_ep(0, "born", 0, -1, None, False)]))
    av = avaliacao_completa(_cls(runs_i, win), cfg)
    check(f"(i) C1 REPRODUZ (N={av['C1']['N_elig']}>=n*, voltas={av['C1']['voltas_elig']})",
          av["C1"]["status"] == "REPRODUZ")
    check("(i) C1 NÃO MATA e NÃO inconclusivo", av["C1"]["status"] not in ("MATA", "INCONCLUSIVO"))
    check(f"(i) C2 SUSTENTA (ord={av['C2']['n_ordenados']}/{av['C2']['n_voltas']}, "
          f"CP-inf={av['C2']['cp'][1]:.3f}>=0.5)", av["C2"]["status"] == "SUSTENTA")
    check("(i) validade OK (relu limpo)", av["validade"]["valido"])

    # ================= MUNDO (ii): taxa-zero (C1 MATA) =====================
    runs_ii = []
    for s in range(1900, 1950):                     # 50 seeds, 1 morte elegível cada, 0 voltas
        runs_ii.append(_run("sp-b3", "softplus", 3.0, 4, s,
                            [_ep(0, "dyn", 1000, -1, None, False)]))
    runs_ii.append(_run("relu", "relu", 300.0, 4, 1900,
                        [_ep(0, "born", 0, -1, None, False)]))
    av = avaliacao_completa(_cls(runs_ii, win), cfg)
    check(f"(ii) C1 MATA (N={av['C1']['N_elig']}>=n*, voltas=0, CP-sup="
          f"{av['C1']['cp'][2]:.4f}<{cfg['C1']['mata_thresh']:.4f})",
          av["C1"]["status"] == "MATA")
    check("(ii) C1 NÃO reproduz e NÃO inconclusivo",
          av["C1"]["status"] not in ("REPRODUZ", "INCONCLUSIVO"))
    check("(ii) C2 INCONCLUSIVO (0 voltas < 6)", av["C2"]["status"] == "INCONCLUSIVO")
    check("(ii) validade OK", av["validade"]["valido"])

    # ================= MUNDO (iii): poucos-eventos (inconclusivos) =========
    runs_iii = [
        _run("sp-b3", "softplus", 3.0, 4, 1900, [_ep(0, "dyn", 1000, 20000, 60000, True)]),
        _run("sp-b3", "softplus", 3.0, 4, 1901, [_ep(0, "dyn", 1200, 25000, 70000, True)]),
        _run("sp-b3", "softplus", 3.0, 4, 1902, [_ep(0, "dyn", 1500, -1, None, False)]),
    ]
    av = avaliacao_completa(_cls(runs_iii, win), cfg)
    check(f"(iii) C1 INCONCLUSIVO (N={av['C1']['N_elig']} < n*={n_star})",
          av["C1"]["status"] == "INCONCLUSIVO")
    check("(iii) C1 NÃO MATA", av["C1"]["status"] != "MATA")
    check(f"(iii) C2 INCONCLUSIVO (voltas={av['C2']['n_voltas']} < 6)",
          av["C2"]["status"] == "INCONCLUSIVO")
    check("(iii) C2 NÃO MATA", av["C2"]["status"] != "MATA")

    # ================= MUNDO (iv): trios desordenados (C2 MATA) ============
    runs_iv = []
    for i, s in enumerate(range(1900, 1908)):        # 8 voltas dinâmicas
        if i < 2:                                    # 2 ordenadas
            eps = [_ep(0, "dyn", 1000, 20000, 80000, True)]
        elif i < 5:                                  # área DEPOIS do res (desordenada)
            eps = [_ep(0, "dyn", 1000, 90000, 80000, True)]
        else:                                        # volta SEM área (t_area=-1)
            eps = [_ep(0, "dyn", 1000, -1, 80000, True)]
        runs_iv.append(_run("sp-b3", "softplus", 3.0, 4, s, eps))
    runs_iv.append(_run("relu", "relu", 300.0, 4, 1900,
                        [_ep(0, "born", 0, -1, None, False)]))
    av = avaliacao_completa(_cls(runs_iv, win), cfg)
    check(f"(iv) C2 MATA (ord={av['C2']['n_ordenados']}/{av['C2']['n_voltas']}, "
          f"CP-inf={av['C2']['cp'][1]:.3f}<0.5)", av["C2"]["status"] == "MATA")
    check("(iv) C2 NÃO sustenta e NÃO inconclusivo (>=6 voltas)",
          av["C2"]["status"] not in ("SUSTENTA", "INCONCLUSIVO"))
    check("(iv) validade OK", av["validade"]["valido"])

    # ================= MUNDO (v): relu-com-volta (INVÁLIDO) ================
    runs_v = [
        _run("sp-b3", "softplus", 3.0, 4, 1900, [_ep(0, "dyn", 1000, 20000, 80000, True)]),
        _run("relu", "relu", 300.0, 4, 1901, [_ep(0, "dyn", 1000, 20000, 80000, True)]),  # BUG
    ]
    av = avaliacao_completa(_cls(runs_v, win), cfg)
    check(f"(v) validade INVÁLIDO (relu com volta: {av['validade']['bugs']})",
          not av["validade"]["valido"] and len(av["validade"]["bugs"]) == 1)

    print(f"== SELF-TEST {'OK — TUDO PASSA' if falhas == 0 else f'FALHOU ({falhas})'} ==")
    return falhas == 0


# ================================ relatório ==================================
def _fmt(t):
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def relatorio(runs, cfg):
    win = cfg["janela_elegivel"]["fim"]
    cls = [classifica_run(r, win) for r in runs]
    av = avaliacao_completa(cls, cfg)
    pb = av["por_braco"]
    L = ["# Confirmatório etapa-4a.3 — resultado\n",
         f"Grade: {len(runs)} runs. Janela elegível [0, {win:.0f}] (H−τ, τ do piloto β3).\n"]
    L.append("## VALIDADE relu")
    L.append("VÁLIDO (relu 0 voltas)." if av["validade"]["valido"]
             else f"**INVÁLIDO — relu com volta: {av['validade']['bugs']}**")
    L.append("")
    L.append("## Por braço")
    L.append("braço | runs | mortes-din | λ/run (boot) | voltas | p(volta|morte) CP | "
             "p(volta|morte elegível) CP | ordenadas/voltas")
    L.append("---|---|---|---|---|---|---|---")
    for b in ["sp-b3", "sp-b10", "sp-b300", "relu"]:
        d = pb.get(b)
        if not d:
            continue
        L.append(f"{b} | {d['n_runs']} | {d['n_dyn']} | {_fmt(d['rate_boot'])} | "
                 f"{d['n_voltas']} | {_fmt(d['p_volta_morte'])} | "
                 f"{_fmt(d['p_volta_elig'])} | {d['n_ordenados']}/{d['n_voltas']}")
    L.append("")
    c1 = av["C1"]
    L.append(f"## C1 (primário {cfg['primario']}): **{c1['status']}**")
    L.append(f"N_elig={c1['N_elig']} (n*={c1['n_star']}), voltas_elig={c1['voltas_elig']}, "
             f"CP p(volta|elegível)={_fmt(c1['cp'])}, mata_thresh={c1['mata_thresh']:.4f}")
    c2 = av["C2"]
    L.append(f"## C2 (coração, {cfg['primario']}): **{c2['status']}**")
    L.append(f"ordenadas={c2['n_ordenados']}/{c2['n_voltas']} (piso {c2['piso']}, "
             f"≥{c2['min_voltas']} voltas), CP-inf={c2['cp'][1]:.3f}")
    L.append("## C4 (escada p(volta|morte) por β; SEM MATA)")
    for b, d in av["C4"].items():
        L.append(f"- {b}: {_fmt(d['p_volta_morte'])} ({d['n_voltas']}/{d['n_dyn']})")
    return "\n".join(L), av


def carrega(resdir):
    runs = []
    for p in sorted(glob.glob(os.path.join(resdir, "*.jsonl.gz"))):
        with gzip.open(p, "rt") as f:
            runs.append(json.load(f))
    return runs


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintetico", action="store_true")
    ap.add_argument("--resultados", default=os.path.join(AQUI, "results-4a3"),
                    help="diretório de results confirmatórios (*.jsonl.gz)")
    a = ap.parse_args()
    cfg = json.load(open(CONFIG))
    if a.sintetico:
        ok = sintetico(cfg)
        raise SystemExit(0 if ok else 1)
    runs = carrega(a.resultados)
    print(f"carregados {len(runs)} runs de {a.resultados}")
    if not runs:
        raise SystemExit("sem resultados confirmatórios ainda (rodar run_4a3.py após pré-registro).")
    txt, _ = relatorio(runs, cfg)
    print(txt)
