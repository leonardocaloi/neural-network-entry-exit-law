"""Analise do CONFIRMATORIO etapa-5 (transformer FFN). Le config_e5.json (numeros
congelados pelo preditor, todos derivados do piloto-de-taxa e5 / rate-results) +
results-e5/ (unidades colapsadas do motor_e5_conf, com HISTERESE e volta SUSTENTADA).

Estimandos pre-registrados (o config manda; nada aqui e decidido a posteriori):
  E-A VALIDADE  : QUALQUER volta sustentada em relu-frozen => INVALIDO (teorema:
                  grad proprio==0 E entrada fixa => terminal; ambas profundidades).
  E-B MONTANTE  : Delta = p_sust(train) - p_sust(frozen) por braco, CI bootstrap por
                  seed. REPRODUZ se CI-inf>0; MATA (montante falso) se CI-sup<=0.
                  Disclosure: piloto ja mostrou Delta>0 decisivo (paridade C1 do 4a.3).
  E-C TRANSFER. : sp-b3-frozen inj -0.3: p(volta sustentada|inj), Clopper-Pearson.
                  MATA se CP-sup < limiar (=1/2 CI-inf do piloto analogo);
                  TRANSFERE se voltas>=n* (=6); senao INCONCLUSIVO.
  E-D BARREIRA  : gelu-frozen: p_sust(prof -2.0) vs p_sust(rasa -0.3). Barreira preve
                  rasa>>profunda. MATA-da-barreira se CP-inf(prof) >= CP-sup(rasa)
                  (barreira nao existe). Sinal de A_drive (deep vs raso) = estimando.
  E-E ESCADA    : p(volta sustentada) por beta {3,10}, frozen e train. Estimacao, sem MATA.

Contagem congelada: volta SUSTENTADA (histerese; unidade termina H viva) unica por
unidade injetada. Trios GT-area (t_area_drive <= t_res) com A_drive reportados.
CIs por BOOTSTRAP DE SEED (clustering). Binomial (Clopper-Pearson) EXATO SEM scipy
por bisseccao na cauda exata. --sintetico valida a logica em 6 mundos com asserts nos
dois sentidos ANTES de qualquer run. alpha e B lidos do config.

Uso:
  ../../venv/bin/python analyze_e5.py --sintetico   # obrigatorio: passa TUDO
  ../../venv/bin/python analyze_e5.py               # roda sobre results-e5/
"""
import glob
import gzip
import json
import math
import os

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(AQUI, "config_e5.json")

try:
    _CFG0 = json.load(open(CONFIG))
except Exception:
    _CFG0 = {}


def _cfg_alpha():
    return float(_CFG0.get("alpha", 0.05))


def _cfg_B():
    return int(_CFG0.get("bootstrap", {}).get("B", 20000))


# ============================ estatistica EXATA (sem scipy) ===================
def _bisect(f, target, incr, lo=0.0, hi=1.0, it=200):
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        val = f(mid)
        if (val < target) == incr:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _binom_sf_ge(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def _binom_cdf_le(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def clopper_pearson(k, n, alpha=None):
    """CI exato de Clopper-Pearson para k/n, por bisseccao na cauda binomial exata.
    (Identico a beta.ppf(a/2,k,n-k+1) / beta.ppf(1-a/2,k+1,n-k), sem scipy.)"""
    if alpha is None:
        alpha = _cfg_alpha()
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    a = alpha / 2.0
    lo = 0.0 if k == 0 else _bisect(lambda p: _binom_sf_ge(k, n, p), a, incr=True)
    hi = 1.0 if k == n else _bisect(lambda p: _binom_cdf_le(k, n, p), a, incr=False)
    return (k / n, float(lo), float(hi))


def _pois_cdf_le(k, mu):
    if k < 0:
        return 0.0
    term = math.exp(-mu)
    s = term
    for i in range(1, k + 1):
        term *= mu / i
        s += term
    return s


def poisson_lower_quantile(mu, q=0.05):
    """Menor inteiro x com P(X<=x) >= q, X~Pois(mu)."""
    term = math.exp(-mu)
    cdf = term
    x = 0
    while cdf < q:
        x += 1
        term *= mu / x
        cdf += term
    return x


# ------------------------- bootstrap de seed (clustering) --------------------
def seed_bootstrap_prop(por_seed_kn, B=None, alpha=None, seed=1):
    """CI de proporcao por cluster de seed. por_seed_kn: lista (k_seed, n_seed)."""
    if B is None:
        B = _cfg_B()
    if alpha is None:
        alpha = _cfg_alpha()
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
    with np.errstate(all="ignore"):
        lo, hi = np.nanpercentile(est, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(p), float(lo), float(hi))


def seed_bootstrap_diff(por_seed_ab, B=None, alpha=None, seed=2):
    """CI da DIFERENCA de proporcoes (train - frozen) por cluster de seed.
    por_seed_ab: dict seed -> (k_tr, n_tr, k_fz, n_fz). Reamostra SEEDS (o cluster)."""
    if B is None:
        B = _cfg_B()
    if alpha is None:
        alpha = _cfg_alpha()
    seeds = list(por_seed_ab.keys())
    if not seeds:
        return (float("nan"), float("nan"), float("nan"))
    ktr = np.array([por_seed_ab[s][0] for s in seeds], float)
    ntr = np.array([por_seed_ab[s][1] for s in seeds], float)
    kfz = np.array([por_seed_ab[s][2] for s in seeds], float)
    nfz = np.array([por_seed_ab[s][3] for s in seeds], float)
    def _d(kt, nt, kf, nf):
        pt = kt.sum() / nt.sum() if nt.sum() > 0 else np.nan
        pf = kf.sum() / nf.sum() if nf.sum() > 0 else np.nan
        return pt - pf
    pt_all = _d(ktr, ntr, kfz, nfz)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(seeds), size=(B, len(seeds)))
    est = np.array([_d(ktr[i], ntr[i], kfz[i], nfz[i]) for i in idx])
    with np.errstate(all="ignore"):
        lo, hi = np.nanpercentile(est, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(pt_all), float(lo), float(hi))


# ============================ classificacao ==================================
def classifica_run(run, tau_por_braco):
    """De um run (unidades colapsadas do motor_e5_conf) devolve contagens.
    p(volta sustentada|inj): inj=denominador; sustained=numerador. Janela: uma volta
    conta p/ 'sustained na janela' se t_res <= H - tau_braco (>= tau de horizonte p/
    confirmar sustentacao). Nota: as mortes injetadas nascem em t=0 (born-dead por
    injecao) => a janela liga no TEMPO DE VOLTA, nao no tempo de morte (difere do 4a.3,
    onde ligava na morte dinamica). O 'sustained' primario e alive-no-fim (histerese);
    a versao janela e reportada como secundaria robusta."""
    arm = run["arm"]; fac = run["fac"]; seed = run["seed"]
    depth = run["depth"]; H = run["cfg"]["passos"]
    tau = tau_por_braco.get(arm, 0.0)
    janela_fim = H - tau
    U = run["unidades"]
    inj = [u for u in U if u["injetada"]]
    n_inj = len(inj)
    n_sust = sum(1 for u in inj if u["sustained"])
    n_rev = sum(1 for u in inj if u["revived"])
    n_hover = sum(1 for u in inj if u["hover"])
    n_sust_janela = sum(1 for u in inj if u["sustained"] and u["t_res"] is not None
                        and u["t_res"] <= janela_fim)
    # trios das voltas sustentadas (t_area_drive <= t_res); volta sem area = falha
    trios = []
    n_area_ok = 0
    for u in inj:
        if u["sustained"]:
            ta = u["t_area_drive"]; tr = u["t_res"]
            area_abriu = ta >= 0
            ordenado = area_abriu and (ta <= (tr if tr is not None else (1 << 30)))
            if ordenado:
                n_area_ok += 1
            trios.append(dict(k=u["k"], depth=u["injdepth"], deep=u["deep"],
                              t_area_drive=ta, t_res=tr, A_drive=u["A_drive"],
                              A_mlp=u["A_mlp"], own_move=u["own_move"], canal=u["canal"],
                              z_park=u["z_park"], zmax_hi=u["zmax_hi"],
                              area_abriu=area_abriu, ordenado=ordenado))
    # sinal A_drive vs A_mlp em TODAS as injetadas (S1 sign-flip; deep vs raso) —
    # sobre TODAS as injetadas (nao so sustentadas), como o piloto mediu.
    inj_sign = [dict(deep=u["deep"], A_drive=u["A_drive"], A_mlp=u["A_mlp"])
                for u in inj if (u["A_drive"] != 0.0 or u["A_mlp"] != 0.0)]
    # natural (nao-injetadas): mortes dinamicas e voltas (validade / lambda_natural)
    nat = [u for u in U if not u["injetada"]]
    n_nat_dyn = sum(1 for u in nat if u["classe1"] == "dyn")
    n_nat_rev = sum(1 for u in nat if u["revived"])
    return dict(
        arm=arm, fac=fac, depth=depth, seed=seed, beta=run["cfg"]["beta"],
        n_inj=n_inj, n_sust=n_sust, n_rev=n_rev, n_hover=n_hover,
        n_sust_janela=n_sust_janela, n_area_ok=n_area_ok, n_sust_trios=len(trios),
        n_nat_dyn=n_nat_dyn, n_nat_rev=n_nat_rev, trios=trios, inj_sign=inj_sign)


def agrega(cls_list, chave):
    """Agrega por chave(cl)->grupo. p_sust = sustained/inj com CP e bootstrap seed."""
    grupos = {}
    for cl in cls_list:
        grupos.setdefault(chave(cl), []).append(cl)
    tab = {}
    for g, items in grupos.items():
        por_seed = {}
        for cl in items:
            d = por_seed.setdefault(cl["seed"], dict(inj=0, sust=0, rev=0))
            d["inj"] += cl["n_inj"]; d["sust"] += cl["n_sust"]; d["rev"] += cl["n_rev"]
        Sx = lambda key: sum(cl[key] for cl in items)  # noqa: E731
        n_inj = Sx("n_inj"); n_sust = Sx("n_sust"); n_rev = Sx("n_rev")
        tab[g] = dict(
            n_runs=len(items), n_seeds=len(por_seed),
            n_inj=n_inj, n_sust=n_sust, n_rev=n_rev, n_hover=Sx("n_hover"),
            n_sust_janela=Sx("n_sust_janela"),
            n_area_ok=Sx("n_area_ok"), n_sust_trios=Sx("n_sust_trios"),
            n_nat_dyn=Sx("n_nat_dyn"), n_nat_rev=Sx("n_nat_rev"),
            cp_sust=clopper_pearson(n_sust, n_inj),
            cp_sust_boot=seed_bootstrap_prop([(d["sust"], d["inj"])
                                              for d in por_seed.values()]),
            trios=[tr for cl in items for tr in cl["trios"]],
            por_seed=por_seed)
    return tab


# ============================ decisoes pre-registradas =======================
def avalia_EA_validade(cls_list):
    """QUALQUER volta sustentada (ou revived) em relu-frozen => INVALIDO."""
    bugs = []
    for cl in cls_list:
        if cl["arm"] == "relu" and cl["fac"] == "frozen":
            if cl["n_sust"] > 0 or cl["n_rev"] > 0 or cl["n_nat_rev"] > 0:
                bugs.append((cl["seed"], cl["depth"], cl["n_sust"], cl["n_rev"], cl["n_nat_rev"]))
    return dict(valido=(len(bugs) == 0), bugs=bugs)


def avalia_EB_montante(cls_list, cfg):
    """Delta = p_sust(train)-p_sust(frozen) por braco (pool profundidades), CI boot seed.
    REPRODUZ se CI-inf>0; MATA se CI-sup<=0; senao INCONCLUSIVO."""
    out = {}
    bseed = cfg["bootstrap"]["seed_montante"]
    for arm in cfg["arms"]:
        tr = [cl for cl in cls_list if cl["arm"] == arm and cl["fac"] == "train"]
        fz = [cl for cl in cls_list if cl["arm"] == arm and cl["fac"] == "frozen"]
        por_seed = {}
        for cl in tr:
            d = por_seed.setdefault(cl["seed"], [0, 0, 0, 0])
            d[0] += cl["n_sust"]; d[1] += cl["n_inj"]
        for cl in fz:
            d = por_seed.setdefault(cl["seed"], [0, 0, 0, 0])
            d[2] += cl["n_sust"]; d[3] += cl["n_inj"]
        por_seed = {s: tuple(v) for s, v in por_seed.items()}
        d, lo, hi = seed_bootstrap_diff(por_seed, seed=bseed)
        ktr = sum(cl["n_sust"] for cl in tr); ntr = sum(cl["n_inj"] for cl in tr)
        kfz = sum(cl["n_sust"] for cl in fz); nfz = sum(cl["n_inj"] for cl in fz)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            status = "INCONCLUSIVO"
        elif lo > 0:
            status = "REPRODUZ"
        elif hi <= 0:
            status = "MATA"
        else:
            status = "INCONCLUSIVO"
        out[arm] = dict(delta=d, ci=(lo, hi), status=status,
                        p_train=(ktr / ntr if ntr else float("nan")),
                        p_frozen=(kfz / nfz if nfz else float("nan")),
                        n_sust_train=ktr, n_train=ntr, n_sust_frozen=kfz, n_frozen=nfz)
    return out


def avalia_EC_transferencia(tab_cell_depth, cfg):
    """sp-b3-frozen inj -0.3: p(volta sustentada|inj) CP. MATA se CP-sup<thresh;
    TRANSFERE se sust>=n*; senao INCONCLUSIVO. (MATA checado 1o; disclosure no config.)"""
    key = ("sp-b3", "frozen", -0.3)
    d = tab_cell_depth.get(key, dict(n_inj=0, n_sust=0, cp_sust=(float("nan"),) * 3))
    N = d["n_inj"]; k = d["n_sust"]
    cp = clopper_pearson(k, N) if N > 0 else (float("nan"), float("nan"), float("nan"))
    thr = cfg["E_C"]["mata_thresh"]
    n_star = cfg["E_C"]["n_star"]
    if N == 0:
        status = "INCONCLUSIVO"
    elif math.isfinite(cp[2]) and cp[2] < thr:
        status = "MATA"
    elif k >= n_star:
        status = "TRANSFERE"
    else:
        status = "INCONCLUSIVO"
    return dict(N_inj=N, sust=k, cp=cp, n_star=n_star, mata_thresh=thr, status=status)


def avalia_ED_barreira(tab_cell_depth, cfg):
    """gelu-frozen: rasa(-0.3) vs profunda(-2.0). 11º PARECER (confound
    profundidade×barreira): a separação INTRA-gelu é necessária mas NÃO
    suficiente — nada garante que a profundidade sozinha não explique o zero
    da profunda. O controle depth-matched JÁ está na grade: sp-b3-frozen a
    -2.0 (mesma profundidade, SEM barreira, g'>0). Regra congelada:
      MATA-BARREIRA: CP-inf(prof-gelu) >= CP-sup(rasa-gelu)  [limpo, inalterado]
      SUSTENTA-BARREIRA: exige (a) intra-gelu rasa>>prof (CP disjuntos, rasa>=n*)
        E (b) cross-arm depth-matched: CP-sup(gelu-frozen-prof) < CP-inf(sp-b3-frozen-prof)
        [gelu revive MENOS que softplus À MESMA profundidade];
      só (a) sem (b) ⟹ "CONSISTENTE-COM-BARREIRA (confundido com profundidade)";
      senão INCONCLUSIVO."""
    rasa = tab_cell_depth.get(("gelu", "frozen", -0.3), dict(n_inj=0, n_sust=0))
    prof = tab_cell_depth.get(("gelu", "frozen", -2.0), dict(n_inj=0, n_sust=0))
    spp = tab_cell_depth.get(("sp-b3", "frozen", -2.0), dict(n_inj=0, n_sust=0))
    cp_r = clopper_pearson(rasa["n_sust"], rasa["n_inj"]) if rasa["n_inj"] else (float("nan"),) * 3
    cp_p = clopper_pearson(prof["n_sust"], prof["n_inj"]) if prof["n_inj"] else (float("nan"),) * 3
    cp_s = clopper_pearson(spp["n_sust"], spp["n_inj"]) if spp["n_inj"] else (float("nan"),) * 3
    n_star = cfg["E_D"]["n_star"]
    intra = (rasa["n_inj"] > 0 and prof["n_inj"] > 0
             and cp_r[1] > cp_p[2] and rasa["n_sust"] >= n_star)
    cross = (spp["n_inj"] > 0 and prof["n_inj"] > 0 and cp_p[2] < cp_s[1])
    if rasa["n_inj"] == 0 or prof["n_inj"] == 0:
        status = "INCONCLUSIVO"
    elif cp_p[1] >= cp_r[2]:
        status = "MATA-BARREIRA"        # profunda >= rasa com confianca => sem barreira
    elif intra and cross:
        status = "SUSTENTA-BARREIRA"    # rasa>>prof E gelu<softplus a -2.0
    elif intra:
        status = "CONSISTENTE-COM-BARREIRA (confundido com profundidade)"
    else:
        status = "INCONCLUSIVO"
    return dict(rasa=dict(n_sust=rasa["n_sust"], n_inj=rasa["n_inj"], cp=cp_r),
                prof=dict(n_sust=prof["n_sust"], n_inj=prof["n_inj"], cp=cp_p),
                controle_sp_prof=dict(n_sust=spp["n_sust"], n_inj=spp["n_inj"], cp=cp_s),
                intra_gelu=bool(intra), cross_depth_matched=bool(cross),
                n_star=n_star, status=status)


def sinal_Adrive_gelu(cls_list):
    """Estimando E-D (S1 sign-flip): fracao de unidades GELU-frozen INJETADAS com
    sinal(A_drive)!=sinal(A_mlp), separado deep (prof) vs raso — sobre TODAS as
    injetadas (nao so sustentadas), como o piloto (80-95% opostos nos profundos)."""
    out = {}
    itens = [s for cl in cls_list if cl["arm"] == "gelu" and cl["fac"] == "frozen"
             for s in cl["inj_sign"]]
    for deep, nome in [(False, "rasa"), (True, "profunda")]:
        sub = [s for s in itens if s["deep"] == deep]
        opostos = sum(1 for s in sub if s["A_drive"] * s["A_mlp"] < 0)
        out[nome] = dict(n=len(sub), opostos=opostos,
                         frac=(opostos / len(sub) if sub else float("nan")))
    return out


def escada_EE(tab_cell):
    """E-E: p(volta sustentada) por beta {3,10}, frozen e train. Estimacao, sem MATA."""
    out = {}
    for arm in ["sp-b3", "sp-b10"]:
        for fac in ["train", "frozen"]:
            d = tab_cell.get((arm, fac))
            if d:
                out[(arm, fac)] = dict(p=d["cp_sust"], n_sust=d["n_sust"], n_inj=d["n_inj"])
    return out


def avaliacao_completa(cls_list, cfg):
    tau = cfg["tau_por_braco"]
    tab_cell = agrega(cls_list, lambda cl: (cl["arm"], cl["fac"]))
    tab_cell_depth = agrega(cls_list, lambda cl: (cl["arm"], cl["fac"], cl["depth"]))
    return dict(
        tab_cell=tab_cell, tab_cell_depth=tab_cell_depth,
        E_A=avalia_EA_validade(cls_list),
        E_B=avalia_EB_montante(cls_list, cfg),
        E_C=avalia_EC_transferencia(tab_cell_depth, cfg),
        E_D=avalia_ED_barreira(tab_cell_depth, cfg),
        E_D_sinal=sinal_Adrive_gelu(cls_list),
        E_E=escada_EE(tab_cell),
    )


# ============================ self-test sintetico ============================
def _u(k, injetada=False, injdepth=None, sustained=False, revived=None, final_alive=None,
       t_res=None, t_area_drive=-1, A_drive=0.0, A_mlp=0.0, own_move=0.0, canal=None,
       classe1=None, deep=False, z_park=None, zmax_hi=None):
    if revived is None:
        revived = sustained
    if final_alive is None:
        final_alive = sustained
    hover = revived and not final_alive
    return dict(k=k, injetada=injetada, injdepth=injdepth, deep=deep, classe1=classe1,
                revived=revived, sustained=sustained, final_alive=final_alive, hover=hover,
                t_res=t_res, z_park=z_park, zmax_hi=zmax_hi, n_eps=1, n_dyn=0,
                t_area_drive=t_area_drive, A_drive=A_drive, A_mlp=A_mlp,
                own_move=own_move, canal=canal)


def _run(arm, fac, depth, seed, unidades, beta=3.0, H=200000):
    return dict(arm=arm, fac=fac, depth=depth, seed=seed,
                cfg=dict(passos=H, beta=beta, delta_h=0.05, freeze_attn=(fac == "frozen")),
                unidades=unidades)


def _cls(runs, cfg):
    return [classifica_run(r, cfg["tau_por_braco"]) for r in runs]


def _mkcell(arm, fac, depth, k_sust, n, seed0=2000, beta=3.0, tarea=1000, tres=50000,
            Ad=1.0, Am=1.0, canal=None, deep=False):
    """n runs (1 injetada cada); k_sust deles com volta sustentada."""
    if canal is None:
        canal = "proprio" if fac == "frozen" else "proprio+montante"
    runs = []
    for i in range(n):
        if i < k_sust:
            u = _u(0, injetada=True, injdepth=depth, sustained=True, t_res=tres,
                   t_area_drive=tarea, A_drive=Ad, A_mlp=Am, canal=canal, deep=deep,
                   z_park=depth, zmax_hi=0.3, classe1="born")
        else:
            u = _u(0, injetada=True, injdepth=depth, sustained=False, revived=False,
                   final_alive=False, canal=None, deep=deep, z_park=depth, zmax_hi=-0.1,
                   classe1="born")
        runs.append(_run(arm, fac, depth, seed0 + i, [u], beta=beta))
    return runs


def sintetico(cfg):
    print("== SELF-TEST SINTETICO analise_e5 (6 mundos, asserts nos dois sentidos) ==")
    falhas = 0

    def check(nome, cond):
        nonlocal falhas
        print(f"  [{'OK' if cond else 'FALHA'}] {nome}")
        falhas += (not cond)

    # ---- estatistica exata: valores conhecidos (bate com scipy) ----
    _, lo, hi = clopper_pearson(0, 10)
    check(f"CP(0/10) hi={hi:.4f}~0.3085", abs(hi - 0.30850) < 1e-3 and lo == 0.0)
    _, lo, hi = clopper_pearson(6, 6)
    check(f"CP(6/6) lo={lo:.4f}~0.5407 (>0.5)", abs(lo - 0.54074) < 1e-3)
    _, lo, hi = clopper_pearson(6, 50)
    check(f"CP(6/50)=[{lo:.4f},{hi:.4f}] (n*=6 borderline)", lo > 0.04 and hi < 0.25)
    thr = cfg["E_C"]["mata_thresh"]
    _, lo0, hi0 = clopper_pearson(0, 50)
    check(f"CP-sup(0/50)={hi0:.4f} > mata_thresh_EC={thr:.4f} "
          "(=> 0 voltas a N=50 e INCONCLUSIVO, NAO mata; disclosure)", hi0 > thr)
    N_mata = 1000
    _, lo5, hi5 = clopper_pearson(0, N_mata)
    check(f"CP-sup(0/{N_mata})={hi5:.5f} < mata_thresh_EC={thr:.5f} "
          f"(=> MATA de E-C e ATINGIVEL em N={N_mata}: falsificavel em principio, "
          "so nao a N=50 — disclosure honesta)", hi5 < thr)

    n_min = cfg["seeds"]  # nº de seeds por celula
    NS = len(n_min)

    # ============ MUNDO (i): TUDO-REPRODUZ ============
    runs = []
    # sp-b3: frozen rasa com muitas voltas (transfere), train ainda mais (montante)
    runs += _mkcell("sp-b3", "frozen", -0.3, 12, NS)
    runs += _mkcell("sp-b3", "frozen", -2.0, 9, NS)   # 9/50: CP-inf>CP-sup(0/50) — o
    # controle depth-matched revive SEM barreira ⟹ cross-arm do E-D fecha (11º parecer)
    runs += _mkcell("sp-b3", "train", -0.3, 22, NS, canal="proprio+montante")
    runs += _mkcell("sp-b3", "train", -2.0, 15, NS, canal="proprio+montante")
    runs += _mkcell("sp-b10", "frozen", -0.3, 8, NS)
    runs += _mkcell("sp-b10", "frozen", -2.0, 2, NS)
    runs += _mkcell("sp-b10", "train", -0.3, 18, NS)
    runs += _mkcell("sp-b10", "train", -2.0, 3, NS)
    # gelu: barreira REAL (rasa >> profunda), com sinal-oposto nos profundos
    runs += _mkcell("gelu", "frozen", -0.3, 14, NS, Ad=1.0, Am=1.0)
    runs += _mkcell("gelu", "frozen", -2.0, 0, NS)
    runs += _mkcell("gelu", "train", -0.3, 16, NS)
    runs += _mkcell("gelu", "train", -2.0, 1, NS)
    # relu: frozen 0 (valido), train com montante
    runs += _mkcell("relu", "frozen", -0.3, 0, NS, canal=None)
    runs += _mkcell("relu", "frozen", -2.0, 0, NS, canal=None)
    runs += _mkcell("relu", "train", -0.3, 8, NS, canal="montante")
    runs += _mkcell("relu", "train", -2.0, 0, NS, canal="montante")
    av = avaliacao_completa(_cls(runs, cfg), cfg)
    check("(i) E-A VALIDO (relu-frozen 0 voltas)", av["E_A"]["valido"])
    check(f"(i) E-B sp-b3 REPRODUZ (Delta={av['E_B']['sp-b3']['delta']:.3f}>0)",
          av["E_B"]["sp-b3"]["status"] == "REPRODUZ")
    check(f"(i) E-B relu REPRODUZ (0->montante, Delta={av['E_B']['relu']['delta']:.3f})",
          av["E_B"]["relu"]["status"] == "REPRODUZ")
    check(f"(i) E-C TRANSFERE (sust={av['E_C']['sust']}>=n*={av['E_C']['n_star']})",
          av["E_C"]["status"] == "TRANSFERE")
    check(f"(i) E-D SUSTENTA-BARREIRA (rasa {av['E_D']['rasa']['n_sust']} >> prof "
          f"{av['E_D']['prof']['n_sust']})", av["E_D"]["status"] == "SUSTENTA-BARREIRA")

    # ============ MUNDO (ii): MONTANTE-AUSENTE (E-B MATA) ============
    # train <= frozen com confianca (montante nao adiciona) => CI-sup(Delta)<=0.
    runs = []
    for arm in ["sp-b3", "sp-b10", "gelu", "relu"]:
        runs += _mkcell(arm, "frozen", -0.3, 12, NS)
        runs += _mkcell(arm, "frozen", -2.0, 4, NS)
        runs += _mkcell(arm, "train", -0.3, 3, NS)
        runs += _mkcell(arm, "train", -2.0, 0, NS)
    # relu-frozen precisa ser 0 p/ validade; refaz relu
    runs = [r for r in runs if not (r["arm"] == "relu" and r["fac"] == "frozen")]
    runs += _mkcell("relu", "frozen", -0.3, 0, NS, canal=None)
    runs += _mkcell("relu", "frozen", -2.0, 0, NS, canal=None)
    av = avaliacao_completa(_cls(runs, cfg), cfg)
    check(f"(ii) E-B sp-b3 MATA (Delta={av['E_B']['sp-b3']['delta']:.3f}<=0, "
          f"CI-sup={av['E_B']['sp-b3']['ci'][1]:.3f}<=0)",
          av["E_B"]["sp-b3"]["status"] == "MATA")
    check("(ii) E-B sp-b3 NAO reproduz", av["E_B"]["sp-b3"]["status"] != "REPRODUZ")
    check("(ii) E-A ainda VALIDO", av["E_A"]["valido"])

    # ============ MUNDO (iii): TRANSFERENCIA-ZERO (E-C INCONCLUSIVO a N=50) ======
    runs = _mkcell("sp-b3", "frozen", -0.3, 0, NS)   # 0 voltas sustentadas em 50
    av = avaliacao_completa(_cls(runs, cfg), cfg)
    check(f"(iii) E-C INCONCLUSIVO (sust=0, CP-sup={av['E_C']['cp'][2]:.4f} > thresh "
          f"{av['E_C']['mata_thresh']:.4f}; N=50 nao decide zero-vs-raro)",
          av["E_C"]["status"] == "INCONCLUSIVO")
    check("(iii) E-C NAO transfere e NAO mata (honesto sobre potencia)",
          av["E_C"]["status"] not in ("TRANSFERE", "MATA"))

    # ============ MUNDO (iv): BARREIRA-INEXISTENTE (E-D MATA-BARREIRA) ===========
    runs = []
    runs += _mkcell("gelu", "frozen", -0.3, 4, NS)     # rasa BAIXA
    runs += _mkcell("gelu", "frozen", -2.0, 16, NS, deep=True)  # profunda ALTA
    av = avaliacao_completa(_cls(runs, cfg), cfg)
    check(f"(iv) E-D MATA-BARREIRA (prof {av['E_D']['prof']['n_sust']} >= rasa "
          f"{av['E_D']['rasa']['n_sust']}; CP-inf(prof)={av['E_D']['prof']['cp'][1]:.3f} "
          f">= CP-sup(rasa)={av['E_D']['rasa']['cp'][2]:.3f})",
          av["E_D"]["status"] == "MATA-BARREIRA")
    check("(iv) E-D NAO sustenta barreira", av["E_D"]["status"] != "SUSTENTA-BARREIRA")

    # ============ MUNDO (v): RELU-FROZEN-COM-VOLTA (E-A INVALIDO) ================
    runs = _mkcell("sp-b3", "frozen", -0.3, 10, NS)
    bug = _run("relu", "frozen", -0.3, 2000,
               [_u(0, injetada=True, injdepth=-0.3, sustained=True, t_res=30000,
                   t_area_drive=1000, canal="proprio", classe1="born")])
    runs.append(bug)
    av = avaliacao_completa(_cls(runs, cfg), cfg)
    check(f"(v) E-A INVALIDO (relu-frozen com volta: {av['E_A']['bugs']})",
          (not av["E_A"]["valido"]) and len(av["E_A"]["bugs"]) == 1)

    # ============ MUNDO (vi): POUCOS-EVENTOS (INCONCLUSIVOS) ====================
    # so 3 seeds, poucas voltas -> E-C < n* e E-D sem separacao
    few = [2000, 2001, 2002]
    runs = []
    for i, s in enumerate(few):
        runs.append(_run("sp-b3", "frozen", -0.3, s,
                         [_u(0, injetada=True, injdepth=-0.3, sustained=(i == 0),
                             t_res=40000, t_area_drive=1000, canal="proprio",
                             classe1="born", z_park=-0.3, zmax_hi=0.2)]))
        runs.append(_run("gelu", "frozen", -0.3, s,
                         [_u(0, injetada=True, injdepth=-0.3, sustained=(i == 0),
                             classe1="born")]))
        runs.append(_run("gelu", "frozen", -2.0, s,
                         [_u(0, injetada=True, injdepth=-2.0, sustained=False, deep=True,
                             classe1="born")]))
    av = avaliacao_completa(_cls(runs, cfg), cfg)
    check(f"(vi) E-C INCONCLUSIVO (sust={av['E_C']['sust']} < n*={av['E_C']['n_star']})",
          av["E_C"]["status"] == "INCONCLUSIVO")
    check(f"(vi) E-D INCONCLUSIVO (rasa {av['E_D']['rasa']['n_sust']}, prof "
          f"{av['E_D']['prof']['n_sust']}; sem separacao)",
          av["E_D"]["status"] == "INCONCLUSIVO")

    # ============ MUNDO (vii): PROFUNDIDADE-PURA (confound do 11º parecer) ======
    # gelu rasa>>prof (intra fecha) MAS softplus-frozen a -2.0 TAMBEM nao revive
    # => nada separa barreira de profundidade => rotulo CONSISTENTE, nunca SUSTENTA
    runs = []
    runs += _mkcell("gelu", "frozen", -0.3, 14, NS, Ad=1.0, Am=1.0)
    runs += _mkcell("gelu", "frozen", -2.0, 0, NS)
    runs += _mkcell("sp-b3", "frozen", -2.0, 0, NS)   # controle depth-matched TAMBEM zero
    runs += _mkcell("sp-b3", "frozen", -0.3, 12, NS)
    runs += _mkcell("relu", "frozen", -0.3, 0, NS, canal=None)
    runs += _mkcell("relu", "frozen", -2.0, 0, NS, canal=None)
    av = avaliacao_completa(_cls(runs, cfg), cfg)
    check(f"(vii) E-D NAO-SUSTENTA sob profundidade-pura (intra={av['E_D']['intra_gelu']}, "
          f"cross={av['E_D']['cross_depth_matched']})",
          av["E_D"]["status"].startswith("CONSISTENTE"))
    check("(vii) E-D nao e MATA nem SUSTENTA (rotulo honesto do confound)",
          av["E_D"]["status"] not in ("SUSTENTA-BARREIRA", "MATA-BARREIRA"))

    print(f"== SELF-TEST {'OK — TUDO PASSA' if falhas == 0 else f'FALHOU ({falhas})'} ==")
    return falhas == 0


# ================================ relatorio ==================================
def _fmt(t):
    if not (isinstance(t, (list, tuple)) and len(t) == 3):
        return str(t)
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def carrega(resdir):
    runs = []
    for p in sorted(glob.glob(os.path.join(resdir, "*.jsonl.gz"))):
        with gzip.open(p, "rt") as f:
            runs.append(json.load(f))
    return runs


def relatorio(runs, cfg):
    cls = _cls(runs, cfg)
    av = avaliacao_completa(cls, cfg)
    L = [f"# Confirmatorio etapa-5 (transformer FFN) — resultado\n",
         f"Runs: {len(runs)}. delta_h={cfg['delta_h']:.4f}. H={cfg['grade']['H']}.\n"]
    L.append("## E-A VALIDADE relu-frozen")
    L.append("VALIDO (relu-frozen 0 voltas sustentadas)." if av["E_A"]["valido"]
             else f"**INVALIDO — relu-frozen com volta: {av['E_A']['bugs']}**")
    L.append("\n## Tabela por celula x profundidade (p_sust = sustained/inj)")
    L.append("cell | depth | inj | sust | rev | hover | p_sust CP | p_sust boot | area_ok/trios")
    L.append("---|---|---|---|---|---|---|---|---")
    for (arm, fac, depth), d in sorted(av["tab_cell_depth"].items()):
        L.append(f"{arm}-{fac} | {depth} | {d['n_inj']} | {d['n_sust']} | {d['n_rev']} | "
                 f"{d['n_hover']} | {_fmt(d['cp_sust'])} | {_fmt(d['cp_sust_boot'])} | "
                 f"{d['n_area_ok']}/{d['n_sust_trios']}")
    L.append(f"\n## E-B MONTANTE (Delta = p_sust train - frozen; boot por seed)")
    for arm, d in av["E_B"].items():
        L.append(f"- {arm}: **{d['status']}** Delta={d['delta']:.3f} CI[{d['ci'][0]:.3f},"
                 f"{d['ci'][1]:.3f}] (train {d['n_sust_train']}/{d['n_train']} vs frozen "
                 f"{d['n_sust_frozen']}/{d['n_frozen']})")
    c = av["E_C"]
    L.append(f"\n## E-C TRANSFERENCIA (sp-b3-frozen -0.3): **{c['status']}**")
    L.append(f"sust={c['sust']}/{c['N_inj']} CP={_fmt(c['cp'])} n*={c['n_star']} "
             f"mata_thresh={c['mata_thresh']:.4f}")
    dd = av["E_D"]
    L.append(f"\n## E-D BARREIRA-GELU (frozen): **{dd['status']}**")
    L.append(f"rasa(-0.3) {dd['rasa']['n_sust']}/{dd['rasa']['n_inj']} CP={_fmt(dd['rasa']['cp'])}; "
             f"profunda(-2.0) {dd['prof']['n_sust']}/{dd['prof']['n_inj']} CP={_fmt(dd['prof']['cp'])}")
    for nome, s in av["E_D_sinal"].items():
        L.append(f"  sinal A_drive!=A_mlp {nome}: {s['opostos']}/{s['n']} "
                 f"(frac={s['frac'] if isinstance(s['frac'],str) else round(s['frac'],2) if s['frac']==s['frac'] else 'nan'})")
    L.append(f"\n## E-E ESCADA p_sust por beta (sem MATA)")
    for (arm, fac), d in sorted(av["E_E"].items()):
        L.append(f"- {arm}-{fac}: {_fmt(d['p'])} ({d['n_sust']}/{d['n_inj']})")
    return "\n".join(L), av


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintetico", action="store_true")
    ap.add_argument("--resultados", default=os.path.join(AQUI, "results-e5"))
    a = ap.parse_args()
    cfg = json.load(open(CONFIG))
    if a.sintetico:
        ok = sintetico(cfg)
        raise SystemExit(0 if ok else 1)
    runs = carrega(a.resultados)
    print(f"carregados {len(runs)} runs de {a.resultados}")
    if not runs:
        raise SystemExit("sem resultados confirmatorios ainda (rodar run_e5.py apos pre-registro).")
    txt, _ = relatorio(runs, cfg)
    print(txt)
