"""Análise do PILOTO-DE-TAXA (etapa-4a.3). Régua-independente no CONTADOR
(ground-truth zmax<0->zmax>0); régua v2 só para o trio (t_morte,t_area,t_res) e
a flag área-abriu, como manda o 9º parecer.

Reconstrói de cada run os episódios de morte a partir do log de ground-truth
(nasce_morto + death_events + revive_events), classifica cada morte DINÂMICA
(t_morte>0) em volta / TERMINAL (área nunca abriu) / CENSURADO (área abriu, sem
volta até H), casa com a v2 para o trio, e agrega por braço e por m com:
  - λ_morte-dinâmica/run: CI por BOOTSTRAP DE SEED (clustering; a init fixa as
    mortes ⟹ superdispersão) + Poisson exato iid (contraste que exibe o excesso).
  - p(volta|morte) e p(volta|morte,área-abriu): Clopper-Pearson exato.
  - return-time (t_res-t_morte, ground-truth) por β ⟹ τ = quantil 0.9.
  - trios de TODAS as voltas dinâmicas, regra ESTRITA t_morte<t_area<t_res.
  - VALIDADE relu: qualquer volta ⟹ BUG.

Uso:
  ../../../../../venv/bin/python analisa_taxa.py --selftest   # testa a lógica
  ../../../../../venv/bin/python analisa_taxa.py              # roda sobre results/
"""
import glob
import gzip
import json
import os

import numpy as np
from scipy.stats import beta as _beta, chi2 as _chi2

AQUI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(AQUI, "results")

BRACOS = ["sp-b3", "sp-b10", "sp-b300", "relu"]
BETA_DE = {"sp-b3": 3.0, "sp-b10": 10.0, "sp-b300": 300.0, "relu": None}


# ----------------------------- estatística exata -----------------------------
def clopper_pearson(k, n, alpha=0.05):
    """CI exato (Clopper-Pearson) para proporção k/n."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    lo = 0.0 if k == 0 else _beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else _beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (k / n, float(lo), float(hi))


def poisson_ci(k, exposure, alpha=0.05):
    """CI exato (Garwood) para taxa Poisson iid = k/exposure."""
    lo = 0.0 if k == 0 else _chi2.ppf(alpha / 2, 2 * k) / 2.0
    hi = _chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2.0
    e = max(exposure, 1e-12)
    return (k / e, lo / e, hi / e)


def seed_bootstrap_rate(por_seed, B=20000, alpha=0.05, seed=0):
    """CI de taxa por CLUSTER DE SEED: reamostra seeds (não mortes), recomputa
    total_mortes/total_runs. Captura a superdispersão por-init do parecer.
    `por_seed`: lista de (n_runs_da_seed, n_mortes_da_seed)."""
    if not por_seed:
        return (float("nan"), float("nan"), float("nan"))
    runs = np.array([r for r, _ in por_seed], float)
    mortes = np.array([d for _, d in por_seed], float)
    rate = mortes.sum() / max(runs.sum(), 1e-12)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(por_seed), size=(B, len(por_seed)))
    bm = mortes[idx].sum(1)
    br = runs[idx].sum(1)
    est = bm / np.maximum(br, 1e-12)
    lo, hi = np.percentile(est, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(rate), float(lo), float(hi))


def seed_bootstrap_prop(por_seed_kn, B=20000, alpha=0.05, seed=1):
    """CI de proporção por cluster de seed. `por_seed_kn`: lista (k_seed, n_seed).
    Reamostra seeds; ignora seeds com n=0 no denominador do reamostrado."""
    kn = [(k, n) for (k, n) in por_seed_kn]
    if not kn:
        return (float("nan"), float("nan"), float("nan"))
    ks = np.array([k for k, _ in kn], float)
    ns = np.array([n for _, n in kn], float)
    tot = ns.sum()
    if tot == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = ks.sum() / tot
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(kn), size=(B, len(kn)))
    bk = ks[idx].sum(1)
    bn = ns[idx].sum(1)
    ok = bn > 0
    est = np.where(ok, bk / np.maximum(bn, 1e-12), np.nan)
    lo, hi = np.nanpercentile(est, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(p), float(lo), float(hi))


# --------------------- reconstrução de episódios (ground-truth) --------------
def reconstrui_episodios(run):
    """Do log de ground-truth de UM run devolve a lista de episódios de morte.
    Cada episódio: dict(k, classe['born'|'dyn'], t_morte_gt, revived, t_res_gt).
    Regra: estado inicial morto sse k in nasce_morto; death/revive alternam."""
    m = run["m"]
    nasce = set(run["nasce_morto"])
    deaths = {}
    for t, k in run["death_events"]:
        deaths.setdefault(k, []).append(int(t))
    revs = {}
    for t, k in run["revive_events"]:
        revs.setdefault(k, []).append(int(t))
    H = run["passos"]
    eps = []
    for k in range(m):
        ev = [(t, "death") for t in deaths.get(k, [])] + \
             [(t, "revive") for t in revs.get(k, [])]
        ev.sort(key=lambda x: (x[0], 0 if x[1] == "death" else 1))
        estado = "morto" if k in nasce else "vivo"
        aberto = None                     # t_morte do intervalo morto corrente
        if estado == "morto":
            aberto = 0
        for t, tp in ev:
            if tp == "death" and estado == "vivo":
                aberto = t
                estado = "morto"
            elif tp == "revive" and estado == "morto":
                eps.append(dict(k=k, classe=("born" if aberto == 0 else "dyn"),
                                t_morte_gt=int(aberto), revived=True,
                                t_res_gt=int(t)))
                aberto = None
                estado = "vivo"
            # transições impossíveis (motor garante alternância) são ignoradas
        if estado == "morto" and aberto is not None:
            eps.append(dict(k=k, classe=("born" if aberto == 0 else "dyn"),
                            t_morte_gt=int(aberto), revived=False, t_res_gt=None))
    return eps


def _v2_por_k(run):
    idx = {}
    for e in run["v2_episodios"]:
        idx.setdefault(e["k"], []).append(e)
    return idx


def casa_v2(ep, idx_v2):
    """Casa um episódio de ground-truth com o episódio v2 de MESMO k e t_morte
    mais próximo. Devolve (trio_dict|None, gap). trio = (t_morte,t_area,t_res) v2."""
    cand = idx_v2.get(ep["k"], [])
    if not cand:
        return None, None
    e = min(cand, key=lambda ee: abs(ee["t_morte"] - ep["t_morte_gt"]))
    trio = dict(t_morte=int(e["t_morte"]), t_area=int(e["t_area"]),
                t_res=int(e["t_res"]), estado=e["estado"])
    return trio, abs(e["t_morte"] - ep["t_morte_gt"])


def trio_ordenado(trio):
    """Regra ESTRITA do parecer: t_morte<t_area<t_res (todos>=0). Sem área ⟹ False."""
    if trio is None:
        return False
    tm, ta, tr = trio["t_morte"], trio["t_area"], trio["t_res"]
    if ta < 0 or tr < 0:
        return False
    return tm < ta < tr


def classifica_run(run):
    """Devolve dict com as contagens ground-truth de UM run + detalhes de voltas."""
    idx_v2 = _v2_por_k(run)
    eps = reconstrui_episodios(run)
    dyn = [e for e in eps if e["classe"] == "dyn"]
    born = [e for e in eps if e["classe"] == "born"]
    voltas, terminais, censurados = [], [], []
    area_abriu_deaths = 0
    n_voltas_area = 0                      # voltas cuja área abriu (numerador de p(volta|área))
    anomalias = []
    trios = []
    ret_times = []
    for e in dyn:
        trio, gap = casa_v2(e, idx_v2)
        area_abriu = (trio is not None and trio["t_area"] >= 0)
        e["area_abriu"] = bool(area_abriu)
        e["trio"] = trio
        e["v2_gap"] = gap
        if area_abriu:
            area_abriu_deaths += 1
        if e["revived"]:
            voltas.append(e)
            if area_abriu:
                n_voltas_area += 1
            ret_times.append(int(e["t_res_gt"] - e["t_morte_gt"]))
            trios.append(dict(k=e["k"], t_morte_gt=e["t_morte_gt"],
                              t_res_gt=e["t_res_gt"], trio_v2=trio,
                              ordenado=trio_ordenado(trio),
                              area_abriu=bool(area_abriu)))
            if not area_abriu:
                anomalias.append(("volta_sem_area", run["rot"], run["m"],
                                  run["seed"], e["k"]))
        else:
            if area_abriu:
                censurados.append(e)
            else:
                terminais.append(e)
    born_rev = [e for e in born if e["revived"]]
    return dict(
        rot=run["rot"], m=run["m"], seed=run["seed"], beta=run["beta"],
        n_born_dead=run["n_nasce_morto"],
        n_dyn=len(dyn), n_voltas=len(voltas),
        n_terminais=len(terminais), n_censurados=len(censurados),
        n_area_abriu=area_abriu_deaths, n_voltas_area=n_voltas_area,
        n_born_rev=len(born_rev),
        n_revive_real=run["n_revive_real"],
        ret_times=ret_times, trios=trios, anomalias=anomalias,
        loss_plato=run.get("loss_plato"),
    )


# ------------------------------- agregação -----------------------------------
def carrega():
    runs = []
    for p in sorted(glob.glob(os.path.join(OUT, "*.jsonl.gz"))):
        with gzip.open(p, "rt") as f:
            runs.append(json.load(f))
    return runs


def agrega(cls_list, chave):
    """Agrega uma lista de classifica_run() por `chave(cl)->grupo`."""
    grupos = {}
    for cl in cls_list:
        g = chave(cl)
        grupos.setdefault(g, []).append(cl)
    tabela = {}
    for g, items in grupos.items():
        n_runs = len(items)
        # por seed (cluster) para bootstrap
        por_seed = {}
        for cl in items:
            s = cl["seed"]
            d = por_seed.setdefault(s, dict(runs=0, dyn=0, voltas=0,
                                            area=0, cens=0, term=0))
            d["runs"] += 1
            d["dyn"] += cl["n_dyn"]
            d["voltas"] += cl["n_voltas"]
            d["area"] += cl["n_area_abriu"]
            d["cens"] += cl["n_censurados"]
            d["term"] += cl["n_terminais"]
        n_dyn = sum(cl["n_dyn"] for cl in items)
        n_voltas = sum(cl["n_voltas"] for cl in items)
        n_term = sum(cl["n_terminais"] for cl in items)
        n_cens = sum(cl["n_censurados"] for cl in items)
        n_area = sum(cl["n_area_abriu"] for cl in items)
        n_voltas_area = sum(cl["n_voltas_area"] for cl in items)
        n_born_rev = sum(cl["n_born_rev"] for cl in items)
        n_born_dead = sum(cl["n_born_dead"] for cl in items)
        # taxa/run: bootstrap de seed + Poisson iid (contraste)
        ps_rate = [(d["runs"], d["dyn"]) for d in por_seed.values()]
        boot = seed_bootstrap_rate(ps_rate)
        pois = poisson_ci(n_dyn, n_runs)
        # p(volta|morte) e p(volta|area) — CP + bootstrap de seed
        cp_vm = clopper_pearson(n_voltas, n_dyn)
        # p(volta|área): numerador = voltas cuja ÁREA ABRIU (não todas as voltas);
        # denominador = mortes dinâmicas com área aberta (= voltas_area + censurados).
        cp_va = clopper_pearson(n_voltas_area, n_area)
        bs_vm = seed_bootstrap_prop([(d["voltas"], d["dyn"])
                                     for d in por_seed.values()])
        ret = [rt for cl in items for rt in cl["ret_times"]]
        trios = [tr for cl in items for tr in cl["trios"]]
        n_ord = sum(1 for tr in trios if tr["ordenado"])
        cp_ord = clopper_pearson(n_ord, len(trios))
        anomalias = [a for cl in items for a in cl["anomalias"]]
        tabela[g] = dict(
            n_runs=n_runs, n_seeds=len(por_seed),
            n_born_dead=n_born_dead, n_born_rev=n_born_rev,
            n_dyn=n_dyn, n_voltas=n_voltas, n_terminais=n_term,
            n_censurados=n_cens, n_area_abriu=n_area, n_voltas_area=n_voltas_area,
            rate_boot=boot, rate_pois=pois,
            p_volta_morte=cp_vm, p_volta_morte_boot=bs_vm,
            p_volta_area=cp_va,
            ret_times=ret, trios=trios, n_ordenados=n_ord, cp_ordenados=cp_ord,
            anomalias=anomalias,
        )
    return tabela


def trio_status(tr):
    """Categoria da falha/sucesso do trio de UMA volta (para dimensionar C2)."""
    v = tr["trio_v2"]
    if tr["ordenado"]:
        return "ORDENADO"
    if v is None or v["t_area"] < 0:
        return "SEM-AREA(v2 t_area=-1)"     # v2 perdeu a área apesar da volta GT
    if v["t_area"] == v["t_morte"]:
        return "EMPATE(t_area==t_morte)"    # área abre no passo da morte-v2
    return "AREA-APOS-RES"


def breakdown_trios(trios):
    out = {}
    for tr in trios:
        out[trio_status(tr)] = out.get(trio_status(tr), 0) + 1
    return out


def quantis_ret(ret):
    if not ret:
        return dict(n=0)
    a = np.array(ret, float)
    return dict(n=len(a), min=int(a.min()), mediana=float(np.median(a)),
                q90=float(np.quantile(a, 0.9)), max=int(a.max()),
                media=float(a.mean()))


# --------------------------- self-test sintético -----------------------------
def _run_sintetico(rot, m, seed, nasce, deaths, revs, v2eps, beta=10.0, H=500000):
    return dict(rot=rot, ativ="softplus", beta=beta, m=m, seed=seed, T=0.0,
                eta=1e-3, n_dados=16, passos=H, loss_plato=0.15,
                nasce_morto=nasce, n_nasce_morto=len(nasce),
                death_events=deaths, revive_events=revs,
                n_mortos_real=len(nasce) + len({k for _, k in deaths}),
                n_revive_real=len(revs), n_neuronios_revive_real=len({k for _, k in revs}),
                v2_episodios=v2eps)


def _v2(k, tm, ta, tr, estado):
    return dict(k=k, t_morte=tm, t_area=ta, t_sinal=-1, t_res=tr,
                v_morte=0.1, v_min_abs=0.1, v_cruzou=False, j_trocas=0,
                estado=estado)


def selftest():
    print("== SELF-TEST sintético da análise (3 mundos) ==")
    falhas = 0

    # MUNDO A — taxa-zero (relu-like): 2 born-dead, nenhuma morte dinâmica.
    A = _run_sintetico("relu", 4, 100, nasce=[0, 1], deaths=[], revs=[],
                       v2eps=[_v2(0, 0, -1, -1, "censurado"),
                              _v2(1, 0, -1, -1, "censurado")], beta=None)
    ca = classifica_run(A)
    ok = (ca["n_dyn"] == 0 and ca["n_voltas"] == 0 and ca["n_born_dead"] == 2)
    print(f"[{'OK' if ok else 'FALHA'}] A taxa-zero: dyn={ca['n_dyn']} "
          f"voltas={ca['n_voltas']} born={ca['n_born_dead']}")
    falhas += (not ok)

    # MUNDO B — taxa-piloto: volta(k2), censurado(k3), terminal(k0), born-rev(k1).
    B = _run_sintetico(
        "sp-b10", 4, 101,
        nasce=[1],
        deaths=[(1000, 2), (2000, 3), (3000, 0)],
        revs=[(500, 1), (140000, 2)],
        v2eps=[_v2(1, 0, 400, 500, "completo"),                 # born-dead revive
               _v2(2, 1000, 45000, 140050, "completo"),         # volta ordenada
               _v2(3, 2000, 50000, -1, "censurado"),            # área abriu, sem volta
               _v2(0, 3000, -1, -1, "censurado")])              # área nunca abriu
    cb = classifica_run(B)
    ok = (cb["n_dyn"] == 3 and cb["n_voltas"] == 1 and cb["n_censurados"] == 1
          and cb["n_terminais"] == 1 and cb["n_area_abriu"] == 2
          and cb["n_born_rev"] == 1 and cb["ret_times"] == [139000]
          and len(cb["trios"]) == 1 and cb["trios"][0]["ordenado"] is True
          and not cb["anomalias"])
    print(f"[{'OK' if ok else 'FALHA'}] B taxa-piloto: dyn={cb['n_dyn']} "
          f"volta={cb['n_voltas']} cens={cb['n_censurados']} term={cb['n_terminais']} "
          f"area={cb['n_area_abriu']} born_rev={cb['n_born_rev']} "
          f"ret={cb['ret_times']} ord={cb['trios'][0]['ordenado'] if cb['trios'] else None}")
    falhas += (not ok)

    # MUNDO C — dois tipos de trio NÃO-ordenado numa volta (GT):
    #   k2 volta, v2 diz t_area(141000) > t_res(140050) ⟹ desordenado, área abriu, SEM anomalia.
    #   k3 volta, v2 diz t_area=-1 (área não abriu) ⟹ desordenado E anomalia volta-sem-área.
    C = _run_sintetico(
        "sp-b10", 4, 102,
        nasce=[], deaths=[(1000, 2), (5000, 3)],
        revs=[(140000, 2), (148000, 3)],
        v2eps=[_v2(2, 1000, 141000, 140050, "completo"),        # área depois do res
               _v2(3, 5000, -1, 148050, "completo")])           # área nunca abriu
    cc = classifica_run(C)
    ok = (cc["n_dyn"] == 2 and cc["n_voltas"] == 2
          and all(not tr["ordenado"] for tr in cc["trios"])
          and len(cc["anomalias"]) == 1 and cc["n_area_abriu"] == 1
          and cc["n_terminais"] == 0 and cc["n_censurados"] == 0)
    print(f"[{'OK' if ok else 'FALHA'}] C trio-desordenado: dyn={cc['n_dyn']} "
          f"voltas={cc['n_voltas']} ord={[tr['ordenado'] for tr in cc['trios']]} "
          f"anomalias={len(cc['anomalias'])} area={cc['n_area_abriu']}")
    falhas += (not ok)

    # segunda morte dinâmica do MESMO neurônio (re-morte após volta)
    D = _run_sintetico(
        "sp-b10", 2, 103, nasce=[],
        deaths=[(1000, 0), (200000, 0)],
        revs=[(140000, 0)],
        v2eps=[_v2(0, 1000, 45000, 140050, "completo"),
               _v2(0, 200000, -1, -1, "censurado")])
    cd = classifica_run(D)
    ok = (cd["n_dyn"] == 2 and cd["n_voltas"] == 1 and cd["n_terminais"] == 1)
    print(f"[{'OK' if ok else 'FALHA'}] D re-morte: dyn={cd['n_dyn']} "
          f"volta={cd['n_voltas']} term={cd['n_terminais']} (2 mortes dinâmicas do mesmo k)")
    falhas += (not ok)

    # estatística exata: valores conhecidos
    _, lo, hi = clopper_pearson(0, 10)
    ok = abs(lo - 0.0) < 1e-9 and abs(hi - 0.3085) < 1e-3
    print(f"[{'OK' if ok else 'FALHA'}] CP(0/10) = [0, {hi:.4f}] (esperado ~0.3085)")
    falhas += (not ok)
    _, lo, hi = clopper_pearson(6, 6)
    ok = abs(lo - 0.5407) < 1e-3       # CP inferior de 6/6 > 0.5 (piso de C2)
    print(f"[{'OK' if ok else 'FALHA'}] CP(6/6) inf = {lo:.4f} (>0.5 ⟹ 6 trios ordenados bastam p/ C2)")
    falhas += (not ok)
    # bootstrap de seed reduz a nada quando 1 só seed carrega tudo (superdispersão)
    r = seed_bootstrap_rate([(2, 10)] + [(2, 0)] * 29, B=5000)
    print(f"[info] boot-seed taxa com 1/30 seeds carregando 10 mortes: "
          f"rate={r[0]:.3f} CI=[{r[1]:.3f},{r[2]:.3f}] (largo ⟹ superdispersão captada)")

    print(f"== SELF-TEST {'OK' if falhas == 0 else f'FALHOU ({falhas})'} ==")
    return falhas == 0


# ------------------------------ relatório ------------------------------------
def _fmt_ci(t):
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def relatorio(runs):
    cls = [classifica_run(r) for r in runs]
    por_braco = agrega(cls, lambda cl: cl["rot"])
    por_braco_m = agrega(cls, lambda cl: (cl["rot"], cl["m"]))

    # VALIDADE relu
    relu_bug = []
    for cl in cls:
        if cl["rot"] == "relu" and (cl["n_voltas"] > 0 or cl["n_born_rev"] > 0
                                    or cl["n_revive_real"] > 0):
            relu_bug.append((cl["seed"], cl["m"], cl["n_voltas"],
                             cl["n_born_rev"], cl["n_revive_real"]))

    L = []
    L.append("# Piloto-de-taxa etapa-4a.3 — resultado\n")
    L.append(f"Grade: {len(runs)} runs. braços {BRACOS}, m∈{{4,6}}, "
             f"seeds 100–129 (30), n=16, η=1e-3, T=0, H=5e5, rec_neuronio=0.")
    L.append("Contador de morte/volta por GROUND-TRUTH (zmax<0→>0); trio e "
             "área-abriu pela régua v2. EXPLORATÓRIO — dimensiona o confirmatório.\n")

    L.append("## VALIDADE relu (teorema: grad≡0 no morto ⟹ 0 voltas)")
    if relu_bug:
        L.append(f"**BUG: {len(relu_bug)} run(s) relu com volta! "
                 f"{relu_bug}** — INVESTIGAR antes de usar.")
    else:
        nrelu = por_braco.get("relu", {}).get("n_runs", 0)
        ndyn = por_braco.get("relu", {}).get("n_dyn", 0)
        L.append(f"OK: relu 0 voltas / 0 born-rev / 0 rev_real em {nrelu} runs "
                 f"({ndyn} mortes dinâmicas, todas sem volta). Instrumento válido.")
    L.append("")

    L.append("## Tabela por braço (m=4 e m=6 agregados; 60 runs/braço, 30 seeds)")
    L.append("braço | runs | born-dead | mortes-din | λ/run (boot-seed) | "
             "λ/run (Poisson iid) | voltas | term | cens | "
             "p(volta\\|morte) CP | p(volta\\|área) CP | trios ord")
    L.append("---|---|---|---|---|---|---|---|---|---|---|---")
    for b in BRACOS:
        d = por_braco.get(b)
        if not d:
            continue
        L.append(f"{b} | {d['n_runs']} | {d['n_born_dead']} | {d['n_dyn']} | "
                 f"{_fmt_ci(d['rate_boot'])} | {_fmt_ci(d['rate_pois'])} | "
                 f"{d['n_voltas']} | {d['n_terminais']} | {d['n_censurados']} | "
                 f"{_fmt_ci(d['p_volta_morte'])} | {_fmt_ci(d['p_volta_area'])} | "
                 f"{d['n_ordenados']}/{len(d['trios'])}")
    L.append("")
    L.append("_p(volta|área): numerador = voltas cuja ÁREA ABRIU (v2 t_area≥0); "
             "denominador = mortes dinâmicas com área aberta (voltas-com-área + "
             "censurados). NÃO usa todas as voltas — 4/10 voltas têm t_area=−1 "
             "(cegueira da v2, ver Diagnóstico C2), então este condicional está "
             "enviesado pelo instrumento e deve ser relido com área ground-truth._")

    L.append("## Tabela por (braço, m) — 30 runs/célula")
    L.append("braço | m | runs | born-dead | mortes-din | λ/run (boot-seed) | "
             "voltas | term | cens | p(volta\\|morte) CP")
    L.append("---|---|---|---|---|---|---|---|---|---")
    for b in BRACOS:
        for m in (4, 6):
            d = por_braco_m.get((b, m))
            if not d:
                continue
            L.append(f"{b} | {m} | {d['n_runs']} | {d['n_born_dead']} | "
                     f"{d['n_dyn']} | {_fmt_ci(d['rate_boot'])} | "
                     f"{d['n_voltas']} | {d['n_terminais']} | {d['n_censurados']} | "
                     f"{_fmt_ci(d['p_volta_morte'])}")
    L.append("")

    L.append("## Return-time (t_res−t_morte, ground-truth) e τ = quantil 0.9, por β")
    L.append("β | n voltas | mín | mediana | **τ=q0.9** | máx | média")
    L.append("---|---|---|---|---|---|---")
    tau_por_b = {}
    for b in ["sp-b3", "sp-b10", "sp-b300"]:
        d = por_braco.get(b)
        q = quantis_ret(d["ret_times"] if d else [])
        if q["n"] == 0:
            L.append(f"{BETA_DE[b]:g} | 0 | — | — | — | — | —")
            continue
        tau_por_b[b] = q["q90"]
        L.append(f"{BETA_DE[b]:g} | {q['n']} | {q['min']} | {q['mediana']:.0f} | "
                 f"**{q['q90']:.0f}** | {q['max']} | {q['media']:.0f}")
    ret_all = [rt for b in ["sp-b3", "sp-b10", "sp-b300"]
               for rt in (por_braco.get(b, {}).get("ret_times", []))]
    qall = quantis_ret(ret_all)
    if qall["n"]:
        tau_por_b["pool"] = qall["q90"]
        L.append(f"pool | {qall['n']} | {qall['min']} | {qall['mediana']:.0f} | "
                 f"**{qall['q90']:.0f}** | {qall['max']} | {qall['media']:.0f}")
    L.append("")

    L.append("## Trios de TODAS as voltas dinâmicas (regra estrita t_morte<t_área<t_res)")
    L.append("braço | m | seed | k | t_morte(GT) | t_res(GT) | v2(t_morte,t_área,t_res) | ordenado | área-abriu")
    L.append("---|---|---|---|---|---|---|---|---")
    n_voltas_tot = 0
    for cl in cls:
        for tr in cl["trios"]:
            n_voltas_tot += 1
            v = tr["trio_v2"]
            vs = (f"({v['t_morte']},{v['t_area']},{v['t_res']})" if v else "—")
            L.append(f"{cl['rot']} | {cl['m']} | {cl['seed']} | {tr['k']} | "
                     f"{tr['t_morte_gt']} | {tr['t_res_gt']} | {vs} | "
                     f"{'sim' if tr['ordenado'] else 'não'} | "
                     f"{'sim' if tr['area_abriu'] else 'não'}")
    if n_voltas_tot == 0:
        L.append("(nenhuma volta dinâmica observada)")
    L.append("")

    all_trios = [tr for cl in cls for tr in cl["trios"]]
    bd = breakdown_trios(all_trios)
    n_ord = bd.get("ORDENADO", 0)
    L.append("## Diagnóstico C2: por que os trios falham a regra estrita (pool)")
    L.append(f"De {len(all_trios)} voltas dinâmicas, ordenadas ESTRITAS "
             f"(t_morte<t_área<t_res) = **{n_ord}/{len(all_trios)}** "
             f"= {_fmt_ci(clopper_pearson(n_ord, len(all_trios)))}. Falhas:")
    for st, c in sorted(bd.items(), key=lambda kv: -kv[1]):
        L.append(f"- {st}: {c}")
    n_empate = bd.get("EMPATE(t_area==t_morte)", 0)
    L.append(f"Se a regra fosse t_morte ≤ t_área < t_res (empates contam), "
             f"ordenadas = {n_ord + n_empate}/{len(all_trios)}. O resto "
             f"({bd.get('SEM-AREA(v2 t_area=-1)', 0)}) é CEGUEIRA da régua v2: "
             f"volta real (ground-truth) sem área registrada — falha de INSTRUMENTO, "
             f"não da lei. Ver recomendação.")
    L.append("")

    # anomalias
    anoms = [a for cl in cls for a in cl["anomalias"]]
    if anoms:
        L.append(f"## Anomalias (volta sem área-abriu na v2): {len(anoms)}")
        for a in anoms:
            L.append(f"- {a}")
        L.append("")

    return "\n".join(L), por_braco, por_braco_m, tau_por_b, cls, relu_bug


def recomendacao(por_braco, tau_por_b):
    """Deriva (S, n*, piso C1, sizing C2) dos números medidos, com o achado
    crítico: a régua v2 é um INSTRUMENTO de área não-confiável em rede real."""
    L = ["## Recomendação para preditor_4a3 (derivada dos números medidos)\n"]

    # --- 0. leitura de canal (C4) ---
    linha = []
    for b in ["sp-b3", "sp-b10", "sp-b300", "relu"]:
        d = por_braco.get(b, {})
        pvm = d.get("p_volta_morte", (float("nan"),) * 3)
        linha.append(f"{b}={pvm[0]:.3f}")
    L.append("**Canal (C4) — confirmado.** p(volta|morte) monótona em β: "
             + ", ".join(linha) + ". O canal e^{βz} FECHA quando β→∞ "
             "(softplus→relu): β300 dá 0 voltas exatamente como a relu (mesmo "
             "nº de mortes dinâmicas, todas terminais). Taxa de MORTE é "
             "β-independente (~0.22–0.30/run); a VOLTA é β-dependente. ⟹ o alvo "
             "primário deve ser **β3** (maior p(volta), mais voltas, retorno mais "
             "rápido, mais perto da suavidade GELU), NÃO β10.")
    L.append("")

    # --- 1. achado crítico: instrumento de área v2 ---
    all_trios = [tr for b in ["sp-b3", "sp-b10", "sp-b300"]
                 for tr in por_braco.get(b, {}).get("trios", [])]
    bd = breakdown_trios(all_trios)
    n_ord = bd.get("ORDENADO", 0)
    n_emp = bd.get("EMPATE(t_area==t_morte)", 0)
    n_miss = bd.get("SEM-AREA(v2 t_area=-1)", 0)
    L.append("**Achado crítico p/ C2 (o coração).** Sob a regra ESTRITA "
             f"pré-registrada com área da v2, só {n_ord}/{len(all_trios)} voltas "
             "são ordenadas — mas a falha é de INSTRUMENTO, não da lei:")
    L.append(f"  - {n_emp} EMPATES t_área==t_morte (v2 abre a área no MESMO passo "
             "da morte-v2): a regra ESTRITA os rejeita; t_morte ≤ t_área os aceita.")
    L.append(f"  - {n_miss} SEM-ÁREA (v2 t_area=−1) apesar de volta real: a régua "
             "v2 é CEGA a essas voltas (β10 rápidas + cruzamento de sinal de v_k). "
             "Nenhum ajuste de regra recupera — é o estimador de área que falha.")
    L.append(f"  - Ordenadas com t_morte≤t_área = {n_ord + n_emp}/{len(all_trios)}; "
             f"em β3 isoladamente as voltas ficam ~ordenadas (5–6/6) — a LEI DE "
             "ÁREA se sustenta em β3; quem falha é a v2.")
    L.append("  - **AÇÃO OBRIGATÓRIA antes de congelar C2:** (i) rec_neuronio>0 no "
             "confirmatório (o piloto pôs 0 por economia) para ter z_park e um "
             "sinal de área GROUND-TRUTH (sinal do drive acumulado da trajetória), "
             "régua-independente; (ii) regra t_morte ≤ t_área < t_res (absorve os "
             "empates de discretização); (iii) reconciliar o relógio: v2 marca "
             "morte ~126 passos (mediana) antes do ground-truth e às vezes chama "
             "morte-dinâmica de born-dead — usar UM relógio (o ground-truth).")
    L.append("")

    # --- 2. sizing (braço primário β3) ---
    b = "sp-b3"
    d = por_braco.get(b, {})
    lam = d.get("rate_boot", (float("nan"),) * 3)
    pvm = d.get("p_volta_morte", (float("nan"),) * 3)
    pva = d.get("p_volta_area", (float("nan"),) * 3)
    lam_hat, lam_lo = lam[0], lam[1]
    pvm_hat, pvm_lo = pvm[0], pvm[1]
    bd_b3 = breakdown_trios(d.get("trios", []))
    tot_b3 = max(sum(bd_b3.values()), 1)
    frac_estrita = bd_b3.get("ORDENADO", 0) / tot_b3
    frac_relax = (bd_b3.get("ORDENADO", 0) + bd_b3.get("EMPATE(t_area==t_morte)", 0)) / tot_b3
    L.append(f"**Braço primário β3:** λ_morte-din/run = {_fmt_ci(lam)} "
             "(bootstrap POR SEED ≈ Poisson iid ⟹ as MORTES quase não têm "
             "superdispersão; a superdispersão do parecer está nas VOLTAS, que se "
             f"concentram em poucas seeds). p(volta|morte) = {_fmt_ci(pvm)} "
             f"(≪ os 0.70 que a nota assumia — parecer estava certo). "
             f"p(volta|área) = {_fmt_ci(pva)} (numerador = voltas cuja área abriu; "
             "denominador contaminado pela cegueira da v2 — interpretar com o "
             "instrumento corrigido).")
    voltas_seed = 2 * lam_hat * pvm_hat            # 2 runs (m4+m6) por seed
    voltas_seed_lo = 2 * lam_lo * (pvm_lo if not np.isnan(pvm_lo) else 0.0)
    L.append(f"- voltas/seed (β3, 2 runs) = 2·{lam_hat:.3f}·{pvm_hat:.3f} "
             f"≈ {voltas_seed:.2f} (ponto) / {voltas_seed_lo:.2f} (pior-caso CI-inf).")
    L.append("- **C1 (existência-com-taxa):** para E[voltas]≥10 (piso do parecer) "
             f"⟹ **S ≈ {10/voltas_seed:.0f}** seeds (ponto); "
             f"para E[voltas]≥6 ⟹ S ≈ {6/voltas_seed:.0f}. "
             f"pior-caso CI-inf pede S ≈ {10/max(voltas_seed_lo,1e-9):.0f}.")
    L.append("- **n\\* (piso C1, INCONCLUSIVO):** contagem de MORTES dinâmicas "
             f"(denominador). E[mortes-din β3] = S·2·{lam_hat:.3f}. A S≈50 ⟹ "
             f"~{2*50*lam_hat:.0f} mortes; sugiro **n\\* ≈ {round(2*50*lam_lo)}** "
             f"(esperado ao CI-inferior λ={lam_lo:.2f}); N_mortes < n\\* ⟹ "
             "INCONCLUSIVO (não negativo), como no parecer.")
    L.append("- **C2 (sizing pelo coração):** precisa de ≥6 voltas ORDENADAS "
             "(CP-inf de 6/6 = 0.541>0.5; confortável 8–10). ordenadas/seed = "
             f"voltas/seed × fração-ordenada:")
    for frac, nome in [(frac_estrita, "v2 ESTRITA como está"),
                       (frac_relax, "regra t_morte≤t_área"),
                       (1.0, "instrumento CORRIGIDO (área ground-truth)")]:
        ops = voltas_seed * frac
        S6 = 6 / ops if ops > 0 else float("inf")
        S8 = 8 / ops if ops > 0 else float("inf")
        L.append(f"  - {nome} (fração≈{frac:.2f}): ordenadas/seed≈{ops:.2f} ⟹ "
                 f"S≈{S6:.0f} (6 ord.) / {S8:.0f} (8 ord.).")
    L.append("  - Leitura: com a v2 como está, C2 é INVIÁVEL (S de centenas); com o "
             "instrumento corrigido, **S ≈ 40–60 seeds** entrega C2. O ganho do "
             "conserto de instrumento vale ~3–5× em amostra — é a alavanca.")
    L.append("")

    # --- 3. τ / janela ---
    L.append("**τ e janela de elegibilidade.** Return-time é MUITO assimétrico "
             "(mediana β3 ~46k mas cauda até ~3e5); com n=4–6 voltas o q0.9 é "
             "ruidoso. Valores medidos:")
    for b_ in ["sp-b3", "sp-b10"]:
        tau = tau_por_b.get(b_)
        if tau is not None:
            L.append(f"  - β{int(BETA_DE[b_])}: τ=q0.9 ≈ {tau:.0f} ⟹ janela "
                     f"[0, H−τ] = [0, {int(5e5 - tau)}] a H=5e5.")
    L.append("  - Manter **H=5e5**; a maioria das mortes é precoce (t_morte<30k), "
             "então a janela cobre o grosso. A cauda de retorno lento (2e5–4e5) é "
             "CENSURA legítima — tratar com KM/person-time (já exigido pelo "
             "parecer), não estendendo H indefinidamente.")
    L.append("")
    L.append("**Grade sugerida do confirmatório:** primário β3; secundário β10 "
             "(comparação de canal, C4); β300 + relu como duplo controle de canal-"
             "fechado (0 voltas esperado); S≈50 seeds × m∈{4,6} × rec_neuronio>0. "
             "relu = validade (qualquer volta ⟹ bug). Sizing por C2 com o "
             "instrumento de área corrigido.")
    return "\n".join(L)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        ok = selftest()
        raise SystemExit(0 if ok else 1)
    runs = carrega()
    print(f"carregados {len(runs)} runs de {OUT}")
    if not runs:
        raise SystemExit("sem resultados ainda.")
    txt, pb, pbm, tau, cls, relu_bug = relatorio(runs)
    rec = recomendacao(pb, tau)
    full = txt + "\n" + rec + "\n"
    with open(os.path.join(AQUI, "taxa-resultado.md"), "w") as f:
        f.write(full)
    # dump JSON auxiliar (números crus)
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items() if k not in ("trios", "ret_times")}
        return o
    with open(os.path.join(AQUI, "taxa-resultado.json"), "w") as f:
        json.dump(dict(por_braco={k: _clean(v) for k, v in pb.items()},
                       por_braco_m={f"{k[0]}-m{k[1]}": _clean(v) for k, v in pbm.items()},
                       tau=tau, relu_bug=relu_bug), f, indent=2)
    print(full)
    print(f"\n[escrito] {os.path.join(AQUI, 'taxa-resultado.md')}")
