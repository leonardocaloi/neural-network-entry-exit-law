"""CONFIRMATORIO H-BARREIRA — ANALISE COM DECISOES CONGELADAS.

Este arquivo e parte do PRE-REGISTRO: escrito e validado sinteticamente ANTES
de qualquer dado real existir. Toda regra de decisao esta em `decide()`; nenhuma
depende do dado observado. Qualquer mudanca aqui depois da primeira leitura de
dado real vai para a secao "desvios" do README-hbarreira.md.

    python analise_hbarreira.py sintetico   # 3 mundos: a=1 SUSTENTA, a=0.7 MATA,
                                            #           sem lei INCONCLUSIVO
    python analise_hbarreira.py real        # le results/hbarreira_runs.jsonl

CONVENCAO DE SINAL (fixada aqui, uma vez): a variavel ajustada e a PROFUNDIDADE
da barreira, y := -beta*z_park > 0. O piloto reportou beta*z_park = -0.94*ln(1/eps)
- 6.1; nesta convencao isso e y = +0.94*ln(1/eps) + 6.1. O mecanismo (o canal
e^{beta z} cruza o piso eps) prediz a = +1 EXATO e b > 0.

Modelo (pre-registrado):
    y(i, eps) = -beta * z_park(i, eps) = a * ln(1/eps) + b + u_i + ruido
OLS empilhado sobre os pontos (instancia i, eps); a incerteza vem de bootstrap
de CLUSTER (reamostra INSTANCIAS inteiras, com todos os seus eps), que e o que
respeita o efeito de instancia u_i.
"""
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(AQUI, "results")
sys.path.insert(0, AQUI)
from preditor_hbarreira import CFG  # noqa: E402  (fonte unica dos criterios)

A_LO, A_HI = CFG["A_LO"], CFG["A_HI"]
BOOT, BSEED = CFG["BOOT"], CFG["BOOT_SEED"]
MINVAL = CFG["MIN_CELULAS_VALIDAS"]
# curvatura: se o termo quadratico deslocar o ajuste em mais de CURV_TOL
# unidades de beta*z ao longo da grade E seu IC95 excluir 0, a LEI LINEAR e
# inadequada e o resultado primario e INCONCLUSIVO-CURVATURA.
CURV_TOL = 1.0


# --------------------------------------------------------------- estimadores
def ajusta(L, y, ordem=1):
    """OLS de y sobre [1, L] (ordem 1) ou [1, L, L^2]. Devolve (a, b) ou (a,b,q)."""
    A = np.vander(L, ordem + 1)          # colunas: L^ordem ... 1
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    if ordem == 1:
        return c[0], c[1]
    return c[1], c[2], c[0]              # (a, b, q)


def clusters(inst, L, y):
    ids = np.unique(inst)
    return ids, [np.nonzero(inst == i)[0] for i in ids]


def boot_ajuste(inst, L, y, ordem=1, seed=BSEED, B=BOOT):
    """Bootstrap de CLUSTER (instancias inteiras). Devolve array (B, k)."""
    ids, idx = clusters(inst, L, y)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(ids), len(ids))
        sel = np.concatenate([idx[j] for j in pick])
        try:
            out.append(ajusta(L[sel], y[sel], ordem))
        except np.linalg.LinAlgError:
            continue
    return np.array(out)


def ic95(v):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def boot_pareado(d1, d2, fn, seed=BSEED + 7, B=BOOT):
    """Diferenca entre duas celulas reamostrando as MESMAS instancias nas duas
    (as celulas compartilham o conjunto de instancias por construcao)."""
    comuns = np.intersect1d(np.unique(d1["inst"]), np.unique(d2["inst"]))
    rng = np.random.default_rng(seed)
    i1 = {i: np.nonzero(d1["inst"] == i)[0] for i in comuns}
    i2 = {i: np.nonzero(d2["inst"] == i)[0] for i in comuns}
    out = []
    for _ in range(B):
        pick = comuns[rng.integers(0, len(comuns), len(comuns))]
        s1 = np.concatenate([i1[i] for i in pick])
        s2 = np.concatenate([i2[i] for i in pick])
        out.append(fn(ajusta(d1["L"][s1], d1["y"][s1]),
                      ajusta(d2["L"][s2], d2["y"][s2])))
    return np.array(out)


# ------------------------------------------------------------ DECISAO (congelada)
def decide(dados, gd=None):
    """dados: {nome_celula: {'inst':.., 'L':.., 'y':.., 'n_tot':int}}.

    Regras, TODAS pre-registradas (README-hbarreira.md §3):
      P-NULL de instrumento: se a fracao de celulas (instancia x eps) com plato
        valido na ANCORA for < 60%, nada e decidido (INSTRUMENTO-NULL).
      PRIMARIA H-BARREIRA (celula 'ancora'):
        IC95(a) contido em [0.9,1.1]      -> SUSTENTA
        IC95(a) disjunto de [0.9,1.1]     -> MATA
        caso contrario                    -> INCONCLUSIVO
        guarda de curvatura: |q|*Lmax^2 > 1.0 com IC95(q) excluindo 0
                                          -> INCONCLUSIVO-CURVATURA (sobrepoe)
      SECUNDARIA (i) invariancia em beta: IC95(a_1e3 - a_1e4) pareado;
        contem 0 ou |delta|<0.1 -> INVARIANTE; senao -> NAO-INVARIANTE.
      SECUNDARIA (ii) memoria de v: EQUIVALENCIA. a nao muda com b2 (pareado,
        tolerancia 0.1) E IC95(b(0.9)-b(0.999)) cabe em [-2,+2] -> MECANISMO-V
        SUSTENTADO; se sair -> REFUTADO; se o IC for mais largo que +-1 ->
        SEM POTENCIA (nao decide). Ver README §4 para a derivacao do 0.07.
      SECUNDARIA (iii) ancora GD (eps -> infinito efetivo): Adam parka MAIS
        FUNDO que GD em todo eps (mediana y_Adam(eps) > y_GD nos 5 eps).
    """
    out = {}
    anc = dados["ancora"]
    frac = len(anc["y"]) / max(anc["n_tot"], 1)
    out["frac_plato_ancora"] = frac
    if frac < MINVAL:
        out["veredito"] = "INSTRUMENTO-NULL"
        out["motivo"] = f"so {frac:.0%} das celulas com plato valido (< {MINVAL:.0%})"
        return out

    a, b = ajusta(anc["L"], anc["y"])
    bs = boot_ajuste(anc["inst"], anc["L"], anc["y"], 1)
    lo, hi = ic95(bs[:, 0])
    out["a"], out["b"], out["ic95_a"] = float(a), float(b), (lo, hi)
    out["ic95_b"] = ic95(bs[:, 1])
    if lo >= A_LO and hi <= A_HI:
        v = "SUSTENTA"
    elif hi < A_LO or lo > A_HI:
        v = "MATA"
    else:
        v = "INCONCLUSIVO"

    aq, bq, q = ajusta(anc["L"], anc["y"], 2)
    bsq = boot_ajuste(anc["inst"], anc["L"], anc["y"], 2)
    qlo, qhi = ic95(bsq[:, 2])
    Lmax = float(anc["L"].max())
    out["q"], out["ic95_q"], out["curv_efeito"] = float(q), (qlo, qhi), float(abs(q) * Lmax ** 2)
    if abs(q) * Lmax ** 2 > CURV_TOL and not (qlo <= 0 <= qhi):
        v = "INCONCLUSIVO-CURVATURA"
    out["veredito"] = v

    # --- (i) invariancia em beta
    if "beta-baixo" in dados:
        d = boot_pareado(dados["beta-baixo"], anc, lambda p1, p2: p1[0] - p2[0])
        dlo, dhi = ic95(d)
        da = ajusta(dados["beta-baixo"]["L"], dados["beta-baixo"]["y"])[0] - a
        out["sec_i"] = dict(delta_a=float(da), ic95=(dlo, dhi),
                            veredito="INVARIANTE" if (dlo <= 0 <= dhi or abs(da) < 0.1)
                            else "NAO-INVARIANTE")
    # --- (ii) memoria de v
    cels = [("b2-090", 0.9), ("b2-099", 0.99), ("ancora", 0.999)]
    if all(c in dados for c, _ in cels):
        bb = {b2: ajusta(dados[c]["L"], dados[c]["y"]) for c, b2 in cels}
        d_a = boot_pareado(dados["b2-090"], anc, lambda p1, p2: p1[0] - p2[0])
        d_b = boot_pareado(dados["b2-090"], anc, lambda p1, p2: p1[1] - p2[1])
        alo, ahi = ic95(d_a)
        blo, bhi = ic95(d_b)
        # PREDICAO (README §4): a nao muda com b2; b muda MUITO POUCO
        # (deslocamento de defasagem estimado a priori em ~0.07 unidades de
        # beta*z, uma ordem abaixo do residuo). Logo o teste e de EQUIVALENCIA:
        # sustenta se IC95(delta_b) cabe em [-DB2_TOL, +DB2_TOL] e a nao muda.
        a_ok = (alo <= 0 <= ahi) or abs(bb[0.9][0] - bb[0.999][0]) < 0.1
        b_ok = (blo >= -CFG["DB2_TOL"] and bhi <= CFG["DB2_TOL"])
        pot = (bhi - blo) / 2 <= CFG["DB2_POT"]
        ver = ("SEM POTENCIA" if not pot else
               "MECANISMO-V SUSTENTADO" if (a_ok and b_ok) else "MECANISMO-V REFUTADO")
        out["sec_ii"] = dict(a_por_b2={k: float(v[0]) for k, v in bb.items()},
                             b_por_b2={k: float(v[1]) for k, v in bb.items()},
                             ordem_b=bool(bb[0.9][1] > bb[0.99][1] > bb[0.999][1]),
                             ic95_delta_a=(alo, ahi), ic95_delta_b=(blo, bhi),
                             veredito=ver)
    # --- (iii) ancora GD
    if gd is not None and len(gd):
        med_gd = float(np.median(gd))          # ja em PROFUNDIDADE (y = -beta*z_park)
        fundo = [float(np.median(anc["y"][anc["L"] == Lv])) > med_gd
                 for Lv in np.unique(anc["L"])]
        out["sec_iii"] = dict(prof_gd=med_gd, adam_mais_fundo_em_todo_eps=bool(all(fundo)),
                              veredito="ANCORA COERENTE" if all(fundo) else "ANCORA VIOLADA")
    return out


# ------------------------------------------------------------------ carga
def carrega(path=None):
    path = path or os.path.join(RES, "hbarreira_runs.jsonl")
    regs = [json.loads(l) for l in open(path)]
    dados, gd = {}, []
    for r in regs:
        if not r["plato"] or r["n_mortes"] != 1:
            continue
        if r["opt"] == "gd":
            if r["beta"] == 1e4:
                gd.append(-r["bz_park"])
            continue
        dados.setdefault(r["celula"], []).append(r)
    tot = {}
    for r in regs:
        if r["opt"] == "adam":
            tot[r["celula"]] = tot.get(r["celula"], 0) + 1
    d = {}
    for c, rs in dados.items():
        d[c] = dict(inst=np.array([r["seed"] for r in rs]),
                    L=np.array([np.log(1.0 / r["eps"]) for r in rs]),
                    y=np.array([-r["bz_park"] for r in rs]),   # PROFUNDIDADE
                    n_tot=tot[c])
    return d, np.array(gd)


# ------------------------------------------------------------- mundo sintetico
def mundo(a_true, seed, n=100, sd_ruido=0.4, sd_inst=3.0, b_true=6.0,
          sem_lei=False, b2_shift=None, q_true=0.0, u_seed=777):
    # u_seed FIXO entre celulas: no experimento real as celulas compartilham as
    # MESMAS instancias, entao o efeito de instancia u_i e comum (e o bootstrap
    # pareado tira ele). O sintetico tem de imitar isso, senao a variancia de
    # delta_b fica inflada e o teste secundario parece menos potente do que e.
    rng = np.random.default_rng(seed)
    eps = np.array(CFG["EPS_GRID"])
    L = np.log(1.0 / eps)
    ids = np.arange(n)
    u = np.random.default_rng(u_seed).normal(0, sd_inst, n)
    inst, Ls, ys = [], [], []
    for i in ids:
        for Lv in L:
            mu = (b_true + u[i] if sem_lei
                  else a_true * Lv + b_true + q_true * Lv ** 2 + u[i])
            if b2_shift:
                mu += b2_shift
            inst.append(i)
            Ls.append(Lv)
            ys.append(mu + rng.normal(0, sd_ruido))
    return dict(inst=np.array(inst), L=np.array(Ls), y=np.array(ys), n_tot=n * len(L))


def sintetico():
    """Os DOIS sentidos (e o terceiro): o teste tem de SUSTENTAR quando a=1,
    MATAR quando a=0.7 e ficar INCONCLUSIVO quando nao ha lei."""
    casos = [
        ("mundo a=1 (mecanismo verdadeiro)", dict(a_true=1.0, seed=1), "SUSTENTA"),
        ("mundo a=0.94 (a do piloto)", dict(a_true=0.94, seed=2), "SUSTENTA"),
        ("mundo a=0.7 (mecanismo falso)", dict(a_true=0.7, seed=3), "MATA"),
        ("mundo a=1.4", dict(a_true=1.4, seed=4), "MATA"),
        ("mundo SEM LEI (z_park nao depende de eps)", dict(a_true=0.0, seed=5, sem_lei=True), "MATA"),
        ("mundo a=0.85, ruido grande, n=6 (indeciso)",
         dict(a_true=0.85, seed=6, n=6, sd_ruido=25.0), "INCONCLUSIVO"),
        ("mundo com CURVATURA (lei nao-linear)",
         dict(a_true=1.0, seed=7, q_true=0.06), "INCONCLUSIVO-CURVATURA"),
    ]
    falhas = 0
    for nome, kw, esperado in casos:
        d = mundo(**kw)
        # celulas secundarias: beta-baixo com o MESMO a; b2 menores => b menos negativo
        dados = {"ancora": d,
                 "beta-baixo": mundo(**{**kw, "seed": kw["seed"] + 100}),
                 "b2-099": mundo(**{**kw, "seed": kw["seed"] + 200, "b2_shift": 0.04}),
                 "b2-090": mundo(**{**kw, "seed": kw["seed"] + 300, "b2_shift": 0.07})}
        gd = np.random.default_rng(kw["seed"]).normal(5.0, 0.5, kw.get("n", 100))
        r = decide(dados, gd)
        ok = r["veredito"] == esperado
        falhas += (not ok)
        print(f"{'OK ' if ok else 'FALHA'} {nome:45s} -> {r['veredito']:24s} "
              f"a={r.get('a', float('nan')):+.3f} IC95={tuple(round(x,3) for x in r.get('ic95_a',(0,0)))} "
              f"(esperado {esperado})")
        if nome.startswith("mundo a=1 "):
            print(f"      sec(i)  {r['sec_i']['veredito']}  delta_a={r['sec_i']['delta_a']:+.3f}")
            print(f"      sec(ii) {r['sec_ii']['veredito']}  b={r['sec_ii']['b_por_b2']}")
            print(f"      sec(iii){r['sec_iii']['veredito']}  prof_gd={r['sec_iii']['prof_gd']:.2f}")
    # controles das secundarias, nos DOIS sentidos
    base = dict(a_true=1.0, seed=11)
    for nome, kw_bb, kw_alt, esperado in [
        ("sec(ii) mundo com b2 IRRELEVANTE (mecanismo eps)", 0.07, None, "MECANISMO-V SUSTENTADO"),
        ("sec(ii) mundo com b2 DOMINANTE (mecanismo v)", 8.0, None, "MECANISMO-V REFUTADO"),
    ]:
        dados = {"ancora": mundo(**base),
                 "b2-099": mundo(**{**base, "seed": 12, "b2_shift": kw_bb / 2}),
                 "b2-090": mundo(**{**base, "seed": 13, "b2_shift": kw_bb})}
        dados["beta-baixo"] = mundo(**{**base, "seed": 14})
        r = decide(dados)
        ok = r["sec_ii"]["veredito"] == esperado
        falhas += (not ok)
        print(f"{'OK ' if ok else 'FALHA'} {nome:45s} -> {r['sec_ii']['veredito']:24s} "
              f"IC95(db)={tuple(round(x,2) for x in r['sec_ii']['ic95_delta_b'])} "
              f"(esperado {esperado})")
    for nome, da, esperado in [("sec(i) beta INVARIANTE", 0.0, "INVARIANTE"),
                               ("sec(i) beta NAO-INVARIANTE", 0.35, "NAO-INVARIANTE")]:
        dados = {"ancora": mundo(a_true=1.0, seed=21),
                 "beta-baixo": mundo(a_true=1.0 + da, seed=22)}
        r = decide(dados)
        ok = r["sec_i"]["veredito"] == esperado
        falhas += (not ok)
        print(f"{'OK ' if ok else 'FALHA'} {nome:45s} -> {r['sec_i']['veredito']:24s} "
              f"delta_a={r['sec_i']['delta_a']:+.3f} (esperado {esperado})")
    for nome, gdv, esperado in [("sec(iii) Adam mais fundo que GD", 5.0, "ANCORA COERENTE"),
                                ("sec(iii) GD mais fundo que Adam", 40.0, "ANCORA VIOLADA")]:
        r = decide({"ancora": mundo(a_true=1.0, seed=31)},
                   np.full(100, gdv))
        ok = r["sec_iii"]["veredito"] == esperado
        falhas += (not ok)
        print(f"{'OK ' if ok else 'FALHA'} {nome:45s} -> {r['sec_iii']['veredito']:24s} "
              f"(esperado {esperado})")
    # controle do P-NULL de instrumento
    d = mundo(a_true=1.0, seed=9)
    d["n_tot"] = int(len(d["y"]) / 0.4)     # so 40% das celulas teriam plato
    r = decide({"ancora": d})
    ok = r["veredito"] == "INSTRUMENTO-NULL"
    falhas += (not ok)
    print(f"{'OK ' if ok else 'FALHA'} {'controle P-NULL de instrumento':45s} -> {r['veredito']}")
    print("\nSINTETICO: " + ("TUDO PASSA" if falhas == 0 else f"{falhas} FALHAS"))
    return falhas


def real():
    d, gd = carrega()
    r = decide(d, gd)
    print(json.dumps(r, indent=1, default=float))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sintetico"
    raise SystemExit(sintetico() if cmd == "sintetico" else real())
