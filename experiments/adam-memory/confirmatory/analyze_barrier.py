"""CONFIRMATORIO H-BARREIRA — ANALISE COM DECISOES CONGELADAS.

Este arquivo e parte do PRE-REGISTRO: escrito e validado sinteticamente ANTES
de qualquer dado real existir. Toda regra de decisao esta em `decide()`; nenhuma
depende do dado observado. Qualquer mudanca aqui depois da primeira leitura de
dado real vai para a secao "desvios" do PROTOCOL.md.

    python analyze_barrier.py sintetico   # mundos de controle, nos DOIS sentidos
    python analyze_barrier.py real        # le results/hbarreira_runs.jsonl

SUBCOMANDO OBRIGATORIO (17o parecer, forte 11): sem argumento, ou com argumento
desconhecido, o script ERRA. Antes o default era `sintetico` e o ramo `else` caia
em `real` -- um typo viraria um PEEK nao declarado no dado do confirmatorio.

CONVENCAO DE SINAL (fixada aqui, uma vez): a variavel ajustada e a PROFUNDIDADE
da barreira, y := -beta*z_park > 0. O piloto reportou beta*z_park = -0.94*ln(1/eps)
- 6.1; nesta convencao isso e y = +0.94*ln(1/eps) + 6.1.

Modelo (pre-registrado):
    y(i, eps) = -beta * z_park(i, eps) = a * ln(1/eps) + b + u_i + ruido
OLS empilhado sobre os pontos (instancia i, eps); a incerteza vem de bootstrap
de CLUSTER (reamostra INSTANCIAS inteiras, com todos os seus eps), que e o que
respeita o efeito de instancia u_i.

DESENHO DEPOIS DO 17o PARECER (adjudicacao (c) restrita) -- ver review-17.md:
  * PRIMARIO UNICO: `a` sobre y BRUTO, grade PRIMARIA sem eps=1, janela
    [0.90,1.10] congelada. NAO ha primario' sobre y* (isso era dupla chance).
  * eps=1 e GD: rodados e reportados como PONTOS DE SATURACAO da secundaria
    (iii). Nunca entram no ajuste primario.
  * y* = y - ln(lr*K_morto): vira DIAGNOSTICO de decomposicao
        a = 1 + dln(lr)/dL + dln(K)/dL + resto
    com o RESTO como teste novo (tolerancia RESTO_TOL derivada ANTES).
    Alerta, NAO morte -- a fisica do resto nao esta derivada.
  * Secundarias (i) e (ii) comparam celulas na GRADE DESCONTADA (y*), porque
    delta_a/delta_b sobre y bruto misturam a recalibracao de lr entre celulas.
"""
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(AQUI, "results")
sys.path.insert(0, AQUI)
from predict_barrier import CFG  # noqa: E402  (fonte unica dos criterios)

A_LO, A_HI = CFG["A_LO"], CFG["A_HI"]
BOOT, BSEED = CFG["BOOT"], CFG["BOOT_SEED"]
MINVAL = CFG["MIN_CELULAS_VALIDAS"]
CURV_TOL = CFG["CURV_TOL"]      # agora vive no CFG => entra no sha256 do config
RESTO_TOL = CFG["RESTO_TOL"]


# --------------------------------------------------------------- estimadores
def ajusta(L, y, ordem=1):
    """OLS de y sobre [1, L] (ordem 1) ou [1, L, L^2]. Devolve (a, b) ou (a,b,q)."""
    A = np.vander(L, ordem + 1)          # colunas: L^ordem ... 1
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    if ordem == 1:
        return c[0], c[1]
    return c[1], c[2], c[0]              # (a, b, q)


def ajusta_fe(inst, L, y):
    """Inclinacao WITHIN (efeitos fixos de instancia): centra L e y DENTRO de
    cada instancia e regride. Devolve so a inclinacao (o intercepto nao e
    identificado sob FE).

    Por que existe (achado do sintetico ao aplicar o 17o parecer, registrado no
    README §3): o OLS empilhado e NAO-VIESADO mesmo com desbalanceamento, mas
    quando as instancias nao cobrem todos os eps (e o DIAGNOSTICO exclui os runs
    truncados, que se concentram no eps pequeno) o efeito de instancia u_i deixa
    de ser ortogonal a L e vaza para a VARIANCIA da inclinacao. Com sd(u)=3.0
    isso alarga o IC do resto de ~0.004 para ~0.02 -- destruindo justamente a
    resolucao que o teste do resto precisa ter. O estimador WITHIN elimina u_i
    por construcao e recupera a resolucao. Usado SO no diagnostico; o PRIMARIO
    continua sendo o OLS empilhado congelado.
    """
    _, g = np.unique(inst, return_inverse=True)
    cnt = np.bincount(g)
    Lc = L - np.bincount(g, L) [g] / cnt[g]
    yc = y - np.bincount(g, y)[g] / cnt[g]
    den = float((Lc ** 2).sum())
    if den <= 0:
        return float("nan")
    return float((Lc * yc).sum() / den)


def clusters(inst):
    ids = np.unique(inst)
    return ids, [np.nonzero(inst == i)[0] for i in ids]


def boot_ajuste(inst, L, y, ordem=1, seed=BSEED, B=BOOT):
    """Bootstrap de CLUSTER (instancias inteiras). Devolve array (B, k)."""
    ids, idx = clusters(inst)
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


def meia(ic):
    return (ic[1] - ic[0]) / 2.0


def boot_pareado(d1, d2, fn, seed=BSEED + 7, B=BOOT, chave="y"):
    """Diferenca entre duas celulas reamostrando as MESMAS instancias nas duas
    (as celulas compartilham o conjunto de instancias por construcao).

    `chave` seleciona a variavel: "y" (bruto) ou "ystar" (grade DESCONTADA do
    vazamento de lr/K). As secundarias (i)/(ii) usam "ystar" -- ver docstring
    do modulo e README §1."""
    Lk, yk = ("L", "y") if chave == "y" else ("L_s", "y_s")
    ik = "inst" if chave == "y" else "inst_s"
    comuns = np.intersect1d(np.unique(d1[ik]), np.unique(d2[ik]))
    rng = np.random.default_rng(seed)
    i1 = {i: np.nonzero(d1[ik] == i)[0] for i in comuns}
    i2 = {i: np.nonzero(d2[ik] == i)[0] for i in comuns}
    out = []
    for _ in range(B):
        pick = comuns[rng.integers(0, len(comuns), len(comuns))]
        s1 = np.concatenate([i1[i] for i in pick])
        s2 = np.concatenate([i2[i] for i in pick])
        out.append(fn(ajusta(d1[Lk][s1], d1[yk][s1]),
                      ajusta(d2[Lk][s2], d2[yk][s2])))
    return np.array(out)


# --------------------------------------------------- diagnostico (nao decide morte)
def diagnostico(d):
    """DECOMPOSICAO PRE-REGISTRADA (17o parecer, item 3):

        a = 1 + d ln(lr)/dL + d ln(K_morto)/dL + resto

    Cada termo MEDIDO. `resto` e o teste novo com resolucao: se |resto| >
    RESTO_TOL com IC95 (bootstrap de cluster) EXCLUINDO 0, o veredito e
    ESTIMACAO-COM-ALERTA -- achado real a investigar, e NAO uma morte de
    H-BARREIRA (a fisica do resto nao esta derivada).

    K_morto e definido operacionalmente como (t_release - t_hit)/dt em PASSOS;
    logo runs TRUNCADOS (sem release) nao tem K_morto e sao EXCLUIDOS DO
    DIAGNOSTICO -- com a contagem reportada. Eles NUNCA saem do primario.
    """
    if not len(d["y_s"]):
        return dict(veredito="SEM DADO PARA DIAGNOSTICO", n_trunc=int(d["n_trunc"]))
    fe = lambda yy: ajusta_fe(d["inst_s"], d["L_s"], yy)   # noqa: E731
    dlr = fe(d["lnlr_s"])
    dK = fe(d["lnK_s"])
    # y* = y - ln(lr*K):  a* = a_ns - dlr - dK  =>  resto := a* - 1
    a_ns = fe(d["y_s"] + d["lnlr_s"] + d["lnK_s"])        # a no MESMO subconjunto
    a_star = fe(d["y_s"])
    resto = a_star - 1.0
    ids, idx = clusters(d["inst_s"])
    rng = np.random.default_rng(BSEED + 31)
    bs = []
    for _ in range(BOOT):
        pick = rng.integers(0, len(ids), len(ids))
        # cada instancia reamostrada e um cluster PROPRIO (senao a demeanagem
        # within misturaria copias da mesma instancia)
        sel = np.concatenate([idx[j] for j in pick])
        rot = np.concatenate([np.full(len(idx[j]), t) for t, j in enumerate(pick)])
        bs.append(ajusta_fe(rot, d["L_s"][sel], d["y_s"][sel]))
    bs = np.array(bs)
    rlo, rhi = ic95(bs[~np.isnan(bs)] - 1.0)
    alerta = abs(resto) > RESTO_TOL and not (rlo <= 0 <= rhi)
    return dict(a_diag=float(a_ns),
                # NOTA DE NOTACAO (2a rodada do 17o parecer, item 2): `a_diag` NAO
                # e o `a` do primario. E o estimador WITHIN (efeitos fixos) sobre o
                # SUBCONJUNTO sem truncados; o primario e OLS empilhado sobre todos
                # os runs com plato. Os dois DIFEREM POR CONSTRUCAO -- estimadores
                # diferentes em amostras diferentes. Nao sao para ser comparados
                # como se um "confirmasse" o outro.
                a_diag_o_que_e="inclinacao WITHIN (FE) no subconjunto sem truncados"
                               " -- NAO e o `a` do primario (OLS empilhado, todos"
                               " os runs com plato); diferem por construcao",
                d_ln_lr_dL=float(dlr), d_ln_K_dL=float(dK),
                resto=float(resto), ic95_resto=(rlo, rhi),
                soma_confere=float(a_ns - (1 + dlr + dK + resto)),
                soma_confere_o_que_e="residuo da identidade; ~0 POR CONSTRUCAO"
                                     " (nao e evidencia de nada)",
                n_usado=int(len(d["y_s"])), n_trunc=int(d["n_trunc"]),
                veredito="ESTIMACAO-COM-ALERTA" if alerta else "RESTO DENTRO DA TOLERANCIA")


# ------------------------------------------------------------ DECISAO (congelada)
def decide(dados, gd=None, sat=None):
    """dados: {celula: dict com as chaves montadas por `carrega()`/`mundo()`}.

    Regras, TODAS pre-registradas (PROTOCOL.md §1 e §3):

      P-NULL de instrumento: se a fracao de celulas (instancia x eps) da ANCORA
        com plato valido, NA GRADE PRIMARIA, for < 60% -> INSTRUMENTO-NULL.

      PRIMARIA H-BARREIRA (celula 'ancora', y BRUTO, grade primaria sem eps=1):
        IC95(a) contido em [0.9,1.1]      -> SUSTENTA
        IC95(a) disjunto de [0.9,1.1]     -> MATA
        caso contrario                    -> INCONCLUSIVO
        guarda de curvatura: |q|*Lmax^2 > CURV_TOL com IC95(q) excluindo 0
                                          -> INCONCLUSIVO-CURVATURA (sobrepoe)
      NAO EXISTE segundo primario. Uma pergunta, uma janela, um veredito.

      DIAGNOSTICO (nao mata): decomposicao de `a` e o RESTO -- ver diagnostico().

      SECUNDARIA (i) invariancia em beta, na grade DESCONTADA y*:
        se meia-largura do IC95(delta_a) > DA_POT      -> SEM POTENCIA
        senao, contem 0 ou |delta_a| < DA_TOL          -> INVARIANTE
        senao                                          -> NAO-INVARIANTE
      SECUNDARIA (ii) memoria de v, na grade DESCONTADA y*: EQUIVALENCIA.
        se meia-largura do IC95(delta_b) > DB2_POT     -> SEM POTENCIA
        senao, a invariante E IC95(delta_b) em [-2,+2] -> MECANISMO-V SUSTENTADO
        senao                                          -> MECANISMO-V REFUTADO
        (|delta_b| < 0.1 do Apendice A.3.3 e PREDICAO-SOB-TESTE, nao decisoria:
         o smoke ja deu 0.40, evidencia contraria previa -- README §7.5.)
      SECUNDARIA (iii) SATURACAO: Adam parka MAIS FUNDO que GD em TODO eps da
        grade EXECUTADA (inclusive eps=1, que aqui e ponto de saturacao).
    """
    out = {}
    anc = dados["ancora"]
    frac = len(anc["y"]) / max(anc["n_tot"], 1)
    out["grade_primaria"] = [float(e) for e in CFG["EPS_PRIMARIO"]]
    out["frac_plato_ancora"] = frac
    if frac < MINVAL:
        out["veredito"] = "INSTRUMENTO-NULL"
        out["motivo"] = f"so {frac:.0%} das celulas com plato valido (< {MINVAL:.0%})"
        return out

    a, b = ajusta(anc["L"], anc["y"])
    bs = boot_ajuste(anc["inst"], anc["L"], anc["y"], 1)
    lo, hi = ic95(bs[:, 0])
    out["a"], out["b"], out["ic95_a"] = float(a), float(b), (lo, hi)
    out["meia_largura_ic_a"] = meia((lo, hi))
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

    # --- DIAGNOSTICO de decomposicao (nao altera o veredito primario)
    out["diagnostico"] = diagnostico(anc)

    # --- (i) invariancia em beta  [grade DESCONTADA]
    if "beta-baixo" in dados:
        d = boot_pareado(dados["beta-baixo"], anc, lambda p1, p2: p1[0] - p2[0],
                         chave="ystar")
        dlo, dhi = ic95(d)
        da = (ajusta(dados["beta-baixo"]["L_s"], dados["beta-baixo"]["y_s"])[0]
              - ajusta(anc["L_s"], anc["y_s"])[0])
        pot = meia((dlo, dhi)) <= CFG["DA_POT"]
        ver = ("SEM POTENCIA" if not pot else
               "INVARIANTE" if (dlo <= 0 <= dhi or abs(da) < CFG["DA_TOL"])
               else "NAO-INVARIANTE")
        out["sec_i"] = dict(base="y* (grade descontada de ln(lr*K))",
                            delta_a=float(da), ic95=(dlo, dhi),
                            meia_largura=meia((dlo, dhi)), veredito=ver)
    # --- (ii) memoria de v  [grade DESCONTADA]
    cels = [("b2-090", 0.9), ("b2-099", 0.99), ("ancora", 0.999)]
    if all(c in dados for c, _ in cels):
        bb = {b2: ajusta(dados[c]["L_s"], dados[c]["y_s"]) for c, b2 in cels}
        d_a = boot_pareado(dados["b2-090"], anc, lambda p1, p2: p1[0] - p2[0],
                           chave="ystar")
        d_b = boot_pareado(dados["b2-090"], anc, lambda p1, p2: p1[1] - p2[1],
                           chave="ystar")
        alo, ahi = ic95(d_a)
        blo, bhi = ic95(d_b)
        a_ok = (alo <= 0 <= ahi) or abs(bb[0.9][0] - bb[0.999][0]) < CFG["DA_TOL"]
        b_ok = (blo >= -CFG["DB2_TOL"] and bhi <= CFG["DB2_TOL"])
        pot = meia((blo, bhi)) <= CFG["DB2_POT"]
        ver = ("SEM POTENCIA" if not pot else
               "MECANISMO-V SUSTENTADO" if (a_ok and b_ok) else "MECANISMO-V REFUTADO")
        db = float(bb[0.9][1] - bb[0.999][1])
        out["sec_ii"] = dict(base="y* (grade descontada de ln(lr*K))",
                             a_por_b2={k: float(v[0]) for k, v in bb.items()},
                             b_por_b2={k: float(v[1]) for k, v in bb.items()},
                             ic95_delta_a=(alo, ahi), ic95_delta_b=(blo, bhi),
                             veredito=ver,
                             # PREDICAO-SOB-TESTE (Apendice A.3.3), NAO decisoria.
                             # Evidencia contraria previa: smoke deu -0.40 (README §7.5).
                             predicao_sob_teste_db_menor_0p1=dict(
                                 delta_b=db, cumpre=bool(abs(db) < CFG["DB2_PREV"]),
                                 etiqueta="predicao-sob-teste, SEM POTENCIA POR DESENHO"
                                          " (meia-largura do IC95(delta_b) ~0.4 contra"
                                          " tolerancia 0.1): este desenho NAO consegue"
                                          " decidir 0.1 nem em principio. NAO decisoria.",
                                 meia_largura=meia((blo, bhi)),
                                 evidencia_contraria_previa="smoke 1 instancia: -0.40"))
    # --- (iii) saturacao: GD e eps=1 como ancoras
    if gd is not None and len(gd):
        med_gd = float(np.median(gd))          # ja em PROFUNDIDADE (y = -beta*z_park)
        La, ya = (sat if sat is not None else (anc["L"], anc["y"]))
        pares = [(float(Lv), float(np.median(ya[La == Lv]))) for Lv in np.unique(La)]
        fundo = [m > med_gd for _, m in pares]
        out["sec_iii"] = dict(prof_gd=med_gd, medianas_por_L=pares,
                              pontos_de_saturacao=[float(e) for e in CFG["EPS_SATURACAO"]]
                              + ["GD"],
                              adam_mais_fundo_em_todo_eps=bool(all(fundo)),
                              veredito="SATURACAO COERENTE" if all(fundo)
                              else "SATURACAO VIOLADA")
    return out


# ------------------------------------------------------------------ carga
def _cel(regs, n_tot):
    """Monta o dict de uma celula a partir dos registros ja filtrados por plato.

    Chaves:
      inst,L,y            -> GRADE PRIMARIA (sem eps=1), y bruto  [PRIMARIO]
      inst_s,L_s,y_s      -> idem, y* = y - ln(lr*K), SO nao-truncados  [DIAG/SEC]
      lnlr_s,lnK_s        -> os dois vazamentos, no mesmo subconjunto
      L_all,y_all         -> grade EXECUTADA inteira (inclui eps=1)  [SEC (iii)]
      n_tot,n_trunc
    """
    prim = [r for r in regs if r["eps"] in CFG["EPS_PRIMARIO"]]
    ns = [r for r in prim if not r["truncado"]]
    lnlr = np.array([np.log(r["lr"]) for r in ns])
    K = np.array([(r["t_release"] - r["t_hit"]) / CFG["DT"] for r in ns])
    y_s = np.array([-r["bz_park"] for r in ns]) - (lnlr + np.log(K))
    return dict(
        inst=np.array([r["seed"] for r in prim]),
        L=np.array([np.log(1.0 / r["eps"]) for r in prim]),
        y=np.array([-r["bz_park"] for r in prim]),
        inst_s=np.array([r["seed"] for r in ns]),
        L_s=np.array([np.log(1.0 / r["eps"]) for r in ns]),
        y_s=y_s, lnlr_s=lnlr, lnK_s=np.log(K),
        L_all=np.array([np.log(1.0 / r["eps"]) for r in regs]),
        y_all=np.array([-r["bz_park"] for r in regs]),
        n_tot=n_tot, n_trunc=len(prim) - len(ns))


def carrega(path=None):
    path = path or os.path.join(RES, "hbarreira_runs.jsonl")
    regs = [json.loads(l) for l in open(path) if l.strip()]
    # FILTRO DE VALIDADE: apenas `plato`. A guarda `n_mortes != 1` foi REMOVIDA
    # (17o parecer, forte 7): com o motor como esta, `morto` so vira True uma vez,
    # logo n_mortes nunca passa de 1 e a guarda nunca podia disparar -- era
    # decorativa e dava falsa sensacao de filtro. n_mortes continua GRAVADO.
    dados, gd = {}, []
    for r in regs:
        if not r["plato"]:
            continue
        if r["opt"] == "gd":
            if r["beta"] == 1e4:
                gd.append(-r["bz_park"])
            continue
        dados.setdefault(r["celula"], []).append(r)
    tot = {}
    for r in regs:
        if r["opt"] == "adam" and r["eps"] in CFG["EPS_PRIMARIO"]:
            tot[r["celula"]] = tot.get(r["celula"], 0) + 1
    d = {c: _cel(rs, tot[c]) for c, rs in dados.items()}
    # diagnostico de instrumento por eps (inclui os pontos de saturacao)
    est = {}
    for r in regs:
        if r["opt"] != "adam":
            continue
        k = f"{r['eps']:.0e}"
        a, n = est.get(k, (0, 0))
        est[k] = (a + int(bool(r.get("estacionario"))), n + 1)
    return d, np.array(gd), {k: f"{a}/{n} estacionarios" for k, (a, n) in est.items()}


# ------------------------------------------------------------- mundo sintetico
def mundo(a_true, seed, n=100, sd_ruido=0.4, sd_inst=3.0, b_true=6.0,
          sem_lei=False, b2_shift=None, q_true=0.0, u_seed=777,
          dlr=-0.006, dK=-0.003, joelho=0.0, frac_trunc=0.1):
    """Mundo sintetico com a MESMA estrutura de `carrega()`.

    dlr, dK  : os vazamentos de projeto (d ln lr/dL, d ln K/dL) -- por default
               os valores medidos no piloto na grade sem eps=1 (soma -0.009).
    joelho   : deslocamento aplicado SO no ponto eps=1 (L=0). E o "mundo do
               desenho velho": a curvatura mora no joelho, que agora esta FORA
               da grade primaria -- a guarda de curvatura NAO deve disparar.
    frac_trunc: fracao de runs sem release (sem K_morto) -> fora do diagnostico.

    u_seed FIXO entre celulas: no experimento real as celulas compartilham as
    MESMAS instancias, entao o efeito de instancia u_i e comum (e o bootstrap
    pareado tira ele). O sintetico tem de imitar isso, senao a variancia de
    delta_b fica inflada e o teste secundario parece menos potente do que e.
    """
    rng = np.random.default_rng(seed)
    L_prim = np.log(1.0 / np.array(CFG["EPS_PRIMARIO"]))
    L_full = np.log(1.0 / np.array(CFG["EPS_GRID"]))
    u = np.random.default_rng(u_seed).normal(0, sd_inst, n)

    def gera(Ls):
        inst, Lo, yo = [], [], []
        for i in range(n):
            for Lv in Ls:
                mu = (b_true + u[i] if sem_lei
                      else a_true * Lv + b_true + q_true * Lv ** 2 + u[i])
                if b2_shift:
                    mu += b2_shift
                if joelho and Lv == 0.0:
                    mu += joelho
                inst.append(i)
                Lo.append(Lv)
                yo.append(mu + rng.normal(0, sd_ruido))
        return np.array(inst), np.array(Lo), np.array(yo)

    inst, L, y = gera(L_prim)
    _, L_all, y_all = gera(L_full)
    # vazamento de projeto: ln(lr) e ln(K) dependem de L com as inclinacoes dadas
    lnlr = dlr * L + np.log(1.2e-6) + rng.normal(0, 0.01, len(L))
    lnK = dK * L + np.log(3.8e5) + rng.normal(0, 0.05, len(L))
    # e o y BRUTO tem de conter esse vazamento, senao o diagnostico nao fecha:
    y = y + (lnlr - lnlr.mean()) + (lnK - lnK.mean())
    keep = rng.random(len(L)) >= frac_trunc
    return dict(inst=inst, L=L, y=y,
                inst_s=inst[keep], L_s=L[keep],
                y_s=y[keep] - (lnlr[keep] + lnK[keep]),
                lnlr_s=lnlr[keep], lnK_s=lnK[keep],
                L_all=L_all, y_all=y_all,
                n_tot=n * len(L_prim), n_trunc=int((~keep).sum()))


def _celulas(base):
    return {"ancora": mundo(**base),
            "beta-baixo": mundo(**{**base, "seed": base["seed"] + 100}),
            "b2-099": mundo(**{**base, "seed": base["seed"] + 200, "b2_shift": 0.04}),
            "b2-090": mundo(**{**base, "seed": base["seed"] + 300, "b2_shift": 0.07})}


def sintetico():
    """Todos os controles, nos DOIS sentidos: o teste tem de SUSTENTAR quando o
    mundo obedece, MATAR quando nao obedece, e nao decidir quando nao ha
    resolucao."""
    casos = [
        ("mundo a=1 (mecanismo verdadeiro)", dict(a_true=1.0, seed=1), "SUSTENTA"),
        ("mundo a=0.97 (predicao derivada do desenho)", dict(a_true=0.97, seed=2), "SUSTENTA"),
        ("mundo a=0.7 (mecanismo falso)", dict(a_true=0.7, seed=3), "MATA"),
        ("mundo a=1.4", dict(a_true=1.4, seed=4), "MATA"),
        ("mundo SEM LEI (z_park nao depende de eps)",
         dict(a_true=0.0, seed=5, sem_lei=True), "MATA"),
        ("mundo a=0.85, ruido grande, n=6 (indeciso)",
         dict(a_true=0.85, seed=6, n=6, sd_ruido=25.0), "INCONCLUSIVO"),
        ("mundo com CURVATURA REAL q=0.005 (efeito 1.7)",
         dict(a_true=1.0, seed=7, q_true=0.005), "INCONCLUSIVO-CURVATURA"),
        # 17o parecer, cosmetico 15: o mundo do DESENHO VELHO -- a curvatura mora
        # SO no joelho de eps=1, que a grade primaria nova exclui. NAO pode disparar.
        ("mundo do JOELHO (so eps=1 deslocado; grade nova exclui)",
         dict(a_true=1.0, seed=8, joelho=-3.0), "SUSTENTA"),
    ]
    falhas = 0
    for nome, kw, esperado in casos:
        dados = _celulas(kw)
        d = dados["ancora"]
        gd = np.random.default_rng(kw["seed"]).normal(5.0, 0.5, kw.get("n", 100))
        r = decide(dados, gd, sat=(d["L_all"], d["y_all"]))
        ok = r["veredito"] == esperado
        falhas += (not ok)
        print(f"{'OK ' if ok else 'FALHA'} {nome:52s} -> {r['veredito']:24s} "
              f"a={r.get('a', float('nan')):+.3f} "
              f"IC95={tuple(round(x,3) for x in r.get('ic95_a',(0,0)))} "
              f"|q|Lmax^2={r.get('curv_efeito', float('nan')):.3f} "
              f"(esperado {esperado})")
        if nome.startswith("mundo a=1 "):
            dg = r["diagnostico"]
            print(f"      meia-largura IC95(a) = {r['meia_largura_ic_a']:.4f}")
            print(f"      diag: a_diag={dg['a_diag']:+.4f} = 1 "
                  f"{dg['d_ln_lr_dL']:+.4f}(lr) {dg['d_ln_K_dL']:+.4f}(K) "
                  f"{dg['resto']:+.4f}(resto)  identidade fecha em "
                  f"{dg['soma_confere']:.1e} (POR CONSTRUCAO)")
            print(f"      diag: a_diag e FE-within sem truncados; NAO e o "
                  f"a={r['a']:+.4f} do primario (OLS empilhado) -- diferem por construcao")
            print(f"      diag: {dg['veredito']}  IC95(resto)="
                  f"{tuple(round(x,4) for x in dg['ic95_resto'])}  "
                  f"n={dg['n_usado']} truncados_excluidos={dg['n_trunc']}")
            print(f"      sec(i)  {r['sec_i']['veredito']}  delta_a={r['sec_i']['delta_a']:+.3f}")
            print(f"      sec(ii) {r['sec_ii']['veredito']}  b={r['sec_ii']['b_por_b2']}")
            print(f"      sec(iii){r['sec_iii']['veredito']}  prof_gd={r['sec_iii']['prof_gd']:.2f}")
        if nome.startswith("mundo do JOELHO"):
            # contraprova: na grade VELHA (com eps=1) o MESMO mundo dispararia.
            La, ya = d["L_all"], d["y_all"]
            _, _, qv = ajusta(La, ya, 2)
            av = ajusta(La, ya)[0]
            print(f"      contraprova: na grade VELHA (com eps=1) o MESMO mundo daria "
                  f"a={av:+.3f} e |q|Lmax^2={abs(qv)*La.max()**2:.3f} "
                  f"({'DISPARARIA' if abs(qv)*La.max()**2 > CURV_TOL else 'nao dispararia'})")

    # --- DIAGNOSTICO do resto, nos DOIS sentidos (teste novo do 17o parecer)
    for nome, kw, esperado in [
        ("diag: mundo SEM resto (a = 1 + vazamentos)",
         dict(a_true=1.0, seed=41), "RESTO DENTRO DA TOLERANCIA"),
        ("diag: mundo COM resto -0.018 (o do piloto)",
         dict(a_true=1.0 - 0.018, seed=42), "ESTIMACAO-COM-ALERTA"),
        ("diag: mundo COM resto +0.05",
         dict(a_true=1.05, seed=43), "ESTIMACAO-COM-ALERTA"),
    ]:
        # o `a_true` do mundo ja e 1+resto ANTES dos vazamentos serem somados a y
        r = diagnostico(mundo(**kw))
        ok = r["veredito"] == esperado
        falhas += (not ok)
        print(f"{'OK ' if ok else 'FALHA'} {nome:52s} -> {r['veredito']:24s} "
              f"resto={r['resto']:+.4f} IC95={tuple(round(x,4) for x in r['ic95_resto'])} "
              f"(esperado {esperado})")

    # --- controles das secundarias, nos DOIS sentidos
    base = dict(a_true=1.0, seed=11)
    for nome, shift, esperado in [
        ("sec(ii) mundo com b2 IRRELEVANTE (mecanismo eps)", 0.07, "MECANISMO-V SUSTENTADO"),
        ("sec(ii) mundo com b2 DOMINANTE (mecanismo v)", 8.0, "MECANISMO-V REFUTADO"),
    ]:
        dados = {"ancora": mundo(**base),
                 "b2-099": mundo(**{**base, "seed": 12, "b2_shift": shift / 2}),
                 "b2-090": mundo(**{**base, "seed": 13, "b2_shift": shift}),
                 "beta-baixo": mundo(**{**base, "seed": 14})}
        r = decide(dados)
        ok = r["sec_ii"]["veredito"] == esperado
        falhas += (not ok)
        print(f"{'OK ' if ok else 'FALHA'} {nome:52s} -> {r['sec_ii']['veredito']:24s} "
              f"IC95(db)={tuple(round(x,2) for x in r['sec_ii']['ic95_delta_b'])} "
              f"(esperado {esperado})")
    # sec(ii) SEM POTENCIA: ruido grande e poucas instancias
    dados = {"ancora": mundo(a_true=1.0, seed=15, n=6, sd_ruido=25.0),
             "b2-099": mundo(a_true=1.0, seed=16, n=6, sd_ruido=25.0),
             "b2-090": mundo(a_true=1.0, seed=17, n=6, sd_ruido=25.0)}
    r = decide(dados)
    ok = r["sec_ii"]["veredito"] == "SEM POTENCIA"
    falhas += (not ok)
    print(f"{'OK ' if ok else 'FALHA'} {'sec(ii) guarda de POTENCIA (ruido grande, n=6)':52s}"
          f" -> {r['sec_ii']['veredito']}")

    for nome, da, esperado in [("sec(i) beta INVARIANTE", 0.0, "INVARIANTE"),
                               ("sec(i) beta NAO-INVARIANTE", 0.35, "NAO-INVARIANTE")]:
        dados = {"ancora": mundo(a_true=1.0, seed=21),
                 "beta-baixo": mundo(a_true=1.0 + da, seed=22)}
        r = decide(dados)
        ok = r["sec_i"]["veredito"] == esperado
        falhas += (not ok)
        print(f"{'OK ' if ok else 'FALHA'} {nome:52s} -> {r['sec_i']['veredito']:24s} "
              f"delta_a={r['sec_i']['delta_a']:+.3f} (esperado {esperado})")
    # sec(i) SEM POTENCIA (17o parecer, blocker 6): IC largo NAO pode virar INVARIANTE
    dados = {"ancora": mundo(a_true=1.0, seed=23, n=6, sd_ruido=25.0),
             "beta-baixo": mundo(a_true=1.0, seed=24, n=6, sd_ruido=25.0)}
    r = decide(dados)
    ok = r["sec_i"]["veredito"] == "SEM POTENCIA"
    falhas += (not ok)
    print(f"{'OK ' if ok else 'FALHA'} {'sec(i) guarda de POTENCIA (ruido grande, n=6)':52s}"
          f" -> {r['sec_i']['veredito']}  meia-largura={r['sec_i']['meia_largura']:.3f}")
    # sec(i) com vazamento DIFERENTE por celula: sobre y bruto pareceria
    # NAO-INVARIANTE; na grade descontada tem de dar INVARIANTE.
    dados = {"ancora": mundo(a_true=1.0, seed=25, dlr=-0.006),
             "beta-baixo": mundo(a_true=1.0, seed=26, dlr=-0.306)}
    r = decide(dados)
    ok = r["sec_i"]["veredito"] == "INVARIANTE"
    falhas += (not ok)
    print(f"{'OK ' if ok else 'FALHA'} "
          f"{'sec(i) confound de lr removido pela grade descontada':52s} -> "
          f"{r['sec_i']['veredito']:24s} delta_a={r['sec_i']['delta_a']:+.3f} "
          f"(y bruto daria {ajusta(dados['beta-baixo']['L'], dados['beta-baixo']['y'])[0] - ajusta(dados['ancora']['L'], dados['ancora']['y'])[0]:+.3f})")

    for nome, gdv, esperado in [("sec(iii) Adam mais fundo que GD", 5.0, "SATURACAO COERENTE"),
                                ("sec(iii) GD mais fundo que Adam", 40.0, "SATURACAO VIOLADA")]:
        d = mundo(a_true=1.0, seed=31)
        r = decide({"ancora": d}, np.full(100, gdv), sat=(d["L_all"], d["y_all"]))
        ok = r["sec_iii"]["veredito"] == esperado
        falhas += (not ok)
        print(f"{'OK ' if ok else 'FALHA'} {nome:52s} -> {r['sec_iii']['veredito']:24s} "
              f"(esperado {esperado})")
    # controle do P-NULL de instrumento
    d = mundo(a_true=1.0, seed=9)
    d["n_tot"] = int(len(d["y"]) / 0.4)     # so 40% das celulas teriam plato
    r = decide({"ancora": d})
    ok = r["veredito"] == "INSTRUMENTO-NULL"
    falhas += (not ok)
    print(f"{'OK ' if ok else 'FALHA'} {'controle P-NULL de instrumento':52s} -> {r['veredito']}")
    print("\nSINTETICO: " + ("TUDO PASSA" if falhas == 0 else f"{falhas} FALHAS"))
    return falhas


def real():
    d, gd, est = carrega()
    anc = d["ancora"]
    r = decide(d, gd, sat=(anc["L_all"], anc["y_all"]))
    r["estacionariedade_por_eps"] = est
    print(json.dumps(r, indent=1, default=float))


if __name__ == "__main__":
    # SUBCOMANDO EXPLICITO OBRIGATORIO -- default e ERRO (17o parecer, forte 11).
    cmd = sys.argv[1] if len(sys.argv) > 1 else None
    if cmd == "sintetico":
        raise SystemExit(sintetico())
    if cmd == "real":
        raise SystemExit(real())
    raise SystemExit("uso: python analyze_barrier.py {sintetico|real}\n"
                     "     (subcomando OBRIGATORIO: sem ele, ou com typo, o script\n"
                     "      ERRA -- para que nenhum engano vire um peek no dado real)")
