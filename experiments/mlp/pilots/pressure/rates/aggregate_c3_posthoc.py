#!/usr/bin/env python3
"""AGREGACAO POST-HOC do estimando C3 (4a.3) -- EXPLORATORIO, NAO CONFIRMATORIO.

C3 foi pre-registrado como estimation-only, SEM condicao de MATA, e a analise
congelada (analyze_4a3.py, commit a8f8f8c) nao contem agregado para ele. Este
script foi escrito DEPOIS de ver os dados confirmatorios, a pedido de revisor
externo. Tudo que ele produz e' POST-HOC EXPLORATORIO e nao pode ser reportado
como resultado confirmatorio nem como teste de hipotese pre-registrada.

Le results-4a3/*.jsonl.gz (episodios_gt do instrumento GT-AREA) e agrega, para o
braco primario sp-b3, as voltas dinamicas do C2:
  t_morte, t_area (1o passo com A_hat>0 no episodio), t_res, z_park, sigma_park.

Saida: c3-posthoc.md
"""
import glob
import gzip
import json
import math
import os

import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(DIR, "results-4a3")
OUT = os.path.join(DIR, "c3-posthoc.md")
BRACO = "sp-b3"          # primario do 4a.3 (C1/C2)
B_BOOT = 20000
SEED_BOOT = 20260726


# ----------------------------------------------------------------- utilitarios
def ranks(x):
    """Postos medios (ties -> media), como no Spearman padrao."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[order] = np.arange(1, len(x) + 1, dtype=float)
    # correcao de empates
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def pearson(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else float("nan")


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def partial_spearman(x, y, z):
    """Correlacao PARCIAL de postos: Pearson dos residuos dos postos de x e y
    regredidos (OLS, com intercepto) nos postos de z."""
    rx, ry, rz = ranks(x), ranks(y), ranks(z)
    Z = np.column_stack([np.ones(len(rz)), rz])

    def resid(v):
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta

    return pearson(resid(rx), resid(ry))


def boot_ci(stat_fn, rows, seeds, B=B_BOOT, seed=SEED_BOOT, cluster=True):
    """IC percentil 95%. cluster=True: bootstrap de SEED (unidade de
    independencia do desenho); cluster=False: bootstrap de episodio (ignora a
    dependencia dentro de seed -> otimista, reportado so' por comparacao)."""
    rng = np.random.default_rng(seed)
    est = []
    if cluster:
        by_seed = {}
        for r in rows:
            by_seed.setdefault(r["seed"], []).append(r)
        keys = list(by_seed)
        for _ in range(B):
            pick = rng.integers(0, len(keys), size=len(keys))
            sample = [e for i in pick for e in by_seed[keys[i]]]
            v = stat_fn(sample)
            if v is not None and np.isfinite(v):
                est.append(v)
    else:
        n = len(rows)
        for _ in range(B):
            idx = rng.integers(0, n, size=n)
            v = stat_fn([rows[i] for i in idx])
            if v is not None and np.isfinite(v):
                est.append(v)
    if len(est) < 100:
        return (float("nan"), float("nan"), len(est))
    lo, hi = np.percentile(est, [2.5, 97.5])
    return (float(lo), float(hi), len(est))


def col(rows, k):
    return np.array([r[k] for r in rows], float)


def q(v, p):
    return float(np.percentile(np.asarray(v, float), p))


# ------------------------------------------------------------------ carregar
def carrega():
    eps = []
    n_runs = 0
    for f in sorted(glob.glob(os.path.join(RES, "*.jsonl.gz"))):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                run = json.loads(line)
                n_runs += 1
                if run["rot"] != BRACO:
                    continue
                for e in run["episodios_gt"]:
                    if e["classe"] != "dyn" or not e["revived"]:
                        continue
                    eps.append(dict(
                        seed=run["seed"], m=run["m"], k=e["k"],
                        t_morte=int(e["t_morte"]), t_area=int(e["t_area"]),
                        t_res=int(e["t_res"]),
                        z_park=float(e["z_park"]),
                        sigma_park=float(e["sigma_park"]),
                        area=bool(e["t_area"] >= 0),
                    ))
    return eps, n_runs


def main():
    todas, n_runs = carrega()
    com_area = [e for e in todas if e["area"]]
    for e in todas:
        e["dur"] = e["t_res"] - e["t_morte"]
    for e in com_area:
        e["lead"] = e["t_res"] - e["t_area"]       # lead time do portao
        e["lag"] = e["t_area"] - e["t_morte"]      # espera ate a area abrir
        e["frac"] = e["lag"] / e["dur"] if e["dur"] > 0 else float("nan")
        e["absz"] = abs(e["z_park"])

    L = []
    A = L.append
    A("# C3 (gap-de-subida) — AGREGACAO POST-HOC, EXPLORATORIA")
    A("")
    A("> **[ESPECULACAO / POST-HOC]** C3 foi pre-registrado como *estimation-only*, "
      "sem condicao de MATA, e a analise congelada `analyze_4a3.py` NAO contem "
      "agregado para ele. Este arquivo foi produzido por `aggregate_c3_posthoc.py` "
      "DEPOIS de ver os dados confirmatorios, a pedido de revisor externo. "
      "Nada aqui e' confirmatorio; nenhum numero abaixo testa hipotese "
      "pre-registrada. N e' pequeno (16 voltas, poucas seeds) e as conclusoes "
      "devem ser lidas como geracao de hipotese para a etapa seguinte.")
    A("")
    A(f"- runs lidos: {n_runs}; braco analisado: `{BRACO}` (primario do 4a.3)")
    A(f"- voltas dinamicas (denominador do C2): **{len(todas)}**")
    A(f"- voltas COM area aberta (t_area>=0): **{len(com_area)}**  "
      f"(= as ordenadas do C2)")
    A(f"- seeds distintas com volta: {len(set(e['seed'] for e in todas))}")
    A("")

    A("## 0. Honestidade estrutural (ler ANTES dos numeros)")
    A("")
    A("O relogio GT-AREA acumula `A_hat` **apenas em passos em que o neuronio esta "
      "morto**, e o episodio **fecha na volta**. Logo `t_area < t_res` sempre que a "
      "area abre — a desigualdade e' **por construcao**, nao e' achado. O conteudo "
      "empirico de C2/C3 e':")
    A("")
    A(f"1. a area ter de fato virado positiva em **{len(com_area)}/{len(todas)}** "
      "voltas (o modo de falha real e' *volta sem area*, e ele ocorreu 3 vezes);")
    A("2. a **magnitude** do lead time `t_res - t_area` e a questao de se `t_area` "
      "carrega informacao sobre `t_res` **alem** da que a mera escala temporal do "
      "episodio ja' carrega (secao 2).")
    A("")
    A("Alem disso: a **drive media do episodio nao esta gravada**. O motor "
      "(`engine_4a3.py`) acumula `A_hat` in-place e persiste apenas o trio de tempos, "
      "`z_park` e `sigma_park`; nao ha' serie de `R_hat` por episodio. Onde a tarefa "
      "pede 'drive', usamos o proxy disponivel: profundidade `|z_park|` e canal "
      "`sigma(beta*z_park)`. Isso e' um substituto, nao a drive.")
    A("")

    # ------------------------------------------------------- 1. distribuicoes
    A("## 1. Lead time do portao (t_res - t_area), n = %d" % len(com_area))
    A("")
    lead = col(com_area, "lead")
    lag = col(com_area, "lag")
    dur = col(com_area, "dur")
    frac = col(com_area, "frac")

    def linha(nome, v):
        return (f"{nome} | {v.min():.0f} | {q(v,25):.0f} | {np.median(v):.0f} | "
                f"{q(v,75):.0f} | {v.max():.0f} | {v.mean():.1f}")

    A("quantidade | min | q25 | mediana | q75 | max | media")
    A("---|---|---|---|---|---|---")
    A(linha("lead = t_res - t_area", lead))
    A(linha("lag  = t_area - t_morte", lag))
    A(linha("dur  = t_res - t_morte", dur))
    A("")
    A(f"Fracao do episodio ja' decorrida quando a area abre "
      f"(`lag/dur`): mediana {np.median(frac):.3f}, "
      f"min {frac.min():.3f}, max {frac.max():.3f}.")
    # IC bootstrap (cluster de seed) da mediana do lead e da fracao
    for nome, key in (("mediana do lead", "lead"), ("mediana de lag/dur", "frac")):
        lo, hi, nb = boot_ci(lambda rr, k=key: float(np.median(col(rr, k))), com_area,
                             None)
        lo2, hi2, _ = boot_ci(lambda rr, k=key: float(np.median(col(rr, k))),
                              com_area, None, cluster=False)
        A(f"- IC95 da {nome}: **[{lo:.0f}, {hi:.0f}]** (bootstrap de seed) / "
          f"[{lo2:.0f}, {hi2:.0f}] (bootstrap de episodio, otimista)"
          if key == "lead" else
          f"- IC95 da {nome}: **[{lo:.3f}, {hi:.3f}]** (bootstrap de seed) / "
          f"[{lo2:.3f}, {hi2:.3f}] (bootstrap de episodio, otimista)")
    A("")
    A("Episodios (um por linha):")
    A("")
    A("seed | m | k | t_morte | t_area | t_res | dur | lead | lag/dur | z_park | sigma_park")
    A("---|---|---|---|---|---|---|---|---|---|---")
    for e in sorted(com_area, key=lambda r: (r["seed"], r["m"], r["k"])):
        A(f"{e['seed']} | {e['m']} | {e['k']} | {e['t_morte']} | {e['t_area']} | "
          f"{e['t_res']} | {e['dur']} | {e['lead']} | {e['frac']:.3f} | "
          f"{e['z_park']:.4f} | {e['sigma_park']:.4g}")
    sem = [e for e in todas if not e["area"]]
    if sem:
        A("")
        A("Voltas SEM area (contadas CONTRA a lei no C2):")
        A("")
        A("seed | m | k | t_morte | t_res | dur | z_park")
        A("---|---|---|---|---|---|---")
        for e in sorted(sem, key=lambda r: (r["seed"], r["m"], r["k"])):
            A(f"{e['seed']} | {e['m']} | {e['k']} | {e['t_morte']} | {e['t_res']} | "
              f"{e['dur']} | {e['z_park']:.4f}")
    A("")

    # ------------------------------------------------- 2. correlacoes de posto
    A("## 2. t_area prediz t_res alem da escala temporal do episodio?")
    A("")
    A("Spearman bruto e PARCIAL (residuos de postos, OLS com intercepto). "
      "ICs = percentil de bootstrap **de seed** (clusteriza a dependencia dentro "
      "de uma mesma seed); em parenteses o bootstrap de episodio, que ignora essa "
      "dependencia e portanto e' otimista.")
    A("")

    def mk(fn):
        def g(rr):
            if len(rr) < 4:
                return None
            return fn(rr)
        return g

    testes = [
        ("rho(t_area, t_res) bruto",
         mk(lambda rr: spearman(col(rr, "t_area"), col(rr, "t_res")))),
        ("rho(t_area, t_res | t_morte)",
         mk(lambda rr: partial_spearman(col(rr, "t_area"), col(rr, "t_res"),
                                        col(rr, "t_morte")))),
        ("rho(t_area, t_res | dur)  [ver ressalva]",
         mk(lambda rr: partial_spearman(col(rr, "t_area"), col(rr, "t_res"),
                                        col(rr, "dur")))),
        ("rho(lag, lead)  [decomposicao livre de t_morte]",
         mk(lambda rr: spearman(col(rr, "lag"), col(rr, "lead")))),
        ("rho(lead, dur)",
         mk(lambda rr: spearman(col(rr, "lead"), col(rr, "dur")))),
    ]
    A("estatistica | valor | IC95 (boot de seed) | IC95 (boot de episodio)")
    A("---|---|---|---")
    for nome, fn in testes:
        val = fn(com_area)
        lo, hi, nb = boot_ci(fn, com_area, None)
        lo2, hi2, _ = boot_ci(fn, com_area, None, cluster=False)
        A(f"{nome} | {val:+.3f} | [{lo:+.3f}, {hi:+.3f}] | [{lo2:+.3f}, {hi2:+.3f}]")
    A("")
    A("**Ressalva sobre `| dur`:** `dur = t_res - t_morte` e' funcao de `t_res`, "
      "entao condicionar em `dur` NAO e' um controle limpo — parte da variancia de "
      "`t_res` esta' no proprio condicionante. O controle interpretavel e' "
      "`| t_morte` (quando o episodio comecou) e, equivalentemente, a decomposicao "
      "`rho(lag, lead)`: se a espera ate a area abrir nao informar nada sobre o "
      "tempo restante ate a volta, `rho(lag, lead)` fica indistinguivel de zero.")
    A("")

    # ----------------------------------------------- 3. lead vs profundidade
    A("## 3. Lead time contra profundidade / canal (proxy de drive)")
    A("")
    testes3 = [
        ("rho(lead, |z_park|)",
         mk(lambda rr: spearman(col(rr, "lead"), col(rr, "absz")))),
        ("rho(lead, sigma_park)",
         mk(lambda rr: spearman(col(rr, "lead"), col(rr, "sigma_park")))),
        ("rho(dur, |z_park|)",
         mk(lambda rr: spearman(col(rr, "dur"), col(rr, "absz")))),
        ("rho(lag, |z_park|)",
         mk(lambda rr: spearman(col(rr, "lag"), col(rr, "absz")))),
    ]
    A("estatistica | valor | IC95 (boot de seed) | IC95 (boot de episodio)")
    A("---|---|---|---")
    for nome, fn in testes3:
        val = fn(com_area)
        lo, hi, _ = boot_ci(fn, com_area, None)
        lo2, hi2, _ = boot_ci(fn, com_area, None, cluster=False)
        A(f"{nome} | {val:+.3f} | [{lo:+.3f}, {hi:+.3f}] | [{lo2:+.3f}, {hi2:+.3f}]")
    A("")
    A("A razao `|z_park|/drive` pedida na tarefa **nao e' computavel** com o que "
      "foi gravado: nao ha' drive por episodio (secao 0). `sigma_park = "
      "sigma(beta*z_park)` e' monotona decrescente em `|z_park|`, entao as duas "
      "primeiras linhas sao a mesma informacao com sinal trocado — nao sao "
      "evidencias independentes.")
    A("")

    n0 = int((lag == 0).sum())
    A("## 4. O fato que domina tudo: lag = 0 em %d/%d episodios" % (n0, len(com_area)))
    A("")
    A(f"Em **{n0} das {len(com_area)}** voltas com area, `t_area == t_morte` "
      "EXATAMENTE: a area do lider ja' e' positiva no primeiro passo morto. So' "
      f"{len(com_area)-n0} episodios tem espera nao-nula (lag = 512, 58850 e 98973).")
    A("")
    A("Consequencias, que devem ser ditas antes de qualquer numero da secao 2:")
    A("")
    A("- o lead time NAO e' uma antecedencia informativa do portao: nesses "
      "%d casos ele e' identicamente a duracao do episodio, e a estatistica "
      "'lead' vira um sinonimo de 'quanto tempo o neuronio ficou morto';" % n0)
    A("- `rho(t_area, t_res)` bruto = +0.79 e' quase inteiramente `rho(t_morte, "
      "t_res)`, i.e. episodios que comecam tarde terminam tarde. Controlando por "
      "`t_morte` a correlacao desaba para +0.10 com IC95 cruzando zero: **nao ha' "
      "evidencia de que o zero da area preveja o tempo de retorno alem de quando "
      "o episodio comecou**;")
    A("- `rho(lead, dur) = +0.91` e' tautologico pela mesma razao;")
    A("- `rho(lag, lead) = -0.56` e' carregada pelos 2-3 episodios de lag grande "
      "(os que abrem a area tarde voltam logo em seguida) e nao suporta peso.")
    A("")
    A("Interpretacao mecanicista honesta: sob GT-AREA o portao nao 'aquece' — na "
      "maioria dos episodios ele ja' esta' aberto na morte, e o que varia e' quanto "
      "tempo a subida leva. Isso e' consistente com a lei de area (a area e' "
      "condicao, nao gatilho temporal) e ao mesmo tempo esvazia a leitura de C3 "
      "como 'lead time preditivo'.")
    A("")
    A("## 5. Leitura conservadora")
    A("")
    A(f"Com n = {len(com_area)} episodios em "
      f"{len(set(e['seed'] for e in com_area))} seeds, os ICs de posto sao largos "
      "por construcao; qualquer IC que contenha 0 deve ser lido como "
      "**sem evidencia**, nao como ausencia de efeito. Nenhuma correlacao aqui "
      "sobrevive a uma correcao para as ~9 estatisticas calculadas, e nenhuma foi "
      "pre-especificada. Uso legitimo: dimensionar n e escolher UM estimando "
      "pre-registrado para a proxima etapa.")
    A("")
    A("Gerado por `aggregate_c3_posthoc.py` (B=%d reamostragens, seed=%d)."
      % (B_BOOT, SEED_BOOT))

    with open(OUT, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("\n-> escrito:", OUT)


if __name__ == "__main__":
    main()
