"""Etapa 4a — ANÁLISE PRÉ-REGISTRADA. Lê results/*.json.gz + config_stage4.json,
emite results-summary.json + figuras. `--sintetico` roda o pipeline COMPLETO em dados
fabricados com verdade conhecida e verifica que cada veredito dispara nos dois
sentidos (regra do 5º parecer: análise testada antes da run).

Unidade estatística: a SEED (Bouthillier). Testes de sinais binomiais exatos
(sem scipy). Todos os números vêm do config; nada é escolhido pós-execução.
"""
import argparse
import glob
import gzip
import json
import os
from math import comb, sqrt

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(AQUI, "config_stage4.json")))
ALPHA = CFG["P_A"]["alpha"]
Q, Q_J = CFG["regua"]["q"], CFG["regua"]["q_j"]
MIN_LIMPOS = CFG["regua"]["min_limpos_por_celula"]
MIN_SEED_PLACEBO = CFG["P_A"]["placebo"]["min_por_seed"]
MIN_EP_SEED = CFG["P_A"]["min_ep_por_seed"]


def phi_v(m):
    return 1.0 / (4.0 * sqrt(m))


def crit_binomial(S, alpha):
    for k in range(S + 1):
        if sum(comb(S, j) for j in range(k, S + 1)) / 2 ** S < alpha:
            return k
    return S + 1


def p_binomial_bilateral(n_pos, S):
    if S == 0:
        return 1.0
    cauda = sum(comb(S, j) for j in range(n_pos, S + 1)) / 2 ** S
    cauda_b = sum(comb(S, j) for j in range(0, n_pos + 1)) / 2 ** S
    return min(1.0, 2 * min(cauda, cauda_b))


def eh_limpo(ep, m):
    return (ep["alheios"] <= Q and not ep["v_cruzou"]
            and ep["v_min_abs"] >= phi_v(m) and ep["j_trocas"] <= Q_J)


def completo(ep):
    return ep["t_res"] >= 0


def carrega(resdir):
    dados = {}
    for arq in glob.glob(os.path.join(resdir, "*.json.gz")):
        with gzip.open(arq, "rt") as f:
            d = json.load(f)
        dados[(d["braco"], d["m"], d["seed"])] = d
    return dados


def celula(dados, braco, m):
    return {s: dados[(braco, m, s)] for s in CFG["grade"]["seeds"]
            if (braco, m, s) in dados}


# ---------------- P-A: memória — nulo CALIBRADO por bootstrap paramétrico ----
# 7º parecer (B-novo): o instrumento trunca t_area em t_res (para de medir na
# ressurreição) ⟹ (i) t_sinal ≤ t_area ≤ t_res mecanicamente — "área vence
# sinal" é DESCRITIVO, não MATA; (ii) o placebo de atraso-permutado cru não
# herda o truncamento e infla o nível (FPR até 58× α medido). Conserto: o nulo
# é SIMULADO com o truncamento — H0: atraso ⊥ duração, atrasos iid de F̂
# (Kaplan-Meier; não-disparo da área = censura à direita na duração da própria
# morte) — e T_obs é comparada à distribuição nula simulada: o viés estrutural
# fica DENTRO da calibração. Nível/potência demonstrados no --sintetico.

def fracao_vitorias(eps):
    """DESCRITIVO apenas (mecânico: área nunca perde quando ambos disparam)."""
    v, tot = 0, 0
    for ep in eps:
        if ep["t_area"] < 0 or ep["t_sinal"] < 0:
            continue
        tot += 1
        v += int(abs(ep["t_res"] - ep["t_area"]) < abs(ep["t_res"] - ep["t_sinal"]))
    return (v / tot) if tot else None, tot


def _arrays_seed(eps):
    dur = np.array([e["t_res"] - e["t_morte"] for e in eps], float)
    delay = np.array([e["t_area"] - e["t_morte"] if e["t_area"] >= 0 else -1.0
                      for e in eps], float)
    return dur, delay, delay >= 0


def _win_frac(dur, delay, fired, rng):
    """Fração de vitórias área-vs-placebo-permutado numa seed (None se <2)."""
    com = np.where(fired)[0]
    if len(com) < MIN_SEED_PLACEBO:
        return None
    d, a = dur[com], delay[com]
    n = len(com)
    j = rng.integers(0, n - 1, size=n)
    j = j + (j >= np.arange(n))
    return float(np.mean((d - a) < np.abs(d - a[j])))


def _km(obs, event):
    """CDF de Kaplan-Meier do atraso (censura à direita = não-disparo).
    Retorna (tempos de evento, F acumulada); massa além do último evento
    permanece censurada (atraso = ∞ ⟹ não-disparo na amostragem)."""
    ordem = np.argsort(obs, kind="stable")
    obs, event = np.asarray(obs)[ordem], np.asarray(event)[ordem].astype(bool)
    S, ts, Fs = 1.0, [], []
    n, i = len(obs), 0
    while i < n:
        t, em_risco, mortes = obs[i], n - i, 0
        while i < n and obs[i] == t:
            mortes += int(event[i]); i += 1
        if mortes:
            S *= 1.0 - mortes / em_risco
            ts.append(float(t)); Fs.append(1.0 - S)
    return np.array(ts), np.array(Fs)


def _amostra_km(ts, Fs, rng, n):
    if len(ts) == 0:
        return np.full(n, np.inf)
    u = rng.random(n)
    idx = np.searchsorted(Fs, u, side="left")
    return np.where(idx < len(ts), ts[np.minimum(idx, len(ts) - 1)], np.inf)


def _T_stat(dados_seeds, rngs):
    fr = [_win_frac(d, a, f, r) for (d, a, f), r in zip(dados_seeds, rngs)]
    fr = [x for x in fr if x is not None]
    return (float(np.mean(fr)) if fr else None), len(fr)


def p_calibrado(dados_seeds, ids, chave, B):
    """p-valor do bootstrap paramétrico sob o nulo truncado. dados_seeds:
    lista de (dur, atraso, disparou) por seed; ids: identificadores das seeds
    (entram no RNG observacional p/ determinismo); chave: (braço, célula)."""
    rngs_obs = [np.random.default_rng(
        [CFG["P_A"]["placebo"]["rng_seed"], *chave, int(s)]) for s in ids]
    T_obs, S_ef = _T_stat(dados_seeds, rngs_obs)
    if T_obs is None:
        return None, None, 0
    obs = np.concatenate([np.where(f, a, d) for d, a, f in dados_seeds])
    ev = np.concatenate([f for _, _, f in dados_seeds])
    ts, Fs = _km(obs, ev)
    rng_mc = np.random.default_rng([CFG["P_A"]["rng_seed_mc"], *chave])
    excede = 0
    for _ in range(B):
        sims = []
        for d, _, _ in dados_seeds:
            atr = _amostra_km(ts, Fs, rng_mc, len(d))
            f2 = atr <= d
            sims.append((d, np.where(f2, atr, -1.0), f2))
        T_b, _ = _T_stat(sims, [rng_mc] * len(sims))
        if T_b is not None and T_b >= T_obs:
            excede += 1
    return (1 + excede) / (B + 1), T_obs, S_ef


def analise_PA(dados, braco):
    ib = [b["nome"] for b in CFG["grade"]["bracos"]].index(braco)
    B = CFG["P_A"]["B"]
    out = {}
    for im, m in enumerate(CFG["grade"]["m"]):
        cel = celula(dados, braco, m)
        dados_seeds, ids, fr_sinal = [], [], []
        n_limpos = sem_area = 0
        for s in sorted(cel):
            eps = sorted(
                (ep for ep in cel[s]["episodios"]
                 if completo(ep) and eh_limpo(ep, m)),
                key=lambda e: (e["t_morte"], e["k"]))
            n_limpos += len(eps)
            sem_area += sum(1 for ep in eps if ep["t_area"] < 0)
            if len(eps) < MIN_EP_SEED:
                continue
            f, _ = fracao_vitorias(eps)
            fr_sinal.append(None if f is None else round(f, 3))
            dados_seeds.append(_arrays_seed(eps))
            ids.append(s)
        p, T_obs, S_ef = (p_calibrado(dados_seeds, ids, (ib, im), B)
                          if dados_seeds else (None, None, 0))
        out[m] = dict(
            n_limpos=n_limpos, valida=bool(n_limpos >= MIN_LIMPOS),
            frac_sem_area=round(sem_area / max(n_limpos, 1), 3),
            T_obs=None if T_obs is None else round(T_obs, 4),
            p_calibrado=None if p is None else round(p, 4), S_efetivo=S_ef,
            fracs_sinal_descritivo=fr_sinal,
            passa=bool(p is not None and p < ALPHA))
    validas = [m for m in out if out[m]["valida"]]
    passam = [m for m in validas if out[m]["passa"]]
    if not validas:
        veredito = "limpos-insuficientes (vira P-E)"
    elif len(passam) >= 2 and 16 in passam:
        veredito = "MEMORIA DEMONSTRADA"
    elif not passam:
        veredito = "MATA: memoria/entry-exit nao sobrevive na rede real"
    else:
        veredito = f"parcial (passa em {passam} de {validas})"
    return dict(celulas={str(m): out[m] for m in out}, veredito=veredito)


def teste_sinais(fracs, alpha):
    """Usado por P-B (Δ pareado entre braços), não mais por P-A."""
    validos = [f for f in fracs if f is not None]
    S = len(validos)
    k = crit_binomial(S, alpha) if S else 999
    n_pos = sum(1 for f in validos if f > 0.5)
    return dict(S=S, n_pos=n_pos, k_critico=k, passa=bool(S and n_pos >= k))


# ---------------- P-B: universalidade in-class ----------------
def stats_seed(d, m):
    eps = d["episodios"]
    n_tot = len(eps)
    n_cens = sum(1 for ep in eps if not completo(ep))
    limpos = [ep for ep in eps if completo(ep) and eh_limpo(ep, m)]
    med = (float(np.median([ep["t_res"] - ep["t_morte"] for ep in limpos]))
           if limpos else None)
    return dict(frac_censurada=(n_cens / n_tot) if n_tot else None,
                mediana_dur=med, n_limpos=len(limpos))


def analise_PB(dados, b1="softplus", b2="erf"):
    piso = CFG["P_B"]["piso_rel"]
    alpha_e = ALPHA / len(CFG["P_B"]["estatisticas"])
    out, celulas_mata = {}, 0
    for m in CFG["grade"]["m"]:
        c1, c2 = celula(dados, b1, m), celula(dados, b2, m)
        seeds = sorted(set(c1) & set(c2))
        gate = all(sum(stats_seed(c[s], m)["n_limpos"] for s in seeds)
                   >= CFG["P_B"]["gate_min_limpos"] for c in (c1, c2))
        res_e = {}
        mata_cel = False
        for est in CFG["P_B"]["estatisticas"]:
            d1 = [stats_seed(c1[s], m)[est] for s in seeds]
            d2 = [stats_seed(c2[s], m)[est] for s in seeds]
            pares = [(a, b) for a, b in zip(d1, d2)
                     if a is not None and b is not None and (a != b)]
            n_pos = sum(1 for a, b in pares if b > a)
            p = p_binomial_bilateral(max(n_pos, len(pares) - n_pos), len(pares))
            m1 = float(np.median([a for a, _ in pares])) if pares else 0.0
            m2 = float(np.median([b for _, b in pares])) if pares else 0.0
            rel = abs(m2 - m1) / max(abs(m1), 1e-12) if pares else 0.0
            sig = bool(p < alpha_e)
            res_e[est] = dict(S=len(pares), p=round(p, 5), delta_rel=round(rel, 4),
                              sig=sig, acima_do_piso=bool(rel > piso))
            mata_cel = mata_cel or (sig and rel > piso)
        mata_cel = mata_cel and gate            # célula sem gate não conta p/ MATA
        celulas_mata += int(mata_cel)
        out[str(m)] = dict(estatisticas=res_e, gate_min_limpos=bool(gate),
                           mata_celula=bool(mata_cel))
    veredito = ("MATA: L5.2 nao ergue (universalidade falha in-class)"
                if celulas_mata >= 2 else "universalidade in-class sustentada")
    return dict(celulas=out, celulas_mata=celulas_mata, veredito=veredito)


# ---------------- P-C: âncora hard vs suave (estimação) ----------------
def analise_PC(dados):
    out = {}
    for m in CFG["grade"]["m"]:
        linha = {}
        for braco in ("relu", "softplus", "erf", "gelu"):
            cel = celula(dados, braco, m)
            fr = [stats_seed(cel[s], m)["frac_censurada"] for s in sorted(cel)]
            fr = [f for f in fr if f is not None]
            if fr:
                linha[braco] = dict(
                    mediana=round(float(np.median(fr)), 3),
                    q25=round(float(np.percentile(fr, 25)), 3),
                    q75=round(float(np.percentile(fr, 75)), 3), S=len(fr))
        out[str(m)] = linha
    return dict(frac_censurada_por_braco=out,
                nota="ancora da nao-comutacao; SEM MATA (6o parecer)")


# ---------------- P-D: back-reaction populacional ----------------
def analise_PD(dados):
    rng = np.random.default_rng(CFG["P_D"]["rng_seed"])
    B = CFG["P_D"]["B_boot"]
    out = {}
    for m in CFG["grade"]["m"]:
        cf, cc = celula(dados, "freeze", m), celula(dados, "softplus", m)
        med_seed, fr_rev_f, fr_rev_c = [], [], []
        for s in sorted(set(cf) & set(cc)):
            cong = set(cf[s]["congelados"])
            def primeiro(d):
                return {ep["k"]: ep for ep in d["episodios"] if ep["t_morte"] == 0}
            ef, ec = primeiro(cf[s]), primeiro(cc[s])
            alvo = [k for k in ef if k not in cong and k in ec]
            if not alvo:
                continue
            dts = [ef[k]["t_res"] - ec[k]["t_res"] for k in alvo
                   if completo(ef[k]) and completo(ec[k])]
            if dts:
                med_seed.append(float(np.median(dts)))
            fr_rev_f.append(np.mean([completo(ef[k]) for k in alvo]))
            fr_rev_c.append(np.mean([completo(ec[k]) for k in alvo]))
        if med_seed:
            arr = np.array(med_seed)
            boots = np.array([np.median(arr[rng.integers(0, len(arr), len(arr))])
                              for _ in range(B)])
            ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
            mediana = float(np.median(arr))
        else:
            ci, mediana = (0.0, 0.0), 0.0
        dur = [ep["t_res"] - ep["t_morte"] for s in sorted(cc)
               for ep in cc[s]["episodios"] if completo(ep) and eh_limpo(ep, m)]
        piso = 0.01 * float(np.median(dur)) if dur else 0.0
        contem0 = bool(ci[0] <= 0.0 <= ci[1])
        if len(med_seed) < CFG["P_D"]["S_min_sem"]:
            classe = "indeterminada"       # 7º parecer: "sem" exige S mínimo
        elif contem0 and abs(mediana) < piso:
            classe = "sem"                 # MATA-condição do README §3
        elif not contem0 and abs(mediana) >= piso:
            classe = "detectada"
        else:
            classe = "indeterminada"
        out[str(m)] = dict(
            S=len(med_seed), mediana_delta=round(mediana, 1),
            ci95=[round(ci[0], 1), round(ci[1], 1)], piso=round(piso, 1),
            frac_revive_freeze=round(float(np.mean(fr_rev_f)), 3) if fr_rev_f else None,
            frac_revive_controle=round(float(np.mean(fr_rev_c)), 3) if fr_rev_c else None,
            classe=classe)
    n_det = sum(1 for m in out if out[m]["classe"] == "detectada")
    if all(out[m]["classe"] == "sem" for m in out):
        veredito = "MATA: sem back-reaction populacional detectavel"
    elif n_det:
        veredito = f"back-reaction detectada em {n_det} de {len(out)} celulas"
    else:
        veredito = "indeterminada (CI largo ou |Delta| entre piso e significancia)"
    return dict(celulas=out, veredito=veredito)


# ---------------- P-E: estimação (superfícies) ----------------
def analise_PE(dados, pa):
    out = {}
    for braco in [b["nome"] for b in CFG["grade"]["bracos"]]:
        for m in CFG["grade"]["m"]:
            cel = celula(dados, braco, m)
            if not cel:
                continue
            tot = cens = limp = saida = cruzou = 0
            durs, t_ress, tms, atrs, tms_a = [], [], [], [], []
            for s in cel:
                for ep in cel[s]["episodios"]:
                    tot += 1
                    cens += int(not completo(ep))
                    saida += int(ep["v_min_abs"] < phi_v(m))
                    cruzou += int(ep["v_cruzou"])
                    if completo(ep) and eh_limpo(ep, m):
                        limp += 1
                        durs.append(ep["t_res"] - ep["t_morte"])
                        t_ress.append(ep["t_res"]); tms.append(ep["t_morte"])
                        if ep["t_area"] >= 0:
                            atrs.append(ep["t_area"] - ep["t_morte"])
                            tms_a.append(ep["t_morte"])
            # descritivo pré-registrado (7º parecer, 2ª rodada): tendência
            # temporal comum — dur e atraso correlacionados com t_morte fariam
            # o nulo iid rejeitar sem memória por-episódio; reportar desarma
            def _corr(a, b):
                return (round(float(np.corrcoef(a, b)[0, 1]), 3)
                        if len(a) >= 3 and np.std(a) > 0 and np.std(b) > 0
                        else None)
            out[f"{braco}-m{m}"] = dict(
                episodios=tot, frac_censurada=round(cens / tot, 3) if tot else None,
                frac_limpa=round(limp / max(tot - cens, 1), 3),
                frac_morte_por_saida=round(saida / tot, 3) if tot else None,
                frac_v_cruzou=round(cruzou / tot, 3) if tot else None,
                mediana_dur=float(np.median(durs)) if durs else None,
                mediana_t_res=float(np.median(t_ress)) if t_ress else None,
                corr_dur_tmorte=_corr(tms, durs),
                corr_atraso_tmorte=_corr(tms_a, atrs))
    qual = {str(m): dict(T_obs=pa["celulas"][str(m)]["T_obs"],
                         p_calibrado=pa["celulas"][str(m)]["p_calibrado"],
                         frac_sem_area=pa["celulas"][str(m)]["frac_sem_area"])
            for m in CFG["grade"]["m"] if str(m) in pa["celulas"]}
    return dict(superficies=out, qualidade_area_por_m=qual)


# ---------------- P-NULL: validade do instrumento ----------------
def analise_PNULL(dados):
    pa = analise_PA(dados, braco="semgrad")
    passou_algum = any(c["passa"] for c in pa["celulas"].values())
    return dict(celulas=pa["celulas"],
                instrumento_valido=bool(not passou_algum),
                veredito=("INSTRUMENTO VICIADO — experimento INVALIDO"
                          if passou_algum else
                          "instrumento ok (difusao pura nao passa no nulo calibrado)"))


def roda_analise(dados):
    pa = analise_PA(dados, "softplus")
    return dict(P_A=pa, P_B=analise_PB(dados), P_C=analise_PC(dados),
                P_D=analise_PD(dados), P_E=analise_PE(dados, pa),
                P_NULL=analise_PNULL(dados))


# ================= MODO SINTÉTICO (valida a análise, não o mundo) ============
def _fabrica(braco, m, seed, modo, shift_pd=0, shift_pb=1.0):
    """Pareamento por-(m,seed,i): mesmos sorteios em todos os braços (modela o
    RNG alinhado do harness); diferenças de braço são aplicadas POR CIMA."""
    eps, cong = [], []
    n_ep = 12
    if braco == "freeze":
        cong = [0, 2]
    for i in range(n_ep):
        rng = np.random.default_rng([4242, m, seed, i])
        nasce_morto = i < 5
        t0 = 0 if nasce_morto else int(rng.integers(1, 40000))
        dur = int(rng.lognormal(8.0, 0.3)) + 200
        if braco == "freeze" and i in cong:
            t_res = -1
        else:
            t_res = t0 + dur + (shift_pd if braco == "freeze" else 0)
        censura = t_res < 0 or t_res > 100000 or (braco == "relu")
        if braco == "erf":
            t_res = t0 + int((t_res - t0) * shift_pb) if t_res >= 0 else t_res
        # INVARIANTE DO INSTRUMENTO (7º parecer): a régua para de medir na
        # ressurreição — atraso que excede a duração vira NÃO-DISPARO (-1),
        # jamais t_area > t_res. O nulo fabricado herda o truncamento real.
        dur_ef = (t_res - t0) if not censura else (100000 - t0)
        if modo == "memoria":
            atr_a, atr_s = int(0.9 * dur), int(0.2 * dur)
        else:                              # nulo: atraso independente da duração
            atr_a = int(rng.lognormal(8.0, 0.3)) + 200
            atr_s = int(rng.lognormal(7.0, 0.3)) + 100
        t_area = t0 + atr_a if atr_a <= dur_ef else -1
        t_sinal = t0 + atr_s if atr_s <= dur_ef else -1
        ep = dict(k=i, t_morte=t0, t_area=t_area, t_sinal=t_sinal,
                  t_res=-1 if censura else t_res, v_morte=0.5, v_min_abs=0.5,
                  v_cruzou=False, j_trocas=0, alheios=0)
        if censura:
            ep["censurado_em"] = 100000
        eps.append(ep)
    return dict(braco=braco, m=m, seed=seed, loss=0.01, passos=100000,
                eta=3e-4, T=1e-3, congelados=cong, n_switches=2 * n_ep,
                episodios=eps, segundos=0.0)


def nivel_potencia():
    """Demonstração exigida pelo 7º parecer: FPR≈α do teste calibrado sob o
    nulo TRUNCADO para n_ep/seed ∈ {8,40,200}; TPR sob mundo-memória."""
    print("-- nível (FPR) do teste calibrado sob nulo truncado, α=0.01 --")
    B, R, S = 199, 100, 10
    for n_ep in (8, 40, 200):
        rej = 0
        for r in range(R):
            rng = np.random.default_rng([9090, n_ep, r])
            ds, ids = [], []
            for s in range(S):
                dur = rng.lognormal(8.0, 0.5, n_ep) + 200
                atr = rng.lognormal(7.8, 0.6, n_ep) + 200
                f = atr <= dur
                ds.append((dur, np.where(f, atr, -1.0), f)); ids.append(s)
            p, _, _ = p_calibrado(ds, ids, (99, n_ep), B)
            rej += int(p is not None and p < ALPHA)
        fpr = rej / R
        print(f"   n_ep/seed={n_ep:3d}: FPR={fpr:.3f} (R={R}, B={B})")
        assert fpr <= 0.05, (n_ep, fpr)
    print("-- potência (TPR) sob memória (atraso=0.9·dur, σ_rel=5%) --")
    rej = 0
    Rp = 40
    for r in range(Rp):
        rng = np.random.default_rng([9191, r])
        ds, ids = [], []
        for s in range(S):
            dur = rng.lognormal(8.0, 0.5, 40) + 200
            atr = np.minimum(0.9 * dur * (1 + 0.05 * rng.standard_normal(40)), dur)
            ds.append((dur, atr, np.ones(40, bool))); ids.append(s)
        p, _, _ = p_calibrado(ds, ids, (98, r), B)
        rej += int(p is not None and p < ALPHA)
    tpr = rej / Rp
    print(f"   TPR={tpr:.3f} (R={Rp}, B={B})")
    assert tpr >= 0.9, tpr
    print("[OK] nível segurado (FPR≈α com truncamento) e potência preservada")


def sintetico():
    print("== ANÁLISE SINTÉTICA (verdade conhecida) ==")
    nivel_potencia()
    bracos = [b["nome"] for b in CFG["grade"]["bracos"]]

    def mundo(modo, shift_pd=0, shift_pb=1.0, modo_semgrad=None):
        return {(b, m, s): _fabrica(b, m, s,
                                    modo_semgrad if (b == "semgrad" and
                                                     modo_semgrad) else modo,
                                    shift_pd, shift_pb)
                for b in bracos for m in CFG["grade"]["m"]
                for s in CFG["grade"]["seeds"]}

    # 1) mundo-memória: P-A demonstra; P-NULL ok (semgrad nulo); P-B sem MATA
    r = roda_analise(mundo("memoria", modo_semgrad="nulo"))
    assert r["P_A"]["veredito"] == "MEMORIA DEMONSTRADA", r["P_A"]["veredito"]
    assert r["P_NULL"]["instrumento_valido"], r["P_NULL"]
    assert "sustentada" in r["P_B"]["veredito"], r["P_B"]["veredito"]
    print("[OK] mundo-memória: P-A=DEMONSTRADA, P-NULL=ok, P-B=sem MATA")
    # 2) mundo-nulo: P-A MATA (área não vence placebo)
    r2 = roda_analise(mundo("nulo"))
    assert r2["P_A"]["veredito"].startswith("MATA"), r2["P_A"]["veredito"]
    print("[OK] mundo-nulo TRUNCADO: P-A dispara MATA (o nulo calibrado por "
          "KM absorve o viés estrutural do truncamento)")
    # 3) semgrad em mundo-memória: P-NULL acusa instrumento viciado
    r3 = roda_analise(mundo("memoria", modo_semgrad="memoria"))
    assert not r3["P_NULL"]["instrumento_valido"]
    print("[OK] P-NULL acusa instrumento viciado quando difusão-pura 'prevê'")
    # 4) P-B: erf com t_res 20% deslocado dispara MATA
    r4 = roda_analise(mundo("memoria", shift_pb=1.25, modo_semgrad="nulo"))
    assert r4["P_B"]["veredito"].startswith("MATA"), r4["P_B"]["veredito"]
    print("[OK] P-B dispara MATA com deslocamento de 25% no braço erf")
    # 5) P-D: shift de +3000 passos nos não-congelados do freeze é detectado
    r5 = roda_analise(mundo("memoria", shift_pd=3000, modo_semgrad="nulo"))
    assert "detectada" in r5["P_D"]["veredito"], r5["P_D"]["veredito"]
    assert r["P_D"]["veredito"].startswith("MATA"), r["P_D"]["veredito"]
    print("[OK] P-D: detecta Δ=3000 e dá MATA com Δ=0")
    # 6) chaves completas e serializável
    json.dumps(r)
    for chave in ("P_A", "P_B", "P_C", "P_D", "P_E", "P_NULL"):
        assert chave in r
    print("[OK] resultados serializáveis, todas as chaves presentes")
    print("== ANÁLISE VALIDADA — apta para o smoke-gate do pré-registro ==")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintetico", action="store_true")
    a = ap.parse_args()
    if a.sintetico:
        sintetico()
    else:
        dados = carrega(os.path.join(AQUI, "results"))
        if not dados:
            raise SystemExit("sem results/ — rodar run.py após o pré-registro")
        r = roda_analise(dados)
        with open(os.path.join(AQUI, "results-summary.json"), "w") as f:
            json.dump(r, f, indent=1, ensure_ascii=False)
        for p in ("P_A", "P_B", "P_D", "P_NULL"):
            print(f"{p}: {r[p]['veredito']}")
        print("results-summary.json escrito")
