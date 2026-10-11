"""Etapa-4a.2 / trilho-2 — NULO ESTRATIFICADO POR NEURÔNIO (arquivo NOVO).

Contexto [RESULTADO 2026-07-25]: o nulo calibrado do 4a (`analysis.py:p_calibrado`,
KM POOLED, H0 iid dentro-da-célula) foi enganado por heterogeneidade por-neurônio
— P-NULL disparou em m=16 (semgrad, p=0.002). Diagnóstico congelado em
`docs/research-notes/2026-07-24-stage4a-invalid-instrument.md`: com poucos neurônios cada
k repete MUITOS episódios com assinatura própria de (duração, atraso); pooled
esses episódios, (dur, atraso) co-variam ENTRE neurônios e a estatística de
vitória-de-área acusa "memória" que é só a assinatura de k. O nulo iid amostra
atrasos ⊥ duração ⟹ T_b≈0.5 ⟹ rejeita T_obs inflado ⟹ falso-positivo.

ESTE ARQUIVO NÃO TOCA em nada pré-registrado: importa `analysis.py` SEM modificar
e só REUSA suas funções de baixo nível (`_win_frac`, `_T_stat`, `_km`,
`_amostra_km`, `p_calibrado`, `ALPHA`, `MIN_SEED_PLACEBO`).

------------------------------------------------------------------------------
DESENHO DO NULO ESTRATIFICADO (escolha justificada)
------------------------------------------------------------------------------
Primário — PERMUTAÇÃO DE ATRASOS DENTRO DO MESMO NEURÔNIO. Para cada k com ≥2
episódios, os atrasos-brutos dos episódios de k são permutados ENTRE SI (durações
fixas); truncamento reaplicado por cima (atraso permutado > duração daquele
episódio ⟹ não-disparo). Neurônios com <2 episódios ficam fixos (não geram
variabilidade nula — o task pede "cair"; equivalente: permutação-identidade).

Por que este é o nulo certo (e não o pooled):
  * A lei T2′ (memória / entry-exit) é POR NEURÔNIO: para um k FIXO, através de
    seus episódios, o atraso da área deve acompanhar a duração DAQUELE episódio.
    Logo memória = associação (dur, atraso) DENTRO de k.
  * A co-variação de (dur, atraso) ENTRE neurônios (k grandes vivem mais e
    disparam mais tarde) é um CONFUNDIDOR, não a lei. A permutação dentro-de-k
    preserva EXATAMENTE as marginais por-neurônio e a co-variação entre-neurônios,
    e destrói só o pareamento episódio-a-episódio dentro de k.
  * Sob H0 "atraso ⊥ dur dentro de k", os atrasos de um neurônio são
    permutáveis (exchangeable) ⟹ o teste é EXATO em nível, sem hipótese
    distribucional (o KM pooled supõe uma F̂ única para toda a célula — é essa
    suposição que quebra).
  * O truncamento do instrumento (a régua para de medir na ressurreição) é
    herdado: reaplico atraso≤dur após permutar.

Complemento/robustez — BOOTSTRAP KM POR-NEURÔNIO COM SHRINKAGE ao pooled
(`p_estratificado_shrink`): amostra atrasos de F̂_k (KM do próprio k) com peso
n_k/(n_k+κ) e do pooled com peso κ/(n_k+κ), truncando por dur. Preserva a escala
de atraso de cada k (mantém a co-variação entre-neurônios) mas força
atraso⊥dur. Rodado como checagem secundária; a permutação é a manchete por ser
exata e sem parâmetro κ.

------------------------------------------------------------------------------
Rodar:  cd experiments/mlp
        nice -19 ../../venv/bin/python stratified_null.py            # tabela completa
        nice -19 ../../venv/bin/python stratified_null.py --smoke    # grade curta
        nice -19 ../../venv/bin/python stratified_null.py --real     # aplica ao semgrad real
"""
import argparse
import sys
import time

import numpy as np

import analysis as A  # SEM modificar; só reuso
from analysis import ALPHA, MIN_SEED_PLACEBO, _T_stat, _win_frac, p_calibrado

# chaves de RNG (não colidem com as do 4a: 777/779 pré-registrados)
K_WORLD, K_OBS, K_MC = 5100, 5200, 5300


# ============================================================================
# NULO ESTRATIFICADO — permutação de atrasos dentro do neurônio
# ============================================================================
def _perm_atraso_dentro_k(delay_raw, k, rng):
    """Permuta os atrasos-brutos entre episódios do MESMO neurônio (durações
    ficam no lugar). Neurônios com <2 episódios não mudam."""
    out = delay_raw.copy()
    for kk in np.unique(k):
        idx = np.where(k == kk)[0]
        if idx.size >= 2:
            out[idx] = delay_raw[idx[rng.permutation(idx.size)]]
    return out


def _obs_seeds(dados_seeds_k):
    """(dur, delay_bruto, k) -> (dur, delay_trunc, fired) no formato de _win_frac."""
    out = []
    for dur, dbruto, _k in dados_seeds_k:
        fired = dbruto <= dur
        out.append((dur, np.where(fired, dbruto, -1.0), fired))
    return out


def p_estratificado(dados_seeds_k, ids, chave, R):
    """p-valor do teste de PERMUTAÇÃO dentro-do-neurônio.
    dados_seeds_k: lista por seed de (dur, delay_bruto, k) — delay_bruto ANTES do
    truncamento (o truncamento é reaplicado). chave: tupla de ints (determinismo).
    Mesma estatística T do pooled (média entre seeds da fração-de-vitórias
    área-vs-placebo, `_win_frac`) — só o NULO muda."""
    rngs_obs = [np.random.default_rng([K_OBS, *chave, int(s)]) for s in ids]
    T_obs, S_ef = _T_stat(_obs_seeds(dados_seeds_k), rngs_obs)
    if T_obs is None:
        return None, None, 0
    rng_mc = np.random.default_rng([K_MC, *chave])
    excede = 0
    for _ in range(R):
        sims = []
        for dur, dbruto, k in dados_seeds_k:
            dp = _perm_atraso_dentro_k(dbruto, k, rng_mc)
            f = dp <= dur
            sims.append((dur, np.where(f, dp, -1.0), f))
        T_b, _ = _T_stat(sims, [rng_mc] * len(sims))
        if T_b is not None and T_b >= T_obs:
            excede += 1
    return (1 + excede) / (R + 1), T_obs, S_ef


# ----- complemento: bootstrap KM por-neurônio com shrinkage ao pooled --------
def p_estratificado_shrink(dados_seeds_k, ids, chave, R, kappa=4.0):
    """Nulo secundário: reamostra atraso de F̂_k (KM do próprio neurônio) com
    shrinkage w=n_k/(n_k+κ) ao pooled. Preserva escala-de-atraso por-k, força
    atraso⊥dur. Truncamento reaplicado por dur."""
    obs = _obs_seeds(dados_seeds_k)
    rngs_obs = [np.random.default_rng([K_OBS, *chave, int(s)]) for s in ids]
    T_obs, S_ef = _T_stat(obs, rngs_obs)
    if T_obs is None:
        return None, None, 0
    # pooled (fallback do shrinkage): atraso observado (=delay se fired, senão dur=censura)
    pool_obs = np.concatenate([np.where(f, dly, dur) for (dur, dly, f) in obs])
    pool_ev = np.concatenate([f for (_, _, f) in obs])
    ts_p, Fs_p = A._km(pool_obs, pool_ev)
    # KM por-neurônio: acumula (obs, event) por k através de todas as seeds
    km_k = {}
    acc = {}
    for dur, dbruto, k in dados_seeds_k:
        fired = dbruto <= dur
        obs_atraso = np.where(fired, dbruto, dur)
        for kk in np.unique(k):
            m = k == kk
            acc.setdefault(int(kk), [[], []])
            acc[int(kk)][0].extend(obs_atraso[m].tolist())
            acc[int(kk)][1].extend(fired[m].astype(int).tolist())
    for kk, (o, e) in acc.items():
        km_k[kk] = A._km(np.array(o, float), np.array(e, int))
    rng_mc = np.random.default_rng([K_MC + 1, *chave])
    excede = 0
    for _ in range(R):
        sims = []
        for dur, dbruto, k in dados_seeds_k:
            atr = np.empty(dur.size)
            for i in range(dur.size):
                kk = int(k[i])
                ts_k, Fs_k = km_k.get(kk, (np.array([]), np.array([])))
                n_k = len(acc[kk][0]) if kk in acc else 0
                w = n_k / (n_k + kappa)
                if ts_k.size and rng_mc.random() < w:
                    atr[i] = A._amostra_km(ts_k, Fs_k, rng_mc, 1)[0]
                else:
                    atr[i] = A._amostra_km(ts_p, Fs_p, rng_mc, 1)[0]
            f = atr <= dur
            sims.append((dur, np.where(f, atr, -1.0), f))
        T_b, _ = _T_stat(sims, [rng_mc] * len(sims))
        if T_b is not None and T_b >= T_obs:
            excede += 1
    return (1 + excede) / (R + 1), T_obs, S_ef


# ============================================================================
# TRÊS MUNDOS-NULO (truncados) + mundo-MEMÓRIA
# ============================================================================
def mundo_iid(rng, n_ep, n_k):
    """(i) iid — o antigo. Atraso ⊥ duração, sem estrutura de neurônio.
    Espelha o nulo do `analise.nivel_potencia` (dur~LN(8,.5)+200, atr~LN(7.8,.6)+200)."""
    dur = rng.lognormal(8.0, 0.5, n_ep) + 200.0
    delay = rng.lognormal(7.8, 0.6, n_ep) + 200.0
    k = np.arange(n_ep) % n_k
    return dur, delay, k


def mundo_hetero(rng, n_ep, n_k, sigma_mu=0.7):
    """(ii) heterogêneo-por-neurônio: cada k tem F_k própria (locação lognormal
    mu_k sorteada por k). dur e atraso do MESMO episódio são INDEPENDENTES dado k
    (⟹ sem memória por-episódio), mas ambos escalam com mu_k ⟹ co-variação
    ENTRE-neurônios ⟹ o pooled vê 'memória' espúria. DEVE quebrar o nulo antigo."""
    k = np.arange(n_ep) % n_k
    mu = rng.normal(8.0, sigma_mu, n_k)
    z1 = rng.standard_normal(n_ep)
    z2 = rng.standard_normal(n_ep)
    dur = np.exp(mu[k] + 0.35 * z1) + 200.0
    delay = np.exp((mu[k] - 0.10) + 0.35 * z2) + 200.0
    return dur, delay, k


def _ar1(rng, L, rho):
    e = np.empty(L)
    e[0] = rng.standard_normal()
    s = np.sqrt(1.0 - rho * rho)
    for t in range(1, L):
        e[t] = rho * e[t - 1] + s * rng.standard_normal()
    return e


def mundo_autocorr(rng, n_ep, n_k, rho=0.5, sigma_mu=0.7):
    """(iii) autocorrelacionado: episódios CONSECUTIVOS do mesmo k correlacionados
    (AR(1), ρ≈0.5) em log-escala, com heterogeneidade por-k. dur e atraso usam
    fluxos AR(1) INDEPENDENTES ⟹ sem memória por-episódio, mas a permutação
    dentro-de-k não é iid (exchangeabilidade estressada). Testa se o estratificado
    aguenta serial-dependência."""
    k = np.arange(n_ep) % n_k
    mu = rng.normal(8.0, sigma_mu, n_k)
    dur = np.empty(n_ep)
    delay = np.empty(n_ep)
    for kk in range(n_k):
        idx = np.where(k == kk)[0]
        if idx.size == 0:
            continue
        ed = _ar1(rng, idx.size, rho)
        el = _ar1(rng, idx.size, rho)
        dur[idx] = np.exp(mu[kk] + 0.35 * ed) + 200.0
        delay[idx] = np.exp((mu[kk] - 0.10) + 0.35 * el) + 200.0
    return dur, delay, k


def mundo_memoria(rng, n_ep, n_k, sigma_mu=0.7):
    """MEMÓRIA (para TPR): atraso = 0.9·dur (±5%) DENTRO de cada episódio, com
    heterogeneidade por-k por cima. 0.9<1 ⟹ todos disparam."""
    k = np.arange(n_ep) % n_k
    mu = rng.normal(8.0, sigma_mu, n_k)
    z1 = rng.standard_normal(n_ep)
    dur = np.exp(mu[k] + 0.35 * z1) + 200.0
    delay = np.minimum(0.9 * dur * (1.0 + 0.05 * rng.standard_normal(n_ep)), dur)
    return dur, delay, k


# ============================================================================
# DRIVERS FPR / TPR
# ============================================================================
def _gera_seeds(world_fn, rng, n_ep, n_k, S):
    dsk, ids = [], []
    for s in range(S):
        dur, delay, k = world_fn(rng, n_ep, n_k)
        dsk.append((dur, delay, k))
        ids.append(s)
    return dsk, ids


def taxa(world_fn, n_ep, n_k, R, S, B, tag, incluir_shrink=False):
    """FPR (ou TPR) de pooled e estratificado sob um mundo, num n_ep."""
    rej_pool = rej_str = rej_shr = 0
    nval_pool = nval_str = nval_shr = 0
    for r in range(R):
        rng = np.random.default_rng([K_WORLD, tag, n_ep, r])
        dsk, ids = _gera_seeds(world_fn, rng, n_ep, n_k, S)
        pooled = _obs_seeds(dsk)
        chave = (tag, n_ep, r)
        p_p, _, _ = p_calibrado(pooled, ids, chave, B)
        p_s, _, _ = p_estratificado(dsk, ids, chave, R=B)
        if p_p is not None:
            nval_pool += 1
            rej_pool += int(p_p < ALPHA)
        if p_s is not None:
            nval_str += 1
            rej_str += int(p_s < ALPHA)
        if incluir_shrink:
            p_h, _, _ = p_estratificado_shrink(dsk, ids, chave, R=B)
            if p_h is not None:
                nval_shr += 1
                rej_shr += int(p_h < ALPHA)
    out = dict(pool=rej_pool / max(nval_pool, 1), strat=rej_str / max(nval_str, 1),
               n_pool=nval_pool, n_str=nval_str)
    if incluir_shrink:
        out["shrink"] = rej_shr / max(nval_shr, 1)
        out["n_shr"] = nval_shr
    return out


def tabela_fpr(R=100, S=12, B=199, n_k=4, ns=(8, 40, 200), shrink_em=None):
    print(f"\n== TABELA FPR (α={ALPHA}) — R={R} mundos/célula, S={S} seeds/mundo, "
          f"B={B} draws-nulos, n_k={n_k} neurônios ==")
    print("  (FPR = fração de mundos-nulo em que o teste REJEITA; alvo ≈ α; "
          "|FPR-α| esperado ~±0.5·√(α(1-α)/R))")
    mundos = [("iid", mundo_iid), ("hetero", mundo_hetero), ("autocorr", mundo_autocorr)]
    resultados = {}
    for tag, (nome, fn) in enumerate(mundos):
        print(f"\n-- mundo ({'i ii iii'.split()[tag]}) {nome} --")
        print(f"   {'n_ep/seed':>10} | {'FPR pooled':>11} | {'FPR estrat':>11}"
              + (" | FPR shrink" if shrink_em == nome else ""))
        resultados[nome] = {}
        for n_ep in ns:
            t0 = time.time()
            r = taxa(fn, n_ep, n_k, R, S, B, tag,
                     incluir_shrink=(shrink_em == nome))
            resultados[nome][n_ep] = r
            extra = f" | {r.get('shrink', float('nan')):>10.3f}" if shrink_em == nome else ""
            flag = "  <-- QUEBRA" if (nome != "iid" and r["pool"] > 2 * ALPHA
                                      and r["pool"] > r["strat"] + 0.02) else ""
            print(f"   {n_ep:>10d} | {r['pool']:>11.3f} | {r['strat']:>11.3f}"
                  f"{extra}{flag}   ({time.time()-t0:.0f}s)")
            sys.stdout.flush()
    return resultados


def tabela_tpr(R=60, S=12, B=199, n_k=4, ns=(8, 40, 200)):
    print(f"\n== TPR sob MEMÓRIA (atraso=0.9·dur ±5%) — R={R}, S={S}, B={B}, "
          f"n_k={n_k} ==")
    print("   (perda de potência do estratificado vs pooled é o preço do nível "
          "honesto)")
    print(f"   {'n_ep/seed':>10} | {'TPR pooled':>11} | {'TPR estrat':>11} | "
          f"{'perda':>7}")
    res = {}
    for n_ep in ns:
        t0 = time.time()
        r = taxa(mundo_memoria, n_ep, n_k, R, S, B, tag=9)
        res[n_ep] = r
        perda = r["pool"] - r["strat"]
        print(f"   {n_ep:>10d} | {r['pool']:>11.3f} | {r['strat']:>11.3f} | "
              f"{perda:>+7.3f}   ({time.time()-t0:.0f}s)")
        sys.stdout.flush()
    return res


# ============================================================================
# APLICAÇÃO AO SEMGRAD REAL (P-NULL disparou pooled p=0.002 em m=16)
# ============================================================================
def _extrai_real(braco, m):
    """Lê results/ (read-only) e monta dados_seeds_k para o braço/m: por seed,
    episódios limpos-completos ordenados, com (dur, atraso_obs, k). Atraso bruto
    real está truncado no horizonte; usa o atraso observado (t_area-t_morte)."""
    import os
    dados = A.carrega(os.path.join(A.AQUI, "results"))
    cel = A.celula(dados, braco, m)
    dsk, ids = [], []
    for s in sorted(cel):
        eps = sorted((ep for ep in cel[s]["episodios"]
                      if A.completo(ep) and A.eh_limpo(ep, m)),
                     key=lambda e: (e["t_morte"], e["k"]))
        if len(eps) < A.MIN_EP_SEED:
            continue
        dur = np.array([e["t_res"] - e["t_morte"] for e in eps], float)
        # atraso observado: t_area-t_morte se disparou, senão um valor > dur
        # (não-disparo) — para a permutação/pooled o truncamento decide fired
        delay = np.array([(e["t_area"] - e["t_morte"]) if e["t_area"] >= 0
                          else (e["t_res"] - e["t_morte"] + 1.0) for e in eps], float)
        k = np.array([e["k"] for e in eps], int)
        dsk.append((dur, delay, k))
        ids.append(s)
    return dsk, ids


def aplica_real(B=1000):
    print("\n== APLICAÇÃO AO DADO REAL (semgrad = P-NULL / difusão pura) ==")
    print("   pooled reproduz a FALHA (p<α ⟹ 'instrumento viciado'); "
          "estratificado deve ISENTAR o instrumento neste regime.")
    print(f"   {'braço':>9} {'m':>4} | {'S':>3} {'#ep':>5} {'#k':>4} | "
          f"{'T_obs':>6} | {'p_pooled':>9} | {'p_estrat':>9}")
    for braco in ("semgrad", "softplus"):
        for m in (16, 64, 256):
            try:
                dsk, ids = _extrai_real(braco, m)
            except Exception as e:
                print(f"   {braco:>9} {m:>4} | (sem dados: {e})")
                continue
            if not dsk:
                print(f"   {braco:>9} {m:>4} | (sem seeds com ≥{A.MIN_EP_SEED} ep)")
                continue
            nep = sum(d[0].size for d in dsk)
            nk = len(set(int(x) for d in dsk for x in d[2]))
            pp, T_obs, _ = p_calibrado(_obs_seeds(dsk), ids, (0, m), B)
            ps, _, _ = p_estratificado(dsk, ids, (0, m), R=B)
            To = "None" if T_obs is None else f"{T_obs:.3f}"
            print(f"   {braco:>9} {m:>4} | {len(dsk):>3} {nep:>5} {nk:>4} | "
                  f"{To:>6} | {('None' if pp is None else f'{pp:.4f}'):>9} | "
                  f"{('None' if ps is None else f'{ps:.4f}'):>9}")
            sys.stdout.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="grade curta p/ tunar")
    ap.add_argument("--real", action="store_true", help="só aplicação ao real")
    ap.add_argument("--full", action="store_true", help="grade completa")
    a = ap.parse_args()
    t0 = time.time()
    if a.real:
        aplica_real()
    elif a.smoke:
        # B>=199: com B=99 o menor p possível é 0.01, que não é < α=0.01
        tabela_fpr(R=40, S=10, B=199, ns=(8, 40))
        tabela_tpr(R=25, S=10, B=199, ns=(8, 40))
    else:  # --full (default)
        tabela_fpr(R=120, S=12, B=199, ns=(8, 40, 200))
        tabela_tpr(R=60, S=12, B=199, ns=(8, 40, 200))
        # robustez do complemento (shrinkage KM por-neurônio) — grade leve
        # (o loop por-episódio é O(n_ep); só n_ep pequenos)
        print("\n== ROBUSTEZ: complemento bootstrap-KM-por-neurônio c/ shrinkage ==")
        tabela_fpr(R=120, S=12, B=199, ns=(8, 40), shrink_em="hetero")
        aplica_real(B=1000)
    print(f"\n[tempo total: {time.time()-t0:.0f}s]")
