"""Checagem adversarial: nível (FPR) do teste-placebo de P-A sob um NULO
FIEL AO INSTRUMENTO — o Regua nunca produz t_area > t_res (a medição para na
ressurreição), então no nulo real o atraso da área é TRUNCADO na duração do
próprio episódio; o placebo herda atrasos de outros episódios SEM esse
truncamento. Pergunta: isso infla P(área vence placebo) acima de 1/2 sem
nenhuma informação por-episódio?

Nulo simulado: dur_i e delay_i independentes (zero informação); episódio só
tem t_area se delay_i <= dur_i (senão t_area=-1, cai do com_area — como no
Regua). Usa as FUNÇÕES REAIS de analise.py (fracao_vitorias_placebo,
teste_sinais) e os números do config (alpha, k_critico, MIN_SEED_PLACEBO).
"""
from pathlib import Path
import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parent / '.'))
import numpy as np
from analise import fracao_vitorias_placebo, teste_sinais, ALPHA, MIN_EP_SEED

def experimento(rng, S=20, n_ep=8, mu_dur=8.0, sig_dur=0.6, mu_del=7.3, sig_del=0.6):
    """Um 'experimento' = S seeds; devolve True se o teste-placebo PASSA."""
    fr_plac = []
    for s in range(S):
        eps = []
        for i in range(n_ep):
            dur = int(rng.lognormal(mu_dur, sig_dur)) + 200
            delay = int(rng.lognormal(mu_del, sig_del)) + 100
            t0 = int(rng.integers(0, 40000))
            t_area = t0 + delay if delay <= dur else -1   # truncamento do Regua
            eps.append(dict(t_morte=t0, t_res=t0 + dur, t_area=t_area))
        com = [e for e in eps if e["t_area"] >= 0]
        if len(com) < MIN_EP_SEED:
            fr_plac.append(None); continue
        f, _ = fracao_vitorias_placebo(com, rng)
        fr_plac.append(f)
    return teste_sinais(fr_plac, ALPHA)["passa"], \
        np.mean([f for f in fr_plac if f is not None]) if any(f is not None for f in fr_plac) else None

if __name__ == "__main__":
    for nome, (mu_del, sig_del, sig_dur) in {
        "atraso~0.5*dur (calibração: rel_err 0.53)": (7.3, 0.6, 0.6),
        "atraso~dur mesma escala": (8.0, 0.3, 0.3),
        "atraso curto, dur espalhada": (6.5, 0.5, 1.0),
    }.items():
        rng = np.random.default_rng(2026)
        n_exp = 400
        passes, fmed = 0, []
        for _ in range(n_exp):
            p, f = experimento(rng, mu_del=mu_del, sig_del=sig_del, sig_dur=sig_dur)
            passes += int(p)
            if f is not None:
                fmed.append(f)
        print(f"{nome}: FPR do teste-placebo = {passes}/{n_exp} = "
              f"{passes/n_exp:.3f} (nominal <= {ALPHA}); "
              f"fração-de-vitórias média sob NULO = {np.mean(fmed):.3f}")
