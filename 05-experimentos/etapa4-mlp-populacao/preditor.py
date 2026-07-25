"""Etapa 4a — PREDITOR: congela números, critérios e grade → config_etapa4.json.

Rodar UMA vez antes do commit de pré-registro. Tudo que a análise usa vem do
config; mudança posterior = desvio documentado no README §6.
Potência: MDE do critério P-A por enumeração binomial exata (sem RNG).
"""
import json
from math import comb


def crit_binomial(S, alpha=0.01):
    """Menor k tal que P(Bin(S,1/2) >= k) < alpha (teste de sinais entre seeds)."""
    for k in range(S + 1):
        if sum(comb(S, j) for j in range(k, S + 1)) / 2 ** S < alpha:
            return k
    return S + 1


def poder_PA(p_ep, S=20, n_ep=30, alpha=0.01):
    """Potência do MATA P-A: por seed, fração-de-vitórias > 1/2 se
    Bin(n_ep, p_ep) >= floor(n_ep/2)+1; sucesso entre seeds via teste de sinais."""
    k_seed = n_ep // 2 + 1
    p_seed = sum(comb(n_ep, j) * p_ep ** j * (1 - p_ep) ** (n_ep - j)
                 for j in range(k_seed, n_ep + 1))
    k_crit = crit_binomial(S, alpha)
    return sum(comb(S, j) * p_seed ** j * (1 - p_seed) ** (S - j)
               for j in range(k_crit, S + 1))


if __name__ == "__main__":
    S, N_EP_MIN = 20, 30
    potencia = {f"{p:.2f}": round(poder_PA(p), 3)
                for p in (0.55, 0.60, 0.62, 0.65, 0.70)}
    mde = next(p for p in (0.55, 0.60, 0.62, 0.65, 0.70)
               if poder_PA(p) >= 0.8)
    cfg = {
        "exp": "exp05", "etapa": "4a", "commit_calibracao": "bd09a6b",
        "dados": {"tipo": "1dreg", "n": 32, "func": "sin(2x)+0.3x, x~U[0,2]",
                  "full_batch": True},
        "arquitetura": {"init_w_b": "N(0,1)", "init_v": "N(0,1/m)"},
        "device": {"pin": "cpu", "torch_deterministic": True,
                   "threads_por_worker": 1,
                   "justificativa": "B4 do 6o parecer: pareamento por seed exige "
                   "device unico; modelos minusculos nao pagam overhead de GPU"},
        "grade": {
            "m": [16, 64, 256], "eta": 3e-4, "passos": 100000,
            "seeds": list(range(1700, 1720)),
            "bracos": [
                {"nome": "softplus", "ativ": "softplus", "beta": 300.0, "T": 1e-3},
                {"nome": "erf", "ativ": "erf", "beta": 300.0, "T": 1e-3},
                {"nome": "gelu", "ativ": "gelu", "beta": None, "T": 1e-3},
                {"nome": "relu", "ativ": "relu", "beta": None, "T": 1e-3},
                {"nome": "T0", "ativ": "softplus", "beta": 300.0, "T": 0.0},
                {"nome": "freeze", "ativ": "softplus", "beta": 300.0, "T": 1e-3,
                 "freeze_meio_mortos": True},
                {"nome": "embaralha", "ativ": "softplus", "beta": 300.0, "T": 1e-3,
                 "embaralha": True},
                {"nome": "semgrad", "ativ": "softplus", "beta": 300.0, "T": 1e-3,
                 "sem_grad": True},
                {"nome": "Tmenor", "ativ": "softplus", "beta": 300.0, "T": 3e-4},
                {"nome": "Tmaior", "ativ": "softplus", "beta": 300.0, "T": 3e-3},
            ],
        },
        "regua": {
            "limiar": "2*sqrt(n*T)/norm((x_jstar,1)); T com piso 1e-7",
            "delta_snapshot": 1, "phi_v": "1/(4*sqrt(m))", "q": 2, "q_j": 0,
            "min_limpos_por_celula": N_EP_MIN,
            "calibracao": {"pareado_toy": "34/34 (DESCRITIVO: mecanico dado o "
                           "truncamento — 7o parecer)", "rel_err_area": 0.534,
                           "rel_err_sinal": 0.851,
                           "erro_intrinseco": "CONSISTENTE com atraso O(1/beta) "
                           "do teorema (300->1000: 4.54%->1.91%, razao 0.42; "
                           "piso residual ~0.8pp nao explicado)"},
        },
        "P_A": {"braco": "softplus", "alpha": 0.01, "B": 1000,
                "rng_seed_mc": 779, "min_ep_por_seed": 3,
                "teste": "bootstrap parametrico sob o nulo TRUNCADO (7o parecer): "
                         "H0 = atraso da area independente da duracao, atrasos "
                         "iid de F-hat (Kaplan-Meier; nao-disparo = censura a "
                         "direita na duracao); estatistica T = media entre seeds "
                         "da fracao-de-vitorias area-vs-placebo-permutado; "
                         "p=(1+#{T_b>=T_obs})/(B+1); passa se p<alpha. Nivel e "
                         "potencia demonstrados no analise.py --sintetico "
                         "(FPR para n_ep/seed em {8,40,200}; TPR memoria)",
                "placebo": {"tipo": "atraso-permutado dentro da mesma seed",
                            "rng_seed": 777, "min_por_seed": 2},
                "vs_sinal": "DESCRITIVO apenas: t_sinal<=t_area<=t_res por "
                            "construcao do instrumento (7o parecer) — nao e MATA",
                "report_sem_area": "fracao de limpos-completos sem disparo da "
                                   "area reportada por celula (evidencia contra "
                                   "a lei se alta; 17% no piloto m=64/20k)",
                "mata": "p_calibrado >= alpha nas celulas validas (agregacao)",
                "agregacao": "avaliado POR celula de m (validas: >=30 limpos); "
                             "memoria demonstrada se p<alpha em >=2 de 3 celulas "
                             "validas INCLUINDO m=16; MATA se falha em TODAS as "
                             "validas; resto = parcial"},
        "P_B": {"bracos": ["softplus", "erf"], "piso_rel": 0.05, "alpha": 0.01,
                "correcao_multipla": "Bonferroni: alpha/3 por estatistica",
                "estatisticas": ["frac_censurada", "mediana_dur", "n_limpos"],
                "teste": "sinais bilateral entre seeds do Delta pareado",
                "gate_min_limpos": N_EP_MIN,
                "mata": "(sig E |Delta|rel>piso) em alguma estatistica, em "
                        ">=2 de 3 celulas de m COM gate de limpos"},
        "P_B_linha": {"braco": "gelu", "tipo": "EXPLORATORIO fora-da-classe"},
        "P_C": {"tipo": "ANCORA/ESTIMACAO", "bracos": ["relu", "softplus"],
                "report": "frac censurada com CI; permanente-por-mecanismo "
                          "(grad=0 verificado) separado de censurado"},
        "P_D": {"bracos": ["freeze", "softplus"], "piso": "1% da duracao "
                "mediana dos episodios LIMPOS-completos do controle",
                "B_boot": 10000, "rng_seed": 778, "S_min_sem": 5,
                "mata": "classe 'sem' (CI95 da mediana(Delta t_res) contem 0 E "
                        "|mediana|<piso E S>=S_min_sem) em todas as celulas; "
                        "'detectada' exige CI fora de 0 E |mediana|>=piso; "
                        "resto = indeterminada"},
        "P_E": {"tipo": "ESTIMACAO", "superficies": [
            "frac_censurada x (T,m)", "medianas t_res e duracao x m",
            "fracao-limpa x m", "qualidade da predicao-area x m",
            "fracao morte-por-saida e v_cruzou x m",
            "corr(dur,t_morte) e corr(atraso,t_morte) por celula — descritivo "
            "de tendencia temporal comum (7o parecer 2a rodada: tendencia "
            "forte comum pode rejeitar o nulo iid sem memoria por-episodio; "
            "reportar desarma a objecao)"]},
        "P_NULL": {"braco": "semgrad",
                   "criterio": "P-A calibrado NAO pode passar em nenhuma celula; "
                               "se passar, instrumento/calibracao viciados -> "
                               "tudo invalido"},
        "potencia_P_A": {
            "metodo": "nivel (FPR) e potencia (TPR) do teste calibrado "
                      "demonstrados por simulacao no --sintetico, sob o nulo "
                      "truncado e sob memoria; ver saida impressa la",
            "condicional": "potencia condicional a episodios/seed suficientes "
                           "(piloto: ~170 completos/seed em 20k passos, m=64 — "
                           "folga grande sobre min_ep_por_seed=3)",
            "referencia_binomial_antiga": {"curva": potencia, "MDE": mde,
                                           "nota": "valia para o teste de sinais "
                                           "abandonado no P-A (7o parecer)"}},
        "resultados": {"formato": "results/{braco}-m{m}-s{seed}.json.gz",
                       "git": "raw fica local (regeneravel, deterministico); "
                              "manifest com SHA256 + agregados sao commitados"},
    }
    with open("config_etapa4.json", "w") as f:
        json.dump(cfg, f, indent=1, ensure_ascii=False)
    print(f"config_etapa4.json escrito. k_critico(S=20)={crit_binomial(S)} "
          f"seeds; potencia P-A: {potencia}; MDE(0.8)={mde}")
