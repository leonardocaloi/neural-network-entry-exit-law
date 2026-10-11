"""Preditor do CONFIRMATÓRIO etapa-4a.3: gera config_4a3.json com TODOS os
números lidos PROGRAMATICAMENTE de rate-results.json (piloto-de-taxa). Nada
digitado à mão — cada número frozen carrega sua ORIGEM (chave no JSON do piloto).

Congela (item a item, conforme a especificação da casa e o 9º parecer):
  - grade: braços {sp-b3(primário), sp-b10, sp-b300, relu} × m∈{4,6} × S=50 seeds
    virgens 1900-1949 × H=5e5 × n=16 × η=1e-3 × T=0;
  - C1: estimando p(volta|morte dinâmica elegível), Clopper-Pearson; janela
    elegível [0, H−τ] com τ = q0.9 do return-time β3 do piloto;
  - gate n* = quantil-inferior-95% de Poisson(λ_CI-inf,boot × 100 runs) das mortes
    elegíveis esperadas em β3;
  - MATA C1: com N≥n*, CP-superior de p(volta) < metade do CI-inferior do piloto;
    N<n* ⟹ INCONCLUSIVO;
  - C2: fração de voltas com t_morte ≤ t_área < t_res; MATA CP-inf<0.5 com ≥6 voltas;
  - C4: escada p(volta|morte) por β, sem MATA;
  - VALIDADE: qualquer volta GT no relu ⟹ inválido;
  - bootstrap por seed (clustering) com RNGs NOVOS (nada de 777/779/5xxx/9090).

Uso: ../../../../../venv/bin/python predict_4a3.py   # escreve config_4a3.json
"""
import json
import os

from analyze_4a3 import poisson_lower_quantile  # única fonte da estatística exata

AQUI = os.path.dirname(os.path.abspath(__file__))
PILOTO = os.path.join(AQUI, "rate-results.json")
CONFIG = os.path.join(AQUI, "config_4a3.json")

# --- constantes de DESENHO (não são "resultados"; são escolhas da casa) -------
H = 500_000
ETA = 1e-3
T = 0.0
N_DADOS = 16
REC_CADA = 100
S = 50
SEEDS = list(range(1900, 1900 + S))     # 1900..1949 virgens
MS = [4, 6]
RUNS_POR_BRACO = S * len(MS)            # 100
BRACOS = [
    dict(nome="sp-b3", ativ="softplus", beta=3.0, papel="primário"),
    dict(nome="sp-b10", ativ="softplus", beta=10.0, papel="canal-C4"),
    dict(nome="sp-b300", ativ="softplus", beta=300.0, papel="controle-canal-fechado"),
    dict(nome="relu", ativ="relu", beta=300.0, papel="validade (grad≡0)"),
]
ALPHA = 0.05
# RNGs NOVOS de bootstrap (proibidos 777/779/5xxx/9090; virgens 1900+ p/ runs)
BOOT_SEED_RATE = 314159
BOOT_SEED_PROP = 271828
BOOT_B = 20000


def main():
    j = json.load(open(PILOTO))
    b3 = j["por_braco"]["sp-b3"]

    # τ = q0.9 do return-time β3 (piloto) ⟹ janela elegível [0, H−τ] ------------
    tau_b3 = j["tau"]["sp-b3"]
    janela_fim = H - tau_b3

    # λ_CI-inf (bootstrap POR SEED, o honesto a clustering — parecer) -----------
    lam_ci_inf = b3["rate_boot"][1]
    mortes_esperadas = lam_ci_inf * RUNS_POR_BRACO      # Poisson mean p/ o gate

    # n* = quantil-inferior-95% de Poisson(mortes_esperadas) --------------------
    n_star = poisson_lower_quantile(mortes_esperadas, q=ALPHA)

    # MATA C1: metade do CI-inferior do piloto p(volta|morte) β3 ----------------
    p_volta_ci_inf = b3["p_volta_morte"][1]
    mata_thresh = 0.5 * p_volta_ci_inf

    # C4: escada p(volta|morte) por β (referência do piloto p/ o report) --------
    pilot_ladder = {b: j["por_braco"][b]["p_volta_morte"]
                    for b in ["sp-b3", "sp-b10", "sp-b300", "relu"]}

    cfg = dict(
        meta=dict(
            etapa="4a.3-confirmatório",
            gerado_por="predict_4a3.py",
            fonte_numeros="rate-results.json (piloto-de-taxa, EXPLORATÓRIO 240 runs)",
            regime="DETERMINÍSTICO (T=0); estocástico fora de escopo",
            status="CANDIDATO A PRÉ-REGISTRO — aguardando 10º parecer",
            instrumento="GT-ÁREA (relógio único ground-truth; sem régua v2 no caminho crítico)",
        ),
        grade=dict(
            bracos=BRACOS, m=MS, S=S, seeds=SEEDS, H=H, n_dados=N_DADOS,
            eta=ETA, T=T, rec_cada=REC_CADA,
            runs_por_braco=RUNS_POR_BRACO, total_runs=len(BRACOS) * RUNS_POR_BRACO,
        ),
        primario="sp-b3",
        janela_elegivel=dict(
            tau_q90_ret_b3=tau_b3, H=H, fim=janela_fim,
            origem="rate-results.json:tau.sp-b3 (q0.9 do return-time GT do braço β3)",
        ),
        C1=dict(
            estimando="p(volta | morte dinâmica elegível), Clopper-Pearson",
            alpha=ALPHA,
            lam_ci_inf_b3_boot=lam_ci_inf,
            runs_b3=RUNS_POR_BRACO,
            mortes_esperadas_ci_inf=mortes_esperadas,
            n_star=n_star,
            mata_thresh=mata_thresh,
            p_volta_ci_inf_pilot_b3=p_volta_ci_inf,
            regra=("N_elig≥n*: CP-superior(voltas_elig, N_elig) < mata_thresh ⟹ MATA "
                   "(mecanismo não reproduz); N_elig<n* ⟹ INCONCLUSIVO (nunca negativo)"),
            origem=dict(
                lam_ci_inf_b3_boot="rate-results.json:por_braco.sp-b3.rate_boot[1]",
                n_star="poisson_lower_quantile(lam_ci_inf×runs_b3, q=alpha)",
                mata_thresh="0.5 × rate-results.json:por_braco.sp-b3.p_volta_morte[1]",
            ),
        ),
        C2=dict(
            estimando=("fração de voltas dinâmicas cuja ÁREA do drive-líder ABRE "
                       "(t_área ≥ 0) durante o episódio morto, antes da volta"),
            alpha=ALPHA, min_voltas=6, piso=0.5,
            honestidade_estrutural=(
                "10º parecer: no relógio único GT, t_morte ≤ t_área < t_res é "
                "GARANTIDO POR CONSTRUÇÃO sempre que a área abre (t_área só é "
                "setada em passos mortos e o episódio fecha na volta). O ÚNICO "
                "modo de falha empírico é 'volta sem área' — distinções "
                "empate/estrito são imateriais à contagem. NÃO reivindicar "
                "'lei de ordenação temporal' como achado."),
            volta_sem_area="conta como falha (evidência contra a lei)",
            regra=("n_voltas≥6: CP-inferior(área-abriu, voltas) < 0.5 ⟹ MATA "
                   "(o drive-líder não explica a volta); n_voltas<6 ⟹ INCONCLUSIVO"),
            desfecho_esperado=(
                "disclosure em paridade com C1: o piloto prevê fração≈1.0 sob o "
                "instrumento GT ⟹ SUSTENTA é o esperado; a potência de MATA vem "
                "inteiramente da taxa de voltas-sem-área (1 em 6 já mata; "
                "voltas conduzidas fora-do-líder produziriam exatamente isso)"),
            nota="denominador = TODAS as voltas dinâmicas; born-dead excluídos",
        ),
        C4=dict(
            estimando="escada p(volta|morte) por β, com CIs, SEM MATA",
            pilot_p_volta_morte=pilot_ladder,
            origem="rate-results.json:por_braco.<braço>.p_volta_morte",
        ),
        validade=dict(
            regra="qualquer volta GT no relu (volta dyn OU born-rev OU n_revive_real>0) ⟹ INVÁLIDO",
            base="teorema: grad≡0 no neurônio relu morto ⟹ 0 voltas",
        ),
        bootstrap=dict(metodo="cluster por seed", B=BOOT_B,
                       seed_rate=BOOT_SEED_RATE, seed_prop=BOOT_SEED_PROP,
                       rng_proibidos=[777, 779, "5xxx", 9090]),
        pilot_ref=dict(
            por_braco={b: dict(rate_boot=j["por_braco"][b]["rate_boot"],
                               rate_pois=j["por_braco"][b]["rate_pois"],
                               p_volta_morte=j["por_braco"][b]["p_volta_morte"],
                               n_dyn=j["por_braco"][b]["n_dyn"],
                               n_voltas=j["por_braco"][b]["n_voltas"])
                       for b in ["sp-b3", "sp-b10", "sp-b300", "relu"]},
            tau=j["tau"],
        ),
    )

    with open(CONFIG, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"[escrito] {CONFIG}")
    print("=== NÚMEROS CONGELADOS (origem entre parênteses) ===")
    print(f"  τ q0.9 ret β3         = {tau_b3:.1f}   (tau.sp-b3)")
    print(f"  janela elegível fim   = {janela_fim:.1f}   (H − τ, H={H})")
    print(f"  λ_CI-inf β3 (boot)    = {lam_ci_inf:.5f}   (por_braco.sp-b3.rate_boot[1])")
    print(f"  mortes esperadas β3   = {mortes_esperadas:.3f}   (λ_CI-inf × {RUNS_POR_BRACO} runs)")
    print(f"  n* (gate INCONCL.)    = {n_star}   (quantil-inf-95% Poisson)")
    print(f"  p(volta) CI-inf pilot = {p_volta_ci_inf:.5f}   (por_braco.sp-b3.p_volta_morte[1])")
    print(f"  MATA C1 thresh        = {mata_thresh:.5f}   (½ × CI-inf pilot)")
    print(f"  C2 piso / min voltas  = 0.5 / 6")
    print(f"  escada C4 (piloto)    = " + ", ".join(
        f"{b}={pilot_ladder[b][0]:.3f}" for b in ['sp-b3', 'sp-b10', 'sp-b300', 'relu']))
    print(f"  grade                 = {len(BRACOS)}×{len(MS)}×{S} = "
          f"{len(BRACOS) * RUNS_POR_BRACO} runs; seeds {SEEDS[0]}..{SEEDS[-1]}")
    print(f"  RNGs boot (novos)     = rate={BOOT_SEED_RATE}, prop={BOOT_SEED_PROP}")


if __name__ == "__main__":
    main()
