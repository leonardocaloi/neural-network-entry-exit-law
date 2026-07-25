"""Preditor do CONFIRMATORIO etapa-5 (transformer FFN): gera config_e5.json com TODOS
os numeros lidos PROGRAMATICAMENTE do piloto-de-taxa e5 (results-taxa/*.json, via a
mesma logica de colapso de-chatter de analise_taxa) + uma CALIBRACAO propria de d_h.
Nada digitado a mao — cada numero congelado carrega sua ORIGEM.

Congela (conforme a especificacao da casa):
  - grade: bracos {sp-b3,sp-b10,gelu,relu} x {attn-treina,attn-frozen} x injecao
    {-0.3,-2.0} x S=50 seeds VIRGENS 2000-2049 x H=2e5 = 800 runs; m_ffn=4, eta=1e-3, T=0.
  - d_h (histerese): DERIVADO — quantil q0.99 da amplitude de flutuacao (meia-p2p em
    janelas L) de zmax de unidades VIVAS em runs FROZEN de calibracao. NAO e magico.
  - tau por braco = q0.9 do return-time (colapso de-chatter, pool train+frozen) do piloto.
  - E-A VALIDADE (relu-frozen: qualquer volta sustentada => invalido; teorema).
  - E-B MONTANTE: Delta(train-frozen); bootstrap por seed decide; MATA se CI-sup<=0.
  - E-C TRANSFERENCIA: sp-b3-frozen -0.3: mata_thresh = 1/2 * CI-inf do piloto (analogo
    ao C1 do 4a.3); n*=6 voltas p/ TRANSFERE.
  - E-D BARREIRA: gelu-frozen rasa vs profunda; MATA-barreira se CP-inf(prof)>=CP-sup(rasa).
  - E-E ESCADA: p_sust por beta; estimacao.
  - bootstrap por seed com RNGs NOVOS (nada de 777/779/5xxx/9090 nem 314159/271828 do 4a.3).

Uso: ../../venv/bin/python preditor_e5.py   # calibra d_h (~1min) + escreve config_e5.json
"""
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
if AQUI not in sys.path:
    sys.path.insert(0, AQUI)
CONFIG = os.path.join(AQUI, "config_e5.json")

from analise_e5 import clopper_pearson            # estatistica exata (sem scipy)
from analise_taxa import collapse, Z_STAR         # colapso de-chatter do piloto

# --- constantes de DESENHO (escolhas da casa, nao "resultados") -------------
H = 200_000
ETA = 1e-3
T = 0.0
N_DADOS = 24          # tokens: N=24 x S=3 (regime do piloto)
S_SEQ = 3
D_MODEL = 16
H_HEADS = 2
M_FFN = 4
REC_CADA = 300
S = 50
SEEDS = list(range(2000, 2000 + S))          # 2000..2049 VIRGENS
INJ_DEPTHS = [-0.3, -2.0]                     # rasa (>z*) ; profunda (<z*, barreira)
ARMS = [
    dict(nome="sp-b3", ativ="softplus", beta=3.0, papel="primario (classe-do-teorema)"),
    dict(nome="sp-b10", ativ="softplus", beta=10.0, papel="escada-beta"),
    dict(nome="gelu", ativ="gelu", beta=1.0, papel="barreira (fora-de-classe)"),
    dict(nome="relu", ativ="relu", beta=1.0, papel="validade / montante puro"),
]
FACTORS = ["train", "frozen"]
ALPHA = 0.05
N_STAR = 6                                    # piso de voltas sustentadas (TAXA-e5.md 7)
# RNGs de bootstrap NOVOS (proibidos 777/779/5xxx/9090 e os 314159/271828 do 4a.3)
BOOT_SEED_PROP = 161803                        # phi
BOOT_SEED_MONTANTE = 141421                    # sqrt2
BOOT_B = 20000

# --- calibracao de d_h (DERIVACAO documentada) ------------------------------
CALIB_SEEDS = [7, 8, 9]        # dedicados: fora das virgens (2000+) e do piloto (100-119)
CALIB_STEPS = 15_000
CALIB_L = 200                  # janela p/ isolar a OSCILACAO do trend (meia-p2p)
CALIB_Q = 0.99                 # quantil ALTO da amplitude


def calibra_deltah():
    """d_h = q0.99 da amplitude de flutuacao (meia-p2p em janelas L) de zmax das
    unidades VIVAS (zmax>0 o run todo) em runs FROZEN de calibracao. Retorna
    (delta_h, metadata). Reproduz o probe: seeds 7-9, 15000 passos, L=200."""
    import torch
    torch.set_num_threads(1)
    from motor_e5_conf import zmax_trace_frozen
    amps_all = []
    por_arm = {}
    for arm in ARMS:
        amps_arm = []
        for s in CALIB_SEEDS:
            tr = zmax_trace_frozen(arm["ativ"], arm["beta"], s, CALIB_STEPS,
                                   m_ffn=M_FFN, N=N_DADOS, S=S_SEQ, d=D_MODEL,
                                   H=H_HEADS, eta=ETA)
            for k in range(tr.shape[1]):
                col = tr[:, k]
                if np.all(col > 0.0):                # unidade VIVA o tempo todo
                    for i in range(0, len(col) - CALIB_L, CALIB_L):
                        w = col[i:i + CALIB_L]
                        amps_arm.append(0.5 * (w.max() - w.min()))
        if amps_arm:
            por_arm[arm["nome"]] = dict(q99=float(np.quantile(amps_arm, CALIB_Q)),
                                        n=len(amps_arm))
            amps_all.extend(amps_arm)
    amps_all = np.asarray(amps_all)
    delta_h = float(np.quantile(amps_all, CALIB_Q))
    meta = dict(
        metodo=("quantil q%.2f da amplitude=meia-p2p(zmax) em janelas de L=%d passos, "
                "sobre unidades VIVAS (zmax>0 todo o run) em runs FROZEN sem injecao"
                % (CALIB_Q, CALIB_L)),
        calib_seeds=CALIB_SEEDS, calib_steps=CALIB_STEPS, janela_L=CALIB_L,
        quantil=CALIB_Q, n_amostras=int(len(amps_all)), por_braco=por_arm,
        nota=("valida a posteriori a escolha d~0.05 do piloto (TAXA-e5.md 0/7); "
              "d_h derivado, nao magico"))
    return delta_h, meta


def le_piloto():
    """Numeros do piloto por (braco[,fator][,profundidade]) via colapso de-chatter."""
    prof_pred = lambda u: u["deep"]              # profunda = z_park/injdepth < z*
    rasa_pred = lambda u: not u["deep"]
    piloto = dict(tau={}, delta_montante={}, por_celula_prof={}, e_c_analogo={})
    for arm in ARMS:
        nome = arm["nome"]
        tres_pool = []
        rev_fac = {}
        for fac in FACTORS:
            units = collapse(f"{nome}-{fac}")[0]
            inj = [u for u in units if u["injetada"]]
            revd = [u for u in inj if u["revived"]]
            rev_fac[fac] = len(revd)
            tres_pool += [u["t_res"] for u in revd if u["t_res"] is not None]
            for depthname, pred in [("rasa", rasa_pred), ("prof", prof_pred)]:
                sub = [u for u in inj if pred(u)]
                rv = [u for u in sub if u["revived"]]
                piloto["por_celula_prof"][f"{nome}-{fac}-{depthname}"] = dict(
                    rev=len(rv), n=len(sub), cp=list(clopper_pearson(len(rv), len(sub))))
        piloto["tau"][nome] = float(np.quantile(tres_pool, 0.9)) if tres_pool else float("nan")
        piloto["delta_montante"][nome] = dict(train=rev_fac["train"], frozen=rev_fac["frozen"],
                                              delta=rev_fac["train"] - rev_fac["frozen"])
    # E-C analogo direto: sp-b3-frozen rasa
    ec = piloto["por_celula_prof"]["sp-b3-frozen-rasa"]
    piloto["e_c_analogo"] = dict(rev=ec["rev"], n=ec["n"], cp=ec["cp"], ci_inf=ec["cp"][1])
    return piloto


def main():
    print("[preditor_e5] calibrando d_h (runs FROZEN de calibracao)...", flush=True)
    delta_h, meta_dh = calibra_deltah()
    print(f"[preditor_e5] d_h = {delta_h:.5f}  ({meta_dh['metodo']})", flush=True)

    piloto = le_piloto()
    tau = piloto["tau"]

    # E-C: mata_thresh = 1/2 * CI-inf do piloto (analogo ao C1 do 4a.3)
    ec_ci_inf = piloto["e_c_analogo"]["ci_inf"]
    ec_mata_thresh = 0.5 * ec_ci_inf
    # reachability honesta: CP-sup(0,N) < thresh so p/ N grande
    def _cp_sup0(n):
        return clopper_pearson(0, n)[2]
    ec_N_min_mata = None
    for n in (50, 100, 200, 500, 800, 1000, 1500, 2000):
        if _cp_sup0(n) < ec_mata_thresh:
            ec_N_min_mata = n
            break

    cfg = dict(
        meta=dict(
            etapa="5-confirmatorio (transformer FFN)",
            gerado_por="preditor_e5.py",
            fonte_numeros=("piloto-de-taxa e5 (results-taxa/*.json, 160 runs EXPLORATORIO, "
                           "colapso de-chatter de analise_taxa) + calibracao propria de d_h"),
            regime="DETERMINISTICO (GD puro T=0); estocastico fora de escopo",
            status="CANDIDATO — aguardando 11o parecer",
            instrumento=("contador-mestre GT com HISTERESE d_h (mata o chatter do canal "
                         "nao-estacionario) + volta SUSTENTADA (viva no fim de H)"),
        ),
        arms=[a["nome"] for a in ARMS],
        grade=dict(
            bracos=ARMS, factors=FACTORS, inj_depths=INJ_DEPTHS,
            S=S, seeds=SEEDS, H=H, m_ffn=M_FFN, n_dados=N_DADOS, S_seq=S_SEQ,
            d_model=D_MODEL, H_heads=H_HEADS, eta=ETA, T=T, rec_cada=REC_CADA,
            celulas=len(ARMS) * len(FACTORS), runs_por_celula=len(INJ_DEPTHS) * S,
            total_runs=len(ARMS) * len(FACTORS) * len(INJ_DEPTHS) * S,
            nota_injecao=("1 unidade injetada por run, na profundidade da grade "
                          "(kill_depth). O piloto injetava 2/run (1 rasa+1 profunda); "
                          "aqui a profundidade e FATOR => 1/run, isolando o canal por "
                          "profundidade. Disclosure: pressao mais branda (3 vivas de 4 vs "
                          "2 do piloto) => se algo, favorece a volta (conservador p/ o MATA)."),
        ),
        seeds=SEEDS,
        alpha=ALPHA,
        delta_h=delta_h,
        delta_h_derivacao=meta_dh,
        tau_por_braco=tau,
        janela=dict(regra="[0, H - tau_braco]", H=H,
                    fim_por_braco={a: H - tau[a] for a in tau},
                    origem="tau_braco = q0.9 do return-time (colapso de-chatter, pool train+frozen)",
                    nota=("injetadas nascem mortas em t=0 => a janela liga no TEMPO DE VOLTA "
                          "(t_res <= H-tau), nao no tempo de morte (difere do 4a.3). "
                          "'sustained' primario = alive-no-fim (histerese); janela e robustez "
                          "secundaria.")),
        E_A=dict(
            estimando="VALIDADE: qualquer volta sustentada em relu-frozen (rasa OU profunda) => INVALIDO",
            base="teorema: grad proprio==0 (relu morto) E entrada fixa (frozen) => terminal, 0 voltas",
            decidibilidade="DECIDIVEL e limpa (piloto: relu-frozen 0/40). Obrigatoria.",
        ),
        E_B=dict(
            estimando="Delta = p_sust(train) - p_sust(frozen) por braco (pool profundidades)",
            regra="CI(boot seed): CI-inf>0 => REPRODUZ; CI-sup<=0 => MATA (montante falso); senao INCONCLUSIVO",
            pilot_delta=piloto["delta_montante"],
            disclosure=(
                "piloto (contador z_alive=0, chatter-propenso) mostrou Delta>0 decisivo "
                "(+6/+6/+6/+8; relu 0->8 = montante puro, own_move=0). MAS a HISTERESE d_h e "
                "ESTRITA: exige zmax>+d_h SUSTENTADO, nao so tocar 0. Teste em s103 mostrou "
                "chatter cru (160 travessias) -> 0 voltas confirmadas a H=8e4. A H=2e5 (>=3tau) "
                "os climbs de montante tem tempo de exceder d_h e sustentar => REPRODUCAO "
                "ESPERADA mas NAO garantida; MATA se o montante for so hover. Decidivel a S=50."),
        ),
        E_C=dict(
            estimando="sp-b3-frozen inj -0.3: p(volta sustentada | inj), Clopper-Pearson",
            n_star=N_STAR,
            mata_thresh=ec_mata_thresh,
            regra=("CP-sup < mata_thresh => MATA (nao transfere); senao sust>=n* => TRANSFERE; "
                   "senao INCONCLUSIVO. (MATA checado 1o.)"),
            pilot=piloto["e_c_analogo"],
            origem=dict(mata_thresh="0.5 * CP-inf(piloto sp-b3-frozen rasa = %d/%d)"
                        % (piloto["e_c_analogo"]["rev"], piloto["e_c_analogo"]["n"]),
                        n_star="TAXA-e5.md 7 (piso de voltas sustentadas)"),
            disclosure=(
                "mata_thresh=%.5f e minusculo => CP-sup(0,N)<thresh so a N>=%s "
                "(CP-sup(0,50)=%.4f > thresh). A N=50 a MATA de E-C e QUASE-INATINGIVEL "
                "(paridade honesta com o C1 do 4a.3, que pedia N>=47 zero-volta): E-C decide "
                "na pratica entre TRANSFERE (>=6 sustentadas) e INCONCLUSIVO (<6). O poder "
                "REAL e o gate positivo: o piloto a H=8e4 deu 2/20 (frozen rasa); projetado a "
                "H=2e5 (~3tau) p~0.28 => >=6/50 esperado. Falsificavel em principio (N grande)."
                % (ec_mata_thresh, ec_N_min_mata, _cp_sup0(50))),
        ),
        E_D=dict(
            estimando=("gelu-frozen: p_sust(profunda -2.0) vs p_sust(rasa -0.3). Barreira "
                       "preve rasa>>profunda no canal PROPRIO. Estimando extra: fracao de "
                       "trios com sinal(A_drive)!=sinal(A_mlp) (S1), deep vs raso."),
            n_star=N_STAR,
            regra=("CP-inf(prof) >= CP-sup(rasa) => MATA-BARREIRA (barreira nao existe); "
                   "CP-inf(rasa) > CP-sup(prof) e rasa>=n* => SUSTENTA-BARREIRA; senao INCONCLUSIVO"),
            pilot=dict(gelu_frozen_rasa=piloto["por_celula_prof"]["gelu-frozen-rasa"],
                       gelu_frozen_prof=piloto["por_celula_prof"]["gelu-frozen-prof"]),
            disclosure=(
                "o piloto injetou a -1.0 (so um pouco abaixo de z*=-0.752) e deu rasa 4/20 = "
                "profunda 4/20: EMPATE, NAO decidiu (P-ii nao sustentada a -1.0). O "
                "confirmatorio injeta a -2.0 (bem abaixo de z*) => se a barreira for terminal, "
                "profunda cai e a separacao aparece. E o teste GENUINO da barreira que o piloto "
                "nao pode fazer. Inversao de sinal de A_drive (S1) reportada como estimando "
                "(piloto: 80-95% opostos nos profundos)."),
        ),
        E_E=dict(
            estimando="escada p(volta sustentada) por beta {3,10}, frozen e train; estimacao, SEM MATA",
            pilot_rasa={f"{a}-{f}": piloto["por_celula_prof"].get(f"{a}-{f}-rasa")
                        for a in ("sp-b3", "sp-b10") for f in FACTORS},
        ),
        contagem=dict(
            regra=("volta SUSTENTADA (histerese; unidade termina H viva) UNICA por unidade "
                   "injetada; trios GT-area (t_area_drive <= t_res) com A_drive; born-dead por "
                   "injecao => t_morte=0."),
        ),
        bootstrap=dict(metodo="cluster por seed", B=BOOT_B,
                       seed_prop=BOOT_SEED_PROP, seed_montante=BOOT_SEED_MONTANTE,
                       rng_proibidos=[777, 779, "5xxx", 9090, 314159, 271828]),
        pilot_ref=piloto,
    )

    with open(CONFIG, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"\n[escrito] {CONFIG}")
    print("=== NUMEROS CONGELADOS (origem entre parenteses) ===")
    print(f"  d_h histerese         = {delta_h:.5f}   (q{CALIB_Q} meia-p2p zmax vivos, "
          f"frozen; n={meta_dh['n_amostras']})")
    for a in tau:
        print(f"  tau q0.9 {a:7s}     = {tau[a]:.1f}   janela_fim={H - tau[a]:.1f}")
    print(f"  E-C mata_thresh       = {ec_mata_thresh:.5f}   (0.5 x CP-inf piloto sp-b3-frozen "
          f"rasa {piloto['e_c_analogo']['rev']}/{piloto['e_c_analogo']['n']})")
    print(f"  E-C n*                = {N_STAR}   (piso voltas sustentadas)")
    print(f"  E-C MATA atingivel    = N>={ec_N_min_mata} (a N=50 CP-sup(0)={_cp_sup0(50):.4f} > thresh "
          f"=> INCONCLUSIVO, nao MATA)")
    print(f"  E-B pilot Delta       = " + ", ".join(
        f"{a}={piloto['delta_montante'][a]['delta']:+d}" for a in piloto['delta_montante']))
    print(f"  E-D pilot gelu-frozen = rasa {piloto['por_celula_prof']['gelu-frozen-rasa']['rev']}/20, "
          f"prof(-1.0) {piloto['por_celula_prof']['gelu-frozen-prof']['rev']}/20 (EMPATE => "
          f"confirmatorio injeta -2.0)")
    print(f"  grade                 = {len(ARMS)}x{len(FACTORS)}x{len(INJ_DEPTHS)}x{S} = "
          f"{cfg['grade']['total_runs']} runs; seeds {SEEDS[0]}..{SEEDS[-1]}")
    print(f"  RNGs boot (novos)     = prop={BOOT_SEED_PROP}, montante={BOOT_SEED_MONTANTE}")


if __name__ == "__main__":
    main()
