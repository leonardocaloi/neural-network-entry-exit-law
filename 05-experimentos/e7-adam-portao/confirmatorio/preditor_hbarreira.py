"""CONFIRMATORIO H-BARREIRA (heranca do piloto e7). CONFIG + MOTOR.

Pre-registro: README-hbarreira.md (escrito ANTES; criterios de morte la).
Decisoes de analise: analise_hbarreira.py (congeladas ANTES).

O config e 100% PROGRAMATICO e BYTE-DETERMINISTICO: `python preditor_hbarreira.py
config` reescreve results/config_hbarreira.json; se o byte mudar depois do
pre-registro, o desvio tem de ir para a secao "desvios" do README.

Estagios (resumiveis; o ambiente reinicia processos de fundo):
    python preditor_hbarreira.py config     # escreve o config
    python preditor_hbarreira.py custo      # aritmetica do custo (nao roda nada)
    python preditor_hbarreira.py smoke      # <5 min, 1 processo, nice -19
    python preditor_hbarreira.py inst       # sorteia/filtra as 101 instancias novas
    python preditor_hbarreira.py calib      # lr por (beta, b2, eps) na inst. de calibracao
    python preditor_hbarreira.py run <cell> # cell = "beta=1e4,b2=0.999,eps=1e-8"
    python preditor_hbarreira.py gd         # baseline GD (ancora eps -> infinito)

O motor e COPIA parametrizada de e7_adam.roda (que copiou o harness de deteccao
de t2-sweep-instancias/sweep_instancias.py, que copiou o integrador canonico de
06-notas/t2-prova/passo4_verificacao.py). Copia deliberada: o confirmatorio nao
pode mudar de comportamento se alguem editar o piloto.
"""
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(AQUI, "results")
RAIZ = os.path.abspath(os.path.join(AQUI, "..", ".."))
SWEEPDIR = os.path.join(RAIZ, "t2-sweep-instancias")

# ------------------------------------------------------------------ CONFIG
# NADA aqui e ajustado depois da primeira execucao. Mudanca = secao "desvios".
CFG = dict(
    tag="e7-confirmatorio-hbarreira",
    data_pre_registro="2026-07-26",
    # --- grade de eps EXECUTADA: 5 pontos, 2 decadas de espacamento
    EPS_GRID=[1e-8, 1e-6, 1e-4, 1e-2, 1.0],
    # --- grade PRIMARIA (17o parecer, adjudicacao (c) restrita): eps=1 NAO
    #     entra no ajuste primario. Em eps=1 o Adam ja degenera em GD (a fase
    #     VIVA ja esta no regime linear), a recalibracao de lr salta x8.9 e o
    #     "joelho" resultante e VAZAMENTO DE PROJETO, nao curvatura da lei.
    #     eps=1 e o GD continuam RODADOS e reportados -- como PONTOS DE
    #     SATURACAO da secundaria (iii), nunca no primario.
    EPS_PRIMARIO=[1e-8, 1e-6, 1e-4, 1e-2],
    EPS_SATURACAO=[1.0],
    # --- celulas: uma ANCORA + dois bracos de um-fator-por-vez (ver README §5)
    #     (o fatorial cheio 2 beta x 3 b2 nao alimenta nenhuma hipotese
    #      pre-registrada: a interacao beta x b2 nao foi predita por ninguem)
    CELULAS=[
        dict(nome="ancora",     beta=1e4, b2=0.999, papel="H-BARREIRA principal"),
        dict(nome="beta-baixo", beta=1e3, b2=0.999, papel="secundaria (i) invariancia em beta"),
        dict(nome="b2-090",     beta=1e4, b2=0.9,   papel="secundaria (ii) memoria de v"),
        dict(nome="b2-099",     beta=1e4, b2=0.99,  papel="secundaria (ii) memoria de v"),
    ],
    B1=0.9,                 # momento de Adam: FIXO (nao e fator)
    # --- instancias: bloco de seeds NOVO, disjunto do piloto (20260726-20261325)
    SEED0=20270000,
    N_SEEDS_MAX=1600,       # sorteia ate achar N_INST+1 que passam (H1)-(H4)
    N_INST=100,             # tamanho do confirmatorio (a 1a que passa e a de CALIBRACAO)
    # --- integracao
    DT=1e-5,
    H=10.0,                 # horizonte (piloto usou 8.0; fase morta ~3.6 no eps pequeno)
    T_CAL=2.0,              # horizonte curto da bissecao de lr (so precisa t_hit)
    IT_CAL=14,              # iteracoes da bissecao em log(lr)
    FACE=0,
    REC=200,                # subamostragem da serie de z na fase morta (criterio de plato)
    # --- criterio de PLATO (desacopla z_park do release: mata o vies de truncamento)
    #     z_park e VALIDO se (a) houve release -- entao o minimo e interior por
    #     construcao (z desceu e voltou a 0), ou (b) o run truncou no horizonte
    #     MAS o minimo ja se sustentou por PLATO_DT dentro de PLATO_TOL/beta.
    #     Sem (a), a fase morta curta do eps grande seria excluida em bloco e a
    #     regressao perderia justamente a ponta direita da grade (visto no smoke).
    PLATO_DT=0.5,           # o minimo tem de se sustentar por este tempo
    PLATO_TOL=0.5,          # em unidades de beta*z: max(beta*z) - beta*z_park na janela
    # --- criterios de decisao (repetidos em analise_hbarreira.py; fonte unica aqui)
    A_LO=0.9, A_HI=1.1,     # H-BARREIRA: MATA se IC95(a) for disjunto de [0.9,1.1]
    BOOT=10000, BOOT_SEED=20270001,
    CURV_TOL=1.0,           # guarda de curvatura: |q|*Lmax^2 acima disto (com IC95(q)
                            # excluindo 0) => INCONCLUSIVO-CURVATURA. (17o parecer,
                            # cosmetico 13: dentro do CFG, logo dentro do sha256.)
    RESTO_TOL=0.01,         # diagnostico: |resto inexplicado| acima disto com IC
                            # excluindo 0 => ESTIMACAO-COM-ALERTA (NAO mata)
    DA_TOL=0.1,             # secundaria (i): tolerancia de equivalencia em delta_a
    DA_POT=0.1,             # se a meia-largura do IC95(delta_a) > isto => SEM POTENCIA
    DB2_TOL=2.0,            # secundaria (ii): |b(0.9)-b(0.999)| predito << 2 (ver README §1)
    DB2_POT=1.0,            # se a meia-largura do IC95(delta_b) > 1, o teste nao tem potencia
    DB2_PREV=0.1,           # predicao-sob-teste do Apendice A.3.3 (NAO decisoria; o
                            # smoke ja deu 0.40, evidencia contraria previa)
    MIN_CELULAS_VALIDAS=0.60,   # < 60% de celulas com plato => P-NULL de instrumento
)
LRF = os.path.join(RES, "lr_calibrado.json")
INSTF = os.path.join(RES, "instancias.jsonl")
RUNS = os.path.join(RES, "hbarreira_runs.jsonl")


def escreve_config():
    os.makedirs(RES, exist_ok=True)
    txt = json.dumps(CFG, indent=1, sort_keys=True) + "\n"
    open(os.path.join(RES, "config_hbarreira.json"), "w").write(txt)
    print(txt)
    print("sha256:", hashlib.sha256(txt.encode()).hexdigest())


# ------------------------------------------------------------------ motor
def softplus(z, b):
    return np.logaddexp(0.0, b * z) / b


def sig(z, b):
    return 1.0 / (1.0 + np.exp(-np.clip(b * z, -500, 500)))


def grad(p, X, Y, b):
    zA, zB = p[0] * X + p[1], p[2] * X + p[3]
    r = softplus(zA, b) + softplus(zB, b) - Y
    sA, sB = sig(zA, b), sig(zB, b)
    return np.concatenate([[np.mean(r * sA * X), np.mean(r * sA)],
                           [np.mean(r * sB * X), np.mean(r * sB)]])


def roda(X, Y, PA0, PB0, beta, opt="adam", lr=None, eps=None, b2=None,
         t_max=None, so_hit=False, guardar=False):
    """Uma trajetoria. opt='gd' (canonico, ancora eps->infinito) ou 'adam'.

    Devolve, alem do harness do piloto, o necessario para H-BARREIRA:
      bz_park  = beta * min z_A(x_face) na fase morta
      plato    = o minimo se sustentou por PLATO_DT dentro de PLATO_TOL/beta
                 (criterio INDEPENDENTE do release: instancia truncada no
                  horizonte ainda entrega z_park valido se ja estacionou)
    """
    F, dt, b = CFG["FACE"], CFG["DT"], beta
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    p = np.concatenate([PA0, PB0]).astype(float)
    m, v = np.zeros(4), np.zeros(4)
    t, k = 0.0, 0
    morto = False
    n_mortes = 0
    t_hit = t_rel = None
    aR = aS = 0.0
    amR = amS = 0.0
    aR_rel = aS_rel = float("nan")
    t_pa = t_ps = t_z0 = None
    zpark, t_park = 0.0, None
    rats, serz, ser = [], [], []
    rat = 0.0
    nmax = int((t_max or CFG["H"]) / dt)
    while k < nmax:
        g = grad(p, X, Y, b)
        if opt == "gd":
            p -= dt * g
        else:
            k1 = k + 1
            m = CFG["B1"] * m + (1 - CFG["B1"]) * g
            v = b2 * v + (1 - b2) * g * g
            mh = m / (1 - CFG["B1"] ** k1)
            vh = v / (1 - b2 ** k1)
            u = mh / (np.sqrt(vh) + eps)
            p -= lr * u
            rat = float(abs(u[0]))
        t += dt
        k += 1
        z0 = p[0] * X[F] + p[1]
        zB0 = p[2] * X[F] + p[3]
        R = Y[F] - softplus(zB0, b)
        if not morto and z0 <= 0 and t > 0.05:
            morto, n_mortes, t_hit = True, n_mortes + 1, t
            if so_hit:
                break
        elif morto:
            if t_rel is None:
                if z0 < zpark:
                    zpark, t_park = z0, t
                rats.append(rat)
                if k % CFG["REC"] == 0:
                    serz.append((t, z0))
            aR += R * dt
            aS += np.sign(R) * dt
            amR, amS = min(amR, aR), min(amS, aS)
            if t_z0 is None and R >= 0:
                t_z0 = t
            if t_pa is None and aR >= 0:
                t_pa = t
            if t_ps is None and aS >= 0:
                t_ps = t
            if z0 >= 0 and t_rel is None:
                t_rel, aR_rel, aS_rel = t, aR, aS
            if t_rel is not None and t_pa is not None and t_ps is not None:
                break
        if guardar and k % CFG["REC"] == 0:
            ser.append((t, z0, R, aR, aS, rat))
    # ---- criterio de PLATO, em pos-processamento da serie subamostrada de z
    # Duas coisas DISTINTAS (17o parecer, forte 9):
    #   `estacionario` : verificacao de platO, calculada SEMPRE (inclusive nos
    #                    runs COM release), = o minimo se sustentou por PLATO_DT
    #                    dentro de PLATO_TOL/beta. E DIAGNOSTICO, reportado por eps.
    #   `plato`        : validade de z_park para a analise = houve release (minimo
    #                    interior por construcao) OU estacionario (truncado que ja
    #                    parou). NAO exige estacionaridade de quem teve release.
    # Se eps=1 falhar `estacionario`, ele e REPORTADO como nao-estacionario --
    # coerente com o papel de ponto de SATURACAO que ele tem (nao entra no primario).
    plato, bz_ret, plato_dur = False, None, None
    if t_park is not None and serz:
        sz = np.array(serz)
        jan = sz[(sz[:, 0] >= t_park) & (sz[:, 0] <= t_park + CFG["PLATO_DT"])]
        plato_dur = float(sz[-1, 0] - t_park)
        if len(jan):
            bz_ret = float(b * (jan[:, 1].max() - zpark))
        estac = bool(plato_dur >= CFG["PLATO_DT"] and bz_ret is not None
                     and bz_ret <= CFG["PLATO_TOL"])
        plato = bool(t_rel is not None or estac)
    else:
        estac = False
    out = dict(t_hit=t_hit, t_release=t_rel, n_mortes=n_mortes, t_zeroR=t_z0,
               t_pred_area=t_pa, t_pred_sign=t_ps,
               aR_rel=float(aR_rel), aS_rel=float(aS_rel),
               amin_R=float(amR), amin_S=float(amS),
               frac_R=float(abs(aR_rel / amR)) if amR < 0 else None,
               frac_S=float(abs(aS_rel / amS)) if amS < 0 else None,
               zA_park=float(zpark), bz_park=float(zpark * b),
               t_park=t_park, plato=plato, bz_retorno=bz_ret,
               estacionario=estac, plato_dur=plato_dur,
               truncado=bool(t_rel is None),
               rat_med=float(np.median(rats)) if rats else None)
    if guardar:
        out["serie"] = np.array(ser)
    return out


# ------------------------------------------------------- instancias (novas)
def instancias():
    """Sorteia do MESMO prior do sweep t2 e filtra por (H1)-(H4), com seeds
    NOVAS. As 13 do piloto e7 sao EXPLORATORIAS e ficam FORA (disclosure §7)."""
    sys.path.insert(0, SWEEPDIR)
    import sweep_instancias as sw
    seeds = list(range(CFG["SEED0"], CFG["SEED0"] + CFG["N_SEEDS_MAX"]))
    with Pool(4) as pool:
        regs = pool.map(sw.uma, seeds, chunksize=8)
    ok = [r for r in regs if r["passou_H1H2"]]
    need = CFG["N_INST"] + 1
    if len(ok) < need:
        raise RuntimeError(f"so {len(ok)} passaram; aumente N_SEEDS_MAX")
    ok = ok[:need]
    with open(INSTF, "w") as fh:
        for i, r in enumerate(ok):
            fh.write(json.dumps(dict(
                nome=f"seed{r['seed']}", seed=r["seed"], X=r["X"], Y=r["Y"],
                PA0=r["PA0"], PB0=r["PB0"],
                papel="calibracao" if i == 0 else "confirmatorio")) + "\n")
    print(f"{len(regs)} sorteadas, {sum(r['passou_H1H2'] for r in regs)} passaram, "
          f"gravadas {need} (1 calibracao + {CFG['N_INST']} confirmatorio)")


def le_instancias(papel="confirmatorio"):
    return [json.loads(l) for l in open(INSTF) if json.loads(l)["papel"] == papel]


# --------------------------------------------------------- calibracao de lr
def calibra_lr(inst, beta, b2, eps, alvo, lo=1e-10, hi=1e-1):
    def th(lr):
        r = roda(inst["X"], inst["Y"], inst["PA0"], inst["PB0"], beta,
                 opt="adam", lr=lr, eps=eps, b2=b2, t_max=CFG["T_CAL"],
                 so_hit=True)
        return r["t_hit"]
    for _ in range(CFG["IT_CAL"]):
        mid = float(np.sqrt(lo * hi))
        v = th(mid)
        if v is None or v > alvo:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def _cal1(arg):
    inst, beta, b2, eps, alvo = arg
    return chave(beta, b2, eps), calibra_lr(inst, beta, b2, eps, alvo)


def chave(beta, b2, eps):
    return f"beta={beta:.0e},b2={b2},eps={eps:.0e}"


def estagio_calib():
    inst = [json.loads(l) for l in open(INSTF)][0]
    alvos, tarefas = {}, []
    for c in CFG["CELULAS"]:
        beta = c["beta"]
        if beta not in alvos:
            g = roda(inst["X"], inst["Y"], inst["PA0"], inst["PB0"], beta, opt="gd")
            alvos[beta] = g["t_hit"]
        for eps in CFG["EPS_GRID"]:
            tarefas.append((inst, beta, c["b2"], eps, alvos[beta]))
    with Pool(4) as pool:
        pares = pool.map(_cal1, tarefas, chunksize=1)
    json.dump(dict(alvo_t_hit={f"{k:.0e}": v for k, v in alvos.items()},
                   inst_calib=inst["seed"], lr=dict(pares)),
              open(LRF, "w"), indent=1, sort_keys=True)
    print(json.dumps(dict(pares), indent=1, sort_keys=True))


# ------------------------------------------------------------------- runs
def _uma(arg):
    inst, cel, eps, lr = arg
    r = roda(inst["X"], inst["Y"], inst["PA0"], inst["PB0"], cel["beta"],
             opt="adam", lr=lr, eps=eps, b2=cel["b2"])
    return dict(nome=inst["nome"], seed=inst["seed"], celula=cel["nome"],
                beta=cel["beta"], b2=cel["b2"], eps=eps, lr=lr, opt="adam", **r)


def _uma_gd(arg):
    inst, beta = arg
    r = roda(inst["X"], inst["Y"], inst["PA0"], inst["PB0"], beta, opt="gd")
    return dict(nome=inst["nome"], seed=inst["seed"], celula="gd", beta=beta,
                b2=None, eps=None, lr=None, opt="gd", **r)


def _ja_rodou(pred):
    """TRAVA ANTI-DUPLICACAO (17o parecer, cosmetico 17). `run` faz append; a
    versao anterior confiava em 'truncar antes de repetir'. Agora ABORTA."""
    if not os.path.exists(RUNS):
        return 0
    n = 0
    with open(RUNS) as fh:
        for l in fh:
            if l.strip() and pred(json.loads(l)):
                n += 1
    return n


def estagio_run(nome_cel, eps, nproc=4):
    cel = [c for c in CFG["CELULAS"] if c["nome"] == nome_cel][0]
    n = _ja_rodou(lambda r: r.get("celula") == nome_cel and r.get("eps") == eps)
    if n:
        raise SystemExit(
            f"ABORTA: {RUNS} ja tem {n} linhas de celula={nome_cel} eps={eps:.0e}. "
            f"Reexecutar duplicaria. Se a rodada anterior foi parcial/invalida, "
            f"remova essas linhas EXPLICITAMENTE e registre em README §9 (desvios).")
    lr = json.load(open(LRF))["lr"][chave(cel["beta"], cel["b2"], eps)]
    ins = le_instancias()
    with Pool(nproc) as pool:
        regs = pool.map(_uma, [(i, cel, eps, lr) for i in ins], chunksize=1)
    with open(RUNS, "a") as fh:
        for r in regs:
            fh.write(json.dumps(r) + "\n")
    ok = [r for r in regs if r["plato"]]
    est = [r for r in regs if r["estacionario"]]
    tru = [r for r in regs if r["truncado"]]
    print(f"{nome_cel} eps={eps:.0e}: {len(ok)}/{len(regs)} com plato, "
          f"{len(est)}/{len(regs)} estacionarios, {len(tru)} truncados, "
          f"mediana bz_park={np.median([r['bz_park'] for r in ok]):.2f}"
          + ("   [PAPEL: saturacao da secundaria (iii), fora do primario]"
             if eps in CFG["EPS_SATURACAO"] else ""))


def estagio_gd(nproc=4):
    n = _ja_rodou(lambda r: r.get("opt") == "gd")
    if n:
        raise SystemExit(f"ABORTA: {RUNS} ja tem {n} linhas de GD. Reexecutar duplicaria.")
    ins = le_instancias()
    betas = sorted({c["beta"] for c in CFG["CELULAS"]})
    with Pool(nproc) as pool:
        regs = pool.map(_uma_gd, [(i, b) for b in betas for i in ins], chunksize=1)
    with open(RUNS, "a") as fh:
        for r in regs:
            fh.write(json.dumps(r) + "\n")
    for b in betas:
        a = [r["bz_park"] for r in regs if r["beta"] == b and r["plato"]]
        print(f"GD beta={b:.0e}: n={len(a)} mediana bz_park={np.median(a):.2f}")


# ------------------------------------------------------------------- custo
# Constantes MEDIDAS em 2026-07-26 nesta maquina (1 nucleo, nice -19, H=8,
# instancia de referencia): Adam eps=1e-8 -> 28.4 s; Adam eps=1 -> 9.0 s;
# GD -> 5.0 s. O run para assim que release E os dois preditores dispararam.
# 17o parecer, cosmetico 14: DECLARADO quais entradas sao medidas e quais nao.
#   MEDIDAS      : eps=1e-8 (28.4) e eps=1 (9.0), mais o GD (5.0).
#   INTERPOLADAS : eps=1e-6, 1e-4, 1e-2 -- as TRES sao interpolacao a olho
#                  entre os dois extremos medidos, NAO cronometragens. Se o
#                  custo real divergir, vai para README §9 (desvios).
CUSTO_MEDIDO = {1e-8, 1.0}
CUSTO_S = {1e-8: 28.4, 1e-6: 26.0, 1e-4: 22.0, 1e-2: 16.0, 1.0: 9.0}
CUSTO_GD = 5.0
PAD = 1.6   # padding: H 8->10, instancias mais lentas que a de referencia,
            # e a fracao que trunca (paga o horizonte inteiro)


def estagio_custo():
    n = CFG["N_INST"]
    por_cel = sum(CUSTO_S[e] for e in CFG["EPS_GRID"]) * n
    tot = por_cel * len(CFG["CELULAS"])
    gd = CUSTO_GD * n * len({c["beta"] for c in CFG["CELULAS"]})
    cal = CFG["IT_CAL"] * CFG["T_CAL"] / 8.0 * 28.4 * len(CFG["EPS_GRID"]) * len(CFG["CELULAS"])
    sweep = 11.8 * 60 * 12 / 600 * CFG["N_SEEDS_MAX"]   # 600 inst = 11.8 min c/ Pool(12)
    itens = [("sweep de instancias (1600 seeds)", sweep),
             ("calibracao de lr (20 celulas x eps)", cal),
             (f"grade Adam ({len(CFG['CELULAS'])} celulas x 5 eps x {n} inst)", tot),
             ("baseline GD (2 betas x 100 inst)", gd)]
    soma = 0.0
    print(f"{'item':45s} {'core-h':>8s}")
    for k, s in itens:
        s *= PAD
        soma += s
        print(f"{k:45s} {s/3600:8.2f}")
    print(f"{'TOTAL (core-hora, com padding %.1fx)'%PAD:45s} {soma/3600:8.2f}")
    for np_ in (3, 4, 6):
        print(f"  Pool({np_}), nice -19  ->  {soma/3600/np_:5.1f} h de parede")
    print("\n  entradas de custo por eps (s/run, 1 nucleo, H=8):")
    for e in CFG["EPS_GRID"]:
        print(f"    eps={e:.0e}: {CUSTO_S[e]:5.1f} s  "
              f"{'MEDIDO' if e in CUSTO_MEDIDO else 'INTERPOLADO (declarado)'}")
    print(f"\nFatorial CHEIO (2 beta x 3 b2 x 5 eps x {n}) seria "
          f"{6*sum(CUSTO_S[e] for e in CFG['EPS_GRID'])*n*PAD/3600:.1f} core-h "
          f"na grade Adam; o desenho um-fator-por-vez usa "
          f"{tot*PAD/3600:.1f} core-h. Ver README §5 para a justificativa.")


# ------------------------------------------------------------------- smoke
def estagio_smoke():
    """< 5 min, 1 processo. NAO e o experimento: so verifica que o motor roda,
    que o criterio de plato dispara e que o config e deterministico."""
    t0 = time.time()
    ref = dict(X=[0.5, 1.5], Y=[0.5, 1.0], PA0=[-0.5, 0.27], PB0=[1.144, 0.261])
    for beta, b2, eps, lr in [(1e4, 0.999, 1e-8, 1.2047458082005154e-06),
                              (1e4, 0.9, 1e-8, 1.2047458082005154e-06),
                              (1e4, 0.999, 1.0, 1.0667621274113613e-05)]:
        r = roda(ref["X"], ref["Y"], ref["PA0"], ref["PB0"], beta,
                 opt="adam", lr=lr, eps=eps, b2=b2)
        print(f"beta={beta:.0e} b2={b2} eps={eps:.0e}: t_hit={r['t_hit']:.4f} "
              f"t_rel={r['t_release']} bz_park={r['bz_park']:.2f} "
              f"plato={r['plato']} estac={r['estacionario']} "
              f"bz_ret={r['bz_retorno']} dur={r['plato_dur']} trunc={r['truncado']}")
    print(f"[smoke] {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "config"
    t0 = time.time()
    if cmd == "config":
        escreve_config()
    elif cmd == "custo":
        estagio_custo()
    elif cmd == "smoke":
        estagio_smoke()
    elif cmd == "inst":
        instancias()
    elif cmd == "calib":
        estagio_calib()
    elif cmd == "gd":
        estagio_gd()
    elif cmd == "run":
        estagio_run(sys.argv[2], float(sys.argv[3]))
    else:
        raise SystemExit(__doc__)
    print(f"[{cmd}] {(time.time()-t0)/60:.1f} min")
