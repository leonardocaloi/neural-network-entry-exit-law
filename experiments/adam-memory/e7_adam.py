"""e7 [EXPLORATORIO]: o que o Adam faz com o portao de memoria (T2').

Motor: mesma instancia de referencia e mesmo harness de deteccao de
experiments/instance-sweep/sweep_instances.py (que copiou o
integrador canonico de theory/verify_step4.py).
A UNICA troca: o passo GD do sistema COMPLETO vira Adam (beta1=0.9,
beta2=0.999, bias-correction, todo theta).

Hipoteses pre-registradas em README.md (H-SIGN, H-EPS, H-NULL).
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(AQUI, "results")
SWEEP = os.path.join(AQUI, "..", "instance-sweep", "sweep_instances.jsonl")

FACE = 0
# DESVIO documentado: H=1.8 era o horizonte do GD (release em 1.13 na
# referencia). Sob Adam a fase morta dura ~3x mais, entao o horizonte foi
# estendido para 8.0 para PODER observar o retorno. Sem isso, todo o piloto
# reportaria "nao volta" por truncamento, o que seria artefato.
T_FIM = 8.0
T_CAL = 1.0          # horizonte curto usado so na calibracao de lr (so precisa t_hit)
DT = 1e-5
BETA = 1e4
B1, B2 = 0.9, 0.999
EPS_DEFAULT = 1e-8
EPS_GRID = [1e-8, 1e-4, 1e-2, 1.0]
N_INST = 12          # + a de referencia = 13 instancias
REC = 50            # subamostragem das series

REF = dict(nome="ref-paper",
           X=[0.5, 1.5], Y=[0.5, 1.0], PA0=[-0.5, 0.27], PB0=[1.144, 0.261])


def softplus(z, b):
    return np.logaddexp(0.0, b * z) / b


def sig(z, b):
    bz = np.clip(b * z, -500, 500)
    return 1.0 / (1.0 + np.exp(-bz))


def grad(p, X, Y, b):
    zA, zB = p[0] * X + p[1], p[2] * X + p[3]
    r = softplus(zA, b) + softplus(zB, b) - Y
    sA, sB = sig(zA, b), sig(zB, b)
    return np.concatenate([[np.mean(r * sA * X), np.mean(r * sA)],
                           [np.mean(r * sB * X), np.mean(r * sB)]])


# --------------------------------------------------------------- integradores
def roda(X, Y, PA0, PB0, b=BETA, opt="adam", lr=None, eps=EPS_DEFAULT,
         dt=DT, guardar=False, t_max=None, so_hit=False):
    """Uma trajetoria. opt='gd' (canonico) ou 'adam'. Mesmo harness de deteccao.

    Preditores calculados na resolucao cheia dentro do loop, a partir da MESMA
    serie R = Y_face - softplus(z_B(x_face)) desta corrida:
      t_pred_area  : primeiro t com int_{t_hit}^t R dt >= 0  (lei velha)
      t_pred_sign  : primeiro t com int_{t_hit}^t sign(R) dt >= 0  (H-SIGN)
      t_zeroR      : primeiro t apos t_hit com R >= 0
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    p = np.concatenate([PA0, PB0]).astype(float)
    m = np.zeros(4)
    v = np.zeros(4)
    t = 0.0
    k = 0
    morto = False
    n_mortes = 0
    t_hit = t_rel = None
    aR = aS = 0.0
    amR = amS = 0.0
    aR_rel = aS_rel = float("nan")
    t_pa = t_ps = t_z0 = None
    ser = []
    rats = []
    zpark = 0.0
    rat = 0.0
    nmax = int((t_max or T_FIM) / dt)
    while k < nmax:
        g = grad(p, X, Y, b)
        if opt == "gd":
            p -= dt * g
        else:
            k1 = k + 1
            m = B1 * m + (1 - B1) * g
            v = B2 * v + (1 - B2) * g * g
            mh = m / (1 - B1 ** k1)
            vh = v / (1 - B2 ** k1)
            u = mh / (np.sqrt(vh) + eps)
            p -= lr * u
            rat = float(abs(u[0]))     # saturacao do Adam na comp. w_A
        t += dt
        k += 1
        z0 = p[0] * X[FACE] + p[1]
        zB0 = p[2] * X[FACE] + p[3]
        R = Y[FACE] - softplus(zB0, b)
        if not morto and z0 <= 0 and t > 0.05:
            morto = True
            n_mortes += 1
            t_hit = t
            if so_hit:
                break
        elif morto:
            if t_rel is None:
                zpark = min(zpark, z0)
                rats.append(rat)
            aR += R * dt
            aS += np.sign(R) * dt
            amR = min(amR, aR)
            amS = min(amS, aS)
            if t_z0 is None and R >= 0:
                t_z0 = t
            if t_pa is None and aR >= 0:
                t_pa = t
            if t_ps is None and aS >= 0:
                t_ps = t
            if z0 >= 0 and t_rel is None:
                t_rel = t
                aR_rel, aS_rel = aR, aS
                if guardar:
                    ser.append((t, z0, R, aR, aS, rat))
                # NAO para no release: os relogios seguem sendo integrados (a
                # continuacao e contrafactual, so para localizar o zero de cada
                # relogio mesmo quando ele cai depois do release).
            if t_rel is not None and t_pa is not None and t_ps is not None:
                break
        if guardar and k % REC == 0:
            ser.append((t, z0, R, aR, aS, rat))
    out = dict(t_hit=t_hit, t_release=t_rel, n_mortes=n_mortes,
               t_zeroR=t_z0, t_pred_area=t_pa, t_pred_sign=t_ps,
               aR_rel=float(aR_rel), aS_rel=float(aS_rel),
               amin_R=float(amR), amin_S=float(amS),
               # fracao do relogio NAO cumprida no instante do release:
               # 0 = a lei bate exatamente, 1 = o relogio nem comecou a voltar
               frac_R=float(abs(aR_rel / amR)) if amR < 0 else None,
               frac_S=float(abs(aS_rel / amS)) if amS < 0 else None,
               zA_park=float(zpark), bz_park=float(zpark * b),
               rat_med=float(np.median(rats)) if rats else None)
    if guardar:
        out["serie"] = np.array(ser)
    return out


def calibra_lr(inst, eps, alvo_t_hit, lo=1e-9, hi=1e-1, it=14):
    """Bissecao em log(lr): t_hit^Adam == t_hit^GD. Sem ajuste posterior."""
    def th(lr):
        r = roda(inst["X"], inst["Y"], inst["PA0"], inst["PB0"],
                 opt="adam", lr=lr, eps=eps, t_max=T_CAL, so_hit=True)
        return r["t_hit"]
    for _ in range(it):
        mid = np.sqrt(lo * hi)
        v = th(mid)
        if v is None:            # nem morreu: lr baixo demais
            lo = mid
        elif v > alvo_t_hit:     # morre tarde: acelera
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


# ------------------------------------------------------------------ analise
def erros(r):
    """erro relativo do preditor, normalizado pela duracao da fase morta."""
    if r["t_release"] is None or r["t_hit"] is None:
        return None, None
    dur = r["t_release"] - r["t_hit"]
    if dur <= 0:
        return None, None
    def e(tp):
        return None if tp is None else abs(tp - r["t_release"]) / dur
    return e(r["t_pred_area"]), e(r["t_pred_sign"])


def instancias():
    ins = [REF]
    with open(SWEEP) as fh:
        ok = [json.loads(l) for l in fh]
    ok = [r for r in ok if r["passou_H1H2"]][:N_INST]
    for r in ok:
        ins.append(dict(nome=f"seed{r['seed']}", X=r["X"], Y=r["Y"],
                        PA0=r["PA0"], PB0=r["PB0"]))
    return ins


def uma(arg):
    inst, eps, lr = arg
    gd = roda(inst["X"], inst["Y"], inst["PA0"], inst["PB0"], opt="gd")
    ad = roda(inst["X"], inst["Y"], inst["PA0"], inst["PB0"],
              opt="adam", lr=lr, eps=eps)
    ea, es = erros(ad)
    ega, egs = erros(gd)
    return dict(nome=inst["nome"], eps=eps, lr=lr,
                gd={k: gd[k] for k in gd}, adam={k: ad[k] for k in ad},
                err_area=ea, err_sign=es, err_area_gd=ega, err_sign_gd=egs)


def resumo(regs, tag, chave="adam"):
    ok = [r for r in regs if r[chave]["t_release"] is not None
          and r[chave]["n_mortes"] == 1]
    d = dict(tag=tag, N=len(regs), N_release_limpo=len(ok))
    campos = [("area", "err_area" if chave == "adam" else "err_area_gd"),
              ("sign", "err_sign" if chave == "adam" else "err_sign_gd")]
    for nome, k in campos:
        a = np.array([r[k] for r in ok if r[k] is not None])
        d[f"n_{nome}"] = int(len(a))
        if len(a):
            d[f"med_{nome}"] = float(np.median(a))
            d[f"p90_{nome}"] = float(np.percentile(a, 90))
    for nome in ("frac_R", "frac_S", "bz_park", "rat_med"):
        a = np.array([r[chave][nome] for r in ok if r[chave].get(nome) is not None])
        if len(a):
            d[f"med_{nome}"] = float(np.median(a))
            d[f"p90_{nome}"] = float(np.percentile(a, 90))
    return d


# ------------------------------------------------------------------ estagios
# O ambiente reinicia processos de fundo, entao o piloto e RESUMIVEL:
# cada estagio roda em primeiro plano, le e escreve results/.
LRF = os.path.join(RES, "lr_calibrado.json")
RUNS = os.path.join(RES, "e7_runs.jsonl")


def _calib1(arg):
    eps, alvo = arg
    lr = calibra_lr(REF, eps, alvo)
    return eps, lr


def estagio_calib():
    gd_ref = roda(REF["X"], REF["Y"], REF["PA0"], REF["PB0"], opt="gd",
                  guardar=True)
    np.savez_compressed(os.path.join(RES, "serie_gd.npz"),
                        gd=gd_ref["serie"],
                        gd_ev=np.array([gd_ref["t_hit"], gd_ref["t_release"],
                                        gd_ref["t_pred_area"] or np.nan,
                                        gd_ref["t_pred_sign"] or np.nan]))
    print(f"GD ref: t_hit={gd_ref['t_hit']:.4f} t_rel={gd_ref['t_release']:.4f} "
          f"pred_area={gd_ref['t_pred_area']} pred_sign={gd_ref['t_pred_sign']}")
    alvo = gd_ref["t_hit"]
    with Pool(4) as pool:
        pares = pool.map(_calib1, [(e, alvo) for e in EPS_GRID])
    lrs = {f"{e:.0e}": lr for e, lr in pares}
    json.dump(dict(alvo_t_hit=alvo, lr=lrs), open(LRF, "w"), indent=1)
    print(json.dumps(lrs, indent=1))


def estagio_run(eps):
    d = json.load(open(LRF))
    lrs = d.get("lr", d)
    lr = lrs[f"{eps:.0e}"]
    ins = instancias()
    with Pool(6) as pool:
        regs = pool.map(uma, [(i, eps, lr) for i in ins], chunksize=1)
    with open(RUNS, "a") as fh:
        for r in regs:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(resumo(regs, f"eps={eps:.0e}")))


def estagio_serie():
    d = json.load(open(LRF))
    lr = d.get("lr", d)[f"{EPS_DEFAULT:.0e}"]
    ad = roda(REF["X"], REF["Y"], REF["PA0"], REF["PB0"], opt="adam",
              lr=lr, eps=EPS_DEFAULT, guardar=True)
    g = np.load(os.path.join(RES, "serie_gd.npz"))
    np.savez_compressed(os.path.join(RES, "serie_ref.npz"),
                        gd=g["gd"], gd_ev=g["gd_ev"], adam=ad["serie"],
                        ad_ev=np.array([ad["t_hit"], ad["t_release"] or np.nan,
                                        ad["t_pred_area"] or np.nan,
                                        ad["t_pred_sign"] or np.nan]))
    print(json.dumps({k: v for k, v in ad.items() if k != "serie"}))


if __name__ == "__main__":
    import sys
    os.makedirs(RES, exist_ok=True)
    t0 = time.time()
    cmd = sys.argv[1]
    if cmd == "calib":
        estagio_calib()
    elif cmd == "run":
        estagio_run(float(sys.argv[2]))
    elif cmd == "serie":
        estagio_serie()
    print(f"[{cmd}] {(time.time()-t0)/60:.1f} min")
