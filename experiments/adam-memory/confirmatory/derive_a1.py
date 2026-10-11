"""Derivacao de a=1 na lei H-BARREIRA  y = -beta*z_park = a*ln(1/eps) + b.

Tres partes, todas reexecutaveis:
  (A) algebra   : SymPy. EMA de v, EDO do sticking, solucao fechada, da/dL.
  (B) crossover : re-roda a instancia de referencia do PILOTO instrumentada
                  (|g|, |mhat|, sqrt(vhat), u) e mede quem cruza eps no
                  instante do estacionamento.
  (C) ajuste    : refaz o ajuste do piloto (results/e7_runs.jsonl) com e sem
                  o ponto eps=1, e mede os vazamentos de projeto d ln lr/dL
                  e d ln K_morto/dL que a derivacao preve entrarem em `a`.

Uso:  nice -19 python derive_a1.py [algebra|crossover|ajuste|tudo]
NAO faz parte do confirmatorio pre-registrado: e derivacao + auditoria do
piloto EXPLORATORIO. Nao toca results/ do confirmatorio.
"""
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
PILOTO = os.path.join(AQUI, "..")
RES_P = os.path.join(PILOTO, "results")
sys.path.insert(0, PILOTO)
import e7_adam as E7  # noqa: E402


# =====================================================================  (A)
def algebra():
    import sympy as sp
    print("=" * 70)
    print("(A) ALGEBRA (SymPy)")
    print("=" * 70)

    # --- A1. sqrt(vhat) em funcao de |g|: o fator de memoria kappa ----------
    b2, rho, d = sp.symbols("beta_2 rho d", positive=True)
    # |g_{k-d}| = |g_k| e^{rho d}  (o gradiente do morto DECAI a taxa rho/passo)
    # v_k = (1-b2) sum_{d>=0} b2^d g_{k-d}^2 = g_k^2 (1-b2) sum (b2 e^{2rho})^d
    q = b2 * sp.exp(2 * rho)                      # razao da geometrica
    kappa2 = sp.simplify((1 - b2) / (1 - q))      # soma fechada, valida sse q<1
    kappa = sp.sqrt(kappa2)
    # confere a soma fechada numericamente (a geometrica so fecha com |q|<1,
    # e SymPy nao assume isso; entao a checagem e numerica e explicita)
    f = sp.lambdify((b2, rho), kappa2)
    pares = [(v, r) for v in (0.9, 0.99, 0.999) for r in (1e-5, 1e-4, 1e-3)
             if v * np.exp(2 * r) < 1]          # so o ramo convergente
    err = max(abs(f(v, r) - (1 - v) * np.sum((v * np.exp(2 * r))
                                             ** np.arange(2000000)))
              for v, r in pares)
    print("  kappa^2 = vhat/g^2 =", kappa2, "   (converge sse b2*e^{2rho} < 1)")
    print(f"  check numerico da geometrica ({len(pares)} pares b2 x rho "
          f"convergentes de 9): erro max {err:.2e}")
    print("  (os 9-", len(pares), "pares divergentes sao o regime rho > "
          "(1-b2)/2: a")
    print("   descida e rapida demais para a memoria de v acompanhar; NAO e o")
    print("   regime do plato, onde rho ~ 7e-5 e kappa=1.00 medido.)")
    print("  rho->0 (congelado)  => kappa ->", sp.limit(kappa, rho, 0))
    print("  d kappa/d rho > 0 :", sp.simplify(sp.diff(kappa2, rho)))
    print("  kappa >= 1 sempre: vhat le o gradiente de ~tau2 passos ATRAS,")
    print("  que era MAIOR. A memoria de v so infla o limiar por um FATOR.")

    # --- A2. a EDO do sticking e sua solucao fechada ------------------------
    y, k, A, C, eps, kap = sp.symbols("y k A C epsilon kappa", positive=True)
    # y := -beta z (profundidade).  |g| = C e^{-y}.  mhat ~ g (sinal consistente)
    # dy/dk = A * |g| / (kappa|g| + eps),  A = beta*lr*c  (c = |x_face| + 1)
    dydk = A * (C * sp.exp(-y)) / (kap * C * sp.exp(-y) + eps)
    print("\n  dy/dk =", sp.simplify(dydk))
    # separavel:  (kappa + (eps/C) e^{y}) dy = A dk
    prim = sp.integrate(1 / dydk, y)
    print("  int dk/dy dy =", sp.simplify(prim), "   =>  A*k")
    # => kappa*y + (eps/C) (e^y - 1) = A k     [y(0)=0 na morte]
    print("  SOLUCAO FECHADA:  kappa*y + (eps/C)(e^y - 1) = A*k")

    # --- A3. da/dL: derivada implicita em L = ln(1/eps) ---------------------
    L = sp.symbols("L", real=True)
    yf = sp.Function("y")(L)
    F = kap * yf + sp.exp(-L) / C * (sp.exp(yf) - 1) - A * k
    dy = sp.solve(sp.Eq(sp.diff(F, L), 0), sp.diff(yf, L))[0]
    dy = sp.simplify(dy)
    print("\n  dy/dL =", dy)
    # A propria solucao da (eps/C)(e^y - 1) = A k - kappa y.  Substituindo,
    # numerador e denominador em termos de E := (eps/C)e^y = A k - kappa y + eps/C
    Yv, Ak = sp.symbols("Y Ak", positive=True)      # Y = kappa*y  (>=0)
    E = Ak - Yv + eps / C                            # = (eps/C) e^y
    num = E - eps / C                                # (eps/C)(e^y - 1)
    den = kap + E                                    # kappa + (eps/C) e^y
    a_expr = sp.simplify(num / den)
    print("  reescrito com E=(eps/C)e^y e a solucao:")
    print("  a = dy/dL =", a_expr)
    print("  1 - a     =", sp.simplify(1 - a_expr))
    print("  serie em 1/Ak:", sp.simplify(sp.series(a_expr, Ak, sp.oo, 2).removeO()))
    print("\n  ==>  a = 1 - (kappa + eps/C)/(A k) + O((A k)^-2).")
    print("  a=1 e EXATO no limite A*k >> kappa*y (orcamento de descida")
    print("  saturada muito maior que a profundidade atingida).")

    # --- A4. a leitura alternativa (a=1/2) ---------------------------------
    print("\n  ALTERNATIVA descartavel: se o portao fosse vhat ~ eps (e nao")
    print("  sqrt(vhat) ~ eps), entao g^2 ~ eps => 2 beta z = ln eps =>")
    print("  y = (1/2) ln(1/eps):  a = 1/2. As duas leituras sao")
    print("  DISTINGUIVEIS por um fator 2 na inclinacao.")
    print("  Ja o limiar sqrt(vhat) = theta*eps (theta=1, 2, ...) muda")
    print("  b por ln(theta) e NAO muda a. a=1 nao depende da convencao.")

    # --- A5. o que entra em b ---------------------------------------------
    print("\n  b = ln C + ln(A*K_morto) = ln C + ln(beta * lr * c * K_morto)")
    print("     C  : prefator de |g| = C e^{beta z}  (~ |R| * escala de X)")
    print("     c  : |x_face| + 1  (geometria: dz = dw*x_face + db)")
    print("     lr, beta, K_morto = duracao da fase morta em PASSOS")
    print("     kappa (=> beta_2) entra so via -kappa*y/(A K), subdominante")
    print("  ==> beta_2 quase nao mexe em b NESTE regime (A K >> kappa y).")
    return True


# =====================================================================  (B)
def _roda_instrumentada(inst, lr, eps, t_max=8.0, dt=E7.DT, b=E7.BETA):
    """Copia fiel de E7.roda(opt='adam') + registro de |g|, |mhat|, sqrt(vhat).

    Registra a serie da fase MORTA (ate o release) subamostrada, e o estado
    exato no instante do minimo de z (o estacionamento).
    """
    X = np.asarray(inst["X"], float)
    Y = np.asarray(inst["Y"], float)
    p = np.concatenate([inst["PA0"], inst["PB0"]]).astype(float)
    m = np.zeros(4)
    v = np.zeros(4)
    t, k = 0.0, 0
    morto = False
    t_hit = t_rel = None
    zmin = 0.0
    snap = None
    ser = []
    kd = 0
    nmax = int(t_max / dt)
    while k < nmax:
        g = E7.grad(p, X, Y, b)
        k1 = k + 1
        m = E7.B1 * m + (1 - E7.B1) * g
        v = E7.B2 * v + (1 - E7.B2) * g * g
        mh = m / (1 - E7.B1 ** k1)
        vh = v / (1 - E7.B2 ** k1)
        u = mh / (np.sqrt(vh) + eps)
        p -= lr * u
        t += dt
        k += 1
        z0 = p[0] * X[E7.FACE] + p[1]
        if not morto and z0 <= 0 and t > 0.05:
            morto, t_hit = True, t
        elif morto and t_rel is None:
            kd += 1
            if z0 < zmin:
                zmin = z0
                snap = dict(t=t, k_morto=kd, bz=b * z0,
                            g=float(abs(g[0])), mh=float(abs(mh[0])),
                            sv=float(np.sqrt(vh[0])), u=float(abs(u[0])),
                            g1=float(abs(g[1])), sv1=float(np.sqrt(vh[1])))
            if kd % 20 == 0:
                ser.append((t, b * z0, abs(g[0]), abs(mh[0]),
                            np.sqrt(vh[0]), abs(u[0])))
            if z0 >= 0:
                t_rel = t
                break
    return dict(t_hit=t_hit, t_release=t_rel, bz_park=b * zmin,
                K_morto=kd, snap=snap, serie=np.array(ser))


def crossover():
    print("=" * 70)
    print("(B) CROSSOVER EMPIRICO no instante do estacionamento")
    print("=" * 70)
    d = json.load(open(os.path.join(RES_P, "lr_calibrado.json")))
    lrs = d.get("lr", d)
    linhas = []
    print("  ATENCAO: no instante EXATO do minimo o drift INVERTE (e a definicao")
    print("  de minimo), entao mhat passa por um zero e as razoes ali sao")
    print("  artefato. Medimos tambem a CHEGADA AO PISO (primeiro t com")
    print("  y >= y_park - 1, isto e, 1/beta acima do fundo) e a MEDIANA no plato.")
    for eps in E7.EPS_GRID:
        lr = lrs[f"{eps:.0e}"]
        r = _roda_instrumentada(E7.REF, lr, eps)
        s, S = r["snap"], r["serie"]
        yy = -S[:, 1]
        ypark = -r["bz_park"]
        plat = yy >= ypark - 1.0
        i0 = int(np.argmax(plat))
        che = dict(zip(("t", "bz", "g", "mh", "sv", "u"), S[i0]))
        med = {k: float(np.median(S[plat, j]))
               for j, k in ((2, "g"), (3, "mh"), (4, "sv"), (5, "u"))}
        print(f"\n  eps={eps:.0e}  lr={lr:.4e}  y_park={ypark:.3f} "
              f"K_morto={r['K_morto']}  t_rel={r['t_release']}  "
              f"plato={int(plat.sum())*20} passos")
        for rot, D in (("minimo ", s), ("chegada", che), ("mediana", med)):
            print(f"    {rot}: |g|={D['g']:.3e} |mh|={D['mh']:.3e} "
                  f"sqrt(vh)={D['sv']:.3e} |u|={D['u']:.3e} || "
                  f"|g|/eps={D['g']/eps:.3e} sqrt(vh)/eps={D['sv']/eps:.3e} "
                  f"kappa={D['sv']/D['g']:.1f}")
        linhas.append(dict(eps=eps, lr=lr, y=ypark, K=r["K_morto"],
                           g=med["g"], mh=med["mh"], sv=med["sv"], u=med["u"],
                           g_che=che["g"], sv_che=che["sv"]))

    print("\n  DIAGNOSTICO a=1: a lei y = a*ln(1/eps) + b EQUIVALE a")
    print("  |g|_park = theta * eps com theta constante em eps, pois")
    print("  |g| = C e^{-y}.  Exatamente: a = 1 - d ln(theta)/d ln(1/eps)")
    print("  (a menos da deriva de C, que e efeito de instancia, nao de eps).")
    Ls = np.array([np.log(1 / d["eps"]) for d in linhas])
    for nome, key in (("|g|_plato", "g"), ("sqrt(vh)_plato", "sv"),
                      ("|g|_chegada", "g_che"), ("sqrt(vh)_chegada", "sv_che")):
        lt = np.array([np.log(d[key] / d["eps"]) for d in linhas])
        p = Ls > 1.0
        print(f"    theta({nome:16s}) = "
              + " ".join(f"{np.exp(x):.2e}" for x in lt)
              + f" | d ln theta/dL: eps<=1e-2 {_ols(Ls[p], lt[p])[0]:+.4f}"
                f"  todos {_ols(Ls, lt)[0]:+.4f}")
    print("    (ordem dos eps: " + ", ".join(f"{d['eps']:.0e}" for d in linhas) + ")")

    # a identificacao quantitativa theta = 1/(A*K) = 1/(beta*lr*c*K_morto)
    c = abs(E7.REF["X"][E7.FACE]) + 1.0
    print(f"\n  IDENTIFICACAO de theta com 1/(beta*lr*c*K_morto), c=|x_face|+1={c}:")
    for d in linhas:
        A = E7.BETA * d["lr"] * c
        pred = 1.0 / (A * d["K"])
        print(f"    eps={d['eps']:.0e}  theta_medido={d['g']/d['eps']:.3e}  "
              f"1/(A*K)={pred:.3e}  razao={d['g']/d['eps']/pred:.2f}")
    print("  (c so e conhecido a menos de O(1); a razao ~1 confirma a ESCALA,")
    print("   e a constancia da razao em eps confirma a ESTRUTURA.)")
    C = [d["g"] * np.exp(d["y"]) for d in linhas]
    print("  C = |g|_park * e^{y_park} (prefator do canal):",
          " ".join(f"{x:.3e}" for x in C))
    print("  termo intrinseco 1-a = (kappa + eps/C)/(A*K):",
          " ".join(f"{(1 + d['eps']/Ci)/(E7.BETA*d['lr']*c*d['K']):.2e}"
                  for d, Ci in zip(linhas, C)))
    np.save(os.path.join(AQUI, "results", "derivacao_crossover.npy"),
            np.array([[d["eps"], d["y"], d["g"], d["mh"], d["sv"], d["u"],
                       d["K"], d["lr"]] for d in linhas]))
    return linhas


# =====================================================================  (C)
def _ols(L, y):
    Ldm = L - L.mean()
    a = float((Ldm * (y - y.mean())).sum() / (Ldm ** 2).sum())
    b = float(y.mean() - a * L.mean())
    res = y - (a * L + b)
    return a, b, float(np.sqrt((res ** 2).sum() / max(len(y) - 2, 1)))


def ajuste():
    print("=" * 70)
    print("(C) AJUSTE do piloto: a com e sem o ponto eps=1")
    print("=" * 70)
    rs = [json.loads(l) for l in open(os.path.join(RES_P, "e7_runs.jsonl"))]
    ok = [r for r in rs if r["adam"]["t_release"] is not None
          and r["adam"]["n_mortes"] == 1]
    L = np.array([np.log(1 / r["eps"]) for r in ok])
    y = np.array([-r["adam"]["bz_park"] for r in ok])
    lr = np.array([r["lr"] for r in ok])
    dur = np.array([r["adam"]["t_release"] - r["adam"]["t_hit"] for r in ok])

    for rot, msk in [("TODOS os 4 eps", np.ones(len(L), bool)),
                     ("so eps <= 1e-2 ", L > 1.0)]:
        a, b, s = _ols(L[msk], y[msk])
        print(f"  {rot}: n={msk.sum():3d}  a={a:.4f}  b={b:.3f}  resid={s:.3f}")

    print("\n  medianas por eps (o joelho em eps=1):")
    print("   eps      L      y_med    lr         K_morto(med)   incl. local")
    ys, Ls, lrm, Km = [], [], [], []
    for e in sorted(set(r["eps"] for r in ok), reverse=True):
        m = np.array([r["eps"] == e for r in ok])
        Ls.append(np.log(1 / e)); ys.append(float(np.median(y[m])))
        lrm.append(float(np.median(lr[m]))); Km.append(float(np.median(dur[m])) / E7.DT)
    for i in range(len(Ls)):
        inc = "" if i == 0 else f"{(ys[i]-ys[i-1])/(Ls[i]-Ls[i-1]):.4f}"
        print(f"  {np.exp(-Ls[i]):.0e}  {Ls[i]:5.2f}  {ys[i]:7.3f}  "
              f"{lrm[i]:.3e}  {Km[i]:10.0f}   {inc}")

    # vazamentos de projeto que a derivacao preve entrarem em `a`
    Ls, lrm, Km = map(np.array, (Ls, lrm, Km))
    peq = Ls > 1.0
    dlr, _, _ = _ols(Ls[peq], np.log(lrm[peq]))
    dK, _, _ = _ols(Ls[peq], np.log(Km[peq]))
    print(f"\n  VAZAMENTOS (b = ln C + ln(beta*lr*c*K) => entram na inclinacao):")
    print(f"    d ln lr / dL      = {dlr:+.5f}   (lr recalibrado por eps)")
    print(f"    d ln K_morto / dL = {dK:+.5f}   (duracao da fase morta)")
    print(f"    a_previsto = 1 + soma = {1+dlr+dK:.4f}")
    print(f"    a_medido (eps<=1e-2)  = {_ols(L[L>1.0], y[L>1.0])[0]:.4f}")

    # ---- (D) guarda de curvatura e regressao DESCONTADA do vazamento -------
    print("\n  (D) GUARDA DE CURVATURA (regra §3 do pre-registro) e correcao")
    lrc = np.array([r["lr"] for r in ok])
    Kc = dur / E7.DT
    ystar = y - np.log(lrc * Kc)      # y* = y - ln(lr*K) = ln C + L + ln(beta*c)
    for rot, m in [("4 eps    ", np.ones(len(L), bool)), ("eps<=1e-2", L > 1.0)]:
        Z = np.vstack([L[m] ** 2, L[m], np.ones(m.sum())]).T
        q, aq, bq = np.linalg.lstsq(Z, y[m], rcond=None)[0]
        gat = abs(q) * L[m].max() ** 2
        a2 = _ols(L[m], ystar[m])[0]
        print(f"   {rot}: q={q:+.5f}  |q|Lmax^2={gat:.3f} "
              f"{'DISPARA' if gat > 1.0 else 'ok     '} | "
              f"a(y bruto)={_ols(L[m], y[m])[0]:.4f}  "
              f"a(y* descontado ln(lr*K))={a2:.4f}")
    print("   ==> o joelho em eps=1 e VAZAMENTO DE PROJETO (lr recalibrado")
    print("       + fase morta mais curta), nao curvatura da lei fisica.")
    return dlr, dK


if __name__ == "__main__":
    os.makedirs(os.path.join(AQUI, "results"), exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tudo"
    if cmd in ("algebra", "tudo"):
        algebra()
    if cmd in ("crossover", "tudo"):
        crossover()
    if cmd in ("ajuste", "tudo"):
        ajuste()
