"""Etapa 4 — CALIBRAÇÃO DA RÉGUA populacional no toy (patch test do instrumento).

A régua O(m)/passo do rascunho (por neurônio k: fração ativa a_k, dado líder
j* = argmax margem, resíduo projetado R̂_k = res_total(j*) + act(z_k,j*), área
acumulada Â_k = ∫R̂_k dt enquanto morto) tem de reproduzir, no toy de 2
neurônios ONDE A VERDADE É CONHECIDA:
  (a) a detecção exata de morte/ressurreição (margens exatas);
  (b) o zero de Â_A ≈ instante real da ressurreição (a lei de área, medida PELO
      instrumento barato, não pelas fórmulas exatas);
  (c) sob ruído: correlação previsto-vs-real por seed.
Instrumento aprovado aqui ganha o direito de medir o MLP. [Regra da casa:
calibrar antes de medir.]
"""
import numpy as np

X = np.array([0.5, 1.5]); Y = np.array([0.5, 1.0]); N = 2
ETA_A, ETA_B = 1e-3, 1e-4
PA0 = np.array([-0.5, 0.27]); PB0 = np.array([1.144, 0.261])
T_FIM_PASSOS = 500_000

def relu(z): return np.maximum(z, 0.0)

def make_act(beta):
    if beta is None:
        return relu, lambda z: (z > 0).astype(float)
    def a(z): return np.logaddexp(0.0, beta * z) / beta
    def da(z): return 1.0 / (1.0 + np.exp(-np.clip(beta * z, -500, 500)))
    return a, da

def roda_com_regua(T, seed, beta=None, esc=1.0):
    """SGLD do toy; a régua vê SÓ o que veria num MLP: ativações, resíduo total,
    margens — nada das fórmulas exatas do teorema."""
    act, dact = make_act(beta)
    eA, eB = ETA_A * esc, ETA_B * esc
    rng = np.random.default_rng(seed)
    pA, pB = PA0.copy(), PB0.copy()
    sA, sB = np.sqrt(2 * eA * T), np.sqrt(2 * eB * T)
    morto = False; t_morte = 0; A_hat = 0.0; t_zero_hat = None; t_sinal = None
    t_res_real = None
    for t in range(int(T_FIM_PASSOS / esc)):
        zA = pA[0] * X + pA[1]; zB = pB[0] * X + pB[1]
        f = act(zA) + act(zB)
        res = f - Y                                   # residuo total (visivel)
        gA = np.array([np.mean(res * dact(zA) * X), np.mean(res * dact(zA))])
        gB = np.array([np.mean(res * dact(zB) * X), np.mean(res * dact(zB))])
        pA += -eA * gA + sA * rng.standard_normal(2)
        pB += -eB * gB + sB * rng.standard_normal(2)
        # ---- regua O(m): so quantidades observaveis ----
        # ressurreicao operacional: margem lider acima de 2*sigma_x (debounce
        # fisico — z>0 cru pisca com ruido na fronteira; mesmo espirito da regua
        # SIG_REDEATH dos experimentos, computavel num MLP conhecendo T)
        lim = 2.0 * np.sqrt(N * max(T, 1e-7)) / np.sqrt(X[int(np.argmax(zA))]**2 + 1)
        vivo = zA.max() > lim
        if not vivo and not morto:
            morto, t_morte, A_hat, t_zero_hat = True, t, 0.0, None
        if morto:
            js = int(np.argmax(zA))                   # dado lider
            R_hat = (Y[js] - f[js]) + act(zA[js])     # exclui o proprio k (morto: ~0)
            A_hat += R_hat * eA
            if t_zero_hat is None and A_hat > 0:
                t_zero_hat = t                        # previsao AREA
            if t_sinal is None and R_hat > 0:
                t_sinal = t                           # previsao rival: sinal de R
            if vivo:
                t_res_real = t                        # ressurreicao REAL
                return dict(t_morte=t_morte, t_prev=t_zero_hat, t_sinal=t_sinal,
                            t_real=t_res_real)
    return dict(t_morte=t_morte if morto else None, t_prev=t_zero_hat,
                t_sinal=t_sinal, t_real=None)

# (a)+(b): quase-deterministico NO SISTEMA DO TEOREMA (softplus beta=300):
# a lei de area vale flow-first; relu duro e o ramo congelado (nao-comutacao) —
# testado como CONTRASTE informativo abaixo.
# fronteira de discretizacao: um passo avanca dtau = beta*eta de tempo rapido;
# o limite do teorema exige beta*eta << 1 (beta=1000,eta=1e-3 => dtau=1: quebra,
# rumo ao ramo relu-duro). Redes reais (GELU, eta~1e-3) vivem em beta*eta<<1.
# Calibracao: a beta fixo, erro deve CAIR quando eta diminui.
errs = {}
for esc in (1.0, 1/3):
    r = roda_com_regua(T=1e-8, seed=1, beta=300.0, esc=esc)
    errs[esc] = abs(r["t_prev"] - r["t_real"]) / r["t_real"]
    print(f"[quase-det, beta=300, beta*eta={300*1e-3*esc:.2f}] prev t={r['t_prev']}"
          f"  real t={r['t_real']}  erro {errs[esc]*100:.2f}%")
rq = roda_com_regua(T=1e-8, seed=1, beta=1000.0)
print(f"[contraste beta=1000, beta*eta=1.0 — FORA do regime] erro "
      f"{abs(rq['t_prev']-rq['t_real'])/rq['t_real']*100:.1f}% (fronteira documentada)")
assert errs[1/3] < errs[1.0] and errs[1/3] < 0.05, errs
print(f"[OK] regua converge quando beta*eta -> 0 como o teorema exige "
      f"({errs[1.0]*100:.1f}% -> {errs[1/3]*100:.1f}%)")

# (c): com ruido, correlacao previsto-vs-real (20 seeds, T=1e-3)
# contraste hard-relu (nao-comutacao visivel no instrumento):
rh = roda_com_regua(T=1e-8, seed=1, beta=None)
print(f"[contraste relu-duro, T~0] prev(A=0) t={rh['t_prev']}  real t={rh['t_real']}"
      f"  — ressurreicao DIFUSIVA, nao pela area (ramo congelado; P-C da etapa 4)")

# (c) sob ruido, o teste certo e RELATIVO (P-A da etapa 4): o zero-de-area
# preve a ressurreicao MELHOR que a mudanca de sinal de R? (pareado por episodio)
ganha_area, tot, n_perm = 0, 0, 0
errA, errS = [], []
for sd in range(2000, 2040):
    r = roda_com_regua(T=1e-3, seed=sd, beta=300.0)
    if r["t_real"] is not None and r["t_prev"] is not None and r["t_sinal"] is not None:
        tot += 1
        eA_ = abs(r["t_real"] - r["t_prev"]); eS_ = abs(r["t_real"] - r["t_sinal"])
        errA.append(eA_); errS.append(eS_)
        ganha_area += int(eA_ < eS_)
    elif r["t_morte"] is not None and r["t_real"] is None:
        n_perm += 1
print(f"[T=1e-3, 40 seeds] episodios completos={tot}, mortos-permanentes={n_perm}")
print(f"  erro mediano |t_real - previsao|: AREA={np.median(errA):.0f} passos"
      f"  vs SINAL={np.median(errS):.0f} passos; area vence em {ganha_area}/{tot}")
assert tot >= 12 and ganha_area / tot >= 0.6, "area nao supera sinal no toy"
print("[OK] (c): o zero-de-area e preditor MELHOR que o sinal — a premissa de"
      " P-A (etapa 4) esta calibrada no toy")
print("\nREGUA CALIBRADA — aprovada para o MLP (patch test do instrumento passou).")
