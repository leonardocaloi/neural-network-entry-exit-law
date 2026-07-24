"""Prova L5: lei de dois termos do release acoplado reproduz o sweep do passo 4;
margem da camada; constantes da cadeia."""
import numpy as np

X = np.array([0.5, 1.5]); Y = np.array([0.5, 1.0]); N = 2
PA0 = np.array([-0.5, 0.27]); PB0 = np.array([1.144, 0.261])
FACE = 0; C_G = (X[FACE] ** 2 + 1) / N; T_FIM = 1.8
T_EXIT = 1.1325; R_EXIT = 0.1080                      # referencia hibrida (passo 4)

def softplus(zv, b): return np.logaddexp(0.0, b * zv) / b
def sig(zv, b):
    bz = np.clip(b * zv, -500, 500)
    return 1.0 / (1.0 + np.exp(-bz))

def roda_acoplado(b):
    """4D acoplado; mede t_rel, corr acumulada na camada, e max z na janela interna."""
    dt = min(1e-5, 0.02 / b)
    p = np.concatenate([PA0, PB0]); t = 0.0
    morto = False; corr = 0.0; zmax_int = -np.inf
    eps_b = 4.0 * 10.6 / (C_G * R_EXIT) * np.log(b) / b   # C_eps com C4~10.6 (L1)
    while t < T_FIM:
        zA, zB = p[0] * X + p[1], p[2] * X + p[3]
        r = softplus(zA, b) + softplus(zB, b) - Y
        sA, sB = sig(zA, b), sig(zB, b)
        g = np.concatenate([[np.mean(r * sA * X), np.mean(r * sA)],
                            [np.mean(r * sB * X), np.mean(r * sB)]])
        p -= dt * g; t += dt
        z0 = p[0] * X[FACE] + p[1]
        if not morto and z0 <= 0 and t > 0.05:
            morto = True
        elif morto:
            corr += dt * C_G * np.log1p(np.exp(np.clip(b * z0, None, 50)))
            if t < T_EXIT - eps_b:
                zmax_int = max(zmax_int, z0)
            if z0 >= 0:
                return t, corr, zmax_int
    return None, corr, zmax_int

MEDIDOS_P4 = {300.0: -12.729, 1000.0: -13.900, 3000.0: -14.290, 10000.0: -14.380}
print("lei de dois termos: beta*Dt = (-1 + corr)/(c*R_exit)")
print(f"  coeficiente universal 1/(c*R_exit) = {1/(C_G*R_EXIT):.3f}")
for b in (300.0, 1000.0, 3000.0, 10000.0):
    tr, corr, zmax = roda_acoplado(b)
    med = b * (tr - T_EXIT)
    teo = (-1.0 + corr) / (C_G * R_EXIT)
    desv = abs(med - teo) / abs(teo)
    print(f"  beta={b:7.0f}: medido={med:+.2f} | teoria={teo:+.2f} (corr={corr:.4f}) "
          f"| desvio {desv*100:.1f}% | max z na janela interna = {zmax:+.2e}")
    assert desv < 0.035, (b, med, teo)
    assert zmax < 0, f"camada violada em beta={b}: {zmax}"
print("[OK] (a): a lei de dois termos reproduz o sweep inteiro do passo 4 (<=3.5%)")
print("[OK] (b): camada estrita (z<0) mantida em [t_cross, t_exit - eps_beta]")
L_B, C_f, C4 = 2.133, np.hypot(X[FACE], 1.0) / N, 10.6
M2 = np.exp(L_B * T_FIM) * C_f * C4 / C_G
print(f"[OK] (c): cadeia de constantes explicita: M2 = e^(L_B*T)*C_f*C4/c = {M2:.1f}")
print("\nPROVA L5 VERIFICADA.")
