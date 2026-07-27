"""Analise dos abertos P7 (Lambda em m) e P8 (quociente de Rayleigh)."""
import json
import os

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(20260727)


def ic_med(a, B=20000):
    a = np.asarray(a, float)
    b = np.median(RNG.choice(a, (B, len(a)), replace=True), axis=1)
    return float(np.median(a)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def p7():
    f = os.path.join(AQUI, "p7_lambda_m.jsonl")
    if not os.path.exists(f):
        return
    R = [json.loads(l) for l in open(f)]
    print("=== P7 — Lambda x m ===")
    print(f"{'m':>3} {'N':>3} | {'Lambda_meas mediana [IC95]':>34} | "
          f"{'lam_max(G_B) mediana [IC95]':>34} | {'T_ep med':>9} | maxrel(ck,meas)")
    tab = {}
    for m in sorted({r["m"] for r in R}):
        S = [r for r in R if r["m"] == m]
        L = np.array([r["Lambda_meas"] for r in S])
        C = np.array([r["Lambda_ck"] for r in S])
        G = np.array([r["lam_max_GB"] for r in S])
        T = np.array([r["T_ep"] for r in S])
        ml, lo, hi = ic_med(L)
        mg, glo, ghi = ic_med(G)
        rel = float(np.max(np.abs(C - L) / L))
        tab[m] = (len(S), ml, lo, hi, mg, glo, ghi, float(np.median(T)),
                  float(np.percentile(L, 25)), float(np.percentile(L, 75)),
                  float(np.percentile(G, 25)), float(np.percentile(G, 75)), rel)
        print(f"{m:>3} {len(S):>3} | {ml:8.3f} [{lo:7.3f},{hi:7.3f}] IQR[{np.percentile(L,25):6.2f},{np.percentile(L,75):6.2f}] | "
              f"{mg:8.3f} [{glo:7.3f},{ghi:7.3f}] IQR[{np.percentile(G,25):6.2f},{np.percentile(G,75):6.2f}] | "
              f"{np.median(T):9.4f} | {rel:.2e}")
    ms = sorted(tab)
    if len(ms) >= 2:
        m0 = ms[0]
        print("\nrazoes a m=%d (mediana):" % m0)
        for m in ms:
            print(f"  m={m:>2}: Lambda x{tab[m][1]/tab[m0][1]:.2f}  "
                  f"lam_max(G_B) x{tab[m][4]/tab[m0][4]:.2f}   (linear seria x{m/m0:.0f})")
        # regressao log-log
        x = np.log([m for m in ms])
        for j, nome in ((1, "Lambda_meas"), (4, "lam_max(G_B)")):
            y = np.log([tab[m][j] for m in ms])
            s = np.polyfit(x, y, 1)[0]
            print(f"  expoente log-log de {nome}: {s:+.3f}  (plano=0, linear=1)")


def p8():
    f = os.path.join(AQUI, "p8_rayleigh.jsonl")
    if not os.path.exists(f):
        return
    R = [json.loads(l) for l in open(f)]
    print("\n=== P8 — quociente de Rayleigh ===")
    print(f"N = {len(R)} instancias, m=4")
    print(f"{'seed':>9} {'T_ep':>8} {'Lam_ep':>8} {'q_sup':>9} {'q_med':>9} "
          f"{'int q dt':>10} {'Lam*T':>9} {'razao':>8} {'log||D||f/i':>11}")
    for r in R:
        raz = r["int_q"] / r["Lam_ep_T_ep"] if r["Lam_ep_T_ep"] else float("nan")
        print(f"{r['seed']:>9} {r['T_ep']:8.4f} {r['Lambda_ep']:8.3f} {r['q_sup']:9.3f} "
              f"{r['q_med']:9.3f} {r['int_q']:10.5f} {r['Lam_ep_T_ep']:9.5f} "
              f"{raz:8.3f} {r['log_growth']:11.4f}")
    for nome in ("q_sup", "q_med", "q_min", "int_q", "Lam_ep_T_ep", "log_growth"):
        a = np.array([r[nome] for r in R])
        m, lo, hi = ic_med(a)
        print(f"  {nome:12s} mediana {m:10.5f} IC95 [{lo:10.5f},{hi:10.5f}]  "
              f"min {a.min():.5f} max {a.max():.5f}")
    raz = np.array([r["int_q"] / r["Lam_ep_T_ep"] for r in R])
    m, lo, hi = ic_med(raz)
    print(f"  razao int_q / (Lam_ep T_ep): mediana {m:.4f} IC95 [{lo:.4f},{hi:.4f}] "
          f"| min {raz.min():.4f} max {raz.max():.4f} | negativas {int((raz<0).sum())}/{len(raz)}")
    eg = np.array([np.exp(r["int_q"]) for r in R])
    el = np.array([np.exp(r["Lam_ep_T_ep"]) for r in R])
    print(f"  fator exp(int q) mediana {np.median(eg):.4f} vs exp(Lam_ep T_ep) mediana {np.median(el):.4f}")
    ra = np.array([r["q_sup"] / r["Lambda_ep"] for r in R])
    m, lo, hi = ic_med(ra)
    print(f"  razao q_sup/Lambda_ep: mediana {m:.4f} IC95 [{lo:.4f},{hi:.4f}] max {ra.max():.4f}")


if __name__ == "__main__":
    p7()
    p8()
