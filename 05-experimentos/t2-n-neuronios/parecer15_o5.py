"""O5: mede rho_B(t_hit)=||theta_B^beta - theta_B^relu|| e rho_A no hit, vs beta.
Predicao do draft (L2' passo 4): rho_B = O(beta^-2), rho_A = O(beta^-1).
RK4 para que o erro de integracao fique <<  beta^-2.
"""
from pathlib import Path
import sys, os, json
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent / '.'))
from verifica_forcing import sorteia, readout, FACE, softplus, sig, hibrido, DT0

def field_relu(P, X, Y, a):
    Z = (P[:,0]*X[:,None]).T + P[:,1][:,None]
    act = (Z > 0).astype(float)
    r = (a[:,None]*np.maximum(Z,0.0)).sum(axis=0) - Y
    return -a[:,None]*np.stack([(act*(r*X)).mean(axis=1), (act*r).mean(axis=1)], axis=1)

def field_sp(P, X, Y, a, b):
    Z = (P[:,0]*X[:,None]).T + P[:,1][:,None]
    r = (a[:,None]*softplus(Z,b)).sum(axis=0) - Y
    s = sig(Z,b)
    return -a[:,None]*np.stack([(s*(r*X)).mean(axis=1), (s*r).mean(axis=1)], axis=1)

def rk4(F, P, dt, n):
    for _ in range(n):
        k1=F(P); k2=F(P+.5*dt*k1); k3=F(P+.5*dt*k2); k4=F(P+dt*k3)
        P = P + dt/6*(k1+2*k2+2*k3+k4)
    return P

def run(seed, m, betas, dt=2e-6):
    X, Y, P0 = sorteia(seed, m)
    h = hibrido(X, Y, P0, DT0)
    if not h["ok"]: return None
    a = readout(m)
    t_hit = h["t_hit"]
    n = int(round(t_hit/dt))
    Pr = rk4(lambda P: field_relu(P,X,Y,a), P0.copy(), dt, n)
    out = {"seed":seed,"m":m,"t_hit":t_hit,"R_hit":h["R_hit"],"rows":[]}
    for b in betas:
        Pb = rk4(lambda P: field_sp(P,X,Y,a,b), P0.copy(), dt, n)
        d = Pb - Pr
        out["rows"].append((b, float(np.linalg.norm(d[1:])), float(np.linalg.norm(d[0]))))
    return out

if __name__ == "__main__":
    betas = [1e3, 2e3, 4e3, 8e3]
    recs = [json.loads(l) for l in open(str(Path(__file__).resolve().parent / 'forcing_m.jsonl'))]
    ok = [r for r in recs if r["passou"]]
    sel = []
    for m in (4,8,16):
        sel += [r["seed"] for r in ok if r["m"]==m][:2]
    print("beta      rho_B          *beta^2     rho_A         *beta")
    for m in (4,8,16):
        for s in [r["seed"] for r in ok if r["m"]==m][:2]:
            o = run(s, m, betas)
            if o is None: print(s,m,"skip"); continue
            print(f"-- seed {s} m={m} t_hit={o['t_hit']:.4f} v=c|R_hit|")
            for b, rB, rA in o["rows"]:
                print(f"{b:8.0f}  {rB:.4e}  {rB*b*b:9.3f}  {rA:.4e}  {rA*b:8.4f}")
