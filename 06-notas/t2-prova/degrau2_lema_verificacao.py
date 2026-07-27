"""Verificacao SIMBOLICA (SymPy) + NUMERICA do LEMA NOVO do Degrau 2:
curvatura negativa bloco-diagonal.

(A) bloco 3x3 simbolico, ordem (w_k, b_k, v_k):  autovalores {0, +||c_k||, -||c_k||}.
(B) Hessiano simbolico completo de U = (1/2n) sum_i (sum_k v_k 1_{k,i}(w_k x_i+b_k) - y_i)^2
    para m=2, n=2: confere Gauss-Newton (1/n)J^T J  +  curvatura residual BLOCO-DIAGONAL,
    com bloco k exatamente [[0,0,c1],[0,0,c2],[c1,c2,0]], c_k=(1/n) sum_i r_i 1_{k,i}(x_i,1).
(C) numerico: lambda_min(Hess) >= -max_k ||c_k||, em 20000 pontos aleatorios (m,n variados).

Rodar:  python degrau2_lema_verificacao.py
"""
import numpy as np
import sympy as sp

print("=== (A) bloco 3x3 simbolico, ordem (w,b,v) ===")
c1, c2 = sp.symbols("c1 c2", real=True)
Bk = sp.Matrix([[0, 0, c1], [0, 0, c2], [c1, c2, 0]])
ev = Bk.eigenvals()
print("  bloco =", Bk.tolist())
print("  autovalores:", {sp.simplify(k): v for k, v in ev.items()})
alvo = {sp.Integer(0): 1, sp.sqrt(c1**2 + c2**2): 1, -sp.sqrt(c1**2 + c2**2): 1}
ok_A = all(any(sp.simplify(k - ka) == 0 for ka in alvo) for k in ev)
print("  == {0, +||c||, -||c||} ?", ok_A)
assert ok_A

print("\n=== (B) Hessiano simbolico completo, m=2, n=2, padrao congelado ===")
m, n = 2, 2
w = sp.symbols("w1 w2", real=True)
b = sp.symbols("b1 b2", real=True)
v = sp.symbols("v1 v2", real=True)
x = sp.symbols("x1 x2", real=True)
y = sp.symbols("y1 y2", real=True)
ind = sp.symbols("i11 i12 i21 i22", real=True)          # 1_{k,i}, constantes (padrao congelado)
I = [[ind[0], ind[1]], [ind[2], ind[3]]]

f = [sum(v[k] * I[k][i] * (w[k] * x[i] + b[k]) for k in range(m)) for i in range(n)]
r = [sp.expand(f[i] - y[i]) for i in range(n)]
U = sp.Rational(1, 2) / n * sum(r[i] ** 2 for i in range(n))

theta = [w[0], b[0], v[0], w[1], b[1], v[1]]            # por unidade
H = sp.hessian(U, theta)

J = sp.Matrix(n, 3 * m, lambda i, j: sp.diff(f[i], theta[j]))
GN = sp.expand(J.T * J / n)
Res = sp.expand(H - GN)

ck = [sp.Matrix([sum(r[i] * I[k][i] * x[i] for i in range(n)) / n,
                 sum(r[i] * I[k][i] for i in range(n)) / n]) for k in range(m)]
Alvo = sp.zeros(3 * m, 3 * m)
for k in range(m):
    Alvo[3 * k + 2, 3 * k + 0] = ck[k][0]
    Alvo[3 * k + 0, 3 * k + 2] = ck[k][0]
    Alvo[3 * k + 2, 3 * k + 1] = ck[k][1]
    Alvo[3 * k + 1, 3 * k + 2] = ck[k][1]
ok_B = sp.simplify(Res - Alvo) == sp.zeros(3 * m, 3 * m)
print("  Hess - (1/n)J^T J  ==  bloco-diagonal com blocos [[0,0,c1],[0,0,c2],[c1,c2,0]] ?", ok_B)
print("  (em particular: ZERO acoplamento residual entre neuronios distintos)")
assert ok_B

print("\n=== (C) cota lambda_min >= -max_k ||c_k||, 20000 pontos aleatorios ===")
rng = np.random.default_rng(20260728)
folga_min, sat = np.inf, 0
N = 20000
for t in range(N):
    mm = int(rng.integers(1, 7)); nn = int(rng.integers(1, 9))
    W = rng.normal(size=mm); B = rng.normal(size=mm); V = rng.normal(size=mm)
    X = rng.normal(size=nn); Y = rng.normal(size=nn)
    Z = W[:, None] * X[None, :] + B[:, None]
    act = (Z > 0).astype(float)
    rr = (V[:, None] * act * Z).sum(axis=0) - Y
    ckn = np.stack([(act * (rr * X)).mean(axis=1), (act * rr).mean(axis=1)], axis=1)
    Lam = float(np.linalg.norm(ckn, axis=1).max())
    Jn = np.zeros((nn, 3 * mm))
    for k in range(mm):
        Jn[:, 3 * k] = V[k] * act[k] * X
        Jn[:, 3 * k + 1] = V[k] * act[k]
        Jn[:, 3 * k + 2] = act[k] * Z[k]
    Hn = Jn.T @ Jn / nn
    for k in range(mm):
        Hn[3 * k + 2, 3 * k] += ckn[k, 0]; Hn[3 * k, 3 * k + 2] += ckn[k, 0]
        Hn[3 * k + 2, 3 * k + 1] += ckn[k, 1]; Hn[3 * k + 1, 3 * k + 2] += ckn[k, 1]
    lmin = float(np.linalg.eigvalsh(Hn)[0])
    folga = lmin + Lam                       # >= 0 exigido
    folga_min = min(folga_min, folga / max(Lam, 1e-12))
    if folga / max(Lam, 1e-12) < 1e-10:
        sat += 1
print(f"  folga relativa MINIMA sobre {N} pontos: {folga_min:+.3e}   (>= 0 exigido)")
print(f"  saturacoes (folga relativa < 1e-10): {sat}/{N} = {100*sat/N:.1f}%")
assert folga_min > -1e-10
print("\nTUDO PASSA.")
