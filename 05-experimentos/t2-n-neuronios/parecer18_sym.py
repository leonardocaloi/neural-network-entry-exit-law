import sympy as sp, itertools, numpy as np
# (1) m=3,n=3 FULL symbolic: block-diagonality of S structural?
for (m,n) in [(3,3),(3,2),(2,4)]:
    w=sp.symbols(f'w0:{m}',real=True); b=sp.symbols(f'b0:{m}',real=True); v=sp.symbols(f'v0:{m}',real=True)
    x=sp.symbols(f'x0:{n}',real=True); y=sp.symbols(f'y0:{n}',real=True)
    I=[[sp.Symbol(f'i{k}_{i}',real=True) for i in range(n)] for k in range(m)]
    f=[sum(v[k]*I[k][i]*(w[k]*x[i]+b[k]) for k in range(m)) for i in range(n)]
    r=[f[i]-y[i] for i in range(n)]
    U=sp.Rational(1,2)/n*sum(ri**2 for ri in r)
    th=[t for k in range(m) for t in (w[k],b[k],v[k])]
    H=sp.hessian(U,th)
    J=sp.Matrix(n,3*m,lambda i,j: sp.diff(f[i],th[j]))
    S=sp.expand(H-J.T*J/n)
    ck=[sp.Matrix([sum(r[i]*I[k][i]*x[i] for i in range(n))/n, sum(r[i]*I[k][i] for i in range(n))/n]) for k in range(m)]
    A=sp.zeros(3*m,3*m)
    for k in range(m):
        A[3*k+2,3*k]=A[3*k,3*k+2]=ck[k][0]; A[3*k+2,3*k+1]=A[3*k+1,3*k+2]=ck[k][1]
    print(f"m={m},n={n}: S == blockdiag arrow-blocks ?", sp.simplify(S-A)==sp.zeros(3*m,3*m))
# (2) independent char poly of 3x3
mu,c1,c2=sp.symbols('mu c1 c2',real=True)
Sk=sp.Matrix([[0,0,c1],[0,0,c2],[c1,c2,0]])
print("charpoly:", sp.factor(sp.expand((Sk-mu*sp.eye(3)).det())))
# (3) 3-layer/2-hidden-layer sanity: does block-diag survive m>1 with SHARED input only? yes (checked above)
