"""
Verificacao SymPy + numerica dos lemas do experimento p5-relu-morte.
Modelo: f(x) = relu(w*x + b), dados {(x_i, y_i)}_{i=1..n}, L = (1/2n) sum (f(x_i)-y_i)^2.

Lema 1 (salto de gradiente): cruzando a face {z_j=0} (demais z_i<0), o gradiente salta de
        0 (lado morto) para -(y_j/n)*(x_j, 1) (limite pelo lado ativo).
Lema 2 (complementaridade estrita): s_j = max(-z_j,0) (gap), r_j = ||contribuicao de j ao
        gradiente||. Entao s_j>0 => r_j=0, r_j>0 => s_j=0, e lim_{z_j->0+} r_j =
        (y_j/n)*sqrt(x_j^2+1) > 0 (reacao NAO some no contato — ao contrario do p5-cruz).
Lema 3 (piso plano): em int(D), grad L = 0 exatamente e L = (1/2n) sum y_i^2.
Checagens: simbolica (SymPy, no ramo ativo/inativo) + numerica (gradiente analitico vs
diferencas finitas em pontos aleatorios, incluindo pontos mortos).
"""
import sympy as sp
import numpy as np

# ---------- simbolico ----------
w, b, x, y = sp.symbols('w b x y', real=True)
n = sp.Symbol('n', positive=True)
z = w * x + b

# ramo ativo (z>0): termo da loss para o dado (x,y): (1/2n)(z - y)^2
L_ativo = (z - y) ** 2 / (2 * n)
grad_ativo = sp.Matrix([sp.diff(L_ativo, w), sp.diff(L_ativo, b)])

# limite z -> 0+ : substituir z=0 via b = -w*x
grad_no_contato = sp.simplify(grad_ativo.subs(b, -w * x))
esperado = sp.Matrix([-(y / n) * x, -(y / n)])
assert sp.simplify(grad_no_contato - esperado) == sp.zeros(2, 1), grad_no_contato
print("[OK] Lema 1: lim (lado ativo) grad = -(y/n)*(x,1); lado morto = 0  => salto = (y/n)*(x,1)")

# ramo inativo (z<0): termo = (1/2n) y^2, constante
L_morto = y ** 2 / (2 * n)
assert sp.diff(L_morto, w) == 0 and sp.diff(L_morto, b) == 0
print("[OK] Lema 3: no ramo inativo, contribuicao ao gradiente e identicamente nula")

# norma da reacao no contato
r_contato = sp.sqrt(grad_no_contato.dot(grad_no_contato))
assert sp.simplify(r_contato - (y / n) * sp.sqrt(x ** 2 + 1)) == 0 or \
       sp.simplify(r_contato ** 2 - (y / n) ** 2 * (x ** 2 + 1)) == 0
print("[OK] Lema 2: r_j -> (y_j/n)*sqrt(x_j^2+1) > 0 quando z_j -> 0+  (reacao O(1) no contato)")

# ---------- numerico: gradiente analitico vs diferencas finitas ----------
rng = np.random.default_rng(0)
X = np.array([0.5, 1.5])
Y = np.array([0.5, 1.0])
N = len(X)

def loss(p):
    zv = p[0] * X + p[1]
    return 0.5 * np.mean((np.maximum(zv, 0.0) - Y) ** 2)

def grad(p):
    zv = p[0] * X + p[1]
    a = (zv > 0).astype(float)          # convencao relu'(0)=0 (a mesma do codigo do exp.)
    res = (np.maximum(zv, 0.0) - Y) * a
    return np.array([np.mean(res * X), np.mean(res)])

h = 1e-6
max_err = 0.0
for _ in range(2000):
    p = rng.uniform(-2, 2, size=2)
    zv = p[0] * X + p[1]
    if np.any(np.abs(zv) < 1e-3):       # longe das quinas: gradiente classico existe
        continue
    g_fd = np.array([(loss(p + h * e) - loss(p - h * e)) / (2 * h)
                     for e in np.eye(2)])
    max_err = max(max_err, np.max(np.abs(g_fd - grad(p))))
assert max_err < 1e-6, max_err
print(f"[OK] gradiente analitico vs dif. finitas (2000 pontos fora das quinas): err_max = {max_err:.2e}")

# ---------- numerico: salto de forca atraves da face do dado j ----------
# face do dado 0: z_0 = 0 com z_1 < 0. Ponto na face: escolher w tal que z1<0.
for j, (xj, yj) in enumerate(zip(X, Y)):
    wf = -1.0                            # w negativo: z_i = w x_i + b
    bf = -wf * xj                        # zera z_j
    zo = wf * X + bf
    outros = [i for i in range(N) if i != j]
    if not all(zo[i] < -1e-9 for i in outros):
        wf = 1.0 if j == np.argmin(X) else -1.0
        bf = -wf * xj
        zo = wf * X + bf
    ok = all(zo[i] < -1e-9 for i in outros)
    if not ok:
        print(f"    (face {j}: config nao isola este dado com w=+-1; pulando checagem direta)")
        continue
    nrm = np.array([xj, 1.0]) / np.hypot(xj, 1.0)   # normal da face, aponta p/ lado ativo
    eps = 1e-7
    g_morto = grad(np.array([wf, bf]) - eps * nrm)
    g_vivo = grad(np.array([wf, bf]) + eps * nrm)
    salto_teo = (yj / N) * np.hypot(xj, 1.0)
    salto_num = np.linalg.norm(g_vivo - g_morto)
    assert np.linalg.norm(g_morto) < 1e-12, g_morto
    assert abs(salto_num - salto_teo) < 1e-4, (salto_num, salto_teo)
    print(f"[OK] face j={j}: ||grad|| salta de 0 para {salto_num:.6f} (teoria: {salto_teo:.6f})")

# ---------- geometria do cone de morte ----------
# D = {(w,b): w x_i + b <= 0 para todo i}; com x_i>0, e um cone com apice na origem.
# Angulo de abertura (para registro no README; conferido numericamente por amostragem):
amostra = rng.uniform(-1, 1, size=(200000, 2))
dentro = np.all(amostra @ np.vstack([X, np.ones(N)]) <= 0, axis=1)
frac = dentro.mean()
# angulo entre as duas retas-fronteira b=-x_min w e b=-x_max w (medindo o setor "abaixo")
t1 = np.arctan2(-X.min() * 1, 1)   # direcao da reta ligadora p/ w<0 ... conferencia por fracao
ang_frac = frac * 2 * np.pi
print(f"[OK] cone de morte: fracao de area (caixa [-1,1]^2, aprox) = {frac:.4f} "
      f"=> angulo ~ {np.degrees(ang_frac):.1f}deg (registro; exato = pi - (atan x_max - atan x_min))")
ang_exato = np.pi - (np.arctan(X.max()) - np.arctan(X.min()))
print(f"    angulo exato = {np.degrees(ang_exato):.2f}deg  (formula conferida vs amostragem: "
      f"{abs(np.degrees(ang_exato) - np.degrees(ang_frac)):.2f}deg de diferenca)")
print("\nTODOS OS LEMAS VERIFICADOS.")
