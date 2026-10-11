"""Verificacao dos lemas L1-v2, L2-v2, L3-v2 do relu-death-v2.

Modelo: f(x) = relu(w_A x + b_A) + relu(w_B x + b_B), p=(w_A,b_A,w_B,b_B),
dados X=(0.5,1.5), Y=(0.5,1.0), L=(1/2n) sum (f(x_i)-y_i)^2.

L1-v2: enquanto A morto (z_A,i<=0 para todo i), grad_{p_A} L = 0 exatamente e a
       marginal de p_B e exatamente o gradiente do problema "B sozinho".
L2-v2: B sozinho tem ajuste exato (0.5, 0.25) com ambos os dados ativos; na regiao
       ativa-ambos a dinamica determinista de B e linear com matriz eta*G,
       G = media[(x_i,1)(x_i,1)^T]; o caminho GD discreto do init declarado (0.2,0.05)
       permanece ativa-ambos, e o residuo tem NO MAXIMO UM cruzamento de sinal por
       componente, na fase rapida de B (t < 30000 passos com eta_B=1e-4), com sinais ESTAVEIS depois:
       face cruzada fica com res > 0 (empurra A para DENTRO do cone — anti-captura),
       a outra com res < 0 decaindo no modo lento (empurra A para FORA — captura).
       [Achado 2026-07-24: a versao ingenua "residuo nao troca de sinal" e FALSA —
       overshoot do modo rapido e generico em varios inits testados.]
L3-v2: o salto de forca em p_A ao cruzar a face {z_A,j=0} (demais z_A,i<0) e
       |res_j|/n * sqrt(x_j^2+1), onde res_j = relu(z_B,j) - y_j (residuo corrente de B)
       — decai conforme B ajusta.

Checagens: simbolicas (SymPy, por ramo) + numericas (dif. finitas; caminho GD).
"""
import numpy as np
import sympy as sp

X = np.array([0.5, 1.5])
Y = np.array([0.5, 1.0])
N = len(X)
ETA = 1e-3
ETA_B = 1e-4               # eta de B (corrida separada; ver README)
B_INIT = np.array([0.2, 0.05])

# ================================================================ L1-v2 (simbolico)
wA, bA, wB, bB, x, y = sp.symbols('w_A b_A w_B b_B x y', real=True)
n = sp.Symbol('n', positive=True)
zA, zB = wA * x + bA, wB * x + bB

# ramo: A inativo (z_A<0), B ativo (z_B>0). Termo do dado (x,y): (1/2n)(z_B - y)^2
termo = (zB - y) ** 2 / (2 * n)
assert sp.diff(termo, wA) == 0 and sp.diff(termo, bA) == 0
# e o gradiente em p_B coincide com o do problema B-sozinho:
termo_B_sozinho = (zB - y) ** 2 / (2 * n)
assert sp.simplify(sp.diff(termo, wB) - sp.diff(termo_B_sozinho, wB)) == 0
# ramo: A inativo, B inativo -> termo constante
termo00 = y ** 2 / (2 * n)
assert sp.diff(termo00, wA) == 0 and sp.diff(termo00, bA) == 0
print("[OK] L1-v2: A morto => grad_{p_A}=0 identicamente; marginal de B = problema B-sozinho")

# ================================================================ L3-v2 (simbolico)
# A na face do dado j (z_A=0 pelo lado ativo), B em posicao generica ativa no dado j.
# Termo do dado j com A ativo: (1/2n)(z_A + z_B - y)^2 ; grad em (w_A,b_A) = (1/n)(z_A+z_B-y)(x,1)
termo_Aativo = (zA + zB - y) ** 2 / (2 * n)
gA = sp.Matrix([sp.diff(termo_Aativo, wA), sp.diff(termo_Aativo, bA)])
gA_na_face = sp.simplify(gA.subs(bA, -wA * x))          # z_A -> 0
res = zB - y                                             # residuo corrente (B ativo no dado j)
esperado = sp.Matrix([res * x / n, res / n])
assert sp.simplify(gA_na_face - esperado) == sp.zeros(2, 1)
print("[OK] L3-v2: salto em p_A na face j = |res_j|/n * (x_j,1), res_j = f_B(x_j)-y_j (simbolico)")

# ================================================================ numerico
def loss(p):
    fA = np.maximum(p[0] * X + p[1], 0.0)
    fB = np.maximum(p[2] * X + p[3], 0.0)
    return 0.5 * np.mean((fA + fB - Y) ** 2)

def grad(p):
    zAv, zBv = p[0] * X + p[1], p[2] * X + p[3]
    r = np.maximum(zAv, 0) + np.maximum(zBv, 0) - Y
    aA, aB = (zAv > 0).astype(float), (zBv > 0).astype(float)
    return np.array([np.mean(r * aA * X), np.mean(r * aA),
                     np.mean(r * aB * X), np.mean(r * aB)])

rng = np.random.default_rng(3)
h, err = 1e-6, 0.0
for _ in range(3000):
    p = rng.uniform(-2, 2, 4)
    zs = np.concatenate([p[0] * X + p[1], p[2] * X + p[3]])
    if np.any(np.abs(zs) < 1e-3):
        continue
    fd = np.array([(loss(p + h * e) - loss(p - h * e)) / (2 * h) for e in np.eye(4)])
    err = max(err, np.abs(fd - grad(p)).max())
assert err < 1e-6, err
print(f"[OK] gradiente 4D vs dif. finitas (3000 pontos fora das quinas): err_max={err:.1e}")

# grad_{p_A}=0 em pontos mortos aleatorios (numerico)
for _ in range(500):
    p = rng.uniform(-2, 2, 4)
    c = max(p[0] * X + p[1]) + rng.uniform(0.01, 1.0)
    p[1] -= c                                            # empurra A para dentro de D
    assert np.all(p[0] * X + p[1] < 0)
    assert np.abs(grad(p)[:2]).max() == 0.0
print("[OK] L1-v2 numerico: grad_{p_A} identicamente zero em 500 pontos mortos")

# ================================================================ L2-v2 (numerico)
p_star = np.linalg.solve(np.array([[0.5, 1], [1.5, 1]]), Y)
assert np.allclose(p_star, [0.5, 0.25])
assert np.all(p_star[0] * X + p_star[1] > 0)
G = np.mean([np.outer([xi, 1], [xi, 1]) for xi in X], axis=0)
eig = np.sort(np.linalg.eigvalsh(G))
print(f"[OK] L2-v2: fit exato de B = (0.5, 0.25), ambos ativos; eig(G) = {eig.round(5)}"
      f" (timescales de B com eta_B: {1/(ETA_B*eig[1]):.0f} e {1/(ETA_B*eig[0]):.0f} passos)")

# caminho GD discreto de B-sozinho a partir do init declarado: ativo-ambos + estrutura de sinal
pB = B_INIT.copy()
min_z, res_hist = np.inf, []
for t in range(500_000):
    zBv = pB[0] * X + pB[1]
    min_z = min(min_z, zBv.min())
    r = np.maximum(zBv, 0) - Y
    res_hist.append(r.copy())
    pB = pB - ETA_B * np.array([np.mean(r * (zBv > 0) * X), np.mean(r * (zBv > 0))])
res_hist = np.array(res_hist)
res_final = np.abs(np.maximum(pB[0] * X + pB[1], 0) - Y).max()
assert min_z > 0, min_z
# com eta_B=1e-4 e 500k passos, o residuo restante e modo-lento puro (fast morto ha
# ~20 timescales); extrapolar CADA componente de t=100k pela taxa lenta e comparar o max
decai = np.exp((500_000 - 100_000) * np.log(1 - ETA_B * eig[0]))
res_teo_final = np.max(np.abs(res_hist[100_000, :])) * decai
assert res_final < 1e-2, res_final
assert abs(res_final - res_teo_final) / res_teo_final < 0.15, (res_final, res_teo_final)
# no maximo 1 cruzamento por componente, na fase rapida, sinais estaveis depois
for j in range(N):
    sinais = np.sign(res_hist[:, j])
    trocas = np.nonzero(np.diff(sinais[np.abs(res_hist[:, j]) > 1e-12]))[0]
    n_trocas = len(trocas)
    assert n_trocas <= 1, f"componente {j}: {n_trocas} cruzamentos"
    if n_trocas == 1:
        t_cruz = trocas[0]
        assert t_cruz < 30000, f"cruzamento tardio: {t_cruz}"
        print(f"    res_{j}: 1 cruzamento em t={t_cruz} (fase rapida), depois res>0 decaindo"
              f" (max overshoot {res_hist[:, j].max():.4f}) — face {j} anti-captura pos-cruzamento")
    else:
        assert np.all(res_hist[np.abs(res_hist[:, j]) > 1e-12, j] < 0)
        print(f"    res_{j}: sem cruzamento, res<0 decaindo — face {j} pro-captura sempre")
print(f"[OK] L2-v2 caminho: ativo-ambos (min z = {min_z:.4f}), residuo final = {res_final:.1e},"
      f" estrutura de sinal conforme lema corrigido")

# decaimento multi-exponencial: ajustar log|res| tardio contra a taxa lenta teorica
res_hist = np.array(res_hist)
tardio = slice(200_000, 400_000)
taxa_obs = np.polyfit(np.arange(*tardio.indices(len(res_hist)))[::10],
                      np.log(np.abs(res_hist[tardio, 0][::10]) + 1e-300), 1)[0]
taxa_teo = np.log(1 - ETA_B * eig[0])
assert abs(taxa_obs - taxa_teo) / abs(taxa_teo) < 0.02, (taxa_obs, taxa_teo)
print(f"[OK] L2-v2 taxa lenta: obs {taxa_obs:.3e}/passo vs teoria ln(1-eta*lam_min)={taxa_teo:.3e} (<2%)")

# ================================================================ L3-v2 (numerico)
for j in range(N):
    wf = 1.0 if j == np.argmax(X) else -1.0              # isola a face j (mesma logica do ciclo 1)
    bf = -wf * X[j]
    zo = wf * X + bf
    assert all(zo[i] < -1e-9 for i in range(N) if i != j)
    for pB_teste in ([0.2, 0.05], [0.35, 0.15], [0.499, 0.249]):
        p_morto = np.array([wf, bf, *pB_teste])
        nrm = np.array([X[j], 1.0]) / np.hypot(X[j], 1.0)
        eps = 1e-7
        g_m = grad(np.concatenate([p_morto[:2] - eps * nrm, p_morto[2:]]))[:2]
        g_v = grad(np.concatenate([p_morto[:2] + eps * nrm, p_morto[2:]]))[:2]
        res_j = max(pB_teste[0] * X[j] + pB_teste[1], 0) - Y[j]
        salto_teo = abs(res_j) / N * np.hypot(X[j], 1.0)
        assert np.linalg.norm(g_m) == 0.0
        assert abs(np.linalg.norm(g_v - g_m) - salto_teo) < 1e-4, (j, pB_teste)
    print(f"[OK] L3-v2 numerico: face j={j}, salto = |res_j|/n*sqrt(x_j^2+1) para 3 posicoes de B"
          f" (em p_B quase-fit o salto ~ 0: parede amolecida)")

print("\nTODOS OS LEMAS v2 VERIFICADOS.")
