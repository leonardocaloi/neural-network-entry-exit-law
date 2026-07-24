"""T2' Passo 0: verificacao simbolica das identidades I1-I4 (ver nota ao lado)."""
import sympy as sp

beta, z, t, c = sp.symbols('beta z t c', positive=True)
R = sp.Function('R')(t)
w = sp.exp(beta * z)
u = sp.exp(-beta * z)
sig = 1 / (1 + sp.exp(-beta * z))

# I1: margem. z' = -c*(softplus(z) - R)*sigma(beta z); separar o termo R e a correcao
softplus = sp.log(1 + sp.exp(beta * z)) / beta
zdot_full = -c * (softplus - R) * sig
zdot_lead = c * R * sig
corr = sp.simplify(zdot_full - zdot_lead)                 # = -c*softplus*sig
assert sp.simplify(corr + c * softplus * sig) == 0
# |softplus| = log(1+w)/beta — cota da nota (checagem estrutural)
assert sp.simplify(softplus - sp.log(1 + w) / beta) == 0
print("[OK] I1: z' = c*R*sigma(beta z)*(1+E_beta), E_beta = -softplus/R, |softplus| = log(1+w)/beta")

# I2: gauge w, tempo rapido tau = beta*t: dw/dtau = (1/beta)*dw/dt = w*z'(lead)
dw_dtau = sp.simplify(sp.diff(w, z) * zdot_lead / beta)
alvo_w = c * R * w**2 / (1 + w)
assert sp.simplify(dw_dtau - alvo_w) == 0
print("[OK] I2: dw/dtau = c*R*w^2/(1+w)  (forma quadratica DM-S, f=1, g=cR/(1+w))")

# I3: gauge u
du_dtau = sp.simplify(sp.diff(u, z) * zdot_lead / beta)
alvo_u = -c * R * u / (1 + u)
assert sp.simplify(du_dtau - alvo_u) == 0
print("[OK] I3: du/dtau = -c*R*u/(1+u)  (forma do ataque de 2026-07-24)")

# I4: quadratura exata: dPhi(u)/dtau = -c*R com Phi = log u + u
Phi = sp.log(u) + u
dPhi = sp.simplify(sp.diff(Phi, z) * zdot_lead / beta)
assert sp.simplify(dPhi + c * R) == 0
print("[OK] I4: d(log u + u)/dtau = -c*R  => release quando a area c*int R dt retorna a zero")

# consistencia sigma/w: sigma(beta z) = w/(1+w) = 1/(1+u)
assert sp.simplify(sig - w / (1 + w)) == 0
assert sp.simplify(sig - 1 / (1 + u)) == 0
print("[OK] sigma(beta z) = w/(1+w) = 1/(1+u)")

print("\nPASSO 0 VERIFICADO — identidades I1-I4 prontas para os passos 1-5.")
