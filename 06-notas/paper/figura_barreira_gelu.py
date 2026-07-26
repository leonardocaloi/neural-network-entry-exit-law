"""Figure 7 (paper j-space): the analytic GELU barrier and the absence of a stratum.

Everything here is exact/analytic: no experimental data is used. The GELU is the
exact (erf) form g(z)=z*Phi(z), so g'(z)=Phi(z)+z*phi(z); the sign change z* is
found with brentq and checked against -0.7518.

Left panel: g'(z) on z in [-3, 0.5], zero at z*, the self-trapped region z<z*
shaded (own gradient pushes the pre-activation further down), and the shallow
band (z*, 0) where the own gradient is still restorative. The two injection
depths of the confirmatory experiment (z=-0.3 shallow, z=-2.0 deep) are marked.

Right panel: the outputs g(z) for GELU, softplus(beta=3), softplus(beta=10) and
ReLU. Reading: ReLU is exactly 0 for z<0 (an exact dead stratum), softplus is
exponentially small but strictly positive (no stratum, only a near-stratum),
GELU is strictly negative and bounded away from 0 (GELU(-0.3) = -0.115): there
is no dead stratum at all for GELU.

Suggested caption
-----------------
Figure 7. The analytic barrier and the non-stratum, for the activations of the
confirmatory grid. Left: the GELU derivative g'(z)=Phi(z)+z phi(z) changes sign
at z* = -0.75179 (brentq). For z<z* (shaded) the unit's own gradient is adverse
-- a self-trapping region -- while on the shallow band z* < z < 0 the own
gradient still points back towards activity. The experiment injects at z=-0.3
(inside the shallow band) and at z=-2.0 (deep inside the self-trapped region),
which is exactly the analytic barrier the deep-injection cells of Figure 6 hit.
Right: the activations themselves. ReLU vanishes identically for z<0, an exact
dead stratum; softplus is exponentially small but strictly positive, so the
stratum is only approached, never reached; GELU is strictly negative, with
GELU(-0.3) = -0.1151, so for GELU there is no dead stratum in the first place
and the deep-injection suppression must come from the sign of g', not from a
loss of rank.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except Exception:
    pass


def gelu(z):
    return z * norm.cdf(z)


def dgelu(z):
    return norm.cdf(z) + z * norm.pdf(z)


def softplus(z, beta):
    return np.log1p(np.exp(-np.abs(beta * z))) / beta + np.maximum(z, 0.0)


def relu(z):
    return np.maximum(z, 0.0)


# --- exact sign change of the GELU derivative
zstar = brentq(dgelu, -3.0, -0.1, xtol=1e-14, rtol=8.9e-16)
assert abs(zstar - (-0.7518)) < 5e-5, f"z*={zstar!r} does not match -0.7518"

Z = np.linspace(-3.0, 0.5, 1400)
DEPTHS = (-0.3, -2.0)
C_GELU, C_B3, C_B10, C_RELU = "#26547C", "#06A77D", "#F4A259", "#EF476F"

fig, (axL, axR) = plt.subplots(1, 2, figsize=(5.5, 2.4))

# ---------------- left: the barrier
axL.axhline(0.0, color="0.55", lw=0.6)
axL.axvspan(-3.0, zstar, color="#EF476F", alpha=0.13, lw=0, zorder=0)
axL.axvspan(zstar, 0.0, color="#06A77D", alpha=0.13, lw=0, zorder=0)
axL.plot(Z, dgelu(Z), color=C_GELU, lw=1.4, zorder=3,
         label=r"$g'(z)=\Phi(z)+z\varphi(z)$")
axL.plot([zstar], [0.0], "o", ms=4, mfc="white", mec=C_GELU, mew=1.0, zorder=4)
axL.annotate(rf"$z^{{*}}={zstar:.4f}$", xy=(zstar, 0.0), xytext=(-2.55, 0.30),
             fontsize=6.6,
             arrowprops=dict(arrowstyle="->", lw=0.6, color="black"))
axL.text(-1.75, 0.62, "self-trapped:\nadverse own-gradient", fontsize=6.4,
         ha="center", va="center", color="#8c2540")
axL.text(-0.20, 0.62, "shallow\nband", fontsize=6.4, ha="center", va="center",
         color="#0a6a52")
for zd in DEPTHS:
    axL.axvline(zd, color="0.25", ls=":", lw=0.9, zorder=2)
    axL.text(zd + 0.04, -0.205, f"$z={zd}$", fontsize=6.2, ha="left",
             va="bottom", color="0.20")
axL.set_xlim(-3.0, 0.5)
axL.set_ylim(-0.22, 1.0)
axL.set_xlabel("pre-activation $z$", fontsize=7.5)
axL.set_ylabel(r"GELU derivative $g'(z)$", fontsize=7.5)
axL.set_title("(a) analytic barrier", fontsize=8)
axL.tick_params(labelsize=7)
axL.legend(fontsize=6.2, loc="upper left", framealpha=0.9)

# ---------------- right: the outputs
axR.axhline(0.0, color="0.55", lw=0.6)
axR.plot(Z, relu(Z), color=C_RELU, lw=1.6, label="ReLU (exactly $0$)")
axR.plot(Z, softplus(Z, 10.0), color=C_B10, lw=1.2, ls="--",
         label=r"softplus $\beta=10$")
axR.plot(Z, softplus(Z, 3.0), color=C_B3, lw=1.2, ls="-.",
         label=r"softplus $\beta=3$")
axR.plot(Z, gelu(Z), color=C_GELU, lw=1.4, label="GELU")
g03 = gelu(-0.3)
axR.plot([-0.3], [g03], "o", ms=3.6, mfc="white", mec=C_GELU, mew=1.0, zorder=4)
axR.annotate(rf"GELU$(-0.3)={g03:.3f}$", xy=(-0.3, g03), xytext=(-1.35, -0.31),
             fontsize=6.6, ha="center",
             arrowprops=dict(arrowstyle="->", lw=0.6, color="black"))
for zd in DEPTHS:
    axR.axvline(zd, color="0.25", ls=":", lw=0.9, zorder=1)
axR.set_xlim(-3.0, 0.5)
axR.set_ylim(-0.35, 0.62)
axR.set_xlabel("pre-activation $z$", fontsize=7.5)
axR.set_ylabel("activation $g(z)$", fontsize=7.5)
axR.set_title("(b) exact stratum vs. no stratum", fontsize=8)
axR.tick_params(labelsize=7)
axR.legend(fontsize=6.0, loc="upper left", framealpha=0.9)

fig.tight_layout(pad=0.4, w_pad=1.2)
for ext in ("pdf", "png"):
    fig.savefig(f"fig7-barreira-gelu.{ext}", dpi=400, bbox_inches="tight")

print("fig7-barreira-gelu: key numbers")
print(f"  z* (brentq)      = {zstar:.10f}   (target -0.7518, ok)")
print(f"  g'(-0.3)         = {dgelu(-0.3):+.6f}  (restorative, > 0)")
print(f"  g'(-2.0)         = {dgelu(-2.0):+.6f}  (adverse, < 0)")
print(f"  GELU(-0.3)       = {gelu(-0.3):+.6f}")
print(f"  GELU(-2.0)       = {gelu(-2.0):+.6f}")
print(f"  ReLU(-0.3)       = {relu(-0.3):+.6f}  (exact zero)")
print(f"  softplus_b3(-0.3)  = {softplus(np.array(-0.3), 3.0):.6e}")
print(f"  softplus_b10(-0.3) = {softplus(np.array(-0.3), 10.0):.6e}")
print(f"  softplus_b3(-2.0)  = {softplus(np.array(-2.0), 3.0):.6e}")
print(f"  softplus_b10(-2.0) = {softplus(np.array(-2.0), 10.0):.6e}")
