"""Independent, small numerical checks of entry initialization and release.

Uses DOP853 with interpolated events, separately from the historical Euler
campaign. Checks three selected archived instances, not the full theorem.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
BETAS = (10_000., 30_000., 100_000., 300_000.)
SEEDS = (20660796, 21061010, 21860945)


def symbolic_identity():
    z, c, r, b, eta = sp.symbols("z c R beta eta", real=True)
    u = sp.exp(-b*z)
    g = sp.log(1+sp.exp(b*z))/b
    dz = c*(r-g)/(1+u) + eta
    derivative = sp.simplify(sp.diff(-b*z+u, z)*dz)
    target = -c*b*r + c*sp.log(1+sp.exp(b*z)) - b*(1+u)*eta
    assert sp.simplify(derivative-target) == 0
    # A zero unit block need not reduce the rank of the whole data Jacobian.
    live = sp.Matrix([[sp.Rational(1, 2), 1], [sp.Rational(3, 2), 1]])
    incoming = sp.Matrix([[sp.Rational(1, 2), 1], [0, 0]])
    assert incoming.row_join(live).rank() == sp.zeros(2, 2).row_join(live).rank() == 2
    assert sp.simplify(1/(1+sp.exp(-b*0))) == sp.Rational(1, 2)
    print("PASS: exact scalar quadrature, including the other-margin remainder")
    print("PASS: Jacobian-rank counterexample and softplus derivative at the kink")


def coupled(record, rtol=2e-12, atol=2e-14):
    m = record["m"]
    x, y, initial = (np.array(record[k]) for k in ("X", "Y", "P0"))
    readout = np.r_[1., np.full(m, 1/np.sqrt(m))]
    c = (x[0]**2+1)/len(x)

    def field(params, beta=None):
        z = params[:, :1]*x + params[:, 1:]
        act = np.maximum(z, 0) if beta is None else np.logaddexp(0, beta*z)/beta
        slope = (z > 0) if beta is None else expit(beta*z)
        residual = readout@act-y
        return -readout[:, None]*np.stack([(slope*(residual*x)).mean(1),
                                         (slope*residual).mean(1)], axis=1)

    def down(t, q):
        return q[0]*x[0]+q[1]

    def up(t, q):
        return down(t, q)

    down.direction, down.terminal = -1, True
    up.direction, up.terminal = 1, True
    settings = dict(method="DOP853", rtol=rtol, atol=atol, max_step=0.001)
    pre = solve_ivp(lambda t, q: field(q.reshape(-1, 2)).ravel(),
                    (0, 1.8), initial.ravel(), events=down, **settings)
    assert pre.success and len(pre.t_events[0]) == 1
    hit = float(pre.t[-1])
    state = pre.y[:, -1].reshape(-1, 2)

    def hybrid(t, q):
        params = q[:-1].reshape(-1, 2)
        drift = field(params)
        drift[0] = 0
        residual = y[0]-readout[1:]@np.maximum(params[1:, 0]*x[0]+params[1:, 1], 0)
        return np.r_[drift.ravel(), residual]

    def area(t, q):
        return q[-1]

    area.direction, area.terminal = 1, True
    reference = solve_ivp(hybrid, (hit, 1.8), np.r_[state.ravel(), 0.], events=area, **settings)
    assert reference.success and len(reference.t_events[0]) == 1
    exit_time = float(reference.t[-1])
    residual_exit = float(hybrid(exit_time, reference.y[:, -1])[-1])
    predicted = -1/(c*residual_exit)
    down.terminal = False
    measurements = []
    for beta in BETAS:
        sol = solve_ivp(lambda t, q: field(q.reshape(-1, 2), beta).ravel(),
                        (0, min(1.8, exit_time+0.1)), initial.ravel(), events=(down, up), **settings)
        assert sol.success and len(sol.t_events[0]) == len(sol.t_events[1]) == 1
        coefficient = float(beta*(sol.t_events[1][0]-exit_time))
        measurements.append(dict(beta=beta, coefficient=coefficient,
                                 relative_error=abs(coefficient/predicted-1),
                                 entry_delay_coefficient=float(beta*(sol.t_events[0][0]-hit))))
    assert measurements[-1]["relative_error"] < 0.015
    assert measurements[-1]["relative_error"] < measurements[0]["relative_error"]
    print(f"PASS: coupled m={m}, seed={record['seed']}, predicted={predicted:.6f}, "
          f"measured at beta=300000: {measurements[-1]['coefficient']:.6f}")
    return dict(m=m, seed=record["seed"], t_hit=hit, t_exit=exit_time,
                predicted=predicted, measurements=measurements)


def from_face():
    measurements = []
    for beta in (200., 1000., 5000., 20000.):
        def field(t, z):
            return (-1+1.2*t-np.logaddexp(0, beta*z[0])/beta)*expit(beta*z[0])

        def up(t, z):
            return z[0]

        up.direction, up.terminal = 1, True
        sol = solve_ivp(field, (0, 2), [0.], events=up, method="DOP853",
                        rtol=2e-12, atol=2e-14, max_step=.002)
        assert sol.success and len(sol.t_events[0]) == 1
        delta = float(sol.t_events[0][0]-2/1.2)
        measurements.append(dict(beta=beta, beta_delta=beta*delta,
                                 beta2_delta_over_log_beta=beta*beta*delta/np.log(beta)))
    assert all(r["beta_delta"] > 0 for r in measurements)
    assert measurements[-1]["beta_delta"] < 0.002
    print("PASS: from-face prescribed forcing: positive subleading displacement; no entry-delay term")
    return measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON report; does not modify archived data")
    args = parser.parse_args()
    symbolic_identity()
    records = [json.loads(x) for x in (ROOT / "05-experimentos/t2-n-neuronios/forcing_m.jsonl").read_text().splitlines()]
    selected = [next(r for r in records if r["seed"] == seed) for seed in SEEDS]
    result = dict(method="DOP853 with event interpolation", rtol=2e-12, atol=2e-14,
                  max_step=.001, coupled=[coupled(r) for r in selected], from_face=from_face())
    refined = coupled(selected[0], rtol=5e-13, atol=5e-15)
    difference = abs(refined["measurements"][-1]["coefficient"]-
                     result["coupled"][0]["measurements"][-1]["coefficient"])
    assert difference < 0.002
    result["tolerance_control"] = dict(rtol=5e-13, atol=5e-15, coefficient_difference=difference)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+"\n")
    print("Selected checks passed. These finite computations do not certify the proofs.")


if __name__ == "__main__":
    main()
