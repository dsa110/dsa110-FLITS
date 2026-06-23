"""Approach A: closed-form Poisson false-alarm probability.

mu_i = R_sr_s * Omega_win * (2*dt) * f_DM(DM_i, ddm);  P_i = 1 - exp(-mu_i).
Expected number of chance associations across the sample = sum_i mu_i.
"""

from __future__ import annotations

import math

import inputs as I


def mu_analytic(dm, *, rate_per_day, omega_win_deg2, dt_s, ddm):
    rate = I.r_sr_s(rate_per_day)
    omega = I.omega_win_sr(omega_win_deg2)
    return rate * omega * (2.0 * dt_s) * I.f_dm(dm, ddm)


def run(bursts, *, rate_per_day, omega_win_deg2, dt_s, ddm):
    out = []
    for b in bursts:
        mu = mu_analytic(
            b["dm"], rate_per_day=rate_per_day, omega_win_deg2=omega_win_deg2, dt_s=dt_s, ddm=ddm
        )
        out.append({"name": b["name"], "dm": b["dm"], "mu": mu, "P": 1.0 - math.exp(-mu)})
    return out
