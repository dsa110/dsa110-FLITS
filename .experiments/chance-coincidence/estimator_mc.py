"""Approach B: direct Monte-Carlo of the background CHIME point process.

For each DSA burst, simulate `realisations` independent backgrounds in the
positional+temporal window (Poisson-distributed count with mean
lambda_box = R_sr_s * Omega_win * 2*dt), draw a DM for each background event
from the same CHIME DM model, and flag a chance association if >=1 background
event also lands within +/- ddm of the burst DM. The fraction of realisations
with a hit is the MC false-alarm probability P_i (with binomial error).

This is a sampling-based, implementation-independent cross-check of the analytic
mu: it exercises the Poisson draw, the DM draw, and the rare-event regime.
Seeded for reproducibility; DM model is shared with Approach A by design.
"""

from __future__ import annotations

import math

import inputs as I
import numpy as np


def _draw_dm(rng, n):
    return np.exp(rng.normal(math.log(I.DM_MEDIAN), I.DM_SIGMA_LN, size=n))


def run(bursts, *, rate_per_day, omega_win_deg2, dt_s, ddm, realisations=2_000_000, seed=20260623):
    rate = I.r_sr_s(rate_per_day)
    omega = I.omega_win_sr(omega_win_deg2)
    lambda_box = rate * omega * (2.0 * dt_s)  # mean # background events in pos+time window
    rng = np.random.default_rng(seed)
    out = []
    for b in bursts:
        counts = rng.poisson(lambda_box, size=realisations)  # events in the box per realisation
        total = int(counts.sum())
        hit = np.zeros(realisations, dtype=bool)
        if total > 0:
            dms = _draw_dm(rng, total)
            within = np.abs(dms - b["dm"]) <= ddm
            # scatter per-event hits back to their realisation via repeat-index
            real_idx = np.repeat(np.arange(realisations), counts)
            hit_real = real_idx[within]
            hit[hit_real] = True
        k = int(hit.sum())
        p = k / realisations
        err = math.sqrt(max(k, 1)) / realisations  # Poisson count error on P
        out.append(
            {
                "name": b["name"],
                "dm": b["dm"],
                "P": p,
                "P_err": err,
                "hits": k,
                "realisations": realisations,
                "lambda_box": lambda_box,
            }
        )
    return out
