"""Shared, sourced inputs for the CHIME-DSA chance-coincidence experiment.

Every astrophysical number here is either cited or explicitly labelled an
assumption. The headline of the experiment is the ORDER OF MAGNITUDE of the
false-alarm probability and the AGREEMENT between the analytic and Monte-Carlo
estimators -- not a precise P. Windows are swept in run.py to show robustness.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).parent

# --- CHIME FRB rate ---------------------------------------------------------
# CHIME/FRB Catalogue 1 (CHIME/FRB Collaboration, Amiri et al. 2021, ApJS 257, 59)
# derived an all-sky rate of ~525 FRBs sky^-1 day^-1 above 5 Jy ms at 600 MHz.
# Use a deliberately CONSERVATIVE (high) value to MAXIMISE the chance probability.
R_SKY_PER_DAY_CENTRAL = 525.0  # cited central value
R_SKY_PER_DAY_CONSERVATIVE = 1000.0  # rounded-up upper bound (assumption, conservative)
FULL_SKY_SR = 4.0 * math.pi  # 4pi sr ~ 41253 deg^2
SECONDS_PER_DAY = 86400.0


def r_sr_s(rate_per_day: float) -> float:
    """FRB rate per steradian per second from an all-sky per-day rate."""
    return rate_per_day / FULL_SKY_SR / SECONDS_PER_DAY


# --- CHIME DM distribution (for the DM-match fraction f_DM) ------------------
# Catalogue 1 extragalactic-DM distribution is broad, peaking a few hundred
# pc cm^-3. Modelled here as log-normal(median=500, sigma_ln=0.7) -- an
# ASSUMPTION (the catalogue file is not on h17); used identically by both
# estimators, so it cancels in the A-vs-B agreement check.
DM_MEDIAN = 500.0
DM_SIGMA_LN = 0.7
DM_RANGE = (50.0, 3500.0)  # support used for the uniform-model sensitivity


def dm_lognormal_pdf(dm: float) -> float:
    mu = math.log(DM_MEDIAN)
    z = (math.log(dm) - mu) / DM_SIGMA_LN
    return math.exp(-0.5 * z * z) / (dm * DM_SIGMA_LN * math.sqrt(2.0 * math.pi))


def f_dm(dm: float, half_width: float) -> float:
    """P(random CHIME DM within +/- half_width of `dm`), local-density approx."""
    return min(1.0, dm_lognormal_pdf(dm) * 2.0 * half_width)


# --- Coincidence windows (baseline; swept in run.py) ------------------------
# Positional tolerance: DSA-110 localises to ~arcsec; CHIME baseband to ~arcmin.
# Baseline uses a deliberately GENEROUS 0.5 deg radius disk (=> larger P).
DEG2_PER_SR = (180.0 / math.pi) ** 2
OMEGA_WIN_BASELINE_DEG2 = math.pi * 0.5**2  # 0.5 deg radius disk ~ 0.785 deg^2
DT_BASELINE_S = 1.0  # +/- 1 s temporal window (conservative)
DDM_BASELINE = 5.0  # +/- 5 pc cm^-3 DM match (generous)


def omega_win_sr(deg2: float) -> float:
    return deg2 / DEG2_PER_SR


def load_bursts() -> list[dict]:
    return json.load(open(HERE / "bursts.json"))
