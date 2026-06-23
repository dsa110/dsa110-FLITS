#!/usr/bin/env python
"""DSA-110 primary-beam gain from the measured Jones E-field cube (DSA110_beam_1.h5).

Stokes-I power beam P = |E_theta|^2 + |E_phi|^2 summed over the X and Y feeds,
normalized to boresight (theta=0) = 1 at each frequency. This is the relative
sensitivity at a burst's angular offset from the pointing centre -- the beam
attenuation factor G in the radiometer flux calibration

    F = (S/N) * SEFD / ( sqrt(n_pol * dnu * dt) * G(theta, phi, nu) ).

It replaces the analytic Airy-disk fallback in dsa110-continuum
(calibration/beam_model.py) with the measured/simulated dish pattern.

Cube layout (N_freq, N_theta, N_phi):
  freq: 1.2-1.6 GHz, 41 pts. NOTE the HDF5 dataset is named `freq_Hz` and carries a
        `freq_units="MHz"` attr, but the stored values are 1.2-1.6 -- unambiguously
        GHz (idx 20 = 1.40 GHz, matching the dish's L-band). Both labels are wrong;
        the values are GHz. Covers the DSA fit band (1.311-1.499 GHz).
  theta: 0-180 deg (zenith angle), 1801 pts @ 0.1 deg; phi: 0-360 deg (azimuth), 73 pts.

The absolute scale still needs the boresight SEFD, which is MEASURED per-day /
per-calibrator (not a constant) by the dsa110-rt SEFD dashboard
(github.com/dsa110/dsa110-rt, served on lxd110h23:5777); use the value nearest a
burst's epoch. See analysis/burst_energies/CALIBRATION_REVIEW.md.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def _default_beam() -> Path:
    """First existing of $FLITS_DSA_BEAM, repo data/DSA110_beam_1.h5, ~/Downloads (back-compat).

    The cube is ~345 MB and gitignored; the canonical stable location is the repo's data/ dir.
    Returns the canonical path even when absent so callers get a clear, repo-relative error.
    """
    repo = Path(__file__).resolve().parents[1]
    canonical = repo / "data" / "DSA110_beam_1.h5"
    for p in (
        os.environ.get("FLITS_DSA_BEAM"),
        canonical,
        Path.home() / "Downloads" / "DSA110_beam_1.h5",
    ):
        if p and Path(p).exists():
            return Path(p)
    return canonical


DEFAULT_BEAM = _default_beam()


def load_power_beam(path: str | Path = DEFAULT_BEAM):
    """(freq_ghz, theta_deg, phi_deg, P) with P normalized to boresight=1 per frequency."""
    import h5py

    with h5py.File(path, "r") as f:
        freq_ghz = f["freq_Hz"][:].astype(float)  # mislabeled units; values are GHz
        theta = f["theta_pts"][:].astype(float)  # deg
        phi = f["phi_pts"][:].astype(float)  # deg
        P = np.zeros((freq_ghz.size, theta.size, phi.size))
        for grp in ("X_pol_Efields", "Y-pol_Efields"):
            P += np.abs(f[f"{grp}/etheta"][:]) ** 2 + np.abs(f[f"{grp}/ephi"][:]) ** 2
    P /= P[:, 0, :].mean(axis=1)[:, None, None]  # boresight (theta=0) -> 1 at each freq
    return freq_ghz, theta, phi, P


def beam_gain(
    theta_deg: float, phi_deg: float, freq_ghz: float, path: str | Path = DEFAULT_BEAM
) -> float:
    """Normalized power-beam gain (boresight=1) at an offset, via trilinear interpolation."""
    from scipy.interpolate import RegularGridInterpolator

    fz, th, ph, P = load_power_beam(path)
    g = RegularGridInterpolator((fz, th, ph), P, bounds_error=False, fill_value=None)
    return float(g([[freq_ghz, theta_deg, np.mod(phi_deg, 360.0)]])[0])


def _check(path: str | Path = DEFAULT_BEAM) -> None:
    if not Path(path).exists():
        print(f"self-check SKIP: beam cube not found at {path}")
        return
    fz, th, ph, P = load_power_beam(path)
    assert abs(P[:, 0, :].mean() - 1.0) < 1e-9, "boresight not normalized to 1"
    assert 1.2 <= fz.min() and fz.max() <= 1.6, ("freq axis not GHz?", fz.min(), fz.max())
    g0 = beam_gain(0.0, 0.0, 1.4, path)
    assert abs(g0 - 1.0) < 1e-6, g0  # boresight = 1
    g_hp = beam_gain(1.8, 0.0, 1.4, path)
    assert 0.2 < g_hp < 0.8, ("half-power radius off", g_hp)  # ~half power near FWHM/2
    g_far = beam_gain(5.0, 0.0, 1.4, path)
    assert g_far < g_hp, ("gain not falling off-axis", g_far, g_hp)
    print(
        f"self-check OK: boresight=1, gain(1.8deg,1.4GHz)={g_hp:.3f} (~half power), "
        f"gain(5deg)={g_far:.3f}; freq {fz.min():.2f}-{fz.max():.2f} GHz"
    )


if __name__ == "__main__":
    _check()
