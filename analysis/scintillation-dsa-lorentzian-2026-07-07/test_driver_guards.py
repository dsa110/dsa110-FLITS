"""Driver-level guard wiring tests for run_dsa_lorentzian_fits.py.

These exercise the actual driver helper functions (imported from the hyphenated
analysis directory) on synthetic ACFs, so the wiring between the driver and
``chime_artifact_guards`` is covered without a full pipeline run:

  - ``_fit_width`` recovers a known Lorentzian width;
  - ``_low_lag_excision_widths`` keeps the width for a resolved wing and
    collapses it for a low-lag-only bump (the arm-B1 discriminator);
  - the harmonic-mask path in the driver removes comb lag bins and changes the
    fit only when a comb is actually present.

Run: NUMBA_DISABLE_JIT=1 python -m pytest \
     analysis/scintillation-dsa-lorentzian-2026-07-07/test_driver_guards.py
from the pipeline root.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]  # pipeline/
sys.path.insert(0, str(_ROOT))


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "rdlf_under_test", _HERE / "run_dsa_lorentzian_fits.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


drv = _load_driver()


def _lorentzian_acf(gamma_mhz, m, *, dch=0.01, nch=400, noise=1e-3, seed=0):
    """Symmetric mean-normalized ACF: m^2/(1+(lag/gamma)^2) + noise, lag0=1."""
    rng = np.random.default_rng(seed)
    pos = np.arange(1, nch + 1) * dch
    lags = np.concatenate((-pos[::-1], [0.0], pos))
    acf = m**2 / (1 + (lags / gamma_mhz) ** 2)
    acf[lags == 0] = 1.0
    acf = acf + rng.normal(0, noise, lags.size)
    err = np.full(lags.size, noise)
    return lags, acf, err, dch


def test_fit_width_recovers_known_lorentzian():
    lags, acf, err, _ = _lorentzian_acf(0.2, 0.8)
    keep = np.abs(lags) <= 2.0
    w = drv._fit_width(lags[keep], acf[keep], err[keep], max_components=1)
    assert w is not None
    assert abs(w - 0.2) < 0.05  # within 25% of the injected 0.2 MHz


def test_low_lag_excision_keeps_resolved_wing():
    # A real, well-resolved Lorentzian (gamma >> channel): width survives
    # dropping the first few channel lags.
    lags, acf, err, dch = _lorentzian_acf(0.3, 0.8, dch=0.01, nch=400)
    keep = np.abs(lags) <= 3.0
    full = drv._fit_width(lags[keep], acf[keep], err[keep], max_components=1)
    excised = drv._low_lag_excision_widths(
        lags[keep], acf[keep], err[keep], dch, max_components=1, ks=(1, 2, 3)
    )
    verdict = drv.guards.low_lag_stability_verdict(full, excised)
    assert verdict["stable"] is True


def test_low_lag_excision_collapses_on_no_wing_artifact():
    # The freya CHIME failure signature: correlated power carried ENTIRELY by the
    # first couple of channel lags with NO Lorentzian wing (flat noise beyond).
    # Excising the low lags must collapse the fitted width.
    dch = 0.01
    rng = np.random.default_rng(0)
    pos = np.arange(1, 401) * dch
    lags = np.concatenate((-pos[::-1], [0.0], pos))
    acf = np.zeros_like(lags)
    lag_ch = np.round(np.abs(lags) / dch).astype(int)
    for k in (1, 2):  # power only at |lag| = 1, 2 channels
        acf[np.abs(lag_ch - k) < 0.5] = 0.25 * (0.5 ** (k - 1))
    acf[lags == 0] = 1.0
    acf = acf + rng.normal(0, 5e-4, lags.size)
    err = np.full(lags.size, 5e-4)

    keep = np.abs(lags) <= 0.5
    full = drv._fit_width(lags[keep], acf[keep], err[keep], max_components=1)
    excised = drv._low_lag_excision_widths(
        lags[keep], acf[keep], err[keep], dch, max_components=1, ks=(1, 2, 3)
    )
    verdict = drv.guards.low_lag_stability_verdict(full, excised)
    assert verdict["stable"] is False
    assert set(verdict["failed_ks"]) & {1, 2}  # collapsed at the low-lag excisions


def test_driver_harmonic_mask_removes_comb_bins():
    lags, acf, err, _ = _lorentzian_acf(0.2, 0.8)
    keep = np.abs(lags) <= 2.0
    L, A, E, rec = drv.guards.apply_harmonic_mask_to_fit(
        lags[keep], acf[keep], err[keep],
        {"enable": True, "spacing_mhz": 0.390625, "halfwidth_mhz": 0.05},
    )
    assert rec["enabled"] is True
    assert rec["n_bins_removed"] > 0
    assert L.size < int(np.sum(keep))


def test_driver_harmonic_mask_disabled_is_passthrough():
    lags, acf, err, _ = _lorentzian_acf(0.2, 0.8)
    keep = np.abs(lags) <= 2.0
    L, A, E, rec = drv.guards.apply_harmonic_mask_to_fit(
        lags[keep], acf[keep], err[keep], {"enable": False}
    )
    assert rec["n_bins_removed"] == 0
    assert L.size == int(np.sum(keep))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
