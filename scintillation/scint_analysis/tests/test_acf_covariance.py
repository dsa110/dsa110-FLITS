"""Tests for the MC correlated-lag ACF covariance (A1 trigger, limb-i likelihood).

Null-conditioned single-screen realizations through the production
calculate_acf path; Ledoit-Wolf-shrunk covariance must be PSD, on the same
scale as the production quadrature diagonal, and positively correlated at
short lag separations (the finite-scintle structure a diagonal likelihood
cannot see). See docs/rse/specs/plan-a1-trigger-calibration.md (Faber2026)
Phase 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_test_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_test_dir.parent.parent.parent))  # FLITS root
sys.path.insert(0, str(_test_dir.parent.parent))  # scintillation dir

from scint_analysis.acf_covariance import mc_acf_covariance  # noqa: E402


def test_covariance_psd_and_diagonal_scale():
    cov, diag_ref = mc_acf_covariance(
        gamma_hwhm_mhz=0.4, mod_index=0.8, band_width_mhz=6.0,
        channel_width_mhz=0.05, snr=25.0, n_real=200, max_lag_bins=60,
        seed=42, return_diag_reference=True,
    )
    n = cov.shape[0]
    assert cov.shape == (n, n) and n >= 40
    np.linalg.cholesky(cov)  # PSD
    ratio = np.sqrt(np.diag(cov)) / np.maximum(diag_ref, 1e-12)
    med = float(np.median(ratio))
    # MC diagonal within a factor 3 of the production quadrature diagonal
    assert 1.0 / 3.0 < med < 3.0


def test_offdiagonal_correlation_positive_at_short_lag_separation():
    cov = mc_acf_covariance(
        gamma_hwhm_mhz=0.4, mod_index=0.8, band_width_mhz=6.0,
        channel_width_mhz=0.05, snr=25.0, n_real=200, max_lag_bins=60,
        seed=42,
    )
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    # neighbours within ~gamma (8 channels) must be positively correlated
    assert float(np.median(np.diag(corr, k=3))) > 0.2


def test_evidence_with_mc_covariance_suppresses_sample_variance_escalation():
    # Few-scintle regime: 6 MHz band, gamma = 1.0 MHz -> ~6 scintles. With the
    # correlated-lag covariance, single-screen truth must not show a strong
    # two-screen preference.
    from scint_analysis.acf_covariance import _one_realization
    from scint_analysis.acf_evidence import evidence_with_mc_covariance

    rng = np.random.default_rng(21)
    n_chan = 120  # 6 MHz / 0.05 MHz
    spec = _one_realization(rng, n_chan=n_chan, gamma_bins=20.0,
                            mod_index=0.8, snr=50.0)
    res = evidence_with_mc_covariance(
        np.ma.masked_invalid(spec), channel_width_mhz=0.05, snr=50.0,
        n_real=150, seed=21, nlive=300, dlogz=0.5, max_lag_bins=60,
    )
    assert res["dlnz"] < 5.0
    assert res["gamma_hat_mhz"] > 0
