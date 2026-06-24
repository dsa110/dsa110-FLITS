"""Phase 6: ACF re-validation harness (Nimmo & Pleunis 2025 bandwidth method).

Seeded synthetic spectra (mirroring tests/test_noise.py fixtures) exercise RFI
flagging, off-pulse masking, the single-screen HWHM Δν, and the two-screen
wide+narrow recovery — the Nimmo/Pleunis fidelity oracle. The injected decorrelation
scales are the known truth; Δν = Lorentzian HWHM and m = sqrt(peak) per Pleunis
Eq 5.1 / Nimmo Eq 4.26.
"""

from __future__ import annotations

import sys
from pathlib import Path

_test_dir = Path(__file__).parent
sys.path.insert(0, str(_test_dir.parent.parent.parent))  # FLITS root
sys.path.insert(0, str(_test_dir.parent.parent))  # scintillation dir

import numpy as np

from scint_analysis.revalidation import (
    emission_size,
    fit_two_screen_acf,
    off_pulse_mask,
    res,
    revalidate_dnu,
    rfi_flag,
)


def test_rfi_spike_flagged():
    rng = np.random.default_rng(0)
    spec = rng.normal(10, 1, 256)
    spec[128] = 80.0  # one RFI channel
    flags = rfi_flag(spec, n_sigma=5)
    assert flags[128] and flags.sum() <= 3


def test_offpulse_mask_excludes_burst():
    prof = np.r_[np.ones(40), 50 * np.ones(8), np.ones(40)]  # burst in the middle
    m = off_pulse_mask(prof, k=3.0)
    assert not m[44] and m[0] and m[-1]


def test_clean_dnu_is_hwhm():
    """Single-screen: Δν recovered as the Lorentzian HWHM (Pleunis Eq 5.1) within an
    order-of-magnitude band of the injected ~10-channel correlation scale."""
    rng = np.random.default_rng(1)
    white = rng.normal(0, 1, 266)
    corr = np.convolve(white, np.ones(10) / 10, mode="valid")[:256]  # Δν ~ 10 chan
    spec = 100 + 20 * corr
    dnu = revalidate_dnu(spec, channel_width_mhz=0.39)
    # ~10-channel boxcar -> HWHM a few channels * 0.39 MHz; generous physical band.
    assert 0.5 < dnu < 8.0


def test_first_lag_two_drops_lag_one():
    """Nimmo's CHIME-upchannelized option (first_lag=2 drops lag 1 too) still recovers
    a comparable single-screen Δν, proving the opt-in lag-1 exclusion is wired."""
    rng = np.random.default_rng(3)
    white = rng.normal(0, 1, 266)
    corr = np.convolve(white, np.ones(10) / 10, mode="valid")[:256]
    spec = 100 + 20 * corr
    dnu1 = revalidate_dnu(spec, channel_width_mhz=0.39, first_lag=1)
    dnu2 = revalidate_dnu(spec, channel_width_mhz=0.39, first_lag=2)
    assert np.isfinite(dnu1) and np.isfinite(dnu2)
    assert 0.5 < dnu2 < 8.0


def test_two_screen_wide_and_narrow_recovered():
    """Inject two decorrelation scales (wide MW + narrow host); the center-omitted
    double-Lorentzian fit recovers two distinct scales and a sane modulation index
    (Pleunis 2505.04576 §5.1, Eq 4.26)."""
    rng = np.random.default_rng(2)
    n = 4096
    wide = np.convolve(rng.normal(0, 1, n + 80), np.ones(80) / 80, "valid")[:n]  # broad
    narrow = np.convolve(rng.normal(0, 1, n + 6), np.ones(6) / 6, "valid")[:n]  # fine
    spec = 100 * (1 + 0.6 * wide) * (1 + 0.6 * narrow)  # 2-screen product
    res_d = fit_two_screen_acf(spec, channel_width_mhz=0.0305)  # DSA-like fine res
    assert res_d["dnu_wide_mhz"] > 5 * res_d["dnu_narrow_mhz"]  # two distinct scales
    assert 0.0 < res_d["m_total"] <= 2.0 and res_d["center_omitted"] is True


def test_emission_size_nimmo_port():
    """res()/emission_size() reproduce the Nimmo et al. 2025 forms: a smaller
    modulation index implies a larger emission region for a fixed screen."""
    phys_res = res(lens_dist_kpc=11.0, lda_m=0.21, scat_lens_ms=0.1)
    assert phys_res > 0
    big = emission_size(phys_res, mod_ind=0.5)
    small = emission_size(phys_res, mod_ind=0.95)
    assert big > small > 0
    # m -> 1 (unresolved) collapses the emission size toward 0.
    assert emission_size(phys_res, mod_ind=0.999) < small


from scint_analysis.revalidation import _lor, compare_lorentzian_components  # noqa: E402


def _synthetic_acf(components, span=2.0, dch=0.01, noise=3e-3, seed=0):
    """Symmetric, lag-0-excluded ACF (as _mean_normalized_acf returns) summing the
    given (gamma, m) Lorentzian components plus white noise."""
    rng = np.random.default_rng(seed)
    pos = np.arange(1, int(span / dch) + 1) * dch
    lags = np.concatenate((-pos[::-1], pos))
    acf = np.zeros(lags.size)
    for g, m in components:
        acf = acf + _lor(lags, g, m)
    return lags, acf + rng.normal(0, noise, lags.size)


def test_single_lorentzian_prefers_one():
    """Known-truth oracle: one injected scale -> the BIC + nested-F-test selector
    refuses the second component (ΔBIC must exceed 'strong' AND the F-test fire)."""
    lags, acf = _synthetic_acf([(0.1, 0.8)])
    out = compare_lorentzian_components(lags, acf, max_components=3)
    assert out["n_preferred"] == 1
    assert out["delta_bic"][2] < 6.0  # second component does not strongly improve BIC


def test_two_lorentzians_prefers_two():
    """Known-truth oracle: two well-separated scales -> selector picks exactly 2 and
    recovers two distinct decorrelation bandwidths."""
    lags, acf = _synthetic_acf([(0.04, 0.7), (0.7, 0.6)])
    out = compare_lorentzian_components(lags, acf, max_components=3)
    assert out["n_preferred"] == 2
    assert out["delta_bic"][2] > 6.0 and out["f_test"][2] < 0.05
    comps = next(f for f in out["fits"] if f["n"] == 2)["components"]
    dnus = sorted(c["dnu_mhz"] for c in comps)
    assert dnus[1] > 5 * dnus[0]  # two genuinely distinct scales
