from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import shift as nd_shift

from dispersion.dm_phase_suite.coherence import coherence_curve
from dispersion.dm_phase_suite.cutoff import width_derived_cutoffs
from dispersion.dm_phase_suite.model import ResolutionEvaluation
from dispersion.dm_phase_suite.quality import (
    bootstrap_central_fraction,
    calibrated_injection_sigma,
)
from dispersion.dm_phase_suite.resolution import (
    block_average,
    resolution_factors,
    select_resolution,
)
from dispersion.dm_phase_suite.search import search_dm
from dispersion.dm_phase_suite.shifts import (
    K_DM_S_MHZ2,
    fractional_zero_fill_shift,
    residual_delay_s,
)


def test_physical_delay_scale_and_sign() -> None:
    freq = np.array([400.0, 800.0])
    delay = residual_delay_s(freq, 1.0, 800.0)
    assert delay[0] == pytest.approx(0.019453125, rel=2e-3)
    assert delay[0] > 0
    assert delay[1] == pytest.approx(0.0)
    assert K_DM_S_MHZ2 > 4000


def test_fractional_shift_is_zero_filled_not_circular() -> None:
    x = np.zeros((1, 64))
    x[0, -2] = 1.0
    shifted = fractional_zero_fill_shift(x, np.array([5.0]), guard_samples=4)
    assert np.max(np.abs(shifted[:, :8])) < 1e-10
    assert np.max(np.abs(shifted)) < 1e-10


def test_shift_then_inverse_recovers_padded_interior() -> None:
    rng = np.random.default_rng(2)
    raw = rng.normal(size=(3, 128))
    kernel = np.exp(-0.5 * (np.arange(-8, 9) / 3.0) ** 2)
    kernel /= kernel.sum()
    x = np.vstack([np.convolve(channel, kernel, mode="same") for channel in raw])
    offsets = np.array([-3.25, 0.0, 4.5])
    y = fractional_zero_fill_shift(x, offsets, guard_samples=8)
    z = fractional_zero_fill_shift(y, -offsets, guard_samples=8)
    np.testing.assert_allclose(z[:, 12:-12], x[:, 12:-12], atol=2e-2, rtol=2e-2)


def _independent_dispersed_waterfall(dm: float, seed: int = 3):
    rng = np.random.default_rng(seed)
    freq = np.linspace(400.0, 800.0, 64)
    dt = 1e-4
    nt = 3000
    t = np.arange(nt)
    base = np.exp(-0.5 * ((t - 1200) / 4.0) ** 2)
    base += 0.65 * np.exp(-0.5 * ((t - 1245) / 6.0) ** 2)
    delays = residual_delay_s(freq, dm, freq.max()) / dt
    clean = np.vstack([nd_shift(base, d, order=3, mode="constant", cval=0.0) for d in delays])
    return clean + 0.04 * rng.normal(size=clean.shape), freq, dt


def test_coherence_curve_recovers_independent_positive_residual() -> None:
    wf, freq, dt = _independent_dispersed_waterfall(0.7)
    grid = np.arange(-0.5, 1.51, 0.1)
    curve = coherence_curve(wf, freq, dt, grid, f_cut_hz=(50.0, 2500.0))
    best = grid[int(np.argmax(curve.score))]
    assert best == pytest.approx(0.7, abs=0.15)
    assert 0 < int(np.argmax(curve.score)) < grid.size - 1


def test_search_is_deterministic_and_returns_physical_absolute_dm() -> None:
    wf, freq, dt = _independent_dispersed_waterfall(0.55, seed=8)
    kwargs = dict(
        waterfall=wf,
        frequencies_mhz=freq,
        sample_time_s=dt,
        reference_dm=500.0,
        coarse_grid=np.arange(-0.5, 1.51, 0.1),
        fine_step=0.02,
        f_cut_hz=(50.0, 2500.0),
    )
    a = search_dm(**kwargs)
    b = search_dm(**kwargs)
    assert a.residual_dm == pytest.approx(b.residual_dm, abs=1e-12)
    assert a.absolute_dm == pytest.approx(500.55, abs=0.12)
    assert not a.edge_peak


def test_resolution_grid_is_native_first_and_block_average_is_exact() -> None:
    assert resolution_factors("chime")[0] == (1, 1)
    wf = np.arange(8 * 16, dtype=float).reshape(8, 16)
    freq = np.arange(8, dtype=float)
    reduced, reduced_freq = block_average(wf, freq, 2, 4)
    assert reduced.shape == (4, 4)
    assert reduced[0, 0] == pytest.approx(np.mean(wf[:2, :4]))
    np.testing.assert_allclose(reduced_freq, [0.5, 2.5, 4.5, 6.5])


def _evaluation(ff: int, tf: int, dm: float, eligible: bool = True) -> ResolutionEvaluation:
    return ResolutionEvaluation(
        frequency_factor=ff,
        time_factor=tf,
        shape=(64, 256),
        residual_dm=dm,
        sigma=0.03,
        profile_snr=12.0,
        coherence_peak_z=8.0,
        bootstrap_success_fraction=1.0,
        edge_peak=False,
        cutoff_stable=True,
        eligible=eligible,
        failure_reasons=(),
    )


def test_resolution_selection_requires_stability_and_prefers_highest_information() -> None:
    rows = [_evaluation(1, 1, 0.50), _evaluation(2, 1, 0.54), _evaluation(1, 2, 1.2)]
    selected = select_resolution(rows)
    assert selected is rows[0]
    assert select_resolution([rows[0]]) is None


def test_width_derived_cutoff_is_lower_for_broader_pulse() -> None:
    t = np.arange(2048)
    narrow = np.exp(-0.5 * ((t - 1000) / 3.0) ** 2)[None, :].repeat(16, axis=0)
    broad = np.exp(-0.5 * ((t - 1000) / 30.0) ** 2)[None, :].repeat(16, axis=0)
    narrow_cut = width_derived_cutoffs(narrow, 1e-4)[1][1]
    broad_cut = width_derived_cutoffs(broad, 1e-4)[1][1]
    assert broad_cut < narrow_cut
    assert broad_cut >= 100.0


def test_bootstrap_central_fraction_rejects_split_modes() -> None:
    clean = np.array([-0.11, -0.10, -0.09, -0.105, -0.095])
    split = np.array([-0.11, -0.10, -0.09, 0.0, 0.01])
    assert bootstrap_central_fraction(clean, -0.10, 0.01) == 1.0
    assert bootstrap_central_fraction(split, -0.10, 0.01) == pytest.approx(0.6)


def test_injection_sigma_uses_supported_snr_floor() -> None:
    assert np.isnan(calibrated_injection_sigma("dsa", 49.9))
    assert calibrated_injection_sigma("dsa", 50.0) == pytest.approx(0.11162548245)
    assert calibrated_injection_sigma("chime", 30.0) == pytest.approx(0.00369433530)
