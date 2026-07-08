from __future__ import annotations

import numpy as np
import pytest

from flits.batch.codetection_data import (
    SUBBURST_PAD_MS,
    band_peak_time,
    chime_toa_shift_ms,
    crop_bands_to_subburst_window,
    subburst_time_window,
)
from flits.batch.codetection_plots import BandSpectrum, _interp_along_time


def _band(
    *,
    peak_ms: float,
    time_ms: np.ndarray,
    tail_ms: float = 0.0,
    tail_amp: float = 0.55,
    label: str = "band",
) -> BandSpectrum:
    profile = np.exp(-0.5 * ((time_ms - peak_ms) / 0.25) ** 2)
    if tail_ms:
        tail = np.where(
            time_ms >= peak_ms,
            tail_amp * np.exp(-(time_ms - peak_ms) / tail_ms),
            0.0,
        )
        profile = np.maximum(profile, tail)
    data = profile[None, :]
    return BandSpectrum(
        freq_mhz=np.array([600.0]),
        time_ms=time_ms,
        data=data,
        model=data.copy(),
        sigma=np.array([1.0]),
        label=label,
    )


def test_subburst_window_caps_long_trailing_half_max_tail():
    early = _band(peak_ms=0.0, time_ms=np.linspace(-8.0, 40.0, 481), label="early")
    late = _band(
        peak_ms=10.0,
        time_ms=np.linspace(-8.0, 40.0, 481),
        tail_ms=30.0,
        tail_amp=0.95,
        label="late",
    )

    x0, x1 = subburst_time_window([early, late])

    assert x0 == pytest.approx(-SUBBURST_PAD_MS, abs=0.11)
    assert x1 < 20.0
    assert x1 > 10.0 + SUBBURST_PAD_MS


def test_crop_bands_to_subburst_window_keeps_requested_window_with_missing_samples():
    t_wide = np.linspace(-10.0, 20.0, 301)
    t_short = np.linspace(-2.0, 20.0, 221)
    bands = [
        _band(peak_ms=0.0, time_ms=t_wide, label="wide"),
        _band(peak_ms=4.0, time_ms=t_short, label="short"),
    ]

    cropped = crop_bands_to_subburst_window(bands, center=False)

    assert cropped[0].time_ms[0] == pytest.approx(-SUBBURST_PAD_MS, abs=0.11)
    assert cropped[1].time_ms[0] == pytest.approx(-SUBBURST_PAD_MS, abs=0.11)
    assert np.isfinite(cropped[0].data[:, 0]).all()
    assert np.isnan(cropped[1].data[:, 0]).all()


def test_time_regrid_marks_out_of_band_samples_as_missing():
    arr = np.array([[1.0, 2.0, 3.0]])
    out = _interp_along_time(
        np.array([-1.0, 0.0, 1.0, 2.0, 3.0]),
        np.array([0.0, 1.0, 2.0]),
        arr,
    )

    assert np.isnan(out[0, 0])
    assert out[0, 1:4] == pytest.approx([1.0, 2.0, 3.0])
    assert np.isnan(out[0, 4])


def test_chime_toa_shift_aligns_cube_peaks_to_crossmatch_offset():
    dsa = _band(peak_ms=2.5, time_ms=np.linspace(0.0, 10.0, 101), label="dsa")
    chime = _band(peak_ms=28.6, time_ms=np.linspace(0.0, 40.0, 401), label="chime")
    offset = 1.32

    shift = chime_toa_shift_ms(dsa, chime, offset)
    chime_aligned = _band(
        peak_ms=28.6 + shift,
        time_ms=np.linspace(0.0, 40.0, 401) + shift,
        label="chime",
    )

    assert band_peak_time(chime_aligned) - band_peak_time(dsa) == pytest.approx(offset, abs=0.05)
