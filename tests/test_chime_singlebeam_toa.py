from __future__ import annotations

import astropy.units as u
import pytest
from astropy.time import Time

from crossmatching.chime_singlebeam import (
    CHIME_BASEBAND_SAMPLE_TIME_S,
    SinglebeamExtractionError,
    extract_singlebeam_toa,
    toa_400_from_peak,
)
from crossmatching.toa_crossmatch import compute_toa


def test_toa_400_from_peak_matches_notebook_convention():
    """Pure timing math: peak sample -> native + 400 MHz TOA via shared compute_toa.

    Mirrors notebook cell 11: t0 = time0[-1], offset = peak*delta_time,
    f_center = bottom channel (~400 MHz), shift to 400 MHz with K_DM.
    """
    peak_index = 24142
    dm = 262.368
    ctime, ctime_offset = 1644261800.0, 0.000123
    f_center = 400.390625  # bottom-of-band channel centre

    native, toa_400 = toa_400_from_peak(
        peak_index=peak_index,
        sample_time_s=CHIME_BASEBAND_SAMPLE_TIME_S,
        time0_ctime=ctime,
        time0_ctime_offset=ctime_offset,
        native_frequency_mhz=f_center,
        dm=dm,
    )

    expected_native = (
        Time(ctime, val2=ctime_offset, format="unix", scale="utc")
        + (peak_index * CHIME_BASEBAND_SAMPLE_TIME_S) * u.s
    )
    expected_400 = compute_toa(
        expected_native, 0 * u.s, f_center * u.MHz, dm * (u.pc / u.cm**3), 400.0 * u.MHz
    )

    assert native.to_value("unix") == pytest.approx(expected_native.to_value("unix"), abs=1e-9)
    assert toa_400.to_value("unix") == pytest.approx(expected_400.to_value("unix"), abs=1e-9)
    # 400 MHz arrives after the 400.39 MHz native channel (delay ~ 1/f^2); a few ms here
    shift_s = toa_400.to_value("unix") - native.to_value("unix")
    assert 0.0 < shift_s < 0.05


def test_extract_fails_loudly_without_baseband_analysis():
    """The canonical extractor needs the CANFAR runtime; absent it, fail loudly.

    Full extraction (dedispersion + peak) is exercised in the CANFAR image, not here.
    """
    try:
        import baseband_analysis  # noqa: F401
    except ModuleNotFoundError:
        with pytest.raises(SinglebeamExtractionError, match="baseband_analysis is required"):
            extract_singlebeam_toa("/nonexistent/singlebeam.h5", dm=262.368)
    else:
        pytest.skip("baseband_analysis present; full extraction runs in the CANFAR image")
