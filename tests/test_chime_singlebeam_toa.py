from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time

from crossmatching.chime_singlebeam import (
    CHIME_BASEBAND_SAMPLE_TIME_S,
    SinglebeamExtractionError,
    extract_singlebeam_toa,
    inspect_singlebeam_layout,
)
from crossmatching.toa_crossmatch import compute_toa

h5py = pytest.importorskip("h5py")


def _write_synthetic_singlebeam(path, *, peak_index=37):
    time0_dtype = np.dtype(
        [
            ("ctime", "<f8"),
            ("ctime_offset", "<f8"),
            ("fpga_count", "<i8"),
        ]
    )
    time0 = np.zeros(4, dtype=time0_dtype)
    time0["ctime"] = 1644261861.0
    time0["ctime_offset"] = np.array([0.000001, 0.000002, 0.000003, 0.000004])
    time0["fpga_count"] = np.arange(4)

    freq_dtype = np.dtype([("centre", "<f8")])
    freq = np.zeros(4, dtype=freq_dtype)
    freq["centre"] = [800.0, 700.0, 600.0, 500.0]

    data = np.ones((4, 2, 128), dtype=np.complex64)
    data[:, :, peak_index] = 20 + 0j

    with h5py.File(path, "w") as h5:
        h5.create_dataset("time0", data=time0)
        h5.create_dataset("index_map/freq", data=freq)
        h5.create_dataset("tiedbeam_baseband", data=data)


def test_inspects_chime_singlebeam_layout_without_reading_payload(tmp_path):
    path = tmp_path / "singlebeam_test.h5"
    _write_synthetic_singlebeam(path)

    layout = inspect_singlebeam_layout(path)

    assert layout.data_path == "tiedbeam_baseband"
    assert layout.time0_path == "time0"
    assert layout.freq_path == "index_map/freq"
    assert layout.data_shape == (4, 2, 128)
    assert layout.frequency_axis == 0
    assert layout.polarization_axis == 1
    assert layout.time_axis == 2
    assert layout.sample_time_s == pytest.approx(CHIME_BASEBAND_SAMPLE_TIME_S)


def test_extracts_candidate_toa_from_synthetic_singlebeam(tmp_path):
    path = tmp_path / "singlebeam_test.h5"
    peak_index = 37
    dm = 219.46
    _write_synthetic_singlebeam(path, peak_index=peak_index)

    result = extract_singlebeam_toa(path, dm=dm)

    native = (
        Time(1644261861.0, val2=0.000003, format="unix", scale="utc")
        + (peak_index * CHIME_BASEBAND_SAMPLE_TIME_S) * u.s
    )
    expected_400 = compute_toa(
        native,
        0 * u.s,
        600.0 * u.MHz,
        dm * (u.pc / u.cm**3),
        400.0 * u.MHz,
    )

    assert result.method == "power_sum_peak_no_coherent_dedispersion"
    assert result.peak_index == peak_index
    assert result.time0_frequency_index == 2
    assert result.native_frequency_mhz == pytest.approx(600.0)
    assert result.native_toa_unix == pytest.approx(native.to_value("unix"), abs=1e-9)
    assert result.toa_unix_400 == pytest.approx(expected_400.to_value("unix"), abs=1e-9)
    assert result.peak_snr_like > 1


def test_singlebeam_toa_requires_time0_metadata(tmp_path):
    path = tmp_path / "missing_time0.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("tiedbeam_baseband", data=np.ones((4, 2, 16), dtype=np.complex64))

    with pytest.raises(SinglebeamExtractionError, match="missing required time0"):
        extract_singlebeam_toa(path, dm=100.0, native_frequency_mhz=600.0)
