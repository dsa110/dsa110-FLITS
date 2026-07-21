"""Unit checks for provenance-preserving CHIME detected products."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from scipy.fft import fft, fftshift

MODULE = Path(__file__).with_name("upchannelize_chime.py")
WINDOWED_MODULE = Path(__file__).with_name("windowed_upchan.py")


def _module():
    spec = importlib.util.spec_from_file_location("upchannelize_chime", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _windowed_module():
    spec = importlib.util.spec_from_file_location("windowed_upchan", WINDOWED_MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _package_reference(wfall, freq_id, *, fftsize, downfreq):
    """Dependency-light copy of baseband_analysis 1.9.0 ``_upchannel``."""
    values = np.swapaxes(np.swapaxes(np.asarray(wfall), 0, 1), 1, 2)
    npol, nsamp, nchan = values.shape
    upchan = fftsize // downfreq
    nblock = nsamp // fftsize
    spectrum = np.zeros((npol, nblock, nchan * upchan), dtype=np.complex64)
    channel_ids = np.zeros(nchan * upchan, dtype=int)
    full_band = np.linspace(800.1953125, 400.1953125, upchan * 1024)
    for pol in range(npol):
        for block in range(nblock):
            for channel in range(nchan):
                time_series = values[
                    pol, block * fftsize : (block + 1) * fftsize, channel
                ].copy()
                transformed = fftshift(fft(time_series))
                transformed = transformed.reshape(upchan, downfreq).mean(axis=1).copy()
                start = channel * upchan
                spectrum[pol, block, start : start + upchan] = transformed
                channel_ids[start : start + upchan] = np.arange(
                    upchan * freq_id[channel], upchan * freq_id[channel] + upchan
                )
    return spectrum, full_band[channel_ids], channel_ids


def test_detected_products_preserve_independent_polarizations_and_stokes_sum():
    module = _module()
    rng = np.random.default_rng(20260714)
    voltages = rng.normal(size=(2, 7, 11)) + 1j * rng.normal(size=(2, 7, 11))

    stokes_i, per_pol = module._detected_products(voltages)

    assert per_pol.shape == (2, 11, 7)
    np.testing.assert_allclose(per_pol[0], np.abs(voltages[0]).T ** 2)
    np.testing.assert_allclose(per_pol[1], np.abs(voltages[1]).T ** 2)
    np.testing.assert_allclose(stokes_i, per_pol.sum(axis=0))


def test_detected_products_rejects_missing_polarization_axis():
    module = _module()

    with np.testing.assert_raises_regex(ValueError, "shape"):
        module._detected_products(np.ones((8, 16), dtype=complex))


def test_nominal_grid_restoration_preserves_measured_values_and_masks_gaps():
    module = _module()
    upchan = 2
    fine_ids = np.array([2, 3, 8, 9])
    package_grid = np.linspace(800.1953125, 400.1953125, 1024 * upchan)
    package_freq = package_grid[fine_ids]
    stokes = np.arange(12, dtype=np.float32).reshape(4, 3)
    per_pol = np.stack((stokes, stokes + 100.0))

    restored = module._restore_nominal_fine_grid(
        stokes, per_pol, package_freq, fine_ids, upchan
    )
    full_stokes, full_per_pol, nominal_freq, full_package_freq, valid = restored

    assert full_stokes.shape == (2048, 3)
    assert full_per_pol.shape == (2, 2048, 3)
    assert valid.shape == (2048,)
    assert valid.sum() == 4
    np.testing.assert_array_equal(full_stokes[fine_ids], stokes)
    np.testing.assert_array_equal(full_per_pol[:, fine_ids], per_pol)
    assert np.isnan(full_stokes[~valid]).all()
    assert np.isnan(full_per_pol[:, ~valid]).all()
    np.testing.assert_array_equal(full_package_freq, package_grid)
    expected_nominal = 800.1953125 - (np.arange(2048) + 0.5) * (0.390625 / upchan)
    np.testing.assert_array_equal(nominal_freq, expected_nominal)


def test_nominal_grid_restoration_rejects_bad_fine_identifiers():
    module = _module()
    stokes = np.ones((2, 3))
    per_pol = np.ones((2, 2, 3))
    package_grid = np.linspace(800.1953125, 400.1953125, 2048)

    with np.testing.assert_raises_regex(ValueError, "unique"):
        module._restore_nominal_fine_grid(
            stokes, per_pol, package_grid[[2, 2]], np.array([2, 2]), 2
        )
    with np.testing.assert_raises_regex(ValueError, "range"):
        module._restore_nominal_fine_grid(
            stokes, per_pol, np.array([1.0, 2.0]), np.array([2, 2048]), 2
        )


def test_nominal_grid_restoration_checks_package_frequency_mapping():
    module = _module()
    stokes = np.ones((2, 3))
    per_pol = np.ones((2, 2, 3))

    with np.testing.assert_raises_regex(ValueError, "package frequency"):
        module._restore_nominal_fine_grid(
            stokes, per_pol, np.array([700.0, 600.0]), np.array([2, 3]), 2
        )


def test_variant_suffix_preserves_historical_names_and_separates_variants():
    module = _module()

    assert module._variant_suffix(None, None) == ""
    assert module._variant_suffix("hann", 4) == "_hann_os4"
    with np.testing.assert_raises_regex(ValueError, "supplied together"):
        module._variant_suffix("hann", None)


def test_rectangular_oversample_two_is_package_equivalent():
    module = _windowed_module()
    rng = np.random.default_rng(20260714)
    wfall = (
        rng.normal(size=(3, 2, 47)) + 1j * rng.normal(size=(3, 2, 47))
    ).astype(np.complex64)
    freq_id = np.array([4, 8, 11])

    actual = module.windowed_upchannel(
        wfall,
        freq_id,
        upchan_factor=4,
        window="rectangular",
        oversample=2,
    )
    expected = _package_reference(wfall, freq_id, fftsize=8, downfreq=2)

    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])
    assert actual[3]["hop_samples"] == 8
    assert actual[3]["frame_center_offset_samples"] == 3.5
    assert actual[3]["normalization"] == "package_exact"


def test_oversample_four_preserves_output_cadence_and_frequency_grid():
    module = _windowed_module()
    rng = np.random.default_rng(18)
    wfall = rng.normal(size=(2, 2, 64)) + 1j * rng.normal(size=(2, 2, 64))
    freq_id = np.array([2, 9])

    two = module.windowed_upchannel(
        wfall, freq_id, upchan_factor=4, window="hann", oversample=2
    )
    four = module.windowed_upchannel(
        wfall, freq_id, upchan_factor=4, window="hann", oversample=4
    )

    assert two[0].shape == (2, 8, 8)
    assert four[0].shape == (2, 7, 8)
    np.testing.assert_array_equal(four[1], two[1])
    np.testing.assert_array_equal(four[2], two[2])
    assert four[3]["fft_size"] == 16
    assert four[3]["downfreq"] == 4
    assert four[3]["hop_samples"] == 8
    assert four[3]["frame_center_offset_samples"] == 7.5


def test_exact_grouped_bin_normalization_preserves_white_noise_power():
    module = _windowed_module()
    rng = np.random.default_rng(1024)
    wfall = (
        rng.normal(size=(1, 2, 131072)) + 1j * rng.normal(size=(1, 2, 131072))
    ).astype(np.complex64)
    freq_id = np.array([5])
    powers = {}
    for window, oversample in (
        ("rectangular", 2),
        ("hann", 2),
        ("hann", 4),
        ("blackmanharris", 2),
        ("blackmanharris", 4),
    ):
        spectrum, _, _, metadata = module.windowed_upchannel(
            wfall,
            freq_id,
            upchan_factor=16,
            window=window,
            oversample=oversample,
        )
        powers[(window, oversample)] = float(np.mean(np.abs(spectrum) ** 2))
        assert np.isclose(metadata["grouped_noise_gain"], 16.0, rtol=1e-12)
    baseline = powers[("rectangular", 2)]
    for value in powers.values():
        assert np.isclose(value, baseline, rtol=0.03)


def test_windows_reduce_fractional_bin_far_sidelobes():
    module = _windowed_module()
    upchan = 64
    size = 2 * upchan
    samples = np.arange(size)
    tone = np.exp(2j * np.pi * 9.37 * samples / size)[None, None, :]
    freq_id = np.array([0])

    spectra = {}
    for window in ("rectangular", "hann", "blackmanharris"):
        spectrum, _, _, _ = module.windowed_upchannel(
            tone,
            freq_id,
            upchan_factor=upchan,
            window=window,
            oversample=2,
        )
        power = np.abs(spectrum[0, 0]) ** 2
        power /= power.max()
        spectra[window] = power

    peak = int(np.argmax(spectra["rectangular"]))
    far = np.ones(upchan, dtype=bool)
    far[max(0, peak - 3) : min(upchan, peak + 4)] = False
    assert spectra["hann"][far].max() < spectra["rectangular"][far].max() / 5
    assert spectra["blackmanharris"][far].max() < spectra["rectangular"][far].max() / 20
