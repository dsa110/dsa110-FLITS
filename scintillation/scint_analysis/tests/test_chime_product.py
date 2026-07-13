from __future__ import annotations

import json

import numpy as np
import pytest

from scintillation.scint_analysis.chime_product import (
    ChimeProductConfig,
    build_chime_products,
    burst_track_mask,
    load_chime_target,
    verify_product_manifest,
    write_chime_products,
)


def _fixture():
    rng = np.random.default_rng(20260712)
    nchan, ntime = 12, 48
    frequencies = np.linspace(600.0, 612.0, nchan, endpoint=False)
    coarse_frequencies = np.array([601.5, 605.5, 609.5])
    parent = np.repeat(np.arange(3), 4)
    offsets = np.array([0, 2, 4])

    channel_gain = np.linspace(8.0, 12.0, nchan)
    spectral_mode = np.linspace(0.7, 1.3, nchan)
    temporal_mode = np.sin(np.linspace(0, 4 * np.pi, ntime))
    power = channel_gain[:, None] * (
        1.0 + 0.16 * spectral_mode[:, None] * temporal_mode[None, :]
    )
    power += rng.normal(0.0, 0.015, power.shape) * channel_gain[:, None]

    burst_mask = np.zeros_like(power, dtype=bool)
    burst_values = np.linspace(2.0, 4.0, nchan)
    for channel, parent_index in enumerate(parent):
        sample = 25 - offsets[parent_index]
        power[channel, sample] += burst_values[channel] * channel_gain[channel]
        burst_mask[channel, sample] = True

    return power, frequencies, coarse_frequencies, offsets, burst_mask, burst_values


def test_builder_removes_rank1_background_and_preserves_masked_burst():
    power, frequencies, coarse_frequencies, offsets, burst_mask, burst_values = _fixture()
    config = ChimeProductConfig(
        target="freya",
        dm=912.4,
        upchannel_factor=64,
        dt_s=0.001,
        off_pulse=(0, 18),
        guard_bins=1,
    )

    result = build_chime_products(
        power,
        frequencies,
        coarse_frequencies,
        coarse_offsets=offsets,
        burst_mask=burst_mask,
        config=config,
    )
    without_burst = power.copy()
    parent = np.repeat(np.arange(3), 4)
    for channel, parent_index in enumerate(parent):
        without_burst[channel, 25 - offsets[parent_index]] -= (
            burst_values[channel] * np.linspace(8.0, 12.0, power.shape[0])[channel]
        )
    control = build_chime_products(
        without_burst,
        frequencies,
        coarse_frequencies,
        coarse_offsets=offsets,
        burst_mask=burst_mask,
        config=config,
    )

    corrected_off = result.corrected[:, :18]
    normalized_off = corrected_off / np.nanmedian(corrected_off, axis=1, keepdims=True)
    assert np.nanmean(np.abs(np.nanmean(normalized_off - 1.0, axis=0))) < 0.02

    # Every dispersed burst sample lands at aligned canvas bin 25. The fitted
    # background may be removed there, but the injected burst amplitude itself
    # must survive to within the seeded noise tolerance.
    recovered_burst = (result.corrected[:, 25] - control.corrected[:, 25]) / np.linspace(
        8.0, 12.0, power.shape[0]
    )
    assert np.allclose(recovered_burst, burst_values, atol=1e-5)
    assert result.manifest["correction"]["algorithm"] == "robust_coarse_rank1_v1"
    assert result.manifest["correction"]["retained_fraction"] > 0.9


def test_builder_alignment_is_padded_not_circular():
    power = np.arange(18, dtype=float).reshape(3, 6) + 1.0
    result = build_chime_products(
        power,
        frequencies_mhz=np.array([600.0, 601.0, 602.0]),
        coarse_frequencies_mhz=np.array([600.0, 601.0, 602.0]),
        coarse_offsets=np.array([0, 1, 3]),
        burst_mask=np.ones_like(power, dtype=bool),
        config=ChimeProductConfig(
            target="synthetic",
            dm=100.0,
            upchannel_factor=1,
            dt_s=1.0,
            off_pulse=(0, 6),
        ),
    )

    assert result.uncorrected.shape == (3, 9)
    assert np.isnan(result.uncorrected[0, 6:]).all()
    assert np.isnan(result.uncorrected[2, :3]).all()
    assert np.array_equal(result.uncorrected[2, 3:], power[2])


def test_writer_preserves_both_products_and_deterministic_manifest(tmp_path):
    power, frequencies, coarse_frequencies, offsets, burst_mask, _ = _fixture()
    config = ChimeProductConfig(
        target="freya",
        dm=912.4,
        upchannel_factor=64,
        dt_s=0.001,
        off_pulse=(0, 18),
    )
    result = build_chime_products(
        power,
        frequencies,
        coarse_frequencies,
        coarse_offsets=offsets,
        burst_mask=burst_mask,
        config=config,
    )

    paths = write_chime_products(result, tmp_path / "freya_chime", input_paths=[])
    assert paths["uncorrected"].exists()
    assert paths["corrected"].exists()
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["schema_version"] == 1
    assert manifest["target"] == "freya"
    assert manifest["products"]["corrected_sha256"]

    second = write_chime_products(result, tmp_path / "repeat", input_paths=[])
    second_manifest = json.loads(second["manifest"].read_text())
    assert manifest["correction"] == second_manifest["correction"]
    assert manifest["alignment"] == second_manifest["alignment"]
    verified = verify_product_manifest(paths["manifest"], paths["corrected"])
    assert verified["valid"] is True

    paths["corrected"].write_bytes(paths["corrected"].read_bytes() + b"tamper")
    rejected = verify_product_manifest(paths["manifest"], paths["corrected"])
    assert rejected["valid"] is False
    assert rejected["reason"] == "corrected product hash mismatch"


def test_writer_refuses_to_overwrite_existing_product_prefix(tmp_path):
    power, frequencies, coarse_frequencies, offsets, burst_mask, _ = _fixture()
    result = build_chime_products(
        power,
        frequencies,
        coarse_frequencies,
        coarse_offsets=offsets,
        burst_mask=burst_mask,
        config=ChimeProductConfig(
            target="freya",
            dm=912.4,
            upchannel_factor=64,
            dt_s=0.001,
            off_pulse=(0, 18),
        ),
    )
    prefix = tmp_path / "freya_chime"
    write_chime_products(result, prefix, input_paths=[])
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_chime_products(result, prefix, input_paths=[])


def test_manifest_verification_binds_product_to_expected_target(tmp_path):
    power, frequencies, coarse_frequencies, offsets, burst_mask, _ = _fixture()
    result = build_chime_products(
        power,
        frequencies,
        coarse_frequencies,
        coarse_offsets=offsets,
        burst_mask=burst_mask,
        config=ChimeProductConfig(
            target="freya",
            dm=912.4,
            upchannel_factor=64,
            dt_s=0.001,
            off_pulse=(0, 18),
        ),
    )
    paths = write_chime_products(result, tmp_path / "freya", input_paths=[])
    verdict = verify_product_manifest(
        paths["manifest"], paths["corrected"], expected_target="hamilton"
    )
    assert verdict["valid"] is False
    assert verdict["reason"] == "manifest target does not match configured burst"


def test_builder_fails_closed_without_required_provenance():
    power = np.ones((2, 4))
    with pytest.raises(ValueError, match="target"):
        build_chime_products(
            power,
            np.array([600.0, 601.0]),
            np.array([600.0, 601.0]),
            coarse_offsets=np.array([0, 0]),
            burst_mask=np.zeros_like(power, dtype=bool),
            config=ChimeProductConfig(
                target="",
                dm=100.0,
                upchannel_factor=1,
                dt_s=1.0,
                off_pulse=(0, 4),
            ),
        )


def test_target_registry_covers_complete_codetection_sample():
    registry = load_chime_target()
    assert set(registry) == {
        "casey",
        "chromatica",
        "freya",
        "hamilton",
        "isha",
        "johndoeII",
        "mahi",
        "oran",
        "phineas",
        "whitney",
        "wilhelm",
        "zach",
    }
    assert registry["freya"]["upchannel_factor"] == 64
    assert registry["isha"]["measurement_eligibility"] == "upper_limit_only"


def test_burst_track_mask_maps_aligned_window_back_before_alignment():
    mask = burst_track_mask(
        n_channels=4,
        n_times=12,
        channel_offsets=np.array([0, 1, 3, 3]),
        aligned_center_bin=7,
        half_width_bins=1,
    )
    assert np.flatnonzero(mask[0]).tolist() == [6, 7, 8]
    assert np.flatnonzero(mask[1]).tolist() == [5, 6, 7]
    assert np.flatnonzero(mask[2]).tolist() == [3, 4, 5]


def test_builder_removes_independent_common_mode_per_coarse_block():
    ntime = 40
    t = np.linspace(0, 2 * np.pi, ntime)
    power = np.full((8, ntime), 10.0)
    power[:4] *= 1.0 + 0.2 * np.sin(t)
    power[4:] *= 1.0 + 0.2 * np.cos(2 * t)
    result = build_chime_products(
        power,
        frequencies_mhz=np.arange(8, dtype=float) + 600.0,
        coarse_frequencies_mhz=np.array([601.5, 605.5]),
        coarse_offsets=np.array([0, 2]),
        burst_mask=np.zeros_like(power, dtype=bool),
        config=ChimeProductConfig(
            target="block-modes",
            dm=100.0,
            upchannel_factor=4,
            dt_s=1.0,
            off_pulse=(0, ntime),
        ),
    )
    for channel, offset in enumerate([0] * 4 + [2] * 4):
        assert np.nanstd(result.corrected[channel, offset : offset + ntime]) < 1e-6
