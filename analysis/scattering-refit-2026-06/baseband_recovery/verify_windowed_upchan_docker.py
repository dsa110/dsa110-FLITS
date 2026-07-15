#!/usr/bin/env python3
"""Dependency-light checks run inside the pinned CHIME production image."""

from __future__ import annotations

import json

import numpy as np
from baseband_analysis.core.sampling import _upchannel
from windowed_upchan import windowed_upchannel


def main() -> int:
    rng = np.random.default_rng(20260714)
    wfall = (
        rng.normal(size=(3, 2, 4096)) + 1j * rng.normal(size=(3, 2, 4096))
    ).astype(np.complex64)
    freq_id = np.array([4, 8, 11])
    upchan = 16

    expected = _upchannel(wfall, freq_id=freq_id, fftsize=2 * upchan, downfreq=2)
    actual = windowed_upchannel(
        wfall,
        freq_id,
        upchan_factor=upchan,
        window="rectangular",
        oversample=2,
    )
    for observed, reference in zip(actual[:3], expected, strict=True):
        np.testing.assert_array_equal(observed, reference)

    powers = {}
    for window, oversample in (
        ("rectangular", 2),
        ("hann", 2),
        ("hann", 4),
        ("blackmanharris", 2),
        ("blackmanharris", 4),
    ):
        spectrum, _, _, metadata = windowed_upchannel(
            wfall,
            freq_id,
            upchan_factor=upchan,
            window=window,
            oversample=oversample,
        )
        powers[f"{window}_os{oversample}"] = float(np.mean(np.abs(spectrum) ** 2))
        np.testing.assert_allclose(metadata["grouped_noise_gain"], upchan, rtol=1e-12)

    print(
        json.dumps(
            {
                "status": "pass",
                "package_rectangular_equivalence": "bit_for_bit",
                "grouped_noise_gain": upchan,
                "mean_detected_powers": powers,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
