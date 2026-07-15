"""Shape-compatible windowed replacement for baseband_analysis ``_upchannel``.

The published fine-channel grid retains ``upchan_factor`` channels per CHIME
coarse channel.  ``oversample`` controls both the FFT length and the number of
adjacent *complex* FFT bins averaged into each published fine channel.  A
fixed hop of ``2 * upchan_factor`` samples preserves the production product's
time cadence when the longer oversample-4 frames overlap.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import fft, fftshift

FREQ_TOP_MHZ = 800.1953125
FREQ_BOTTOM_MHZ = 400.1953125
SUPPORTED_WINDOWS = ("rectangular", "hann", "blackmanharris")
SUPPORTED_OVERSAMPLES = (2, 4)


def _window_values(name: str, size: int) -> np.ndarray:
    if name == "rectangular":
        return np.ones(size, dtype=float)
    if name == "hann":
        return np.hanning(size)
    if name == "blackmanharris":
        # Four-term minimum Blackman-Harris window, written locally so the
        # h17 worker does not acquire a new SciPy dependency.
        index = np.arange(size, dtype=float)
        phase = 2.0 * np.pi * index / (size - 1)
        return (
            0.35875
            - 0.48829 * np.cos(phase)
            + 0.14128 * np.cos(2.0 * phase)
            - 0.01168 * np.cos(3.0 * phase)
        )
    raise ValueError(f"unsupported window {name!r}; choose from {SUPPORTED_WINDOWS}")


def _grouped_noise_gain(window: np.ndarray, downfreq: int) -> float:
    """White-noise power gain of FFT plus adjacent complex-bin averaging."""
    size = window.size
    samples = np.arange(size, dtype=float)
    bin_average = np.mean(
        np.exp(-2j * np.pi * np.arange(downfreq)[:, None] * samples[None, :] / size),
        axis=0,
    )
    return float(np.sum(np.abs(window * bin_average) ** 2))


def windowed_upchannel(
    wfall: np.ndarray,
    freq_id: np.ndarray,
    *,
    upchan_factor: int,
    window: str,
    oversample: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Upchannelize ``(coarse_channel, polarization, time)`` voltages.

    The first three return values reproduce the private package function's
    spectrum, frequency, and channel-ID contract.  The fourth records the
    framing and exact noise-power normalization needed to interpret the
    resulting detected products.
    """
    values = np.asarray(wfall)
    ids = np.asarray(freq_id)
    if values.ndim != 3:
        raise ValueError("wfall must have shape (coarse_channel, polarization, time)")
    if ids.ndim != 1 or ids.shape[0] != values.shape[0]:
        raise ValueError("freq_id must be 1-D and match the coarse-channel axis")
    if not isinstance(upchan_factor, (int, np.integer)) or upchan_factor < 1:
        raise ValueError("upchan_factor must be a positive integer")
    if oversample not in SUPPORTED_OVERSAMPLES:
        raise ValueError(f"oversample must be one of {SUPPORTED_OVERSAMPLES}")
    if window not in SUPPORTED_WINDOWS:
        raise ValueError(f"window must be one of {SUPPORTED_WINDOWS}")

    # Match baseband_analysis: (channel, pol, time) -> (pol, time, channel).
    values = np.swapaxes(np.swapaxes(values, 0, 1), 1, 2)
    npol, nsamp, nchan = values.shape
    fft_size = int(upchan_factor * oversample)
    downfreq = int(oversample)
    hop = int(2 * upchan_factor)
    nblock = 0 if nsamp < fft_size else 1 + (nsamp - fft_size) // hop

    weights = _window_values(window, fft_size)
    raw_gain = _grouped_noise_gain(weights, downfreq)
    target_gain = float(upchan_factor)
    scale = float(np.sqrt(target_gain / raw_gain))
    if window == "rectangular" and oversample == 2:
        # Avoid even a roundoff-level multiply: this path is a regression
        # oracle for baseband_analysis 1.9.0's private implementation.
        scale = 1.0
        normalization = "package_exact"
    else:
        normalization = "exact_grouped_white_noise_power"

    spectrum = np.zeros(
        (npol, nblock, nchan * upchan_factor), dtype=np.complex64
    )
    channel_ids = np.zeros(nchan * upchan_factor, dtype=int)
    full_band = np.linspace(
        FREQ_TOP_MHZ, FREQ_BOTTOM_MHZ, upchan_factor * 1024
    )

    # Deliberately retain the package loop order and float32 destination.  It
    # makes the rectangular-2 result bit-for-bit comparable and bounds the
    # scientific change to framing, windowing, and documented normalization.
    for pol in range(npol):
        for block in range(nblock):
            start_sample = block * hop
            for channel in range(nchan):
                time_series = values[
                    pol, start_sample : start_sample + fft_size, channel
                ].copy()
                if window != "rectangular":
                    time_series *= weights
                transformed = fftshift(fft(time_series))
                transformed = transformed.reshape(upchan_factor, downfreq).mean(axis=1)
                if scale != 1.0:
                    transformed *= scale
                start_channel = channel * upchan_factor
                spectrum[
                    pol, block, start_channel : start_channel + upchan_factor
                ] = transformed
                channel_ids[start_channel : start_channel + upchan_factor] = np.arange(
                    upchan_factor * ids[channel],
                    upchan_factor * ids[channel] + upchan_factor,
                )

    metadata = {
        "implementation": "windowed_upchannel_v1",
        "window": window,
        "upchannel_factor": int(upchan_factor),
        "oversample": int(oversample),
        "fft_size": fft_size,
        "downfreq": downfreq,
        "hop_samples": hop,
        "frame_center_offset_samples": (fft_size - 1) / 2.0,
        "normalization": normalization,
        "raw_grouped_noise_gain": raw_gain,
        "normalization_scale": scale,
        "grouped_noise_gain": raw_gain * scale**2,
    }
    return spectrum, full_band[channel_ids], channel_ids, metadata
