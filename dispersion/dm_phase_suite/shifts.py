from __future__ import annotations

import numpy as np

K_DM_S_MHZ2 = 1.0 / 2.41e-4


def residual_delay_s(
    frequencies_mhz: np.ndarray,
    residual_dm: float,
    reference_frequency_mhz: float | None = None,
) -> np.ndarray:
    """Physical residual delay; positive DM makes low frequencies later."""
    freq = np.asarray(frequencies_mhz, dtype=float)
    if freq.ndim != 1 or freq.size == 0 or np.any(~np.isfinite(freq)) or np.any(freq <= 0):
        raise ValueError("frequencies_mhz must be a finite positive 1-D array")
    ref = float(np.max(freq) if reference_frequency_mhz is None else reference_frequency_mhz)
    return K_DM_S_MHZ2 * float(residual_dm) * (freq**-2 - ref**-2)


def fractional_zero_fill_shift(
    waterfall: np.ndarray,
    shifts_samples: np.ndarray,
    *,
    guard_samples: int = 8,
) -> np.ndarray:
    """Fractionally shift channels without allowing circular edge wrap.

    Positive shifts move samples later (right); negative shifts move them earlier.
    """
    wf = np.asarray(waterfall, dtype=float)
    shifts = np.asarray(shifts_samples, dtype=float)
    if wf.ndim != 2 or shifts.shape != (wf.shape[0],):
        raise ValueError("waterfall must be (frequency,time) and one shift is required per channel")
    if not np.all(np.isfinite(shifts)):
        raise ValueError("shifts must be finite")
    pad = int(np.ceil(np.max(np.abs(shifts)))) + max(1, int(guard_samples))
    padded = np.pad(wf, ((0, 0), (pad, pad)), mode="constant")
    fft = np.fft.fft(padded, axis=1)
    bins = np.fft.fftfreq(padded.shape[1])
    phase = np.exp(-2j * np.pi * shifts[:, None] * bins[None, :])
    shifted = np.fft.ifft(fft * phase, axis=1).real
    out = shifted[:, pad : pad + wf.shape[1]]
    # Numerical Fourier ringing beyond a hard edge is not data. Enforce exact
    # zero-fill for samples whose inverse-mapped coordinate lies outside input.
    ntime = wf.shape[1]
    for channel, shift in enumerate(shifts):
        if shift > 0:
            out[channel, : min(ntime, int(np.ceil(shift)))] = 0.0
        elif shift < 0:
            out[channel, max(0, ntime - int(np.ceil(-shift))) :] = 0.0
    return out


def dedisperse_residual(
    waterfall: np.ndarray,
    frequencies_mhz: np.ndarray,
    sample_time_s: float,
    residual_dm: float,
    *,
    guard_samples: int = 8,
) -> np.ndarray:
    if not np.isfinite(sample_time_s) or sample_time_s <= 0:
        raise ValueError("sample_time_s must be positive and finite")
    delays = residual_delay_s(frequencies_mhz, residual_dm)
    return fractional_zero_fill_shift(
        waterfall,
        -delays / float(sample_time_s),
        guard_samples=guard_samples,
    )
