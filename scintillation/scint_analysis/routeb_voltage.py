"""Route-B common-mode-immune Δν_d statistics for CHIME scintillation.

Every earlier CHIME route worked on a quantity in which the instrumental
common mode ``g(ν)`` survives.  The retained detected intensity is

    I_p(ν, t) = g(ν, t) · [ f_p · B(t) · s(ν) + N_p(ν, t) ]

with ``g`` the common instrumental response (measured amplitude
``A = 0.586``, width ``≈ 35 kHz``; multiplies *everything*), ``s(ν)`` the
scintillation gain (common to both polarizations, multiplies *only* the
source term), ``B(t)`` the burst profile, ``f_p`` the polarization flux and
``N_p`` independent receiver noise.  A polarization cross-ACF of ``I_p``
retains ``g`` because ``g`` is shared by both pols
(``research-chime-scint-instrumental-common-mode.md``).

Route B forms the on/off *ratio* first,

    R_p(ν) = ⟨I_p⟩_on / ⟨I_p⟩_off − 1 ,

in which ``g(ν)`` cancels **algebraically** whenever it is stable across the
frame (⟨I_p⟩_on and ⟨I_p⟩_off both carry the same ``g(ν)`` factor, so it
divides out exactly).  The measured on/off cross-ACF identity (±0.017,
research record) is the direct evidence of that frame-scale stability.  What
survives the ratio is ``f_b · s(ν)`` plus polarization-independent radiometer
structure; the pol0×pol1 cross-ACF of ``R_p`` then removes the independent
radiometer term, leaving the common scintillation Lorentzian.

Three statistics share the ratio construction (all predeclared in
``experiment-chime-scint-routeb-voltage.md``):

* ``S1`` on/off ratio cross-ACF: ``cross(R_p0, R_p1)``.
* ``S2`` time-split ratio cross-ACF: split on- and off-pulse into two time
  halves, cross the two half-ratios.  Independent noise between the halves
  kills the noise bias at every lag, at a √2 sensitivity cost.
* ``S3`` voltage variant: the same ratio built from per-pol ``|V_p|²`` on the
  complex fine-channel voltages before Stokes aggregation, preserving the
  P1 worker's grouped-bin noise normalization.

Blinding: the on-pulse window is samples 250–350.  No function here computes a
statistic on a sample that lands in that window unless the caller passes
``allow_unblind=True`` — the one-shot unblinded computation is the
orchestrator's, performed only after gates G1 and G2 pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .cross_acf import CrossACF, blockwise_cross_acf, fit_cross_lorentzian

# On-pulse window (samples); half-open [lo, hi).  Any statistic that touches a
# sample in this range while blinded is refused.
ON_PULSE_GUARD: tuple[int, int] = (250, 350)


class BlindingError(RuntimeError):
    """Raised when a Route-B statistic would read the blinded on-pulse window."""


def samples_from_window(window: tuple[int, int]) -> np.ndarray:
    """Expand a half-open ``(lo, hi)`` sample window to an index array."""
    lo, hi = int(window[0]), int(window[1])
    return np.arange(lo, hi, dtype=int)


def assert_offpulse_samples(
    samples, *, allow_unblind: bool = False, name: str = "samples"
) -> np.ndarray:
    """Refuse a set of time-sample indices that overlaps the on-pulse window.

    The blinding boundary is structural: the on-pulse window ``ON_PULSE_GUARD``
    may only be read when the caller explicitly asserts ``allow_unblind=True``
    (the orchestrator's one-shot unblinding).  ``samples`` may be an index
    array or a ``(lo, hi)`` window tuple.
    """
    index = np.asarray(samples, dtype=int).ravel()
    inside = (index >= ON_PULSE_GUARD[0]) & (index < ON_PULSE_GUARD[1])
    if inside.any() and not allow_unblind:
        raise BlindingError(
            f"{name} overlaps the blinded on-pulse window {ON_PULSE_GUARD}: "
            f"{int(inside.sum())} sample(s) in [{ON_PULSE_GUARD[0]}, "
            f"{ON_PULSE_GUARD[1]}). Pass allow_unblind=True only for the "
            "sanctioned one-shot unblinded computation."
        )
    return index


def row_nanmean(values: np.ndarray) -> np.ndarray:
    """Per-channel time mean, NaN channels -> NaN, no all-NaN-slice warning."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    return np.divide(
        np.nansum(np.where(finite, values, 0.0), axis=1),
        count,
        out=np.full(values.shape[0], np.nan),
        where=count > 0,
    )


def _ratio(on_mean: np.ndarray, off_mean: np.ndarray) -> np.ndarray:
    """``on/off − 1`` with a positive-denominator guard (NaN where undefined)."""
    on = np.asarray(on_mean, dtype=float)
    off = np.asarray(off_mean, dtype=float)
    valid = np.isfinite(on) & np.isfinite(off) & (off > 0)
    ratio = np.full(on.shape, np.nan, dtype=float)
    ratio[valid] = on[valid] / off[valid] - 1.0
    return ratio


def lorentzian_gain_field(
    rng: np.random.Generator, *, n_channels: int, width_channels: float
) -> np.ndarray:
    """Unit-variance Gaussian field with a Lorentzian frequency ACF.

    The circular covariance is ``C(d) = 1 / (1 + (d / width_channels)²)`` so
    that ``C(width_channels) = 1/2`` — ``width_channels`` is the **HWHM in fine
    channels**, matching ``fit_cross_lorentzian`` (whose ``width`` parameter is
    the same HWHM ``γ`` reported directly as Δν_d).  Spectral synthesis:
    filter white noise by ``sqrt(FFT(C))``, then normalize to unit variance so
    the injected modulation ``m`` is exactly the fractional RMS of the gain.

    This reproduces the injection convention of the B4/C1 harness
    (``validate_freya_highband_crossacf._stationary_lorentzian``) so Route-B
    calibration is comparable to the earlier routes.
    """
    n = int(n_channels)
    distances = np.minimum(np.arange(n), n - np.arange(n))
    covariance = 1.0 / (1.0 + (distances / float(width_channels)) ** 2)
    power = np.maximum(np.real(np.fft.fft(covariance)), 0.0)
    white = np.fft.fft(rng.normal(size=n))
    sample = np.real(np.fft.ifft(white * np.sqrt(power)))
    return (sample - sample.mean()) / sample.std()


@dataclass(frozen=True)
class RouteBResult:
    """A Route-B statistic evaluation: the cross-ACF, its fit, and the ratios."""

    statistic: str
    cross: CrossACF
    fit: dict | None
    ratios: list[np.ndarray]


def _fit(cross: CrossACF, *, channel_width_mhz, first_lag_bin, fit_max_mhz, block_length):
    return fit_cross_lorentzian(
        cross,
        channel_width_mhz=channel_width_mhz,
        first_lag_bin=first_lag_bin,
        fit_max_mhz=fit_max_mhz,
        block_length=block_length,
    )


def _mean_frame(dyn: np.ndarray, samples: np.ndarray, gain: np.ndarray | None) -> np.ndarray:
    frame = dyn[:, samples]
    if gain is not None:
        frame = frame * gain[:, None]
    return row_nanmean(frame)


def s1_ratio_cross_acf(
    dynamic_by_pol: Sequence[np.ndarray],
    on_samples,
    off_samples,
    block_ids: np.ndarray,
    *,
    channel_width_mhz: float,
    on_gain: np.ndarray | None = None,
    max_lag_bins: int = 40,
    first_lag_bin: int = 2,
    fit_max_mhz: float = 0.25,
    block_length: int | None = 64,
    allow_unblind: bool = False,
) -> RouteBResult:
    """S1 — on/off ratio cross-ACF between the two polarizations.

    ``dynamic_by_pol`` is ``[I_pol0, I_pol1]``, each ``(n_channels, n_times)``.
    ``on_samples``/``off_samples`` are time-sample index arrays (or windows).
    ``on_gain`` is an optional per-channel multiplicative gain applied to the
    on-window frames — the injection hook for G1 (a real burst carries the
    common mode ``g``; the injected gain rides on the same ``g``-bearing
    off-pulse data, so the ratio still cancels ``g``).
    """
    on = assert_offpulse_samples(on_samples, allow_unblind=allow_unblind, name="on_samples")
    off = assert_offpulse_samples(off_samples, allow_unblind=allow_unblind, name="off_samples")
    ratios = [
        _ratio(_mean_frame(dyn, on, on_gain), _mean_frame(dyn, off, None))
        for dyn in dynamic_by_pol
    ]
    cross = blockwise_cross_acf(
        ratios[0],
        ratios[1],
        block_ids,
        normalization_left=1.0,
        normalization_right=1.0,
        max_lag_bins=max_lag_bins,
    )
    fit = _fit(
        cross,
        channel_width_mhz=channel_width_mhz,
        first_lag_bin=first_lag_bin,
        fit_max_mhz=fit_max_mhz,
        block_length=block_length,
    )
    return RouteBResult("S1", cross, fit, ratios)


def _split_halves(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split an index array into two disjoint interleaved halves.

    Interleaving (``::2`` / ``1::2``) rather than a contiguous cut keeps the two
    halves statistically exchangeable even if the off-pulse baseline drifts
    slowly across the window.
    """
    return samples[0::2], samples[1::2]


def s2_time_split_cross_acf(
    dynamic_by_pol: Sequence[np.ndarray],
    on_samples,
    off_samples,
    block_ids: np.ndarray,
    *,
    channel_width_mhz: float,
    on_gain: np.ndarray | None = None,
    max_lag_bins: int = 40,
    first_lag_bin: int = 2,
    fit_max_mhz: float = 0.25,
    block_length: int | None = 64,
    allow_unblind: bool = False,
) -> RouteBResult:
    """S2 — time-split ratio cross-ACF.

    Split the on- and off-pulse windows into two disjoint time halves, form a
    half-ratio ``R^{(h)}_p`` in each, average the two polarizations into
    ``S^{(h)}(ν) = (R^{(h)}_p0 + R^{(h)}_p1) / 2``, and cross ``S^{(1)}`` with
    ``S^{(2)}``.  The noise in the two halves is independent, so its cross-ACF
    has zero expectation at **every** lag (not just the noise-dominated lag 0)
    — this removes the noise bias S1 carries, at a √2 sensitivity cost.  The
    common scintillation ``δ(ν)`` is shared by both halves, so the pol-mean
    keeps the recovered amplitude on the same ``(f_b·m)²`` scale as S1.
    """
    on = assert_offpulse_samples(on_samples, allow_unblind=allow_unblind, name="on_samples")
    off = assert_offpulse_samples(off_samples, allow_unblind=allow_unblind, name="off_samples")
    on_a, on_b = _split_halves(on)
    off_a, off_b = _split_halves(off)

    def half(on_h: np.ndarray, off_h: np.ndarray) -> np.ndarray:
        per_pol = [
            _ratio(_mean_frame(dyn, on_h, on_gain), _mean_frame(dyn, off_h, None))
            for dyn in dynamic_by_pol
        ]
        return 0.5 * (per_pol[0] + per_pol[1])

    field1 = half(on_a, off_a)
    field2 = half(on_b, off_b)
    cross = blockwise_cross_acf(
        field1,
        field2,
        block_ids,
        normalization_left=1.0,
        normalization_right=1.0,
        max_lag_bins=max_lag_bins,
    )
    fit = _fit(
        cross,
        channel_width_mhz=channel_width_mhz,
        first_lag_bin=first_lag_bin,
        fit_max_mhz=fit_max_mhz,
        block_length=block_length,
    )
    return RouteBResult("S2", cross, fit, [field1, field2])


def voltage_intensity(voltage: np.ndarray) -> np.ndarray:
    """Per-pol detected intensity ``|V|²`` from complex fine-channel voltages."""
    return np.abs(np.asarray(voltage)) ** 2


def s3_voltage_cross_acf(
    voltage_by_pol: Sequence[np.ndarray],
    on_samples,
    off_samples,
    block_ids: np.ndarray,
    *,
    channel_width_mhz: float,
    on_gain: np.ndarray | None = None,
    max_lag_bins: int = 40,
    first_lag_bin: int = 2,
    fit_max_mhz: float = 0.25,
    block_length: int | None = 64,
    allow_unblind: bool = False,
) -> RouteBResult:
    """S3 — ratio cross-ACF on per-pol ``|V_p|²`` before Stokes aggregation.

    Structurally S1 applied to the voltage-derived per-pol intensity.  Using
    ``|V_p|²`` computed from the complex fine-channel voltages (rather than the
    retained Stokes-I product) preserves the exact grouped-bin noise
    normalization of the P1 voltage worker on h17, which allows exact noise-bias
    bookkeeping; it is expected to agree with S1 in the mean.
    """
    intensity = [voltage_intensity(v) for v in voltage_by_pol]
    result = s1_ratio_cross_acf(
        intensity,
        on_samples,
        off_samples,
        block_ids,
        channel_width_mhz=channel_width_mhz,
        on_gain=on_gain,
        max_lag_bins=max_lag_bins,
        first_lag_bin=first_lag_bin,
        fit_max_mhz=fit_max_mhz,
        block_length=block_length,
        allow_unblind=allow_unblind,
    )
    return RouteBResult("S3", result.cross, result.fit, result.ratios)


STATISTICS = {
    "S1": s1_ratio_cross_acf,
    "S2": s2_time_split_cross_acf,
    "S3": s3_voltage_cross_acf,
}
