"""P3′ delay-domain matched (optimal quadratic) Δν_d estimator.

Predeclared in ``experiment-chime-scint-p3-optimal-estimator.md`` (Faber2026)
and amended by the owner-sanctioned §P3′ amendment (2026-07-15): the P2 S2
split-ratio construction is kept unchanged; the statistic downstream of the
ratio spectra is replaced by a matched filter in the delay domain.

Chain (all frozen):

1. Split ratio fields ``r1(ν), r2(ν)`` — P2's S2 construction verbatim
   (disjoint interleaved on/off time halves, per-split polarization mean), so
   the instrumental common mode ``g(ν)`` cancels algebraically and the noise
   is independent between the two splits.
2. Transform — **global** mean removal only (the P3′ amendment retires the
   64-channel block demeaning: it erased scintles wider than one coarse
   block, which is exactly the Gate-0 detectability window; the ratio already
   provides the common-mode protection demeaning existed for), NaN→0 (the
   RFI mask acts as a window shared with the Monte-Carlo templates), full-band
   rfft, DC bin dropped.
3. Cross power ``P(k) = Re[R1(k) conj R2(k)]`` — noise-bias-free at every
   delay because the splits carry independent noise.
4. Matched amplitude against Monte-Carlo templates ``T(k; Δν_d)`` =
   mean delay power of unit-variance Lorentzian-ACF gain fields pushed
   through the identical mask + transform (this bakes in every window/mask
   transfer, the T4 requirement), with empirical weights ``w = 1/Var_null(k)``
   and the frozen delay-bin exclusion ``k < KMIN`` (the envelope control: an
   intrinsic burst spectral envelope — common to both polarizations and both
   time halves, hence invisible to the null campaign — lives in the lowest
   delay bins; structure smoother than ~12.8 MHz is excised at a measured
   ~7 % SNR cost):

       a_hat = Σ_k w P T / Σ_k w T²,   σ_a = (Σ_k w T²)^(−1/2)

5. Detection score per grid point ``z = (a_hat − ⟨a_hat⟩_null)/σ_null`` with
   the null mean and σ taken from the calibration null half; the trials-
   corrected threshold comes from the evaluation null half (seeds are frozen
   in the experiment harness).

Blinding: identical structural guard to Route B — no function here reads a
time sample inside ``routeb_voltage.ON_PULSE_GUARD`` unless the caller passes
``allow_unblind=True`` (the orchestrator's one-shot unblinding only).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from . import routeb_voltage as rb

# ---- frozen P3' hyperparameters (record §P3' amendment) ----------------------
KMIN = 11  # delay bins k < KMIN excluded (envelope control; ~12.8 MHz scale)
N_SCAN = 25  # log grid over the Gate-0 prior sweep
DNU_SCAN_KHZ = np.geomspace(20.0, 400.0, N_SCAN)
N_TEMPLATE = 200  # Monte-Carlo realizations per template
TEMPLATE_SEED_BASE = 750_000  # + 1000*grid_index + j ; disjoint from G1/G2/0b
N_VAR_BANDS = 48  # log-spaced smoothing bands for Var_null(k)


def split_ratio_fields(
    dynamic_by_pol: Sequence[np.ndarray],
    on_samples,
    off_samples,
    *,
    on_gain: np.ndarray | None = None,
    allow_unblind: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """P2's S2 split-ratio fields (pol-mean halves), with the blinding guard.

    Interleaved (``::2``/``1::2``) halves keep the two splits exchangeable
    under slow baseline drift; the returned fields carry independent receiver
    noise and the shared signal ``f_b·m·δ(ν)`` (constant offsets die in the
    global demeaning of the transform).
    """
    on = rb.assert_offpulse_samples(on_samples, allow_unblind=allow_unblind, name="on_samples")
    off = rb.assert_offpulse_samples(off_samples, allow_unblind=allow_unblind, name="off_samples")
    on_a, on_b = rb._split_halves(on)
    off_a, off_b = rb._split_halves(off)

    def half(on_h: np.ndarray, off_h: np.ndarray) -> np.ndarray:
        per_pol = [
            rb._ratio(rb._mean_frame(dyn, on_h, on_gain), rb._mean_frame(dyn, off_h, None))
            for dyn in dynamic_by_pol
        ]
        return 0.5 * (per_pol[0] + per_pol[1])

    return half(on_a, off_a), half(on_b, off_b)


def delay_transform(field: np.ndarray) -> np.ndarray:
    """Global-demeaned rfft with the DC bin dropped (P3′: no block demeaning)."""
    x = np.asarray(field, dtype=float)
    x = x - np.nanmean(x)
    x = np.where(np.isfinite(x), x, 0.0)
    return np.fft.rfft(x)[1:]


def cross_power(field1: np.ndarray, field2: np.ndarray) -> np.ndarray:
    return np.real(delay_transform(field1) * np.conj(delay_transform(field2)))


def smooth_variance(variance: np.ndarray, n_bands: int = N_VAR_BANDS) -> np.ndarray:
    """Log-band-averaged Var(k): per-bin sample variances over O(100) nulls are
    too noisy for stable 1/Var weights across ~11k bins; band-averaging is
    conservative (can only understate the achievable SNR)."""
    n = variance.size
    if n <= n_bands:
        return np.asarray(variance, dtype=float)
    edges = np.unique(np.geomspace(1, n, n_bands + 1).astype(int))
    smoothed = np.empty(n, dtype=float)
    for lo, hi in zip(edges[:-1], edges[1:]):
        hi = max(hi, lo + 1)
        smoothed[lo - 1 : hi] = variance[lo - 1 : hi].mean()
    return smoothed


def lorentzian_template(
    dnu_khz: float,
    grid_index: int,
    *,
    n_channels: int,
    channel_width_khz: float,
    good_mask: np.ndarray,
    n_realizations: int = N_TEMPLATE,
    seed_base: int = TEMPLATE_SEED_BASE,
) -> np.ndarray:
    """T(k) = <|D(k)|²> over unit-variance Lorentzian-ACF gain fields through
    the identical mask + transform. Per-realization seeds keep the bank
    auditable; the rfft is batched for speed."""
    width_channels = float(dnu_khz) / float(channel_width_khz)
    bad = ~np.asarray(good_mask, dtype=bool)
    fields = np.empty((n_realizations, n_channels), dtype=float)
    for j in range(n_realizations):
        rng = np.random.default_rng(seed_base + 1000 * grid_index + j)
        fields[j] = rb.lorentzian_gain_field(
            rng, n_channels=n_channels, width_channels=width_channels
        )
    fields[:, bad] = 0.0  # matches NaN->0 in delay_transform for masked bins
    fields -= fields.mean(axis=1, keepdims=True)
    spectra = np.fft.rfft(fields, axis=1)[:, 1:]
    return (np.abs(spectra) ** 2).mean(axis=0)


@dataclass
class MatchedScan:
    """Frozen matched-filter scan over the Δν_d grid.

    ``variance`` is the smoothed null Var(k) (calibration half); ``templates``
    is ``(N_SCAN, n_bins)``. Null mean/σ per grid point are attached by
    ``calibrate`` and used by ``zscan``.
    """

    dnu_khz: np.ndarray
    templates: np.ndarray
    variance: np.ndarray
    kmin: int = KMIN
    null_mean: np.ndarray | None = field(default=None)
    null_sigma: np.ndarray | None = field(default=None)

    def __post_init__(self) -> None:
        sl = slice(self.kmin - 1, None)  # bin index 0 is delay bin k=1
        weights = 1.0 / np.asarray(self.variance, dtype=float)[sl]
        t_cut = np.asarray(self.templates, dtype=float)[:, sl]
        self._slice = sl
        self._weighted_templates = weights[None, :] * t_cut  # (N_SCAN, n_kept)
        self._denominator = np.sum(self._weighted_templates * t_cut, axis=1)
        self.sigma_analytic = self._denominator**-0.5

    def amplitudes(self, power: np.ndarray) -> np.ndarray:
        """a_hat(Δν_d) for one cross-power spectrum."""
        return self._weighted_templates @ np.asarray(power, dtype=float)[self._slice] / self._denominator

    def calibrate(self, null_powers: np.ndarray) -> None:
        """Attach the null mean and empirical σ per grid point (calibration
        null half). The mean subtraction removes the small positive offset the
        split cross power carries on real data (~1σ, Gate 0b)."""
        a_null = np.array([self.amplitudes(p) for p in np.asarray(null_powers)])
        self.null_mean = a_null.mean(axis=0)
        self.null_sigma = a_null.std(axis=0, ddof=1)

    def zscan(self, power: np.ndarray) -> dict:
        """Null-calibrated detection scan: z per grid point, max, argmax."""
        if self.null_mean is None or self.null_sigma is None:
            raise RuntimeError("MatchedScan.calibrate must run before zscan")
        a_hat = self.amplitudes(power)
        z = (a_hat - self.null_mean) / self.null_sigma
        peak = int(np.argmax(z))
        return {
            "a_hat": a_hat,
            "z": z,
            "z_max": float(z[peak]),
            "dnu_khz_argmax": float(self.dnu_khz[peak]),
            "argmax_index": peak,
        }

    def nearest_grid_index(self, dnu_khz: float) -> int:
        return int(np.argmin(np.abs(np.log(self.dnu_khz) - np.log(float(dnu_khz)))))
