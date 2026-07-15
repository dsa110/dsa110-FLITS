"""P4 intrinsic-envelope models for the CHIME ratio spectrum (exploratory).

Predeclared in ``experiment-chime-scint-p4-envelope-model.md`` (Faber2026):
freya's on-pulse ratio spectrum is dominated by broad intrinsic spectral
structure (P3′ unblinding: â ≈ 10⁻³, ~11× the scintillation ceiling). These
models estimate the smooth envelope ``E(ν)`` so the multiplicative residual

    r(ν) = R(ν)/E(ν) − 1

can be searched for a scintle with the P3′ matched scan. Three frozen
families:

* **M1 spline** — cubic least-squares spline with knots every Λ MHz.
* **M2 Gaussian process** — squared-exponential kernel, length scale ℓ,
  posterior mean fit on a 4× decimated channel grid (the kernel factorization
  depends only on the grid and ℓ, so it is precomputed once per scale and
  each fit is a solve + chunked cross-covariance product).
* **M3 delay low-pass** — keep delay bins ``k < k_env`` (structure smoother
  than ~140.8/k_env MHz) as the envelope estimate.

The scale parameter trades envelope leakage (too little smoothing → the
residual keeps envelope structure) against scintle absorption (too much →
the model eats the signal). Neither side is judged here: the E2 injection
calibration measures both through the full chain, with templates rebuilt
through the same model+subtract transfer (the P3/Gate-0b lesson).

Everything here is post-unblinding, owner-sanctioned 2026-07-15; callers
read the on-pulse window explicitly via the Route-B blinding flag.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import LSQUnivariateSpline
from scipy.linalg import cho_factor, cho_solve

ENVELOPE_CLIP_PERCENTILE = 5.0  # record: clip E below its 5th positive percentile
GP_DECIMATION = 4


def _finite(nu: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(values)
    return np.asarray(nu, dtype=float)[ok], np.asarray(values, dtype=float)[ok]


def _valid_knots(x: np.ndarray, candidates: np.ndarray, min_points: int = 8) -> np.ndarray:
    """Drop candidate knots until every knot span holds >= min_points data
    points (Schoenberg-Whitney for k=3); knots inside masked gaps (e.g. the
    30 MHz LTE hole) are skipped and the spline continues smoothly across."""
    keep: list[float] = []
    last = x[0]
    for t in candidates:
        if np.count_nonzero((x > last) & (x < t)) >= min_points:
            keep.append(float(t))
            last = t
    while keep and np.count_nonzero(x > keep[-1]) < min_points:
        keep.pop()
    return np.asarray(keep)


def fit_spline(nu_mhz: np.ndarray, spectrum: np.ndarray, scale_mhz: float) -> np.ndarray:
    """M1: cubic least-squares spline with interior knots every ``scale_mhz``."""
    x, y = _finite(nu_mhz, spectrum)
    order = np.argsort(x)
    x, y = x[order], y[order]
    knots = np.arange(x[0] + scale_mhz, x[-1] - scale_mhz / 2, scale_mhz)
    knots = _valid_knots(x, knots[(knots > x[3]) & (knots < x[-4])])
    spline = LSQUnivariateSpline(x, y, knots, k=3)
    return spline(np.asarray(nu_mhz, dtype=float))


class GPEnvelope:
    """M2: squared-exponential GP posterior mean, factorized once per scale.

    ``noise_variance`` is the per-channel radiometer variance of the ratio
    spectrum (measured off-pulse); it regularizes the kernel solve. The
    posterior mean is linear in the data, so per-spectrum cost after the
    one-time Cholesky is one solve plus a chunked cross-covariance product.
    """

    def __init__(
        self,
        nu_mhz: np.ndarray,
        good_mask: np.ndarray,
        scale_mhz: float,
        noise_variance: float,
        decimation: int = GP_DECIMATION,
    ) -> None:
        nu = np.asarray(nu_mhz, dtype=float)
        good = np.asarray(good_mask, dtype=bool)
        self.train_index = np.flatnonzero(good)[::decimation]
        x = nu[self.train_index]
        d2 = (x[:, None] - x[None, :]) ** 2
        kernel = np.exp(-0.5 * d2 / scale_mhz**2)
        kernel[np.diag_indices_from(kernel)] += max(noise_variance, 1e-12)
        self._cho = cho_factor(kernel, lower=True)
        self._train_x = x
        self._nu = nu
        self._scale = float(scale_mhz)

    def fit(self, spectrum: np.ndarray) -> np.ndarray:
        """Posterior mean; accepts one spectrum or a batch (n_spectra, n_chan).

        The posterior mean is linear in the data, so a batch shares the single
        Cholesky factorization: one multi-RHS solve + one chunked matmul.
        """
        x = np.asarray(spectrum, dtype=float)
        squeeze = x.ndim == 1
        y = np.atleast_2d(x)[:, self.train_index]
        fill = np.nanmean(y, axis=1, keepdims=True)
        y = np.where(np.isfinite(y), y, fill)
        alpha = cho_solve(self._cho, (y - fill).T)  # (n_train, n_spectra)
        envelope = np.empty((self._nu.size, alpha.shape[1]), dtype=float)
        chunk = 2048
        for lo in range(0, self._nu.size, chunk):
            hi = min(lo + chunk, self._nu.size)
            d2 = (self._nu[lo:hi, None] - self._train_x[None, :]) ** 2
            envelope[lo:hi] = np.exp(-0.5 * d2 / self._scale**2) @ alpha
        out = envelope.T + fill
        return out[0] if squeeze else out


def fit_delay_lowpass(spectrum: np.ndarray, k_env: int) -> np.ndarray:
    """M3: envelope = inverse transform of delay bins ``k < k_env`` (DC kept).

    Accepts one spectrum or a batch (last axis = channels).
    """
    x = np.asarray(spectrum, dtype=float)
    mean = np.nanmean(x, axis=-1, keepdims=True)
    filled = np.where(np.isfinite(x), x, mean)
    transform = np.fft.rfft(filled - mean, axis=-1)
    transform[..., int(k_env):] = 0.0
    return np.fft.irfft(transform, n=x.shape[-1], axis=-1) + mean


def residual(spectrum: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    """Multiplicative residual ``R/E − 1`` with the frozen clip rule.

    ``E`` below the 5th percentile of its positive values (per spectrum) is
    masked (NaN residual) rather than divided — a scintle multiplies the
    envelope, so only the well-measured envelope supports the ratio.
    Accepts one spectrum or a batch (last axis = channels).
    """
    R = np.asarray(spectrum, dtype=float)
    E = np.asarray(envelope, dtype=float)
    if R.ndim == 1:
        positive = E[np.isfinite(E) & (E > 0)]
        floor = np.percentile(positive, ENVELOPE_CLIP_PERCENTILE)
        ok = np.isfinite(R) & np.isfinite(E) & (E >= floor)
        out = np.full(R.shape, np.nan)
        out[ok] = R[ok] / E[ok] - 1.0
        return out
    return np.stack([residual(r, e) for r, e in zip(R, E)])


FAMILY_SCALES = {
    "M1_spline": (0.5, 1.0, 2.0, 5.0, 10.0),  # knot spacing, MHz
    "M2_gp": (0.5, 1.0, 2.0, 5.0, 10.0),  # SE length scale, MHz
    "M3_delaycut": (25, 50, 100, 200),  # k_env (smoother than 140.8/k MHz)
}


class EnvelopeChain:
    """One (family, scale) model+subtract chain, reusable across spectra."""

    def __init__(
        self,
        family: str,
        scale,
        nu_mhz: np.ndarray,
        good_mask: np.ndarray,
        noise_variance: float,
    ) -> None:
        if family not in FAMILY_SCALES:
            raise ValueError(f"unknown family {family!r}")
        self.family = family
        self.scale = scale
        self._nu = np.asarray(nu_mhz, dtype=float)
        self._gp = (
            GPEnvelope(nu_mhz, good_mask, float(scale), noise_variance)
            if family == "M2_gp"
            else None
        )

    def envelope(self, spectrum: np.ndarray) -> np.ndarray:
        """Fit the envelope; accepts one spectrum or a batch (rows = spectra)."""
        x = np.asarray(spectrum, dtype=float)
        if self.family == "M1_spline":
            if x.ndim == 1:
                return fit_spline(self._nu, x, float(self.scale))
            return np.stack([fit_spline(self._nu, row, float(self.scale)) for row in x])
        if self.family == "M2_gp":
            return self._gp.fit(x)
        return fit_delay_lowpass(x, int(self.scale))

    def residual(self, spectrum: np.ndarray) -> np.ndarray:
        return residual(spectrum, self.envelope(spectrum))
