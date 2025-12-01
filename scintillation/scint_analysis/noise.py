# noise_model.py
"""Characterise three flavours of off‑pulse data and synthesise matching noise

* **Intensity**  – strictly positive, exponential/Gamma, mean ≫ 0.
* **Flux‑Gaussian** – mean‑subtracted, nearly symmetric about 0.
* **Flux‑ShiftedGamma** – mean‑subtracted but **skewed** (support extends from
  a negative floor up to a long positive tail).  This arises when each channel’s
  *mean* has been removed from an originally Gamma‑like intensity stream.

Mode is auto‑detected.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy import signal, stats

log = logging.getLogger(__name__)

EPS = np.finfo(np.float32).tiny

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _acf_1d(x: NDArray[np.floating], nlags: int) -> NDArray[np.floating]:
    """NaN‑safe unbiased ACF up to *nlags* (O(N·nlags))."""
    x = np.asarray(x, dtype=np.float64)
    if np.isnan(x).any():
        x = np.where(np.isnan(x), np.nanmedian(x), x)
    x -= x.mean()
    var = np.dot(x, x)
    if var == 0:
        return np.zeros(nlags + 1)
    out = np.empty(nlags + 1)
    for k in range(nlags + 1):
        out[k] = np.dot(x[:-k or None], x[k:]) / var
    return out


def _lag1(acf: NDArray[np.floating]) -> float:
    return float(np.clip(acf[1], -0.99, 0.99))


def _robust_std(a: NDArray[np.floating], axis: Union[int, None] = None) -> NDArray[np.floating]:
    """σ ≈ 1.4826·MAD, NaN‑safe."""
    med = np.nanmedian(a, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(a - med), axis=axis)
    return 1.4826 * mad

# -----------------------------------------------------------------------------
# Dataclass
# -----------------------------------------------------------------------------

Kind = Literal["intensity", "flux_gauss", "flux_shiftedgamma"]

@dataclass
class NoiseDescriptor:
    kind: Kind
    nt: int
    nchan: int
    # Gaussian/shift parameters
    mu: float          # mean for intensity branch
    sigma: float       # stdev for Gaussian branch
    shift: float       # additive shift in shifted‑Gamma branch
    # Gamma params (intensity or shifted‑gamma)
    gamma_k: float
    gamma_theta: float
    # correlation
    phi_t: float
    phi_f: float
    # slow systematics (variance/gain curves, unit‑mean)
    g_t: NDArray[np.floating]
    b_f: NDArray[np.floating]

    # ------------------------------------------------------------------
    # (de)serialisation
    # ------------------------------------------------------------------
    def to_json(self, path: Union[str, Path]) -> None:
        d = asdict(self)
        d["g_t"], d["b_f"] = self.g_t.tolist(), self.b_f.tolist()
        Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "NoiseDescriptor":
        d = json.loads(Path(path).read_text())
        d["g_t"], d["b_f"] = np.asarray(d["g_t"], dtype=np.float32), np.asarray(d["b_f"], dtype=np.float32)
        return cls(**d)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # internal generators
    # ------------------------------------------------------------------
    def _correlated_white(self, rng: np.random.Generator, shape: tuple[int, int]) -> NDArray[np.float32]:
        w = rng.standard_normal(shape, dtype=np.float32)
        if abs(self.phi_t) > 0:
            w = signal.lfilter([1.0], [1.0, -self.phi_t], w, axis=0).astype(np.float32)
        if abs(self.phi_f) > 0:
            w = signal.lfilter([1.0], [1.0, -self.phi_f], w, axis=1).astype(np.float32)
        return w

    # ------------------------------------------------------------------
    # public sampler
    # ------------------------------------------------------------------
    def sample(self, *, seed: Optional[int] = None) -> NDArray[np.float32]:
        rng = np.random.default_rng(seed)

        if self.kind == "intensity":
            V = (rng.standard_normal((self.nt, self.nchan), dtype=np.float32) +
                 1j * rng.standard_normal((self.nt, self.nchan), dtype=np.float32))
            if abs(self.phi_t) > 0:
                V = signal.lfilter([1], [1, -self.phi_t], V, axis=0).astype(np.complex64)
            if abs(self.phi_f) > 0:
                V = signal.lfilter([1], [1, -self.phi_f], V, axis=1).astype(np.complex64)
            V *= np.sqrt(np.outer(self.g_t, self.b_f)).astype(np.float32)
            I = (V.real**2 + V.imag**2).astype(np.float32)
            if not np.isclose(self.gamma_k, 1.0, atol=0.05):
                G = rng.gamma(self.gamma_k, self.gamma_theta, size=I.shape).astype(np.float32)
                I *= G / (self.gamma_k * self.gamma_theta)
            return I * (self.mu / 2.0)

        if self.kind == "flux_gauss":
            W = self._correlated_white(rng, (self.nt, self.nchan)) * self.sigma
            W *= np.sqrt(np.outer(self.g_t, self.b_f)).astype(np.float32)
            return W

        # shifted‑Gamma flux
        pos = rng.gamma(self.gamma_k, self.gamma_theta, size=(self.nt, self.nchan)).astype(np.float32)
        pos -= self.shift  # centre back to have negative tail down to -shift
        pos *= np.sqrt(np.outer(self.g_t, self.b_f)).astype(np.float32)
        # add correlations in *additive* domain
        pos = self._correlated_white(rng, (self.nt, self.nchan)) * 0 + pos  # lfilter expects add domain; skip for now
        return pos

# -----------------------------------------------------------------------------
# NaN handling
# -----------------------------------------------------------------------------

def _inpaint(I: NDArray[np.floating]) -> NDArray[np.floating]:
    if not np.isnan(I).any():
        return I
    log.debug("NaNs detected in noise estimation array – inpainting with medians.")
    I = I.copy()
    row_med = np.nanmedian(I, axis=1)
    col_med = np.nanmedian(I, axis=0)
    global_med = np.nanmedian(I)
    row_med = np.where(np.isnan(row_med), global_med, row_med)
    col_med = np.where(np.isnan(col_med), global_med, col_med)
    inds = np.where(np.isnan(I))
    I[inds] = 0.5 * (row_med[inds[0]] + col_med[inds[1]])
    return I

# -----------------------------------------------------------------------------
# Main estimator
# -----------------------------------------------------------------------------

def estimate_noise_descriptor(I: NDArray[np.floating], nlags: int = 50) -> NoiseDescriptor:
    """Analyse *I* and return a matching *NoiseDescriptor* (auto‑mode)."""
    if I.ndim != 2:
        raise ValueError("Input must be 2‑D time × freq")
    I = _inpaint(np.asarray(I, dtype=np.float64))
    nt, nchan = I.shape

    # decide branch
    min_, max_, med = np.min(I), np.max(I), np.median(I)
    if min_ >= 0 and med > 1e-3:  # intensity
        kind: Kind = "intensity"
    else:
        # compute skewness
        skew = float(stats.skew(I, axis=None, nan_policy="omit"))
        if abs(skew) < 0.2:
            kind = "flux_gauss"
        else:
            kind = "flux_shiftedgamma"

    # ---------- branch‑specific params ----------
    if kind == "intensity":
        mu = float(I.mean())
        var = I.var(ddof=0)
        var = max(float(var), EPS)
        gamma_k, gamma_theta = mu**2 / var, var / mu
        sigma = 0.0
        shift = 0.0
        g_t = np.clip(np.nanmedian(I, axis=1), EPS, None).astype(np.float32)
        b_f = np.clip(np.nanmedian(I, axis=0), EPS, None).astype(np.float32)
    elif kind == "flux_gauss":
        mu = 0.0
        sigma = float(_robust_std(I))
        gamma_k = gamma_theta = 0.0
        shift = 0.0
        g_t = np.clip(_robust_std(I, axis=1), EPS, None).astype(np.float32)
        b_f = np.clip(_robust_std(I, axis=0), EPS, None).astype(np.float32)
    else:  # flux_shiftedgamma
        shift = abs(min_) + EPS
        gamma_data = I + shift  # now positive
        mu = 0.0
        sigma = 0.0
        mean, var = gamma_data.mean(), gamma_data.var(ddof=0)
        var = max(float(var), EPS)
        gamma_k, gamma_theta = mean**2 / var, var / mean
        g_t = np.clip(_robust_std(gamma_data, axis=1), EPS, None).astype(np.float32)
        b_f = np.clip(_robust_std(gamma_data, axis=0), EPS, None).astype(np.float32)

    # normalise curves
    g_t /= g_t.mean()
    b_f /= b_f.mean()

    # correlations
    acf_t = np.mean([_acf_1d(I[:, j], nlags) for j in range(nchan)], axis=0)
    acf_f = np.mean([_acf_1d(I[i, :], nlags) for i in range(nt)], axis=0)
    phi_t, phi_f = _lag1(acf_t), _lag1(acf_f)

    return NoiseDescriptor(kind, nt, nchan, mu, sigma, shift, gamma_k, gamma_theta, phi_t, phi_f, g_t, b_f)

# -----------------------------------------------------------------------------
# Sanity test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    raw = rng.gamma(shape=2.0, scale=1.0, size=(1024, 64)).astype(np.float32)
    residual = raw - np.mean(raw, axis=0)  # skewed around 0
    desc = estimate_noise_descriptor(residual)
    fake = desc.sample(seed=42)
    print("Detected", desc.kind, "Skew(original)", stats.skew(residual), "Skew(fake)", stats.skew(fake))
