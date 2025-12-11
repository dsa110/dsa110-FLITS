# noise_model.py
"""Toolkit to characterise and re‑synthesise off‑pulse noise in a
radio‑telescope dynamic spectrum, now **robust to NaNs**.

The key entry points are:

>>> desc = estimate_noise_descriptor(I)
>>> I_fake = desc.sample(seed=42)

where `I` is a 2‑D NumPy array (time × frequency).  Any NaNs are
interpreted as masked pixels and automatically in‑painted using the
row/column medians so subsequent statistics remain finite.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import signal

# -----------------------------------------------------------------------------
# Constants & helpers
# -----------------------------------------------------------------------------

EPS = np.finfo(np.float32).tiny  # ≈1.18 × 10⁻³⁸


def _acf_1d(x: NDArray[np.floating], nlags: int) -> NDArray[np.floating]:
    """Unbiased autocorrelation (slow O(N·nlags), but NaN‑safe)."""
    x = np.asarray(x, dtype=np.float64)
    if np.isnan(x).any():
        # Replace NaNs by the median of finite values in *this* vector
        med = np.nanmedian(x)
        x = np.where(np.isnan(x), med, x)
    x -= x.mean()
    var = np.dot(x, x)
    if var == 0:
        return np.zeros(nlags + 1)
    acf = np.empty(nlags + 1)
    for k in range(nlags + 1):
        acf[k] = np.dot(x[: x.size - k], x[k:]) / var
    return acf


def _lag1_coeff(acf: NDArray[np.floating]) -> float:
    return float(np.clip(acf[1], -0.99, 0.99))

# -----------------------------------------------------------------------------
# Descriptor
# -----------------------------------------------------------------------------

@dataclass
class NoiseDescriptor:
    nt: int
    nchan: int
    gamma_k: float
    gamma_theta: float
    phi_t: float
    phi_f: float
    g_t: NDArray[np.floating]
    b_f: NDArray[np.floating]

    # ---------- (de)serialisation ----------
    def to_json(self, path: str | Path) -> None:
        d = asdict(self)
        d["g_t"], d["b_f"] = self.g_t.tolist(), self.b_f.tolist()
        Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "NoiseDescriptor":
        d = json.loads(Path(path).read_text())
        d["g_t"], d["b_f"] = np.asarray(d["g_t"], dtype=np.float32), np.asarray(
            d["b_f"], dtype=np.float32
        )
        return cls(**d)  # type: ignore[arg-type]

    # ---------- sampling ----------
    def _correlated_gaussian(self, rng: np.random.Generator) -> NDArray[np.complex64]:
        V = rng.standard_normal((self.nt, self.nchan), dtype=np.float32) + 1j * rng.standard_normal(
            (self.nt, self.nchan), dtype=np.float32
        )
        if abs(self.phi_t) > 0:
            V = signal.lfilter([1.0], [1.0, -self.phi_t], V, axis=0).astype(np.complex64)
        if abs(self.phi_f) > 0:
            V = signal.lfilter([1.0], [1.0, -self.phi_f], V, axis=1).astype(np.complex64)
        return V

    def sample(self, *, seed: Optional[int] = None) -> NDArray[np.float32]:
        rng = np.random.default_rng(seed)
        V = self._correlated_gaussian(rng)
        V *= np.sqrt(np.outer(self.g_t, self.b_f)).astype(np.float32)
        I = (V.real ** 2 + V.imag ** 2).astype(np.float32)
        if not np.isclose(self.gamma_k, 1.0, atol=0.05):
            G = rng.gamma(shape=self.gamma_k, scale=self.gamma_theta, size=I.shape).astype(np.float32)
            I *= G / (self.gamma_k * self.gamma_theta)
        return I

# -----------------------------------------------------------------------------
# Estimator (NaN‑aware)
# -----------------------------------------------------------------------------

def _inpaint_nans(I: NDArray[np.floating]) -> NDArray[np.floating]:
    """Replace NaNs by the average of row & column medians (simple, fast)."""
    if not np.isnan(I).any():
        return I  # nothing to do
    warnings.warn("Input contains NaNs – treating them as masked pixels.")
    I = I.copy()
    row_med = np.nanmedian(I, axis=1)
    col_med = np.nanmedian(I, axis=0)
    # Fallback to global median if an entire row/col is NaN
    global_med = np.nanmedian(I)
    row_med = np.where(np.isnan(row_med), global_med, row_med)
    col_med = np.where(np.isnan(col_med), global_med, col_med)
    inds = np.where(np.isnan(I))
    I[inds] = 0.5 * (row_med[inds[0]] + col_med[inds[1]])
    return I


def estimate_noise_descriptor(I: NDArray[np.floating], nlags: int = 50) -> NoiseDescriptor:
    """Fit a *NoiseDescriptor* from an off‑pulse dynamic spectrum *I*.

    Any NaNs in *I* are interpreted as masked pixels and are automatically
    in‑painted using the local row/column medians before statistics are
    computed.
    """
    if I.ndim != 2:
        raise ValueError("Input array must be 2‑D (time × frequency)")
    I = np.asarray(I, dtype=np.float64)
    if np.isnan(I).all():
        raise ValueError("Input contains only NaNs – nothing to fit!")
    I = _inpaint_nans(I)
    nt, nchan = I.shape

    # (1) marginal PDF (Gamma MOM, NaN‑free now)
    mean, var = I.mean(), I.var(ddof=0)
    k_hat, theta_hat = mean**2 / var, var / mean
    gamma_k, gamma_theta = (1.0, 1.0) if 0.9 <= k_hat <= 1.1 else (float(k_hat), float(theta_hat))

    # (2) correlations
    acf_t = np.mean([_acf_1d(I[:, j], nlags) for j in range(nchan)], axis=0)
    acf_f = np.mean([_acf_1d(I[i, :], nlags) for i in range(nt)], axis=0)
    phi_t, phi_f = _lag1_coeff(acf_t), _lag1_coeff(acf_f)

    # (3) slow systematics
    g_t = np.clip(np.nanmedian(I, axis=1).astype(np.float32), EPS, None)
    b_f = np.clip(np.nanmedian(I, axis=0).astype(np.float32), EPS, None)
    g_t /= g_t.mean()
    b_f /= b_f.mean()

    return NoiseDescriptor(nt, nchan, gamma_k, gamma_theta, phi_t, phi_f, g_t, b_f)

# -----------------------------------------------------------------------------
# Self‑test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    I_syn = rng.exponential(scale=1.0, size=(2048, 64)).astype(np.float32)
    # Inject NaNs to test robustness
    I_syn[100:110, 10:20] = np.nan
    desc = estimate_noise_descriptor(I_syn)
    fake = desc.sample(seed=1)
    print("Descriptor OK – nan‑safe.  Fake spectrum mean:", fake.mean())
