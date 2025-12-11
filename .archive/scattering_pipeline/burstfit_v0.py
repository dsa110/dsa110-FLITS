"""
burstfit.py
===========

Physics kernel + lightweight MCMC wrapper for modelling **fast-radio-burst
dynamic spectra** with dispersion, intra-channel smearing and thin-screen
scattering.

Public API
----------
* :class:`FRBParams`  – dataclass container for model parameters.
* :class:`FRBModel`   – forward model & Gaussian likelihood.
* :class:`FRBFitter`  – emcee front-end with box priors.
* :func:`compute_bic` – Bayesian Information Criterion helper.
* :func:`build_priors`
* :func:`downsample`
* :func:`plot_dynamic`
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Sequence, Tuple

from flits.common.constants import DM_DELAY_MS

import emcee
import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve

__all__ = [
    "FRBParams",
    "FRBModel", 
    "FRBFitter",
    "compute_bic",
    "build_priors",
    "downsample",
    "plot_dynamic",
    "goodness_of_fit", 
]

# ----------------------------------------------------------------------
# Module-level constants
# ----------------------------------------------------------------------
DM_SMEAR_MS = 1.622e-3        # intra-channel smearing, ms GHz

log = logging.getLogger(__name__)
if not log.handlers:
    log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

# ----------------------------------------------------------------------
# Log-probability wrapper
# ----------------------------------------------------------------------

#Module-level wrapper for emcee + multiprocessing
def _log_prob_wrapper(theta, model, priors, order, key):
    """Module-level function for multiprocessing compatibility."""
    # 1. Check priors
    for value, name in zip(theta, order[key]):
        lo, hi = priors[name]
        if not (lo <= value <= hi):
            return -np.inf
    
    # 2. Compute likelihood
    params = FRBParams.from_sequence(theta, key)
    # This now correctly uses the log_likelihood method on the model instance
    return model.log_likelihood(params, key)

# ----------------------------------------------------------------------
# Dataclass – model parameters
# ----------------------------------------------------------------------
@dataclass
class FRBParams:
    """Parameter container for the scattering model."""

    c0: float
    t0: float
    gamma: float
    zeta: float = 0.0
    tau_1ghz: float = 0.0

    # helper methods ---------------------------------------------------
    def as_tuple(self) -> Tuple[float, ...]:
        return tuple(asdict(self).values())

    @classmethod
    def to_sequence(self, model_key: str = "M3") -> Sequence[float]:
        """Pack parameters into a flat sequence for a given model_key."""
        key_map = {
            "M0": ("c0", "t0", "gamma"),
            "M1": ("c0", "t0", "gamma", "zeta"),
            "M2": ("c0", "t0", "gamma", "tau_1ghz"),
            "M3": ("c0", "t0", "gamma", "zeta", "tau_1ghz"),
        }
        return [getattr(self, k) for k in key_map[model_key]]
    def from_sequence(
        cls, seq: Sequence[float], model_key: str = "M3"
    ) -> "FRBParams":
        """Unpack a flat param vector according to *model_key*."""
        key_map = {
            "M0": ("c0", "t0", "gamma"),
            "M1": ("c0", "t0", "gamma", "zeta"),
            "M2": ("c0", "t0", "gamma", "tau_1ghz"),
            "M3": ("c0", "t0", "gamma", "zeta", "tau_1ghz"),
        }
        kwargs = {k: v for k, v in zip(key_map[model_key], seq)}
        # fill optional keys with defaults
        kwargs.setdefault("zeta", 0.0)
        kwargs.setdefault("tau_1ghz", 0.0)
        return cls(**kwargs)  # type: ignore[arg-type]

# ----------------------------------------------------------------------
# Forward model
# ----------------------------------------------------------------------
class FRBModel:
    """
    Forward model + (Gaussian) log-likelihood for a dynamic spectrum.

    Parameters
    ----------
    time, freq
        1-D axes in **ms** and **GHz**; `freq` must be ascending.
    data
        Optional 2-D array (nfreq, ntime).  If supplied, a robust noise
        estimate is computed for the likelihood.
    dm_init
        Dispersion measure already **removed** from the dynamic spectrum
        (so the burst is roughly aligned).
    beta
        Dispersion exponent; 2.0 for cold plasma, 2.002 for relativistic.
    noise_std
        Per-channel noise RMS (overrides internal MAD estimate).
    off_pulse
        Slice / indices marking off-pulse samples for noise estimation.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        time: NDArray[np.floating],
        freq: NDArray[np.floating],
        *,
        data: NDArray[np.floating] | None = None,
        dm_init: float = 0.0,
        beta: float = 2.0,
        noise_std: NDArray[np.floating] | None = None,
        off_pulse: slice | Sequence[int] | None = None,
    ) -> None:
        self.time = np.asarray(time, dtype=float)
        self.freq = np.asarray(freq, dtype=float)

        if data is not None:
            self.data = np.asarray(data, dtype=float)
            if self.data.shape != (self.freq.size, self.time.size):
                raise ValueError("data must have shape (nfreq, ntime)")
        else:
            self.data = None

        self.dm_init = float(dm_init)
        self.beta = float(beta)

        if not np.allclose(np.diff(self.time), self.time[1] - self.time[0]):
            raise ValueError("time axis must be uniform")
        self.dt = self.time[1] - self.time[0]

        # noise estimate ------------------------------------------------
        if noise_std is None and self.data is not None:
            self.noise_std = self._estimate_noise(off_pulse)
        else:
            self.noise_std = noise_std

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _dispersion_delay(
        self, dm_err: float = 0.0, ref_freq: float | None = None
    ) -> NDArray[np.floating]:
        if ref_freq is None:
            ref_freq = self.freq.max()
        return DM_DELAY_MS * dm_err * (self.freq ** -self.beta -
                                       ref_freq ** -self.beta)

    def _smearing_sigma(
        self, dm: float, zeta: float
    ) -> NDArray[np.floating]:
        sig_dm = DM_SMEAR_MS * dm * self.freq ** (-self.beta - 1.0)
        return np.hypot(sig_dm, zeta)

    def _estimate_noise(self, off_pulse):
        if off_pulse is None:
            q = self.time.size // 4
            idx = np.r_[0:q, -q:0]
        elif isinstance(off_pulse, slice):        
            idx = off_pulse
        else:
            idx = np.asarray(off_pulse)
        mad = np.median(
            np.abs(
                self.data[:, idx]
                - np.median(self.data[:, idx], axis=1, keepdims=True)
            ),
            axis=1,
        )
        return 1.4826 * np.clip(mad, 1e-3, None)

    # ------------------------------------------------------------------
    # public callables
    # ------------------------------------------------------------------
    def __call__(self, p: FRBParams, model: str = "M3") -> NDArray[np.floating]:
        """Return model dynamic spectrum for parameters *p*."""
        amp = p.c0 * (self.freq / self.freq[self.freq.size // 2]) ** p.gamma
        mu  = p.t0 + self._dispersion_delay(0.0)[:, None]
        
        if model in {"M1", "M3"}:          # smearing on
            sig = self._smearing_sigma(self.dm_init, p.zeta)[:, None]
        else:                              # M0, M2  → smearing off
            sig = self._smearing_sigma(self.dm_init, 0.0)[:, None]

        sig = np.clip(sig, 1e-6, None)     # 1 µs floor prevents σ=0 → NaN

        gauss = (
            amp[:, None] *
            np.exp(-0.5 * ((self.time - mu) / sig) ** 2) /
            (np.sqrt(2 * np.pi) * sig)
        )

        if model in {"M2", "M3"} and p.tau_1ghz > 0:
            alpha = 4.0  # thin-screen Kolmogorov
            tau = p.tau_1ghz * (self.freq / 1.0) ** (-alpha)
            kernel = np.exp(
                -np.maximum(self.time, 0)[None, :] / tau[:, None]
            )
            kernel /= kernel.sum(axis=1, keepdims=True)
            return fftconvolve(gauss, kernel, mode="same", axes=1)

        if model not in {"M0", "M1", "M2", "M3"}:
            raise ValueError(f"unknown model '{model}'")

        return gauss
    
    # ------------------------------------------------------------------
    def log_prior(self, p: FRBParams, model: str, priors: Dict[str, Tuple[float, float]]) -> float:
        """Compute log prior probability for parameters."""
        param_names = {
            "M0": ("c0", "t0", "gamma"),
            "M1": ("c0", "t0", "gamma", "zeta"),
            "M2": ("c0", "t0", "gamma", "tau_1ghz"),
            "M3": ("c0", "t0", "gamma", "zeta", "tau_1ghz"),
        }[model]

        for name in param_names:
            value = getattr(p, name)
            lo, hi = priors[name]
            if not (lo <= value <= hi):
                return -np.inf
        return 0.0

    # ------------------------------------------------------------------
    def log_likelihood(self, p: FRBParams, model: str = "M3") -> float:
        if self.data is None or self.noise_std is None:
            raise RuntimeError("need observed data + noise_std for likelihood")
        resid = (self.data - self(p, model)) / self.noise_std[:, None]
        return -0.5 * np.sum(resid ** 2)

# ----------------------------------------------------------------------
# Sampler wrapper
# ----------------------------------------------------------------------

class FRBFitter:
    """
    Thin wrapper around *emcee* that:

    * builds a walker ensemble only for the parameters relevant to
      ``model_key``; no broadcast mismatches.
    * supports the modern ``pool=`` interface (pass a Pool or ``None``).

    Parameters
    ----------
    model
        An :class:`FRBModel` instance with data & likelihood.
    priors
        Dict ``{name: (low, high)}`` *for all five* possible parameters.
        The subset needed by each model is picked internally.
    n_steps
        MCMC length.
    pool
        A `multiprocessing.Pool`-like object or ``None``.
    """

    _ORDER = {
        "M0": ("c0", "t0", "gamma"),
        "M1": ("c0", "t0", "gamma", "zeta"),
        "M2": ("c0", "t0", "gamma", "tau_1ghz"),
        "M3": ("c0", "t0", "gamma", "zeta", "tau_1ghz"),
    }

    # ------------------------------------------------------------------
    def __init__(
        self,
        model: FRBModel,
        priors: Dict[str, Tuple[float, float]],
        *,
        n_steps: int = 1000,
        n_walkers_mult: int = 8,
        pool=None,
    ):
        self.model = model
        self.priors = priors
        self.n_steps = n_steps
        self.n_walkers_mult = n_walkers_mult
        self.pool = pool

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _slice(self, theta, key):
        """Return theta restricted to names in _ORDER[key]."""
        return np.array([theta[self._ORDER["M3"].index(n)] for n in self._ORDER[key]])

    def _log_prior(self, theta, key):
        for value, name in zip(theta, self._ORDER[key]):
            lo, hi = self.priors[name]
            if not (lo <= value <= hi):
                return -np.inf
        return 0.0

    #def _log_prob(self, theta, key):
    #    lp = self._log_prior(theta, key)
    #    if not np.isfinite(lp):
    #        return -np.inf
    #    params = FRBParams.from_sequence(theta, key)   # use *key*, not "M3"
    #    return lp + self.model.log_likelihood(params, key)

    def _init_walkers(self, p0: FRBParams, key: str, nwalk: int):
        names = self._ORDER[key]
        centre = np.array([getattr(p0, n) for n in names])
        widths = np.array([0.05 * (self.priors[n][1] - self.priors[n][0]) for n in names])
        lower, upper = zip(*(self.priors[n] for n in names))
        walkers = np.random.normal(centre, widths, size=(nwalk, len(names)))
        return np.clip(walkers, lower, upper)

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------
    def sample(self, p0: FRBParams, model_key: str = "M3"):
        """Run the MCMC sampler."""
        names = self._ORDER[model_key]
        ndim = len(names)
        nwalk = max(self.n_walkers_mult * ndim, 2 * ndim)

        p_walkers = self._init_walkers(p0, model_key, nwalk)

        sampler = emcee.EnsembleSampler(
            nwalk, ndim, _log_prob_wrapper,
            args=(self.model, self.priors, self._ORDER, model_key),
            pool=self.pool,
        )
        sampler.run_mcmc(p_walkers, self.n_steps, progress=True)
        return sampler

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

POSITIVE = {"c0", "zeta", "tau_1ghz"}

def build_priors(init: FRBParams, scale: float = 3.0):
    """Return box (uniform) priors centred on *init*.
    For parameters in _POSITIVE, the lower bound is clamped at 0.0.
    """
    pri = {}
    for name, val in asdict(init).items():
        width = scale * (abs(val) if val != 0 else 1.0)
        lo = val - width
        hi = val + width
        if name in _POSITIVE: # <-- This check is important
            lo = max(0.0, lo)
        pri[name] = (lo, hi)
    return pri

def compute_bic(logL_max: float, k: int, n: int) -> float:
    """Bayesian Information Criterion (lower = preferred)."""
    return -2.0 * logL_max + k * np.log(n)

def downsample(data: NDArray[np.floating], f_factor=1, t_factor=1):
    """Block-average by integer factors along (freq, time)."""
    if f_factor == 1 and t_factor == 1:
        return data
    nf, nt = data.shape
    nf_ds = nf // f_factor * f_factor
    nt_ds = nt // t_factor * t_factor
    d = data[:nf_ds, :nt_ds].reshape(
        nf_ds // f_factor, f_factor, nt_ds // t_factor, t_factor
    )
    return d.mean(axis=(1, 3))

def goodness_of_fit(data: NDArray[np.floating], 
                   model: NDArray[np.floating], 
                   noise_std: NDArray[np.floating] | None = None,
                   n_params: int = 5) -> Dict[str, float]:
    """
    Compute goodness-of-fit metrics for the model.
    
    Parameters
    ----------
    data : array_like
        Observed dynamic spectrum (nfreq, ntime)
    model : array_like  
        Model dynamic spectrum (nfreq, ntime)
    noise_std : array_like, optional
        Per-channel noise standard deviation. If None, estimated from residuals.
    n_params : int
        Number of model parameters (for DOF calculation)
        
    Returns
    -------
    dict
        Dictionary containing:
        - chi2: Total chi-squared
        - chi2_reduced: Reduced chi-squared  
        - ndof: Number of degrees of freedom
        - residual_rms: RMS of residuals
        - residual_autocorr: Autocorrelation of time-collapsed residuals
    """
    residual = data - model
    
    if noise_std is None:
        # Estimate noise from first and last quarters of time series
        n_time = residual.shape[1]
        q = n_time // 4
        off_pulse = np.concatenate([residual[:, :q], residual[:, -q:]], axis=1)
        noise_std = np.std(off_pulse, axis=1, keepdims=True)
    elif noise_std.ndim == 1:
        noise_std = noise_std[:, np.newaxis]
    
    # Chi-squared calculation
    chi2 = np.sum((residual / noise_std) ** 2)
    ndof = data.size - n_params
    chi2_reduced = chi2 / ndof
    
    # Residual autocorrelation of time-collapsed profile
    residual_profile = np.sum(residual, axis=0)
    # Normalize for autocorrelation
    residual_profile = residual_profile - np.mean(residual_profile)
    
    # Compute autocorrelation using numpy's correlate
    autocorr = np.correlate(residual_profile, residual_profile, mode='same')
    # Normalize so autocorr[center] = 1
    autocorr = autocorr / autocorr[len(autocorr)//2]
    
    return {
        'chi2': float(chi2),
        'chi2_reduced': float(chi2_reduced),
        'ndof': int(ndof),
        'residual_rms': float(np.std(residual)),
        'residual_autocorr': autocorr
    }

def plot_dynamic(
    ax,
    dyn: NDArray[np.floating],
    time: NDArray[np.floating],
    freq: NDArray[np.floating],
    **imshow_kw,
):
    """Imshow wrapper with correct axes."""
    imshow_kw.setdefault("aspect", "auto")
    imshow_kw.setdefault("origin", "lower")
    extent = [time[0], time[-1], freq[0], freq[-1]]
    return ax.imshow(dyn, extent=extent, **imshow_kw)