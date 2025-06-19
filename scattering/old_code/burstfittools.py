"""
 burstfittools (v1.1 – 2025‑06‑16)
 ------------------------------------------------
 Fast Radio Burst dynamic‑spectrum modelling + MCMC fitting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
import emcee

__all__ = [
    "FRBModelParams",
    "FRBModel",
    "FRBFitter",
    "compute_bic",
]

# -----------------------------------------------------------------------------
# Constants & logging
# -----------------------------------------------------------------------------
DM_DELAY_CONST_MS = 4.148808  # ms GHz² (pc cm⁻³)⁻¹
INTRA_CHANNEL_CONST_MS = 1.622e-3  # ms GHz

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Dataclass
# -----------------------------------------------------------------------------

@dataclass
class FRBModelParams:
    c0: float
    t0: float
    gamma: float
    zeta: float | None = None
    tau_1ghz: float | None = None

    def as_tuple(self) -> Tuple[float, ...]:
        return tuple(v for v in asdict(self).values() if v is not None)

# -----------------------------------------------------------------------------
# FRB model
# -----------------------------------------------------------------------------

class FRBModel:
    def __init__(self, data: NDArray[np.float64], time: NDArray[np.float64], freq: NDArray[np.float64], dm_init: float, beta: float = 2.0):
        self.data = data.astype(float)
        self.time = time.astype(float)
        self.freq = freq.astype(float)
        self.dm_init = dm_init
        self.beta = beta
        self.n_ch, self.n_t = self.data.shape
        dt = np.diff(self.time)
        if not np.allclose(dt, dt[0]):
            raise ValueError("Time axis must be uniform.")
        self.dt = dt[0]
        self.noise_std = self._estimate_noise()

    # helpers
    def dispersion_delay(self, dm_err: float = 0.0) -> NDArray[np.float64]:
        ref = self.freq.max()
        return DM_DELAY_CONST_MS * dm_err * (self.freq ** -self.beta - ref ** -self.beta)

    def intra_channel_smearing(self, dm: float, zeta: float = 0.0) -> NDArray[np.float64]:
        sig_dm = INTRA_CHANNEL_CONST_MS * dm * self.freq ** (-self.beta - 1)
        return np.hypot(sig_dm, zeta)

    # model
    def __call__(self, p: FRBModelParams, model_type: str = "M0") -> NDArray[np.float64]:
        if model_type not in {"M0", "M1", "M2", "M3"}:
            raise ValueError("invalid model_type")
        c0, t0, g = p.c0, p.t0, p.gamma
        z, tau = p.zeta or 0.0, p.tau_1ghz or 0.0
        ref = self.freq[self.n_ch // 2]
        amp = c0 * (self.freq / ref) ** g
        mu = t0 + self.dispersion_delay(0.0)[:, None]
        sig = self.intra_channel_smearing(self.dm_init, z)[:, None]
        gauss = amp[:, None] * np.exp(-0.5 * ((self.time - mu) / sig) ** 2) / (np.sqrt(2 * np.pi) * sig)
        if tau > 0 and model_type in {"M2", "M3"}:
            alpha = 4.0
            tau_i = tau * (self.freq / 1.0) ** (-alpha)
            t_rel = self.time - self.time.min()
            pbf = np.where(t_rel[None, :] >= 0, np.exp(-t_rel[None, :] / tau_i[:, None]), 0.0)
            pbf /= pbf.sum(axis=1, keepdims=True)
            pad = self.n_t * 2 - 1
            n_fft = int(2 ** np.ceil(np.log2(pad)))
            spec = np.fft.rfft(gauss, n=n_fft, axis=1)
            kern = np.fft.rfft(pbf, n=n_fft, axis=1)
            scat = np.fft.irfft(spec * kern, n=n_fft, axis=1)[:, : self.n_t]
            return np.flipud(scat)
        return np.flipud(gauss)

    def log_likelihood(self, p: FRBModelParams, model_type: str = "M0") -> float:
        res = (self.data - self(p, model_type)) / self.noise_std[:, None]
        return -0.5 * np.sum(res ** 2)

    def _estimate_noise(self) -> NDArray[np.float64]:
        q = self.n_t // 4
        idx = np.r_[0:q, 3 * q:]
        rms = np.std(self.data[:, idx], axis=1, ddof=1)
        return np.clip(rms, 1e-3, None)
    
    #def _estimate_noise(self) -> NDArray[np.float64]:
    #    # Identify off-pulse regions (e.g., first and last quarters of the time axis)
    #    off_pulse_indices = np.concatenate([
    #        np.arange(0, self.n_t // 4),
    #        np.arange(3 * self.n_t // 4, self.n_t)
    #    ])
    #    off_pulse_data = self.data[:, off_pulse_indices]
    #    # Estimate noise using the off-pulse data
    #    noise_std = np.std(off_pulse_data, axis=1)
    #    #noise_std_log = np.std(np.log(off_pulse_data + np.abs(np.min(off_pulse_data)) + 1e-9), axis=1)
    #    noise_std = np.maximum(noise_std, 1e-3)
    #    print('Noise Sigma: ', noise_std)
    #    #noise_mean_log = np.mean(np.log(off_pulse_data + np.abs(np.min(off_pulse_data)) + 1e-9), axis=1)
    #    return noise_std

# -----------------------------------------------------------------------------
# Fitter
# -----------------------------------------------------------------------------

class FRBFitter:
    def __init__(self, model: FRBModel, prior_bounds: Dict[str, Tuple[float, float]], n_steps: int = 300, n_walkers_multiplier: int = 10):
        self.model = model
        self.prior_bounds = prior_bounds
        self.n_steps = n_steps
        self.n_walkers_multiplier = n_walkers_multiplier

    # internal helpers -------------------------------------------------
    def _log_prior(self, params: Sequence[float], m: str) -> float:
        keys = {
            "M0": ("c0", "t0", "gamma"),
            "M1": ("c0", "t0", "gamma", "zeta"),
            "M2": ("c0", "t0", "gamma", "tau_1ghz"),
            "M3": ("c0", "t0", "gamma", "zeta", "tau_1ghz"),
        }[m]
        for v, k in zip(params, keys):
            lo, hi = self.prior_bounds[k]
            if not (lo <= v <= hi):
                return -np.inf
        return 0.0

    def _log_prob(self, params: Sequence[float], m: str) -> float:
        lp = self._log_prior(params, m)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.model.log_likelihood(self._unpack(params, m), m)

    def _walkers(self, m: str, init: FRBModelParams | None = None) -> NDArray[np.float64]:
        keys = {
            "M0": ("c0", "t0", "gamma"),
            "M1": ("c0", "t0", "gamma", "zeta"),
            "M2": ("c0", "t0", "gamma", "tau_1ghz"),
            "M3": ("c0", "t0", "gamma", "zeta", "tau_1ghz"),
        }[m]
        ndim = len(keys)
        nwalk = max(self.n_walkers_multiplier * ndim, 50)
        p0 = np.zeros((nwalk, ndim))
        if init is None:
            # uniform within bounds
            for j, k in enumerate(keys):
                lo, hi = self.prior_bounds[k]
                p0[:, j] = np.random.uniform(lo, hi, nwalk)
        else:
            center = init.as_tuple()
            for j, (k, c) in enumerate(zip(keys, center)):
                lo, hi = self.prior_bounds[k]
                scale = 0.05 * (hi - lo)
                p0[:, j] = np.clip(np.random.normal(c, scale, nwalk), lo, hi)
        return p0

    def _unpack(self, params: Sequence[float], m: str) -> FRBModelParams:
        maps = {
            "M0": (0, 1, 2, None, None),
            "M1": (0, 1, 2, 3, None),
            "M2": (0, 1, 2, None, 3),
            "M3": (0, 1, 2, 3, 4),
        }
        i = maps[m]
        return FRBModelParams(
            c0=params[i[0]],
            t0=params[i[1]],
            gamma=params[i[2]],
            zeta=(params[i[3]] if i[3] is not None else None),
            tau_1ghz=(params[i[4]] if i[4] is not None else None),
        )

    # public API --------------------------------------------------------
    def fit(
        self,
        init: FRBModelParams | None = None,
        model_flags: Tuple[bool, bool, bool, bool] = (True, True, True, True),
    ) -> Dict[str, Dict[str, object]]:
        out, n = {}, self.model.data.size
        for flag, m in zip(model_flags, ("M0", "M1", "M2", "M3")):
            if not flag:
                continue
            p0 = self._walkers(m, init)
            ndim = p0.shape[1]
            sampl = emcee.EnsembleSampler(p0.shape[0], ndim, self._log_prob, args=(m,))
            sampl.run_mcmc(p0, self.n_steps, progress=False)
            lnL_max = sampl.get_log_prob().max()
            bic_val = compute_bic(lnL_max, ndim, n)
            out[m] = {"sampler": sampl, "bic": bic_val, "lnL_max": lnL_max}
            logger.info("%s done (BIC %.1f)", m, bic_val)
        return out
# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------

def compute_bic(lnL_max: float, k: int, n: int) -> float:
    """Bayesian Information Criterion (smaller = better)."""
    return -2.0 * lnL_max + k * np.log(n)


def downsample_data(data, f_factor = 1, t_factor = 1):   

    # Check data shape
    print(f'Power Shape (frequency axis): {data.shape[0]}')
    print(f'Power Shape (time axis): {data.shape[1]}')

    # Downsample in frequency
    # Ensure nearest multiple is not greater than the frequency axis length
    nrst_mltpl_f = f_factor * (data.shape[0] // f_factor)
    print(f'Nearest Multiple To Downsampling Factor (frequency): {nrst_mltpl_f}')

    # Clip the frequency axis to the nearest multiple
    data_clip_f = data[:nrst_mltpl_f, :]

    # Downsample along the frequency axis (y-axis)
    data_ds_f = data_clip_f.reshape([
        nrst_mltpl_f // f_factor, f_factor,
        data_clip_f.shape[1]
    ]).mean(axis=1)

    # Downsample in time
    # Ensure nearest multiple is not greater than the time axis length
    nrst_mltpl_t = t_factor * (data_ds_f.shape[1] // t_factor)
    print(f'Nearest Multiple To Downsampling Factor (time): {nrst_mltpl_t}')

    # Clip the time axis to the nearest multiple
    data_clip_t = data_ds_f[:, :nrst_mltpl_t]

    # Downsample along the time axis (x-axis)
    data_ds_t = data_clip_t.reshape([
        data_clip_t.shape[0],  # Frequency axis remains the same
        nrst_mltpl_t // t_factor, t_factor
    ]).mean(axis=2)

    # Output the final downsampled data
    print(f'Downsampled Data Shape: {data_ds_t.shape}')

    return data_ds_t
