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

from flits.common.constants import DM_DELAY_MS

# ----------------------------------------------------------------------
# Module-level constants
# ----------------------------------------------------------------------
DM_SMEAR_MS = 8.3e-6     # intra-channel smearing, ms GHz⁻³ MHz⁻¹ -> this is a more common formulation
                         # The previous value was likely a typo or for a specific setup.
                         # This standard form is: 8.3 * 1e6 * DM * dnu_MHz / nu_GHz**3 / 1e3
                         # We will apply dnu_MHz later, so this constant is 8.3e-6 ms GHz^3 MHz^-1

# ## FIX ##: Restored the set of physically non-negative parameters.
_POSITIVE = {"c0", "zeta", "tau_1ghz"}        # keep in ONE place only
_MIN_POS  = 1e-6   


log = logging.getLogger(__name__)
if not log.handlers:
    log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

# ----------------------------------------------------------------------
# Log-probability wrapper
# ----------------------------------------------------------------------

# ## REFACTOR ##: This is the correct, module-level wrapper for emcee + multiprocessing.
# It takes the model object itself as an argument, making it stateless and pickleable.
def _log_prob_wrapper(theta_raw, model, priors, order, key, log_weight_pos,
                      log_params_names=None, alpha_gauss=None, likelihood_kind="gaussian", student_nu=5.0,
                      param_mode="single", K: int = 1):
    """Module-level function for multiprocessing compatibility."""
    # Map raw parameters (possibly in log-space) to linear domain
    names = order[key]
    log_params_names = set(log_params_names or [])
    theta = []
    for raw, name in zip(theta_raw, names):
        if name in log_params_names:
            theta.append(np.exp(raw))
        else:
            theta.append(raw)

    logp = 0.0

    # 1. Check top-hat priors
    for value, name in zip(theta, names):
        lo, hi = priors[name]
        if not (lo <= value <= hi):
            return -np.inf

    # 2. Optional Gaussian prior on alpha
    if alpha_gauss is not None and ("alpha" in names):
        mu, sigma = alpha_gauss
        if sigma is not None and sigma > 0:
            idx = names.index("alpha")
            a = theta[idx]
            logp += -0.5 * ((a - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))

    # 3. Jeffreys 1/x weight for positive params (still sampling in linear units)
    if log_weight_pos:
        for name, v in zip(names, theta):
            if name in _POSITIVE:
                logp += -np.log(max(v, 1e-30))

    # 4. Compute likelihood
    if param_mode == "single":
        params = FRBParams.from_sequence(theta, key)
        if likelihood_kind == "gaussian":
            return logp + model.log_likelihood(params, key)
        elif likelihood_kind == "studentt":
            return logp + model.log_likelihood_student_t(params, key, nu=float(student_nu))
        else:
            return logp + model.log_likelihood(params, key)
    else:
        # multi-component: parse shared + component params and sum models
        # shared
        def get(name):
            return theta[names.index(name)] if name in names else None
        gamma = get("gamma") if "gamma" in names else -1.6
        tau1 = get("tau_1ghz") if "tau_1ghz" in names else 0.0
        alpha = get("alpha") if "alpha" in names else 4.4
        delta_dm = get("delta_dm") if "delta_dm" in names else 0.0
        # sum components
        model_sum = np.zeros_like(model.data)
        for idx in range(1, int(K)+1):
            c0 = get(f"c0_{idx}")
            t0 = get(f"t0_{idx}")
            zeta = get(f"zeta_{idx}")
            if c0 is None or t0 is None or zeta is None:
                continue
            p_i = FRBParams(c0=c0, t0=t0, gamma=gamma, zeta=zeta, tau_1ghz=tau1, alpha=alpha, delta_dm=delta_dm)
            model_sum = model_sum + model(p_i, "M3")
        # compute likelihood against summed model
        if likelihood_kind == "gaussian":
            noise_std_safe = np.clip(model.noise_std, 1e-9, None)
            resid = (model.data - model_sum) / noise_std_safe[:, None]
            return logp + (-0.5 * np.sum(resid ** 2))
        else:
            # student-t
            noise_std_safe = np.clip(model.noise_std, 1e-9, None)
            resid = (model.data - model_sum) / noise_std_safe[:, None]
            r2_over_nu = (resid ** 2) / float(student_nu)
            term = -0.5 * (float(student_nu) + 1.0) * np.log1p(r2_over_nu)
            const = -0.5 * model.data.size * np.log(float(student_nu) * np.pi) - np.sum(np.log(noise_std_safe))
            return logp + const + float(np.sum(term))

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
    alpha: float = 4.4       # frequency scaling exponent τ ∝ ν^{-alpha}
    delta_dm: float = 0.0    # residual DM error around dm_init

    # ## FIX ##: Added the to_sequence method that was missing.
    def to_sequence(self, model_key: str = "M3") -> Sequence[float]:
        """Pack parameters into a flat sequence for a given model_key."""
        key_map = {
            "M0": ("c0", "t0", "gamma"),
            "M1": ("c0", "t0", "gamma", "zeta"),
            "M2": ("c0", "t0", "gamma", "tau_1ghz"),
            # M3 now includes alpha and delta_dm as part of the parameterization
            "M3": ("c0", "t0", "gamma", "zeta", "tau_1ghz", "alpha", "delta_dm"),
        }
        return [getattr(self, k) for k in key_map[model_key]]

    # ## FIX ##: Added the @classmethod decorator. This was a critical bug.
    @classmethod
    def from_sequence(
        cls, seq: Sequence[float], model_key: str = "M3"
    ) -> "FRBParams":
        """Unpack a flat param vector according to *model_key*."""
        key_map = {
            "M0": ("c0", "t0", "gamma"),
            "M1": ("c0", "t0", "gamma", "zeta"),
            "M2": ("c0", "t0", "gamma", "tau_1ghz"),
            "M3": ("c0", "t0", "gamma", "zeta", "tau_1ghz", "alpha", "delta_dm"),
        }
        kwargs = {k: v for k, v in zip(key_map[model_key], seq)}
        # fill optional keys with defaults
        kwargs.setdefault("zeta", 0.0)
        kwargs.setdefault("tau_1ghz", 0.0)
        return cls(**kwargs)

# ----------------------------------------------------------------------
# Forward model
# ----------------------------------------------------------------------
class FRBModel:
    """
    Forward model + (Gaussian) log-likelihood for a dynamic spectrum.
    """
    def __init__(
        self,
        time: NDArray[np.floating],
        freq: NDArray[np.floating],
        *,
        data: NDArray[np.floating] | None = None,
        dm_init: float = 0.0,
        df_MHz: float = 0.390625, #0.03051757812, #Channel width is needed for smearing
        beta: float = 2.0,
        noise_std: NDArray[np.floating] | None = None,
        off_pulse: slice | Sequence[int] | None = None,
    ) -> None:
        self.time = np.asarray(time, dtype=float)
        self.freq = np.asarray(freq, dtype=float)
        self.df_MHz = float(df_MHz)

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

        if noise_std is None and self.data is not None:
            self.noise_std = self._estimate_noise(off_pulse)
        else:
            self.noise_std = noise_std

    def _dispersion_delay(
        self, dm_err: float = 0.0, ref_freq: float | None = None
    ) -> NDArray[np.floating]:
        if ref_freq is None:
            ref_freq = self.freq.max()
        return DM_DELAY_MS * dm_err * (self.freq ** -self.beta - ref_freq ** -self.beta)

    # ## REFACTOR ##: Smearing calculation is now more explicit.
    def _smearing_sigma(self, dm: float, zeta: float) -> NDArray[np.floating]:
        """Calculates total Gaussian width from intrinsic pulse width (zeta)
        and intra-channel DM smearing."""
        # DM smearing time (ms) = 8.3e-6 * DM * df_MHz / nu_GHz^3
        sig_dm = DM_SMEAR_MS * dm * self.df_MHz * (self.freq ** -3.0)
        # Add in quadrature with intrinsic width
        return np.hypot(sig_dm, zeta)

    def _estimate_noise(self, off_pulse):
        if self.data is None: return None
        if off_pulse is None:
            q = self.time.size // 4
            idx = np.r_[0:q, -q:0]
        else:
            idx = np.asarray(off_pulse)
        
        # Ensure indices are within bounds
        idx = idx[idx < self.data.shape[1]]

        mad = np.median(
            np.abs(self.data[:, idx] - np.median(self.data[:, idx], axis=1, keepdims=True)),
            axis=1,
        )
        return 1.4826 * np.clip(mad, 1e-6, None) # Use a smaller floor

    def __call__(self, p: FRBParams, model_key: str = "M3") -> NDArray[np.floating]:
        """Return model dynamic spectrum for parameters *p*."""
        ref_freq = np.median(self.freq)
        amp = p.c0 * (self.freq / ref_freq) ** p.gamma
        # include residual DM offset around dm_init
        mu = p.t0 + self._dispersion_delay(p.delta_dm)[:, None]

        if model_key in {"M1", "M3"}:
            sig = self._smearing_sigma(self.dm_init, p.zeta)[:, None]
        else:
            sig = self._smearing_sigma(self.dm_init, 0.0)[:, None]

        # Guard against non-physical width
        sig = np.clip(sig, 1e-6, None)

        gauss = (1 / (np.sqrt(2 * np.pi) * sig)) * np.exp(-0.5 * ((self.time - mu) / sig) ** 2)

        # --- FIX: Implement safe division to prevent NaN ---
        gauss_sum = np.sum(gauss, axis=1, keepdims=True)
        # Clip the sum to a tiny positive number to avoid 0/0 division
        safe_gauss_sum = np.clip(gauss_sum, 1e-30, None)
        gauss_norm = gauss / safe_gauss_sum

        profile = amp[:, None] * gauss_norm

        if model_key in {"M2", "M3"} and p.tau_1ghz > 1e-6:
            alpha = 4.0
            # use parameterized frequency scaling exponent for scattering
            alpha = getattr(p, "alpha", 4.4)
            tau = p.tau_1ghz * (self.freq / 1.0) ** (-alpha)
            t_kernel = self.time - self.time[0]

            kernel = np.exp(-t_kernel[None, :] / np.clip(tau, 1e-6, None)[:, None])

            # --- FIX: Implement safe division for the kernel as well ---
            kernel_sum = np.sum(kernel, axis=1, keepdims=True)
            safe_kernel_sum = np.clip(kernel_sum, 1e-30, None)
            kernel_norm = kernel / safe_kernel_sum

            # --- FIX: Use causal convolution (mode='full' truncated) instead of mode='same'
            # mode='same' centers the output, which incorrectly shifts the peak earlier.
            # For a causal exponential scattering kernel, we want the first N samples
            # of the full convolution, which preserves the peak position and only delays it.
            n_time = profile.shape[1]
            result_full = fftconvolve(profile, kernel_norm, mode="full", axes=1)
            return result_full[:, :n_time]

        if model_key not in {"M0", "M1", "M2", "M3"}:
            raise ValueError(f"unknown model '{model_key}'")

        return profile

    def log_likelihood(self, p: FRBParams, model: str = "M3") -> float:
        if self.data is None or self.noise_std is None:
            raise RuntimeError("need observed data + noise_std for likelihood")
        
        # Protect against all-zero noise_std if a channel is dead
        noise_std_safe = np.clip(self.noise_std, 1e-9, None)

        resid = (self.data - self(p, model)) / noise_std_safe[:, None]
        return -0.5 * np.sum(resid ** 2)

    def log_likelihood_student_t(self, p: FRBParams, model: str = "M3", nu: float = 5.0) -> float:
        if self.data is None or self.noise_std is None:
            raise RuntimeError("need observed data + noise_std for likelihood")
        noise_std_safe = np.clip(self.noise_std, 1e-9, None)
        resid = (self.data - self(p, model)) / noise_std_safe[:, None]
        # Student-t log-pdf up to constant per element
        # logL = sum( - (nu+1)/2 * log(1 + r^2/nu) ) - N * 0.5*log(nu*pi) - sum(log(sigma))
        r2_over_nu = (resid ** 2) / nu
        term = -0.5 * (nu + 1.0) * np.log1p(r2_over_nu)
        const = -0.5 * self.data.size * np.log(nu * np.pi) - np.sum(np.log(noise_std_safe))
        return const + float(np.sum(term))

# ----------------------------------------------------------------------
# Sampler wrapper
# ----------------------------------------------------------------------
class FRBFitter:
    """Thin wrapper around *emcee*."""
    _ORDER = {
        "M0": ("c0", "t0", "gamma"),
        "M1": ("c0", "t0", "gamma", "zeta"),
        "M2": ("c0", "t0", "gamma", "tau_1ghz"),
        # M3 extended to include alpha and delta_dm
        "M3": ("c0", "t0", "gamma", "zeta", "tau_1ghz", "alpha", "delta_dm"),
    }

    def __init__(
        self,
        model: FRBModel,
        priors: Dict[str, Tuple[float, float]],
        *,
        n_steps: int = 1000,
        n_walkers_mult: int = 8,
        pool=None,
        log_weight_pos=False,
        sample_log_params: bool = True,
        alpha_prior: Tuple[float, float] | None = None,
        likelihood_kind: str = "gaussian",
        student_nu: float = 5.0,
        walker_width_frac: float = 0.01,
        **kwargs
    ):
        self.model = model
        self.priors = priors
        self.n_steps = n_steps
        self.n_walkers_mult = n_walkers_mult
        self.pool = pool
        self.log_weight_pos = log_weight_pos
        self.sample_log_params = sample_log_params
        self._log_param_names = {"c0", "zeta", "tau_1ghz"} if sample_log_params else set()
        self.alpha_prior = alpha_prior
        self.likelihood_kind = likelihood_kind
        self.student_nu = student_nu
        self.custom_order: dict[str, tuple[str, ...]] = {}
        self.walker_width_frac = max(1e-4, float(walker_width_frac))

    def _init_walkers(self, p0, key: str, nwalk: int):
        names = self._ORDER[key] if key in self._ORDER else self.custom_order[key]
        lower, upper = zip(*(self.priors[n] for n in names))
        lower = np.array(lower, dtype=float)
        upper = np.array(upper, dtype=float)

        centre = []
        widths = []
        frac = self.walker_width_frac
        for i, n in enumerate(names):
            if hasattr(p0, n):
                val = getattr(p0, n)
            elif isinstance(p0, dict):
                val = p0.get(n, (lower[i] + upper[i]) / 2.0)
            else:
                val = (lower[i] + upper[i]) / 2.0
            if n in self._log_param_names:
                lo = max(lower[i], 1e-30)
                hi = max(upper[i], lo * (1.0 + 1e-6))
                centre.append(np.log(max(val, 1e-30)))
                widths.append(frac * (np.log(hi) - np.log(lo)))
            else:
                centre.append(val)
                widths.append(frac * (upper[i] - lower[i]))

        centre = np.array(centre)
        widths = np.clip(np.array(widths), 1e-6, None)

        walkers = np.random.normal(centre, widths, size=(nwalk, len(names)))

        # Clip in appropriate domain
        for j, n in enumerate(names):
            if n in self._log_param_names:
                walkers[:, j] = np.clip(walkers[:, j], np.log(max(lower[j], 1e-30)), np.log(max(upper[j], 1e-30)))
            else:
                walkers[:, j] = np.clip(walkers[:, j], lower[j], upper[j])

        return walkers

    def sample(self, p0, model_key: str = "M3"):
        """Run the MCMC sampler."""
        names = self._ORDER[model_key] if model_key in self._ORDER else self.custom_order[model_key]
        ndim = len(names)
        nwalk = max(self.n_walkers_mult * ndim, 2 * ndim)

        p_walkers = self._init_walkers(p0, model_key, nwalk)

        # log-params by full name
        log_param_fullnames = set()
        if self.sample_log_params:
            base = self._log_param_names
            for n in names:
                root = n.split('_')[0]
                if root in base:
                    log_param_fullnames.add(n)

        sampler = emcee.EnsembleSampler(
            nwalkers=nwalk,
            ndim=ndim,
            log_prob_fn=_log_prob_wrapper,
            args=(self.model, self.priors, (self.custom_order if model_key in self.custom_order else self._ORDER), model_key, self.log_weight_pos, list(log_param_fullnames), self.alpha_prior, self.likelihood_kind, self.student_nu, "multi" if model_key=="M3_multi" else "single", len(names) if model_key=="M3_multi" else 1),
            pool=self.pool,
        )
        sampler.run_mcmc(p_walkers, self.n_steps, progress=True)
        return sampler

    def build_multicomp_order(self, K: int) -> tuple[str, ...]:
        order = ["gamma", "tau_1ghz", "alpha", "delta_dm"]
        for i in range(1, K+1):
            order.extend([f"c0_{i}", f"t0_{i}", f"zeta_{i}"])
        self.custom_order["M3_multi"] = tuple(order)
        return self.custom_order["M3_multi"]

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def build_priors(
    init: "FRBParams",
    *,
    scale: float = 6.0,            # half-width multiplier (wide!)
    abs_min: float = _MIN_POS,     # floor for positive parameters
    abs_max: dict[str, float] | None = None,  # optional hard ceilings
    log_weight_pos: bool = False,  # True → Jeffreys p(x)∝1/x   (still linear sampling)
) -> dict[str, tuple[float, float]]:
    """
    Build simple linear-space top-hat priors that *won’t* strangle the chain.

    Parameters
    ----------
    init
        The optimiser-derived initial parameter set.
    scale
        Half-width multiplier around each init value.
    abs_min
        Lower floor for every positive-definite parameter.
    abs_max
        Optional per-parameter upper caps, e.g. {"tau_1ghz": 1e5}.
    log_weight_pos
        If True, you still sample in linear units but will later *add*
        -log(x) to the log-prior for each positive parameter.
    """
    from dataclasses import asdict
    pri = {}
    ceiling = abs_max or {}
    for name, val in asdict(init).items():
        w     = max(scale * max(abs(val), 1e-3), 0.5)     # ≥ 0.5 half-width
        lower = val - w
        upper = val + w
        if name in _POSITIVE:               # enforce positivity
            lower = max(lower, abs_min)
        if name in ceiling:                 # honour hard caps
            upper = min(upper, ceiling[name])
        pri[name] = (lower, upper)
    return pri, log_weight_pos

def compute_bic(logL_max: float, k: int, n: int) -> float:
    """Bayesian Information Criterion (lower = preferred)."""
    return -2.0 * logL_max + k * np.log(n)

def downsample(data: NDArray[np.floating], f_factor=1, t_factor=1):
    """Block-average by integer factors along (freq, time)."""
    if f_factor == 1 and t_factor == 1:
        return data
    nf, nt = data.shape
    # Ensure dimensions are divisible by factors
    nf_new = nf - (nf % f_factor)
    nt_new = nt - (nt % t_factor)
    d = data[:nf_new, :nt_new].reshape(
        nf_new // f_factor, f_factor, nt_new // t_factor, t_factor
    )
    return d.mean(axis=(1, 3))

def goodness_of_fit(data: NDArray[np.floating],
                   model: NDArray[np.floating],
                   noise_std: NDArray[np.floating],
                   n_params: int) -> Dict[str, Any]:
    """Compute goodness-of-fit metrics."""
    residual = data - model
    noise_std_safe = np.clip(noise_std, 1e-9, None)[:, np.newaxis]
    
    chi2 = np.sum((residual / noise_std_safe) ** 2)
    ndof = data.size - n_params
    chi2_reduced = chi2 / ndof if ndof > 0 else np.inf

    residual_profile = np.sum(residual, axis=0)
    residual_profile -= np.mean(residual_profile)
    
    autocorr = np.correlate(residual_profile, residual_profile, mode='same')
    center_val = autocorr[len(autocorr)//2]
    if center_val > 0:
        autocorr /= center_val

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
    imshow_kw.setdefault("interpolation", "nearest")
    extent = [time[0], time[-1], freq[0], freq[-1]]
    return ax.imshow(dyn, extent=extent, **imshow_kw)
