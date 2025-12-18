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
from scipy import stats

__all__ = [
    "FRBParams",
    "FRBModel",
    "FRBFitter",
    "compute_bic",
    "build_priors",
    "downsample",
    "plot_dynamic",
    "goodness_of_fit",
    "gelman_rubin",
]

from flits.common.constants import DM_DELAY_MS
from flits.scattering import (
    scatter_broaden,
    tau_per_freq,
    log_normal_prior,
    gaussian_prior,
)

from .validation_thresholds import (
    DM_MIN, DM_MAX,
    AMP_MIN, AMP_MAX,
    WIDTH_MIN, WIDTH_MAX,
    ALPHA_GOOD_MIN, ALPHA_MARGINAL_MAX,
    CHI_SQ_RED_MARGINAL_MAX,
    CHI_SQ_RED_SUSPICIOUSLY_LOW,
    R_SQ_POOR_MIN,
    R_SQ_MARGINAL_MIN,
    RESIDUAL_NORMALITY_PVALUE,
    RESIDUAL_AUTOCORR_DW_MIN,
    RESIDUAL_AUTOCORR_DW_MAX,
)

# ----------------------------------------------------------------------
# Module-level constants
# ----------------------------------------------------------------------
DM_SMEAR_MS = 8.3e-6  # intra-channel smearing, ms GHz⁻³ MHz⁻¹ -> this is a more common formulation
# The previous value was likely a typo or for a specific setup.
# This standard form is: 8.3 * 1e6 * DM * dnu_MHz / nu_GHz**3 / 1e3
# We will apply dnu_MHz later, so this constant is 8.3e-6 ms GHz^3 MHz^-1

# ## FIX ##: Restored the set of physically non-negative parameters.
_POSITIVE = {"c0", "zeta", "tau_1ghz"}  # keep in ONE place only
_MIN_POS = 1e-6


log = logging.getLogger(__name__)
if not log.handlers:
    log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

# ----------------------------------------------------------------------
# Log-probability wrapper
# ----------------------------------------------------------------------


# ## REFACTOR ##: This is the correct, module-level wrapper for emcee + multiprocessing.
# It takes the model object itself as an argument, making it stateless and pickleable.
def _log_prob_wrapper(
    theta_raw,
    model,
    priors,
    order,
    key,
    log_weight_pos,
    log_params_names=None,
    alpha_gauss=None,
    likelihood_kind="gaussian",
    student_nu=5.0,
    param_mode="single",
    K: int = 1,
    tau_prior: tuple[float, float] | None = None,
):
    """Module-level function for multiprocessing compatibility.

    Parameters
    ----------
    tau_prior : (mu, sigma) or None
        Log-normal prior on tau_1ghz. Default None (no prior).
    alpha_gauss : (mu, sigma) or None
        Gaussian prior on alpha (will be converted to tuple for apply_physical_priors).
    """
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

    # 2. Apply physical priors on scattering parameters
    logp = apply_physical_priors(
        logp,
        theta,
        names,
        tau_prior=tau_prior,
        alpha_prior=alpha_gauss,  # backward compat: alpha_gauss → alpha_prior
    )
    if logp == -np.inf:
        return -np.inf

    # Extract tau_1ghz if it's a parameter in the current model
    tau_1ghz = 0.0
    if "tau_1ghz" in names:
        tau_1ghz = theta[names.index("tau_1ghz")]
    if not (0.001 < tau_1ghz < 15): return -np.inf

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
            return logp + model.log_likelihood_student_t(
                params, key, nu=float(student_nu)
            )
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
        for idx in range(1, int(K) + 1):
            c0 = get(f"c0_{idx}")
            t0 = get(f"t0_{idx}")
            zeta = get(f"zeta_{idx}")
            if c0 is None or t0 is None or zeta is None:
                continue
            p_i = FRBParams(
                c0=c0,
                t0=t0,
                gamma=gamma,
                zeta=zeta,
                tau_1ghz=tau1,
                alpha=alpha,
                delta_dm=delta_dm,
            )
            model_sum = model_sum + model(p_i, "M3")
        # compute likelihood against summed model
        if likelihood_kind == "gaussian":
            noise_std_safe = np.clip(model.noise_std, 1e-9, None)
            resid = (model.data - model_sum) / noise_std_safe[:, None]
            return logp + (-0.5 * np.sum(resid**2))
        else:
            # student-t
            noise_std_safe = np.clip(model.noise_std, 1e-9, None)
            resid = (model.data - model_sum) / noise_std_safe[:, None]
            r2_over_nu = (resid**2) / float(student_nu)
            term = -0.5 * (float(student_nu) + 1.0) * np.log1p(r2_over_nu)
            const = -0.5 * model.data.size * np.log(float(student_nu) * np.pi) - np.sum(
                np.log(noise_std_safe)
            )
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
    alpha: float = 4.4  # frequency scaling exponent τ ∝ ν^{-alpha}
    delta_dm: float = 0.0  # residual DM error around dm_init

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
    def from_sequence(cls, seq: Sequence[float], model_key: str = "M3") -> "FRBParams":
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
        df_MHz: float = 0.390625,  # 0.03051757812, #Channel width is needed for smearing
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
        return DM_DELAY_MS * dm_err * (self.freq**-self.beta - ref_freq**-self.beta)

    # ## REFACTOR ##: Smearing calculation is now more explicit.
    def _smearing_sigma(self, dm: float, zeta: float) -> NDArray[np.floating]:
        """Calculates total Gaussian width from intrinsic pulse width (zeta)
        and intra-channel DM smearing."""
        # DM smearing time (ms) = 8.3e-6 * DM * df_MHz / nu_GHz^3
        sig_dm = DM_SMEAR_MS * dm * self.df_MHz * (self.freq**-3.0)
        # Add in quadrature with intrinsic width
        return np.hypot(sig_dm, zeta)

    def _estimate_noise(self, off_pulse):
        if self.data is None:
            return None
        if off_pulse is None:
            q = self.time.size // 4
            idx = np.r_[0:q, -q:0]
        else:
            idx = np.asarray(off_pulse)

        # Ensure indices are within bounds
        idx = idx[idx < self.data.shape[1]]

        mad = np.median(
            np.abs(
                self.data[:, idx] - np.median(self.data[:, idx], axis=1, keepdims=True)
            ),
            axis=1,
        )
        return 1.4826 * mad  # Do not clip to 1e-6, allow 0 for dead channels

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

        gauss = (1 / (np.sqrt(2 * np.pi) * sig)) * np.exp(
            -0.5 * ((self.time - mu) / sig) ** 2
        )

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

        # Ignore channels with ~0 noise (dead/masked)
        valid = self.noise_std > 1e-9
        if not np.any(valid):
            return -np.inf

        noise_valid = self.noise_std[valid]
        data_valid = self.data[valid]
        
        # Calculate model only for valid channels is hard due to optimized call structure, 
        # so calculate full model and slice
        model_out = self(p, model)
        model_valid = model_out[valid]
        
        resid = (data_valid - model_valid) / noise_valid[:, None]
        return -0.5 * np.sum(resid**2)

    def log_likelihood_student_t(
        self, p: FRBParams, model: str = "M3", nu: float = 5.0
    ) -> float:
        if self.data is None or self.noise_std is None:
            raise RuntimeError("need observed data + noise_std for likelihood")
        # Ignore channels with ~0 noise
        valid = self.noise_std > 1e-9
        if not np.any(valid):
             return -np.inf
             
        noise_valid = self.noise_std[valid]
        data_valid = self.data[valid]
        model_valid = self(p, model)[valid]
        
        resid = (data_valid - model_valid) / noise_valid[:, None]
        
        # Student-t log-pdf up to constant per element
        # logL = sum( - (nu+1)/2 * log(1 + r^2/nu) ) - N * 0.5*log(nu*pi) - sum(log(sigma))
        r2_over_nu = (resid**2) / nu
        term = -0.5 * (nu + 1.0) * np.log1p(r2_over_nu)
        
        # Constant term accounts for valid pixels only
        n_pix = data_valid.size
        const = -0.5 * n_pix * np.log(nu * np.pi) - np.sum(np.log(noise_valid)) * data_valid.shape[1]
        
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
        **kwargs,
    ):
        self.model = model
        self.priors = priors
        self.n_steps = n_steps
        self.n_walkers_mult = n_walkers_mult
        self.pool = pool
        self.log_weight_pos = log_weight_pos
        self.sample_log_params = sample_log_params
        self._log_param_names = (
            {"c0", "zeta", "tau_1ghz"} if sample_log_params else set()
        )
        self.alpha_prior = alpha_prior
        self.likelihood_kind = likelihood_kind
        self.student_nu = student_nu
        self.custom_order: dict[str, tuple[str, ...]] = {}
        self.walker_width_frac = max(1e-4, float(walker_width_frac))

    def _is_log_param(self, name: str) -> bool:
        """Check if a parameter should be sampled in log-space.

        Handles both base parameters ('c0', 'zeta', 'tau_1ghz') and
        multi-component variants ('c0_1', 'zeta_2', etc.) consistently.
        """
        if not self.sample_log_params:
            return False
        # For multi-component params like 'c0_1', check if root matches
        root = name.split("_")[0]
        return root in self._log_param_names or name in self._log_param_names

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

            # Use consistent log-param checking
            if self._is_log_param(n):
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

        # Clip in appropriate domain using consistent log-param checking
        for j, n in enumerate(names):
            if self._is_log_param(n):
                walkers[:, j] = np.clip(
                    walkers[:, j],
                    np.log(max(lower[j], 1e-30)),
                    np.log(max(upper[j], 1e-30)),
                )
            else:
                walkers[:, j] = np.clip(walkers[:, j], lower[j], upper[j])

        return walkers

    def sample(self, p0, model_key: str = "M3"):
        """Run the MCMC sampler."""
        names = (
            self._ORDER[model_key]
            if model_key in self._ORDER
            else self.custom_order[model_key]
        )
        ndim = len(names)
        nwalk = max(self.n_walkers_mult * ndim, 2 * ndim)

        p_walkers = self._init_walkers(p0, model_key, nwalk)

        # Build list of log-space params using consistent checking
        log_param_fullnames = [n for n in names if self._is_log_param(n)]

        sampler = emcee.EnsembleSampler(
            nwalkers=nwalk,
            ndim=ndim,
            log_prob_fn=_log_prob_wrapper,
            args=(
                self.model,
                self.priors,
                (self.custom_order if model_key in self.custom_order else self._ORDER),
                model_key,
                self.log_weight_pos,
                log_param_fullnames,
                self.alpha_prior,
                self.likelihood_kind,
                self.student_nu,
                "multi" if model_key == "M3_multi" else "single",
                len(names) if model_key == "M3_multi" else 1,
            ),
            pool=self.pool,
        )
        sampler.run_mcmc(p_walkers, self.n_steps, progress=True)
        return sampler

    def build_multicomp_order(self, K: int) -> tuple[str, ...]:
        order = ["gamma", "tau_1ghz", "alpha", "delta_dm"]
        for i in range(1, K + 1):
            order.extend([f"c0_{i}", f"t0_{i}", f"zeta_{i}"])
        self.custom_order["M3_multi"] = tuple(order)
        return self.custom_order["M3_multi"]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def gelman_rubin(sampler, discard: int = 0) -> dict[str, float]:
    """Compute Gelman-Rubin R̂ statistic for MCMC convergence diagnostics.

    R̂ < 1.1 is typically considered evidence of convergence.
    R̂ < 1.01 is considered excellent convergence.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        The sampler object after running MCMC.
    discard : int
        Number of steps to discard as burn-in.

    Returns
    -------
    dict
        Dictionary mapping parameter index to R̂ value.
        Also includes 'max_rhat' and 'converged' keys.
    """
    # Get chains: shape (n_steps, n_walkers, n_params)
    chain = sampler.get_chain(discard=discard)
    n_steps, n_walkers, n_params = chain.shape

    if n_steps < 10:
        return {"max_rhat": np.inf, "converged": False, "warning": "Too few steps"}

    results = {}
    rhats = []

    for i in range(n_params):
        # For each parameter, compute R̂
        # Treat each walker as a separate chain
        chains = chain[:, :, i].T  # shape: (n_walkers, n_steps)

        n = chains.shape[1]  # number of samples per chain
        m = chains.shape[0]  # number of chains (walkers)

        # Chain means
        chain_means = np.mean(chains, axis=1)
        # Overall mean
        overall_mean = np.mean(chain_means)

        # Between-chain variance
        B = n / (m - 1) * np.sum((chain_means - overall_mean) ** 2)

        # Within-chain variance
        chain_vars = np.var(chains, axis=1, ddof=1)
        W = np.mean(chain_vars)

        # Pooled variance estimate
        if W < 1e-30:
            # Chains haven't moved - not converged
            rhat = np.inf
        else:
            var_hat = (1 - 1 / n) * W + (1 / n) * B
            rhat = np.sqrt(var_hat / W)

        results[f"param_{i}"] = float(rhat)
        rhats.append(rhat)

    max_rhat = float(np.max(rhats))
    results["max_rhat"] = max_rhat
    results["converged"] = max_rhat < 1.1
    results["well_converged"] = max_rhat < 1.01

    return results


def apply_physical_priors(
    log_prob: float,
    theta: Sequence[float],
    names: Sequence[str],
    *,
    tau_prior: tuple[float, float] | None = None,
    alpha_prior: tuple[float, float] | None = None,
) -> float:
    """Apply optional physical priors on scattering parameters.

    Parameters
    ----------
    log_prob : float
        Current log-probability (before physical priors).
    theta : sequence of float
        Parameter values.
    names : sequence of str
        Parameter names corresponding to theta.
    tau_prior : (mu, sigma) or None
        Log-normal prior on tau_1ghz: log(τ) ~ N(mu, sigma^2).
        If None, no prior applied.
    alpha_prior : (mu, sigma) or None
        Gaussian prior on alpha: α ~ N(mu, sigma^2).
        Default Kolmogorov: α ≈ 4.4. If None, no prior applied.

    Returns
    -------
    float
        Updated log-probability with physical priors included.
    """
    for name, val in zip(names, theta):
        # Log-normal prior on tau_1ghz (scattering timescale)
        if name == "tau_1ghz" and tau_prior is not None:
            mu, sigma = tau_prior
            if val <= 0.0:
                return -np.inf
            log_prob += log_normal_prior(val, mu, sigma)

        # Gaussian prior on alpha (frequency scaling exponent)
        elif name == "alpha" and alpha_prior is not None:
            mu, sigma = alpha_prior
            log_prob += gaussian_prior(val, mu, sigma)

    return log_prob


def build_priors(
    init: "FRBParams",
    *,
    scale: float = 6.0,  # half-width multiplier (wide!)
    abs_min: float = _MIN_POS,  # floor for positive parameters
    abs_max: dict[str, float] | None = None,  # optional hard ceilings
    log_weight_pos: bool = False,  # True → Jeffreys p(x)∝1/x   (still linear sampling)
) -> tuple[dict[str, tuple[float, float]], bool]:
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
    pri = {}
    
    # Map parameters to their specific bounds
    # If not in this map, use defaults
    param_bounds = {
        "c0": (AMP_MIN, AMP_MAX),
        "tau_1ghz": (WIDTH_MIN, WIDTH_MAX),
        "zeta": (WIDTH_MIN, WIDTH_MAX),
        "delta_dm": (-DM_MAX, DM_MAX), # Delta DM can be negative but bounded
        "gamma": (-5.0, 5.0), # Spectral index bounds
        "t0": (init.t0 - 2*max(init.tau_1ghz, 10.0), init.t0 + 2*max(init.tau_1ghz, 10.0)), # Dynamic t0 window
        "alpha": (ALPHA_GOOD_MIN, ALPHA_MARGINAL_MAX) # Scattering index
    }
    
    ceiling = abs_max or {}
    
    for name, val in asdict(init).items():
        # Determine specific bounds for this parameter
        if name in param_bounds:
            hard_min, hard_max = param_bounds[name]
        else:
             hard_min, hard_max = abs_min, 1e6

        # Calculate dynamic window around init, but clamp to hard bounds
        w = max(scale * max(abs(val), 1e-3), 0.5)
        lower = max(val - w, hard_min)
        upper = min(val + w, hard_max)
        
        # Override with explicit ceiling if provided
        if name in ceiling:
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


def goodness_of_fit(
    data: NDArray[np.floating],
    model: NDArray[np.floating],
    noise_std: NDArray[np.floating],
    n_params: int,
) -> Dict[str, Any]:
    """Compute comprehensive goodness-of-fit metrics and validation flags."""
    residual = data - model
    # Ensure noise broadcast correctly
    noise_std_safe = np.clip(np.atleast_1d(noise_std), 1e-9, None)
    if noise_std_safe.ndim == 1 and data.ndim == 2:
         noise_std_safe = noise_std_safe[:, np.newaxis]

    # Create valid mask based on noise threshold
    valid_mask = noise_std > 1e-9
    # Ensure mask broadcasts to data shape if needed
    if valid_mask.ndim == 1 and data.ndim == 2:
        valid_mask = valid_mask[:, np.newaxis]
    valid_mask = np.broadcast_to(valid_mask, data.shape)

    # Use only valid pixels for statistics
    resid_valid = residual[valid_mask]
    # Broadcast noise_std_safe to data shape before masking
    noise_std_full = np.broadcast_to(noise_std_safe, data.shape)
    noise_valid = noise_std_full[valid_mask]
    data_valid = data[valid_mask]

    n_valid = resid_valid.size
    
    # 1. Chi-squared
    chi2 = np.sum((resid_valid / noise_valid) ** 2)
    ndof = n_valid - n_params
    chi2_reduced = chi2 / ndof if ndof > 0 else np.inf

    # 2. R-squared
    ss_res = np.sum((resid_valid / noise_valid) ** 2)
    # Weighted calculation on valid data only
    weights = 1.0 / noise_valid**2
    mean_data = np.average(data_valid, weights=weights)
    ss_tot = np.sum(((data_valid - mean_data) / noise_valid) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else -np.inf

    # 3. Normality Test (Shapiro-Wilk)
    # Downsample for speed if needed
    data_flat = resid_valid.flatten()
    test_resids = data_flat[::max(1, len(data_flat)//5000)]
    try:
        _, normality_pvalue = stats.shapiro(test_resids)
        normality_pass = normality_pvalue > RESIDUAL_NORMALITY_PVALUE
    except Exception:
        normality_pvalue = 0.0
        normality_pass = False

    # 4. Bias Test
    bias_mean = np.mean(resid_valid)
    bias_std = np.std(resid_valid)
    bias_sem = bias_std / np.sqrt(resid_valid.size)
    bias_nsigma = abs(bias_mean) / bias_sem if bias_sem > 0 else 0.0
    bias_pass = bias_nsigma < 3.0

    # 5. Durbin-Watson (Autocorrelation)
    # We can compute it on the flattened valid array
    diffs = np.diff(data_flat)
    dw_stat = np.sum(diffs ** 2) / np.sum(data_flat ** 2)
    autocorr_pass = RESIDUAL_AUTOCORR_DW_MIN <= dw_stat <= RESIDUAL_AUTOCORR_DW_MAX
    
    # 6. Quality Flag
    quality = "PASS"
    notes = []
    
    if chi2_reduced > 10.0:
        quality = "FAIL"
        notes.append(f"Catastrophic chi2_red ({chi2_reduced:.1f})")
    elif chi2_reduced > CHI_SQ_RED_MARGINAL_MAX:
        quality = "FAIL" # Strict failure per guide
        notes.append(f"High chi2_red ({chi2_reduced:.2f})")
    elif chi2_reduced < CHI_SQ_RED_SUSPICIOUSLY_LOW:
        quality = "MARGINAL"
        notes.append(f"Low chi2_red ({chi2_reduced:.2f})")
        
    if r_squared < R_SQ_POOR_MIN:
        quality = "FAIL"
        notes.append(f"Poor R2 ({r_squared:.3f})")
    elif r_squared < R_SQ_MARGINAL_MIN:
        if quality == "PASS": quality = "MARGINAL"
        notes.append(f"Marginal R2 ({r_squared:.3f})")
        
    if not normality_pass:
        if quality == "PASS": quality = "MARGINAL"
        notes.append(f"Non-normal residuals (p={normality_pvalue:.1e})")

    # Autocorrelation of profile
    residual_profile = np.sum(residual, axis=0)
    residual_profile -= np.mean(residual_profile)
    start_autocorr = np.correlate(residual_profile, residual_profile, mode="same")
    center_val = start_autocorr[len(start_autocorr) // 2]
    norm_autocorr = start_autocorr / center_val if center_val > 0 else start_autocorr

    return {
        "chi2": float(chi2),
        "chi2_reduced": float(chi2_reduced),
        "r_squared": float(r_squared),
        "ndof": int(ndof),
        "residual_rms": float(np.std(resid_valid)),
        "residual_autocorr": norm_autocorr, # Keep array for plotting
        "normality_pvalue": float(normality_pvalue),
        "bias_nsigma": float(bias_nsigma),
        "durbin_watson": float(dw_stat),
        "quality_flag": quality,
        "validation_notes": notes
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
