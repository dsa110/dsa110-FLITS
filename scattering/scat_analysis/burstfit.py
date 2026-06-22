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
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import emcee
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.linalg import eigh
from scipy.special import erfcx

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

# ----------------------------------------------------------------------
# Constants & Helper Functions (Inlined from flits to avoid circular deps)
# ----------------------------------------------------------------------

# Cold-plasma dispersion delay in ms GHz^2 (pc cm^-3)^-1
DM_DELAY_MS = 4.148808

DM_SMEAR_MS = 8.3e-3  # intra-channel smearing, ms GHz⁻³ MHz⁻¹ (= 2*DM_DELAY_MS/1000; was 8.3e-6, a µs-in-s value mislabelled ms)

# Validation Thresholds
DM_MIN = 1e-3
DM_MAX = 3000
# delta_dm is a RESIDUAL DM error around the (catalog) dm_init the data was dedispersed
# at, so it is small -- not the full DM_MAX. A wide ±DM_MAX prior lets a broad (e.g.
# DM-smeared DSA) model escape into an unphysical huge-delta_dm degeneracy; bound it.
DM_RESID_MAX = 50.0
AMP_MIN = 0.01
AMP_MAX = 10000
WIDTH_MIN = 1e-4
WIDTH_MAX = 1000
ALPHA_GOOD_MIN = 3.0
ALPHA_MARGINAL_MAX = 6.0
CHI_SQ_RED_MARGINAL_MAX = 3.0
CHI_SQ_RED_SUSPICIOUSLY_LOW = 0.3
CHI_SQ_RED_GOOD_MAX = 1.5  # PASS ceiling: model fits to within the noise
CHI_SQ_RED_FAIL_MAX = 10.0  # above this the fit is rejected outright
R_SQ_POOR_MIN = 0.50
R_SQ_MARGINAL_MIN = 0.70
RESIDUAL_NORMALITY_PVALUE = 0.05
RESIDUAL_AUTOCORR_DW_MIN = 1.0
RESIDUAL_AUTOCORR_DW_MAX = 3.0


def log_normal_prior(x: float, mu: float, sigma: float) -> float:
    """Log-probability for log-normal distribution (for τ_ms or similar)."""
    if x <= 0.0:
        return -np.inf
    return -0.5 * ((np.log(x) - mu) / sigma) ** 2 - np.log(sigma * x)


def gaussian_prior(x: float, mu: float, sigma: float) -> float:
    """Log-probability for Gaussian (normal) distribution."""
    if sigma <= 0.0:
        return 0.0
    return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))


# ## FIX ##: Restored the set of physically non-negative parameters.
_POSITIVE = {"c0", "zeta", "tau_1ghz"}  # keep in ONE place only
_MIN_POS = 1e-6


log = logging.getLogger(__name__)
if not log.handlers:
    log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)


def analytic_gaussian_exp_convolution(t, mu, sig, tau):
    """
    Analytic convolution of a Gaussian G(t; mu, sig) and an exponential E(t; tau).

    This uses the erfcx-based stable formulation:
    f(t) = (1/2*tau) * exp(-(t-mu)^2 / (2*sig^2)) * erfcx(sig/(sqrt(2)*tau) - (t-mu)/(sqrt(2)*sig))

    Stability:
    If tau -> 0 (scattering is negligible compared to smearing), this returns G(t).
    If sig -> 0 (smearing is negligible compared to scattering), this returns E(t).
    """
    if t.ndim == 1:
        t = t[None, :]

    # Pre-calculate common terms
    t_minus_mu = t - mu

    # --- STABILITY GUARD: Gaussian Limit ---
    # If tau is extremely small relative to sig, convolution is just the Gaussian.
    # This prevents numerical overflow in erfcx(b) / inv_tau when tau -> 0.
    is_gauss = (tau < 1e-9) | (sig > 100 * tau)

    # Standard Gaussian part (unnormalized)
    gauss_exp = np.exp(-0.5 * (t_minus_mu / sig) ** 2)

    # Initialize result with Gaussian limit
    # f(t) = (1/sqrt(2pi)sig) * exp(-0.5*(t-mu/sig)^2)
    res = (1.0 / (np.sqrt(2.0 * np.pi) * sig)) * gauss_exp

    # Only evaluate the erfcx part where it is NOT a pure Gaussian
    # This avoids NaNs from inf * 0 in the deep tails.
    mask = ~is_gauss.squeeze()
    if np.any(mask):
        # We need to handle masks carefully for 2D broadcast
        # mask is (nfreq,), results are (nfreq, ntime)
        inv_tau_m = 1.0 / tau[mask]
        t_m = t_minus_mu[mask]
        sig_m = sig[mask]

        b = (sig_m / (np.sqrt(2.0) * tau[mask])) - (t_m / (np.sqrt(2.0) * sig_m))

        # --- NUMERICAL STABILITY FIX ---
        # When b is large negative (deep tail or tiny sigma), erfcx(b) overflows
        # and gauss_exp underflows. The product should be finite.
        # Use asymptotic form: gauss_exp * erfcx(b) ~= 2 * exp(k^2 - 2kx)
        # where k = sig/(sqrt(2)*tau) and x = (t-mu)/(sqrt(2)*sig)
        # k^2 - 2kx = 0.5*(sig/tau)^2 - (t-mu)/tau

        # Threshold for erfcx overflow (approx -26 for float64)
        safe_mask = b > -25.0

        res_m = np.zeros_like(b)
        inv_tau_expanded = np.broadcast_to(inv_tau_m, b.shape)

        # Safe region: use standard formula
        if np.any(safe_mask):
            g_exp = gauss_exp[mask]
            res_m[safe_mask] = (
                (0.5 * inv_tau_expanded[safe_mask]) * g_exp[safe_mask] * erfcx(b[safe_mask])
            )

        # Overflow region: use asymptotic approximation
        # res_m = (1/tau) * exp(0.5*(sig/tau)^2 - (t-mu)/tau)
        overflow_mask = ~safe_mask
        if np.any(overflow_mask):
            # k^2 term: 0.5 * (sig/tau)^2
            k2 = 0.5 * (sig_m / tau[mask]) ** 2
            # -2kx term: -(t-mu)/tau
            minus_2kx = -t_m / tau[mask]
            exponent = k2 + minus_2kx

            res_m[overflow_mask] = inv_tau_expanded[overflow_mask] * np.exp(exponent[overflow_mask])

        # Replace NaNs (if any remain in deep tails) with 0
        np.nan_to_num(res_m, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        res[mask] = res_m

    return res


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
    components: Sequence[str] | None = None,
    tau_prior: tuple[float, float] | None = None,
):
    """Module-level function for multiprocessing compatibility.

    Parameters
    ----------
    tau_prior : (mu, sigma) or None
        Log-normal prior on tau_1ghz. Default None (no prior).
    alpha_gauss : (mu, sigma) or None
        Gaussian prior on alpha (will be converted to tuple for apply_physical_priors).
    components : sequence of str or None
        List of model keys for mixed multi-component fitting (e.g. ["M0", "M3"]).
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
    if not (0.001 < tau_1ghz < 15):
        return -np.inf

    # 3. Jeffreys 1/x weight for positive params (still sampling in linear units)
    if log_weight_pos:
        for name, v in zip(names, theta):
            if name in _POSITIVE:
                logp += -np.log(max(v, 1e-30))

    # 4. Compute likelihood
    if components is not None:
        # --- Mixed Multi-Component Mode ---
        # Parse global delta_dm (only fitted if "M3" is present)
        delta_dm = 0.0
        if "delta_dm" in names:
            delta_dm = theta[names.index("delta_dm")]

        model_sum = np.zeros_like(model.data)

        for i, model_type in enumerate(components):
            suffix = f"_{i + 1}"

            # Helper to extract parameter for this component
            def get_p(root, default=0.0):
                full_name = f"{root}{suffix}"
                if full_name in names:
                    return theta[names.index(full_name)]
                return default

            # Core parameters (always present for all models)
            c0 = get_p("c0")
            t0 = get_p("t0")
            gamma = get_p("gamma", -1.6)

            # Model-dependent parameters
            zeta = get_p("zeta", 0.0)  # M1, M3
            tau_1ghz = get_p("tau_1ghz", 0.0)  # M2, M3
            alpha = get_p("alpha", 4.4)  # M3 (fixed to 4.4 for M2)

            p_i = FRBParams(
                c0=c0,
                t0=t0,
                gamma=gamma,
                zeta=zeta,
                tau_1ghz=tau_1ghz,
                alpha=alpha,
                delta_dm=delta_dm,
            )
            model_sum = model_sum + model(p_i, model_type)

        # Compute likelihood against summed model
        noise_std_safe = np.clip(model.noise_std, 1e-9, None)
        resid = (model.data - model_sum) / noise_std_safe[:, None]

        if likelihood_kind == "gaussian":
            return logp + (-0.5 * np.sum(resid**2))
        else:
            # student-t
            r2_over_nu = (resid**2) / float(student_nu)
            term = -0.5 * (float(student_nu) + 1.0) * np.log1p(r2_over_nu)
            const = -0.5 * model.data.size * np.log(float(student_nu) * np.pi) - np.sum(
                np.log(noise_std_safe)
            )
            return logp + const + float(np.sum(term))

    elif param_mode == "single":
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

    # Aliases for compatibility with flits.params
    @property
    def amplitude(self) -> float:
        return self.c0

    @property
    def width(self) -> float:
        return self.zeta

    @property
    def tau_alpha(self) -> float:
        return self.alpha

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
    def from_sequence(cls, seq: Sequence[float], model_key: str = "M3") -> FRBParams:
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
# Gaussian-process spectral marginal (diffractive scintillation)
# ----------------------------------------------------------------------
def _gp_amplitude_logL(
    ahat: NDArray[np.floating],
    v: NDArray[np.floating],
    freq_MHz: NDArray[np.floating],
    delta_nu_d_MHz: float,
    mu_degree: int = 1,
    sigma_g2: float | None = None,
) -> tuple[float, float, NDArray[np.floating], float]:
    """Spectral GP marginal over the matched-filter gain estimates `ahat`.

    Diffractive scintillation makes the per-channel gain estimates correlated in
    frequency with a Lorentzian ACF of half-width `delta_nu_d` (the scintillation
    bandwidth). We model the gain-estimate vector as

        ahat ~ N(mu, Sigma),  Sigma = sigma_g^2 C(delta_nu_d) + diag(v)
        C[f,f'] = 1 / (1 + ((nu_f - nu_f') / delta_nu_d)^2)   (unit-amplitude Lorentzian)
        mu = X beta  (smooth intrinsic envelope, Vandermonde of centred+scaled freq)

    mu is profiled by generalized least squares (flat prior on beta -> the
    +0.5 ln|X^T Sigma^-1 X| Jacobian term), and sigma_g^2 by 1-D maximum
    likelihood unless overridden via `sigma_g2`.

    Returns ``(logL_amp, sigma_g2_ml, mu, modulation_index)`` where ``logL_amp``
    is the SPECTRAL block only (the temporal -0.5 sum chi2min is added by the
    caller). It comprises

        -0.5 (ahat-mu)^T Sigma^-1 (ahat-mu)      [GP residual]
        -0.5 ln|Sigma|                            [GP Occam, generalizes -0.5 ln S_kk]
        +0.5 ln|X^T Sigma^-1 X|                   [GLS mu-profiling Jacobian]
        -0.5 sum_f ln(2 pi v_f)                   [ahat|g white-noise normalizer]

    Numerics: whiten the correlation by the diagonal, M = D^{-1/2} C D^{-1/2}
    = Q L Q^T (one symmetric eigendecomposition per call), then for any sigma_g^2
    both ln|Sigma| = sum ln v + sum ln(sigma_g^2 L_k + 1) and the quadratic /
    GLS forms are O(n) in the rotated basis. The sigma_g^2 sweep therefore costs
    one O(n^3) eigendecomposition + O(n) per inner evaluation (cf. derivation
    notes / SPEC.profile_strategy).
    """
    ahat = np.asarray(ahat, dtype=float)
    v = np.clip(np.asarray(v, dtype=float), 1e-30, None)
    nu = np.asarray(freq_MHz, dtype=float)
    n = ahat.size

    # Lorentzian correlation on live channels (unit amplitude).
    dnu = nu[:, None] - nu[None, :]
    C = 1.0 / (1.0 + (dnu / float(delta_nu_d_MHz)) ** 2)

    # Whiten by the noise diagonal: M = D^{-1/2} C D^{-1/2} = Q L Q^T.
    dinv_sqrt = 1.0 / np.sqrt(v)
    M = (dinv_sqrt[:, None] * C) * dinv_sqrt[None, :]
    M = 0.5 * (M + M.T)  # symmetrize against round-off before eigh
    L, Q = eigh(M)
    L = np.clip(L, 0.0, None)  # C is PSD; clip tiny negative round-off

    # Rotate the data and the design matrix into the eigenbasis ONCE.
    # Vandermonde of centred+scaled freq keeps X^T Sigma^-1 X well-conditioned.
    span = float(nu.max() - nu.min())
    nu_c = (nu - nu.mean()) / (span if span > 0 else 1.0)
    X = np.vander(nu_c, N=int(mu_degree) + 1, increasing=True)  # (n, p+1)
    a_w = Q.T @ (dinv_sqrt * ahat)  # Q^T D^{-1/2} ahat
    X_w = Q.T @ (dinv_sqrt[:, None] * X)  # Q^T D^{-1/2} X   (n, p+1)
    logdet_v = float(np.sum(np.log(v)))

    def _profile_mu(s2: float) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """GLS beta and rotated residual z = Q^T D^{-1/2} (ahat - mu) at sigma_g^2=s2."""
        d_eig = s2 * L + 1.0  # eigenvalues of D^{-1/2} Sigma D^{-1/2}
        inv_d = 1.0 / d_eig
        # X^T Sigma^-1 X = X_w^T diag(inv_d) X_w ; X^T Sigma^-1 ahat = X_w^T diag(inv_d) a_w
        XtSiX = (X_w * inv_d[:, None]).T @ X_w
        XtSia = (X_w * inv_d[:, None]).T @ a_w
        beta = np.linalg.solve(XtSiX, XtSia)
        z = a_w - X_w @ beta  # rotated residual
        return beta, z

    def _neglogL_s2(log_s2: float, return_full: bool = False):
        s2 = float(np.exp(log_s2))
        beta, z = _profile_mu(s2)
        d_eig = s2 * L + 1.0
        quad = float(np.sum(z * z / d_eig))
        logdet_sigma = logdet_v + float(np.sum(np.log(d_eig)))
        # GLS Jacobian (flat prior on beta): +0.5 ln|X^T Sigma^-1 X|
        inv_d = 1.0 / d_eig
        XtSiX = (X_w * inv_d[:, None]).T @ X_w
        sgn, logdet_XtSiX = np.linalg.slogdet(XtSiX)
        jac = 0.5 * logdet_XtSiX if sgn > 0 else -1e30
        logL = -0.5 * quad - 0.5 * logdet_sigma + jac - 0.5 * (n * np.log(2.0 * np.pi) + logdet_v)
        if return_full:
            return logL, beta, s2
        return -logL

    if sigma_g2 is not None:
        # Caller-forced amplitude (used by the decorrelated-wide flat-limit test).
        logL, beta, s2_ml = _neglogL_s2(np.log(float(sigma_g2)), return_full=True)
    else:
        # Bounded 1-D ML over log(sigma_g^2). Golden-section cannot diverge near
        # the (non-convex) unresolved boundary, unlike Newton. Range anchored on
        # the data scale: median(v) sets the noise floor, var(ahat) the ceiling.
        from scipy.optimize import minimize_scalar

        med_v = float(np.median(v))
        scale = max(med_v, float(np.var(ahat)), 1e-30)
        lo, hi = np.log(scale) - 18.0, np.log(scale) + 18.0
        res = minimize_scalar(
            _neglogL_s2, bounds=(lo, hi), method="bounded", options={"xatol": 1e-3}
        )
        logL, beta, s2_ml = _neglogL_s2(res.x, return_full=True)

    mu = X @ beta
    # Modulation index of the (envelope-removed) gain estimates: std of the
    # fractional residual. Independent sub-resolution scintillation probe.
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = (ahat - mu) / np.where(np.abs(mu) > 1e-30, mu, np.nan)
    mod_index = float(np.nanstd(frac)) if np.any(np.isfinite(frac)) else float("nan")
    return float(logL), float(s2_ml), mu, mod_index


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

        # Pre-calculate validity mask (exclude dead channels and NaNs)
        if self.data is not None and self.noise_std is not None:
            self.valid = (self.noise_std > 1e-9) & (np.isfinite(np.nanmean(self.data, axis=1)))
        else:
            self.valid = None

    def _dispersion_delay(
        self, dm_err: float = 0.0, ref_freq: float | None = None
    ) -> NDArray[np.floating]:
        if ref_freq is None:
            ref_freq = self.freq.max()
        return DM_DELAY_MS * dm_err * (self.freq**-self.beta - ref_freq**-self.beta)

    # ## REFACTOR ##: Smearing calculation is now more explicit.
    def _smearing_sigma(self, dm: float, zeta: float) -> NDArray[np.floating]:
        """Total Gaussian width from intrinsic pulse width (zeta) and
        intra-channel DM smearing.

        Intra-channel smearing is the dispersive delay across ONE *native* channel
        at the dedispersion DM, so self.df_MHz must be the NATIVE channel width
        (io.py sets it from telescope.df_MHz_raw, not the downsampled width). It is
        a boxcar of full width dt_DM; modelled as a Gaussian its variance-matched
        sigma is dt_DM/sqrt(12). Pass dm=0 for coherently-dedispersed data (CHIME,
        smearing removed) and the catalog DM for incoherently-dedispersed data (DSA).
        """
        # dt_DM (ms) = 8.3e-3 * DM(pc cm^-3) * df_MHz(native) * nu_GHz^-3  (Lorimer & Kramer 2005)
        dt_dm = DM_SMEAR_MS * dm * self.df_MHz * (self.freq**-3.0)
        sig_dm = dt_dm / np.sqrt(12.0)  # uniform (boxcar) -> Gaussian-equivalent sigma
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
            np.abs(self.data[:, idx] - np.median(self.data[:, idx], axis=1, keepdims=True)),
            axis=1,
        )
        return 1.4826 * mad  # Do not clip to 1e-6, allow 0 for dead channels

    def __call__(
        self, p: FRBParams, model_key: str = "M3", freq_subset: NDArray[np.bool_] | None = None
    ) -> NDArray[np.floating]:
        """Return model dynamic spectrum for parameters *p*.

        Args:
            p: Parameters
            model_key: Model variant
            freq_subset: Optional boolean mask of frequencies to evaluate.
                         If None, evaluates all frequencies.
        """
        if freq_subset is None:
            freq = self.freq
        else:
            freq = self.freq[freq_subset]

        n_freq = freq.size
        n_time = self.time.size

        ref_freq = np.median(self.freq)  # Keep global ref_freq for consistency
        amp = p.c0 * (freq / ref_freq) ** p.gamma

        # Dispersion delay
        dd_full = self._dispersion_delay(p.delta_dm)
        if freq_subset is None:
            mu = p.t0 + dd_full[:, None]
        else:
            mu = p.t0 + dd_full[freq_subset, None]

        if model_key in {"M1", "M3"}:
            sig_full = self._smearing_sigma(self.dm_init, p.zeta)
        else:
            sig_full = self._smearing_sigma(self.dm_init, 0.0)

        if freq_subset is None:
            sig = sig_full[:, None]
        else:
            sig = sig_full[freq_subset, None]

        # Guard against non-physical width
        sig = np.clip(sig, 1e-6, None)

        if model_key in {"M2", "M3"} and p.tau_1ghz > 1e-6:
            alpha = getattr(p, "alpha", 4.4)
            tau = p.tau_1ghz * (freq / 1.0) ** (-alpha)
            tau = np.clip(tau, 1e-6, None)[:, None]

            # Use analytic convolution for speed and precision
            return amp[:, None] * analytic_gaussian_exp_convolution(self.time, mu, sig, tau)

        # Non-scattering models (M0, M1) or negligible tau
        gauss = (1.0 / (np.sqrt(2.0 * np.pi) * sig)) * np.exp(-0.5 * ((self.time - mu) / sig) ** 2)
        # Normalize
        gauss_sum = np.sum(gauss, axis=1, keepdims=True)
        safe_gauss_sum = np.clip(gauss_sum, 1e-30, None)

        if model_key not in {"M0", "M1", "M2", "M3"}:
            raise ValueError(f"unknown model '{model_key}'")

        return amp[:, None] * (gauss / safe_gauss_sum)

    def log_likelihood(self, p: FRBParams, model: str = "M3") -> float:
        if self.data is None or self.noise_std is None:
            raise RuntimeError("need observed data + noise_std for likelihood")

        # Use pre-calculated validity mask
        if self.valid is None or not np.any(self.valid):
            return -np.inf

        # Broadcast noise to full shape before masking
        ns = self.noise_std
        if ns.ndim == 1:
            ns = ns[:, None]
        noise_std_full = np.broadcast_to(ns, self.data.shape)
        noise_valid = noise_std_full[self.valid]
        data_valid = self.data[self.valid]

        # Calculate model only for valid channels
        model_valid = self(p, model, freq_subset=self.valid)

        resid = (data_valid - model_valid) / noise_valid
        return -0.5 * np.sum(resid**2)

    def log_likelihood_gain_marginal(self, p: FRBParams, model: str = "M3") -> float:
        """Gaussian log-L with a per-channel amplitude (gain) marginalized analytically.

        The model factorizes as g_f * K_f(t): K_f is the UNIT-amplitude scattering
        kernel (evaluated with c0=1, gamma=0) carrying the per-channel temporal
        shape tau(f), and g_f is a free per-channel gain that absorbs the burst
        spectrum AND diffractive scintillation. With a flat prior on g_f, the gain
        integral is the matched-filter (F-statistic) marginal likelihood per channel

            ln L_f = -0.5 (S_dd - S_dk^2/S_kk)/sig_f^2 - 0.5 ln(S_kk) + 0.5 ln(2 pi sig_f^2)

        with S_dd=sum_t d^2, S_dk=sum_t d K, S_kk=sum_t K^2 over the on-pulse window.
        This whitens AMPLITUDE residuals (scintillation no longer inflates chi2) so
        the per-channel chi2 is a valid scattering goodness-of-fit gate; the profiled
        g_f (see gain_spectrum) is the scintillation/Delta-nu_d probe. A purely
        temporal (shape) misfit -- e.g. a model that cannot reach the burst peak --
        is NOT absorbed by g_f and remains visible.
        """
        if self.data is None or self.noise_std is None:
            raise RuntimeError("need observed data + noise_std for likelihood")
        if self.valid is None or not np.any(self.valid):
            return -np.inf

        K = self(replace(p, c0=1.0, gamma=0.0), model, freq_subset=self.valid)
        d = self.data[self.valid]
        sig = np.clip(self.noise_std[self.valid], 1e-9, None)
        var = sig**2
        S_dd = np.einsum("ij,ij->i", d, d)
        S_dk = np.einsum("ij,ij->i", d, K)
        S_kk = np.einsum("ij,ij->i", K, K)
        ok = S_kk > 1e-30
        S_kk_safe = np.where(ok, S_kk, 1.0)
        # gain=0 baseline (S_dd/var) where the model has no support in a channel
        chi2min = np.where(ok, (S_dd - S_dk**2 / S_kk_safe) / var, S_dd / var)
        occam = np.where(ok, -0.5 * np.log(S_kk_safe), 0.0)
        const = 0.5 * np.log(2.0 * np.pi * var)
        ll = float(np.sum(-0.5 * chi2min + occam + const))
        return ll if np.isfinite(ll) else -np.inf

    def gain_spectrum(self, p: FRBParams, model: str = "M3") -> NDArray[np.floating]:
        """Profiled per-channel gain g_f = S_dk/S_kk at parameters *p* (all channels).

        g_f is the matched-filter amplitude of the unit kernel K_f in each channel:
        the burst spectrum modulated by diffractive scintillation. Autocorrelating
        g_f over frequency yields the scintillation bandwidth Delta-nu_d (the second
        scattering probe). Channels with no model support return 0.
        """
        K = self(replace(p, c0=1.0, gamma=0.0), model)
        d = self.data
        S_dk = np.einsum("ij,ij->i", d, K)
        S_kk = np.einsum("ij,ij->i", K, K)
        return np.where(S_kk > 1e-30, S_dk / np.where(S_kk > 1e-30, S_kk, 1.0), 0.0)

    def log_likelihood_gain_marginal_gp(
        self,
        p: FRBParams,
        model: str = "M3",
        delta_nu_d_MHz: float | None = None,
        mu_degree: int = 1,
        sigma_g2: float | None = None,
    ) -> float:
        """Gain-marginal log-L with a frequency-correlated (scintillation) GP prior.

        Generalizes :meth:`log_likelihood_gain_marginal`: the per-channel gains
        g_f are no longer independent with a flat prior but a Gaussian process
        whose Lorentzian frequency ACF has half-width ``delta_nu_d_MHz`` (the
        diffractive scintillation bandwidth). The temporal matched-filter
        statistics are IDENTICAL to the flat path; only the amplitude block
        changes (see :func:`_gp_amplitude_logL`).

        Per consensus_logL the per-band marginal is

            logL = -0.5 sum_f chi2min_f                          [temporal, shared with flat]
                   + 0.5 sum_f ln(2 pi sig_f^2)                  [data/MF normalization const]
                   + logL_amp(ahat, v, nu, delta_nu_d)          [spectral GP block]

        where ahat_f = S_dk,f/S_kk,f, v_f = sig_f^2/S_kk,f and
        chi2min_f = (S_dd - S_dk^2/S_kk)/sig_f^2.

        Dispatch / fallback:
        * ``delta_nu_d_MHz is None`` -> returns ``log_likelihood_gain_marginal``
          VERBATIM (exact flat-prior regression anchor; the GP math is never
          duplicated).
        * fewer than ``mu_degree + 2`` live, model-supported channels -> same
          flat fallback (cannot profile mu + the GP simultaneously).
        """
        if delta_nu_d_MHz is None:
            return self.log_likelihood_gain_marginal(p, model)
        if self.data is None or self.noise_std is None:
            raise RuntimeError("need observed data + noise_std for likelihood")
        if self.valid is None or not np.any(self.valid):
            return -np.inf

        # --- temporal matched-filter statistics (identical to the flat path) ---
        K = self(replace(p, c0=1.0, gamma=0.0), model, freq_subset=self.valid)
        d = self.data[self.valid]
        sig = np.clip(self.noise_std[self.valid], 1e-9, None)
        var = sig**2
        S_dd = np.einsum("ij,ij->i", d, d)
        S_dk = np.einsum("ij,ij->i", d, K)
        S_kk = np.einsum("ij,ij->i", K, K)
        ok = S_kk > 1e-30  # channels with model support; GP runs on these
        n_supported = int(np.count_nonzero(ok))

        # Temporal residual is delta_nu_d-INDEPENDENT and shared with the flat path.
        S_kk_safe = np.where(ok, S_kk, 1.0)
        chi2min = np.where(ok, (S_dd - S_dk**2 / S_kk_safe) / var, S_dd / var)
        temporal = float(np.sum(-0.5 * chi2min))
        # data/MF normalization const (matches the flat code's +0.5 ln 2pi sig^2).
        const = float(np.sum(0.5 * np.log(2.0 * np.pi * var)))

        if n_supported < int(mu_degree) + 2:
            # Cannot profile mu + GP; fall back to the flat marginal (regression-safe).
            return self.log_likelihood_gain_marginal(p, model)

        # --- spectral GP block on the supported channels ---
        ahat = S_dk[ok] / S_kk[ok]
        v = var[ok] / S_kk[ok]
        nu_MHz = self.freq[self.valid][ok] * 1.0e3  # GHz -> MHz
        try:
            logL_amp, _s2, _mu, _m = _gp_amplitude_logL(
                ahat,
                v,
                nu_MHz,
                float(delta_nu_d_MHz),
                mu_degree=int(mu_degree),
                sigma_g2=sigma_g2,
            )
        except np.linalg.LinAlgError:
            return -np.inf

        ll = temporal + const + logL_amp
        return ll if np.isfinite(ll) else -np.inf

    def scint_gain_summary(
        self,
        p: FRBParams,
        model: str = "M3",
        delta_nu_d_MHz: float | None = None,
        mu_degree: int = 1,
    ) -> dict[str, Any]:
        """Recover the GLS smooth mean mu, residual gains, ML sigma_g^2 and the
        modulation index at parameters *p* and a given ``delta_nu_d_MHz``.

        Returns a dict over the LIVE+model-supported channels: ``freq_MHz``,
        ``ahat`` (matched-filter gain estimates), ``mu`` (smooth GLS envelope),
        ``resid`` (ahat - mu, the per-channel scintillation residual),
        ``sigma_g2``, ``modulation_index``, ``delta_nu_d_MHz``.
        """
        if self.data is None or self.noise_std is None:
            raise RuntimeError("need observed data + noise_std for likelihood")
        K = self(replace(p, c0=1.0, gamma=0.0), model, freq_subset=self.valid)
        d = self.data[self.valid]
        sig = np.clip(self.noise_std[self.valid], 1e-9, None)
        var = sig**2
        S_dk = np.einsum("ij,ij->i", d, K)
        S_kk = np.einsum("ij,ij->i", K, K)
        ok = S_kk > 1e-30
        ahat = S_dk[ok] / S_kk[ok]
        v = var[ok] / S_kk[ok]
        nu_MHz = self.freq[self.valid][ok] * 1.0e3
        if delta_nu_d_MHz is None or int(np.count_nonzero(ok)) < int(mu_degree) + 2:
            # No GP -> smooth mean is the flat-weighted polynomial, residual=ahat-mu.
            span = float(nu_MHz.max() - nu_MHz.min()) if nu_MHz.size else 1.0
            nu_c = (nu_MHz - nu_MHz.mean()) / (span if span > 0 else 1.0)
            X = np.vander(nu_c, N=int(mu_degree) + 1, increasing=True)
            beta, *_ = np.linalg.lstsq(X, ahat, rcond=None)
            mu = X @ beta
            return dict(
                freq_MHz=nu_MHz,
                ahat=ahat,
                mu=mu,
                resid=ahat - mu,
                sigma_g2=float("nan"),
                modulation_index=float(
                    np.nanstd((ahat - mu) / np.where(np.abs(mu) > 1e-30, mu, np.nan))
                ),
                delta_nu_d_MHz=delta_nu_d_MHz,
            )
        _ll, s2, mu, mod = _gp_amplitude_logL(
            ahat, v, nu_MHz, float(delta_nu_d_MHz), mu_degree=int(mu_degree)
        )
        return dict(
            freq_MHz=nu_MHz,
            ahat=ahat,
            mu=mu,
            resid=ahat - mu,
            sigma_g2=s2,
            modulation_index=mod,
            delta_nu_d_MHz=float(delta_nu_d_MHz),
        )

    def log_likelihood_student_t(self, p: FRBParams, model: str = "M3", nu: float = 5.0) -> float:
        if self.data is None or self.noise_std is None:
            raise RuntimeError("need observed data + noise_std for likelihood")
        # Use pre-calculated validity mask
        if self.valid is None or not np.any(self.valid):
            return -np.inf

        # Broadcast noise to full shape before masking
        ns = self.noise_std
        if ns.ndim == 1:
            ns = ns[:, None]
        noise_std_full = np.broadcast_to(ns, self.data.shape)
        noise_valid = noise_std_full[self.valid]
        data_valid = self.data[self.valid]
        model_valid = self(p, model, freq_subset=self.valid)

        resid = (data_valid - model_valid) / noise_valid

        # Student-t log-pdf up to constant per element
        # logL = sum( - (nu+1)/2 * log(1 + r^2/nu) ) - N * 0.5*log(nu*pi) - sum(log(sigma))
        r2_over_nu = (resid**2) / nu
        term = -0.5 * (nu + 1.0) * np.log1p(r2_over_nu)

        # Constant term accounts for valid pixels only
        n_pix = data_valid.size
        const = (
            -0.5 * n_pix * np.log(nu * np.pi) - np.sum(np.log(noise_valid)) * data_valid.shape[1]
        )

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
        priors: dict[str, tuple[float, float]],
        *,
        n_steps: int = 1000,
        n_walkers_mult: int = 8,
        pool=None,
        log_weight_pos=False,
        sample_log_params: bool = True,
        alpha_prior: tuple[float, float] | None = None,
        likelihood_kind: str = "gaussian",
        student_nu: float = 5.0,
        walker_width_frac: float = 0.01,
        components: Sequence[str] | None = None,
        **kwargs,
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
        self.components = components

        # Pre-build order if components are provided
        if self.components:
            self.build_mixed_model_order(self.components)

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
        # Use "mixed" key if components are provided
        if self.components:
            model_key = "mixed"

        names = self._ORDER[model_key] if model_key in self._ORDER else self.custom_order[model_key]
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
                "multi" if model_key in ["M3_multi", "mixed"] else "single",
                len(names) if model_key == "M3_multi" else 1,
                self.components,
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

    def build_mixed_model_order(self, components: Sequence[str]) -> tuple[str, ...]:
        """Build parameter list for mixed multi-component models.

        Parameters
        ----------
        components : list of str
            List of model keys (e.g. ["M0", "M3", "M1"]).

        Returns
        -------
        tuple of str
            Ordered parameter names.
        """
        order = []

        # 1. Global Parameters
        # Only fit delta_dm if at least one component is M3 (the full model)
        if any(c == "M3" for c in components):
            order.append("delta_dm")

        # 2. Per-Component Parameters
        for i, c in enumerate(components):
            idx = i + 1
            # Core parameters (all models)
            order.extend([f"c0_{idx}", f"t0_{idx}", f"gamma_{idx}"])

            # Intrinsic width (M1, M3)
            if c in ["M1", "M3"]:
                order.append(f"zeta_{idx}")

            # Scattering timescale (M2, M3)
            if c in ["M2", "M3"]:
                order.append(f"tau_1ghz_{idx}")

            # Scattering index (M3 only - M2 has fixed alpha)
            if c == "M3":
                order.append(f"alpha_{idx}")

        self.custom_order["mixed"] = tuple(order)
        return self.custom_order["mixed"]


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
    init: FRBParams,
    *,
    scale: float = 6.0,  # half-width multiplier (wide!)
    abs_min: float = _MIN_POS,  # floor for positive parameters
    abs_max: dict[str, float] | None = None,  # optional hard ceilings
    log_weight_pos: bool = False,  # True → Jeffreys p(x)∝1/x   (still linear sampling)
    absolute_bounds: bool = False,  # True → init-independent physical priors (for nested sampling)
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
        "delta_dm": (
            -DM_RESID_MAX,
            DM_RESID_MAX,
        ),  # residual DM around dm_init (small), not full DM_MAX
        "gamma": (-5.0, 5.0),  # Spectral index bounds
        "t0": (
            init.t0 - 2 * max(init.tau_1ghz, 10.0),
            init.t0 + 2 * max(init.tau_1ghz, 10.0),
        ),  # Dynamic t0 window
        "alpha": (ALPHA_GOOD_MIN, ALPHA_MARGINAL_MAX),  # Scattering index
    }

    ceiling = abs_max or {}

    for name, val in asdict(init).items():
        # Determine specific bounds for this parameter
        if name in param_bounds:
            hard_min, hard_max = param_bounds[name]
        else:
            hard_min, hard_max = abs_min, 1e6

        if absolute_bounds and name != "t0":
            # Init-independent physical prior: use the hard bounds directly.
            # The init-anchored val ± scale·|val| window collapses to ~0 half-width
            # when val ≈ 0 (e.g. gamma=0 → [-0.5, 0.5]), which can exclude the true
            # value and trap a global sampler in a wrong mode (observed: a trivial
            # init railed M3 to τ→0, ΔlnZ≈320 vs a data-driven init, purely because
            # its gamma=0 window excluded the true γ≈-2.6). For evidence-based model
            # selection the prior MUST NOT depend on the init at all. t0 is excluded:
            # its window is anchored on init.t0, which is the data profile-peak
            # estimate (not a fit value), so it is already init-robust.
            lower, upper = hard_min, hard_max
        else:
            # Calculate dynamic window around init, but clamp to hard bounds
            # (appropriate for walker-based methods that start near init).
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
    d = data[:nf_new, :nt_new].reshape(nf_new // f_factor, f_factor, nt_new // t_factor, t_factor)
    return d.mean(axis=(1, 3))


def classify_fit_quality(
    chi2_reduced: float,
    r_squared: float | None = None,
    normality_pvalue: float | None = None,
) -> tuple[str, list]:
    """Classify fit quality from chi2_reduced (PASS / MARGINAL / FAIL).

    The reduced chi-squared is the primary, trustworthy gate when the noise is
    well estimated: a value near 1 means the model fits to within the noise. The
    noise-weighted R^2 is deliberately NOT a gate here -- for faint (low-S/N)
    bursts the signal variance is small relative to the noise, so R^2 is low even
    for a correct model (a fit at chi2_red ~ 1 can have R^2 well below 0.5).
    Treating R^2 < 0.5 as a hard failure (the previous behavior) spuriously
    rejected good faint-burst fits. R^2 and the residual-normality p-value are
    reported as informational notes only; note that at the ~10^4-pixel sample
    sizes here the Shapiro normality p-value is ~0 even for excellent fits.

    Returns (flag, notes) where flag is "PASS", "MARGINAL", or "FAIL".
    """
    notes: list = []
    if chi2_reduced is None or not np.isfinite(chi2_reduced):
        return "FAIL", ["Non-finite chi2_red"]

    if chi2_reduced > CHI_SQ_RED_FAIL_MAX:
        flag = "FAIL"
        notes.append(f"Catastrophic chi2_red ({chi2_reduced:.1f})")
    elif chi2_reduced > CHI_SQ_RED_GOOD_MAX:
        flag = "MARGINAL"
        notes.append(f"Elevated chi2_red ({chi2_reduced:.2f})")
    elif chi2_reduced < CHI_SQ_RED_SUSPICIOUSLY_LOW:
        flag = "MARGINAL"
        notes.append(f"Suspiciously low chi2_red ({chi2_reduced:.2f}); noise may be overestimated")
    else:
        flag = "PASS"

    # Informational only -- these do NOT change the flag.
    if r_squared is not None and np.isfinite(r_squared) and r_squared < R_SQ_MARGINAL_MIN:
        notes.append(
            f"Low weighted R2 ({r_squared:.3f}) [informational; expected for low-S/N bursts]"
        )
    if (
        normality_pvalue is not None
        and np.isfinite(normality_pvalue)
        and normality_pvalue < RESIDUAL_NORMALITY_PVALUE
    ):
        notes.append(f"Non-normal residuals (p={normality_pvalue:.1e}) [informational at large N]")
    return flag, notes


def goodness_of_fit(
    data: NDArray[np.floating],
    model: NDArray[np.floating],
    noise_std: NDArray[np.floating],
    n_params: int,
) -> dict[str, Any]:
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
    # SciPy stats.shapiro warns when N > 5000 as the p-value computation becomes
    # inaccurate. We calculate a slice step to strictly cap the sample size at 5000.
    data_flat = resid_valid.flatten()
    step = max(1, int(np.ceil(len(data_flat) / 5000)))
    test_resids = data_flat[::step]
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
    # bias_pass = bias_nsigma < 3.0

    # 5. Durbin-Watson (Autocorrelation)
    # We can compute it on the flattened valid array
    diffs = np.diff(data_flat)
    dw_stat = np.sum(diffs**2) / np.sum(data_flat**2)
    # autocorr_pass = RESIDUAL_AUTOCORR_DW_MIN <= dw_stat <= RESIDUAL_AUTOCORR_DW_MAX

    # 6. Quality Flag -- driven by reduced chi-squared (the trustworthy gate when
    # noise is well estimated). R^2 and normality are informational only; see
    # classify_fit_quality for why noise-weighted R^2 is not a gate for faint
    # bursts.
    quality, notes = classify_fit_quality(chi2_reduced, r_squared, normality_pvalue)

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
        "residual_autocorr": norm_autocorr,  # Keep array for plotting
        "normality_pvalue": float(normality_pvalue),
        "bias_nsigma": float(bias_nsigma),
        "durbin_watson": float(dw_stat),
        "quality_flag": quality,
        "validation_notes": notes,
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
