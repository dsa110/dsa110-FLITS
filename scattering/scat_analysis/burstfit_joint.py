"""
burstfit_joint.py
=================

Joint two-telescope scattering fit (CHIME ~0.6 GHz + DSA ~1.4 GHz).

Single-band M3 fits with alpha fixed = 4 give *inconsistent* tau_1ghz between
CHIME and DSA for the same sightline (observed up to ~15x), which is only
possible if the scattering index alpha != 4. A single band cannot separate
alpha from tau (the tau-alpha degeneracy): tau(nu) = tau_1ghz * nu^-alpha, and
one band fixes the product at its own frequency, not the slope.

This module fits both bands simultaneously with a *shared* (tau_1ghz, alpha) and
*per-telescope* (c0, t0, gamma, zeta, delta_dm). The ~1 GHz lever arm between
the two bands measures alpha directly: the ratio tau_C/tau_D pins the slope,
the shared tau_1ghz pins the normalization at 1 GHz.

12-parameter vector (M3 both bands, alpha free):

    [tau_1ghz, alpha | c0_C, t0_C, gamma_C, zeta_C, ddm_C | c0_D, t0_D, gamma_D, zeta_D, ddm_D]
     ^shared sightline   ^CHIME intrinsic/timing            ^DSA intrinsic/timing

Independent noise -> the joint log-likelihood is the sum of the two single-band
Gaussian log-likelihoods (reuses the nch^2-fixed FRBModel.log_likelihood).

zeta is kept per-telescope (not shared): intrinsic width is achromatic in
principle, but the measured zeta also absorbs unmodelled per-band structure, so
the conservative choice is to let each band fit its own. The alpha lever arm
comes from the shared tau, independent of how zeta is treated.

Usage
-----
```python
from burstfit_joint import fit_joint_scattering
res = fit_joint_scattering(
    model_C=model_chime, init_C=init_chime,
    model_D=model_dsa,   init_D=init_dsa,
    alpha_bounds=(2.0, 6.0), nlive=600, nproc=8,
)
print(res["percentiles"]["alpha"], res["percentiles"]["tau_1ghz"])
```
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .burstfit import FRBModel, FRBParams, build_priors

log = logging.getLogger(__name__)

__all__ = [
    "fit_joint_scattering",
    "JOINT_PARAM_NAMES",
    "JOINT_PARAM_NAMES_GAIN",
    "JOINT_PARAM_NAMES_GAIN_GP",
    "JOINT_PARAM_NAMES_GAIN_MULTI",
    "_gain_marginal_multi_band",
]

# Joint 12-vector layout. First two are the shared sightline params; the rest
# are per-telescope (suffix _C = CHIME, _D = DSA).
JOINT_PARAM_NAMES: tuple[str, ...] = (
    "tau_1ghz",
    "alpha",
    "c0_C",
    "t0_C",
    "gamma_C",
    "zeta_C",
    "delta_dm_C",
    "c0_D",
    "t0_D",
    "gamma_D",
    "zeta_D",
    "delta_dm_D",
)
# Positive params sampled log-uniform (Jeffreys), mirroring burstfit_nested.
_LOG_NAMES = frozenset({"tau_1ghz", "c0_C", "zeta_C", "c0_D", "zeta_D"})

# Gain-marginalized 8-vector layout: the per-channel amplitude (gain) is
# integrated analytically (matched-filter likelihood), so c0 and gamma drop out
# of the sampled vector -- the gain absorbs the burst spectrum AND scintillation.
# Only the temporal/scattering params remain. Lower-dim => easier sampling, and
# the 2D residual whitens so chi2 becomes a valid goodness-of-fit gate.
JOINT_PARAM_NAMES_GAIN: tuple[str, ...] = (
    "tau_1ghz",
    "alpha",
    "t0_C",
    "zeta_C",
    "delta_dm_C",
    "t0_D",
    "zeta_D",
    "delta_dm_D",
)
_LOG_NAMES_GAIN = frozenset({"tau_1ghz", "zeta_C", "zeta_D"})

# Gain-marginalized + scintillation-GP 10-vector layout. Adds a per-band
# scintillation bandwidth Delta_nu_d (MHz). The flat per-channel gain prior is
# replaced by a Lorentzian-ACF Gaussian process (see
# FRBModel.log_likelihood_gain_marginal_gp); the smooth spectral envelope (mu)
# and the GP amplitude (sigma_g) are profiled analytically, so ONLY Delta_nu_d
# is added to the sampled vector (8 -> 10 dim).
JOINT_PARAM_NAMES_GAIN_GP: tuple[str, ...] = (
    "tau_1ghz",
    "alpha",
    "t0_C",
    "zeta_C",
    "delta_dm_C",
    "Delta_nu_d_C",
    "t0_D",
    "zeta_D",
    "delta_dm_D",
    "Delta_nu_d_D",
)
_LOG_NAMES_GAIN_GP = frozenset({"tau_1ghz", "zeta_C", "zeta_D", "Delta_nu_d_C", "Delta_nu_d_D"})


# ----------------------------------------------------------------------
# Multi-component (N>=1 pulses per band) gain-marginal layout.
#
# Each band carries n_comp temporal components; per component a (t0, zeta) pair
# is sampled (suffix _C{i}/_D{i}, i=1..n_comp). (tau_1ghz, alpha) stay shared
# across BOTH bands and ALL components -- one sightline, one scattering law.
# delta_dm stays per-band (one cold-plasma column per telescope).
#
#   [tau_1ghz, alpha,
#    t0_C1, zeta_C1, ..., t0_C{nC}, zeta_C{nC}, delta_dm_C,
#    t0_D1, zeta_D1, ..., t0_D{nD}, zeta_D{nD}, delta_dm_D]
#
# n_comp=1 both bands -> 8-vector ordering IDENTICAL in content to
# JOINT_PARAM_NAMES_GAIN (names differ by the _C1/_D1 suffix only); the existing
# 8-vector path is untouched.
# ----------------------------------------------------------------------
def JOINT_PARAM_NAMES_GAIN_MULTI(n_C: int = 1, n_D: int = 1) -> tuple[str, ...]:
    """Param-name tuple for the N-component-per-band gain-marginal fit."""
    names: list[str] = ["tau_1ghz", "alpha"]
    for i in range(1, int(n_C) + 1):
        names += [f"t0_C{i}", f"zeta_C{i}"]
    names.append("delta_dm_C")
    for i in range(1, int(n_D) + 1):
        names += [f"t0_D{i}", f"zeta_D{i}"]
    names.append("delta_dm_D")
    return tuple(names)


def _gain_marginal_multi_band(
    model: FRBModel,
    params_list: Sequence[FRBParams],
    model_keys: Sequence[str],
    s2: float | None = None,
    eig_rel_floor: float = 1e-6,
) -> tuple[float, dict[str, Any]]:
    """Per-channel linear-Gaussian gain-marginal evidence for ONE band.

    N temporal component kernels K_1..K_N per channel f; the per-component gains
    g ~ N(0, s2 I_N) carry the burst spectrum + scintillation. With noise var
    sigma_f^2 the per-channel marginal (Gaussian g integrated analytically) is

        M_ij = sum_t K_i,t K_j,t          (NxN, per channel)
        b_i  = sum_t d_t K_i,t            (N)
        S_dd = sum_t d_t^2

        ln Z_f = -0.5*[ S_dd/sigma^2 - b^T (M + (sigma^2/s2) I)^-1 b / sigma^2 ]
                 - 0.5*T*ln(2 pi sigma^2)                       (FULL data norm)
                 - 0.5*ln det( I_N + (s2/sigma^2) M )           (proper Occam)

    (The quadratic divisor is sigma^2, not sigma^4 -- verified against the brute
    Gaussian evidence d^T Sigma_d^-1 d, Sigma_d = sigma^2 I_T + s2 K K^T, via
    Woodbury; the SPEC's sigma^4 was a transcription slip.)

    The Occam term GROWS with N and with s2 -- the valid finite-variance penalty
    that the flat-improper version (-0.5 ln det M ~ +N ln s2 as s2->inf) got
    wrong, rewarding spurious merged components. As s2->inf, ln Z_f reduces to the
    flat F-stat profile -0.5*chi2min_f - 0.5*ln det M_f + 0.5*N*ln s2 (the last
    is a divergent param-INDEPENDENT constant that cancels in any ln Z DIFFERENCE).

    s2 is the gain-prior variance HYPERPARAMETER. If ``s2 is None`` it is profiled
    by 1-D ML on a SHARED-per-band value (state in the design notes); pass a float
    to fix it. Returns ``(lnZ, diag)`` with diagnostics:
    ``frac_culled`` (channels dropped by the eigenvalue guard), ``max_abs_g``
    (per component), ``s2``, ``n_supported``.

    Eigenvalue conditioning guard: a channel is culled when
    min eig(M_f) / max eig(M_f) < ``eig_rel_floor`` (near-degenerate kernels in
    the DAMAGE band where two components nearly merge -- M_f singular -> the full-N
    solve explodes |g|). A culled-but-SUPPORTED channel (real signal, collinear
    kernels) falls back to a rank-1 proper-prior evidence on its top eigenpair --
    NOT to the gain=0 baseline -- so a merge stays Occam-penalized (a reward at
    large fixed s2 otherwise; see the inline note). Only a genuinely unsupported
    channel (emax ~ 0, no signal) gets the gain=0 baseline
    -0.5 S_dd/sigma^2 - 0.5 T ln(2 pi sigma^2). ``frac_culled`` counts all
    not-full-rank-N channels (rank-1 fallback + unsupported).
    """
    if model.data is None or model.noise_std is None:
        raise RuntimeError("need data + noise_std")
    valid = model.valid
    if valid is None or not np.any(valid):
        return -np.inf, {"frac_culled": 1.0, "max_abs_g": None, "s2": s2, "n_supported": 0}

    Ks = np.stack(
        [
            model(replace(p, c0=1.0, gamma=0.0), mk, freq_subset=valid)
            for p, mk in zip(params_list, model_keys)
        ]
    )  # (N, F, T)
    N, F, T = Ks.shape
    d = model.data[valid]  # (F, T)
    sig = np.clip(model.noise_std[valid], 1e-9, None)  # (F,)
    var = sig**2  # (F,)

    S_dd = np.einsum("ft,ft->f", d, d)  # (F,)
    b = np.einsum("nft,ft->fn", Ks, d)  # (F, N)
    M = np.einsum("nft,mft->fnm", Ks, Ks)  # (F, N, N)

    # Eigenvalue conditioning guard (per channel). M is symmetric PSD. We keep the
    # eigenVECTORS (eigh, not eigvalsh) so a culled channel can fall back to its
    # rank-1 top-eigenpair proper evidence instead of the gain=0 baseline -- see
    # below for why that distinction is load-bearing.
    evals, evecs = np.linalg.eigh(M)  # (F, N) asc, (F, N, N)
    emax = evals[:, -1]
    emin = evals[:, 0]
    supported = emax > 1e-30  # any signal at all
    cond_ok = np.zeros(F, dtype=bool)
    cond_ok[supported] = (
        emin[supported] / np.where(emax[supported] > 0, emax[supported], 1.0) >= eig_rel_floor
    )
    ok = supported & cond_ok  # well-conditioned channels
    # Culled-but-supported channels: kernels collinear (a near-merge), but there
    # IS signal. Route them to a rank-1 model on the top eigenvector (one
    # effective kernel), NOT to gain=0. This matters because at large fixed s2 the
    # gain=0 baseline -0.5 S_dd/var - 0.5 T ln(2pi var) sits ABOVE the proper N=1
    # lnZ (which carries the divergent +0.5 ln(s2/var) Occam per channel), so
    # culling-to-baseline would REWARD a degenerate merge by ~+0.5 F ln(s2/var)
    # (e.g. +676 nats at s2=1e8) -- reintroducing the very bug the prior fixes for
    # any caller that bypasses the ordered transform. The rank-1 fallback is
    # continuous with the N=1 proper model, so the merge is a penalty, not a reward.
    cull = supported & ~cond_ok  # (F,)
    eye = np.eye(N)

    def _lnZ_at(s2v: float) -> tuple[float, NDArray[np.floating]]:
        # gain=0 baseline only for genuinely unsupported (no-signal) channels.
        base = -0.5 * S_dd / var - 0.5 * T * np.log(2.0 * np.pi * var)
        lnZ_f = base.copy()
        g_all = np.zeros((F, N))
        if np.any(ok):
            Mok = M[ok]  # (G, N, N)
            bok = b[ok]  # (G, N)
            varok = var[ok]  # (G,)
            ridge = (varok / s2v)[:, None, None] * eye[None]
            A = Mok + ridge  # (G, N, N)
            g = np.linalg.solve(A, bok[:, :, None])[:, :, 0]  # (G, N), the MAP gain
            quad = np.einsum("gn,gn->g", g, bok)  # b^T A^-1 b
            # ln det( I + (s2/var) M ) via eigvals of M (per channel).
            ev_ok = evals[ok]  # (G, N)
            logdet_occam = np.sum(
                np.log1p((s2v / varok)[:, None] * np.clip(ev_ok, 0.0, None)),
                axis=1,
            )  # (G,)
            lnZ_ok = (
                -0.5 * (S_dd[ok] / varok - quad / varok)
                - 0.5 * T * np.log(2.0 * np.pi * varok)
                - 0.5 * logdet_occam
            )
            lnZ_f[ok] = lnZ_ok
            g_all[ok] = g
        if np.any(cull):
            # rank-1 proper evidence on the top eigenpair (scalar effective kernel
            # with norm^2 = emax, projected data b.v_top). MAP gain along v_top is
            # then distributed back onto the components via v_top for diagnostics.
            emx = emax[cull]  # (C,)
            vtop = evecs[cull][:, :, -1]  # (C, N)
            bproj = np.einsum("cn,cn->c", b[cull], vtop)  # (C,)
            varc = var[cull]  # (C,)
            Ac = emx + varc / s2v
            gc = bproj / Ac  # (C,) scalar MAP along v_top
            quadc = gc * bproj
            occ_c = np.log1p((s2v / varc) * np.clip(emx, 0.0, None))
            lnZ_f[cull] = (
                -0.5 * (S_dd[cull] / varc - quadc / varc)
                - 0.5 * T * np.log(2.0 * np.pi * varc)
                - 0.5 * occ_c
            )
            g_all[cull] = gc[:, None] * vtop  # (C, N)
        return float(np.sum(lnZ_f)), g_all

    if s2 is None:
        # 1-D ML over log s2, range anchored on the data scale: var(ahat) where
        # ahat=b/diag(M) is the matched-filter gain ~ sets the signal amplitude.
        diagM = np.einsum("fnn->fn", M)
        with np.errstate(divide="ignore", invalid="ignore"):
            ahat = np.where(diagM > 0, b / np.where(diagM > 0, diagM, 1.0), 0.0)
        scale = max(float(np.var(ahat[ok])) if np.any(ok) else 1.0, 1e-12)
        from scipy.optimize import minimize_scalar

        lo, hi = np.log(scale) - 18.0, np.log(scale) + 18.0
        res = minimize_scalar(
            lambda ls: -_lnZ_at(float(np.exp(ls)))[0],
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-3},
        )
        s2_used = float(np.exp(res.x))
    else:
        s2_used = float(s2)

    lnZ, g_all = _lnZ_at(s2_used)
    max_abs_g = [float(np.max(np.abs(g_all[:, i]))) if F else 0.0 for i in range(N)]
    # NB: n_supported and frac_culled use DIFFERENT denominators. n_supported counts
    # only well-conditioned (full-rank-N) channels (`ok`), whereas frac_culled =
    # mean(~ok) also counts rank-1-fallback channels as culled -- so in general
    # n_supported != (1 - frac_culled) * F.
    diag = {
        "frac_culled": float(np.mean(~ok)),
        "max_abs_g": max_abs_g,
        "s2": s2_used,
        "n_supported": int(np.count_nonzero(ok)),
    }
    return (lnZ if np.isfinite(lnZ) else -np.inf), diag


def _dnu_d_bounds(freq_GHz: NDArray[np.floating]) -> tuple[float, float]:
    """Log-uniform Delta_nu_d prior bounds (MHz) from a band's freq axis.

    Resolvable range: lower = 0.3 * channel width (below this the GP is
    unresolved and degrades to a flat upper limit); upper = band / 3 (a few
    scintles minimum). Derived per band from the data, not hardcoded.
    """
    nu = np.asarray(freq_GHz, dtype=float)
    chan_w_MHz = float(np.median(np.abs(np.diff(nu)))) * 1.0e3
    band_MHz = float(nu.max() - nu.min()) * 1.0e3
    return (0.3 * chan_w_MHz, band_MHz / 3.0)


def _joint_prior_spec(
    init_C: FRBParams,
    init_D: FRBParams,
    alpha_bounds: tuple[float, float],
) -> list[tuple[str, tuple[float, float], bool]]:
    """Assemble per-index (name, (lo, hi), is_log) from the single-band priors.

    Reuses build_priors(absolute_bounds=True) per telescope so the joint prior is
    init-independent (required for a global sampler) except t0, whose window is
    anchored on each band's data profile-peak estimate. alpha is widened to
    alpha_bounds to allow shallower-than-Kolmogorov slopes (the whole point).
    """
    pC, _ = build_priors(init_C, absolute_bounds=True)
    pD, _ = build_priors(init_D, absolute_bounds=True)
    # tau_1ghz bound is the absolute WIDTH_MIN..WIDTH_MAX (identical in pC/pD).
    by_name = {
        "tau_1ghz": pC["tau_1ghz"],
        "alpha": tuple(alpha_bounds),
        "c0_C": pC["c0"],
        "t0_C": pC["t0"],
        "gamma_C": pC["gamma"],
        "zeta_C": pC["zeta"],
        "delta_dm_C": pC["delta_dm"],
        "c0_D": pD["c0"],
        "t0_D": pD["t0"],
        "gamma_D": pD["gamma"],
        "zeta_D": pD["zeta"],
        "delta_dm_D": pD["delta_dm"],
    }
    return [(n, by_name[n], n in _LOG_NAMES) for n in JOINT_PARAM_NAMES]


def _joint_prior_spec_gain(
    init_C: FRBParams,
    init_D: FRBParams,
    alpha_bounds: tuple[float, float],
) -> list[tuple[str, tuple[float, float], bool]]:
    """Prior spec for the 8-vector gain-marginalized fit (no c0, gamma)."""
    pC, _ = build_priors(init_C, absolute_bounds=True)
    pD, _ = build_priors(init_D, absolute_bounds=True)
    by_name = {
        "tau_1ghz": pC["tau_1ghz"],
        "alpha": tuple(alpha_bounds),
        "t0_C": pC["t0"],
        "zeta_C": pC["zeta"],
        "delta_dm_C": pC["delta_dm"],
        "t0_D": pD["t0"],
        "zeta_D": pD["zeta"],
        "delta_dm_D": pD["delta_dm"],
    }
    return [(n, by_name[n], n in _LOG_NAMES_GAIN) for n in JOINT_PARAM_NAMES_GAIN]


def _joint_prior_spec_gain_gp(
    init_C: FRBParams,
    init_D: FRBParams,
    alpha_bounds: tuple[float, float],
    model_C: FRBModel,
    model_D: FRBModel,
) -> list[tuple[str, tuple[float, float], bool]]:
    """Prior spec for the 10-vector gain+scintillation-GP fit.

    Reuses the 8 temporal entries from `_joint_prior_spec_gain`, then appends a
    per-band log-uniform Delta_nu_d with bounds [0.3*chan_width, band/3] computed
    from each model's freq axis (data-derived, not hardcoded).
    """
    base = {n: (b, lg) for n, b, lg in _joint_prior_spec_gain(init_C, init_D, alpha_bounds)}
    dnu_C = _dnu_d_bounds(model_C.freq)
    dnu_D = _dnu_d_bounds(model_D.freq)
    by_name = dict(base)
    by_name["Delta_nu_d_C"] = (dnu_C, True)
    by_name["Delta_nu_d_D"] = (dnu_D, True)
    return [(n, by_name[n][0], n in _LOG_NAMES_GAIN_GP) for n in JOINT_PARAM_NAMES_GAIN_GP]


def _joint_prior_spec_gain_multi(
    init_C: FRBParams,
    init_D: FRBParams,
    alpha_bounds: tuple[float, float],
    n_C: int = 1,
    n_D: int = 1,
) -> list[tuple[str, tuple[float, float], bool]]:
    """Prior spec for the N-component-per-band gain-marginal fit.

    Each component reuses the single-band (t0, zeta) bounds; the t0 window is the
    SAME for every component in a band (the ordered transform separates them).
    """
    pC, _ = build_priors(init_C, absolute_bounds=True)
    pD, _ = build_priors(init_D, absolute_bounds=True)
    spec: list[tuple[str, tuple[float, float], bool]] = [
        ("tau_1ghz", pC["tau_1ghz"], True),
        ("alpha", tuple(alpha_bounds), False),
    ]
    for i in range(1, int(n_C) + 1):
        spec.append((f"t0_C{i}", pC["t0"], False))
        spec.append((f"zeta_C{i}", pC["zeta"], True))
    spec.append(("delta_dm_C", pC["delta_dm"], False))
    for i in range(1, int(n_D) + 1):
        spec.append((f"t0_D{i}", pD["t0"], False))
        spec.append((f"zeta_D{i}", pD["zeta"], True))
    spec.append(("delta_dm_D", pD["delta_dm"], False))
    return spec


class _JointPriorTransform:
    """Picklable unit-cube -> parameter transform for the 12-vector.

    Module-level callable (not a closure) so dynesty.pool can ship it to workers.
    Log-uniform on the flagged positive params, uniform on the rest.
    """

    def __init__(self, spec: list[tuple[str, tuple[float, float], bool]]):
        self.lo = np.array([s[1][0] for s in spec], dtype=float)
        self.hi = np.array([s[1][1] for s in spec], dtype=float)
        # only log-sample where flagged AND both bounds strictly positive
        self.is_log = np.array([bool(s[2] and s[1][0] > 0 and s[1][1] > 0) for s in spec])
        # precompute log-bounds with safe placeholders (log(1)=0) on linear axes
        self._loglo = np.log(np.where(self.is_log, self.lo, 1.0))
        self._loghi = np.log(np.where(self.is_log, self.hi, 1.0))

    def __call__(self, u: NDArray[np.floating]) -> NDArray[np.floating]:
        lin = self.lo + u * (self.hi - self.lo)
        logu = np.exp(self._loglo + u * (self._loghi - self._loglo))
        return np.where(self.is_log, logu, lin)


class _JointPriorTransformOrdered(_JointPriorTransform):
    """Ordered + min-separation transform for the multi-component vector.

    Within each band the per-component t0 group is SORTED ascending (breaks the
    label-swap degeneracy: N! identical posterior modes collapse to one), then
    forced to obey t0_{i+1} - t0_i >= dt_min. The min-separation is enforced by
    re-mapping the unit cube of the t0 group onto the simplex
    {t0_1 <= ... <= t0_N, gaps >= dt_min} so EVERY cube point lands in the
    feasible region (no rejected volume, no -inf from the transform -- dynesty's
    cube map must stay total). A cube whose feasible width
    (hi - lo - (N-1)*dt_min) is <= 0 collapses the group to a single point and
    the likelihood (degenerate kernels) is culled by the eigenvalue guard, so the
    merge is penalized by the Occam term, not rewarded -- which is the whole fix.

    dt_min defaults to a few channel time-samples (>= the kernel can resolve);
    the caller passes the band time grids so it is data-derived, not hardcoded.
    """

    def __init__(self, spec, t0_groups, dt_min):
        super().__init__(spec)
        # t0_groups: list of index arrays into the param vector, one per band,
        # giving the positions of that band's t0_C1..t0_C{n} (already ascending).
        self.t0_groups = [np.asarray(g, dtype=int) for g in t0_groups]
        self.dt_min = float(dt_min)

    def __call__(self, u: NDArray[np.floating]) -> NDArray[np.floating]:
        x = super().__call__(u)
        for grp in self.t0_groups:
            n = grp.size
            if n < 2:
                continue
            lo = self.lo[grp[0]]
            hi = self.hi[grp[0]]
            # Feasible width after reserving (n-1)*dt_min of separation.
            usable = hi - lo - (n - 1) * self.dt_min
            uu = np.sort(u[grp])  # n sorted unit-cube coords -> ordered
            if usable > 0:
                # place n ordered points in [0, usable], then add cumulative dt_min
                pts = lo + uu * usable + np.arange(n) * self.dt_min
            else:
                # band too narrow for n separated comps -> collapse (culled by guard)
                pts = np.full(n, lo + uu.mean() * (hi - lo))
            x[grp] = pts
        return x


class _JointLogLikelihood:
    """Picklable joint log-likelihood: ll_CHIME(pC) + ll_DSA(pD).

    Two FRBModels sharing (tau_1ghz, alpha); independent noise -> additive.
    Both FRBModels hold only numpy arrays + scalars, so this pickles.
    """

    def __init__(self, model_C: FRBModel, model_D: FRBModel):
        self.model_C = model_C
        self.model_D = model_D

    def __call__(self, theta: NDArray[np.floating]) -> float:
        tau, alpha = float(theta[0]), float(theta[1])
        pC = FRBParams(
            c0=theta[2],
            t0=theta[3],
            gamma=theta[4],
            zeta=theta[5],
            tau_1ghz=tau,
            alpha=alpha,
            delta_dm=theta[6],
        )
        pD = FRBParams(
            c0=theta[7],
            t0=theta[8],
            gamma=theta[9],
            zeta=theta[10],
            tau_1ghz=tau,
            alpha=alpha,
            delta_dm=theta[11],
        )
        ll = self.model_C.log_likelihood(pC, "M3") + self.model_D.log_likelihood(pD, "M3")
        return ll if np.isfinite(ll) else -1e100


class _JointLogLikelihoodGain:
    """Joint gain-marginalized log-L: matched-filter L over both bands.

    8-vector theta = [tau, alpha, t0_C, zeta_C, ddm_C, t0_D, zeta_D, ddm_D]. Per
    band the per-channel amplitude is integrated out analytically
    (FRBModel.log_likelihood_gain_marginal), so c0/gamma are not sampled.
    """

    def __init__(self, model_C: FRBModel, model_D: FRBModel):
        self.model_C = model_C
        self.model_D = model_D

    def __call__(self, theta: NDArray[np.floating]) -> float:
        tau, alpha = float(theta[0]), float(theta[1])
        pC = FRBParams(
            c0=1.0,
            t0=theta[2],
            gamma=0.0,
            zeta=theta[3],
            tau_1ghz=tau,
            alpha=alpha,
            delta_dm=theta[4],
        )
        pD = FRBParams(
            c0=1.0,
            t0=theta[5],
            gamma=0.0,
            zeta=theta[6],
            tau_1ghz=tau,
            alpha=alpha,
            delta_dm=theta[7],
        )
        ll = self.model_C.log_likelihood_gain_marginal(
            pC, "M3"
        ) + self.model_D.log_likelihood_gain_marginal(pD, "M3")
        return ll if np.isfinite(ll) else -1e100


class _JointLogLikelihoodGainGP:
    """Joint gain-marginal log-L with a scintillation GP prior on the gains.

    10-vector theta layout (JOINT_PARAM_NAMES_GAIN_GP):
      [0] tau_1ghz  [1] alpha
      [2] t0_C  [3] zeta_C  [4] delta_dm_C  [5] Delta_nu_d_C
      [6] t0_D  [7] zeta_D  [8] delta_dm_D  [9] Delta_nu_d_D

    Per band the per-channel gains are integrated analytically under a Lorentzian
    Gaussian-process prior (FRBModel.log_likelihood_gain_marginal_gp), profiling
    the smooth envelope (GLS) and GP amplitude (ML); c0/gamma are not sampled.
    Independent noise -> the joint logL is additive, exactly as the flat path.
    """

    def __init__(self, model_C: FRBModel, model_D: FRBModel, mu_degree: int = 1):
        self.model_C = model_C
        self.model_D = model_D
        self.mu_degree = int(mu_degree)

    def __call__(self, theta: NDArray[np.floating]) -> float:
        tau, alpha = float(theta[0]), float(theta[1])
        pC = FRBParams(
            c0=1.0,
            t0=theta[2],
            gamma=0.0,
            zeta=theta[3],
            tau_1ghz=tau,
            alpha=alpha,
            delta_dm=theta[4],
        )
        pD = FRBParams(
            c0=1.0,
            t0=theta[6],
            gamma=0.0,
            zeta=theta[7],
            tau_1ghz=tau,
            alpha=alpha,
            delta_dm=theta[8],
        )
        ll = self.model_C.log_likelihood_gain_marginal_gp(
            pC, "M3", delta_nu_d_MHz=float(theta[5]), mu_degree=self.mu_degree
        ) + self.model_D.log_likelihood_gain_marginal_gp(
            pD, "M3", delta_nu_d_MHz=float(theta[9]), mu_degree=self.mu_degree
        )
        return ll if np.isfinite(ll) else -1e100


class _JointLogLikelihoodGainMulti:
    """Joint N-component-per-band gain-marginal log-L (proper finite gain prior).

    Picklable: holds 2 FRBModels, the per-band component counts (n_C, n_D), and
    the s2 hyperparameter policy. theta layout = JOINT_PARAM_NAMES_GAIN_MULTI:

      [tau, alpha,
       t0_C1, zeta_C1, ..., t0_C{nC}, zeta_C{nC}, delta_dm_C,
       t0_D1, zeta_D1, ..., t0_D{nD}, zeta_D{nD}, delta_dm_D]

    Per band the per-channel per-component gains are integrated under a proper
    N(0, s2 I) prior (_gain_marginal_multi_band); independent noise -> additive.
    s2_policy: None -> ML-profile a shared s2 per band per call; a float -> fixed.
    """

    def __init__(
        self,
        model_C: FRBModel,
        model_D: FRBModel,
        n_C: int = 1,
        n_D: int = 1,
        s2: float | None = None,
    ):
        self.model_C = model_C
        self.model_D = model_D
        self.n_C = int(n_C)
        self.n_D = int(n_D)
        self.s2 = s2

    def _band_params(self, theta, off, n, tau, alpha, ddm):
        out = []
        for i in range(n):
            t0 = float(theta[off + 2 * i])
            zeta = float(theta[off + 2 * i + 1])
            out.append(
                FRBParams(
                    c0=1.0, t0=t0, gamma=0.0, zeta=zeta, tau_1ghz=tau, alpha=alpha, delta_dm=ddm
                )
            )
        return out

    def __call__(self, theta: NDArray[np.floating]) -> float:
        tau, alpha = float(theta[0]), float(theta[1])
        oC = 2
        ddm_C = float(theta[oC + 2 * self.n_C])
        psC = self._band_params(theta, oC, self.n_C, tau, alpha, ddm_C)
        oD = oC + 2 * self.n_C + 1
        ddm_D = float(theta[oD + 2 * self.n_D])
        psD = self._band_params(theta, oD, self.n_D, tau, alpha, ddm_D)

        lnZ_C, _ = _gain_marginal_multi_band(self.model_C, psC, ["M3"] * self.n_C, s2=self.s2)
        lnZ_D, _ = _gain_marginal_multi_band(self.model_D, psD, ["M3"] * self.n_D, s2=self.s2)
        ll = lnZ_C + lnZ_D
        return ll if np.isfinite(ll) else -1e100


def _weighted_percentiles(
    samples: NDArray[np.floating],
    weights: NDArray[np.floating],
    names: tuple[str, ...] = JOINT_PARAM_NAMES,
) -> dict[str, dict[str, float]]:
    """Weighted 16/50/84 percentiles per column (mirror NestedSamplingResult)."""
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(names):
        s = samples[:, i]
        idx = np.argsort(s)
        ss, sw = s[idx], weights[idx]
        cdf = np.cumsum(sw)
        cdf /= cdf[-1]
        p16, p50, p84 = ss[np.searchsorted(cdf, [0.16, 0.50, 0.84])]
        out[name] = {
            "median": float(p50),
            "lower": float(p16),
            "upper": float(p84),
            "err_minus": float(p50 - p16),
            "err_plus": float(p84 - p50),
        }
    return out


def fit_joint_scattering(
    *,
    model_C: FRBModel,
    init_C: FRBParams,
    model_D: FRBModel,
    init_D: FRBParams,
    alpha_bounds: tuple[float, float] = (2.0, 6.0),
    nlive: int = 600,
    dlogz: float = 0.5,
    nproc: int | None = None,
    sample: str = "rwalk",
    verbose: bool = True,
    marginalize_gain: bool = False,
    marginalize_gain_gp: bool = False,
    mu_degree: int = 1,
    components_C: int = 1,
    components_D: int = 1,
    gain_s2: float | None = None,
    dt_min: float | None = None,
    force_multi: bool = False,
    **dynesty_kwargs,
) -> dict[str, Any]:
    """Run the joint CHIME+DSA nested fit; return posterior summary.

    Parameters
    ----------
    model_C, model_D : FRBModel
        CHIME and DSA burst models, each with data + noise loaded.
    init_C, init_D : FRBParams
        Per-band data-driven inits (used only to anchor the t0 prior window and
        scale-free absolute bounds).
    alpha_bounds : (lo, hi)
        Uniform prior on the shared scattering index. Default (2, 6) is wide
        enough to detect shallow (sub-Kolmogorov) slopes.
    nlive, dlogz, nproc, sample
        dynesty knobs (12-dim -> nlive ~600+ recommended).

    Returns
    -------
    dict with keys: param_names, percentiles, log_evidence, log_evidence_err,
    samples, weights, alpha_bounds.
    """
    from dynesty import NestedSampler

    if model_C.data is None or model_D.data is None:
        raise ValueError("both FRBModels must have data loaded")

    multi = bool(force_multi) or int(components_C) > 1 or int(components_D) > 1
    ptform = None
    if multi:
        names = JOINT_PARAM_NAMES_GAIN_MULTI(components_C, components_D)
        spec = _joint_prior_spec_gain_multi(
            init_C, init_D, alpha_bounds, components_C, components_D
        )
        loglike = _JointLogLikelihoodGainMulti(
            model_C, model_D, n_C=components_C, n_D=components_D, s2=gain_s2
        )
        # dt_min: a few channel time-samples of each band (data-derived). The
        # binding constraint is the tighter (smaller-dt) band's resolution.
        if dt_min is None:
            dts = []
            for m in (model_C, model_D):
                t = np.asarray(m.time, dtype=float)
                dts.append(float(np.median(np.abs(np.diff(t)))) * 3.0)
            dt_min = max(dts)
        # index groups of each band's t0 components within the vector.
        idx = {n: i for i, n in enumerate(names)}
        grp_C = [idx[f"t0_C{i}"] for i in range(1, int(components_C) + 1)]
        grp_D = [idx[f"t0_D{i}"] for i in range(1, int(components_D) + 1)]
        ptform = _JointPriorTransformOrdered(spec, [grp_C, grp_D], dt_min=dt_min)
    elif marginalize_gain_gp:
        names = JOINT_PARAM_NAMES_GAIN_GP
        spec = _joint_prior_spec_gain_gp(init_C, init_D, alpha_bounds, model_C, model_D)
        loglike = _JointLogLikelihoodGainGP(model_C, model_D, mu_degree=mu_degree)
    elif marginalize_gain:
        names = JOINT_PARAM_NAMES_GAIN
        spec = _joint_prior_spec_gain(init_C, init_D, alpha_bounds)
        loglike = _JointLogLikelihoodGain(model_C, model_D)
    else:
        names = JOINT_PARAM_NAMES
        spec = _joint_prior_spec(init_C, init_D, alpha_bounds)
        loglike = _JointLogLikelihood(model_C, model_D)
    ndim = len(spec)
    if ptform is None:
        ptform = _JointPriorTransform(spec)
    if ndim >= 12 and nlive < 800:
        log.info(
            f"Joint multi-component fit ndim={ndim} >= 12: recommend "
            f"nlive>=800-1000 (got {nlive}) for reliable evidence."
        )

    if verbose:
        log.info(
            f"Joint CHIME+DSA fit: ndim={ndim}, nlive={nlive}, "
            f"alpha~U{alpha_bounds}, marginalize_gain={marginalize_gain}, "
            f"marginalize_gain_gp={marginalize_gain_gp}"
        )

    if nproc is not None and nproc > 1:
        # fork so workers inherit memory instead of re-importing __main__ (spawn
        # default crashes); identical pattern to burstfit_nested.
        import multiprocessing as _mp

        try:
            _mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
        from dynesty import pool as dypool

        with dypool.Pool(int(nproc), loglike, ptform) as pool:
            sampler = NestedSampler(
                pool.loglike,
                pool.prior_transform,
                ndim,
                nlive=nlive,
                sample=sample,
                pool=pool,
                queue_size=int(nproc),
                **dynesty_kwargs,
            )
            sampler.run_nested(dlogz=dlogz, print_progress=verbose)
            results = sampler.results
    else:
        sampler = NestedSampler(loglike, ptform, ndim, nlive=nlive, sample=sample, **dynesty_kwargs)
        sampler.run_nested(dlogz=dlogz, print_progress=verbose)
        results = sampler.results

    weights = np.exp(results.logwt - results.logz[-1])
    weights /= weights.sum()

    return {
        "param_names": list(names),
        "percentiles": _weighted_percentiles(results.samples, weights, names),
        "log_evidence": float(results.logz[-1]),
        "log_evidence_err": float(results.logzerr[-1]),
        "samples": results.samples,
        "weights": weights,
        "alpha_bounds": tuple(alpha_bounds),
        "ncall": int(np.sum(results.ncall)),  # dynesty .ncall is per-iteration, sum for total
    }


def demo() -> None:
    """Self-check: the shared-tau likelihood must prefer the true alpha.

    Builds two synthetic single-band bursts (CHIME 0.6 / DSA 1.4 GHz) scattered
    with a common tau_1ghz, alpha_true, then verifies the joint log-likelihood
    (profiled crudely over the per-band amplitudes only) peaks at alpha_true and
    rejects a wrong alpha. No sampler -- a fast logic gate, not a fit.
    """
    rng = np.random.default_rng(0)
    tau_true, alpha_true = 1.0, 4.0
    truth = dict(c0=20.0, gamma=0.0, zeta=0.3, tau_1ghz=tau_true, alpha=alpha_true)

    def make(fmin, fmax, nch):
        freq = np.linspace(fmin, fmax, nch)
        time = np.arange(220) * 0.05
        m = FRBModel(time=time, freq=freq, data=np.zeros((nch, time.size)), dm_init=0.0)
        p = FRBParams(t0=time.mean(), delta_dm=0.0, **truth)
        clean = m(p, "M3")
        noisy = clean + rng.normal(0, 0.05 * clean.max(), clean.shape)
        return FRBModel(time=time, freq=freq, data=noisy, dm_init=0.0), p

    mC, pC = make(0.40, 0.80, 16)
    mD, pD = make(1.31, 1.50, 16)
    ll = _JointLogLikelihood(mC, mD)

    def vec(alpha):
        # [tau, alpha | c0_C,t0_C,g_C,z_C,dd_C | c0_D,t0_D,g_D,z_D,dd_D]
        return np.array(
            [
                tau_true,
                alpha,
                pC.c0,
                pC.t0,
                pC.gamma,
                pC.zeta,
                0.0,
                pD.c0,
                pD.t0,
                pD.gamma,
                pD.zeta,
                0.0,
            ]
        )

    ll_true = ll(vec(alpha_true))
    ll_wrong = ll(vec(alpha_true + 1.5))
    assert ll_true > ll_wrong, f"true alpha not preferred: {ll_true} <= {ll_wrong}"
    # and the shared-tau model with a wrong tau is worse too
    bad = vec(alpha_true)
    bad[0] = tau_true * 5
    assert ll_true > ll(bad), "true tau not preferred"
    print(
        f"demo OK: ll(alpha={alpha_true})={ll_true:.0f} > "
        f"ll(alpha={alpha_true + 1.5})={ll_wrong:.0f}"
    )


if __name__ == "__main__":
    demo()
