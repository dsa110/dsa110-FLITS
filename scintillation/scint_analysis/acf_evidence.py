"""Nested-sampling evidence comparison for ACF component count (A1 trigger, limb i).

Implements the evidence-based escalation trigger of the Faber2026 A1 design
(docs/rse/specs/plan-a1-trigger-calibration.md, owner direction 2026-07-13):
a second broadening component may be fitted only when the Bayesian evidence
prefers a two-component ACF model over a single Lorentzian at an
injection-calibrated dlnZ threshold, or the burst-profile PPC limb fires.

Conventions (must match the production estimator):
  - gamma is the Lorentzian HWHM and equals the reported decorrelation
    bandwidth Delta nu_d directly (analysis.calculate_acf, HWHM interpolation;
    no FWHM conversion) — the scinttools_v3 FWHM=2*gamma convention is the
    documented hazard, do not reintroduce it.
  - Lag-0 is never fitted: the symmetric lag-0 point in the production ACF is
    a synthetic token (acf=1, err=1e-9), not a measurement. This module
    requires strictly positive one-sided lags, matching the one-sided
    convention of revalidation.compare_lorentzian_components (ndata must not
    be inflated by ACF symmetry).
  - M2 is the physical two-screen form lor1 + lor2 + lor1*lor2 + c of the
    CANFAR-era reference recipe (reference_arc RECIPE.md §3g): two screens
    modulate multiplicatively, so the cross term is part of the prediction.
    Width ordering is enforced by parameterizing gamma2 = f * gamma1 with a
    prior f >= f_lo > 1.

The dlnZ threshold consumed by ``escalation_trigger_verdict`` is not chosen
here: it comes from the injection calibration campaign
(simulation/trigger_calibration.py) and is recorded as an ADR-0008
calibration entry. Rail semantics follow the re-trust contract: posterior
mass piled on a prior edge is model-family rejection, never a detection.
"""

from __future__ import annotations

import numpy as np

# Rail constants mirror the ADR-0008 rail-classifier SSOT values
# (EDGE_WIDTH_FRAC / EDGE_MASS_FRAC in flits/fitting rails); duplicated here
# as module constants so the evidence engine has no import-time dependency on
# the scattering side.
EDGE_WIDTH_FRAC = 0.05
EDGE_MASS_FRAC = 0.30


def lorentzian_1(lags, gamma, m2, c):
    """Single-screen ACF model: c + m2 / (1 + (lag/gamma)^2). gamma = HWHM."""
    return c + m2 / (1.0 + (lags / gamma) ** 2)


def two_screen_model(lags, gamma1, m2_1, f, m2_2, c):
    """Two-screen ACF model: lor1 + lor2 + lor1*lor2 + c, gamma2 = f*gamma1.

    The multiplicative cross term is the physical prediction for two
    independent scintillating screens (reference recipe form).
    """
    l1 = m2_1 / (1.0 + (lags / gamma1) ** 2)
    l2 = m2_2 / (1.0 + (lags / (f * gamma1)) ** 2)
    return c + l1 + l2 + l1 * l2


def _mvn_loglike_factory(lags, acf, cov):
    """Multivariate-normal log-likelihood with a fixed covariance.

    The Cholesky factor is computed once; each call solves one triangular
    system. The covariance is supplied by the caller — diagonal for quick
    looks, the MC correlated-lag covariance (acf_covariance.mc_acf_covariance)
    for trigger use, so the second component cannot feed on finite-scintle
    sample variance.
    """
    from scipy.linalg import cho_factor, solve_triangular

    chol, lower = cho_factor(cov, lower=True)
    # Constant terms omitted: only evidence *differences* between models fitted
    # to the same (data, covariance) are consumed, and both models share them.

    def loglike_for(model):
        def loglike(theta):
            r = acf - model(lags, *theta)
            z = solve_triangular(chol, r, lower=lower)
            return -0.5 * float(z @ z)

        return loglike

    return loglike_for


def _weighted_median(x, w):
    idx = np.argsort(x)
    cw = np.cumsum(w[idx])
    return float(x[idx][np.searchsorted(cw, 0.5 * cw[-1])])


def _edge_mass_flags(samples, weights, bounds,
                     edge_frac=EDGE_WIDTH_FRAC, mass_frac=EDGE_MASS_FRAC):
    """Flag parameters whose posterior mass piles onto a prior edge.

    bounds: list of (name, lo, hi, is_log). A parameter is flagged when more
    than ``mass_frac`` of the posterior mass lies within ``edge_frac`` of
    either prior edge (in log space for log-uniform priors) — the ADR-0008
    edge-mass rail criterion. A flagged M2 fit is model-family rejection,
    never a two-screen detection.
    """
    flags = []
    w = weights / weights.sum()
    for i, (name, lo, hi, is_log) in enumerate(bounds):
        x = np.log(samples[:, i]) if is_log else samples[:, i]
        l, h = (np.log(lo), np.log(hi)) if is_log else (lo, hi)
        edge = edge_frac * (h - l)
        if w[(x < l + edge) | (x > h - edge)].sum() > mass_frac:
            flags.append(name)
    return flags


def _run_nested(loglike, prior_transform, ndim, nlive, dlogz, seed,
                maxcall=2_000_000):
    """One dynesty run; returns evidence, weighted samples, per-dim medians.

    Pattern follows scattering/scat_analysis/burstfit_nested.py (logz[-1],
    logzerr[-1], normalized exp(logwt - logz) weights). Sampling is 'rwalk':
    the M2 prior contains a near-plateau region (gamma1 and f both large make
    the model effectively constant over the fitted lag range), where the
    default uniform-ellipsoid proposal can stall for some noise realizations;
    random walks degrade gracefully there. ``maxcall`` bounds the run — a cap
    hit raises RuntimeError ('evidence_failed' path: flag, exclude, replay by
    seed) rather than returning a silently unconverged evidence.
    """
    from dynesty import NestedSampler

    rng = np.random.default_rng(seed)
    sampler = NestedSampler(loglike, prior_transform, ndim,
                            nlive=nlive, rstate=rng, sample="rwalk")
    sampler.run_nested(dlogz=dlogz, print_progress=False, maxcall=maxcall)
    r = sampler.results
    if int(np.sum(r.ncall)) >= maxcall:
        raise RuntimeError(
            f"nested run hit maxcall={maxcall} before dlogz={dlogz} "
            f"(seed={seed}, ndim={ndim}): evidence_failed"
        )
    w = np.exp(r.logwt - r.logz[-1])
    w = w / w.sum()
    med = [
        _weighted_median(r.samples[:, i], w) for i in range(ndim)
    ]
    return {
        "logz": float(r.logz[-1]),
        "logz_err": float(r.logzerr[-1]),
        "samples": np.asarray(r.samples),
        "weights": w,
        "median": med,
        "n_like_calls": int(np.sum(r.ncall)),
    }


def compare_acf_evidence(lags, acf, cov, channel_width_mhz, band_width_mhz,
                         nlive=500, dlogz=0.1, seed=0, f_lo=3.0, f_hi=300.0):
    """dlnZ for two-screen (M2) vs single-Lorentzian (M1) on a one-sided ACF.

    Parameters
    ----------
    lags, acf : 1-D arrays
        One-sided positive lags (MHz) and ACF values. Lag-0 must be excluded.
    cov : 2-D array
        Covariance of ``acf``. Use the MC correlated-lag covariance for
        trigger decisions; a diagonal is acceptable only for smoke tests.
    channel_width_mhz, band_width_mhz : float
        Set the gamma prior range: log-uniform on
        [0.5 * channel_width, band_width / 4].
    f_lo, f_hi : float
        Width-ratio prior range (log-uniform). f_lo = 3 encodes "genuinely
        distinct scale"; near-equal-scale ambiguity is limb (ii)'s job.

    Returns
    -------
    dict with keys "m1", "m2" (each: logz, logz_err, samples, weights,
    params_median, rail_flags), "dlnz" (= lnZ_M2 - lnZ_M1), "dlnz_err".
    """
    lags = np.asarray(lags, dtype=float)
    acf = np.asarray(acf, dtype=float)
    if lags.ndim != 1 or acf.shape != lags.shape:
        raise ValueError("lags and acf must be matching 1-D arrays")
    if not np.all(lags > 0):
        raise ValueError("one-sided positive lags required (lag-0 excluded)")

    g_lo = 0.5 * channel_width_mhz
    g_hi = band_width_mhz / 4.0
    if not g_hi > g_lo:
        raise ValueError("band_width_mhz/4 must exceed 0.5*channel_width_mhz")

    loglike_for = _mvn_loglike_factory(lags, acf, np.asarray(cov, dtype=float))

    def pt1(u):
        # gamma log-U [g_lo, g_hi]; m2 U [0.01, 2]; c U [-0.5, 0.5]
        return np.array([
            g_lo * (g_hi / g_lo) ** u[0],
            0.01 + 1.99 * u[1],
            -0.5 + u[2],
        ])

    def pt2(u):
        # gamma1, m2_1, f (log-U), m2_2, c
        return np.array([
            g_lo * (g_hi / g_lo) ** u[0],
            0.01 + 1.99 * u[1],
            f_lo * (f_hi / f_lo) ** u[2],
            0.01 + 1.99 * u[3],
            -0.5 + u[4],
        ])

    bounds1 = [("gamma", g_lo, g_hi, True),
               ("m2", 0.01, 2.0, False),
               ("c", -0.5, 0.5, False)]
    bounds2 = [("gamma1", g_lo, g_hi, True),
               ("m2_1", 0.01, 2.0, False),
               ("f", f_lo, f_hi, True),
               ("m2_2", 0.01, 2.0, False),
               ("c", -0.5, 0.5, False)]

    m1 = _run_nested(loglike_for(lorentzian_1), pt1, 3, nlive, dlogz, seed)
    m2 = _run_nested(loglike_for(two_screen_model), pt2, 5, nlive, dlogz,
                     seed + 1)

    m1["rail_flags"] = _edge_mass_flags(m1["samples"], m1["weights"], bounds1)
    m2["rail_flags"] = _edge_mass_flags(m2["samples"], m2["weights"], bounds2)

    m1["params_median"] = dict(zip(("gamma", "m2", "c"), m1["median"]))
    g1, m2_1, f, m2_2, c = m2["median"]
    m2["params_median"] = {"gamma1": g1, "m2_1": m2_1, "f": f,
                           "gamma2": g1 * f, "m2_2": m2_2, "c": c}

    return {
        "m1": m1,
        "m2": m2,
        "dlnz": m2["logz"] - m1["logz"],
        "dlnz_err": float(np.hypot(m1["logz_err"], m2["logz_err"])),
        "priors": {"gamma_mhz": (g_lo, g_hi), "f": (f_lo, f_hi),
                   "m2": (0.01, 2.0), "c": (-0.5, 0.5)},
    }


def escalation_trigger_verdict(dlnz, rail_flags, ppc_pvalues, calibration):
    """A1 escalation verdict: limb (i) dlnZ, limb (ii) burst-profile PPC.

    Semantics fixed by the A1 design (plan-a1-trigger-calibration.md):
      - a railed second component (posterior edge-pile) is model-family
        rejection — never a two-screen detection, regardless of dlnZ;
      - PPC band [0.05, 0.95] is the rung-iv contract; ``lag1_acf`` is the
        registered statistic sensitive to unmodeled temporal structure;
      - no escalation is 'censored', never a single-screen claim (a host-side
        screen below channel resolution is invisible to limb i).

    ``calibration`` must carry the injection-calibrated ``dlnz_threshold``
    (ADR-0008 calibration entry; reports/a1_trigger_calibration.json).
    """
    threshold = calibration["dlnz_threshold"]
    if rail_flags:
        return {"escalate": False, "verdict": "model_family_rejection",
                "reasons": [], "rail_flags": list(rail_flags)}
    reasons = []
    if dlnz >= threshold:
        reasons.append("dlnz")
    p = ppc_pvalues.get("lag1_acf")
    if p is not None and not (0.05 <= p <= 0.95):
        reasons.append("ppc_lag1_acf")
    if reasons:
        return {"escalate": True, "verdict": "escalate", "reasons": reasons}
    return {"escalate": False, "verdict": "no_escalation_censored",
            "reasons": []}


def evidence_with_mc_covariance(masked_spectrum, channel_width_mhz, snr,
                                n_real=500, seed=0, nlive=500, dlogz=0.1,
                                max_lag_bins=None, f_lo=3.0, f_hi=300.0):
    """End-to-end limb-i comparison: production ACF -> matched MC covariance
    -> dlnZ.

    Fits nothing to the spectrum beyond the production ACF's own HWHM
    estimate (interpolated half-max, the same convention calculate_acf uses
    for its finite-scintle error): the covariance is conditioned on the
    single-screen null with that gamma-hat — correct conditioning for a
    false-escalation decision.
    """
    from .acf_covariance import mc_acf_covariance
    from .analysis import calculate_acf

    acf_obj = calculate_acf(masked_spectrum, channel_width_mhz,
                            max_lag_bins=max_lag_bins)
    if acf_obj is None:
        raise ValueError("ACF calculation failed on the supplied spectrum")

    pos = acf_obj.lags > 0
    lags = acf_obj.lags[pos]
    acf = acf_obj.acf[pos]

    # gamma-hat: interpolated half-max of the one-sided ACF (production
    # convention); modulation-index estimate from the smallest-lag value.
    half_max = 0.5 * float(np.max(acf))
    gamma_hat = float(np.interp(half_max, acf[::-1], lags[::-1]))
    gamma_hat = max(gamma_hat, 0.5 * channel_width_mhz)
    m_hat = float(np.sqrt(np.clip(acf[0], 0.05, 1.0)))

    n_unmasked = int(np.ma.count(masked_spectrum))
    band_width_mhz = n_unmasked * channel_width_mhz

    cov = mc_acf_covariance(
        gamma_hwhm_mhz=gamma_hat, mod_index=m_hat,
        band_width_mhz=band_width_mhz, channel_width_mhz=channel_width_mhz,
        snr=snr, n_real=n_real, max_lag_bins=len(lags) + 1, seed=seed,
    )
    n_lag = cov.shape[0]
    res = compare_acf_evidence(
        lags[:n_lag], acf[:n_lag], cov,
        channel_width_mhz=channel_width_mhz, band_width_mhz=band_width_mhz,
        nlive=nlive, dlogz=dlogz, seed=seed, f_lo=f_lo, f_hi=f_hi,
    )
    res["gamma_hat_mhz"] = gamma_hat
    res["mod_index_hat"] = m_hat
    res["n_lags"] = n_lag
    return res
