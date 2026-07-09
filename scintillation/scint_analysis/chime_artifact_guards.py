"""Artifact-control guards for CHIME upchannelized scintillation products.

CHIME/FRB upchannelized (gen-3) dynamic spectra carry instrumental structure
that an ACF-based scintillation-bandwidth fit can mistake for a real
decorrelation scale. The ``experiment-freya-chime-instrumental-origin``
experiment (docs/rse/specs/, arms A/B1/C) established, for FRB 20230325A
(freya), that the canonical CHIME Delta_nu_d = 35.19 +/- 4.42 kHz is an
instrumental noise-correlation scale, not scintillation:

  - **Off-pulse ACF null FAILS** (arm A): burst-free noise slices fit the same
    tens-of-kHz scale as the on-pulse window (median 37.7 kHz), so the
    correlation lives in the product chain, not the burst.
  - **Low-lag excision collapses the fit** (arm B1): dropping the first few
    channel lags makes the apparent Lorentzian disappear (35.19 -> 31.2 (N=2)
    -> 23.4 (N=3) -> degenerate), i.e. there is no resolved wing. A real
    Lorentzian keeps its wing (contrast: DSA arm C, gamma stable 713->554 kHz
    across N=2..8).
  - **Harmonic masking is a systematic, not a cure** (arm B1): masking the
    coarse-channel comb at k*0.390625 MHz moves the fit 35.19 -> 42.21 kHz;
    both are retracted-as-instrumental scales.

This module promotes those one-off experiment arms into standing, first-class
guards (experiment doc "Next Steps" item 1) that any CHIME scintillation run
can apply. Every function here is a **pure** function of already-computed
numbers (no I/O, no fitting) so it is cheap and unit-testable; the driver
(``run_dsa_lorentzian_fits.py``) is responsible for computing the on/off ACFs
and excised fits and handing the resulting widths in.

Design contract: for ``telescope != "chime"`` the provenance gate never
downgrades a result (DSA products are not subject to the upchannelization
comb), so wiring these guards into a shared driver leaves the DSA path's
``measurement`` status intact.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .analysis import harmonic_lag_mask

# CHIME coarse-channel spacing: 400 MHz / 1024 native channels.
CHIME_COARSE_SPACING_MHZ = 0.390625
DEFAULT_HARMONIC_HALFWIDTH_MHZ = 0.05

# Mitigations a CHIME scintillation *measurement* must explicitly record. Each
# maps to the config path whose ``enable`` flag we require to be truthy.
CHIME_REQUIRED_MITIGATIONS = ("grid_regularization", "bandpass_normalization", "harmonic_mask")

MEASUREMENT = "measurement"
DIAGNOSTIC_ONLY = "diagnostic_only"


def _enabled(block: Any) -> bool:
    """True iff a config sub-block is a dict with a truthy ``enable`` key."""
    return isinstance(block, dict) and bool(block.get("enable"))


def apply_harmonic_mask_to_fit(
    lags: np.ndarray,
    acf: np.ndarray,
    err: np.ndarray | None,
    harmonic_cfg: dict | None,
):
    """Apply the coarse-channel harmonic lag mask to a fit-window ACF slice.

    Thin, testable wrapper around :func:`analysis.harmonic_lag_mask` for the
    driver path, which fits ``compare_lorentzian_components`` on raw sliced
    arrays and would otherwise silently ignore ``analysis.fitting.harmonic_mask``
    (the confirmed ``--band chime`` trap: the CHIME YAML enables the mask, but
    the driver never read it).

    Parameters
    ----------
    lags, acf : np.ndarray
        Fit-window lags (MHz) and ACF values (already sliced to the fit range).
    err : np.ndarray or None
        Per-lag ACF error, sliced consistently, or None.
    harmonic_cfg : dict or None
        The ``analysis.fitting.harmonic_mask`` block. When falsy or
        ``enable`` is not set, the arrays pass through unchanged.

    Returns
    -------
    (lags, acf, err, record) : tuple
        Masked arrays (comb-harmonic lag bins removed) and a JSON-able
        ``record`` dict: ``enabled``, ``spacing_mhz``, ``halfwidth_mhz``,
        ``n_bins_removed``, ``n_bins_kept``.
    """
    lags = np.asarray(lags, dtype=float)
    acf = np.asarray(acf, dtype=float)
    err = None if err is None else np.asarray(err, dtype=float)

    if not _enabled(harmonic_cfg):
        return (
            lags,
            acf,
            err,
            {
                "enabled": False,
                "spacing_mhz": None,
                "halfwidth_mhz": None,
                "n_bins_removed": 0,
                "n_bins_kept": int(lags.size),
            },
        )

    spacing = float(harmonic_cfg.get("spacing_mhz", CHIME_COARSE_SPACING_MHZ))
    halfwidth = float(harmonic_cfg.get("halfwidth_mhz", DEFAULT_HARMONIC_HALFWIDTH_MHZ))
    keep = harmonic_lag_mask(lags, spacing, halfwidth)
    record = {
        "enabled": True,
        "spacing_mhz": spacing,
        "halfwidth_mhz": halfwidth,
        "n_bins_removed": int(np.sum(~keep)),
        "n_bins_kept": int(np.sum(keep)),
    }
    sliced_err = None if err is None else err[keep]
    return lags[keep], acf[keep], sliced_err, record


def chime_provenance_status(config: dict | None) -> dict:
    """Fail-closed provenance gate for a CHIME scintillation measurement.

    For ``telescope: chime`` a result is only a *measurement* if the artifact
    controls the freya experiment identified as necessary are all explicitly
    enabled in the config: gapped-grid regularization, per-channel bandpass
    flat-fielding, and the coarse-channel harmonic mask. Any missing/disabled
    mitigation demotes the result to ``diagnostic_only`` (ChatGPT rec #2 /
    experiment "Next Steps" #1: incomplete mitigation must not yield a number).

    Non-CHIME telescopes are never demoted by this gate.

    Returns a JSON-able dict with ``telescope``, per-mitigation ``records``,
    ``missing`` (list), ``is_chime``, and ``status`` (``measurement`` or
    ``diagnostic_only``).
    """
    cfg = config or {}
    telescope = str(cfg.get("telescope", "") or "").strip().lower()
    analysis_cfg = cfg.get("analysis", {}) or {}
    fitting_cfg = analysis_cfg.get("fitting", {}) or {}

    records = {
        "grid_regularization": _enabled(analysis_cfg.get("grid_regularization")),
        "bandpass_normalization": _enabled(analysis_cfg.get("bandpass_normalization")),
        "harmonic_mask": _enabled(fitting_cfg.get("harmonic_mask")),
    }

    is_chime = telescope == "chime"
    if not is_chime:
        return {
            "telescope": telescope,
            "is_chime": False,
            "records": records,
            "missing": [],
            "status": MEASUREMENT,
            "reason": "non-CHIME telescope not subject to upchannelization comb gate",
        }

    missing = [m for m in CHIME_REQUIRED_MITIGATIONS if not records[m]]
    status = MEASUREMENT if not missing else DIAGNOSTIC_ONLY
    return {
        "telescope": telescope,
        "is_chime": True,
        "records": records,
        "missing": missing,
        "status": status,
        "reason": (
            "all required CHIME mitigations enabled"
            if not missing
            else f"missing/disabled CHIME mitigations: {', '.join(missing)}"
        ),
    }


def off_pulse_null_verdict(
    on_dnu_mhz: float | None,
    off_dnu_mhz: list[float] | np.ndarray | None,
    *,
    bracket_ratio: float = 2.0,
    min_off_fits: int = 3,
) -> dict:
    """Off-pulse ACF null test (experiment arm A; ChatGPT rec #3).

    A real scintillation width should live only in the burst. If burst-free
    (off-pulse) noise slices, fit with the identical ACF machinery, return the
    same decorrelation scale as the on-pulse fit, the correlation is in the
    product chain, not the sky -> the null FAILS and the on-pulse value is
    instrumental.

    The test compares the on-pulse width to the *median* off-pulse width. The
    null **fails** (``null_pass=False``) when the off-pulse median lands within
    a factor ``bracket_ratio`` of the on-pulse width (the freya CHIME failure:
    on 35.19 kHz, off median 37.7 kHz -> ratio ~1.07). It **passes** when the
    off-pulse fits are absent/scattered or land at a very different scale (the
    DSA behaviour: off-pulse effectively white).

    Parameters
    ----------
    on_dnu_mhz : float or None
        On-pulse fitted decorrelation bandwidth (MHz). None/non-finite -> the
        test is inconclusive (``null_pass=None``).
    off_dnu_mhz : sequence of float or None
        Fitted off-pulse widths (MHz), one per successful noise slice.
    bracket_ratio : float
        max(on/off_median, off_median/on) below which the two scales are
        "the same" and the null fails. Default 2.0 (within a factor of 2).
    min_off_fits : int
        Minimum number of successful off-pulse fits required to judge the
        null; fewer -> the off-pulse is too white/scattered to fit, null passes.

    Returns
    -------
    dict
        ``null_pass`` (bool or None), ``on_dnu_mhz``, ``off_median_dnu_mhz``,
        ``off_n_fits``, ``ratio``, ``reason``.
    """
    on_ok = on_dnu_mhz is not None and np.isfinite(on_dnu_mhz) and on_dnu_mhz > 0
    off = np.asarray([v for v in (off_dnu_mhz or []) if v is not None], dtype=float)
    off = off[np.isfinite(off) & (off > 0)]

    if not on_ok:
        return {
            "null_pass": None,
            "on_dnu_mhz": on_dnu_mhz,
            "off_median_dnu_mhz": float(np.median(off)) if off.size else None,
            "off_n_fits": int(off.size),
            "ratio": None,
            "reason": "no valid on-pulse width; null inconclusive",
        }

    if off.size < min_off_fits:
        return {
            "null_pass": True,
            "on_dnu_mhz": float(on_dnu_mhz),
            "off_median_dnu_mhz": float(np.median(off)) if off.size else None,
            "off_n_fits": int(off.size),
            "ratio": None,
            "reason": (
                f"only {off.size} off-pulse fits (< {min_off_fits}); "
                "off-pulse too white to fit a consistent scale -> null passes"
            ),
        }

    off_median = float(np.median(off))
    ratio = float(max(on_dnu_mhz / off_median, off_median / on_dnu_mhz))
    null_pass = ratio > bracket_ratio
    return {
        "null_pass": bool(null_pass),
        "on_dnu_mhz": float(on_dnu_mhz),
        "off_median_dnu_mhz": off_median,
        "off_n_fits": int(off.size),
        "ratio": ratio,
        "reason": (
            f"off-pulse median {off_median * 1e3:.1f} kHz brackets on-pulse "
            f"{on_dnu_mhz * 1e3:.1f} kHz (ratio {ratio:.2f} <= {bracket_ratio}); "
            "scale is instrumental -> null FAILS"
            if not null_pass
            else f"off-pulse median {off_median * 1e3:.1f} kHz differs from on-pulse "
            f"{on_dnu_mhz * 1e3:.1f} kHz (ratio {ratio:.2f} > {bracket_ratio}) -> null passes"
        ),
    }


def low_lag_stability_verdict(
    dnu_full_mhz: float | None,
    dnu_by_excision: dict[int, float | None] | None,
    *,
    collapse_ratio: float = 0.5,
) -> dict:
    """Low-lag excision stability test (experiment arm B1; ChatGPT rec #4).

    A genuine Lorentzian scintillation wing is not carried by the first few
    channel lags: dropping them should leave the fitted width roughly intact.
    If excising the lowest ``k`` positive-lag bins makes the width collapse (or
    the fit fail), the "Lorentzian" had no wing and is a low-lag artifact --
    exactly the freya CHIME failure (35.19 -> 23.4 kHz by N=3, degenerate by
    N=6). Contrast DSA (arm C): gamma stable within ~25% out to N=8.

    Parameters
    ----------
    dnu_full_mhz : float or None
        Width from the full fit (no excision).
    dnu_by_excision : dict[int, float or None]
        Map of excised-bin count ``k`` -> refitted width (MHz), or None when
        that refit failed/degenerated.
    collapse_ratio : float
        If any excised width drops below ``collapse_ratio * dnu_full`` (or the
        refit failed), the wing is judged unstable. Default 0.5 (halving).

    Returns
    -------
    dict
        ``stable`` (bool or None), ``dnu_full_mhz``, ``dnu_by_excision``,
        ``min_ratio``, ``failed_ks`` (list of k that failed/collapsed),
        ``reason``.
    """
    if dnu_full_mhz is None or not np.isfinite(dnu_full_mhz) or dnu_full_mhz <= 0:
        return {
            "stable": None,
            "dnu_full_mhz": dnu_full_mhz,
            "dnu_by_excision": dnu_by_excision or {},
            "min_ratio": None,
            "failed_ks": [],
            "reason": "no valid full-fit width; stability inconclusive",
        }

    by_k = dnu_by_excision or {}
    if not by_k:
        return {
            "stable": None,
            "dnu_full_mhz": float(dnu_full_mhz),
            "dnu_by_excision": {},
            "min_ratio": None,
            "failed_ks": [],
            "reason": "no excision refits provided; stability inconclusive",
        }

    ratios: dict[int, float] = {}
    failed_ks: list[int] = []
    for k, dnu in sorted(by_k.items()):
        if dnu is None or not np.isfinite(dnu) or dnu <= 0:
            failed_ks.append(int(k))
            continue
        ratios[int(k)] = float(dnu) / float(dnu_full_mhz)

    collapsed = [k for k, r in ratios.items() if r < collapse_ratio]
    failed_ks = sorted(set(failed_ks) | set(collapsed))
    min_ratio = min(ratios.values()) if ratios else None
    stable = len(failed_ks) == 0
    return {
        "stable": bool(stable),
        "dnu_full_mhz": float(dnu_full_mhz),
        "dnu_by_excision": {str(k): (float(v) if v is not None else None) for k, v in by_k.items()},
        "min_ratio": min_ratio,
        "failed_ks": failed_ks,
        "reason": (
            "width survives low-lag excision (no collapse) -> wing is resolved"
            if stable
            else f"width collapses/fails at excision k={failed_ks} "
            "-> no resolved wing, low-lag artifact"
        ),
    }


def harmonic_mask_systematic(
    dnu_unmasked_mhz: float | None,
    dnu_masked_mhz: float | None,
) -> dict:
    """Harmonic-mask sensitivity as a *systematic*, not a correction (rec #5).

    The coarse-channel harmonic mask reduces bias from known comb harmonics,
    but it does not prove the residual is physical (freya CHIME: masking moved
    35.19 -> 42.21 kHz and *both* are instrumental). We therefore report the
    with/without-mask widths and their fractional difference as a systematic
    uncertainty band, never as a corrected value.

    Returns a dict with both widths, the absolute and fractional difference,
    and a ``systematic_frac`` = |masked - unmasked| / unmasked.
    """
    def _val(x):
        return float(x) if (x is not None and np.isfinite(x) and x > 0) else None

    u = _val(dnu_unmasked_mhz)
    m = _val(dnu_masked_mhz)
    if u is None or m is None:
        return {
            "dnu_unmasked_mhz": u,
            "dnu_masked_mhz": m,
            "abs_diff_mhz": None,
            "systematic_frac": None,
            "reason": "need both masked and unmasked widths to report the systematic",
        }
    abs_diff = abs(m - u)
    return {
        "dnu_unmasked_mhz": u,
        "dnu_masked_mhz": m,
        "abs_diff_mhz": abs_diff,
        "systematic_frac": abs_diff / u,
        "reason": "harmonic-mask sensitivity reported as a systematic band, not a correction",
    }


def finalize_measurement_status(
    provenance: dict,
    *,
    off_pulse_null: dict | None = None,
    low_lag_stability: dict | None = None,
) -> dict:
    """Combine the provenance gate with the physical null/stability verdicts.

    For CHIME, a result is a ``measurement`` only if provenance passes AND the
    off-pulse null passes AND the low-lag excision is stable. A failed null or
    a collapsing wing forces ``diagnostic_only`` even when every mitigation was
    enabled (the freya CHIME case: full mitigation stack, but arm A/B1 fail).
    Non-CHIME results keep their provenance status untouched.

    Returns a dict: ``status``, ``downgraded`` (bool), ``failed_checks`` (list).
    """
    if not provenance.get("is_chime"):
        return {
            "status": provenance.get("status", MEASUREMENT),
            "downgraded": False,
            "failed_checks": [],
        }

    failed: list[str] = []
    if provenance.get("status") == DIAGNOSTIC_ONLY:
        failed.append("provenance:" + ",".join(provenance.get("missing", [])))
    if off_pulse_null is not None and off_pulse_null.get("null_pass") is False:
        failed.append("off_pulse_null")
    if low_lag_stability is not None and low_lag_stability.get("stable") is False:
        failed.append("low_lag_stability")

    status = MEASUREMENT if not failed else DIAGNOSTIC_ONLY
    return {"status": status, "downgraded": bool(failed), "failed_checks": failed}
