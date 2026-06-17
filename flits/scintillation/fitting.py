"""Model functions and fitters for scintillation analysis.

Lorentzian and power-law models plus lmfit-based fitters migrated from the
legacy `scint_pipeline_funcs` module: a Lorentzian fit to an ACF, and the
scintillation-bandwidth vs frequency power-law relation \u0394\u03bd_d \u221d \u03bd^\u03b1.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from lmfit import Model, Parameters

__all__ = [
    "lorentzian",
    "lorentzian_with_const",
    "double_lorentzian_with_const",
    "power_law",
    "fit_lorentzian_acf",
    "fit_scint_bandwidth_freq_relation",
]


def lorentzian(
    x: NDArray[np.floating], amp: float, cen: float, wid: float
) -> NDArray[np.floating]:
    """Standard Lorentzian profile.

    Parameters
    ----------
    x : ndarray
        Independent variable (e.g. ACF lag).
    amp : float
        Peak amplitude.
    cen : float
        Centre of the profile.
    wid : float
        Half-width at half-maximum (HWHM).

    Returns
    -------
    ndarray
        Lorentzian evaluated at ``x``.
    """
    return amp * wid ** 2 / ((x - cen) ** 2 + wid ** 2)


def lorentzian_with_const(
    x: NDArray[np.floating], amp: float, cen: float, wid: float, c: float
) -> NDArray[np.floating]:
    """Lorentzian plus a constant offset (see :func:`lorentzian`)."""
    return lorentzian(x, amp, cen, wid) + c


def double_lorentzian_with_const(
    x: NDArray[np.floating],
    amp1: float,
    cen1: float,
    wid1: float,
    amp2: float,
    cen2: float,
    wid2: float,
    c: float,
) -> NDArray[np.floating]:
    """Sum of two Lorentzians plus a constant offset (see :func:`lorentzian`)."""
    return lorentzian(x, amp1, cen1, wid1) + lorentzian(x, amp2, cen2, wid2) + c


def power_law(
    x: NDArray[np.floating], amp: float, index: float
) -> NDArray[np.floating]:
    """Power-law model ``amp * x ** index`` (x assumed positive)."""
    return amp * np.power(x, index)


def fit_lorentzian_acf(
    lags: NDArray[np.floating],
    acf: NDArray[np.floating],
    errs: Optional[NDArray[np.floating]] = None,
    center_guess: float = 0.0,
    const_offset: bool = True,
) -> Tuple[Optional[Parameters], Optional[Model], Optional[Any]]:
    """Fit a Lorentzian (optionally with offset) to the core of an ACF.

    Only lags where the ACF exceeds 10 % of its peak are fitted, isolating the
    central scintillation peak.

    Parameters
    ----------
    lags : ndarray
        ACF lags, centred near zero.
    acf : ndarray
        ACF values.
    errs : ndarray, optional
        1-sigma uncertainties for weighted fitting (``weights = 1 / errs**2``).
    center_guess : float, optional
        Initial guess for the Lorentzian centre. Default 0.0.
    const_offset : bool, optional
        Include a constant offset term. Default True.

    Returns
    -------
    params : lmfit.Parameters or None
        Best-fit parameters, or None if the fit could not be performed.
    model : lmfit.Model or None
        The model used, or None.
    result : lmfit.model.ModelResult or None
        The full fit result, or None.
    """
    lags = np.asarray(lags, dtype=np.float64)
    acf = np.asarray(acf, dtype=np.float64)

    center_idx = np.argmax(acf)
    fit_mask = acf > 0.1 * acf[center_idx]
    if not np.any(fit_mask):
        print("Warning: No suitable data points found for Lorentzian fit.")
        return None, None, None

    x_fit = lags[fit_mask]
    y_fit = acf[fit_mask]
    weights = None
    if errs is not None:
        weights = 1.0 / np.asarray(errs, dtype=np.float64)[fit_mask] ** 2

    params = Parameters()
    params.add("amp", value=float(np.max(y_fit)), min=0)
    params.add("cen", value=center_guess, vary=True)
    params.add("wid", value=float(np.std(x_fit)), min=1e-9)
    if const_offset:
        model = Model(lorentzian_with_const)
        params.add("c", value=float(np.min(y_fit)))
    else:
        model = Model(lorentzian)

    try:
        result = model.fit(y_fit, params, x=x_fit, weights=weights)
        return result.params, model, result
    except Exception as exc:  # noqa: BLE001 - report and degrade gracefully
        print(f"Error during Lorentzian fit: {exc}")
        return None, None, None


def fit_scint_bandwidth_freq_relation(
    freqs: NDArray[np.floating],
    scint_widths: NDArray[np.floating],
    errs: Optional[NDArray[np.floating]] = None,
) -> Tuple[Optional[Parameters], Optional[Model], Optional[Any]]:
    """Fit the \u0394\u03bd_d \u221d \u03bd^\u03b1 power-law relation.

    Parameters
    ----------
    freqs : ndarray
        Sub-band centre frequencies.
    scint_widths : ndarray
        Measured scintillation bandwidths (e.g. Lorentzian HWHM).
    errs : ndarray, optional
        1-sigma uncertainties for weighted fitting.

    Returns
    -------
    params : lmfit.Parameters or None
        Best-fit parameters (``amp``, ``index``), or None on failure.
    model : lmfit.Model or None
        The power-law model, or None.
    result : lmfit.model.ModelResult or None
        The full fit result, or None.

    Raises
    ------
    ValueError
        If ``freqs`` and ``scint_widths`` (or ``errs``) differ in length.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    scint_widths = np.asarray(scint_widths, dtype=np.float64)
    if len(freqs) != len(scint_widths):
        raise ValueError(
            "Frequency and scintillation width arrays must have the same length."
        )

    weights = None
    if errs is not None:
        errs = np.asarray(errs, dtype=np.float64)
        if len(errs) != len(freqs):
            raise ValueError(
                "Error array must have the same length as frequency array."
            )
        weights = 1.0 / errs ** 2

    valid = (
        (freqs > 0)
        & (scint_widths > 0)
        & ~np.isnan(freqs)
        & ~np.isnan(scint_widths)
    )
    if errs is not None:
        valid &= ~np.isnan(errs) & (errs > 0)
    if np.sum(valid) < 2:
        print("Warning: Fewer than 2 valid points for power law fit.")
        return None, None, None

    x_fit = freqs[valid]
    y_fit = scint_widths[valid]
    weights_fit = weights[valid] if weights is not None else None

    # Seed the non-linear fit with a log-log linear estimate.
    try:
        coeffs = np.polyfit(
            np.log(x_fit),
            np.log(y_fit),
            1,
            w=y_fit * weights_fit if weights_fit is not None else None,
        )
        alpha_guess = coeffs[0]
        amp_guess = np.exp(coeffs[1])
    except Exception:  # noqa: BLE001 - fall back to a Kolmogorov-ish guess
        alpha_guess = 4.0
        amp_guess = float(np.mean(y_fit / (x_fit ** alpha_guess)))

    model = Model(power_law)
    params = Parameters()
    params.add("amp", value=amp_guess, min=1e-12)
    params.add("index", value=alpha_guess)

    try:
        result = model.fit(y_fit, params, x=x_fit, weights=weights_fit)
        return result.params, model, result
    except Exception as exc:  # noqa: BLE001 - report and degrade gracefully
        print(f"Error during power law fit: {exc}")
        return None, None, None
