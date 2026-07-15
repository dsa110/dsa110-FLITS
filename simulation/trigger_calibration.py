"""A1 escalation-trigger injection calibration (Faber2026 plan, Phases 3-4).

Truth-known single- and two-screen dynamic spectra rendered to per-subband
ACFs through the production ``calculate_acfs_for_subbands`` path (the sim_gate
pattern: injections go through the estimator actually used on science data,
not an idealization), plus the calibration-campaign kernel that maps the
null dlnZ distribution to false-escalation thresholds.

Conventions: gamma = HWHM = Delta nu_d (production convention, C1 = 1);
one-sided lag-0-excluded ACFs; two multiplicative screens for the two-screen
truth (the M2 cross-term physics).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_FLITS_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_FLITS_ROOT / "scintillation"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scint_analysis.acf_covariance import _one_realization  # noqa: E402
from scint_analysis.analysis import calculate_acfs_for_subbands  # noqa: E402
from scint_analysis.core import DynamicSpectrum  # noqa: E402

_BASE_CONFIG = {
    "analysis": {
        "acf": {"num_subbands": 4, "use_snr_subbanding": False,
                "max_lag_mhz": 45.0},
        "noise": {"disable_template": True},
        "self_noise": {"disable": True},
        "rfi_masking": {"use_symmetric_noise_window": True},
    }
}


def _interp_hwhm(lags, acf):
    """Interpolated half-max width — the production gamma-hat convention."""
    half = 0.5 * float(np.max(acf))
    return float(np.interp(half, acf[::-1], lags[::-1]))


def _render_dynamic_spectrum(rng, gain_spectrum, n_chan, snr,
                             n_time=256, n_burst=32):
    """2D dynamic spectrum: scintillated burst + unit radiometer noise.

    The burst occupies the central ``n_burst`` time bins with a flat profile;
    amplitude is set so the *time-averaged on-burst* spectrum has per-channel
    S/N ``snr`` (noise of the average = 1/sqrt(n_burst))."""
    power = rng.normal(0.0, 1.0, (n_chan, n_time))
    t0 = (n_time - n_burst) // 2
    amp = snr / np.sqrt(n_burst)
    power[:, t0:t0 + n_burst] += amp * gain_spectrum[:, None]
    return power, (t0, t0 + n_burst)


def _run_subband_acfs(rng, gain_spectrum, band_width_mhz, channel_width_mhz,
                      snr, num_subbands):
    n_chan = gain_spectrum.size
    power, burst_lims = _render_dynamic_spectrum(rng, gain_spectrum, n_chan,
                                                 snr)
    freqs = 600.0 + channel_width_mhz * np.arange(n_chan)
    times = 1e-3 * np.arange(power.shape[1])
    ds = DynamicSpectrum(power, freqs, times)

    config = {
        "analysis": {
            **_BASE_CONFIG["analysis"],
            "acf": {**_BASE_CONFIG["analysis"]["acf"],
                    "num_subbands": num_subbands},
        }
    }
    res = calculate_acfs_for_subbands(ds, config, burst_lims, noise_desc=None)

    subbands = []
    for acf, lags, err, cfreq, cw in zip(
        res["subband_acfs"], res["subband_lags_mhz"],
        res["subband_acfs_err"], res["subband_center_freqs_mhz"],
        res["subband_channel_widths_mhz"],
    ):
        acf = np.asarray(acf)
        lags = np.asarray(lags)
        pos = lags > 0
        subbands.append({
            "lags": lags[pos],
            "acf": acf[pos],
            "err": np.asarray(err)[pos] if err is not None else None,
            "center_freq_mhz": float(cfreq),
            "channel_width_mhz": float(cw),
            "dnu_hwhm_est_mhz": _interp_hwhm(lags[pos], acf[pos]),
        })
    return subbands


def inject_single_screen(dnu_hwhm_mhz, band_width_mhz, channel_width_mhz,
                         snr, num_subbands, seed):
    """Single-screen injection through the production subband-ACF path."""
    rng = np.random.default_rng(seed)
    n_chan = int(round(band_width_mhz / channel_width_mhz))
    gain = _one_realization(rng, n_chan,
                            gamma_bins=dnu_hwhm_mhz / channel_width_mhz,
                            mod_index=1.0, snr=np.inf)
    subbands = _run_subband_acfs(rng, gain, band_width_mhz,
                                 channel_width_mhz, snr, num_subbands)
    return {
        "truth": {"screens": 1, "dnu_hwhm_mhz": float(dnu_hwhm_mhz),
                  "mod_index": 1.0, "snr": float(snr)},
        "subbands": subbands,
        "convention": "HWHM",
        "channel_width_mhz": float(channel_width_mhz),
        "first_fit_lag": 1,
        "seed": int(seed),
    }


def inject_two_screen(dnu1_hwhm_mhz, dnu2_hwhm_mhz, m2_ratio,
                      band_width_mhz, channel_width_mhz, snr,
                      num_subbands, seed):
    """Two multiplicative independent screens (narrow x wide)."""
    rng = np.random.default_rng(seed)
    n_chan = int(round(band_width_mhz / channel_width_mhz))
    g1 = _one_realization(rng, n_chan,
                          gamma_bins=dnu1_hwhm_mhz / channel_width_mhz,
                          mod_index=1.0, snr=np.inf)
    m2 = float(np.sqrt(np.clip(m2_ratio, 0.0, 1.0)))
    g2 = _one_realization(rng, n_chan,
                          gamma_bins=dnu2_hwhm_mhz / channel_width_mhz,
                          mod_index=m2, snr=np.inf)
    gain = g1 * g2
    subbands = _run_subband_acfs(rng, gain, band_width_mhz,
                                 channel_width_mhz, snr, num_subbands)
    return {
        "truth": {"screens": 2, "dnu1_hwhm_mhz": float(dnu1_hwhm_mhz),
                  "dnu2_hwhm_mhz": float(dnu2_hwhm_mhz),
                  "f": float(dnu2_hwhm_mhz / dnu1_hwhm_mhz),
                  "m2_ratio": float(m2_ratio), "snr": float(snr)},
        "subbands": subbands,
        "convention": "HWHM",
        "channel_width_mhz": float(channel_width_mhz),
        "first_fit_lag": 1,
        "seed": int(seed),
    }


# --------------------------------------------------------------------------
# Phase 4: calibration-campaign kernel
# --------------------------------------------------------------------------


def _burst_dlnz(injection, snr, n_real_cov, nlive, dlogz, seed):
    """Summed dlnZ over a burst's valid subbands, each with its matched MC
    covariance (null-conditioned on that subband's own gamma-hat)."""
    from scint_analysis.acf_covariance import mc_acf_covariance
    from scint_analysis.acf_evidence import compare_acf_evidence

    total = 0.0
    n_used = 0
    for k, s in enumerate(injection["subbands"]):
        lags, acf = s["lags"], s["acf"]
        if lags.size < 20:
            continue
        band = lags.size * s["channel_width_mhz"] * 4.0  # ~subband width
        n_chan_band = int(round(band / s["channel_width_mhz"]))
        cov = mc_acf_covariance(
            gamma_hwhm_mhz=max(s["dnu_hwhm_est_mhz"],
                               0.5 * s["channel_width_mhz"]),
            mod_index=float(np.sqrt(np.clip(acf[0], 0.05, 1.0))),
            band_width_mhz=n_chan_band * s["channel_width_mhz"],
            channel_width_mhz=s["channel_width_mhz"],
            snr=snr, n_real=n_real_cov,
            max_lag_bins=lags.size + 1, seed=seed + 17 * k,
        )
        n_lag = cov.shape[0]
        res = compare_acf_evidence(
            lags[:n_lag], acf[:n_lag], cov,
            channel_width_mhz=s["channel_width_mhz"],
            band_width_mhz=band,
            nlive=nlive, dlogz=dlogz, seed=seed + 1000 + 17 * k,
        )
        total += res["dlnz"]
        n_used += 1
    if n_used == 0:
        raise RuntimeError("no valid subbands in injection")
    return total


def null_dlnz_cell(dnu_hwhm_mhz, snr, band_width_mhz, channel_width_mhz,
                   num_subbands, n_real, seed, nlive=500, dlogz=0.1,
                   n_real_cov=150):
    """dlnZ null sample: n_real single-screen injections through the full
    trigger path. Failed evidence runs (maxcall cap) are recorded as NaN and
    excluded by ``threshold_table`` — counted, never silently dropped."""
    out = []
    for r in range(n_real):
        s = seed + 1000 * r
        inj = inject_single_screen(dnu_hwhm_mhz, band_width_mhz,
                                   channel_width_mhz, snr, num_subbands,
                                   seed=s)
        try:
            out.append(_burst_dlnz(inj, snr, n_real_cov, nlive, dlogz,
                                   seed=s))
        except RuntimeError:
            out.append(np.nan)
    return out


def power_dlnz_cell(f, m2_ratio, dnu1_hwhm_mhz, snr, band_width_mhz,
                    channel_width_mhz, num_subbands, n_real, seed,
                    nlive=500, dlogz=0.1, n_real_cov=150):
    """dlnZ sample under two-screen truth (detection-power curve input)."""
    out = []
    for r in range(n_real):
        s = seed + 1000 * r
        inj = inject_two_screen(dnu1_hwhm_mhz, f * dnu1_hwhm_mhz, m2_ratio,
                                band_width_mhz, channel_width_mhz, snr,
                                num_subbands, seed=s)
        try:
            out.append(_burst_dlnz(inj, snr, n_real_cov, nlive, dlogz,
                                   seed=s))
        except RuntimeError:
            out.append(np.nan)
    return out


def threshold_table(null_samples_by_cell, rates=(0.005, 0.01, 0.05)):
    """Conservative envelope: for each rate, the max over cells of that
    cell's (1 - rate) dlnZ quantile. NaNs (evidence_failed) are excluded per
    cell; a cell with <50% finite samples raises."""
    table = {}
    for rate in rates:
        per_cell = []
        for cell, sample in null_samples_by_cell.items():
            arr = np.asarray(sample, dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size < 0.5 * arr.size or finite.size == 0:
                raise ValueError(
                    f"cell {cell}: {arr.size - finite.size}/{arr.size} "
                    "evidence failures; cannot set a threshold"
                )
            per_cell.append(float(np.quantile(finite, 1.0 - rate)))
        table[rate] = max(per_cell)
    return table
