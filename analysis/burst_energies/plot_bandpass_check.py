#!/usr/bin/env python
"""Bandpass diagnostic: is the DSA gamma_D rail a sensitivity artifact or astrophysics?

For each burst with a joint fit and a staged DSA .npy, overplot (per channel, vs frequency):
  - the S/N spectrum that was actually fit (sum over the on-pulse of data/noise), and the
    fitted power law c0_D (nu/nu_ref)^gamma_D evaluated on it; both normalized to peak;
  - the FLUX-CALIBRATED spectrum sigma_S(nu) * sn_integrated (per-channel fluence [Jy*ms]),
    normalized to peak.

Hypothesis (figures.manifest.json): if gamma_D ~ -5 is DSA bandpass rolloff / beam-edge
sensitivity loss rather than an astrophysical steep spectrum, dividing by the per-channel
noise (the z-score) and multiplying by sigma_S(nu) FLATTENS the calibrated spectrum -- its
log-log effective slope is markedly less negative than gamma_D. A calibrated slope that still
tracks gamma_D would instead support a genuine steep spectrum.

Run from the repo root: python analysis/burst_energies/plot_bandpass_check.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analysis.dsa_beam import beam_gain  # noqa: E402
from analysis.flux_cal import (  # noqa: E402
    _dsa_burst_config,
    burst_epoch_position,
    dsa_beam_offset,
    dsa_pointing_dec,
    dsa_sigma_jy,
    load_dsa_sefd,
    sn_spectrum_from_npy,
)

JOINT_DIR = REPO / "analysis" / "scattering-refit-2026-06" / "joint_json"
OUT_DIR = REPO / "analysis" / "burst_energies"
NU_REF = 1.405e9  # DSA band centre [Hz] (matches calculate_burst_energies)


def joint_dsa_params():
    """nick (lower) -> (c0_D, gamma_D) median, for bursts whose joint fit has per-band amplitudes."""
    out = {}
    for p in sorted(JOINT_DIR.glob("*_joint_fit.json")):
        d = json.loads(p.read_text())
        pct = d.get("percentiles", {})
        if "c0_D" in pct and "gamma_D" in pct:
            out[d["burst"].lower()] = (pct["c0_D"]["median"], pct["gamma_D"]["median"])
    return out


def loglog_slope(freq_hz, y):
    """Effective power-law slope of y(nu) over channels where y>0 (log-log least squares)."""
    m = np.isfinite(y) & (y > 0)
    if m.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(freq_hz[m]), np.log(y[m]), 1)[0])


def main() -> None:
    fits = joint_dsa_params()
    nicks = []
    for cfg in sorted((REPO / "configs" / "batch" / "dsa").glob("*_dsa.yaml")):
        nick = cfg.name[: -len("_dsa.yaml")]
        npy, _, _ = _dsa_burst_config(nick)
        if nick in fits and npy.exists():
            nicks.append(nick)
    if not nicks:
        sys.exit("no bursts with both a joint gamma_D fit and a staged DSA .npy (data/dsa/)")

    ncol = 3
    nrow = -(-len(nicks) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.1 * nrow), squeeze=False)
    summary = []
    for ax, nick in zip(axes.flat, nicks, strict=False):
        c0_D, gamma_D = fits[nick]
        npy, ff, tf = _dsa_burst_config(nick)
        freq_hz, sn_int, dt_ms, dnu_hz = sn_spectrum_from_npy(npy, "dsa", ff, tf)
        _, _, dec = burst_epoch_position(nick)
        theta, phi = dsa_beam_offset(dec, dsa_pointing_dec(nick))
        g_ref = beam_gain(theta, phi, NU_REF / 1e9)
        sigma = dsa_sigma_jy(
            freq_hz, dnu_hz, load_dsa_sefd(nick), dt_ms / 1e3, theta, phi, beam_gain
        )
        cal = sigma * dt_ms * sn_int  # per-channel fluence [Jy*ms]
        model = c0_D * (freq_hz / NU_REF) ** gamma_D  # fitted S/N power law

        f_ghz = freq_hz / 1e9
        sn_slope = loglog_slope(freq_hz, sn_int)
        cal_slope = loglog_slope(freq_hz, cal)
        d_slope = cal_slope - sn_slope  # the log-log slope sigma_S(nu) adds; >0 = flatter
        nrm = lambda a: a / np.nanmax(np.abs(a))  # noqa: E731 peak-normalize for shape comparison
        ax.plot(f_ghz, nrm(sn_int), "o", ms=3, color="0.45", label="S/N spectrum (fit)")
        ax.plot(
            f_ghz, nrm(model), "--", color="0.45", lw=1.2, label=rf"$\nu^{{{gamma_D:.1f}}}$ fit"
        )
        ax.plot(f_ghz, nrm(cal), "s", ms=3, color="C3", label="flux-calibrated")
        ax.set_title(
            f"{nick}  off={theta:.1f}deg G={g_ref:.2f}\n"
            rf"$\gamma_D$={gamma_D:.1f}  S/N={sn_slope:.1f}$\to$cal={cal_slope:.1f}",
            fontsize=8,
        )
        ax.tick_params(labelsize=7)
        ax.axhline(0, color="0.85", lw=0.6, zorder=0)
        summary.append(
            {
                "burst": nick,
                "theta_deg": round(theta, 2),
                "G_ref": round(g_ref, 3),
                "gamma_D": round(gamma_D, 2),
                "sn_slope": round(sn_slope, 2),
                "cal_slope": round(cal_slope, 2),
                "d_slope_from_sigmaS": round(d_slope, 2),
                "flattened_by_cal": d_slope > 0.3,  # sigma_S(nu) measurably flattens this burst
            }
        )
    for ax in axes.flat[len(nicks) :]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=6, loc="upper right")
    fig.supxlabel("frequency [GHz]", fontsize=9)
    fig.supylabel("peak-normalized per-channel amplitude", fontsize=9)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "bandpass_check.png"
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")

    n_flat = sum(s["flattened_by_cal"] for s in summary)
    expectation = (
        "Per burst: red flux-calibrated squares vs grey S/N points + dashed nu^gamma_D fit, each "
        "peak-normalized. The title shows S/N->cal log-log slope. HYPOTHESIS: sigma_S(nu) rises "
        "toward the band edges where the beam gain G falls, so calibrating FLATTENS off-axis "
        "bursts (cal slope less negative than the S/N slope) while leaving on-axis bursts (G~1, "
        "flat sigma_S) unchanged. Confirm visually: for the off-axis railed bursts (freya G=0.20, "
        "chromatica G=0.25) the red squares fall LESS steeply than the grey points/dashed fit; for "
        "on-axis phineas (G=1.0) red and grey overlap (steep spectrum persists -> NOT a beam "
        "artifact there). Check axes 1.31-1.50 GHz, annotated slopes match the visual, no empty "
        "panels."
    )
    manifest = {
        "generated_by": "analysis/burst_energies/plot_bandpass_check.py",
        "figures": [{"path": "bandpass_check.png", "expectation": expectation}],
        "per_burst": summary,
        "n_flattened_by_calibration": f"{n_flat}/{len(summary)}",
        "interpretation": (
            "gamma_D rail is PARTLY a beam-edge sensitivity artifact (flattens for off-axis "
            "bursts) but persists on-axis (phineas) -- not uniformly instrumental."
        ),
        "caveat_absolute_scale": (
            "SHAPE/slope comparison is robust; ABSOLUTE Jy scale (band-avg fluences 218-6860 Jy*ms) "
            "is high vs published DSA fluences and is a separate cross-check (SEFD/n_pol/beam), not "
            "asserted here."
        ),
        "caveat_slope_proxy": (
            "sn_slope/cal_slope are naive log-log fits over positive channels, a proxy for the "
            "MCMC gamma_D (2D fit); they differ in magnitude, so compare cal vs sn (the sigma_S "
            "effect), not cal vs gamma_D."
        ),
    }
    (OUT_DIR / "figures.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"wrote {OUT_DIR / 'figures.manifest.json'}; flattened-by-calibration: {n_flat}/{len(summary)}"
    )
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
