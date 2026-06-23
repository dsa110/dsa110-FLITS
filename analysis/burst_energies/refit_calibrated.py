#!/usr/bin/env python
"""Rigorous test of the gamma_D rail: re-fit the FLUX-CALIBRATED DSA spectra.

The naive bandpass proxy (plot_bandpass_check.py) just measured a log-log slope. This instead
re-runs the actual MCMC scattering fit (model M2: c0, t0, gamma, tau_1ghz) twice per burst, with an
identical setup, on:
  - the S/N data d(nu,t)               (uniform per-channel noise -- what the original fit saw), and
  - the flux-calibrated data d*sigma_S(nu), with per-channel noise sigma_S(nu).

sigma_S(nu) carries the measured beam G(nu) and the coherent-beam SEFD, so calibrating DOWN-WEIGHTS
the beam-edge channels. This handles the two confounds the proxy could not: negative-S/N channels
(full 2D Gaussian likelihood, not positive-only log-log) and the beam model (the full measured G(nu)
shape enters, not a single slope), and it re-optimizes tau/profile jointly.

Analytic backbone (the oracle this cross-checks): for a power-law spectrum the noise-weighted fit
satisfies gamma_cal = gamma_SN + slope(sigma_S) exactly -- multiplying the data by sigma_S tilts the
spectrum, and the matching 1/sigma_S noise weighting leaves the channel weights unchanged, so the
recovered index shifts by exactly the beam slope. slope(sigma_S) > 0 for an off-axis burst (sigma_S
rises toward the band edges where G falls), so calibrating makes gamma LESS negative (relaxes the
rail); on-axis (G~1, flat sigma_S) it does not move. The MCMC d_gamma should reproduce slope_sigmaS
within the posterior width; a large mismatch means the fit has not converged or the spectrum is not a
clean power law.

Run from the repo root: python analysis/burst_energies/refit_calibrated.py [burst ...]
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
sys.path.insert(0, str(REPO / "scattering"))  # io.py uses package-relative imports

from scat_analysis.burstfit import FRBFitter, FRBModel, FRBParams, build_priors  # noqa: E402
from scat_analysis.config_utils import load_telescope_block  # noqa: E402
from scat_analysis.pipeline.io import BurstDataset  # noqa: E402

from analysis.dsa_beam import beam_gain  # noqa: E402
from analysis.flux_cal import (  # noqa: E402
    _dsa_burst_config,
    burst_epoch_position,
    dsa_beam_offset,
    dsa_pointing_dec,
    dsa_sigma_jy,
    load_dsa_sefd_beam,
)

OUT_DIR = REPO / "analysis" / "burst_energies"
JOINT_DIR = REPO / "analysis" / "scattering-refit-2026-06" / "joint_json"
RAILED = ["chromatica", "oran", "phineas", "zach", "freya"]  # gamma_D ~ -5 in the joint fit
N_STEPS = 5000
DISCARD_FRAC = 0.5
SEED = 0  # reproducible walker draws
GAMMA_FLOOR = -10.0  # open below the default -5 hard bound so the calibration shift is not clipped
# at the rail (else gamma_cal = gamma_SN + slope(sigma_S) is invisible there)


def joint_gamma_d(nick):
    """Original joint-fit gamma_D median for a burst (None if no joint fit), for context."""
    p = JOINT_DIR / f"{nick}_joint_fit.json"
    if not p.exists():
        return None
    pct = json.loads(p.read_text()).get("percentiles", {})
    return pct.get("gamma_D", {}).get("median")


def _fit_gamma(model, init, n_steps=None):
    """(median, lo16, hi84) of the M2 gamma posterior for one FRBModel.

    absolute_bounds=True gives gamma the init-independent hard prior [-5, 5]; the default
    init-anchored window collapses to [-0.5, 0.5] for gamma=0 (burstfit.py:1320-1327) and would
    strangle the chain before it could reach the -5 rail this test is probing. walker_width_frac is
    widened from the 0.01 default so the ensemble starts spread across the prior (a tight ball at
    gamma=0 cannot diffuse to a -5 mode in a broad narrow-band posterior).
    """
    n_steps = n_steps or N_STEPS
    priors, _ = build_priors(init, absolute_bounds=True)
    priors = {**priors, "gamma": (GAMMA_FLOOR, 5.0)}  # open the -5 floor (see GAMMA_FLOOR)
    fitter = FRBFitter(model, priors, n_steps=n_steps, n_walkers_mult=8, walker_width_frac=0.3)
    sampler = fitter.sample(init, model_key="M2")
    gcol = FRBFitter._ORDER["M2"].index("gamma")  # gamma sampled in linear space
    chain = sampler.get_chain(discard=int(DISCARD_FRAC * n_steps), flat=True)[:, gcol]
    return float(np.median(chain)), float(np.percentile(chain, 16)), float(np.percentile(chain, 84))


def refit_burst(nick, tel):
    """Fit S/N and flux-calibrated DSA spectra for one burst; return the gamma comparison."""
    npy, ff, tf = _dsa_burst_config(nick)
    ds = BurstDataset(
        npy, npy, telescope=tel, f_factor=ff, t_factor=tf, onpulse_crop=True, onpulse_thresh=3.0
    )
    m = ds.model  # S/N data (z-scored), noise_std ~ 1 per channel
    _, _, dec = burst_epoch_position(nick)
    theta, phi = dsa_beam_offset(dec, dsa_pointing_dec(nick))
    sigma = dsa_sigma_jy(
        m.freq * 1e9,
        ds.df_MHz * 1e6,
        load_dsa_sefd_beam(nick),
        ds.dt_ms / 1e3,
        theta,
        phi,
        beam_gain,
    )  # per-channel sigma_S(nu) [Jy]
    slope_sigma = float(np.polyfit(np.log(m.freq * 1e9), np.log(sigma), 1)[0])  # analytic d_gamma

    prof = np.nansum(m.data, axis=0)
    t0 = float(m.time[int(np.argmax(prof))])
    dt = float(m.time[1] - m.time[0])

    def init_for(data):
        p = np.nansum(data, axis=0)
        return FRBParams(
            c0=float(np.nanmax(p)) or 1.0,
            t0=t0,
            gamma=0.0,
            zeta=dt,
            tau_1ghz=0.1,
            alpha=4.0,
            delta_dm=0.0,
        )

    g_sn = _fit_gamma(m, init_for(m.data))
    cal = m.data * sigma[:, None]
    m_cal = FRBModel(
        time=m.time,
        freq=m.freq,
        data=cal,
        df_MHz=m.df_MHz,
        dm_init=m.dm_init,
        noise_std=m.noise_std * sigma,
    )
    g_cal = _fit_gamma(m_cal, init_for(cal))
    d_gamma = g_cal[0] - g_sn[0]
    return {
        "burst": nick,
        "theta_deg": round(theta, 2),
        "G_ref": round(beam_gain(theta, phi, 1.405), 3),
        "gamma_D_joint": (round(j, 2) if (j := joint_gamma_d(nick)) is not None else None),
        "gamma_SN": round(g_sn[0], 2),
        "gamma_SN_lo": round(g_sn[1], 2),
        "gamma_SN_hi": round(g_sn[2], 2),
        "gamma_cal": round(g_cal[0], 2),
        "gamma_cal_lo": round(g_cal[1], 2),
        "gamma_cal_hi": round(g_cal[2], 2),
        "d_gamma_mcmc": round(d_gamma, 2),  # >0 = relaxed (less negative)
        "d_gamma_analytic": round(slope_sigma, 2),  # oracle: gamma_cal - gamma_SN should equal this
        "relaxed": d_gamma > 0.5,
    }


def main() -> None:
    np.random.seed(SEED)
    nicks = sys.argv[1:] or RAILED
    tel = load_telescope_block(str(REPO / "scattering" / "configs" / "telescopes.yaml"), "dsa")
    rows = [refit_burst(n, tel) for n in nicks]

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    x = np.arange(len(rows))
    for off, key, c, lab in (
        (-0.14, "SN", "0.45", "S/N fit"),
        (0.14, "cal", "C3", "flux-calibrated"),
    ):
        med = [r[f"gamma_{key}"] for r in rows]
        lo = [r[f"gamma_{key}"] - r[f"gamma_{key}_lo"] for r in rows]
        hi = [r[f"gamma_{key}_hi"] - r[f"gamma_{key}"] for r in rows]
        ax.errorbar(x + off, med, yerr=[lo, hi], fmt="o", color=c, label=lab, capsize=3)
    jx = [i for i, r in enumerate(rows) if r["gamma_D_joint"] is not None]
    ax.scatter(
        jx,
        [rows[i]["gamma_D_joint"] for i in jx],
        marker="_",
        s=400,
        color="C0",
        label="joint gamma_D (orig)",
        zorder=5,
    )
    ax.axhline(-4.0, color="0.8", lw=0.8, ls=":")  # Kolmogorov-ish reference
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['burst']}\nG={r['G_ref']:.2f}\nslope_s={r['d_gamma_analytic']:+.1f}" for r in rows],
        fontsize=8,
    )
    ax.set_ylabel(r"DSA spectral index $\gamma_D$ (M2)")
    ax.set_title("gamma_D: S/N fit vs flux-calibrated re-fit (rigorous MCMC); blue = joint orig")
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = OUT_DIR / "refit_calibrated.png"
    fig.savefig(png, dpi=130)

    n_relaxed = sum(r["relaxed"] for r in rows)
    manifest = {
        "generated_by": "analysis/burst_energies/refit_calibrated.py",
        "figures": [
            {
                "path": "refit_calibrated.png",
                "expectation": (
                    "Per burst: grey = gamma_D from the S/N fit, red = from the flux-calibrated re-fit (M2 "
                    "MCMC, error bars 16-84 pct); blue dash = the original joint gamma_D. x-labels give the "
                    "beam gain G and the analytic prediction slope_s = slope(sigma_S). HYPOTHESIS: red sits "
                    "ABOVE grey by ~slope_s (rail relaxes when beam-edge channels are down-weighted), "
                    "strongest for off-axis low-G bursts (freya G=0.20, chromatica G=0.25, slope_s large +); "
                    "for on-axis phineas (G=1.0, slope_s~0) red ~= grey. The gamma floor is opened to -10 "
                    "so the shift is not clipped at the old -5 rail. CHECK: (1) red-minus-grey tracks the "
                    "slope_s annotation per burst; (2) no panel has collapsed error bars; (3) gamma_SN "
                    "reaching well below -5 for the steep bursts means the single-band DSA falloff is REAL "
                    "(the joint -5 blue dashes were the old prior floor, not the true preference) -- so the "
                    "rail is a genuinely steep DSA spectrum that calibration only partly (off-axis) relaxes."
                ),
            }
        ],
        "per_burst": rows,
        "n_relaxed_of": f"{n_relaxed}/{len(rows)}",
        "method": (
            f"M2 MCMC, {N_STEPS} steps, discard {DISCARD_FRAC:.0%}, seed {SEED}; identical setup S/N "
            "vs calibrated; calibrated noise = sigma_S(nu) so beam-edge channels down-weight. "
            "d_gamma_analytic = slope(sigma_S) is the oracle the MCMC d_gamma should match."
        ),
    }
    (OUT_DIR / "refit_calibrated.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {png} and refit_calibrated.manifest.json; relaxed {n_relaxed}/{len(rows)}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
