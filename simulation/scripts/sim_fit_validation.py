#!/usr/bin/env python3
"""End-to-end validation of the simulate -> fit routine.

Simulates a scattered FRB with the two-screen `FRBScintillator`, runs the
`scattering.scat_analysis.burstfit` MCMC fitter on the simulated dynamic
spectrum, and compiles every diagnostic figure along the way:

  1. sim_dynamic_spectrum.png  - simulated waterfall, band-integrated profile,
                                 spectrum, spectral ACF (+ theoretical tau_s/nu_s/RP)
  2. fit_data_model_residual.png - data | best-fit model | residual + profile overlay
  3. fit_corner.png            - posterior corner (physical units), true tau marked
  4. fit_chains.png            - walker traces per parameter + log-prob
  5. recovered_vs_true.png     - tau posterior vs injected tau at the band centre
  6. ensemble_recovery.png     - scintillation averaged over N screen realisations
                                 -> clean pulse-broadening tail -> clean tau recovery

A single realisation is fully scintillated, so its band-integrated profile is a
spiky multipath IRF and tau recovery is loose (the exponential-PBF fitter and the
coherent multipath simulator are different physical models). The ensemble panel
is the quantitative validation: averaging independent screen realisations recovers
the ensemble-average exponential PBF, which the fitter recovers cleanly.

Run:  python simulation/scripts/sim_fit_validation.py [--out DIR] [--ensemble N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import astropy.units as u

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SIM = os.path.join(_REPO, "simulation")
for p in (_REPO, _SIM):
    if p not in sys.path:
        sys.path.insert(0, p)

from engine import SimCfg, FRBScintillator  # noqa: E402
from screen import ScreenCfg  # noqa: E402
from instrument import InstrumentalCfg  # noqa: E402
from sim_fit_bridge import sim_grids_to_fitter  # noqa: E402
from scattering.scat_analysis.burstfit import (  # noqa: E402
    FRBModel, FRBParams, FRBFitter, build_priors, gelman_rubin,
)
from tools.figure_manifest import write_manifest  # noqa: E402

M2_ORDER = ("c0", "t0", "gamma", "tau_1ghz")
LOG_COLS = {"c0", "tau_1ghz"}  # fitter samples these in log-space


def make_cfg(seed_mw=1234, seed_host=5678, nchan=128):
    """Host-dominated single screen (MW negligible); equal N for the cross term.

    Regime chosen so the scattering tail is well-resolved and high-SNR:
    host L=40 AU -> tau_s ~ 0.3 ms (~57 time bins); peak 200 Jy vs SEFD 0.5 Jy
    -> high per-bin SNR; nu_s ~ 0.5 kHz << channel width -> near-smooth spectrum.
    """
    return SimCfg(
        peak_flux=200 * u.Jy, nu0=800 * u.MHz, bw=25.0 * u.MHz, nchan=nchan,
        z_host=0.192, D_mw=2.3 * u.kpc, D_host_src=2.0 * u.kpc,
        mw=ScreenCfg(N=96, L=0.2 * u.AU, rng_seed=seed_mw),
        host=ScreenCfg(N=96, L=40.0 * u.AU, rng_seed=seed_host),
        intrinsic_pulse="delta",
        instrument=InstrumentalCfg(sefd=0.5 * u.Jy),
    )


def spectral_acf(spectrum):
    s = spectrum - np.nanmean(spectrum)
    ac = np.correlate(s, s, mode="full")
    ac = ac[ac.size // 2:]
    return ac / ac[0]


def fit_dynamic_spectrum(data, time_ms, freq_ghz, df_MHz, *, tau_init_ms=1.0,
                         n_steps=600, n_walkers_mult=8):
    """Fit M2 (c0, t0, gamma, tau_1ghz). Returns (sampler, model, median_params)."""
    model = FRBModel(time=time_ms, freq=freq_ghz, data=data, dm_init=0.0, df_MHz=df_MHz)
    prof = np.nansum(data, axis=0)
    init = FRBParams(
        c0=float(np.nanmax(prof)) or 1.0,
        t0=float(time_ms[int(np.argmax(prof))]),
        gamma=0.0,
        zeta=float(time_ms[1] - time_ms[0]),
        tau_1ghz=tau_init_ms, alpha=4.0, delta_dm=0.0,
    )
    priors, _ = build_priors(init)
    # build_priors floors the t0 half-window at 10 ms; for a ~2 ms burst that lets
    # t0 wander out of the data and the fit rails. Constrain t0 to the data span and
    # tau to sub-ms (the injected scattering time is tens of us).
    priors["t0"] = (float(time_ms[0]), float(time_ms[-1]))
    priors["tau_1ghz"] = (1e-3, 2.0)
    fitter = FRBFitter(model, priors, n_steps=n_steps, n_walkers_mult=n_walkers_mult)
    sampler = fitter.sample(init, model_key="M2")

    chain = sampler.get_chain(discard=n_steps // 2, flat=True)
    med = {}
    for j, name in enumerate(M2_ORDER):
        v = np.median(chain[:, j])
        med[name] = float(np.exp(v)) if name in LOG_COLS else float(v)
    best = FRBParams(c0=med["c0"], t0=med["t0"], gamma=med["gamma"],
                     zeta=init.zeta, tau_1ghz=med["tau_1ghz"], alpha=4.0, delta_dm=0.0)
    return sampler, model, best, med


def physical_chain(sampler, n_steps):
    chain = sampler.get_chain(discard=n_steps // 2, flat=True).copy()
    for j, name in enumerate(M2_ORDER):
        if name in LOG_COLS:
            chain[:, j] = np.exp(chain[:, j])
    return chain


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_SIM, "validation_out"))
    ap.add_argument("--ensemble", type=int, default=24, help="screen realisations (0 to skip)")
    ap.add_argument("--nsteps", type=int, default=600)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.default_rng(0)
    summary = {}

    # ----- Stage 1: simulate -------------------------------------------------
    print("[1/4] Simulating burst ...")
    cfg = make_cfg()
    sim = FRBScintillator(cfg)
    I_t_nu, t_s, f_hz = sim.synthesise_dynamic_spectrum(duration=4.0 * u.ms, rng=rng)
    obs = sim.calculate_theoretical_observables()
    data, t_ms, f_ghz, df = sim_grids_to_fitter(I_t_nu, t_s, f_hz)
    nu0_ghz = cfg.nu0.to_value(u.GHz)
    tau_true_ms = max(obs["tau_s_mw_s"], obs["tau_s_host_s"]) * 1e3
    summary["theory"] = {
        "tau_s_host_ms": obs["tau_s_host_s"] * 1e3, "tau_s_mw_ms": obs["tau_s_mw_s"] * 1e3,
        "nu_s_host_khz": obs["nu_s_host_hz"] / 1e3, "RP": float(sim.resolution_power()),
        "nu0_ghz": nu0_ghz, "dt_ms": float(t_ms[1] - t_ms[0]),
    }

    prof = data.sum(0)
    spec = data.sum(1)
    pk = int(prof.argmax())
    acf = spectral_acf(data[:, pk])

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    extent = [t_ms[0], t_ms[-1], f_ghz[0], f_ghz[-1]]
    ax[0, 0].imshow(data, aspect="auto", origin="lower", extent=extent, cmap="magma")
    ax[0, 0].set(title="Simulated dynamic spectrum I(t,nu)", xlabel="time [ms]", ylabel="freq [GHz]")
    ax[0, 1].plot(t_ms, prof, lw=1, color="0.1")
    ax[0, 1].axvline(t_ms[pk], color="tab:red", ls=":", lw=1, label="peak")
    ax[0, 1].set(title="Band-integrated profile", xlabel="time [ms]", ylabel="flux")
    ax[0, 1].legend(fontsize=8)
    ax[1, 0].plot(f_ghz, spec, lw=0.7, color="tab:blue")
    ax[1, 0].set(title=f"Time-integrated spectrum (mod. index {spec.std()/spec.mean():.2f})",
                 xlabel="freq [GHz]", ylabel="flux")
    lags_khz = np.arange(acf.size) * df * 1e3
    ax[1, 1].plot(lags_khz[:acf.size // 3], acf[:acf.size // 3], lw=1, color="tab:green")
    ax[1, 1].axhline(0, color="0.7", lw=0.6)
    ax[1, 1].set(title=f"Spectral ACF (nu_s,host ~ {obs['nu_s_host_hz']/1e3:.1f} kHz)",
                 xlabel="freq lag [kHz]", ylabel="ACF")
    fig.suptitle(f"Stage 1 - simulated burst | tau_s,host={tau_true_ms*1e3:.1f} us  "
                 f"RP={sim.resolution_power():.3f}  dt={t_ms[1]-t_ms[0]:.3f} ms", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "sim_dynamic_spectrum.png"), dpi=130)
    plt.close(fig)

    # ----- Stage 2: fit ------------------------------------------------------
    print("[2/4] Fitting simulated burst (M2) ...")
    sampler, model, best, med = fit_dynamic_spectrum(data, t_ms, f_ghz, df, n_steps=args.nsteps)
    model_spec = model(best, model_key="M2")
    noise_std = model.noise_std if model.noise_std is not None else np.std(data[:, :data.shape[1] // 4])
    ns = np.clip(np.atleast_1d(noise_std), 1e-9, None)
    ns = ns[:, None] if ns.ndim == 1 else ns
    red_chi2 = float(np.sum(((data - model_spec) / ns) ** 2) / (data.size - len(M2_ORDER)))
    gr = gelman_rubin(sampler)
    tau_fit_1ghz = med["tau_1ghz"]
    tau_fit_nu0 = tau_fit_1ghz * nu0_ghz ** (-4.0)
    summary["fit_single"] = {
        "tau_1ghz_ms": tau_fit_1ghz, "tau_at_nu0_ms": tau_fit_nu0,
        "t0_ms": med["t0"], "gamma": med["gamma"], "c0": med["c0"],
        "reduced_chi2": red_chi2,
        "acceptance": float(np.mean(sampler.acceptance_fraction)),
        "gelman_rubin_max": float(max(gr.values())) if gr else None,
    }

    # data | model | residual
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    vmax = np.nanpercentile(data, 99)
    for a, d, ttl in zip(ax, [data, model_spec, data - model_spec],
                         ["data", "best-fit model (M2)", "residual"]):
        im = a.imshow(d, aspect="auto", origin="lower", extent=extent, cmap="magma",
                      vmin=-vmax if ttl == "residual" else 0, vmax=vmax)
        a.set(title=ttl, xlabel="time [ms]")
        plt.colorbar(im, ax=a, fraction=0.046)
    ax[0].set_ylabel("freq [GHz]")
    fig.suptitle(f"Stage 2 - fit | tau_1GHz={tau_fit_1ghz:.3f} ms  "
                 f"red.chi2={summary['fit_single']['reduced_chi2']:.2f}  "
                 f"acc={summary['fit_single']['acceptance']:.2f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fit_data_model_residual.png"), dpi=130)
    plt.close(fig)

    # corner (physical units)
    try:
        import corner
        chain_phys = physical_chain(sampler, args.nsteps)
        truths = [None, None, None, tau_fit_1ghz]  # mark fitted tau; true is at nu0 (diff ref)
        cfig = corner.corner(chain_phys, labels=[r"$c_0$", r"$t_0$ [ms]", r"$\gamma$",
                             r"$\tau_{1GHz}$ [ms]"], truths=truths, show_titles=True,
                             title_fmt=".3f", quantiles=[0.16, 0.5, 0.84])
        cfig.suptitle("Stage 2 - posterior (M2)", y=1.02, fontsize=11)
        cfig.savefig(os.path.join(args.out, "fit_corner.png"), dpi=120, bbox_inches="tight")
        plt.close(cfig)
    except Exception as e:  # corner missing or degenerate posterior
        print(f"  corner skipped: {e}")

    # chains
    full = sampler.get_chain()  # (nsteps, nwalkers, ndim)
    fig, ax = plt.subplots(len(M2_ORDER) + 1, 1, figsize=(9, 9), sharex=True)
    for j, name in enumerate(M2_ORDER):
        ax[j].plot(full[:, :, j], color="0.3", alpha=0.25, lw=0.5)
        ax[j].set_ylabel(name + (" (log)" if name in LOG_COLS else ""))
    ax[-1].plot(sampler.get_log_prob(), color="tab:purple", alpha=0.25, lw=0.5)
    ax[-1].set(ylabel="log prob", xlabel="step")
    fig.suptitle("Stage 2 - MCMC chains", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fit_chains.png"), dpi=120)
    plt.close(fig)

    # recovered vs true
    chain_phys = physical_chain(sampler, args.nsteps)
    tau_post_nu0 = chain_phys[:, M2_ORDER.index("tau_1ghz")] * nu0_ghz ** (-4.0)
    fig, a = plt.subplots(figsize=(7, 4.2))
    a.hist(tau_post_nu0 * 1e3, bins=50, color="tab:blue", alpha=0.7, density=True,
           label=r"fit $\tau(\nu_0)$ posterior")
    a.axvline(tau_true_ms * 1e3, color="tab:red", lw=2, label=f"injected tau_s = {tau_true_ms*1e3:.1f} us")
    a.set(title="Stage 3 - recovered vs injected scattering time (band centre)",
          xlabel=r"$\tau$ at $\nu_0$ [us]", ylabel="density")
    a.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "recovered_vs_true.png"), dpi=130)
    plt.close(fig)

    # ----- Stage 4: ensemble-averaged clean recovery -------------------------
    if args.ensemble > 0:
        print(f"[3/4] Ensemble: averaging {args.ensemble} screen realisations ...")
        stack = None
        for k in range(args.ensemble):
            c = make_cfg(seed_mw=1000 + k, seed_host=2000 + k)
            s = FRBScintillator(c)
            Ik, tk, fk = s.synthesise_dynamic_spectrum(duration=4.0 * u.ms,
                                                        rng=np.random.default_rng(100 + k))
            stack = Ik if stack is None else stack + Ik
        I_avg = stack / args.ensemble
        data_e, t_e, f_e, df_e = sim_grids_to_fitter(I_avg, tk, fk)
        print("[4/4] Fitting ensemble-averaged burst ...")
        samp_e, model_e, best_e, med_e = fit_dynamic_spectrum(data_e, t_e, f_e, df_e, n_steps=args.nsteps)
        tau_e_nu0 = med_e["tau_1ghz"] * nu0_ghz ** (-4.0)
        prof_e = data_e.sum(0)
        model_e_spec = model_e(best_e, model_key="M2")

        # Empirical PBF e-folding from the clean ensemble profile: this is what an
        # exponential-tail fit *should* recover, and it differs from the nominal
        # tau_s (1/e scattering ANGLE) by an O(few) factor because the screen field
        # extends to +/-2 sigma. Log-linear fit over the tail above the noise floor.
        pe = int(prof_e.argmax())
        tt_ms = t_e[pe:] - t_e[pe]
        tail = prof_e[pe:] - np.median(prof_e[:max(1, pe // 2)])  # subtract baseline
        thr = 0.05 * tail.max()
        m = tail > thr
        tau_eff_ms = float(-1.0 / np.polyfit(tt_ms[m], np.log(tail[m]), 1)[0]) if m.sum() > 3 else np.nan
        summary["fit_ensemble"] = {
            "n_realisations": args.ensemble, "tau_1ghz_ms": med_e["tau_1ghz"],
            "tau_at_nu0_ms": tau_e_nu0, "tau_at_nu0_us": tau_e_nu0 * 1e3,
            "tau_s_nominal_us": tau_true_ms * 1e3,
            "tau_eff_empirical_us": tau_eff_ms * 1e3,
            "ratio_fit_to_empirical": tau_e_nu0 / tau_eff_ms,
            "ratio_empirical_to_nominal": tau_eff_ms / tau_true_ms,
        }
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
        ax[0].imshow(data_e, aspect="auto", origin="lower",
                     extent=[t_e[0], t_e[-1], f_e[0], f_e[-1]], cmap="magma")
        ax[0].set(title=f"Ensemble mean I(t,nu)  (N={args.ensemble}, scintillation suppressed)",
                  xlabel="time [ms]", ylabel="freq [GHz]")
        pe = int(prof_e.argmax())
        ax[1].plot(t_e - t_e[pe], prof_e / prof_e.max(), color="0.1", lw=1.2, label="ensemble profile")
        ax[1].plot(t_e - t_e[pe], model_e_spec.sum(0) / model_e_spec.sum(0).max(),
                   color="tab:red", lw=1.2, ls="--", label="fit model")
        tt = np.linspace(0, (t_e[-1] - t_e[pe]), 200)
        ax[1].plot(tt, np.exp(-tt / tau_true_ms), color="tab:green", lw=1, ls=":",
                   label=f"exp(-t/tau_s_nominal), {tau_true_ms*1e3:.0f}us")
        if np.isfinite(tau_eff_ms):
            ax[1].plot(tt, np.exp(-tt / tau_eff_ms), color="tab:orange", lw=1.2,
                       label=f"exp(-t/tau_eff_empirical), {tau_eff_ms*1e3:.0f}us")
        ax[1].set(title=f"PBF tail | fit tau(nu0)={tau_e_nu0*1e3:.0f}us  "
                        f"empirical={tau_eff_ms*1e3:.0f}us  nominal tau_s={tau_true_ms*1e3:.0f}us",
                  xlabel="t - t_peak [ms]", ylabel="norm. flux", yscale="log",
                  ylim=(1e-3, 1.5), xlim=(-0.1, min(2.0, t_e[-1] - t_e[pe])))
        ax[1].legend(fontsize=8)
        fig.suptitle("Stage 4 - ensemble-averaged clean recovery", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "ensemble_recovery.png"), dpi=130)
        plt.close(fig)

    # ----- figure manifest (drives the figure-review gate) -------------------
    figs = [
        ("sim_dynamic_spectrum.png",
         "Simulated burst: waterfall shows a pulse + decaying tail; band-integrated "
         "profile peaks then decays; time-integrated spectrum should be NEAR-SMOOTH "
         "(low modulation index, many scintles/channel); spectral ACF should decorrelate "
         "near nu_s_host (~sub-kHz to kHz), i.e. a narrow spike, NOT a band-wide envelope."),
        ("fit_data_model_residual.png",
         "data | best-fit M2 model | residual; model tracks the gross pulse, residual holds "
         "leftover scintillation, no gross spatial misfit; colour ranges comparable."),
        ("fit_corner.png",
         "Posterior corner for c0,t0,gamma,tau_1ghz in physical units; well-formed contours; "
         "tau column finite and bounded."),
        ("fit_chains.png",
         "Walker traces per parameter + log-prob vs step; chains stationary/mixed after burn-in "
         "(no permanent drift or stuck walkers)."),
        ("recovered_vs_true.png",
         "tau(nu0) posterior histogram with the injected tau_s as a vertical line; posterior "
         "should be a sensible distribution (ideally near the line)."),
    ]
    if args.ensemble > 0:
        figs.append(("ensemble_recovery.png",
                     "Left: ensemble-mean waterfall with scintillation visibly suppressed and a "
                     "clear pulse. Right (log y): normalised PBF tail with fit, empirical-efold, "
                     "and nominal-tau_s exponentials; fit and empirical curves should track the "
                     "data tail, nominal tau_s should be steeper."))
    write_manifest(args.out, figs)

    # ----- summary -----------------------------------------------------------
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n=== VALIDATION SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nFigures + summary.json written to: {args.out}")
    return summary


if __name__ == "__main__":
    main()
