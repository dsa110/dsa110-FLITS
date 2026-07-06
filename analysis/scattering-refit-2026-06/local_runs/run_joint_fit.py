#!/usr/bin/env python
"""Driver: joint CHIME+DSA scattering fit for one burst.

Reads the two single-band HPCC run-configs (<b>_chime_run.yaml, <b>_dsa_run.yaml),
rebuilds each band's preprocessed FRBModel + data-driven init exactly as the
single-band pipeline does (same freq-orientation flip, trim, noise estimate),
then runs the shared-(tau,alpha) joint sampler from burstfit_joint.

Writes <RUNS>/data/joint/<b>_joint_fit.json with the shared alpha / tau_1ghz
posteriors + per-band params, for direct comparison against the single-band
tau_1ghz rails.

  python run_joint_fit.py <burst> [nlive] [nproc]
"""

import argparse
import json
import os
import sys

REPO = os.environ.get("FLITS_REPO", "/home/jfaber/flits/dsa110-FLITS")
RUNS = os.environ.get("FLITS_RUNS", "/central/scratch/jfaber/flits-runs")
sys.path.insert(0, f"{REPO}/scattering")  # so `scat_analysis` imports

import numpy as np
import yaml
from scat_analysis.burstfit import FRBParams
from scat_analysis.burstfit_init import data_driven_initial_guess
from scat_analysis.burstfit_joint import fit_joint_scattering
from scat_analysis.config_utils import load_telescope_block
from scat_analysis.pipeline.io import BurstDataset
from scat_analysis.pipeline.optimization import refine_initial_guess_mle


def prepare(cfg_path, name, outdir):
    """Rebuild a single band's FRBModel + data-driven init from its run-config."""
    cfg = yaml.safe_load(open(cfg_path))
    tel = load_telescope_block(cfg["telcfg_path"], cfg["telescope"])
    ds = BurstDataset(
        cfg["path"],
        outdir,
        name=name,
        telescope=tel,
        f_factor=int(cfg["f_factor"]),
        t_factor=int(cfg["t_factor"]),
        outer_trim=float(cfg.get("outer_trim", 0.15)),
        onpulse_crop=os.environ.get("FLITS_ONPULSE_CROP", "1") == "1",
        onpulse_pad_factor=float(os.environ.get("FLITS_ONPULSE_PAD", "0.5")),
        onpulse_thresh=float(os.environ.get("FLITS_ONPULSE_THRESH", "3.0")),
    )
    model = ds.model
    dm_init = float(cfg.get("dm_init", 0.0))
    model.dm_init = dm_init
    init = data_driven_initial_guess(
        data=model.data,
        freq=model.freq,
        time=model.time,
        dm=dm_init,
        verbose=False,
    ).params
    init = refine_initial_guess_mle(model, init)
    return model, init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("burst")
    ap.add_argument("nlive", nargs="?", type=int, default=600)
    ap.add_argument("nproc", nargs="?", type=int, default=8)
    # beta is the sampled parameter (ADR-0006); --beta-lo/--beta-hi drive the
    # prior directly. The legacy --alpha-lo/--alpha-hi pair is a deprecated
    # alias mapped through beta_bounds_from_alpha_bounds inside the library
    # (only alpha >= 4 is reachable on the thin-screen branch, so the old
    # default (2.0, 6.0) silently truncates to beta in [3, 4]).
    ap.add_argument("--beta-lo", dest="beta_lo", type=float, default=None)
    ap.add_argument("--beta-hi", dest="beta_hi", type=float, default=None)
    ap.add_argument("--alpha-lo", type=float, default=2.0)
    ap.add_argument("--alpha-hi", type=float, default=6.0)
    ap.add_argument(
        "--marginalize-gain",
        action="store_true",
        help="per-channel gain marginalized (absorbs scintillation); 8-dim fit",
    )
    ap.add_argument(
        "--marginalize-gain-gp",
        "--scint",
        dest="marginalize_gain_gp",
        action="store_true",
        help="gain marginalized with a Lorentzian scintillation GP prior; "
        "samples Delta_nu_d per band (10-dim fit)",
    )
    ap.add_argument(
        "--mu-degree",
        type=int,
        default=1,
        help="polynomial degree of the smooth GLS spectral envelope (GP path)",
    )
    ap.add_argument(
        "--components-C",
        dest="components_C",
        type=int,
        default=1,
        help="number of temporal components (sub-pulses) in the CHIME band",
    )
    ap.add_argument(
        "--components-D",
        dest="components_D",
        type=int,
        default=1,
        help="number of temporal components (sub-pulses) in the DSA band",
    )
    ap.add_argument(
        "--force-multi",
        dest="force_multi",
        action="store_true",
        help="run the multi-component likelihood even at C1D1, so its lnZ "
        "is normalization-matched to C2/D2 runs (model-selection baseline)",
    )
    ap.add_argument(
        "--shared-zeta",
        dest="shared_zeta",
        action="store_true",
        help="ONE frequency-evolving intrinsic width zeta(nu)=zeta_1ghz*nu^x_zeta "
        "shared across both bands (gain-marginal, 8-dim) instead of per-band zeta; "
        "gives a single coherent burst across the full band",
    )
    ap.add_argument(
        "--gain-s2",
        dest="gain_s2",
        type=float,
        default=None,
        help="fix the gain-prior variance s2 instead of ML-profiling it per fit. "
        "Profiled s2 (default) gives a profile/empirical-Bayes Z; for a CLEAN "
        "cross-N Bayes factor (component model selection) fix s2 to one value "
        "across the whole ladder. Output is tagged _s2-<val> so it does not "
        "overwrite the profiled run.",
    )
    # The per-band --pbf-C/--pbf-D/--beta-C/--beta-D knobs are gone: ADR-0006
    # removed the FLITS_PBF selector and model.pbf/.pbf_beta have zero kernel
    # consumers -- the PBF family is driven by the sampled beta itself
    # (power-law tail below 4, exponential member at the beta=4 limit).
    a = ap.parse_args()
    if (a.beta_lo is None) != (a.beta_hi is None):
        ap.error("--beta-lo and --beta-hi must be given together")
    beta_bounds = (a.beta_lo, a.beta_hi) if a.beta_lo is not None else None
    multi = a.components_C > 1 or a.components_D > 1 or a.force_multi

    cfg_dir = f"{RUNS}/configs"
    out_dir = f"{RUNS}/data/joint"
    os.makedirs(out_dir, exist_ok=True)

    cC = f"{cfg_dir}/{a.burst}_chime_run.yaml"
    cD = f"{cfg_dir}/{a.burst}_dsa_run.yaml"
    for c in (cC, cD):
        if not os.path.exists(c):
            sys.exit(f"missing config: {c}")

    print(f"[{a.burst}] preparing CHIME + DSA models ...", flush=True)
    model_C, init_C = prepare(cC, f"{a.burst}_chime", out_dir)
    model_D, init_D = prepare(cD, f"{a.burst}_dsa", out_dir)
    print(
        f"[{a.burst}] CHIME init: tau={init_C.tau_1ghz:.3g} a={init_C.alpha:.2g} | "
        f"DSA init: tau={init_D.tau_1ghz:.3g} a={init_D.alpha:.2g}",
        flush=True,
    )

    res = fit_joint_scattering(
        model_C=model_C,
        init_C=init_C,
        model_D=model_D,
        init_D=init_D,
        beta_bounds=beta_bounds,
        alpha_bounds=None if beta_bounds else (a.alpha_lo, a.alpha_hi),
        nlive=a.nlive,
        nproc=a.nproc,
        marginalize_gain=a.marginalize_gain,
        marginalize_gain_gp=a.marginalize_gain_gp,
        shared_zeta=a.shared_zeta,
        mu_degree=a.mu_degree,
        components_C=a.components_C,
        components_D=a.components_D,
        force_multi=a.force_multi,
        gain_s2=a.gain_s2,
    )

    pct = res["percentiles"]
    names = res["param_names"]

    def med(n):  # median (+err_plus/-err_minus)
        d = pct[n]
        return d["median"], d["err_minus"], d["err_plus"]

    a_m, a_lo, a_hi = med("alpha")
    b_m, b_lo, b_hi = med("beta")
    t_m, t_lo, t_hi = med("tau_1ghz")
    summary = {
        "burst": a.burst,
        "marginalize_gain": bool(a.marginalize_gain),
        "marginalize_gain_gp": bool(a.marginalize_gain_gp),
        "shared_zeta": bool(a.shared_zeta),
        # beta first: gate_one's beta-native path keys off "beta" in fit and
        # rails against beta_bounds; alpha is the derived report-only value.
        "beta": {"median": b_m, "err_minus": b_lo, "err_plus": b_hi},
        "beta_bounds": list(res["beta_bounds"]),
        "alpha": {"median": a_m, "err_minus": a_lo, "err_plus": a_hi},
        "tau_1ghz": {"median": t_m, "err_minus": t_lo, "err_plus": t_hi},
        "log_evidence": res["log_evidence"],
        "log_evidence_err": res["log_evidence_err"],
        "alpha_bounds": list(res["alpha_bounds"]),
        "components_C": a.components_C,
        "components_D": a.components_D,
        "gain_s2": a.gain_s2,
        "percentiles": {n: pct[n] for n in names},
        "ncall": res["ncall"],
    }

    # Recover the per-channel gain spectra at the medians (scintillation probe).
    gain_C = gain_D = None
    scint = {}
    if (a.marginalize_gain or a.marginalize_gain_gp or a.shared_zeta) and not multi:
        p = {k: v["median"] for k, v in pct.items()}
        if a.shared_zeta:
            # ONE width law -> per-band zeta is the array zeta_1ghz*nu^x_zeta on
            # that band's full channel axis (matches _JointLogLikelihoodGainSharedZeta).
            zc = p["zeta_1ghz"] * np.asarray(model_C.freq, float) ** p["x_zeta"]
            zd = p["zeta_1ghz"] * np.asarray(model_D.freq, float) ** p["x_zeta"]
        else:
            zc, zd = p["zeta_C"], p["zeta_D"]
        # beta, not alpha: FRBParams is beta-native post-ADR-0006 (alpha is a
        # derived property; the alpha= kwarg TypeErrors). beta -> alpha is
        # monotone, so the beta median IS the alpha median's preimage.
        pC = FRBParams(
            c0=1.0,
            t0=p["t0_C"],
            gamma=0.0,
            zeta=zc,
            tau_1ghz=t_m,
            beta=p["beta"],
            delta_dm=p["delta_dm_C"],
        )
        pD = FRBParams(
            c0=1.0,
            t0=p["t0_D"],
            gamma=0.0,
            zeta=zd,
            tau_1ghz=t_m,
            beta=p["beta"],
            delta_dm=p["delta_dm_D"],
        )
        gain_C = model_C.gain_spectrum(pC, "M3")
        gain_D = model_D.gain_spectrum(pD, "M3")
        summary["gain_recovered"] = True

    if a.marginalize_gain_gp:
        import numpy as _np

        # Per-band Delta_nu_d posterior medians + channel width + unresolved flag
        # + modulation-index sub-resolution estimate (from the GLS residual gains).
        def _chan_w_MHz(freq_GHz):
            return float(_np.median(_np.abs(_np.diff(_np.asarray(freq_GHz))))) * 1e3

        dnu_C = med("Delta_nu_d_C")
        dnu_D = med("Delta_nu_d_D")
        cw_C, cw_D = _chan_w_MHz(model_C.freq), _chan_w_MHz(model_D.freq)
        sumC = model_C.scint_gain_summary(pC, "M3", delta_nu_d_MHz=dnu_C[0], mu_degree=a.mu_degree)
        sumD = model_D.scint_gain_summary(pD, "M3", delta_nu_d_MHz=dnu_D[0], mu_degree=a.mu_degree)
        scint = {
            "Delta_nu_d_C": {
                "median": dnu_C[0],
                "err_minus": dnu_C[1],
                "err_plus": dnu_C[2],
                "chan_width_MHz": cw_C,
                "unresolved": bool(dnu_C[0] < cw_C),
                "modulation_index": sumC["modulation_index"],
                "modindex_dnu_d_MHz": float(sumC["modulation_index"] ** 2 * cw_C),
                "sigma_g2": sumC["sigma_g2"],
            },
            "Delta_nu_d_D": {
                "median": dnu_D[0],
                "err_minus": dnu_D[1],
                "err_plus": dnu_D[2],
                "chan_width_MHz": cw_D,
                "unresolved": bool(dnu_D[0] < cw_D),
                "modulation_index": sumD["modulation_index"],
                "modindex_dnu_d_MHz": float(sumD["modulation_index"] ** 2 * cw_D),
                "sigma_g2": sumD["sigma_g2"],
            },
        }
        summary["scint"] = scint

    if multi:
        tag = f"_C{a.components_C}D{a.components_D}"
    elif a.shared_zeta:
        tag = "_sharedzeta"  # keep the per-band baseline json/npz for comparison
    else:
        tag = ""
    if a.gain_s2 is not None:  # fixed-s2 ladder: never clobber the profiled run
        tag += f"_s2-{a.gain_s2:g}"
    # No _pbf-* tag: beta drives the PBF family (ADR-0006), so beta-campaign
    # outputs are suffix-free and cannot collide with the legacy alpha-era
    # *_pbf-exp-exp.json artifacts.
    out = f"{out_dir}/{a.burst}_joint_fit{tag}.json"
    json.dump(summary, open(out, "w"), indent=2)

    # Persist the full weighted posterior + recovered gains + per-band freq axes so
    # corner plots / tau(nu) ladders / scintillation (Delta-nu_d) analysis can be
    # built without re-running the sampler.
    npz = dict(
        samples=res["samples"],
        weights=res["weights"],
        param_names=np.array(names, dtype=object),
        alpha_bounds=np.array(res["alpha_bounds"], dtype=float),
        freq_C=model_C.freq,
        freq_D=model_D.freq,
    )
    if gain_C is not None:
        npz["gain_C"] = gain_C
        npz["gain_D"] = gain_D
    if a.marginalize_gain_gp:
        # Posterior Delta_nu_d columns (so scint_acf.py can cross-check the fit's
        # Delta_nu_d against its own ACF estimate) + GLS mean/residual per band.
        ci = list(names).index("Delta_nu_d_C")
        di = list(names).index("Delta_nu_d_D")
        npz["Delta_nu_d_C_samples"] = res["samples"][:, ci]
        npz["Delta_nu_d_D_samples"] = res["samples"][:, di]
        npz["scint_freq_C_MHz"] = sumC["freq_MHz"]
        npz["scint_ahat_C"] = sumC["ahat"]
        npz["scint_mu_C"] = sumC["mu"]
        npz["scint_freq_D_MHz"] = sumD["freq_MHz"]
        npz["scint_ahat_D"] = sumD["ahat"]
        npz["scint_mu_D"] = sumD["mu"]
    np.savez_compressed(f"{out_dir}/{a.burst}_joint_samples{tag}.npz", **npz)

    # Rail check on the SAMPLED parameter: ADR-0004 flags a median within ~3
    # sigma of either beta prior bound (a beta=4 rail is the exponential limit
    # / alpha=4-as-limit case and the ADR-0007 re-open trigger).
    bb_lo, bb_hi = res["beta_bounds"]
    edge = (
        " [BETA AT PRIOR EDGE]" if (b_m - 3.0 * b_lo <= bb_lo or b_m + 3.0 * b_hi >= bb_hi) else ""
    )
    print(
        f"\n[{a.burst}] JOINT  beta = {b_m:.3f} (+{b_hi:.3f}/-{b_lo:.3f}){edge}"
        f"   alpha = {a_m:.2f} (+{a_hi:.2f}/-{a_lo:.2f})"
        f"   tau_1GHz = {t_m:.3g} (+{t_hi:.2g}/-{t_lo:.2g}) ms"
        f"   lnZ = {res['log_evidence']:.1f}",
        flush=True,
    )
    if a.marginalize_gain_gp:
        for b, s in scint.items():
            flag = "UNRESOLVED (upper limit)" if s["unresolved"] else "RESOLVED"
            print(
                f"[{a.burst}] {b} = {s['median']:.3g} MHz "
                f"(chan {s['chan_width_MHz']:.3g} MHz) [{flag}]  "
                f"m={s['modulation_index']:.3g} -> dnu_d~{s['modindex_dnu_d_MHz']:.3g} MHz",
                flush=True,
            )
    print(f"[{a.burst}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
