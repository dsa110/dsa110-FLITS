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
from pathlib import Path

REPO = os.environ.get("FLITS_REPO", "/home/jfaber/flits/dsa110-FLITS")
RUNS = os.environ.get("FLITS_RUNS", "/central/scratch/jfaber/flits-runs")
sys.path.insert(0, f"{REPO}/scattering")  # so `scat_analysis` imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so joint_tf_prep imports

import joint_tf_prep
import numpy as np
import scat_analysis.burstfit_joint as burstfit_joint_module
import scat_analysis.controlled_run as controlled_run_module
import scat_analysis.joint_fit_diagnostics as diagnostics_module
import scat_analysis.joint_model_grid as model_grid_module
import yaml
from scat_analysis.burstfit import FRBParams
from scat_analysis.burstfit_init import data_driven_initial_guess
from scat_analysis.burstfit_joint import fit_joint_scattering
from scat_analysis.config_utils import load_telescope_block
from scat_analysis.controlled_run import (
    DEPRECATED_ZACH_GUARDS,
    ControlledRunError,
    controlled_python_argv,
    finalize_receipt,
    identity_sha256,
    preflight,
    processing_environment_identity,
    reverify_preflight,
)
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
    )
    model = ds.model
    dm_init = float(cfg.get("dm_init", 0.0))
    model.dm_init = dm_init
    return model, _init_for(model, dm_init)


def _init_for(model, dm_init):
    """Data-driven initial guess + MLE refine for a prepared band model."""
    model.dm_init = dm_init
    init = data_driven_initial_guess(
        data=model.data,
        freq=model.freq,
        time=model.time,
        dm=dm_init,
        verbose=False,
    ).params
    return refine_initial_guess_mle(model, init)


def prepare_joint(cC, cD, burst, outdir):
    """CHIME + DSA prepared together with the S/N-driven resolution + robust common
    window (joint_tf_prep.prepare_pair). Returns (model_C, init_C, model_D, init_D)
    and logs the chosen per-band resolution/window. Set FLITS_JOINT_AUTO_TF=0 to
    fall back to the config's fixed f_factor/t_factor + legacy per-band crop."""
    dm_C = float(yaml.safe_load(open(cC)).get("dm_init", 0.0))
    dm_D = float(yaml.safe_load(open(cD)).get("dm_init", 0.0))
    if joint_tf_prep._env_auto():
        (model_C, mkC), (model_D, mkD) = joint_tf_prep.prepare_pair(cC, cD, burst, outdir)
        print(f"[{burst}] AUTO-TF CHIME: {mkC.caption()}", flush=True)
        print(f"[{burst}] AUTO-TF DSA  : {mkD.caption()}", flush=True)
        marginal = [band for band, meta in (("CHIME", mkC), ("DSA", mkD)) if not meta.snr_qualified]
        if marginal:
            raise SystemExit(
                f"[{burst}] AUTO-TF MARGINAL: {', '.join(marginal)} did not clear the "
                f"S/N target; refusing to publish a joint fit"
            )
        return model_C, _init_for(model_C, dm_C), model_D, _init_for(model_D, dm_D)
    model_C, init_C = prepare(cC, f"{burst}_chime", outdir)
    model_D, init_D = prepare(cD, f"{burst}_dsa", outdir)
    return model_C, init_C, model_D, init_D


def main(*, controlled: bool = False):
    ap = argparse.ArgumentParser()
    ap.add_argument("burst")
    ap.add_argument("nlive", nargs="?", type=int, default=600)
    ap.add_argument("nproc", nargs="?", type=int, default=8)
    ap.add_argument(
        "--seed",
        type=int,
        required=controlled,
        help="dynesty random seed; required and recorded by the controlled entrypoint",
    )
    ap.add_argument(
        "--contract",
        type=Path,
        required=controlled,
        help="frozen source/input/configuration contract to verify before sampling",
    )
    ap.add_argument(
        "--receipt",
        type=Path,
        required=controlled,
        help="new preflight/output receipt path; existing files are rejected",
    )
    ap.add_argument(
        "--dlogz",
        type=float,
        default=0.5,
        help="nested-sampling stopping threshold in log-evidence units",
    )
    ap.add_argument(
        "--resolved-identity-output",
        type=Path,
        help="write the resolved pre-sampler identity for contract preparation",
    )
    # beta is the sampled parameter (ADR-0006); --beta-lo/--beta-hi drive the
    # prior directly. --alpha-lo/--alpha-hi is the deprecated alias (mapped via
    # beta_bounds_from_alpha_bounds; only alpha >= 4 is reachable thin-screen).
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
        "--gain-s2",
        dest="gain_s2",
        type=int,
        default=None,
        help="fix the per-channel gain-prior variance s2 instead of profiling it. "
        "REQUIRED for a valid cross-N Bayes factor (ADR-0003): the profiled-s2 lnZ "
        "is not comparable across component count, so component-count model selection "
        "must hold s2 fixed (and use the multi likelihood throughout, which this also "
        "enables). Output is tagged _s2-<v>. Run a small grid (e.g. 10, 100) and accept "
        "the extra component only if ΔlnZ(N+1 vs N) > 5 consistently across s2.",
    )
    ap.add_argument(
        "--fixed-delta-dm-C",
        type=float,
        default=None,
        help="hold the CHIME residual DM fixed at this value instead of sampling it",
    )
    ap.add_argument(
        "--fixed-delta-dm-D",
        type=float,
        default=None,
        help="hold the DSA residual DM fixed at this value instead of sampling it",
    )
    # Shared zeta is the DEFAULT: ONE frequency-evolving intrinsic width
    # zeta(nu)=zeta_1ghz*nu^x_zeta across both bands models a single coherent
    # burst over the full CHIME+DSA band, which is the physically motivated
    # baseline (the writeup concluded per-band zeta over-fits intrinsic width).
    # Pass --per-band-zeta to give each band its own zeta (the old default).
    ap.add_argument(
        "--per-band-zeta",
        dest="shared_zeta",
        action="store_false",
        default=True,
        help="give CHIME and DSA each their own intrinsic width zeta instead of "
        "the default single shared zeta(nu) across both bands",
    )
    # The per-band --pbf-C/--pbf-D/--beta-C/--beta-D knobs are gone: ADR-0006
    # removed the FLITS_PBF selector and model.pbf/.pbf_beta have zero kernel
    # consumers -- the sampled beta drives the PBF family.
    a = ap.parse_args()
    if (a.beta_lo is None) != (a.beta_hi is None):
        ap.error("--beta-lo and --beta-hi must be given together")
    beta_bounds = (a.beta_lo, a.beta_hi) if a.beta_lo is not None else None
    # gain_s2 fixed also forces the multi likelihood (burstfit_joint threads it there),
    # so a fixed-s2 C1D1 is normalization-matched to the C2 rungs it is compared against.
    multi = a.components_C > 1 or a.components_D > 1 or a.force_multi or a.gain_s2 is not None
    if controlled and a.gain_s2 is None:
        ap.error("controlled component-count fits require a fixed --gain-s2")
    if controlled and (a.marginalize_gain or a.marginalize_gain_gp):
        ap.error("controlled component-count fits do not accept alternate gain likelihoods")
    if controlled and (a.fixed_delta_dm_C is not None or a.fixed_delta_dm_D is not None):
        ap.error("controlled component-count fits do not accept fixed residual dispersion measure")
    effective_shared_zeta = bool(a.shared_zeta) and not multi

    cfg_dir = f"{RUNS}/configs"
    out_dir = f"{RUNS}/data/joint"
    os.makedirs(out_dir, exist_ok=True)
    if multi:
        tag = f"_C{a.components_C}D{a.components_D}"
    elif not a.shared_zeta:
        tag = "_perbandzeta"
    else:
        tag = ""
    if a.gain_s2 is not None:
        tag += f"_s2-{a.gain_s2}"
    out = Path(out_dir) / f"{a.burst}_joint_fit{tag}.json"
    samples_out = Path(out_dir) / f"{a.burst}_joint_samples{tag}.npz"
    model_grid_out = Path(out_dir) / f"{a.burst}_jointmodel{tag}.npz"
    diagnostics_out = Path(out_dir) / f"{a.burst}_joint_diagnostics{tag}.json"
    panel_out = Path(out_dir) / f"{a.burst}_joint_panel{tag}.svg"
    if controlled:
        for output_path in (
            out,
            samples_out,
            model_grid_out,
            diagnostics_out,
            panel_out,
        ):
            if output_path.exists():
                ap.error(f"refusing to overwrite controlled output: {output_path}")

    cC = f"{cfg_dir}/{a.burst}_chime_run.yaml"
    cD = f"{cfg_dir}/{a.burst}_dsa_run.yaml"
    for c in (cC, cD):
        if not os.path.exists(c):
            sys.exit(f"missing config: {c}")

    receipt = None
    processing_environment = processing_environment_identity(Path(REPO), Path(RUNS))
    invocation = {
        "burst": a.burst,
        "nlive": a.nlive,
        "nproc": a.nproc,
        "dlogz": a.dlogz,
        "sample": "rwalk",
        "seed": a.seed,
        "beta_bounds": list(beta_bounds) if beta_bounds is not None else None,
        "alpha_bounds": None if beta_bounds else [a.alpha_lo, a.alpha_hi],
        "marginalize_gain": bool(a.marginalize_gain),
        "marginalize_gain_gp": bool(a.marginalize_gain_gp),
        "mu_degree": a.mu_degree,
        "components_C": a.components_C,
        "components_D": a.components_D,
        "force_multi": bool(a.force_multi),
        "gain_s2": a.gain_s2,
        "fixed_delta_dm_C": a.fixed_delta_dm_C,
        "fixed_delta_dm_D": a.fixed_delta_dm_D,
        "shared_zeta": effective_shared_zeta,
    }
    if controlled:
        configs = {
            "chime_config": Path(cC).resolve(),
            "dsa_config": Path(cD).resolve(),
        }
        loaded_configs = {
            band: yaml.safe_load(path.read_text(encoding="utf-8")) for band, path in configs.items()
        }
        resolved_files = {
            **configs,
            "chime_input": Path(loaded_configs["chime_config"]["path"]).expanduser().resolve(),
            "dsa_input": Path(loaded_configs["dsa_config"]["path"]).expanduser().resolve(),
            "chime_telescope_config": Path(loaded_configs["chime_config"]["telcfg_path"])
            .expanduser()
            .resolve(),
            "dsa_telescope_config": Path(loaded_configs["dsa_config"]["telcfg_path"])
            .expanduser()
            .resolve(),
            "environment_lock": (Path(REPO) / "uv.lock").resolve(),
            "controlled_entrypoint": Path(sys.argv[0]).resolve(),
            "fit_driver": Path(__file__).resolve(),
            "joint_tf_prep_source": Path(joint_tf_prep.__file__).resolve(),
            "burstfit_joint_source": Path(burstfit_joint_module.__file__).resolve(),
            "controlled_run_source": Path(controlled_run_module.__file__).resolve(),
            "model_grid_source": Path(model_grid_module.__file__).resolve(),
            "diagnostic_source": Path(diagnostics_module.__file__).resolve(),
        }
        if a.receipt.exists():
            ap.error(f"receipt already exists: {a.receipt}")
        receipt = preflight(
            contract_path=a.contract,
            repo=Path(REPO),
            invocation=invocation,
            resolved_files=resolved_files,
            argv=controlled_python_argv(sys.argv),
            cwd=Path.cwd(),
            environment_variables=processing_environment,
        )
        a.receipt.parent.mkdir(parents=True, exist_ok=True)
        a.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[{a.burst}] preparing CHIME + DSA models ...", flush=True)
    model_C, init_C, model_D, init_D = prepare_joint(cC, cD, a.burst, out_dir)
    print(
        f"[{a.burst}] CHIME init: tau={init_C.tau_1ghz:.3g} a={init_C.alpha:.2g} | "
        f"DSA init: tau={init_D.tau_1ghz:.3g} a={init_D.alpha:.2g}",
        flush=True,
    )

    resolved_identity_callback = None
    if controlled:
        contract = json.loads(a.contract.read_text(encoding="utf-8"))

        def verify_resolved_identity(identity):
            digest = identity_sha256(identity)
            if a.resolved_identity_output is not None:
                a.resolved_identity_output.parent.mkdir(parents=True, exist_ok=True)
                a.resolved_identity_output.write_text(
                    json.dumps(
                        {"identity": identity, "identity_sha256": digest},
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            if digest != contract.get("resolved_fit_identity_sha256"):
                raise ControlledRunError(
                    "resolved likelihood, priors, or fitted support do not match contract"
                )
            receipt["resolved_fit_identity"] = identity
            receipt["resolved_fit_identity_sha256"] = digest
            reverify_preflight(receipt)
            receipt["post_preparation_reverification_passed"] = True
            a.receipt.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        resolved_identity_callback = verify_resolved_identity

    res = fit_joint_scattering(
        model_C=model_C,
        init_C=init_C,
        model_D=model_D,
        init_D=init_D,
        beta_bounds=beta_bounds,
        alpha_bounds=None if beta_bounds else (a.alpha_lo, a.alpha_hi),
        nlive=a.nlive,
        dlogz=a.dlogz,
        nproc=a.nproc,
        seed=a.seed,
        marginalize_gain=a.marginalize_gain,
        marginalize_gain_gp=a.marginalize_gain_gp,
        shared_zeta=effective_shared_zeta,
        mu_degree=a.mu_degree,
        components_C=a.components_C,
        components_D=a.components_D,
        force_multi=a.force_multi,
        gain_s2=a.gain_s2,
        fixed_delta_dm_C=a.fixed_delta_dm_C,
        fixed_delta_dm_D=a.fixed_delta_dm_D,
        resolved_identity_callback=resolved_identity_callback,
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
        "shared_zeta": effective_shared_zeta,
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
        # None => s2 was profiled (lnZ NOT cross-N comparable; ADR-0003). A float =>
        # fixed s2, so this lnZ IS a valid cross-N rung at that s2.
        "gain_s2": a.gain_s2,
        "gain_model": "proper_gaussian" if multi else "ordinary_least_squares",
        "force_multi": bool(a.force_multi),
        "mu_degree": a.mu_degree,
        "nlive": a.nlive,
        "nproc": a.nproc,
        "fixed_parameters": res.get("fixed_parameters", {}),
        "percentiles": pct,
        "ncall": res["ncall"],
        "seed": res["seed"],
        "dlogz": a.dlogz,
        "sample": "rwalk",
    }
    if controlled:
        summary.update(
            {
                "controlled_contract_sha256": receipt["contract"]["sha256"],
                "resolved_fit_identity_sha256": receipt["resolved_fit_identity_sha256"],
                "source_revision": receipt["source"]["revision"],
                "required_post_fit_guards": DEPRECATED_ZACH_GUARDS,
            }
        )

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
        # derived property; the alpha= kwarg TypeErrors).
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

    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Persist the full weighted posterior + recovered gains + per-band freq axes so
    # corner plots / tau(nu) ladders / scintillation (Delta-nu_d) analysis can be
    # built without re-running the sampler.
    npz = dict(
        samples=res["samples"],
        weights=res["weights"],
        log_weight=res["log_weight"],
        log_evidence_history=res["log_evidence_history"],
        log_evidence_error_history=res["log_evidence_error_history"],
        ncall_history=res["ncall_history"],
        param_names=np.array(names, dtype=object),
        beta_bounds=np.array(res["beta_bounds"], dtype=float),
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
    np.savez_compressed(samples_out, **npz)
    if controlled:
        model_grid = model_grid_module.build_model_grid_arrays(model_C, model_D, summary)
        np.savez_compressed(model_grid_out, **model_grid)
        diagnostics = diagnostics_module.build_diagnostics(
            summary,
            model_grid,
            samples=res["samples"],
            weights=res["weights"],
            param_names=res["param_names"],
        )
        diagnostics_module.write_diagnostics(diagnostics_out, diagnostics)
        diagnostics_module.render_fit_panel(model_grid, panel_out)
        finalize_receipt(
            a.receipt,
            {
                "fit_summary": out,
                "weighted_samples": samples_out,
                "model_grid": model_grid_out,
                "diagnostics": diagnostics_out,
                "panel": panel_out,
            },
        )

    # Rail check on the SAMPLED parameter (ADR-0004: median within ~3 sigma of
    # a beta prior bound; the beta=4 rail is the ADR-0007 re-open trigger).
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
