"""
Core orchestration logic for the BurstFit pipeline.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import inspect
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..burstfit import (
    FRBFitter,
    FRBParams,
    build_priors,
    goodness_of_fit,
)
from ..burstfit_modelselect import fit_models_bic
from ..burstfit_nested import fit_models_evidence
from ..config_utils import load_telescope_block
from ..pool_utils import build_pool
from flits.utils.reporting import print_fit_summary

from .io import BurstDataset
from .optimization import refine_initial_guess_mle, auto_burn_thin
from .diagnostics import BurstDiagnostics, create_fit_summary_plot, create_four_panel_plot

log = logging.getLogger(__name__)


def build_safe_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the JSON-serializable fit-results dict, including posterior errors.

    Crucially this persists per-parameter posterior percentiles (median, 16/84,
    err_minus/err_plus) for both the best model and every scanned model -- the
    nested sampler computes them but the earlier save path discarded them, so no
    fit (even a good one) yielded tau +/- sigma. ``best_params_percentiles`` is
    the headline: it carries tau_1ghz with its uncertainty. Results without a
    ``percentiles`` attribute (e.g. the emcee/BIC branch) degrade to None.
    """
    best_params = results.get("best_params")
    if dataclasses.is_dataclass(best_params) and not isinstance(best_params, type):
        best_params = dataclasses.asdict(best_params)
    elif hasattr(best_params, "__dict__"):
        best_params = dict(best_params.__dict__)

    all_results = results.get("all_results", {}) or {}
    best_key = results.get("best_key")

    all_res_summary: Dict[str, Any] = {}
    for k, v in all_results.items():
        if k == "bayes_factors":
            continue
        all_res_summary[k] = {
            "log_evidence": getattr(v, "log_evidence", None),
            "log_evidence_err": getattr(v, "log_evidence_err", None),
            "percentiles": getattr(v, "percentiles", None),
        }

    best_res = all_results.get(best_key) if isinstance(all_results, dict) else None
    best_percentiles = getattr(best_res, "percentiles", None)

    return {
        "best_model": best_key,
        "best_params": best_params,
        "best_params_percentiles": best_percentiles,
        "param_names": results.get("param_names"),
        "goodness_of_fit": results.get("goodness_of_fit"),
        "dm_init": results.get("dm_init"),
        "convergence": results.get("loop_stats"),
        "all_results": all_res_summary,
    }


class BurstPipeline:
    """Main orchestrator for the fitting pipeline."""

    def __init__(
        self,
        inpath: str | Path,
        outpath: str | Path,
        name: str,
        *,
        dm_init: float = 0.0,
        **kwargs,
    ):
        """
        Initializes the pipeline.

        Args:
            name: FRB name
            inpath: Path to the input .npy data file.
            outpath: Path to the output files.
            dm_init: Initial dispersion measure for the data.
            **kwargs: Keyword arguments for pipeline configuration. These are
                      intelligently split between BurstDataset and the pipeline.
        """
        self.inpath = inpath
        self.outpath = Path(outpath)
        self.outpath.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.dm_init = dm_init

        # --- FIX: Intelligently separate kwargs for different components ---

        # Get the names of all valid arguments for the BurstDataset constructor
        dataset_params = inspect.signature(BurstDataset).parameters
        dataset_arg_names = list(dataset_params.keys())

        # Create a dictionary with only the kwargs that BurstDataset accepts
        self.dataset_kwargs = {
            k: v for k, v in kwargs.items() if k in dataset_arg_names
        }

        # Store the remaining kwargs for the pipeline itself (e.g., 'steps')
        self.pipeline_kwargs = {
            k: v for k, v in kwargs.items() if k not in dataset_arg_names
        }

        # Create the multiprocessing pool
        self.pool = build_pool(
            self.pipeline_kwargs.get("nproc"),
            auto_ok=self.pipeline_kwargs.get("yes", False),
        )

        # Optional seed init-guess
        self.seed_single: FRBParams | None = None
        self.seed_multi: dict[str, float] | None = None
        init_guess_path = self.pipeline_kwargs.get(
            "init_guess"
        ) or self.pipeline_kwargs.get("init_guess_path")
        if init_guess_path:
            try:
                with (
                    Path(init_guess_path).expanduser().open("r", encoding="utf-8") as fh
                ):
                    seed = json.load(fh)
                mk = seed.get("model_key", "M3")
                if mk == "M3":
                    self.seed_single = FRBParams(
                        c0=float(seed.get("c0", 1.0)),
                        t0=float(seed.get("t0", 0.0)),
                        gamma=float(seed.get("gamma", -1.6)),
                        zeta=float(seed.get("zeta", 0.1)),
                        tau_1ghz=float(seed.get("tau_1ghz", 0.1)),
                        alpha=float(seed.get("alpha", 4.4)),
                        delta_dm=float(seed.get("delta_dm", 0.0)),
                    )
                elif mk == "M3_multi":
                    shared = seed.get("shared", {})
                    comp = seed.get("components", [])
                    d: dict[str, float] = {}
                    d["gamma"] = float(shared.get("gamma", -1.6))
                    d["tau_1ghz"] = float(shared.get("tau_1ghz", 0.1))
                    d["alpha"] = float(shared.get("alpha", 4.4))
                    d["delta_dm"] = float(shared.get("delta_dm", 0.0))
                    for i, c in enumerate(comp, start=1):
                        d[f"c0_{i}"] = float(c.get("c0", 1.0))
                        d[f"t0_{i}"] = float(c.get("t0", 0.0))
                        d[f"zeta_{i}"] = float(c.get("zeta", 0.1))
                    self.seed_multi = d
                    # ensure ncomp matches seed
                    self.pipeline_kwargs["ncomp"] = max(1, len(comp))
            except Exception as e:
                warnings.warn(f"Failed to read init-guess '{init_guess_path}': {e}")

        # Resolve telescope config if it's a string
        if "telescope" in self.dataset_kwargs and isinstance(
            self.dataset_kwargs["telescope"], str
        ):
            tel_name = self.dataset_kwargs["telescope"]
            telcfg_path = self.dataset_kwargs.get(
                "telcfg_path", "scattering/configs/telescopes.yaml"
            )
            try:
                self.dataset_kwargs["telescope"] = load_telescope_block(
                    telcfg_path, tel_name
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to load telescope config for '{tel_name}' from '{telcfg_path}': {e}"
                )

    def run_full(
        self,
        model_scan=True,
        diagnostics=True,
        plot=True,
        save=True,
        show=True,
        model_keys=("M0", "M1", "M2", "M3"),
        **kwargs,
    ):
        """Main pipeline execution flow."""
        with self.pool or contextlib.nullcontext(self.pool) as pool:
            # --- FIX: Use the filtered kwargs to instantiate BurstDataset ---
            self.dataset = BurstDataset(
                self.inpath, self.outpath, **self.dataset_kwargs
            )
            self.dataset.model.dm_init = self.dm_init

            # Optional DM refinement via phase-coherence method
            if self.pipeline_kwargs.get("refine_dm", False):
                log.info("DM refinement enabled, running phase-coherence estimation...")
                try:
                    from ..dm_preprocessing import refine_dm_init

                    catalog_dm = self.dm_init  # Original value from config/bursts.yaml

                    self.dm_init = refine_dm_init(
                        dataset=self.dataset,
                        catalog_dm=catalog_dm,
                        enable_dm_estimation=True,
                        dm_search_window=self.pipeline_kwargs.get(
                            "dm_search_window", 5.0
                        ),
                        dm_grid_resolution=self.pipeline_kwargs.get(
                            "dm_grid_resolution", 0.01
                        ),
                        n_bootstrap=self.pipeline_kwargs.get("dm_n_bootstrap", 200),
                    )

                    # Update model's dm_init
                    self.dataset.model.dm_init = self.dm_init
                    log.info(
                        f"✓ DM refined: {catalog_dm:.3f} → {self.dm_init:.3f} pc/cm³"
                    )
                except Exception as e:
                    log.error(f"DM refinement failed: {e}")
                    log.info(f"Continuing with catalog DM: {self.dm_init:.3f} pc/cm³")

            n_steps = self.pipeline_kwargs.get("steps", 2000)

            # Seed initial guess from file if provided
            if self.seed_single is not None:
                init_guess = self.seed_single
            else:
                init_guess = self._get_initial_guess(self.dataset.model)

            # Automated MLE Refinement of Initial Guess
            if self.pipeline_kwargs.get("auto_guess", True):
                init_guess = refine_initial_guess_mle(self.dataset.model, init_guess)

            # Configure priors/likelihood controls
            alpha_fixed = self.pipeline_kwargs.get("alpha_fixed")
            alpha_mu = self.pipeline_kwargs.get("alpha_mu", 4.4)
            alpha_sigma = self.pipeline_kwargs.get("alpha_sigma", 0.6)
            delta_dm_sigma = self.pipeline_kwargs.get("delta_dm_sigma", 0.1)
            likelihood_kind = self.pipeline_kwargs.get("likelihood", "gaussian")
            studentt_nu = float(self.pipeline_kwargs.get("studentt_nu", 5.0))
            sample_log_params = bool(
                self.pipeline_kwargs.get("sample_log_params", True)
            )

            # Components
            ncomp = int(self.pipeline_kwargs.get("ncomp", 1))
            auto_components = bool(self.pipeline_kwargs.get("auto_components", False))

            sampler = None
            mcmc_diag = None

            if model_scan and ncomp == 1:
                sampler_name = self.pipeline_kwargs.get("fitting_method", "emcee")
                log.info(f"DEBUG: fitting_method='{sampler_name}'")

                if sampler_name == "nested":
                    log.info(
                        "Starting model selection using Nested Sampling (dynesty)..."
                    )
                    best_key, ns_results = fit_models_evidence(
                        model=self.dataset.model,
                        init=init_guess,
                        model_keys=model_keys,
                        priors=None,  # Will use defaults in nested module
                        nlive=int(self.pipeline_kwargs.get("nlive", 400)),
                        dlogz=float(self.pipeline_kwargs.get("dlogz", 0.5)),
                        alpha_prior=(alpha_mu, alpha_sigma)
                        if alpha_fixed is None
                        else None,
                        alpha_fixed=alpha_fixed,
                        likelihood_kind=likelihood_kind,
                        student_nu=studentt_nu,
                        walks=int(self.pipeline_kwargs.get("nlive_walks", 15)),
                        nproc=int(self.pipeline_kwargs.get("nproc") or 1),
                    )

                    # Convert NS result to pipeline format
                    from dynesty.utils import resample_equal

                    best_res = ns_results[best_key]
                    flat_chain = resample_equal(best_res.samples, best_res.weights)

                    results = {
                        "best_key": best_key,
                        "best_params": best_res.get_best_params(),
                        "flat_chain": flat_chain,
                        "model_instance": self.dataset.model,
                        "param_names": best_res.param_names,
                        "goodness_of_fit": {
                            "log_evidence": best_res.log_evidence,
                            "log_evidence_err": best_res.log_evidence_err,
                        },
                        "dm_init": self.dm_init,
                        "loop_stats": {"ncall": best_res.ncall},
                        "all_results": ns_results,
                    }

                    # For nested, we don't have a 'sampler' object in the emcee sense
                    # So we construct a dummy sampler or skip steps that require it
                    sampler = None

                else:
                    log.info("Starting model selection scan (BIC)...")
                    best_key, all_res = fit_models_bic(
                        model=self.dataset.model,
                        init=init_guess,
                        n_steps=n_steps // 2,
                        pool=pool,
                        model_keys=model_keys,
                        sample_log_params=sample_log_params,
                        alpha_prior=(
                            (alpha_mu, alpha_sigma)
                            if alpha_fixed is None
                            else (alpha_fixed, None)
                        ),
                        likelihood_kind=likelihood_kind,
                        student_nu=studentt_nu,
                        walker_width_frac=self.pipeline_kwargs.get(
                            "walker_width_frac", 0.01
                        ),
                    )
                    sampler = all_res[best_key][0]
                    # Pack results for emcee (legacy) path
                    
                    # Compute convergence stats for the best model from the scan
                    burn, thin, convergence_info = auto_burn_thin(sampler)
                    
                    chain = sampler.get_chain(discard=burn, thin=thin, flat=True)
                    log.info(f"Emcee chain shape: {chain.shape}")
                    param_names = FRBFitter._ORDER[best_key]
                    
                    # Better estimate than crude median of raw chain
                    best_idx = np.argmax(sampler.get_log_prob(discard=burn, thin=thin, flat=True))
                    best_params_vec = chain[best_idx].copy() # Ensure copy

                    # Handle log parameters conversion
                    if sample_log_params:
                         log_params = {"c0", "zeta", "tau_1ghz"}
                         for i, pname in enumerate(param_names):
                             # Check base name (e.g. c0_1 -> c0)
                             base = pname.split("_")[0]
                             if pname in log_params or base in log_params:
                                 best_params_vec[i] = np.exp(best_params_vec[i])

                    results = {
                        "best_key": best_key,
                        "best_params": FRBParams.from_sequence(
                            best_params_vec, best_key
                        ),
                        "param_names": param_names,
                        "goodness_of_fit": {},  # Populated later
                        "dm_init": self.dm_init,
                        "loop_stats": {},
                        "chain_stats": {
                            "burn_in": burn,
                            "thin": thin,
                            "convergence": convergence_info,
                        },
                        "flat_chain": chain,
                        "sampler": sampler,
                        "model_instance": self.dataset.model,
                        "is_multi": False,
                    }

            elif ncomp == 1:
                # Direct single model fit code... (keeping existing structure but wrapping in else)
                best_key = "M3"
                if self.pipeline_kwargs.get("sampler") == "nested":
                    log.info("Fitting model M3 directly with Nested Sampling...")
                    # Implement direct nested fit logic here if needed, or fall through
                    pass

                log.info(f"Fitting model {best_key} directly...")
                # right before sampling
                priors, use_logw = build_priors(
                    init_guess,
                    scale=6.0,
                    abs_max={"tau_1ghz": 5e4, "zeta": 5e4},
                    log_weight_pos=True,
                )  # Jeffreys weighting in prior weight
                # give generous bounds to the two broadening params
                priors["tau_1ghz"] = (1e-6, 5e4)  # ms
                priors["zeta"] = (1e-6, 5e4)  # ms
                # alpha prior bounds
                if alpha_fixed is not None:
                    priors["alpha"] = (float(alpha_fixed), float(alpha_fixed))
                    alpha_prior = (float(alpha_fixed), None)
                else:
                    lo_a = max(0.1, float(alpha_mu) - 6.0 * float(alpha_sigma))
                    hi_a = float(alpha_mu) + 6.0 * float(alpha_sigma)
                    priors["alpha"] = (lo_a, hi_a)
                    alpha_prior = (float(alpha_mu), float(alpha_sigma))
                # delta_dm bounds (top-hat prior)
                dm_w = float(delta_dm_sigma)
                priors["delta_dm"] = (-3.0 * dm_w, 3.0 * dm_w)

                fitter = FRBFitter(
                    self.dataset.model,
                    priors,
                    n_steps=n_steps,
                    pool=pool,
                    log_weight_pos=use_logw,
                    sample_log_params=sample_log_params,
                    alpha_prior=alpha_prior,
                    likelihood_kind=likelihood_kind,
                    student_nu=studentt_nu,
                    walker_width_frac=self.pipeline_kwargs.get(
                        "walker_width_frac", 0.01
                    ),
                )

                # UPDATED: Unpack tuple return (sampler, diagnostics)
                sampler, mcmc_diag = fitter.sample(init_guess, model_key=best_key)
            else:
                # Multi-component with shared PBF
                K = ncomp
                log.info(f"Fitting multi-component model with K={K} (shared PBF)...")
                # Build initial multi guess
                if self.seed_multi is not None:
                    init_multi = self.seed_multi
                else:
                    init_multi = self._get_initial_guess_multi(
                        self.dataset.model, K, base=init_guess
                    )
                # Build priors for shared + component params
                priors = {}
                # shared from build_priors around base guess
                shared_priors, use_logw = build_priors(
                    init_guess,
                    scale=6.0,
                    abs_max={"tau_1ghz": 5e4, "zeta": 5e4},
                    log_weight_pos=True,
                )
                priors.update(
                    {
                        k: v
                        for k, v in shared_priors.items()
                        if k in ("gamma", "tau_1ghz")
                    }
                )
                # alpha, delta_dm
                if alpha_fixed is not None:
                    priors["alpha"] = (float(alpha_fixed), float(alpha_fixed))
                    alpha_prior = (float(alpha_fixed), None)
                else:
                    lo_a = max(0.1, float(alpha_mu) - 6.0 * float(alpha_sigma))
                    hi_a = float(alpha_mu) + 6.0 * float(alpha_sigma)
                    priors["alpha"] = (lo_a, hi_a)
                    alpha_prior = (float(alpha_mu), float(alpha_sigma))
                dm_w = float(delta_dm_sigma)
                priors["delta_dm"] = (-3.0 * dm_w, 3.0 * dm_w)

                # per-component bounds
                tmin, tmax = (
                    float(self.dataset.time.min()),
                    float(self.dataset.time.max()),
                )
                for i in range(1, K + 1):
                    priors[f"c0_{i}"] = (1e-6, 1e9)
                    priors[f"t0_{i}"] = (tmin, tmax)
                    priors[f"zeta_{i}"] = (1e-6, 5e4)

                fitter = FRBFitter(
                    self.dataset.model,
                    priors,
                    n_steps=n_steps,
                    pool=pool,
                    log_weight_pos=use_logw,
                    sample_log_params=sample_log_params,
                    alpha_prior=alpha_prior,
                    likelihood_kind=likelihood_kind,
                    student_nu=studentt_nu,
                    walker_width_frac=self.pipeline_kwargs.get(
                        "walker_width_frac", 0.01
                    ),
                )
                names = fitter.build_multicomp_order(K)

                # UPDATED: Unpack tuple return (sampler, diagnostics)
                sampler, mcmc_diag = fitter.sample(init_multi, model_key="M3_multi")
                best_key = "M3_multi"

            if sampler is not None:
                log.info("Processing MCMC chains...")
                burn, thin, convergence_info = auto_burn_thin(sampler)
                flat_chain = sampler.get_chain(discard=burn, thin=thin, flat=True)
                if flat_chain.shape[0] == 0:
                    raise RuntimeError(
                        "MCMC chain is empty after burn-in and thinning. Check sampler settings or increase n_steps."
                    )

                if best_key == "M3_multi":
                    # keep theta_best and names for downstream
                    idx_best = int(
                        np.argmax(
                            sampler.get_log_prob(discard=burn, thin=thin, flat=True)
                        )
                    )
                    theta_best = flat_chain[idx_best]

                    param_names = list(fitter.custom_order["M3_multi"])  # type: ignore[attr-defined]
                    results = {
                        "best_key": best_key,
                        "sampler": sampler,
                        "flat_chain": flat_chain,
                        "param_names": param_names,
                        "dm_init": self.dm_init,
                        "model_instance": self.dataset.model,
                        "chain_stats": {
                            "burn_in": burn,
                            "thin": thin,
                            "convergence": convergence_info,
                        },
                        "is_multi": True,
                        "K": K,
                        "theta_best": theta_best,
                        "mcmc_diagnostics": mcmc_diag,  # Add MCMC diagnostics
                    }
                else:
                    best_params = FRBParams.from_sequence(
                        flat_chain[
                            np.argmax(
                                sampler.get_log_prob(discard=burn, thin=thin, flat=True)
                            )
                        ],
                        best_key,
                    )

                    param_names = (
                        FRBFitter._ORDER[best_key]
                        if best_key in FRBFitter._ORDER
                        else []  # Should not happen for standard BIC scan
                    )

                    # Calculate goodness of fit
                    # Fix: pass correct arguments matching definition
                    gof = goodness_of_fit(
                        self.dataset.data,
                        self.dataset.model(best_params, best_key),
                        self.dataset.model.noise_std,
                        len(param_names),
                    )
                    loop_stats = {
                        "burn_in": burn,
                        "thin": thin,
                        "convergence": convergence_info,
                    }

                    results = {
                        "best_key": best_key,
                        "best_params": best_params,
                        "param_names": param_names,
                        "goodness_of_fit": gof,
                        "dm_init": self.dm_init,
                        "loop_stats": loop_stats,
                        "chain_stats": {
                            "burn_in": burn,
                            "thin": thin,
                            "convergence": convergence_info,
                        },
                        "flat_chain": flat_chain,
                        "sampler": sampler,
                        "model_instance": self.dataset.model,
                        "is_multi": False,
                        "mcmc_diagnostics": mcmc_diag,  # Add MCMC diagnostics
                    }

            # Diagnostics and plotting should happen after results is definitely populated
            if results is None:
                raise RuntimeError(
                    "Results dictionary was not populated by any fitting path."
                )

            if diagnostics:
                # Skip diagnostics if chain is badly non-converged (R̂ > 5)
                # Only applicable if sampler is not None (i.e., emcee)
                if sampler is not None:
                    max_rhat = (
                        results["chain_stats"]
                        .get("convergence", {})
                        .get("max_rhat", 1.0)
                    )
                    if max_rhat > 5.0:
                        log.warning(
                            f"Skipping diagnostics: chain not converged (R̂ = {max_rhat:.2f} > 5)"
                        )
                        results["diagnostics"] = {
                            "skipped": True,
                            "reason": f"R̂ = {max_rhat:.2f} too high",
                        }
                    else:
                        try:
                            diag_runner = BurstDiagnostics(self.dataset, results)
                            results["diagnostics"] = diag_runner.run_all(
                                sb_steps=n_steps // 4, pool=pool
                            )
                        except Exception as e:
                            log.warning(f"Diagnostics failed: {e}")
                            results["diagnostics"] = {"skipped": True, "reason": str(e)}
                else:  # Nested sampling path, no Rhat
                    try:
                        diag_runner = BurstDiagnostics(self.dataset, results)
                        results["diagnostics"] = diag_runner.run_all(
                            sb_steps=n_steps // 4, pool=pool
                        )
                    except Exception as e:
                        log.warning(f"Diagnostics failed: {e}")
                        results["diagnostics"] = {"skipped": True, "reason": str(e)}

            if best_key == "M3_multi":
                model_dyn = self._build_multi_model(results)
                results["goodness_of_fit"] = goodness_of_fit(
                    self.dataset.data,
                    model_dyn,
                    self.dataset.model.noise_std,
                    len(results["param_names"]),
                )
            else:
                results["goodness_of_fit"] = goodness_of_fit(
                    self.dataset.data,
                    self.dataset.model(results["best_params"], best_key),
                    self.dataset.model.noise_std,
                    len(results["param_names"]),
                )
            log.info(
                f"Best model: {best_key} | χ²/dof = {results['goodness_of_fit']['chi2_reduced']:.2f}"
            )

            # --- CONSOLIDATED FIT REPORTING ---
            print_fit_summary(results)
            # ----------------------------------

            if save:
                import json

                # Helper to convert numpy types
                class NumpyEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.integer):
                            return int(obj)
                        if isinstance(obj, np.floating):
                            return float(obj)
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        return super().default(obj)

                # Prepare safe dict (persists posterior percentiles -> tau +/- sigma)
                safe_results = build_safe_results(results)

                json_path = os.path.join(self.outpath, f"{self.name}_fit_results.json")
                with open(json_path, "w") as f:
                    json.dump(safe_results, f, indent=4, cls=NumpyEncoder)
                log.info(f"Saved fit results to {json_path}")

            if plot:
                try:
                    from ..visualization import plot_scattering_diagnostic

                    log.info("Generating publication-quality diagnostic plot...")
                    
                    # Prepare arguments for visualization
                    # Generate model array
                    best_params = results["best_params"]
                    best_key = results["best_key"]
                    if results.get("is_multi"):
                        model_arr = self._build_multi_model(results)
                    else:
                        model_arr = self.dataset.model(best_params, best_key)
                        
                    output_plot_path = self.outpath / f"{self.name}_diagnostic.png"
                    
                    plot_scattering_diagnostic(
                        data=self.dataset.data,
                        model=model_arr,
                        freq=self.dataset.freq,
                        time=self.dataset.time,
                        params=best_params,
                        results=results,
                        output_path=output_plot_path,
                        burst_name=self.name,
                        telescope=getattr(self.dataset.telescope, "name", "Unknown"),
                    )

                    # Fit-quality view: profile overlay + sigma-residual whiteness,
                    # which the stock waterfalls cannot show (resid_sigma ~1 good).
                    if not results.get("is_multi"):
                        from ..visualization import plot_fit_quality

                        plot_fit_quality(
                            data=self.dataset.data,
                            model=model_arr,
                            freq=self.dataset.freq,
                            time=self.dataset.time,
                            noise=self.dataset.model.noise_std,
                            valid=self.dataset.model.valid,
                            params=best_params,
                            results=results,
                            output_path=self.outpath / f"{self.name}_fitquality.png",
                            burst_name=self.name,
                            telescope=getattr(self.dataset.telescope, "name", "Unknown"),
                        )
                except Exception as e:
                    log.warning(
                        f"Modular plotting failed: {e}. Falling back to legacy plots."
                    )
                    create_fit_summary_plot(
                        self.dataset, results, save=save, show=False
                    )
                    create_four_panel_plot(self.dataset, results, save=save, show=False)

            return results

    def _get_initial_guess(self, model) -> "FRBParams":
        """Generate data-driven initial guess for MCMC.

        Uses the burstfit_init module to extract parameter estimates
        directly from the data instead of hardcoded values.
        """
        log.info("Finding data-driven initial guess for MCMC...")

        # Try data-driven estimation first
        try:
            from ..burstfit_init import data_driven_initial_guess

            result = data_driven_initial_guess(
                data=model.data,
                freq=model.freq,
                time=model.time,
                dm=self.dm_init,
                verbose=True,
            )

            init_guess = result.params
            log.info("Data-driven initial guess:")
            log.info(f"  c0      = {init_guess.c0:.2f}")
            log.info(f"  t0      = {init_guess.t0:.3f} ms")
            log.info(f"  gamma   = {init_guess.gamma:.2f}")
            log.info(f"  zeta    = {init_guess.zeta:.3f} ms")
            log.info(f"  tau_1ghz= {init_guess.tau_1ghz:.3f} ms")
            log.info(f"  alpha   = {init_guess.alpha:.2f}")

            # Store diagnostics for later inspection
            self._init_guess_diagnostics = result.diagnostics

            return init_guess

        except Exception as e:
            log.warning(f"Data-driven guess failed: {e}. Falling back to optimization.")

        # Fallback: quick optimization-based guess
        f_ds = 1
        t_ds = 1

        # Build down-sampled arrays
        data_ds = model.data[::f_ds, ::t_ds]
        time_ds = model.time[::t_ds]
        freq_ds = model.freq[::f_ds]

        from ..burstfit import FRBModel
        model_ds = FRBModel(
            data=data_ds,
            time=time_ds,
            freq=freq_ds,
            dm_init=self.dm_init,
            df_MHz=model.df_MHz,
        )

        prof = np.nansum(model_ds.data, axis=0)
        if np.all(prof == 0):
            return FRBParams(c0=0, t0=model_ds.time.mean(), gamma=0, zeta=0, tau_1ghz=0)

        # Data-derived rough guess (better than pure hardcodes)
        t0_idx = np.argmax(prof)
        t0 = model_ds.time[t0_idx]
        c0 = np.sum(prof)

        # Estimate spectral index from data
        spectrum = np.nansum(model_ds.data, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            log_freq = np.log(freq_ds)
            log_flux = np.log(np.maximum(spectrum, 1e-10))
            mask = np.isfinite(log_flux) & np.isfinite(log_freq)
        if mask.sum() > 3:
            try:
                gamma = np.polyfit(log_freq[mask], log_flux[mask], 1)[0]
                gamma = np.clip(gamma, -5, 2)
            except Exception:
                gamma = -1.6
        else:
            gamma = -1.6

        # Estimate width from profile variance
        weights = np.maximum(prof - np.percentile(prof, 10), 0)
        weights /= np.sum(weights) + 1e-30
        t_var = np.sum((model_ds.time - t0) ** 2 * weights)
        width = 2.355 * np.sqrt(max(t_var, 1e-6))

        # Initial zeta and tau: split observed width
        zeta = max(0.1, width * 0.4)
        tau_1ghz = max(0.1, width * 0.4)

        rough_guess = FRBParams(
            c0=c0,
            t0=t0,
            gamma=gamma,
            zeta=zeta,
            tau_1ghz=tau_1ghz,
            alpha=4.0,  # Thin screen default
        )

        # Refine with L-BFGS-B
        priors, use_logw = build_priors(
            rough_guess,
            scale=1.5,
            abs_max={"tau_1ghz": 5e4, "zeta": 5e4},
            log_weight_pos=True,
        )
        model_key = "M3"
        x0 = rough_guess.to_sequence(model_key)
        bounds = [priors[n] for n in FRBFitter._ORDER[model_key]]

        def nll(theta):
            p = FRBParams.from_sequence(theta, model_key)
            ll = model_ds.log_likelihood(p, model_key)
            return -ll if np.isfinite(ll) else np.inf

        from scipy.optimize import minimize
        res = minimize(
            nll,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-7},
        )
        if not res.success:
            warnings.warn("Initial guess optimization failed. Using rough guess.")
            return rough_guess
        log.info("Refined initial guess found via optimization.")
        return FRBParams.from_sequence(res.x, model_key)

    def _get_initial_guess_multi(
        self, model, K: int, base: "FRBParams"
    ) -> dict[str, float]:
        # Smooth profile and find K peaks
        prof = np.nansum(model.data, axis=0)
        if model.dt > 0:
            sigma_samps = max(1, int((0.1 / 2.355) / model.dt))
            if sigma_samps > 1:
                from scipy.ndimage import gaussian_filter1d
                prof = gaussian_filter1d(prof, sigma_samps)
        idxs = np.argpartition(prof, -K)[-K:]
        idxs = np.sort(idxs)
        # initial guesses
        total = np.sum(prof)
        init: dict[str, float] = {
            "gamma": base.gamma,
            "tau_1ghz": max(base.tau_1ghz, 1e-3),
            "alpha": getattr(base, "alpha", 4.4),
            "delta_dm": 0.0,
        }
        for j, ix in enumerate(idxs, start=1):
            init[f"t0_{j}"] = model.time[ix]
            init[f"c0_{j}"] = max(total / K, 1e-3)
            init[f"zeta_{j}"] = max(getattr(base, "zeta", 0.05), 1e-3)
        return init

    def _build_multi_model(self, results: Dict[str, Any]):
        names = results["param_names"]
        theta = results["theta_best"]
        K = int(results["K"])
        model = results["model_instance"]

        # helper
        def get(name):
            return theta[names.index(name)] if name in names else None

        gamma = get("gamma")
        tau1 = get("tau_1ghz")
        alpha = get("alpha")
        delta_dm = get("delta_dm")
        model_sum = np.zeros_like(model.data)
        for i in range(1, K + 1):
            c0 = get(f"c0_{i}")
            t0 = get(f"t0_{i}")
            zeta = get(f"zeta_{i}")
            p = FRBParams(
                c0=c0,
                t0=t0,
                gamma=gamma,
                zeta=zeta,
                tau_1ghz=tau1,
                alpha=alpha,
                delta_dm=delta_dm,
            )
            model_sum = model_sum + model(p, "M3")
        return model_sum

def _main():
    p = argparse.ArgumentParser(description="Run BurstFit pipeline on a .npy file.")
    # Add all possible arguments here
    p.add_argument("inpath", type=Path, help="Input .npy file")
    p.add_argument("--frb", type=str, help="Event name")
    p.add_argument("--outpath", type=Path, help="Output filepath")
    p.add_argument("--dm_init", type=float, default=0.0)
    p.add_argument("--telescope", default="CHIME")
    p.add_argument("--telcfg", default="telescopes.yaml")
    p.add_argument("--sampcfg", default="sampler.yaml")
    p.add_argument("--nproc", type=int, default=None)
    p.add_argument("--yes", action="store_true", help="Bypass pool confirmation")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--f_factor", type=int, default=1)
    p.add_argument("--t_factor", type=int, default=1)
    p.add_argument(
        "--outer-trim",
        dest="outer_trim",
        type=float,
        help="Fraction to trim from each time edge (0-0.49)",
    )
    p.add_argument(
        "--flip-freq",
        dest="flip_freq",
        action="store_true",
        help="Flip frequency axis (high at top)",
    )
    p.add_argument(
        "--no-flip-freq",
        dest="flip_freq",
        action="store_false",
        help="Do not flip frequency axis",
    )
    # New modeling controls
    p.add_argument(
        "--alpha-fixed",
        type=float,
        default=None,
        help="Fix alpha frequency scaling exponent",
    )
    p.add_argument(
        "--alpha-mu", type=float, default=4.4, help="Gaussian prior mean for alpha"
    )
    p.add_argument(
        "--alpha-sigma", type=float, default=0.6, help="Gaussian prior sigma for alpha"
    )
    p.add_argument(
        "--delta-dm-sigma",
        type=float,
        default=0.1,
        help="Top-hat prior sigma for delta DM (pc cm^-3)",
    )
    p.add_argument(
        "--likelihood", type=str, choices=["gaussian", "studentt"], default="gaussian"
    )
    p.add_argument(
        "--studentt-nu", type=float, default=5.0, help="Student-t degrees of freedom"
    )
    p.add_argument(
        "--no-logspace",
        dest="sample_log_params",
        action="store_false",
        help="Disable log-space sampling for positive params",
    )
    # Seeding / walkers
    p.add_argument(
        "--init-guess",
        type=Path,
        default=None,
        help="Path to JSON seed for initial guess (single or multi)",
    )
    p.add_argument(
        "--walker-width-frac",
        type=float,
        default=0.01,
        help="Initial walker cloud width as fraction of prior span",
    )
    # Multi-component controls
    p.add_argument(
        "--ncomp",
        type=int,
        default=1,
        help="Number of Gaussian components (shared PBF)",
    )
    p.add_argument(
        "--auto-components",
        action="store_true",
        help="Greedy BIC-based component selection (placeholder)",
    )
    # Earmarks / placeholders
    p.add_argument(
        "--anisotropy-enabled",
        action="store_true",
        help="Earmark: enable anisotropy option (not implemented)",
    )
    p.add_argument(
        "--anisotropy-axial-ratio",
        type=float,
        default=1.0,
        help="Earmark: anisotropy axial ratio (not implemented)",
    )
    p.add_argument(
        "--baseline-order",
        type=int,
        default=0,
        help="Earmark: polynomial baseline order to marginalize (not implemented)",
    )
    p.add_argument(
        "--correlated-resid",
        action="store_true",
        help="Earmark: AR(1)/GP residual model (not implemented)",
    )
    p.add_argument(
        "--fitting-method",
        dest="fitting_method",
        type=str,
        choices=["emcee", "nested"],
        default="emcee",
        help="Sampler choice (emcee or nested)",
    )
    # Add flags for boolean pipeline controls
    p.add_argument("--no-scan", dest="model_scan", action="store_false")
    p.add_argument("--no-diag", dest="diagnostics", action="store_false")
    p.add_argument("--no-plot", dest="plot", action="store_false")
    p.set_defaults(model_scan=True, diagnostics=True, plot=True, flip_freq=False)
    args = p.parse_args()

    # --- FIX: Pass all arguments as a dict to the pipeline constructor ---
    # The new __init__ will sort them out automatically.
    pipeline_kwargs = vars(args)

    # Extract required args and provide sensible defaults
    inpath = pipeline_kwargs.pop("inpath")
    outpath = pipeline_kwargs.pop("outpath") or inpath.parent
    name = pipeline_kwargs.pop("frb") or inpath.stem
    dm_init = pipeline_kwargs.pop("dm_init")

    # Harmonize config key names for dataset constructor
    telcfg_cli = pipeline_kwargs.pop("telcfg", None)
    if telcfg_cli is not None:
        pipeline_kwargs["telcfg_path"] = telcfg_cli
    sampcfg_cli = pipeline_kwargs.pop("sampcfg", None)
    if sampcfg_cli is not None:
        pipeline_kwargs["sampcfg_path"] = sampcfg_cli

    pipe = BurstPipeline(
        name=name,
        inpath=inpath,  # positional arg extracted above
        outpath=outpath,
        dm_init=dm_init,
        **pipeline_kwargs,
    )

    pipe.run_full(
        model_scan=args.model_scan, diagnostics=args.diagnostics, plot=args.plot
    )

if __name__ == "__main__":
    _main()
