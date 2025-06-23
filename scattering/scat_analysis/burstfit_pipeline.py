"""
burstfit_pipeline.py
====================

Object-oriented orchestrator for the BurstFit pipeline. This module connects
the data loading, preprocessing, fitting, diagnostics, and plotting modules
into a coherent, runnable sequence.
"""
from __future__ import annotations

import logging
import warnings
import argparse
import inspect
import contextlib
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import scipy as sp
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

from .burstfit import (
    FRBModel,
    FRBFitter,
    FRBParams,
    build_priors,
    plot_dynamic,
    goodness_of_fit,
    downsample,
)
from .burstfit_modelselect import fit_models_bic
from .burstfit_robust import (
    subband_consistency,
    leave_one_out_influence,
    plot_influence,
    fit_subband_profiles,
    plot_subband_profiles,
    dm_optimization_check,
)
from .config_utils import load_telescope_block, load_sampler_block
from .pool_utils import build_pool

# --- Setup Logging ---
log = logging.getLogger("burstfit.pipeline")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s | %(name)s] %(message)s")


###############################################################################
# 0. PLOTTING FUNCTIONS
###############################################################################

def create_four_panel_plot(
    dataset: "BurstDataset",
    results: Dict[str, Any],
    *,
    output_path: Path | None = None,
    show: bool = True
):
    """Creates a four-panel diagnostic plot comparing data, model, and residuals."""
    log.info("Generating four-panel diagnostic plot...")
    
    best_p, best_key = results["best_params"], results["best_key"]
    model_instance = results["model_instance"]
    
    data, time, freq = dataset.data, dataset.time, dataset.freq
    time_centered = time - (time[0] + (time[-1] - time[0]) / 2)
    extent = [time_centered[0], time_centered[-1], freq[0], freq[-1]]

    clean_model = model_instance(best_p, best_key)
    synthetic_noise = np.random.normal(0.0, model_instance.noise_std[:, None], size=data.shape)
    synthetic_data = clean_model + synthetic_noise
    residual = data - synthetic_data

    def _normalize_panel_data(arr_2d, off_pulse_data):
        mean_off, std_off = np.nanmean(off_pulse_data), np.nanstd(off_pulse_data)
        if std_off < 1e-9: return arr_2d
        arr_norm = (arr_2d - mean_off) / std_off
        peak = np.nanmax(arr_norm)
        return arr_norm / peak if peak > 0 else arr_norm

    q = data.shape[1] // 4
    data_off_pulse = data[:, np.r_[0:q, -q:0]]
    
    data_norm = _normalize_panel_data(data, data_off_pulse)
    model_norm = _normalize_panel_data(clean_model, data_off_pulse)
    synthetic_norm = _normalize_panel_data(synthetic_data, data_off_pulse)

    fig, axes = plt.subplots(
        nrows=2, ncols=8,
        gridspec_kw={'height_ratios': [1, 2.5], 'width_ratios': [2, 0.5] * 4},
        figsize=(24, 8)
    )
    
    panel_data = [
        (data_norm, 'Data', r'I$_{\mathrm{data}}$'),
        (model_norm, 'Model', r'I$_{\mathrm{model}}$'),
        (synthetic_norm, 'Synth.', r'I$_{\mathrm{model+noise}}$'),
        (residual, 'Residual', r'I$_{\mathrm{residual}}$'),
    ]

    for i, (panel_ds, title, label) in enumerate(panel_data):
        col_idx = i * 2
        ax_ts, ax_sp, ax_wf = axes[0, col_idx], axes[1, col_idx + 1], axes[1, col_idx]

        ts = np.nansum(panel_ds, axis=0)
        sp = np.nansum(panel_ds, axis=1)

        ax_ts.step(time_centered, ts, where='mid', c='k', lw=1.5, label=label)
        ax_ts.legend(loc='upper right', fontsize=14, frameon=False)
        
        cmap = 'coolwarm' if title == 'Residual' else 'plasma'
        vmax = np.nanpercentile(np.abs(panel_ds), 99.5) if title != 'Residual' else np.nanstd(panel_ds) * 3
        vmin = 0 if title != 'Residual' else -vmax
        ax_wf.imshow(panel_ds, extent=extent, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto', origin='lower')

        ax_sp.step(sp, freq, where='mid', c='k', lw=1.5)
        
        ax_ts.set_yticks([]); ax_ts.tick_params(axis='x', labelbottom=False); ax_ts.set_xlim(extent[0], extent[1])
        ax_sp.set_xticks([]); ax_sp.tick_params(axis='y', labelleft=False); ax_sp.set_ylim(extent[2], extent[3])
        ax_wf.set_xlabel('Time [ms]')
        if i == 0: ax_wf.set_ylabel('Frequency [GHz]')
        else: ax_wf.tick_params(axis='y', labelleft=False)
        axes[0, col_idx + 1].axis('off')

    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    fig.suptitle("Four-Panel Fit Summary", fontsize=20, weight='bold')

    if output_path:
        log.info(f"Saving 4-panel plot to {output_path}")
        fig.savefig(output_path, bbox_inches='tight', dpi=150)
    if show: plt.show()
    else: plt.close(fig)
    return fig

def create_sixteen_panel_plot(
    dataset: "BurstDataset",
    results: Dict[str, Any],
    *,
    output_path: Path | None = None,
    show: bool = True
):
    """Creates a comprehensive 16-panel diagnostic plot summarizing the fit."""
    log.info("Generating 16-panel comprehensive diagnostics plot...")
    
    best_key, best_p = results["best_key"], results["best_params"]
    sampler, gof = results["sampler"], results.get("goodness_of_fit")
    chain_stats, flat_chain = results.get("chain_stats", {}), results["flat_chain"]
    param_names, diag_results = results["param_names"], results.get("diagnostics", {})
    model_instance = results["model_instance"]
    
    model_dyn = model_instance(best_p, best_key)
    residual = dataset.data - model_dyn

    fig, axes = plt.subplots(4, 4, figsize=(20, 20), constrained_layout=True)
    ax = axes.ravel()
    
    # Panels 0-2: Data, Model, Residual
    vmin, vmax = np.percentile(dataset.data, [1, 99])
    plot_dynamic(ax[0], dataset.data, dataset.time, dataset.freq, vmin=vmin, vmax=vmax, cmap='plasma'); ax[0].set_title("Data")
    plot_dynamic(ax[1], model_dyn, dataset.time, dataset.freq, vmin=vmin, vmax=vmax, cmap='plasma'); ax[1].set_title(f"Model ({best_key})")
    res_std = np.std(residual); plot_dynamic(ax[2], residual, dataset.time, dataset.freq, vmin=-3*res_std, vmax=3*res_std, cmap='coolwarm'); ax[2].set_title("Residual")

    # Panel 3: Residual Histogram
    res_norm = residual / model_instance.noise_std[:, None]
    ax[3].hist(res_norm.flatten(), bins=100, density=True, color='gray', label='Residuals')
    x_pdf = np.linspace(-4, 4, 100); ax[3].plot(x_pdf, sp.stats.norm.pdf(x_pdf), 'm-', lw=2, label='N(0,1)'); ax[3].set_title('Residual Distribution'); ax[3].legend()

    # Panel 4-5: Time Profile and Spectrum
    ax[4].plot(dataset.time, np.sum(dataset.data, axis=0), 'k-', label='Data'); ax[4].plot(dataset.time, np.sum(model_dyn, axis=0), 'm--', lw=2, label='Model'); ax[4].set_title('Time Profile'); ax[4].legend(); ax[4].set_xlabel('Time [ms]')
    ax[5].plot(dataset.freq, np.sum(dataset.data, axis=1), 'k-', label='Data'); ax[5].plot(dataset.freq, np.sum(model_dyn, axis=1), 'm--', lw=2, label='Model'); ax[5].set_title('Frequency Spectrum'); ax[5].legend(); ax[5].set_xlabel('Frequency [GHz]')

    # Panel 6-8: Diagnostics
    if diag_results.get("influence") is not None: plot_influence(ax[6], diag_results["influence"], dataset.freq)
    else: ax[6].text(0.5, 0.5, 'Influence\nNot Run', ha='center', va='center'); ax[6].set_axis_off()

    if diag_results.get("subband_2d") is not None:
        p_name, s_res, _ = diag_results["subband_2d"]
        if p_name:
            valid = [(i,v,e) for i, (v,e) in enumerate(s_res) if np.isfinite(v) and e > 0]
            if valid:
                idx, vals, errs = zip(*valid)
                edges = np.linspace(0, dataset.freq.size, len(s_res) + 1, dtype=int)
                band_centers = np.array([dataset.freq[edges[i]:edges[i+1]].mean() for i in range(len(s_res))])[list(idx)]
                ax[7].errorbar(band_centers, vals, yerr=errs, fmt="o", c='k', capsize=3, label=f"Sub-band {p_name}")
                global_val = getattr(best_p, p_name); ax[7].axhline(global_val, color="m", ls="--", label="Global Fit")
                ax[7].legend(); ax[7].set_ylabel(p_name)
    else: ax[7].text(0.5, 0.5, '2D Sub-band\nNot Run', ha='center', va='center'); ax[7].set_axis_off()
    
    if diag_results.get("profile1d") is not None: plot_subband_profiles(ax[8], *diag_results["profile1d"], best_p, fontsize=10)
    else: ax[8].text(0.5, 0.5, '1D Profile\nNot Run', ha='center', va='center'); ax[8].set_axis_off()

    # Panel 9-10: MCMC chains
    chain = sampler.get_chain(); burn = chain_stats.get("burn_in", 0)
    ax[9].plot(chain[:, ::10, 0], 'k', alpha=0.3); ax[9].axvline(burn, color='m', ls='--'); ax[9].set_title(f'Trace: {param_names[0]}'); ax[9].set_xlabel('Step')
    if len(param_names) > 1:
        corr = np.corrcoef(flat_chain.T); np.fill_diagonal(corr, 0); i, j = np.unravel_index(np.abs(corr).argmax(), corr.shape)
        ax[10].scatter(flat_chain[:,i], flat_chain[:,j], s=1, alpha=0.1, c='k'); ax[10].set_xlabel(param_names[i]); ax[10].set_ylabel(param_names[j]); ax[10].set_title(f'Highest Correlation (ρ={corr[i,j]:.2f})')
    
    # Panel 11-12: ACF and DM Check
    if gof: ax[11].plot((np.arange(len(gof['residual_autocorr'])) - len(gof['residual_autocorr'])//2) * dataset.dt_ms, gof['residual_autocorr'], 'k-'); ax[11].set_title('Residual ACF'); ax[11].set_xlabel('Lag [ms]')
    if diag_results.get("dm_check") is not None: dms, snrs = diag_results['dm_check']; ax[12].plot(dms, snrs, 'o-k'); ax[12].set_title('DM Optimization'); ax[12].set_xlabel(r'ΔDM (pc cm$^{-3}$)')

    # Panels 13-15: Text Summaries
    for i in [13, 14, 15]: ax[i].set_axis_off()
    if gof: ax[13].text(0.05, 0.95, f"GoF:\nχ²/dof = {gof['chi2_reduced']:.2f}", va='top', fontfamily='monospace', fontsize=12)
    p_summary = "Best Fit (Median & 1σ):\n" + "\n".join([f"{n}: {np.median(flat_chain[:,i]):.3f} ± {np.std(flat_chain[:,i]):.3f}" for i, n in enumerate(param_names)])
    ax[14].text(0.05, 0.95, p_summary, va='top', fontfamily='monospace', fontsize=12)
    ax[15].text(0.05, 0.95, f"File:\n{dataset.path.name}", va='top', fontfamily='monospace', fontsize=12)
    
    fig.suptitle(f"Comprehensive Fit Diagnostics: {dataset.path.name}", fontsize=24, weight='bold')
    if output_path: fig.savefig(output_path, bbox_inches='tight', dpi=150)
    if show: plt.show()
    else: plt.close(fig)
    return fig

###############################################################################
# 1. DATASET LOADER
###############################################################################
class BurstDataset:
    """Loads and preprocesses a burst from a .npy file."""
    def __init__(
        self,
        path: str | Path,
        *,
        telescope: str = "CHIME",
        telcfg_path: str = "telescopes.yaml",
        sampcfg_path: str = "sampler.yaml",
        f_factor: int = 1,
        t_factor: int = 1,
        outer_trim: float = 0.45,
        smooth_ms: float = 0.1,
        center_burst: bool = True,
        flip_freq: bool = False,
        lazy: bool = False,
    ):
        self.path = Path(path)
        self.telname, self.telparams = load_telescope_block(telcfg_path, telescope=telescope)
        self.sampname, self.sampparams = load_sampler_block(sampcfg_path)
        self.f_factor, self.t_factor = f_factor, t_factor
        self.outer_trim, self.smooth_ms = outer_trim, smooth_ms
        self.center_burst, self.flip_freq = center_burst, flip_freq
        self.data = self.freq = self.time = self.df_MHz = self.dt_ms = self.model = None
        if not lazy: self.load()

    def load(self):
        if self.data is not None: return
        raw = self._load_raw()
        if self.flip_freq: raw = np.flipud(raw)
        
        # Build axes for the raw data to use in preprocessing
        raw_freq, raw_time, _, _ = self._build_axes(raw.shape, f_factor=1, t_factor=1)
        ds = self._bandpass_correct(raw, raw_time)
        ds = self._trim_buffer(ds)
        self.data = self._downsample_and_renormalize(ds)
        
        # Re-build axes for final downsampled data shape
        self.freq, self.time, self.df_MHz, self.dt_ms = self._build_axes(self.data.shape)

        if self.center_burst: self._centre_burst()
        
        self.model = FRBModel(time=self.time, freq=self.freq, data=self.data, df_MHz=self.df_MHz)

    def _load_raw(self):
        if not self.path.exists(): raise FileNotFoundError(f"Data not found: {self.path}")
        try:
            data = np.load(self.path); return np.nan_to_num(data.astype(np.float64))
        except Exception as e: raise IOError(f"Failed to load {self.path}: {e}")

    def _build_axes(self, shape, f_factor=None, t_factor=None):
        f_factor = f_factor if f_factor is not None else self.f_factor
        t_factor = t_factor if t_factor is not None else self.t_factor
        
        # Get raw shape from config, not from current array shape
        p = self.telparams
        n_ch_raw = int(p.get("n_ch_raw", shape[0] * f_factor))
        
        df_MHz = p["df_MHz_raw"] * f_factor
        dt_ms = p["dt_ms_raw"] * t_factor
        
        final_n_ch = shape[0]
        final_n_t = shape[1]

        freq = np.linspace(p["f_min_GHz"], p["f_max_GHz"], final_n_ch)
        time = np.arange(final_n_t) * dt_ms
        return freq, time, df_MHz, dt_ms

    def _bandpass_correct(self, arr, time_axis):
        q = time_axis.size // 4; off_pulse_idx = np.r_[0:q, -q:0]
        mu = np.nanmean(arr[:, off_pulse_idx], axis=1, keepdims=True)
        sig = np.nanstd(arr[:, off_pulse_idx], axis=1, keepdims=True)
        sig[sig < 1e-9] = np.nan
        return np.nan_to_num((arr - mu) / sig, nan=0.0)

    def _trim_buffer(self, arr):
        n_trim = int(self.outer_trim * arr.shape[1])
        return arr[:, n_trim:-n_trim] if n_trim > 0 else arr

    def _downsample_and_renormalize(self, arr):
        ds_arr = downsample(arr, self.f_factor, self.t_factor)
        peak = np.nanmax(ds_arr)
        return ds_arr / peak if peak > 0 else ds_arr

    def _centre_burst(self):
        prof = np.nansum(self.data, axis=0)
        if self.smooth_ms > 0 and self.dt_ms > 0:
            sigma_samps = (self.smooth_ms / 2.355) / self.dt_ms
            prof = gaussian_filter1d(prof, sigma=sigma_samps)
        shift = self.data.shape[1] // 2 - np.argmax(prof)
        self.data = np.roll(self.data, shift, axis=1)

###############################################################################
# 2. DIAGNOSTICS WRAPPER
###############################################################################
class BurstDiagnostics:
    """A container for running and storing all post-fit diagnostic checks."""
    def __init__(self, dataset: "BurstDataset", results: Dict[str, Any]):
        self.dataset = dataset
        self.results_in = results
        self.diag_results: Dict[str, Any] = {}

    def run_all(self, sb_steps: int = 500, pool=None):
        log.info("Running all post-fit diagnostics...")
        best_p = self.results_in["best_params"]
        best_key = self.results_in["best_key"]
        dm_init = self.results_in["dm_init"]
        model_instance = self.results_in["model_instance"]
        model_dyn = model_instance(best_p, best_key)

        self.diag_results['influence'] = leave_one_out_influence(self.dataset.data, model_dyn)
        self.diag_results['dm_check'] = dm_optimization_check(self.dataset.data, self.dataset.freq, self.dataset.time, dm_init)
        self.diag_results['subband_2d'] = subband_consistency(
            self.dataset.data, self.dataset.freq, self.dataset.time, dm_init, 
            self.dataset.df_MHz, best_p, model_key=best_key, n_steps=sb_steps, pool=pool
        )
        self.diag_results['profile1d'] = fit_subband_profiles(self.dataset, best_p, dm_init)
        log.info("Diagnostics complete.")
        return self.diag_results

###############################################################################
# 3. PIPELINE FAÇADE
###############################################################################
class BurstPipeline:
    """Main orchestrator for the fitting pipeline."""
    def __init__(self, path: str | Path, *, dm_init: float = 0.0, **kwargs):
        """
        Initializes the pipeline.

        Args:
            path: Path to the input .npy data file.
            dm_init: Initial dispersion measure for the data.
            **kwargs: Keyword arguments for pipeline configuration. These are
                      intelligently split between BurstDataset and the pipeline.
        """
        self.path = path
        self.dm_init = dm_init

        # --- FIX: Intelligently separate kwargs for different components ---

        # Get the names of all valid arguments for the BurstDataset constructor
        dataset_params = inspect.signature(BurstDataset).parameters
        dataset_arg_names = list(dataset_params.keys())

        # Create a dictionary with only the kwargs that BurstDataset accepts
        self.dataset_kwargs = {k: v for k, v in kwargs.items() if k in dataset_arg_names}

        # Store the remaining kwargs for the pipeline itself (e.g., 'steps')
        self.pipeline_kwargs = {k: v for k, v in kwargs.items() if k not in dataset_arg_names}

        # Create the multiprocessing pool
        self.pool = build_pool(
            self.pipeline_kwargs.get("nproc"), 
            auto_ok=self.pipeline_kwargs.get("yes", False)
        )

    def run_full(self, model_scan=True, diagnostics=True, plot=True, show=True, model_keys=("M0","M1","M2","M3"), **kwargs):
        """Main pipeline execution flow."""
        with self.pool or contextlib.nullcontext(self.pool) as pool:
            # --- FIX: Use the filtered kwargs to instantiate BurstDataset ---
            self.dataset = BurstDataset(self.path, **self.dataset_kwargs)
            self.dataset.model.dm_init = self.dm_init

            n_steps = self.pipeline_kwargs.get('steps', 2000)

            init_guess = self._get_initial_guess(self.dataset.model)

            if model_scan:
                log.info("Starting model selection scan (BIC)...")
                best_key, all_res = fit_models_bic(
                    model=self.dataset.model, init=init_guess, n_steps=n_steps//2, pool=pool, model_keys = model_keys,
                )
                sampler = all_res[best_key][0]
            else:
                best_key = "M3"; log.info(f"Fitting model {best_key} directly...")
                # right before sampling
                priors = build_priors(init_guess, scale=6.0)
                # give generous log-uniform‐ish bounds to the two broadening params
                priors["tau_1ghz"] = (1e-4, 5e4)   # ms
                priors["zeta"]     = (1e-4, 5e4)   # ms
                fitter = FRBFitter(self.dataset.model, priors, n_steps=n_steps, pool=pool)
                sampler = fitter.sample(init_guess, model_key=best_key)

            log.info("Processing MCMC chains...")
            burn, thin = auto_burn_thin(sampler)
            flat_chain = sampler.get_chain(discard=burn, thin=thin, flat=True)
            if flat_chain.shape[0] == 0:
                raise RuntimeError("MCMC chain is empty after burn-in and thinning. Check sampler settings or increase n_steps.")

            best_params = FRBParams.from_sequence(flat_chain[np.argmax(sampler.get_log_prob(discard=burn, thin=thin, flat=True))], best_key)

            results = {
                "best_key": best_key, "best_params": best_params, "sampler": sampler, 
                "flat_chain": flat_chain, "param_names": FRBFitter._ORDER[best_key], 
                "dm_init": self.dm_init, "model_instance": self.dataset.model, 
                "chain_stats": {"burn_in": burn, "thin": thin}
            }

            if diagnostics:
                diag_runner = BurstDiagnostics(self.dataset, results)
                results['diagnostics'] = diag_runner.run_all(sb_steps=n_steps//4, pool=pool)

            results['goodness_of_fit'] = goodness_of_fit(
                self.dataset.data, self.dataset.model(best_params, best_key), 
                self.dataset.model.noise_std, len(results['param_names'])
            )
            log.info(f"Best model: {best_key} | χ²/dof = {results['goodness_of_fit']['chi2_reduced']:.2f}")

            if plot:
                p_path_sixt = self.path.with_name(f"{self.path.stem}_diagnostics.pdf")
                create_sixteen_panel_plot(self.dataset, results, output_path=p_path_sixt, show=show)
                p_path_four = self.path.with_name(f"{self.path.stem}_fullmodel.pdf")
                create_four_panel_plot(self.dataset, results, output_path=p_path_four, show=show)

            return results

    def _get_initial_guess(self, model: "FRBModel") -> "FRBParams":
        log.info("Finding initial guess for MCMC...")
        prof = np.nansum(model.data, axis=0)
        if np.all(prof == 0): return FRBParams(c0=0, t0=model.time.mean(), gamma=0, zeta=0, tau_1ghz=0)
        rough_guess = FRBParams(c0=np.sum(prof), t0=model.time[np.argmax(prof)], gamma=-1.6, zeta=0.1, tau_1ghz=0.1)
        priors = build_priors(rough_guess, scale=1.5); model_key = "M3"
        x0 = rough_guess.to_sequence(model_key); bounds = [priors[n] for n in FRBFitter._ORDER[model_key]]
        def nll(theta):
            p = FRBParams.from_sequence(theta, model_key); ll = model.log_likelihood(p, model_key)
            return -ll if np.isfinite(ll) else np.inf
        res = minimize(nll, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-7})
        if not res.success: warnings.warn("Initial guess optimization failed. Using rough guess."); return rough_guess
        log.info("Refined initial guess found via optimization.")
        return FRBParams.from_sequence(res.x, model_key)

def auto_burn_thin(sampler, safety_factor_burn=3.0, safety_factor_thin=0.5):
    try:
        tau = sampler.get_autocorr_time(tol=0.01)
        burn = int(safety_factor_burn * np.nanmax(tau)); thin = max(1, int(safety_factor_thin * np.nanmin(tau)))
        burn = min(burn, sampler.iteration // 2)
        log.info(f"Auto-determined burn-in: {burn}, thinning: {thin}")
        return burn, thin
    except Exception as e:
        warnings.warn(f"Could not estimate autocorr time: {e}. Using defaults.")
        return sampler.iteration // 4, 1

###############################################################################
# 4. CLI WRAPPER
###############################################################################
def _main():
    p = argparse.ArgumentParser(description="Run BurstFit pipeline on a .npy file.")
    # Add all possible arguments here
    p.add_argument("path", type=Path, help="Input .npy file")
    p.add_argument("--dm_init", type=float, default=0.0)
    p.add_argument("--telescope", default="CHIME")
    p.add_argument("--telcfg", default="telescopes.yaml")
    p.add_argument("--sampcfg", default="sampler.yaml")
    p.add_argument("--nproc", type=int, default=None)
    p.add_argument("--yes", action="store_true", help="Bypass pool confirmation")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--f_factor", type=int, default=1)
    p.add_argument("--t_factor", type=int, default=1)
    # Add flags for boolean pipeline controls
    p.add_argument("--no-scan", dest='model_scan', action='store_false')
    p.add_argument("--no-diag", dest='diagnostics', action='store_false')
    p.add_argument("--no-plot", dest='plot', action='store_false')
    p.set_defaults(model_scan=True, diagnostics=True, plot=True)
    args = p.parse_args()
    
    # --- FIX: Pass all arguments as a dict to the pipeline constructor ---
    # The new __init__ will sort them out automatically.
    pipeline_kwargs = vars(args)
    
    pipe = BurstPipeline(
        path=pipeline_kwargs.pop('path'), # path is a positional arg
        dm_init=pipeline_kwargs.pop('dm_init'), # dm_init is also explicit
        **pipeline_kwargs
    )
    
    pipe.run_full(
        model_scan=args.model_scan, 
        diagnostics=args.diagnostics, 
        plot=args.plot
    )

if __name__ == "__main__":
    _main()