"""
burstfit_pipeline.py
====================

Object-oriented orchestrator for the BurstFit pipeline.
"""
from __future__ import annotations

import logging
import warnings
import argparse
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

# ## REFACTOR ##: All physics, fitting, and primary diagnostics are now imported
# from the other modules. The pipeline file is purely for orchestration.
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

log = logging.getLogger("burstfit.pipeline")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

plt_txt_font = 18
legend_font = 16
title_font = 24

###############################################################################
# 0. Plotting Functions
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
    
    # --- Unpack results ---
    best_p = results["best_params"]
    best_key = results["best_key"]
    model_instance = results["model_instance"]
    
    data = dataset.data
    time = dataset.time
    freq = dataset.freq
    time_centered = time - (time[0] + (time[-1] - time[0]) / 2)
    extent = [time_centered[0], time_centered[-1], freq[0], freq[-1]]

    clean_model = model_instance(best_p, best_key)
    synthetic_noise = np.random.normal(0.0, model_instance.noise_std[:, None], size=data.shape)
    synthetic_data = clean_model + synthetic_noise
    residual = data - synthetic_data

    # --- Normalization ---
    def _normalize_panel_data(arr_2d, off_pulse_data):
        mean_off = np.nanmean(off_pulse_data)
        std_off = np.nanstd(off_pulse_data)
        if std_off < 1e-9: return arr_2d # Avoid division by zero
        arr_norm = (arr_2d - mean_off) / std_off
        return arr_norm / np.nanmax(arr_norm) if np.nanmax(arr_norm) > 0 else arr_norm

    q = data.shape[1] // 4
    off_pulse_indices = np.r_[0:q, -q:0]
    data_off_pulse = data[:, off_pulse_indices]
    
    data_norm = _normalize_panel_data(data, data_off_pulse)
    model_norm = _normalize_panel_data(clean_model, data_off_pulse)
    synthetic_norm = _normalize_panel_data(synthetic_data, data_off_pulse)

    # --- Plotting ---
    fig, axes = plt.subplots(nrows=2, ncols=8,
        gridspec_kw={'height_ratios': [1, 2.5], 'width_ratios': [2, 0.5] * 4},
        figsize=(24, 8)
    )
    plt.style.use('default')
    
    panel_data = [
        (data_norm, 'data', r'I$_{\mathrm{data}}$'),
        (model_norm, 'model', r'I$_{\mathrm{model}}$'),
        (synthetic_norm, 'synth', r'I$_{\mathrm{model+noise}}$'),
        (residual, 'residual', r'I$_{\mathrm{residual}}$'),
    ]

    for i, (panel_ds, title, label) in enumerate(panel_data):
        col_idx = i * 2
        ax_ts, ax_sp, ax_wf = axes[0, col_idx], axes[1, col_idx + 1], axes[1, col_idx]

        ts = np.nansum(panel_ds, axis=0)
        sp = np.nansum(panel_ds, axis=1)

        # Time series
        ax_ts.step(time_centered, ts, where='mid', c='k', lw=1.5, label=label)
        ax_ts.legend(loc='upper right', fontsize=14, frameon=False)
        ax_ts.set_yticks([])
        ax_ts.tick_params(axis='x', labelbottom=False)

        # Waterfall
        cmap = 'coolwarm' if title == 'residual' else 'plasma'
        vmax = np.nanpercentile(panel_ds, 99.5) if title != 'residual' else np.nanstd(panel_ds) * 3
        vmin = np.nanpercentile(panel_ds, 1) if title != 'residual' else -vmax
        ax_wf.imshow(panel_ds, extent=extent, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
        ax_wf.set_xlabel('Time [ms]')
        if i == 0: ax_wf.set_ylabel('Frequency [GHz]')
        else: ax_wf.tick_params(axis='y', labelleft=False)

        # Spectrum
        ax_sp.step(sp, freq, where='mid', c='k', lw=1.5)
        ax_sp.set_xticks([])
        ax_sp.tick_params(axis='y', labelleft=False)
        axes[0, col_idx + 1].axis('off')

    plt.subplots_adjust(hspace=0.05, wspace=0.05)

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
    """
    Creates a comprehensive 16-panel diagnostic plot summarizing the fit.
    """
    log.info("Generating 16-panel comprehensive diagnostics plot...")

    # --- Unpack all necessary data from the results dictionary ---
    best_key = results["best_key"]
    best_p = results["best_params"]
    sampler = results["sampler"]
    gof = results.get("goodness_of_fit")
    chain_stats = results.get("chain_stats", {})
    flat_chain = results["flat_chain"]
    p_names = results["p_names"]
    diag_results = results.get("diagnostics", {})
    model_instance = results["model_instance"]
    
    model_dyn = model_instance(best_p, best_key)
    residual = dataset.data - model_dyn

    # --- Setup Figure ---
    fig, axes = plt.subplots(4, 4, figsize=(20, 20), constrained_layout=True)
    ax = axes.ravel()

    # Use a flattened array of axes for easy sequential access
    ax = axes.ravel()

    # --- 3. Populate Panels Systematically ---

    # Panel 0: Data Dynamic Spectrum
    vmin = np.percentile(dataset.data, 1)
    vmax = np.percentile(dataset.data, 99)
    plot_dynamic(ax[0], dataset.data, dataset.time, dataset.freq, cmap="plasma", vmin=vmin, vmax=vmax)
    ax[0].set_title("Data")

    # Panel 1: Model Dynamic Spectrum
    plot_dynamic(ax[1], model_dyn, dataset.time, dataset.freq, cmap="plasma", vmin=vmin, vmax=vmax)
    ax[1].set_title(f"Model ({best_key})")

    # Panel 2: Residual Dynamic Spectrum
    res_std = np.std(residual)
    plot_dynamic(ax[2], residual, dataset.time, dataset.freq, cmap="coolwarm", vmin=-3*res_std, vmax=3*res_std)
    ax[2].set_title("Residual")

    # Panel 3: Residual Histogram
    residual_normalized = residual / noise_std[:, None]
    ax[3].hist(residual_normalized.flatten(), bins=100, density=True, color='gray', label='Residuals')
    x_norm = np.linspace(-4, 4, 100)
    ax[3].plot(x_norm, np.exp(-0.5*x_norm**2)/np.sqrt(2*np.pi), 'm-', lw=2, label='N(0,1)')
    ax[3].set_title('Residual Distribution')
    ax[3].legend(fontsize=legend_font)

    # Panel 4: Time Profile (Data vs. Model)
    ax[4].plot(dataset.time, np.sum(dataset.data, axis=0), 'k-', label='Data')
    ax[4].plot(dataset.time, np.sum(model_dyn, axis=0), 'm--', lw=2, label='Model')
    ax[4].fill_between(dataset.time, np.sum(residual, axis=0), color='gray', alpha=0.5, label='Residual')
    ax[4].set_title('Time Profile')
    ax[4].legend(fontsize=legend_font)
    ax[4].set_xlabel('Time [ms]')
    ax[4].set_ylabel('Flux')


    # Panel 5: Frequency Spectrum (Data vs. Model)
    ax[5].plot(dataset.freq, np.sum(dataset.data, axis=1), 'k-', label='Data')
    ax[5].plot(dataset.freq, np.sum(model_dyn, axis=1), 'm--', lw=2, label='Model')
    ax[5].set_title('Frequency Spectrum')
    ax[5].legend(fontsize=legend_font)
    ax[5].set_xlabel('Frequency [GHz]')


    # Panel 6: Channel Influence Diagnostic
    if diag and diag.influence_results is not None:
        plot_influence(ax[6], diag.influence_results, dataset.freq)
        ax[6].set_title('Channel Influence (Δχ²)')
    else:
        ax[6].text(0.5, 0.5, 'Influence Diagnostic Not Run', ha='center', va='center')
        ax[6].set_axis_off()

    # Panel 7: 2D Sub-band Consistency
    if diag and diag.subband_results is not None:
        p_name, s_res, s_chains = diag.subband_results
        if p_name:
            valid = [(i, v, e) for i, (v, e) in enumerate(s_res) if np.isfinite(v) and e > 0]
            if valid:
                idx, vals, errs = zip(*valid)
                n_bands = len(s_res)
                band_ctr = (dataset.freq[0] + (dataset.freq[-1] - dataset.freq[0]) * (np.array(idx) + 0.5) / n_bands)
                ax[7].errorbar(band_ctr, vals, yerr=errs, fmt="o", capsize=5, ms=8, color="k", label=f"{p_name} per sub-band")
                global_val = getattr(best_p, p_name)
                ax[7].axhline(global_val, color="m", ls="--", label=f"Global Fit: {global_val:.3f}")
                ax[7].set_ylabel(p_name)
                ax[7].legend(fontsize=legend_font)
        else:
            ax[7].text(0.5, 0.5, "Model M0\n(no broadening)", ha="center", va="center")
        ax[7].set_title("2-D Sub-Band Re-Fit")
        ax[7].set_xlabel("Frequency [GHz]")
    else:
        ax[7].text(0.5, 0.5, '2D Sub-band Diagnostic Not Run', ha='center', va='center')
        ax[7].set_axis_off()

    # Panel 8: 1D Sub-band Profile Fits
    if diag_results.get("profile1d") is not None: plot_subband_profiles(ax[8], *diag_results["profile1d"], best_p)
    else: ax[8].text(0.5, 0.5, '1D Profile Check Not Run', **{'ha':'center', 'va':'center'}); ax[8].set_axis_off()

    # Panel 9-10: MCMC chains
    chain = sampler.get_chain(); burn = chain_stats.get("burn_in", 0)
    ax[9].plot(chain[:, ::5, 0], 'k', alpha=0.3); ax[9].axvline(burn, color='m', ls='--'); ax[9].set_title(f'Trace: {param_names[0]}'); ax[9].set_xlabel('Step')
    if len(param_names) > 1:
        corr = np.corrcoef(flat_chain.T); np.fill_diagonal(corr, 0); i, j = np.unravel_index(np.abs(corr).argmax(), corr.shape)
        ax[10].scatter(flat_chain[:,i], flat_chain[:,j], s=1, alpha=0.1, c='k'); ax[10].set_xlabel(param_names[i]); ax[10].set_ylabel(param_names[j]); ax[10].set_title(f'Highest Correlation (ρ={corr[i,j]:.2f})')
    
    # Panel 11-12: ACF and DM Check
    if gof: ax[11].plot(np.arange(len(gof['residual_autocorr'])), gof['residual_autocorr'], 'k-'); ax[11].set_title('Residual ACF');
    if diag_results.get("dm_check") is not None: dms, snrs = diag_results['dm_check']; ax[12].plot(dms, snrs, 'o-k'); ax[12].set_title('DM Optimization'); ax[12].set_xlabel(r'ΔDM (pc cm$^{-3}$)')

    # Panels 13-15: Text Summaries
    for i in [13, 14, 15]: ax[i].set_axis_off()
    if gof: ax[13].text(0, 0.95, f"GoF:\nχ²/dof = {gof['chi2_reduced']:.2f}", va='top', fontfamily='monospace')
    p_summary = "Best Fit:\n" + "\n".join([f"{n}: {getattr(best_p, n):.3f}" for n in param_names])
    ax[14].text(0, 0.95, p_summary, va='top', fontfamily='monospace')
    ax[15].text(0, 0.95, f"File:\n{dataset.path.name}", va='top', fontfamily='monospace')
    
    if output_path: 
        log.info(f"Saving 16-panel plot to {output_path}")
        fig.savefig(output_path, bbox_inches='tight', dpi=150)
    if show: plt.show()
    else: plt.close(fig)
    return fig

###############################################################################
# 1. Dataset loader
###############################################################################

class BurstDataset:
    """Load *and* preprocess a burst cut‑out stored as a 2‑D ``.npy``.

    The preprocessing pipeline is split into discrete helpers so each
    step is unit‑testable and can be swapped out (e.g. different
    band‑pass correction).  The full load runs automatically unless
    ``lazy=True``.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        path: str | Path,
        *,
        telescope: str = "DSA-110",
        telcfg_path: str | Path = "telescopes.yaml",
        sampcfg_path: str | Path = "sampler.yaml",
        f_factor: int = 1,
        t_factor: int = 1,
        outer_trim: float = 0.45,
        smooth_ms: float = 0.5,
        center_burst: bool = True,
        flip_freq: bool = True, # flip if raw stored high→low
        off_idx: slice | Sequence[int] = slice(0, 1000),
        lazy: bool = False,
    ) -> None:
        self.path = Path(path)
        self.telparams = load_telescope_block(telcfg_path, telescope=telescope)
        self.telname = self.telparams.name
        self.sampparams = load_sampler_block(sampcfg_path)
        self.sampname = self.sampparams.name
        assert 0.0 <= outer_trim < 0.5, "outer_trim must be < 0.5"
        self.f_factor = f_factor
        self.t_factor = t_factor
        self.outer_trim = outer_trim
        self.smooth_ms = smooth_ms   # FWHM of Gaussian in *ms*
        self.center_burst = center_burst
        self.flip_freq = flip_freq
        self.off_idx = off_idx

        # will be populated by _load()
        self.data: NDArray[np.floating] | None = None
        self.freq: NDArray[np.floating] | None = None
        self.time: NDArray[np.floating] | None = None
        self.df_MHz: float | None = None
        self.dt_ms: float | None = None

        if not lazy:
            self.load()

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------
    def load(self):
        """Run the full preprocessing chain (idempotent)."""
        if self.data is not None:
            return  # already loaded
        raw = self._load_raw()
        if self.flip_freq:
            raw = np.flipud(raw)
        ds = self._bandpass_correct(raw)
        ds = self._trim_buffer(ds)
        ds = self._downsample(ds)
        ds = self._normalise(ds)
        self.data = ds
        self.freq, self.time = self._build_axes(ds)
        if self.center_burst:
            self._centre_burst()
        self.model = FRBModel(
            time=self.time,
            freq=self.freq,
            data=self.data,
            dm_init=0.0, # Placeholder, will be set in pipeline
            df_MHz=self.df_MHz
        )
            
    def validate(self) -> Dict[str, Any]:
        """Validate loaded data and return diagnostics."""
        if self.data is None:
            self.load()
        
        diagnostics = {
            'shape': self.data.shape,
            'n_dead_channels': np.sum(np.all(self.data == 0, axis=1)),
            'n_nan_pixels': np.sum(np.isnan(self.data)),
            'snr_estimate': self._estimate_snr(),
            'freq_range': (self.freq.min(), self.freq.max()),
            'time_span': self.time[-1] - self.time[0],
            'bandwidth': (self.freq[-1] - self.freq[0]) * 1000,  # MHz
        }
        
        # Check for common issues
        if diagnostics['n_dead_channels'] > 0.5 * self.data.shape[0]:
            warnings.warn("More than 50% of channels appear dead!")
        
        if diagnostics['snr_estimate'] < 5:
            warnings.warn("Low S/N detected - results may be unreliable")
        
        return diagnostics
        
    def quicklook(self, title: str = "Dynamic spectrum"):
        if self.data is None:
            self.load()
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_dynamic(ax, self.data, self.time, self.freq, cmap="plasma")
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Frequency [GHz]")
        ax.set_title(title)
        plt.tight_layout()

    # ------------------------------------------------------------------
    # pipeline helpers (private)
    # ------------------------------------------------------------------
    def _load_raw(self):
        """Load raw data with comprehensive error checking."""
        if not self.path.exists():
            raise FileNotFoundError(f"Data file not found: {self.path}")

        try:
            data = np.load(self.path)
            if data.ndim != 2:
                raise ValueError(f"Expected 2D array, got shape {data.shape}")
            return np.nan_to_num(data.astype(float))
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {self.path}: {e}")

    def _estimate_snr(self) -> float:
        """Estimate burst S/N."""
        profile = np.sum(self.data, axis=0)
        noise = np.std(profile[:len(profile)//4])
        signal = np.max(profile) - np.median(profile)
        return signal / noise if noise > 0 else 0.0
            
    def _bandpass_correct(self, arr):
        """Per‑channel z‑score with dead‑channel protection.

        Channels whose off‑pulse variance is exactly zero (all NaNs or
        a hardware flag of zeros) would yield division‑by‑zero → ``±inf``
        after normalisation.  Here we set ``σ=NaN`` for those channels,
        carry the NaNs through the division, then convert to zeros so
        downstream averaging keeps them dark instead of blinding white.
        """
        if isinstance(self.off_idx, slice):
            idx = np.arange(arr.shape[1])[self.off_idx]
        else:
            idx = np.asarray(self.off_idx)

        mu = np.nanmean(arr[:, idx], axis=1, keepdims=True)
        sig = np.nanstd(arr[:, idx], axis=1, keepdims=True)
        sig[sig == 0] = np.nan  # mark zero‑variance (dead) channels

        arr_bp = (arr - mu) / sig
        return np.nan_to_num(arr_bp, nan=0.0, posinf=0.0, neginf=0.0)

    def _trim_buffer(self, arr):
        n_left = int(self.outer_trim * arr.shape[1])
        return arr[:, n_left:-n_left]

    def _downsample(self, arr):
        return downsample(arr, self.f_factor, self.t_factor)

    def _normalise(self, arr):
        # Adjust off_idx for downsampling
        if isinstance(self.off_idx, slice):
            start = self.off_idx.start // self.t_factor if self.off_idx.start else 0
            stop = self.off_idx.stop // self.t_factor if self.off_idx.stop else arr.shape[1]
            idx = np.arange(arr.shape[1])[start:stop]
        else:
            idx = np.asarray(self.off_idx) // self.t_factor

        # Ensure indices are within bounds
        idx = idx[idx < arr.shape[1]]

        off = arr[:, idx]
        arr = (arr - np.nanmean(off)) / np.nanstd(off)
        return arr / np.nanmax(arr)
    
    def _centre_burst(self):
        """
        Roll dynamic spectrum so that the *smoothed* burst envelope peaks at
        the centre of the time axis.
        """
        # 1) burst profile
        prof = self.data.sum(axis=0)

        # 2) Gaussian smooth  (convert FWHM ms -> sigma in samples)
        sigma  = (self.smooth_ms / (2.355 * self.dt_ms))
        kernel = np.exp(-0.5*((np.arange(-4*sigma, 4*sigma+1))/sigma)**2)
        kernel /= kernel.sum()
        prof_s = np.convolve(prof, kernel, mode="same")

        # 3) roll so peak -> centre
        t_peak = np.argmax(prof_s)
        mid    = self.data.shape[1] // 2
        shift  = mid - t_peak
        self.data = np.roll(self.data, shift, axis=1)
        self.time = self.time + shift * self.dt_ms
        
        

    def _build_axes(self, ds):
        p = self.telparams
        self.df_MHz = p.df_MHz_raw * self.f_factor
        self.dt_ms = p.dt_ms_raw * self.t_factor
        freq = np.linspace(p.f_min_GHz, p.f_max_GHz, ds.shape[0])
        time = np.arange(ds.shape[1]) * self.dt_ms
        return freq, time

###############################################################################
# 2. Fitting layer
###############################################################################

class BurstFitter:
    """Fit an FRB pulse model with emcee, dynesty, … depending on YAML."""

    def __init__(
        self,
        dataset: "BurstDataset",
        *,
        dm_init: float = 0.0,
        n_steps: int = 2000,
        pool=None,
        sampcfg_path: str | Path = "sampler.yaml",
        sampler_name: str | None = None,
    ) -> None:

        # ---- read the sampler block -----------------------------------
        engine, params = load_sampler_block(sampcfg_path, sampler_name)
        self._engine  = engine          # 'emcee', 'dynesty', …
        self._params  = params          # keep full dict for later

        # optional public copies (drop if you don’t need them)
        self.sampname    = engine
        self.sampparams  = params

        # ---- knobs with defaults --------------------------------------
        self.n_walkers_mult = int(params.get("n_walkers_mult", 8))
        self.n_steps_warm   = int(params.get("n_steps_warm",   500))
        self.thin_warm      = int(params.get("thin_warm",      4))
        self.n_steps        = n_steps
        self.pool           = pool

        # ---- data & forward model -------------------------------------
        self.ds = dataset
        self.model = FRBModel(
            time=dataset.time,
            freq=dataset.freq,
            data=dataset.data,
            dm_init=dm_init,
        )
        
    @staticmethod
    def _estimate_smear_scatter(time, prof, freqs,
                                beta: float = -4.0,
                                smooth_sigma: float = 1.0):
        """Return zeta_est, tau_1GHz_est."""
        if smooth_sigma and smooth_sigma > 0:
            prof_sm = gaussian_filter1d(prof, sigma=smooth_sigma)
        else:
            prof_sm = prof

        i0   = np.argmax(prof_sm)
        half = prof_sm[i0] / 2
        left  = np.where(prof_sm[:i0] < half)[0]
        right = np.where(prof_sm[i0:] < half)[0]

        if len(left) and len(right):
            zeta_est = time[i0 + right[0]] - time[left[-1]]
        else:
            zeta_est = 0.1 * (time[-1] - time[0])   # fallback

        tail_end  = min(i0 + int(0.5 * len(prof_sm)), len(prof_sm))
        prof_tail = prof_sm[i0:tail_end]
        mask      = prof_tail > 0
        t_tail    = time[i0:tail_end][mask]

        if len(t_tail) >= 2:
            y_tail = np.log(prof_tail[mask])
            slope, _ = np.polyfit(t_tail, y_tail, 1)
            tau_obs = abs(1 / slope) if slope else zeta_est
        else:
            tau_obs = zeta_est

        nu_med = np.median(freqs)
        tau_1ghz = tau_obs * (nu_med / 1.0) ** beta
        return float(zeta_est), float(tau_1ghz)


    def _guess(self) -> FRBParams:
        """Estimate initial parameters with improved robustness."""
        # Ensure data is loaded
        if self.ds.data is None:
            self.ds.load()

        # Amplitude: peak of frequency-summed profile
        prof = np.nansum(self.ds.data, axis=0)
        c0 = float(np.nanmax(prof))

        # Time: use weighted centroid for robustness
        prof_positive = np.maximum(prof, 0)
        if np.sum(prof_positive) > 0:
            t0 = float(np.sum(self.ds.time * prof_positive) / np.sum(prof_positive))
        else:
            t0 = float(self.ds.time[np.argmax(prof)])

        # Spectral index: fit to frequency-collapsed spectrum
        freq_profile = np.nansum(self.ds.data, axis=1)
        valid_freq = freq_profile > 0.1 * np.nanmax(freq_profile)

        if np.sum(valid_freq) >= 3:
            # Fit log(S) = gamma * log(f) + C
            log_freq = np.log10(self.ds.freq[valid_freq])
            log_flux = np.log10(freq_profile[valid_freq])

            # Weighted fit to reduce noise impact
            weights = freq_profile[valid_freq] / np.nanmax(freq_profile[valid_freq])
            coeffs = np.polyfit(log_freq, log_flux, 1, w=weights)
            gamma = float(coeffs[0])
        else:
            gamma = -1.0  # Default fallback

        # Smearing and scattering estimates
        zeta_est, tau1ghz_est = self._estimate_smear_scatter(
            self.ds.time, prof, self.ds.freq
        )

        return FRBParams(
            c0=c0,
            t0=t0,
            gamma=gamma,
            zeta=zeta_est,
            tau_1ghz=tau1ghz_est
        )

    def _adapt_initial_guess(self, model_key: str = "M3") -> FRBParams:
        """Optimize initial guess using bounded optimization."""

        # Get initial guess
        p0 = self._guess()
        priors = build_priors(p0, scale=2.0)  # Tighter bounds for optimization

        # Get parameter names for this model
        p_names = {
            "M0": ("c0", "t0", "gamma"),
            "M1": ("c0", "t0", "gamma", "zeta"),
            "M2": ("c0", "t0", "gamma", "tau_1ghz"),
            "M3": ("c0", "t0", "gamma", "zeta", "tau_1ghz"),
        }[model_key]

        # Pack initial values
        x0 = np.array([getattr(p0, name) for name in p_names])

        # Set bounds
        bounds = [(priors[name][0], priors[name][1]) for name in p_names]

        def _neg_log_posterior(x):
            # Unpack parameters
            param_dict = {name: val for name, val in zip(p_names, x)}
            # Fill in defaults for unused parameters
            for name in ["c0", "t0", "gamma", "zeta", "tau_1ghz"]:
                if name not in param_dict:
                    param_dict[name] = 0.0

            params = FRBParams(**param_dict)

            # Check priors
            for name, val in zip(p_names, x):
                lo, hi = priors[name]
                if not (lo <= val <= hi):
                    return 1e10  # Large penalty

            # Compute likelihood
            try:
                ll = self.model.log_likelihood(params, model_key)
                return -ll  # Minimize negative log likelihood
            except:
                return 1e10

        # Optimize with bounds
        result = minimize(
            _neg_log_posterior,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )

        if not result.success:
            warnings.warn(f"Initial guess optimization failed: {result.message}")

        # Return optimized parameters
        return FRBParams.from_sequence(result.x, model_key)
    
    def fit(self, model_key="M3"):
        
        def _stateless_args(model, priors, order, key):
            return (model.time, model.freq, model.data,
                    model.dm_init, model.noise_std,
                    priors, order, key)
        
        names = FRBFitter(self.model, None, n_steps=0)._ORDER[model_key]
        ndim  = len(names)
        
        if self._engine == "emcee":
            
            # Setup
            names = FRBFitter._ORDER[model_key]
            ndim  = len(names)
            nwalk = max(self.n_walkers_mult * ndim, 2 * ndim)
            
            # Adaptive initial guess
            p0 = self._adapt_initial_guess(model_key)

            # Warm-up sampler
            warm_sampler = emcee.EnsembleSampler(
                nwalk, ndim, _log_prob_stateless,
                args=_stateless_args(self.model,
                                     build_priors(p0, 3.0),
                                     FRBFitter._ORDER,
                                     model_key),
                pool=self.pool,
            )
            
            # initialise walkers around p0
            p0_warm = np.array([p0.to_sequence(model_key)
                    + 1e-4*np.random.randn(ndim)
                    for _ in range(nwalk)])
            warm_sampler.run_mcmc(p0_warm, self.n_steps_warm,
                                  thin_by=self.thin_warm, progress=True)

            # Estimate covariance & reinitialize (with jitter to avoid singular cov)
            warm   = warm_sampler.get_chain(flat=True, thin=self.thin_warm)
            mean   = warm.mean(axis=0)
            cov    = np.cov(warm, rowvar=False)
            cov   += np.eye(ndim) * 1e-6 * np.trace(cov)/ndim
            p0_ens = np.random.multivariate_normal(mean, cov, size=nwalk)
            
            nwalk = max(self.n_walkers_mult * ndim, 2 * ndim)
            sampler = emcee.EnsembleSampler(
                nwalk, ndim, _log_prob_stateless,
                args=_stateless_args(self.model,
                                     build_priors(p0, 3.0),
                                     FRBFitter._ORDER,
                                     model_key),
                pool=self.pool,
            )
            sampler.run_mcmc(p0_ens, self.n_steps, progress=True)
            return sampler

        elif self._engine == "dynesty":
            raise NotImplementedError("Dynesty backend not wired yet.")
        else:
            raise ValueError(f"Sampler '{self._engine}' not implemented yet.")

        

###############################################################################
# 3. Diagnostic layer
###############################################################################

class BurstDiagnostics:
    def __init__(self, dataset: BurstDataset, best_p: FRBParams, dm_init: float = 0.0):
        if dataset.data is None:
            dataset.load()
        self.ds = dataset
        self.best_p = best_p
        self.dm_init = dm_init
        self._subband_results = None
        self._influence_results = None
        self._subband_profile_results = None

    def subband(self, best_key: str, sb_steps: int):
        par_name, res, chains = subband_consistency(
        self.ds.data,          
        self.ds.freq,          
        self.ds.time,          
        self.dm_init,          # residual DM used in the global fit
        self.best_p,           # FRBParams from the global best fit
        model_key=best_key,    # ensure slice re-fit uses the same model
        n_steps=sb_steps,
        # can also add n_sub=4, n_steps=600, pool=self.ds.pool if desired
        )
        self._subband_results = res        # list[(mean, σ) per slice]
        self._subband_param   = par_name   # 'tau_1ghz', 'zeta', or None
        self._subband_chains = chains
        return par_name, res

    def influence(self, best_key: str, plot=False):
        delta = leave_one_out_influence(
            self.ds.data, self.ds.freq, self.ds.time, 
            self.dm_init, self.best_p, model_key=best_key
        )
        self._influence_results = delta
        if plot:
            fig, ax = plt.subplots(figsize=(8, 3))
            plot_influence(ax, delta, self.ds.freq)
        return delta

    def profile1d(self, n_sub: int = 4):
        """Return (ν_centres, τ̂, σ_τ) from fast 1-D fits."""
        centres, tau1d, tau1d_err = fit_subband_profiles(
            self.ds, self.best_p, dm_init=self.dm_init, n_sub=n_sub
        )
        self._subband_profile_results = (centres, tau1d, tau1d_err)
        return self._subband_profile_results
    
###############################################################################
# 4. Pipeline façade
###############################################################################

class BurstPipeline:
    def __init__(
        self,
        path: str | Path,
        *,
        dm_init: float = 0.0,
        telescope: str = "DSA-110",
        telcfg_path: str | Path = "telescopes.yaml",
        sampcfg_path: str | Path = "sampler.yaml",
        sampler_name: str | None = None,
        n_steps: int = 2000,
        sb_steps: int = 1000,
        f_factor: int = 1,
        t_factor: int = 1,
        outer_trim: float = 0.45,
        center_burst: bool = True,   # default
        smooth_ms: float = 0.5,  # widen for very noisy data
        pool=None,
    ) -> None:
        
        self.dataset = BurstDataset(
            path,
            telescope=telescope,
            telcfg_path=telcfg_path,
            sampcfg_path=sampcfg_path,
            f_factor=f_factor,
            t_factor=t_factor,
            outer_trim=outer_trim,
            center_burst=center_burst,
            smooth_ms=smooth_ms,
        )
        
        self.sb_steps      = sb_steps
        
        self.dm_init        = dm_init
        
        self.pool           = pool
        self.telcfg_path    = telcfg_path
        self.sampcfg_path   = sampcfg_path
        
        self.fitter = FRBFitter(
            self.dataset.model, # Pass the FRBModel instance from the dataset
            {}, # Priors will be built on the fly
            n_steps=n_steps,
            pool=self.pool,
        )

    def run_full(self, *, dm_init=None, sb_steps=None, model_scan=True, diagnostics=True, plot=False, show=False, verbose=True):
        """
        Run the full burst fitting pipeline.
        
        Parameters
        ----------
        model_scan : bool
            Whether to scan through models M0-M3 using BIC
        diagnostics : bool
            Whether to run diagnostic checks (subband, influence)
        plot : bool
            Whether to create plots
        verbose : bool
            Whether to print detailed diagnostic information
        """
        # Ensure data is loaded
        if self.dataset.data is None:
            self.dataset.load()
            
        # Pick whichever DM was passed in, else the one stored during __init__
        if dm_init is None:
            self.dataset.model.dm_init = self.dm_init
            
        n_steps = self.kwargs.get('n_steps', 2000)
        
        # --- Model Selection & Fitting ---
        log.info("Finding initial guess...")
        # Initial guess needs a fitter instance, but we create a lightweight one
        temp_fitter = FRBFitter(self.dataset.model, {})
        init_guess = temp_fitter._adapt_initial_guess() # Need to implement this
        
        # Model selection or direct fit
        if model_scan:
            log.info("Starting model selection scan (BIC)...")
            init_guess = self.fitter._adapt_initial_guess()
            best_key, all_results = fit_models_bic(
                model=self.dataset.model,
                freq=self.dataset.freq,
                time=self.dataset.time,
                init=init_guess,
                n_steps=n_steps // 2,
                pool=self.pool
            )
            sampler = all_results[best_key][0]
        else:
            best_key = "M3"
            log.info(f"Skipping scan, fitting model {best_key}...")
            priors = build_priors(init_guess, scale=3.0)
            fitter = FRBFitter(self.dataset.model, priors, n_steps=n_steps, pool=self.pool)
            sampler = fitter.sample(init_guess, best_key)
            
        if model_scan:
            if verbose:
                print("\n=== Model Selection (BIC) ===")
            init = self.fitter._adapt_initial_guess()
            best_key, res = fit_models_bic(
                data=self.dataset.data,
                freq=self.dataset.freq,
                time=self.dataset.time,
                dm_init=dm_init,
                init=init,  
                n_steps=self.fitter.n_steps // 2,
                pool=self.pool
            )
            sampler = res[best_key][0]
            n_steps_used = self.fitter.n_steps // 2
        else:
            best_key = "M3"
            if verbose:
                print(f"\n=== Fitting Model {best_key} ===")
            sampler = self.fitter.fit(best_key)
            n_steps_used = self.fitter.n_steps

        # Auto burn-in and thinning
        def auto_burn_thin(sampler, safety_factor_burn=2.0, safety_factor_thin=0.5):
            """Estimate burn-in and thinning; fall back gracefully on NaNs."""
            try:
                tau = sampler.get_autocorr_time(tol=0)
                if np.any(~np.isfinite(tau)):          # catches NaN or inf
                    raise ValueError("NaN/infinite τ")
            except Exception as e:
                warnings.warn(f"Could not estimate autocorr time: {e}; "
                              "using defaults (burn=iter/4, thin=1).")
                return sampler.iteration // 4, 1

            burn = int(safety_factor_burn * np.max(tau))
            thin = max(1, int(safety_factor_thin * np.min(tau)))
            burn = min(burn, sampler.iteration // 2)   # never discard >50 %
            return burn, thin

        burn, thin = auto_burn_thin(sampler)
        flat = sampler.get_chain(discard=burn, thin=thin, flat=True)
        log_probs_flat = sampler.get_log_prob(discard=burn, thin=thin, flat=True)
        
        # Find the best parameters
        best_idx = np.argmax(log_probs_flat)
        best_p = FRBParams.from_sequence(flat[best_idx], best_key)
        
        if verbose:
            print(f"\nBest-fit parameters ({best_key}):")
            p_names = {
                "M0": ["c0", "t0", "gamma"],
                "M1": ["c0", "t0", "gamma", "zeta"],
                "M2": ["c0", "t0", "gamma", "tau_1ghz"],
                "M3": ["c0", "t0", "gamma", "zeta", "tau_1ghz"],
            }[best_key]
            
            for i, name in enumerate(p_names):
                val = getattr(best_p, name)
                # Get uncertainty from chain
                param_std = np.std(flat[:, i])
                print(f"  {name}: {val:.3f} ± {param_std:.3f}")

        # Run diagnostics
        diag = None
        if diagnostics:
            if verbose:
                print("\n=== Running Diagnostics ===")
            
            diag = BurstDiagnostics(self.dataset, best_p, dm_init=dm_init)
            
            if sb_steps is None:
                sb_steps = self.sb_steps
            
            # Sub-band consistency check
            if verbose:
                print("\n1. Sub-band Consistency Check")
            p_name_sb, subband_results = diag.subband(best_key, sb_steps)
            
            if best_key in ["M2", "M3"] and verbose:
                print(f"   Sub-band τ_1GHz values:")
                for i, (tau, std) in enumerate(subband_results):
                    if not np.isnan(tau):
                        print(f"   Band {i+1}: τ = {tau:.3f} ± {std:.3f} ms")
                    else:
                        print(f"   Band {i+1}: No scattering (model {best_key} has no tau)")
                
                # Check consistency if we have valid measurements
                valid_results = [(tau, std) for tau, std in subband_results 
                               if not np.isnan(tau) and std > 0]
                
                if len(valid_results) > 1:
                    global_tau = best_p.tau_1ghz
                    chi2_tau = sum((tau - global_tau)**2 / std**2 
                                  for tau, std in valid_results)
                    dof = len(valid_results) - 1
                    
                    p_value = 1 - sp.stats.chi2.cdf(chi2_tau, df=dof)
                    print(f"\n   Consistency test: χ² = {chi2_tau:.1f}, p = {p_value:.3f}")
                    
                    if p_value < 0.05:
                        print("   ⚠️  WARNING: Scattering time varies significantly across bands!")
                        print("      Possible causes:")
                        print("      - Frequency-dependent scattering screen")
                        print("      - Multiple scattering components")
                        print("      - RFI or calibration issues in some bands")
                    else:
                        print("   ✓ Scattering times are consistent across bands")
            
            # Leave-one-out influence check
            if verbose:
                print("\n2. Channel Influence Check")
            influence_results = diag.influence(best_key, plot=False)
            
            # Find outliers
            influence_std = np.std(influence_results)
            influence_mean = np.mean(influence_results)
            outliers = np.where(np.abs(influence_results - influence_mean) > 3 * influence_std)[0]
            
            if verbose:
                if len(outliers) > 0:
                    print(f"   Found {len(outliers)} influential channels (>3σ):")
                    for idx in outliers:
                        freq_val = self.dataset.freq[idx]
                        influence_val = influence_results[idx]
                        print(f"   Channel {idx}: {freq_val:.3f} GHz (Δχ² = {influence_val:.1f})")
                    print("\n   ⚠️  Consider masking these channels and re-fitting")
                else:
                    print("   ✓ No overly influential channels detected")
                    
            # Sub-banded 1D profile fit check
            if verbose:
                print("\n3. 1-D Sub-band Profile Check")

            centres, tau1d, tau1d_err = diag.profile1d()

            if verbose:
                for k, (nu, tt, ee) in enumerate(zip(centres, tau1d, tau1d_err), 1):
                    print(f"   Band {k}: ν = {nu:.3f} GHz   τ = {tt:.3f} ± {ee:.3f} ms")

            # Store all results
            diag.subband_results = subband_results
            diag.influence_results = influence_results

        # Calculate goodness of fit
        model = FRBModel(
            time=self.dataset.time, 
            freq=self.dataset.freq, 
            data=self.dataset.data,
            dm_init=dm_init
        )
        model_dyn = model(best_p, best_key)
        
        n_params = len(p_names)
        gof = goodness_of_fit(
            self.dataset.data, 
            model_dyn, 
            model.noise_std,
            n_params=n_params
        )
        
        if verbose:
            print("\n=== Goodness of Fit ===")
            print(f"χ² = {gof['chi2']:.1f}")
            print(f"χ²_reduced = {gof['chi2_reduced']:.2f}")
            print(f"DOF = {gof['ndof']}")
            
            if gof['chi2_reduced'] > 2:
                print("⚠️  Poor fit quality (χ²_reduced > 2)")
                print("   Consider:")
                print("   - Checking for RFI or bad channels")
                print("   - Trying a more complex model")
                print("   - Adjusting data preprocessing")
            elif gof['chi2_reduced'] < 0.5:
                print("⚠️  Possibly overfit (χ²_reduced < 0.5)")
                print("   Consider using a simpler model")
        
        # Plotting
        if plot:
            output_file_sixteenpanel = Path(f"{self.dataset.path.stem}_diagnostic_summary.pdf")
            create_sixteen_panel_plot(
                dataset=self.dataset,
                best_p=best_p,
                best_key=best_key,
                noise_std=self.fitter.model.noise_std, # Get noise from the model instance
                output_path=output_file_sixteenpanel,
                show=show # Pass the 'show' argument from run_full
            )
            
            output_file_fourpanel = Path(f"{self.dataset.path.stem}_four_panel.pdf")
            create_four_panel_plot(
                dataset=self.dataset,
                best_p=best_p,
                best_key=best_key,
                noise_std=self.fitter.model.noise_std, # Get noise from the model instance
                output_path=output_file_fourpanel,
                show=show # Pass the 'show' argument from run_full
            )
                
        else:
            return fig, {
                "best_key": best_key,
                "best_params": best_p,
                "sampler": sampler,
                "diagnostics": diag,
                "profile1d": (centres, tau1d, tau1d_err),
                "subband_param": (p_name_sb, subband_results),
                "goodness_of_fit": gof,
                "chain_stats": {
                    "burn_in": burn,
                    "thin": thin,
                    "n_samples": flat.shape[0],
                    "acceptance_fraction": np.mean(sampler.acceptance_fraction) if hasattr(sampler, 'acceptance_fraction') else None
                }
            }

###############################################################################
# 5. CLI wrapper
###############################################################################


def _main():

    parser = argparse.ArgumentParser(description="Run BurstFit pipeline on a .npy burst cut‑out")
    parser.add_argument("npy", type=Path, help="Input .npy file")
    parser.add_argument("--dm_init", type=float, default=0.0, help="Initial DM corresponding to the .npy burst")
    parser.add_argument("--telescope", default="DSA-110")
    # Telescope / sampler configuration files
    parser.add_argument("--telcfg",  default="telescopes.yaml",
                        help="YAML file that contains telescope constants")
    parser.add_argument("--sampcfg", default="sampler.yaml",
                        help="YAML file that contains sampler settings")
    # Optional override: pick a sampler that is *inside* sampcfg
    parser.add_argument("--sampler",
                        help="Use this sampler block instead of the YAML default "
                             "(e.g. 'dynesty', 'zeus')")
    parser.add_argument("--nproc", type=int, default=None, help="Pool size (0=serial, omit=auto‑detect)")
    parser.add_argument("--yes", action="store_true", help="Skip pool confirmation prompt")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--no-scan", action="store_true")
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    pool = build_pool(args.nproc, auto_ok=args.yes, label="BurstFit")
    with pool or contextlib.nullcontext():
        pipe = BurstPipeline(
            args.npy,
            dm_init=args.dm_init,
            telescope=args.telescope,
            telcfg_path=args.telcfg,
            sampcfg_path=args.sampcfg,
            sampler_name=args.sampler, 
            n_steps=args.steps,
            sb_steps=args.steps,
            pool=pool,
        )
        res = pipe.run_full(model_scan=not args.no_scan, diagnostics=not args.no_diagnostics, plot=args.plot)
        print("Best model:", res["best_key"])
        print("Best parameters:", res["best_params"])

if __name__ == "__main__":
    _main()
