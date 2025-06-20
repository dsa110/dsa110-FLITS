"""
burstfit_pipeline.py
====================

Object‑oriented **orchestrator** that glues together all core modules
(`burstfit.py`, `burstfit_modelselect.py`, `burstfit_robust.py`),
reads telescope‑specific constants from `telescopes.yaml` via
`config_utils.load_telescope()`, and reads sampler constants from 
`sampler.yaml` via `config_utils.load_sampler()`.

Typical use
-----------
```
python
from burstfit_pipeline import BurstPipeline
pipe = BurstPipeline("burst.npy", telescope="CHIME", pool=6)
result = pipe.run_full(model_scan=True, diagnostics=True)
print(result["best_params"])
```
At the command line:
```bash
python burstfit_pipeline.py burst.npy --telescope DSA-110 --plot
```

Dependencies
------------
* PyYAML (for `config_utils`)
* NumPy, Matplotlib, emcee (pulled in by `burstfit`)
"""
from __future__ import annotations

import logging
import warnings
import pickle
import contextlib
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d  
from scipy.optimize import minimize  
import emcee

from burstfit import (
    FRBModel,
    FRBFitter,
    FRBParams,
    build_priors,
    plot_dynamic,
)
from burstfit_modelselect import fit_models_bic
from burstfit_robust import (
    subband_consistency,
    leave_one_out_influence,
    plot_influence,
)

from config_utils import load_telescope_block, load_sampler_block, load_sampler_choice, clear_config_cache
clear_config_cache()

from pool_utils import build_pool

log = logging.getLogger("burstfit.pipeline")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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
        self.telname, self.telparams = load_telescope_block(telcfg_path, telescope=telescope)
        self.sampname, self.sampparams = load_sampler_block(sampcfg_path)
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
    # publicd
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
        from burstfit import downsample
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
        self.df_MHz = p["df_MHz_raw"] * self.f_factor
        self.dt_ms = p["dt_ms_raw"] * self.t_factor
        freq = np.linspace(p["f_min_GHz"], p["f_max_GHz"], ds.shape[0])
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
        param_names = {
            "M0": ("c0", "t0", "gamma"),
            "M1": ("c0", "t0", "gamma", "zeta"),
            "M2": ("c0", "t0", "gamma", "tau_1ghz"),
            "M3": ("c0", "t0", "gamma", "zeta", "tau_1ghz"),
        }[model_key]

        # Pack initial values
        x0 = np.array([getattr(p0, name) for name in param_names])

        # Set bounds
        bounds = [(priors[name][0], priors[name][1]) for name in param_names]

        def _neg_log_posterior(x):
            # Unpack parameters
            param_dict = {name: val for name, val in zip(param_names, x)}
            # Fill in defaults for unused parameters
            for name in ["c0", "t0", "gamma", "zeta", "tau_1ghz"]:
                if name not in param_dict:
                    param_dict[name] = 0.0

            params = FRBParams(**param_dict)

            # Check priors
            for name, val in zip(param_names, x):
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

    def subband(self, best_key: str):
        self._subband_results = subband_consistency(
            self.ds.data, self.ds.freq, self.ds.time, 
            self.dm_init, self.best_p, model_key=best_key
        )
        return self._subband_results

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

###############################################################################
# 4. Pipeline façade
###############################################################################

class BurstPipeline:
    def __init__(
        self,
        path: str | Path,
        *,
        telescope: str = "DSA-110",
        telcfg_path: str | Path = "telescopes.yaml",
        sampcfg_path: str | Path = "sampler.yaml",
        sampler_name: str | None = None,
        n_steps: int = 2000,
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
        
        self.pool           = pool
        self.telcfg_path    = telcfg_path
        self.sampcfg_path   = sampcfg_path
        
        self.fitter = BurstFitter(
            self.dataset,
            n_steps=n_steps,
            pool=self.pool,
            sampcfg_path=sampcfg_path,
            sampler_name=sampler_name,     
        )

    def run_full(self, *, model_scan=True, diagnostics=True, plot=False, verbose=True):
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
        
        # Model selection or direct fit
        if model_scan:
            if verbose:
                print("\n=== Model Selection (BIC) ===")
            init = self.fitter._adapt_initial_guess()
            best_key, res = fit_models_bic(
                data=self.dataset.data,
                freq=self.dataset.freq,
                time=self.dataset.time,
                dm_init=0.0,
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
            """Estimate burn-in and thin by sampler autocorrelation times."""
            try:
                tau = sampler.get_autocorr_time(tol=0)
            except Exception as e:
                warnings.warn(f"Could not estimate autocorr time: {e}. Falling back to defaults.")
                return sampler.iteration // 4, 1

            tau_max = np.max(tau)
            tau_min = np.min(tau)

            burn = int(safety_factor_burn * tau_max)
            thin = max(1, int(safety_factor_thin * tau_min))

            # ensure we don't exceed total samples
            burn = min(burn, sampler.iteration // 2)
            return burn, thin

        burn, thin = auto_burn_thin(sampler)
        flat = sampler.get_chain(discard=burn, thin=thin, flat=True)
        log_probs_flat = sampler.get_log_prob(discard=burn, thin=thin, flat=True)
        
        # Find the best parameters
        best_idx = np.argmax(log_probs_flat)
        best_p = FRBParams.from_sequence(flat[best_idx], best_key)
        
        if verbose:
            print(f"\nBest-fit parameters ({best_key}):")
            param_names = {
                "M0": ["c0", "t0", "gamma"],
                "M1": ["c0", "t0", "gamma", "zeta"],
                "M2": ["c0", "t0", "gamma", "tau_1ghz"],
                "M3": ["c0", "t0", "gamma", "zeta", "tau_1ghz"],
            }[best_key]
            
            for i, name in enumerate(param_names):
                val = getattr(best_p, name)
                # Get uncertainty from chain
                param_std = np.std(flat[:, i])
                print(f"  {name}: {val:.3f} ± {param_std:.3f}")

        # Run diagnostics
        diag = None
        if diagnostics:
            if verbose:
                print("\n=== Running Diagnostics ===")
            
            diag = BurstDiagnostics(self.dataset, best_p)
            
            # Sub-band consistency check
            if verbose:
                print("\n1. Sub-band Consistency Check")
            subband_results = diag.subband(best_key)
            
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
                    
                    import scipy.stats
                    p_value = 1 - scipy.stats.chi2.cdf(chi2_tau, df=dof)
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
                        freq_val = self.dataset.freq[idx * self.dataset.f_factor]  # Account for downsampling
                        influence_val = influence_results[idx]
                        print(f"   Channel {idx}: {freq_val:.3f} GHz (Δχ² = {influence_val:.1f})")
                    print("\n   ⚠️  Consider masking these channels and re-fitting")
                else:
                    print("   ✓ No overly influential channels detected")
            
            # Store all results
            diag.subband_results = subband_results
            diag.influence_results = influence_results

        # Calculate goodness of fit
        model = FRBModel(
            time=self.dataset.time, 
            freq=self.dataset.freq, 
            data=self.dataset.data,
            dm_init=0.0
        )
        model_dyn = model(best_p, best_key)
        
        from burstfit import goodness_of_fit
        n_params = len(param_names)
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
            # Quick look at data
            self.dataset.quicklook()

            # Create comprehensive diagnostic plot
            fig = plt.figure(figsize=(16, 10))
            
            # 1. Data
            ax1 = plt.subplot(3, 4, 1)
            plot_dynamic(ax1, self.dataset.data, self.dataset.time, 
                        self.dataset.freq, cmap="viridis")
            ax1.set_title("Data")
            ax1.set_xlabel("Time [ms]")
            ax1.set_ylabel("Frequency [GHz]")
            
            # 2. Model
            ax2 = plt.subplot(3, 4, 2)
            plot_dynamic(ax2, model_dyn, self.dataset.time, 
                        self.dataset.freq, cmap="viridis")
            ax2.set_title(f"Model ({best_key})")
            ax2.set_xlabel("Time [ms]")
            
            # 3. Residual
            ax3 = plt.subplot(3, 4, 3)
            residual = self.dataset.data - model_dyn
            vmax = np.percentile(np.abs(residual), 95)
            plot_dynamic(ax3, residual, self.dataset.time, 
                        self.dataset.freq, cmap="RdBu_r", 
                        vmin=-vmax, vmax=vmax)
            ax3.set_title("Residual")
            ax3.set_xlabel("Time [ms]")
            
            # 4. Residual histogram
            ax4 = plt.subplot(3, 4, 4)
            residual_normalized = residual / model.noise_std[:, None]
            ax4.hist(residual_normalized.flatten(), bins=50, density=True, 
                    alpha=0.7, label='Residuals')
            x = np.linspace(-4, 4, 100)
            ax4.plot(x, np.exp(-0.5*x**2)/np.sqrt(2*np.pi), 'r-', 
                    label='N(0,1)')
            ax4.set_xlabel('Normalized Residual')
            ax4.set_ylabel('Density')
            ax4.set_title('Residual Distribution')
            ax4.legend()
            
            # 5. Time profile
            ax5 = plt.subplot(3, 4, 5)
            data_profile = np.sum(self.dataset.data, axis=0)
            model_profile = np.sum(model_dyn, axis=0)
            residual_profile = np.sum(residual, axis=0)
            ax5.plot(self.dataset.time, data_profile, 'k-', label='Data', alpha=0.7)
            ax5.plot(self.dataset.time, model_profile, 'r--', label='Model', linewidth=2)
            ax5.fill_between(self.dataset.time, residual_profile, alpha=0.3, 
                           color='gray', label='Residual')
            ax5.set_xlabel('Time [ms]')
            ax5.set_ylabel('Flux')
            ax5.set_title('Time Profile')
            ax5.legend()
            
            # 6. Spectrum
            ax6 = plt.subplot(3, 4, 6)
            data_spectrum = np.sum(self.dataset.data, axis=1)
            model_spectrum = np.sum(model_dyn, axis=1)
            ax6.plot(self.dataset.freq, data_spectrum, 'k-', label='Data', alpha=0.7)
            ax6.plot(self.dataset.freq, model_spectrum, 'r--', label='Model', linewidth=2)
            ax6.set_xlabel('Frequency [GHz]')
            ax6.set_ylabel('Flux')
            ax6.set_title('Frequency Spectrum')
            ax6.legend()
            
            # 7. Channel influence (if diagnostics run)
            if diag and hasattr(diag, 'influence_results'):
                ax7 = plt.subplot(3, 4, 7)
                plot_influence(ax7, diag.influence_results, self.dataset.freq)
                ax7.set_title('Channel Influence')
            
            # 8. Sub-band consistency (for M2/M3)
            if diag and best_key in ["M2", "M3"] and hasattr(diag, 'subband_results'):
                ax8 = plt.subplot(3, 4, 8)
                valid_results = [(i, tau, err) for i, (tau, err) in enumerate(diag.subband_results)
                               if not np.isnan(tau) and err > 0]
                if valid_results:
                    indices, sub_taus, sub_errs = zip(*valid_results)
                    n_bands = len(diag.subband_results)
                    band_centers = self.dataset.freq[0] + (self.dataset.freq[-1] - self.dataset.freq[0]) * (np.array(indices) + 0.5) / n_bands
                    
                    ax8.errorbar(band_centers, sub_taus, yerr=sub_errs, 
                               fmt='o', capsize=5, markersize=8, label='Sub-bands')
                    ax8.axhline(best_p.tau_1ghz, color='r', linestyle='--', linewidth=2,
                              label=f'Global: {best_p.tau_1ghz:.3f} ms')
                    
                    # Add shaded region for global uncertainty
                    if 'tau_1ghz' in param_names:
                        tau_idx = param_names.index('tau_1ghz')
                        tau_std = np.std(flat[:, tau_idx])
                        ax8.fill_between([self.dataset.freq[0], self.dataset.freq[-1]], 
                                       best_p.tau_1ghz - tau_std,
                                       best_p.tau_1ghz + tau_std,
                                       alpha=0.3, color='red')
                    
                    ax8.set_xlabel('Frequency [GHz]')
                    ax8.set_ylabel('τ_1GHz [ms]')
                    ax8.set_title('Sub-band Scattering')
                    ax8.legend()
            
            # 9. Parameter evolution (trace plots for 2 key parameters)
            if sampler is not None:
                ax9 = plt.subplot(3, 4, 9)
                chain = sampler.get_chain()
                # Plot first parameter evolution
                for i in range(min(10, chain.shape[1])):  # Show up to 10 walkers
                    ax9.plot(chain[:, i, 0], alpha=0.5)
                ax9.axvline(burn, color='r', linestyle='--', label=f'Burn-in ({burn} steps)')
                ax9.set_xlabel('Step')
                ax9.set_ylabel(param_names[0])
                ax9.set_title('Chain Evolution')
                ax9.legend()
                
                # 10. Corner plot preview (2D correlation)
                ax10 = plt.subplot(3, 4, 10)
                if len(param_names) >= 2 and flat.shape[0] > 100:
                    # Subsample for speed
                    idx = np.random.choice(flat.shape[0], min(1000, flat.shape[0]), replace=False)
                    ax10.scatter(flat[idx, 0], flat[idx, 1], alpha=0.5, s=1)
                    ax10.scatter(flat[best_idx, 0], flat[best_idx, 1], 
                               color='red', s=100, marker='*', label='Best fit')
                    ax10.set_xlabel(param_names[0])
                    ax10.set_ylabel(param_names[1])
                    ax10.set_title('Parameter Correlation')
                    ax10.legend()
            
            # 11. Goodness of fit summary
            ax11 = plt.subplot(3, 4, 11)
            ax11.axis('off')
            summary_text = f"""Fit Summary:
Model: {best_key}
χ² = {gof['chi2']:.1f}
χ²_reduced = {gof['chi2_reduced']:.2f}
DOF = {gof['ndof']}

Parameters:"""
            for name in param_names:
                val = getattr(best_p, name)
                if name in param_names:
                    idx = param_names.index(name)
                    std = np.std(flat[:, idx])
                    summary_text += f"\n{name} = {val:.3f} ± {std:.3f}"
            
            ax11.text(0.1, 0.9, summary_text, transform=ax11.transAxes,
                     verticalalignment='top', fontfamily='monospace', fontsize=10)
            
            # 12. Autocorrelation of residuals
            ax12 = plt.subplot(3, 4, 12)
            autocorr = gof['residual_autocorr']
            lags = np.arange(len(autocorr)) - len(autocorr)//2
            ax12.plot(lags * (self.dataset.time[1] - self.dataset.time[0]), autocorr)
            ax12.axhline(0, color='k', linestyle='-', alpha=0.3)
            ax12.axhline(1.96/np.sqrt(len(autocorr)), color='r', linestyle='--', alpha=0.5)
            ax12.axhline(-1.96/np.sqrt(len(autocorr)), color='r', linestyle='--', alpha=0.5)
            ax12.set_xlabel('Lag [ms]')
            ax12.set_ylabel('Autocorrelation')
            ax12.set_title('Residual Autocorrelation')
            ax12.set_xlim(-5, 5)  # Show ±5 ms
            
            plt.suptitle(f'BurstFit Comprehensive Diagnostics - {self.dataset.path.name}', 
                        fontsize=14)
            plt.tight_layout()
            plt.show()

        return {
            "best_key": best_key,
            "best_params": best_p,
            "sampler": sampler,
            "diagnostics": diag,
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
    import argparse

    parser = argparse.ArgumentParser(description="Run BurstFit pipeline on a .npy burst cut‑out")
    parser.add_argument("npy", type=Path, help="Input .npy file")
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
            telescope=args.telescope,
            telcfg_path=args.telcfg,
            sampcfg_path=args.sampcfg,
            sampler_name=args.sampler, 
            n_steps=args.steps,
            pool=pool,
        )
        res = pipe.run_full(model_scan=not args.no_scan, diagnostics=not args.no_diagnostics, plot=args.plot)
        print("Best model:", res["best_key"])
        print("Best parameters:", res["best_params"])

if __name__ == "__main__":
    import contextlib
    _main()
