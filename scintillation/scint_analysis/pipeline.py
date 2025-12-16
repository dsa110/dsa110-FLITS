# ==============================================================================
# File: scint_analysis/scint_analysis/pipeline.py
# ==============================================================================
import os
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
# Make sure to import the new noise module
from . import analysis, core, noise, plotting

log = logging.getLogger(__name__)

class ScintillationAnalysis:
    """
    An object-oriented controller for running the end-to-end scintillation pipeline.
    """
    def __init__(self, config):
        self.config = config
        self.masked_spectrum = None
        self.noise_descriptor = None
        self.acf_results = None
        self.all_subband_fits = None 
        self.final_results = None  
        self.all_powerlaw_fits = None
        self.intra_pulse_results = None
        self.data_prepared = False
        
        self.cache_dir = self.config.get('pipeline_options', {}).get('cache_directory', './cache')
        if self.config.get('pipeline_options', {}).get('save_intermediate_steps'):
            os.makedirs(self.cache_dir, exist_ok=True)
            log.info(f"Intermediate results will be cached in: {self.cache_dir}")

    def _get_cache_path(self, stage_name):
        """Generates a standard path for a cache file."""
        burst_id = self.config.get('burst_id', 'unknown_burst')
        return os.path.join(self.cache_dir, f"{burst_id}_{stage_name}.pkl")
        
    def _create_diagnostic_plots(self, burst_lims, off_pulse_lims, baseline_info=None):
        """Internal helper to generate and save diagnostic plots."""
        diag_config = self.config.get('pipeline_options', {}).get('diagnostic_plots', {})
        if not diag_config.get('enable', False):
            return
            
        log.info("Generating diagnostic plots...")
        plot_dir = diag_config.get('directory', './plots/diagnostics')
        os.makedirs(plot_dir, exist_ok=True)
        burst_id = self.config.get('burst_id', 'unknown_burst')
        
        # --- On-pulse and Off-pulse Window Plots ---
        try:
            # 1. Prepare and plot the on-pulse window
            on_pulse_power = self.masked_spectrum.power[:, burst_lims[0]:burst_lims[1]]
            on_pulse_times = self.masked_spectrum.times[burst_lims[0]:burst_lims[1]]
            on_pulse_ds_obj = core.DynamicSpectrum(
                on_pulse_power, self.masked_spectrum.frequencies, on_pulse_times
            )
            on_pulse_save_path = os.path.join(plot_dir, f"{burst_id}_on_pulse_diagnostic.png")
            
            plotting.plot_pulse_window_diagnostic(
                on_pulse_ds_obj,
                title="On-Pulse Region",
                save_path=on_pulse_save_path
            )

            # 2. Prepare and plot the off-pulse (noise) window
            off_pulse_power = self.masked_spectrum.power[:, off_pulse_lims[0]:off_pulse_lims[1]]
            off_pulse_times = self.masked_spectrum.times[off_pulse_lims[0]:off_pulse_lims[1]]
            off_pulse_ds_obj = core.DynamicSpectrum(
                off_pulse_power, self.masked_spectrum.frequencies, off_pulse_times
            )
            off_pulse_save_path = os.path.join(plot_dir, f"{burst_id}_off_pulse_diagnostic.png")
            
            plotting.plot_pulse_window_diagnostic(
                off_pulse_ds_obj,
                title="Off-Pulse (Noise) Region",
                save_path=off_pulse_save_path
            )

            log.info(f"On/Off pulse diagnostic plots saved to: {plot_dir}")

        except Exception as e:
            log.error(f"Failed to generate on/off pulse diagnostic plots: {e}")
            
        if baseline_info:
            log.info("Generating baseline fit diagnostic plot.")
            baseline_save_path = os.path.join(plot_dir, f"{burst_id}_baseline_diagnostic.png")
            plotting.plot_baseline_fit(
                off_pulse_spectrum=baseline_info['original_data'],
                fitted_baseline=baseline_info['model'],
                frequencies=self.masked_spectrum.frequencies,
                poly_order=baseline_info['poly_order'],
                save_path=baseline_save_path
            )

    def prepare_data(self):
        """
        Loads data from file and performs initial RFI masking.
        Populates self.masked_spectrum.
        """
        if self.data_prepared:
            log.info("Data already prepared. Skipping.")
            return

        log.info("--- Preparing Data ---")
        
        self.cache_dir = self.config.get('pipeline_options', {}).get('cache_directory', './cache')
        if self.config.get('pipeline_options', {}).get('save_intermediate_steps'):
            os.makedirs(self.cache_dir, exist_ok=True)
            log.info(f"Intermediate results will be cached in: {self.cache_dir}")

        processed_spec_cache = self._get_cache_path('processed_spectrum')
        
        if os.path.exists(processed_spec_cache) and not self.config.get('pipeline_options', {}).get('force_recalc', False):
            log.info(f"Loading cached processed spectrum from {processed_spec_cache}")
            with open(processed_spec_cache, 'rb') as f:
                # The cache now only needs to store the masked spectrum
                self.masked_spectrum = pickle.load(f)
            
        else:
            log.info("Loading and processing raw data...")
            #spectrum = core.DynamicSpectrum.from_numpy_file(self.config['input_data_path'])
            # --- optional down-sampling factors ---------------------------------
            ds_cfg   = self.config.get('pipeline_options', {}).get('downsample', {})
            f_factor = int(ds_cfg.get('f_factor', 1))
            t_factor = int(ds_cfg.get('t_factor', 1))

            # --------------------------------------------------------------------
            spectrum = core.DynamicSpectrum.from_numpy_file(
                self.config['input_data_path']
            ).downsample(f_factor, t_factor)
            # The mask_rfi function now correctly uses the manual window if present
            self.masked_spectrum = spectrum.mask_rfi(self.config)
            
            if self.config.get('pipeline_options', {}).get('save_intermediate_steps'):
                with open(processed_spec_cache, 'wb') as f:
                    pickle.dump(self.masked_spectrum, f)
        
        self.data_prepared = True
        log.info("--- Data Preparation Finished ---")

    def run(self):
        """
        Executes the full scintillation analysis pipeline from start to finish.
        """
        self.prepare_data() # Ensures data is loaded

        log.info(f"--- Starting Scintillation Pipeline for {self.config['burst_id']} ---")

        rfi_config = self.config.get('analysis', {}).get('rfi_masking', {})
        
        # --- CENTRALIZED WINDOW DETERMINATION ---
        manual_on_pulse = rfi_config.get('manual_burst_window')
        if manual_on_pulse and len(manual_on_pulse) == 2:
            burst_lims = manual_on_pulse
            log.warning(f"RUN: Using manually specified on-pulse window: {burst_lims}")
        else:
            log.info("RUN: Using automated burst detection for on-pulse window.")
            burst_lims = self.masked_spectrum.find_burst_envelope(
                thres=rfi_config.get('find_burst_thres', 5.0),
                padding_factor=rfi_config.get('padding_factor', 0.2)
            )
            
        manual_off_pulse = rfi_config.get('manual_noise_window')
        if manual_off_pulse and len(manual_off_pulse) == 2:
            off_pulse_lims = manual_off_pulse
            log.warning(f"RUN: Using manually specified off-pulse (noise) window: {off_pulse_lims}")
        else:
            noise_end_bin = burst_lims[0] - 200 # Default buffer
            off_pulse_lims = (max(0, noise_end_bin - 500), noise_end_bin) # Default off-pulse
            log.info(f"RUN: Using automated off-pulse window: {off_pulse_lims}")
        # --- END CENTRALIZED WINDOW DETERMINATION ---

        # --- BASELINE SUBTRACTION (MOVED HERE) ---
        baseline_info_for_plotting = None
        baseline_config = self.config.get('analysis', {}).get('baseline_subtraction', {})
        if baseline_config.get('enable', False):
            log.info("Applying polynomial baseline subtraction...")
            if off_pulse_lims[1] > off_pulse_lims[0] + 50: # Check for a valid off-pulse region
                poly_order = baseline_config.get('poly_order', 1)
                # Use the finalized off_pulse_lims to get the spectrum for baseline fitting
                off_pulse_spectrum_1d = self.masked_spectrum.get_spectrum(off_pulse_lims)
                
                # Create a temporary variable to hold the spectrum before subtraction for the plot
                spec_before_baseline = self.masked_spectrum
                
                self.masked_spectrum, baseline_model = self.masked_spectrum.subtract_poly_baseline(
                    off_pulse_spectrum_1d, poly_order=poly_order
                )
                if baseline_model is not None:
                    baseline_info_for_plotting = {
                        'original_data': spec_before_baseline.get_spectrum(off_pulse_lims),
                        'model': baseline_model,
                        'poly_order': poly_order
                    }
            else:
                log.warning("Not enough off-pulse data to model baseline. Skipping subtraction.")

        # --- DIAGNOSTIC PLOTS ---
        # This function is now called AFTER the final windows are determined.
        self._create_diagnostic_plots(burst_lims, off_pulse_lims, baseline_info=baseline_info_for_plotting)

        # --- NOISE CHARACTERIZATION ---
        if self.config.get('analysis', {}).get('noise', {}).get('disable', False):
            log.info("Noise modelling disabled by config.")
            self.noise_descriptor = None
        elif off_pulse_lims[1] > off_pulse_lims[0] + 100:
            log.info("Characterizing off-pulse noise...")
            off_pulse_data = self.masked_spectrum.power.data[:, off_pulse_lims[0]:off_pulse_lims[1]].T
            self.noise_descriptor = noise.estimate_noise_descriptor(off_pulse_data)
            log.info(f"Noise characterization complete. Detected kind: '{self.noise_descriptor.kind}'")
        else:
            log.warning("Not enough pre-burst data for robust noise characterization. Skipping.")
            self.noise_descriptor = None
            
        # --- ACF CALCULATION ---
        acf_results_cache = self._get_cache_path('acf_results')
        if os.path.exists(acf_results_cache) and not self.config.get('pipeline_options', {}).get('force_recalc', False):
            log.info(f"Loading cached ACF results from {acf_results_cache}")
            with open(acf_results_cache, 'rb') as f:
                self.acf_results = pickle.load(f)
        else:
            log.info("Calculating ACFs for all sub-bands...")
            self.acf_results = analysis.calculate_acfs_for_subbands(
                self.masked_spectrum, self.config, burst_lims=burst_lims, noise_desc=self.noise_descriptor
            )
            if self.config.get('pipeline_options', {}).get('save_intermediate_steps'):
                with open(acf_results_cache, 'wb') as f:
                    pickle.dump(self.acf_results, f)
                log.info(f"Saved ACF results to cache: {acf_results_cache}")
                
        # --- HALT CHECK ---
        if self.config.get('pipeline_options', {}).get('halt_after_acf', False):
            log.info("'halt_after_acf' is set to True. Halting pipeline as requested.")
            return
        
        # --- Run the intra-pulse analysis ---
        acf_config = self.config.get('analysis', {}).get('acf', {})
        if acf_config.get('enable_intra_pulse_analysis', False):
            ### FIX: Log message moved inside the conditional check ###
            log.info(f"Running intra-pulse analysis...")
            if self.noise_descriptor:
                self.intra_pulse_results = analysis.analyze_intra_pulse_scintillation(
                    self.masked_spectrum,
                    burst_lims,
                    self.config,
                    self.noise_descriptor
                )
            else:
                log.warning("Cannot run intra-pulse analysis without a valid noise descriptor. Skipping.")
        
        # --- Stage 4: Fit Models and Derive Parameters ---
        if not self.acf_results or not self.acf_results['subband_acfs']:
            log.error("ACF results are empty, cannot proceed to fitting. Exiting.")
            return

        log.info("Fitting models and deriving final scintillation parameters...")
        self.final_results, self.all_subband_fits, self.all_powerlaw_fits = analysis.analyze_scintillation_from_acfs(
            self.acf_results, self.config
        )
        
        # --- 2D GLOBAL SCINTILLATION FIT ---
        self.fit_2d_result = None
        fit_2d_config = self.config.get('analysis', {}).get('fit_2d', {})
        if fit_2d_config.get('enable', True):  # Enabled by default
            log.info("Running 2D global scintillation fit across all sub-bands...")
            self.fit_2d_result = self._run_2d_scintillation_fit(fit_2d_config)
        
        log.info("--- Pipeline execution finished. ---")
    
    def _run_2d_scintillation_fit(self, fit_2d_config):
        """
        Run 2D global scintillation fit across all sub-bands.
        
        This enforces physical frequency scaling: γ(ν) = γ₀ × (ν/ν_ref)^α
        and provides direct measurement of the scaling index α.
        """
        try:
            from .fitting_2d import fit_2d_scintillation, Scintillation2DResult
        except ImportError as e:
            log.warning(f"Could not import fitting_2d module: {e}. Skipping 2D fit.")
            return None
        
        if self.acf_results is None or not self.acf_results.get('subband_acfs'):
            log.warning("No ACF results available for 2D fitting.")
            return None
        
        try:
            result = fit_2d_scintillation(
                self.acf_results,
                model_type=fit_2d_config.get('model_type', 'lorentzian'),
                fit_range_mhz=fit_2d_config.get('fit_range_mhz', 25.0),
                nu_ref=fit_2d_config.get('nu_ref', None),
                gamma_0_init=fit_2d_config.get('gamma_0_init', 1.0),
                alpha_init=fit_2d_config.get('alpha_init', 4.0),
                m_0_init=fit_2d_config.get('m_0_init', 0.5),
                vary_alpha=fit_2d_config.get('vary_alpha', True),
                include_self_noise=fit_2d_config.get('include_self_noise', False),
            )
            
            log.info(
                f"2D fit complete: γ₀ = {result.gamma_0:.3f} ± {result.gamma_0_err:.3f} MHz, "
                f"α = {result.alpha:.2f} ± {result.alpha_err:.2f}, "
                f"χ²_red = {result.redchi:.2f}"
            )
            
            # Store in final_results for convenience
            if self.final_results is not None:
                self.final_results['fit_2d'] = {
                    'gamma_0': result.gamma_0,
                    'gamma_0_err': result.gamma_0_err,
                    'alpha': result.alpha,
                    'alpha_err': result.alpha_err,
                    'm_0': result.m_0,
                    'm_0_err': result.m_0_err,
                    'nu_ref': result.nu_ref,
                    'redchi': result.redchi,
                    'success': result.success,
                }
            
            return result
            
        except Exception as e:
            log.error(f"2D scintillation fit failed: {e}")
            return None