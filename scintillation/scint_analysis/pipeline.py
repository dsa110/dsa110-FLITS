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
        
        self.cache_dir = self.config.get('pipeline_options', {}).get('cache_directory', './cache')
        if self.config.get('pipeline_options', {}).get('save_intermediate_steps'):
            os.makedirs(self.cache_dir, exist_ok=True)
            log.info(f"Intermediate results will be cached in: {self.cache_dir}")

    def _get_cache_path(self, stage_name):
        """Generates a standard path for a cache file."""
        burst_id = self.config.get('burst_id', 'unknown_burst')
        return os.path.join(self.cache_dir, f"{burst_id}_{stage_name}.pkl")
        
    def _create_diagnostic_plots(self, burst_lims, off_pulse_lims):
        """Internal helper to generate and save diagnostic plots."""
        diag_config = self.config.get('pipeline_options', {}).get('diagnostic_plots', {})
        if not diag_config.get('enable', False):
            return
            
        log.info("Generating diagnostic plots...")
        plot_dir = diag_config.get('directory', './plots/diagnostics')
        os.makedirs(plot_dir, exist_ok=True)
        burst_id = self.config.get('burst_id', 'unknown_burst')

        try:
            # --- On-pulse plots ---
            on_pulse_power = self.masked_spectrum.power[:, burst_lims[0]:burst_lims[1]]
            on_pulse_times = self.masked_spectrum.times[burst_lims[0]:burst_lims[1]]

            # Create a temporary DynamicSpectrum object for plotting
            on_pulse_ds_obj = core.DynamicSpectrum(on_pulse_power, self.masked_spectrum.frequencies, on_pulse_times)
            on_pulse_ts = np.ma.mean(on_pulse_power, axis=0)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
            # Pass the full DynamicSpectrum object to the plotting function
            plotting.plot_dynamic_spectrum(on_pulse_ds_obj, ax=ax1)
            ax1.set_title("On-Pulse Region")

            ax2.plot(on_pulse_times, on_pulse_ts)
            ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Mean Power")
            ax2.grid(True, alpha=0.5)
            plt.tight_layout()
            plt.show()
            plt.savefig(os.path.join(plot_dir, f"{burst_id}_on_pulse_diagnostic.png"), dpi=150)
            plt.close(fig)

            # --- Off-pulse plots ---
            off_pulse_power = self.masked_spectrum.power[:, off_pulse_lims[0]:off_pulse_lims[1]]
            off_pulse_times = self.masked_spectrum.times[off_pulse_lims[0]:off_pulse_lims[1]]

            # Create a temporary DynamicSpectrum object for plotting
            off_pulse_ds_obj = core.DynamicSpectrum(off_pulse_power, self.masked_spectrum.frequencies, off_pulse_times)
            off_pulse_ts = np.ma.mean(off_pulse_power, axis=0)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
            plotting.plot_dynamic_spectrum(off_pulse_ds_obj, ax=ax1)
            ax1.set_title("Off-Pulse (Noise) Region")

            ax2.plot(off_pulse_times, off_pulse_ts)
            ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Mean Power")
            ax2.grid(True, alpha=0.5)
            plt.tight_layout()
            plt.show()
            plt.savefig(os.path.join(plot_dir, f"{burst_id}_off_pulse_diagnostic.png"), dpi=150)
            plt.close(fig)

            log.info(f"Diagnostic plots saved to: {plot_dir}")
        except Exception as e:
            log.error(f"Failed to generate diagnostic plots: {e}")


    def run(self):
        """
        Executes the full scintillation analysis pipeline from start to finish.
        """
        log.info(f"--- Starting Scintillation Pipeline for {self.config['burst_id']} ---")

        processed_spec_cache = self._get_cache_path('processed_spectrum')

        if os.path.exists(processed_spec_cache):
            log.info(f"Loading cached processed spectrum from {processed_spec_cache}")
            with open(processed_spec_cache, 'rb') as f:
                self.masked_spectrum = pickle.load(f)
        else:
            log.info("Loading and processing raw data...")
            spectrum = core.DynamicSpectrum.from_numpy_file(self.config['input_data_path'])
            rfi_masked_spectrum = spectrum.mask_rfi(self.config)

            baseline_config = self.config.get('analysis', {}).get('baseline_subtraction', {})
            if baseline_config.get('enable', False):
                log.info("Applying polynomial baseline subtraction...")
                burst_lims_pre = rfi_masked_spectrum.find_burst_envelope(
                    thres=self.config.get('analysis',{}).get('rfi_masking',{}).get('find_burst_thres', 5),
                    padding_factor=self.config.get('analysis',{}).get('rfi_masking',{}).get('padding_factor', 0.2)
                )
                off_pulse_end_bin = burst_lims_pre[0] - 200
                
                if off_pulse_end_bin < 100:
                    log.warning("Not enough pre-burst data to model baseline. Skipping subtraction.")
                    self.masked_spectrum = rfi_masked_spectrum
                else:
                    poly_order = baseline_config.get('poly_order', 1)
                    off_pulse_spectrum_1d = rfi_masked_spectrum.get_spectrum((0, off_pulse_end_bin))
                    self.masked_spectrum = rfi_masked_spectrum.subtract_poly_baseline(off_pulse_spectrum_1d, poly_order=poly_order)
            else:
                log.info("Skipping optional baseline subtraction.")
                self.masked_spectrum = rfi_masked_spectrum

            if self.config.get('pipeline_options', {}).get('save_intermediate_steps'):
                with open(processed_spec_cache, 'wb') as f:
                    pickle.dump(self.masked_spectrum, f)
                log.info(f"Saved processed spectrum to cache: {processed_spec_cache}")
        
        log.info("Locating burst and defining analysis windows...")
        burst_lims = self.masked_spectrum.find_burst_envelope(
                    thres=self.config.get('analysis',{}).get('rfi_masking',{}).get('find_burst_thres', 5),
                    padding_factor=self.config.get('analysis',{}).get('rfi_masking',{}).get('padding_factor', 0.2)
                )
        noise_end_bin = burst_lims[0] - 200
        
        self._create_diagnostic_plots(burst_lims, (0, noise_end_bin))
        
        if noise_end_bin < 100:
            log.warning("Not enough pre-burst data for robust noise characterization. Skipping.")
            self.noise_descriptor = None
        else:
            log.info("Characterizing off-pulse noise...")
            off_pulse_data = self.masked_spectrum.power.data[:, 0:noise_end_bin].T
            self.noise_descriptor = noise.estimate_noise_descriptor(off_pulse_data)
            log.info(f"Noise characterization complete. Detected kind: '{self.noise_descriptor.kind}'")
        
        # --- Stage 3: Calculate ACFs ---
        acf_results_cache = self._get_cache_path('acf_results')
        # Check for cached ACF results
        if os.path.exists(acf_results_cache) and os.path.getmtime(acf_results_cache) > os.path.getmtime(processed_spec_cache):
            log.info(f"Loading cached ACF results from {acf_results_cache}")
            with open(acf_results_cache, 'rb') as f:
                self.acf_results = pickle.load(f)
        else:
            log.info("Calculating ACFs for all sub-bands...")
            self.acf_results = analysis.calculate_acfs_for_subbands(
                self.masked_spectrum, self.config, burst_lims=burst_lims, noise_desc=self.noise_descriptor
            )
            if self.config.get('pipeline_options', {}).get('save_intermediate_steps'):
                # Use standard open() function on the string path
                with open(acf_results_cache, 'wb') as f:
                    pickle.dump(self.acf_results, f)
                log.info(f"Saved ACF results to cache: {acf_results_cache}")
        
        # --- Stage 4: Fit Models and Derive Parameters ---
        if not self.acf_results or not self.acf_results['subband_acfs']:
            log.error("ACF results are empty, cannot proceed to fitting. Exiting.")
            return

        log.info("Fitting models and deriving final scintillation parameters...")
        self.final_results, self.all_subband_fits, self.all_powerlaw_fits = analysis.analyze_scintillation_from_acfs(
            self.acf_results, self.config
        )
        
        log.info("--- Pipeline execution finished. ---")
