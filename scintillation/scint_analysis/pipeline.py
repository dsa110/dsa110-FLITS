# ==============================================================================
# File: scint_analysis/scint_analysis/pipeline.py
# ==============================================================================
import os
import pickle
import logging
from .core import DynamicSpectrum
from . import analysis

log = logging.getLogger(__name__)

class ScintillationAnalysis:
    """
    An object-oriented controller for running the end-to-end scintillation pipeline.
    """
    def __init__(self, config):
        self.config = config
        # Attributes to store results from each stage
        self.masked_spectrum = None
        self.acf_results = None
        self.all_subband_fits = None 
        self.final_results = None  
        self.powlaw_params = None
        
        self.cache_dir = config.get('pipeline_options', {}).get('cache_directory', './cache')
        if config.get('pipeline_options', {}).get('save_intermediate_steps'):
            os.makedirs(self.cache_dir, exist_ok=True)
            log.info(f"Intermediate results will be cached in: {self.cache_dir}")

    def _get_cache_path(self, stage_name):
        """Generates a standard path for a cache file."""
        burst_id = self.config.get('burst_id', 'unknown_burst')
        return os.path.join(self.cache_dir, f"{burst_id}_{stage_name}.pkl")

    def run(self):
        """
        Executes the full scintillation analysis pipeline from start to finish.
        """
        log.info(f"--- Starting Scintillation Pipeline for {self.config['burst_id']} ---")

        # --- Stage 1: Load and Mask Data ---
        masked_spec_cache = self._get_cache_path('masked_spectrum')
        if os.path.exists(masked_spec_cache):
            log.info(f"Loading cached masked spectrum from {masked_spec_cache}")
            with open(masked_spec_cache, 'rb') as f:
                self.masked_spectrum = pickle.load(f)
        else:
            log.info("Loading and processing raw data...")
            spectrum = DynamicSpectrum.from_numpy_file(self.config['input_data_path'])
            self.masked_spectrum = spectrum.mask_rfi(self.config)
            if self.config.get('pipeline_options', {}).get('save_intermediate_steps'):
                with open(masked_spec_cache, 'wb') as f:
                    pickle.dump(self.masked_spectrum, f)
                log.info(f"Saved masked spectrum to cache: {masked_spec_cache}")
        
        # --- Stage 2: Calculate ACFs ---
        acf_results_cache = self._get_cache_path('acf_results')
        if os.path.exists(acf_results_cache):
            log.info(f"Loading cached ACF results from {acf_results_cache}")
            with open(acf_results_cache, 'rb') as f:
                self.acf_results = pickle.load(f)
        else:
            log.info("Calculating ACFs for all sub-bands...")
            self.acf_results = analysis.calculate_acfs_for_subbands(self.masked_spectrum, self.config)
            if self.config.get('pipeline_options', {}).get('save_intermediate_steps'):
                with open(acf_results_cache, 'wb') as f:
                    pickle.dump(self.acf_results, f)
                log.info(f"Saved ACF results to cache: {acf_results_cache}")

        # --- Stage 3: Fit Models and Derive Parameters ---
        if not self.acf_results or not self.acf_results['subband_acfs']:
            log.error("ACF results are empty, cannot proceed to fitting. Exiting.")
            return

        log.info("Fitting models and deriving final scintillation parameters...")
        self.final_results, self.all_subband_fits, self.powlaw_params = analysis.analyze_scintillation_from_acfs(
            self.acf_results, self.config
        )
        
        log.info("--- Pipeline execution finished. ---")