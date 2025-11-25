# ==============================================================================
# File: scint_analysis/run_analysis.py
# ==============================================================================
import argparse
import logging
import json
import numpy as np
from scint_analysis import config, pipeline, plotting

class NumpyJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for NumPy data types.
    This converts NumPy types to standard Python types for JSON serialization.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

def main():
    """
    Main function to run the scintillation analysis pipeline from the command line.
    """
    # 1. Set up Command-Line Argument Parser
    parser = argparse.ArgumentParser(description="Run a scintillation analysis pipeline on FRB data.")
    parser.add_argument("burst_config_path", type=str, help="Path to the burst-specific YAML configuration file.")
    args = parser.parse_args()

    # 2. Load Configuration
    try:
        loaded_config = config.load_config(args.burst_config_path)
    except Exception as e:
        print(f"Error: Could not load configuration. {e}")
        return

    # 3. Set up Logging
    log_level = loaded_config.get('pipeline_options', {}).get('log_level', 'INFO').upper()
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # 4. Initialize and Run the Pipeline
    # Resolve fallback paths for helper configs if not provided in YAML
    # This mirrors robust resolution used by the scattering CLI.
    from pathlib import Path as _P
    base_dir = _P(args.burst_config_path).parent
    def _resolve_cfg(base_dir: _P, filename: str) -> _P:
        cands = [
            base_dir / filename,
            base_dir.parent / filename,
            base_dir.parent.parent / filename,
        ]
        for c in cands:
            if c.exists():
                return c
        return cands[-1]

    loaded_config.setdefault('pipeline_options', {})
    # no explicit helper files here, but ensure any relative paths are rooted
    if 'input_data_path' in loaded_config:
        loaded_config['input_data_path'] = str(_P(loaded_config['input_data_path']))

    scint_pipeline = pipeline.ScintillationAnalysis(loaded_config)
    scint_pipeline.run()

    # 5. Handle and Save Results
    if not scint_pipeline.final_results:
        logging.error("Pipeline did not produce any results.")
        return

    # Save results to a JSON file using the custom encoder
    burst_id = loaded_config.get('burst_id', 'output')
    output_path = f"./{burst_id}_analysis_results.json"
    try:
        with open(output_path, 'w') as f:
            json.dump(scint_pipeline.final_results, f, indent=4, cls=NumpyJSONEncoder)
        logging.info(f"Final results successfully saved to {output_path}")
    except TypeError as e:
        logging.error(f"Could not serialize results to JSON. {e}")

    # 6. Generate Final Plots
    if scint_pipeline.acf_results and scint_pipeline.all_subband_fits:
        logging.info("Generating final analysis overview plot...")
        plotting.plot_analysis_overview(
            scint_pipeline.final_results, 
            scint_pipeline.acf_results, 
            scint_pipeline.all_subband_fits,
            scint_pipeline.all_powerlaw_fits
        )
    else:
        logging.warning("Intermediate results not available, skipping overview plot.")

if __name__ == "__main__":
    main()
