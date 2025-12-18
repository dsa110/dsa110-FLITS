import json
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from . import plotting

log = logging.getLogger(__name__)

def run_consistency_check(
    scat_results_path: str,
    scint_results_path: str,
    burst_id: str = None,
    c_factor: float = 1.16,
    output_dir: str = None
):
    """
    Load results from both scattering and scintillation pipelines and
    generate a consistency plot.
    
    Parameters
    ----------
    scat_results_path : str
        Path to scattering _fit_results.json
    scint_results_path : str
        Path to scintillation _analysis_results.json
    burst_id : str, optional
        ID for the burst (default derived from filenames)
    c_factor : float
        Proportionality constant C.
    output_dir : str, optional
        Directory to save results
    """
    # 1. Load Results
    try:
        with open(scat_results_path, 'r') as f:
            scat_results = json.load(f)
    except Exception as e:
        log.error(f"Failed to load scattering results: {e}")
        return

    try:
        with open(scint_results_path, 'r') as f:
            scint_results = json.load(f)
    except Exception as e:
        log.error(f"Failed to load scintillation results: {e}")
        return

    # 2. Extract burst ID if not provided
    if burst_id is None:
        burst_id = Path(scat_results_path).stem.split('_fit_results')[0]
    
    # 3. Prepare data for plotter
    # The scintillation JSON might have slightly different structure than expected by the plotter
    # We need to maps subband_measurements to subband_gamma correctly
    # Checking Scintillation JSON structure from previous view:
    # components -> component_1 -> subband_measurements -> [freq_mhz, bw, bw_err, ...]
    
    scint_plot_data = {
        'subband_center_freqs_mhz': [],
        'subband_gamma': [],
        'subband_gamma_err': []
    }
    
    # Try to find sub-band measurements in the best model/component
    # This assumes we want component_1 for now or a flattened list
    components = scint_results.get('components', {})
    for comp_id, comp_data in components.items():
        measurements = comp_data.get('subband_measurements', [])
        for m in measurements:
            scint_plot_data['subband_center_freqs_mhz'].append(m.get('freq_mhz'))
            scint_plot_data['subband_gamma'].append(m.get('bw'))
            scint_plot_data['subband_gamma_err'].append(m.get('bw_err', 0) if m.get('bw_err') is not None else 0)
    
    # Convert to arrays and sort by frequency
    nu = np.array(scint_plot_data['subband_center_freqs_mhz'])
    gamma = np.array(scint_plot_data['subband_gamma'])
    err = np.array(scint_plot_data['subband_gamma_err'])
    
    sort_idx = np.argsort(nu)
    scint_plot_data['subband_center_freqs_mhz'] = nu[sort_idx]
    scint_plot_data['subband_gamma'] = gamma[sort_idx]
    scint_plot_data['subband_gamma_err'] = err[sort_idx]

    # 4. Generate Plot
    if output_dir:
        save_path = Path(output_dir) / f"{burst_id}_scat_scint_consistency.png"
    else:
        save_path = f"{burst_id}_scat_scint_consistency.png"

    log.info(f"Generating consistency plot for {burst_id}...")
    fig = plotting.plot_scat_scint_consistency(
        scint_plot_data,
        scat_results,
        c_factor=c_factor,
        save_path=str(save_path)
    )
    
    return fig

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Scintillation-Scattering consistency check.")
    parser.add_argument("scat_json", type=str, help="Path to scattering fit results JSON.")
    parser.add_argument("scint_json", type=str, help="Path to scintillation analysis results JSON.")
    parser.add_argument("--burst_id", type=str, help="Burst ID.")
    parser.add_argument("--c_factor", type=float, default=1.16, help="Proportionality constant C.")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory.")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    run_consistency_check(
        args.scat_json,
        args.scint_json,
        burst_id=args.burst_id,
        c_factor=args.c_factor,
        output_dir=args.outdir
    )
