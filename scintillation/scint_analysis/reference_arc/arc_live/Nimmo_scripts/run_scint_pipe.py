# run_scintillation_pipeline.py
#
# An automated pipeline for scintillation analysis. This script takes
# pre-processed spectral data as NumPy arrays and runs a full analysis,
# saving plots and results.

import numpy as np
import argparse
import json
import os

# Import the main analysis class from our functions library
from scintillation_analysis_functions import Scintillation

def main():
    """Main function to run the scintillation analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="A generalized pipeline for FRB scintillation analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Input Files ---
    parser.add_argument('--spec_on', type=str, required=True,
                        help="Path to the on-burst spectrum .npy file (1D array).")
    parser.add_argument('--spec_off', type=str, required=True,
                        help="Path to the off-burst noise spectrum .npy file (1D array).")
    parser.add_argument('--freqs', type=str, required=True,
                        help="Path to the channel frequencies .npy file in MHz (1D array).")

    # --- Analysis Parameters ---
    parser.add_argument('--event_name', type=str, default="MyFRB",
                        help="Name of the event for titles and filenames.")
    parser.add_argument('--model', type=str, default='double_lorentz',
                        choices=['single_lorentz', 'double_lorentz', 'triple_lorentz'],
                        help="ACF model to fit for the full-band analysis.")
    parser.add_argument('--n_subbands', type=int, default=16,
                        help="Number of sub-bands for frequency-resolved analysis.")
    parser.add_argument('--ref_freq', type=float, default=1400.0,
                        help="Reference frequency in MHz for power-law fit normalization.")
    
    # --- Output Parameters ---
    parser.add_argument('--outdir', type=str, default='./scintillation_results',
                        help="Directory to save output plots and results.")
    
    args = parser.parse_args()
    
    # --- Create output directory ---
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        print(f"Created output directory: {args.outdir}")

    # --- Load Data ---
    print("Loading data...")
    try:
        spectrum = np.load(args.spec_on)
        noise_spec = np.load(args.spec_off)
        freqs_mhz = np.load(args.freqs)
    except FileNotFoundError as e:
        print(f"Error: Input file not found. {e}")
        return

    # --- Initialize Scintillation Object ---
    scint_analyzer = Scintillation(
        spectrum=spectrum,
        freqs_mhz=freqs_mhz,
        noise_spec=noise_spec,
        event_name=args.event_name
    )

    # --- Run Full-Band Analysis ---
    scint_analyzer.run_full_band_acf()
    scint_analyzer.fit_acf(model_type=args.model)
    scint_analyzer.plot_acf_fit(xlim=(-0.5, 0.5), save_dir=args.outdir) # Example xlim

    # --- Run Sub-band Analysis ---
    scint_analyzer.run_subband_analysis(n_subbands=args.n_subbands)
    scint_analyzer.fit_and_plot_subbands(ref_freq_mhz=args.ref_freq, save_dir=args.outdir)

    # --- Save Results ---
    results_to_save = {}
    # Extract full-band fit results
    if scint_analyzer.fit_result:
        results_to_save['full_band_fit'] = scint_analyzer.fit_result.params.valuesdict()
    # Extract sub-band fit results
    if scint_analyzer.subband_results:
        pl_fit = scint_analyzer.subband_results.get('power_law_fit')
        if pl_fit:
            results_to_save['sub_band_power_law_fit'] = {
                'alpha': pl_fit.params['alpha'].value,
                'alpha_err': pl_fit.params['alpha'].stderr,
                'amplitude_at_ref_freq': pl_fit.params['a'].value,
                'amplitude_err': pl_fit.params['a'].stderr
            }

    results_path = os.path.join(args.outdir, f"{args.event_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=4)
        
    print(f"\nAnalysis complete. All results saved in {args.outdir}")

if __name__ == '__main__':
    main()
