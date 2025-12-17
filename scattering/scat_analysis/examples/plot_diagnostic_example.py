#!/usr/bin/env python3
"""
Example: Using the generalized scattering diagnostic plot function

This script demonstrates how to use plot_scattering_diagnostic() to create
standardized diagnostic plots for any FRB scattering fit.
"""
import json
import numpy as np
from pathlib import Path

from scattering.scat_analysis import plot_scattering_diagnostic
from scattering.scat_analysis.burstfit import FRBModel, FRBParams


def plot_from_fit_results(
    results_json: Path,
    data_npy: Path,
    time: np.ndarray,
    freq: np.ndarray,
    burst_name: str = "FRB",
    output_path: Optional[Path] = None
):
    """
    Generate diagnostic plot from saved fit results and data.
    
    Parameters
    ----------
    results_json : Path
        Path to JSON file with fit results (must contain 'best_params', 
        'best_model', and optionally 'goodness_of_fit')
    data_npy : Path
        Path to numpy file with preprocessed data (n_freq, n_time)
    time : np.ndarray
        Time axis in milliseconds
    freq : np.ndarray  
        Frequency axis in GHz
    burst_name : str
        Name of the burst for plot title
    output_path : Path, optional
        Where to save the plot
    """
    # Load results
    with open(results_json) as f:
        results = json.load(f)
    
    # Extract parameters and metadata
    params = FRBParams(**results["best_params"])
    model_key = results["best_model"]
    goodness_of_fit = results.get("goodness_of_fit")
    
    # Load data
    data = np.load(data_npy)
    
    # Generate model
    model_gen = FRBModel(time=time, freq=freq, data=data)
    model = model_gen(params, model_key)
    
    # Create plot
    fig = plot_scattering_diagnostic(
        data=data,
        model=model,
        time=time,
        freq=freq,
        params=params,
        model_key=model_key,
        goodness_of_fit=goodness_of_fit,
        processing_info={
            "likelihood": "Student-t",
            "fitting_method": "Nested sampling"
        },
        burst_name=burst_name,
        output_path=output_path,
        dpi=150
    )
    
    return fig


# Example usage
if __name__ == "__main__":
    # Example 1: Generate plot for Freya
    freya_fig = plot_from_fit_results(
        results_json=Path("scattering/scat_process/freya_chime_I_912_4067_32000b_cntr_bpc_fit_results.json"),
        data_npy=Path("data/chime/freya_chime_I_912_4067_32000b_cntr_bpc.npy"),
        # Note: You would need to provide the actual time/freq arrays
        # These would come from your pipeline or be reconstructed from config
        burst_name="Freya",
        output_path=Path("output/freya_diagnostic_new.png")
    )
    
    print("✓ Generated Freya diagnostic plot")
    
    # Example 2: Use directly with arrays (after fitting)
    from scattering.scat_analysis import FRBModel, FRBParams
    
    # ... after running your fit ...
    # data, model_array, time, freq, best_params are available
    
    # fig = plot_scattering_diagnostic(
    #     data=data,
    #     model=model_array,
    #     time=time,
    #     freq=freq,
    #     params=best_params,
    #     model_key="M3",
    #     goodness_of_fit={
    #         "chi2_reduced": 1.15,
    #         "r_squared": 0.97,
    #         "quality_flag": "Excellent"
    #     },
    #     processing_info={
    #         "t_factor": 4,
    #         "f_factor": 32,
    #         "likelihood": "Student-t",
    #         "fitting_method": "Nested sampling (dynesty)"
    #     },
    #     burst_name="My FRB",
    #     output_path=Path("my_diagnostic.png")
    # )
