"""
Reporting utilities for FLITS fitting pipeline.
"""
import logging
import numpy as np

log = logging.getLogger(__name__)

def print_fit_summary(results: dict):
    """
    Prints a consolidated summary table of the fit results to stdout.
    
    Args:
        results (dict): The results dictionary from the pipeline containing:
            - best_key (str)
            - goodness_of_fit (dict)
            - best_params (FRBParams or dict)
            - param_names (list)
            - flat_chain (array, optional)
    """
    # Extract main info
    model = results.get("best_key", "Unknown")
    gof = results.get("goodness_of_fit", {})
    chi2 = gof.get("chi2_reduced", float("nan"))
    r2 = gof.get("r_squared", float("nan"))
    quality = gof.get("quality_flag", "Unknown")
    
    params = results.get("best_params")
    param_names = results.get("param_names", [])
    flat_chain = results.get("flat_chain")
    
    # Prepare lines
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"FIT SUMMARY: Model {model}")
    lines.append("-" * 60)
    
    # Goodness of Fit Section
    lines.append("VIALIDATION STATUS:")
    lines.append(f"  Flag:        {quality}")
    lines.append(f"  Reduced Chi2: {chi2:.3f}")
    lines.append(f"  R-squared:    {r2:.4f}")
    
    # Detailed Diagnostics (if available from recent run)
    if "diagnostics" in results and isinstance(results["diagnostics"], dict):
        diag = results["diagnostics"]
        if "residual_analysis" in diag:
            # It's an object, try to extract fields if simple
            # (Users might just look at the plot, but basic flags help)
            pass

    lines.append("-" * 60)
    
    # Parameter Section
    lines.append(f"{'Parameter':<15} | {'Best Fit':<12} | {'Uncertainty':<12}")
    lines.append("-" * 45)
    
    for i, name in enumerate(param_names):
        val = np.nan
        err = np.nan
        
        # Attempt to get stats from chain
        if flat_chain is not None:
             # Ensure array
             arr = np.asanyarray(flat_chain)
             if arr.ndim == 2 and arr.shape[1] >= len(param_names):
                 # Check matching indices
                 vals = arr[:, i]
                 val = np.median(vals)
                 err = np.std(vals)
        
        # Fallback to best_params if val is nan
        if np.isnan(val):
            # Try attribute access
            if hasattr(params, name):
                val = getattr(params, name)
                err = 0.0
            # Try dict access
            elif isinstance(params, dict) and name in params:
                 val = params[name]
                 err = 0.0
            
        # Format string
        if np.isnan(val):
             s_val = "NaN"
             s_err = "NaN"
        else:
             s_val = f"{val:.5g}"
             s_err = f"{err:.5g}" if err > 0 else "N/A"
             
        lines.append(f"{name:<15} | {s_val:<12} | {s_err:<12}")
        
    lines.append("=" * 60)
    lines.append("")
    
    # Print to console
    print("\n".join(lines))
