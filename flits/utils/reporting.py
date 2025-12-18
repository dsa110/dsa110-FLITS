import math
import numpy as np
import logging

log = logging.getLogger(__name__)

def _format_with_uncertainty(val: float, err: float) -> str:
    """
    Formats value and error using parenthetical notation (e.g., 5.44(1)).
    """
    if np.isnan(val):
        return "NaN"
    if np.isnan(err) or err <= 0:
        return f"{val:.5g}"
        
    # Determine precision from error
    # Log10 of error gives magnitude
    try:
        power = math.floor(math.log10(err))
    except ValueError:
        return f"{val:.5g}"
        
    # We want to print up to the digit specified by power
    # decimal_places = -power
    
    decimal_places = -int(power)
    
    # Round value to that number of decimal places
    if decimal_places < 0:
        # Error is > 1 (e.g. 10). Round to nearest 10.
        # For simplicity in this context, we usually just want standard float formatting 
        # but let's stick to the logic:
        # err=12 (power 1). decimal_places = -1.
        # round(1234, -1) -> 1230.
        # We can just return standard scientific or fixed notation if needed
        # But let's try to support it.
        # val=1234, err=10. -> 1230(10).
        rounded_val = round(val, decimal_places)
        rounded_err = round(err, decimal_places)
        
        # If we are in >1 domain, maybe just use standard +/-? 
        # User asked for 5.4(1).
        # Let's handle the common case (decimals) strictly and fallback for huge numbers.
        return f"{val:.5g}({err:.5g})"
        
    # Standard decimal case
    fmt = f"{{:.{decimal_places}f}}"
    val_str = fmt.format(val)
    
    # Calculate error digit(s) in the last place
    # e.g. err=0.103 (pow -1, dec 1). 0.103 * 10^1 = 1.03 -> 1
    # e.g. err=0.096 (pow -2, dec 2). 0.096 * 10^2 = 9.6 -> 10
    err_digits = round(err * (10 ** decimal_places))
    
    # If error rounds to 0 (shouldn't happen with floor logic unless err very close to next magnitude down?), clamp to 1?
    if err_digits == 0: err_digits = 1
    
    return f"{val_str}({int(err_digits)})"

def get_fit_summary_lines(results: dict, table_format: str = "ascii") -> list[str]:
    """
    Returns the consolidated fit summary as a list of strings.
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
    
    lines = []
    
    if table_format == "markdown":
        lines.append(f"### Fit Summary: Model {model}")
        lines.append("")
        lines.append(f"**Validation Status**: {quality}")
        lines.append(f"- **Reduced Chi2**: `{chi2:.3f}`")
        lines.append(f"- **R-squared**: `{r2:.4f}`")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("| :--- | :--- |")
    else:  # ascii
        lines.append("")
        lines.append("=" * 60)
        lines.append(f"FIT SUMMARY: Model {model}")
        lines.append("-" * 60)
        lines.append("VIALIDATION STATUS:")
        lines.append(f"  Flag:        {quality}")
        lines.append(f"  Reduced Chi2: {chi2:.3f}")
        lines.append(f"  R-squared:    {r2:.4f}")
        lines.append("-" * 60)
        lines.append(f"{'Parameter':<15} | {'Value (Unc)':<20}")
        lines.append("-" * 45)
    
    for i, name in enumerate(param_names):
        val = np.nan
        err = np.nan
        
        # Attempt to get stats from chain
        if flat_chain is not None:
             arr = np.asanyarray(flat_chain)
             if arr.ndim == 2 and arr.shape[1] >= len(param_names):
                 vals = arr[:, i]
                 val = np.median(vals)
                 err = np.std(vals)
        
        # Fallback to best_params
        if np.isnan(val):
            if hasattr(params, name):
                val = getattr(params, name)
                err = 0.0
            elif isinstance(params, dict) and name in params:
                 val = params[name]
                 err = 0.0
            
        # Format string
        val_str = _format_with_uncertainty(val, err)
             
        if table_format == "markdown":
            lines.append(f"| `{name}` | {val_str} |")
        else:
            lines.append(f"{name:<15} | {val_str:<20}")
        
    if table_format != "markdown":
        lines.append("=" * 60)
        lines.append("")
        
    return lines

def print_fit_summary(results: dict, table_format: str = "ascii"):
    """
    Prints a consolidated summary table of the fit results to stdout.
    
    Args:
        results (dict): The results dictionary from the pipeline.
        table_format (str): 'ascii' (default) or 'markdown'.
    """
    lines = get_fit_summary_lines(results, table_format)
    # Print to console
    print("\n".join(lines))
