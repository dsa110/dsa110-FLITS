#!/usr/bin/env python3
"""
Generalized diagnostic visualization for FRB scattering fits.

This module provides a reusable function to create comprehensive diagnostic plots
for scattering analysis results, including data, model, residuals, and fit parameters.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from scattering.scat_analysis.burstfit import FRBParams


def plot_scattering_diagnostic(
    data: np.ndarray,
    model: np.ndarray,
    time: np.ndarray,
    freq: np.ndarray,
    params: FRBParams,
    model_key: str,
    goodness_of_fit: Optional[Dict[str, float]] = None,
    processing_info: Optional[Dict[str, Any]] = None,
    burst_name: str = "FRB",
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (14, 10),
    dpi: int = 150,
    scale_model: bool = True,
    show: bool = False
) -> plt.Figure:
    """
    Create a comprehensive four-panel diagnostic plot for scattering analysis.
    
    This function generates a standardized visualization showing:
    1. Data dynamic spectrum
    2. Model dynamic spectrum
    3. Residual (Data - Model)
    4. Frequency-averaged time profiles
    5. Fit parameters and quality metrics (text summary)
    
    Parameters
    ----------
    data : np.ndarray
        Data dynamic spectrum, shape (n_freq, n_time)
    model : np.ndarray
        Model dynamic spectrum, shape (n_freq, n_time)
    time : np.ndarray
        Time axis in milliseconds, shape (n_time,)
    freq : np.ndarray
        Frequency axis in GHz, shape (n_freq,)
    params : FRBParams
        Best-fit model parameters
    model_key : str
        Model identifier (e.g., "M0", "M1", "M2", "M3")
    goodness_of_fit : dict, optional
        Dictionary with fit quality metrics:
        - 'chi2_reduced': Reduced chi-squared
        - 'r_squared': R-squared coefficient
        - 'quality_flag': Quality assessment string
    processing_info : dict, optional
        Dictionary with processing details:
        - 't_factor': Time downsampling factor
        - 'f_factor': Frequency downsampling factor
        - 'likelihood': Likelihood function used
        - 'fitting_method': Fitting method (e.g., "Nested sampling")
    burst_name : str, default "FRB"
        Name of the burst for plot title
    output_path : Path, optional
        Path to save the figure. If None, figure is not saved.
    figsize : tuple, default (14, 10)
        Figure size in inches (width, height)
    dpi : int, default 150
        Resolution for saved figure
    scale_model : bool, default True
        If True, scale model to match data peak intensity for visualization
    show : bool, default False
        If True, call plt.show() to display figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object
    
    Examples
    --------
    Basic usage with fit results:
    
    >>> from scattering.scat_analysis.burstfit import FRBModel, FRBParams
    >>> from scattering.scat_analysis.visualization import plot_scattering_diagnostic
    >>> 
    >>> # After running fit...
    >>> fig = plot_scattering_diagnostic(
    ...     data=dataset.data,
    ...     model=model_spectrum,
    ...     time=dataset.time,
    ...     freq=dataset.freq,
    ...     params=best_params,
    ...     model_key="M3",
    ...     goodness_of_fit={"chi2_reduced": 1.23, "r_squared": 0.95},
    ...     burst_name="FRB20191108A",
    ...     output_path=Path("output/diagnostics.png")
    ... )
    
    With full metadata:
    
    >>> fig = plot_scattering_diagnostic(
    ...     data=data,
    ...     model=model,
    ...     time=time,
    ...     freq=freq,
    ...     params=params,
    ...     model_key="M3",
    ...     goodness_of_fit={
    ...         "chi2_reduced": 1.15,
    ...         "r_squared": 0.97,
    ...         "quality_flag": "Excellent"
    ...     },
    ...     processing_info={
    ...         "t_factor": 4,
    ...         "f_factor": 32,
    ...         "likelihood": "Student-t",
    ...         "fitting_method": "Nested sampling (dynesty)"
    ...     },
    ...     burst_name="Freya",
    ...     output_path=Path("diagnostics/freya_fit.png"),
    ...     dpi=200
    ... )
    
    Notes
    -----
    - The function automatically handles intensity scaling and colormap ranges
    - Residuals use a diverging colormap (RdBu_r) centered at zero
    - Time profiles are overlaid with legend for easy comparison
    - All panels use consistent time/frequency axes for alignment
    """
    # Scale model if requested
    if scale_model:
        data_peak = np.max(np.sum(data, axis=0))
        model_peak = np.max(np.sum(model, axis=0))
        if model_peak > 0:
            scale_factor = data_peak / model_peak
            model_scaled = model * scale_factor
        else:
            model_scaled = model
            scale_factor = 1.0
    else:
        model_scaled = model
        scale_factor = 1.0
    
    # Calculate residuals
    residual = data - model_scaled
    
    # Generate time profiles
    data_prof = np.sum(data, axis=0)
    model_prof = np.sum(model_scaled, axis=0)
    residual_prof = data_prof - model_prof
    
    # Create figure with gridspec
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Determine intensity ranges
    vmin_data = np.percentile(data, 0.5)
    vmax_data = np.percentile(data, 99.5)
    res_vmax = np.percentile(np.abs(residual), 99)
    
    # Panel 1: Data dynamic spectrum
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(
        data, 
        aspect='auto', 
        origin='lower',
        extent=[time[0], time[-1], freq[0], freq[-1]],
        vmin=vmin_data, 
        vmax=vmax_data, 
        cmap='viridis',
        interpolation='nearest'
    )
    ax1.set_xlabel('Time (ms)', fontsize=11)
    ax1.set_ylabel('Frequency (GHz)', fontsize=11)
    ax1.set_title('Data', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Intensity')
    
    # Panel 2: Model dynamic spectrum
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(
        model_scaled,
        aspect='auto',
        origin='lower',
        extent=[time[0], time[-1], freq[0], freq[-1]],
        vmin=vmin_data,
        vmax=vmax_data,
        cmap='viridis',
        interpolation='nearest'
    )
    ax2.set_xlabel('Time (ms)', fontsize=11)
    ax2.set_ylabel('Frequency (GHz)', fontsize=11)
    ax2.set_title(f'Model ({model_key})', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Intensity')
    
    # Panel 3: Residual dynamic spectrum
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(
        residual,
        aspect='auto',
        origin='lower',
        extent=[time[0], time[-1], freq[0], freq[-1]],
        vmin=-res_vmax,
        vmax=res_vmax,
        cmap='RdBu_r',
        interpolation='nearest'
    )
    ax3.set_xlabel('Time (ms)', fontsize=11)
    ax3.set_ylabel('Frequency (GHz)', fontsize=11)
    ax3.set_title('Residual (Data - Model)', fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=ax3, label='Intensity')
    
    # Panel 4: Time profiles
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(time, data_prof, 'k-', linewidth=1.5, label='Data', alpha=0.7)
    ax4.plot(time, model_prof, 'r-', linewidth=2, label='Model', alpha=0.8)
    ax4.plot(time, residual_prof, 'b-', linewidth=1, label='Residual', alpha=0.6)
    ax4.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax4.set_xlabel('Time (ms)', fontsize=11)
    ax4.set_ylabel('Intensity', fontsize=11)
    ax4.set_title('Frequency-Averaged Profiles', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Parameters and fit quality
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Build parameter text
    param_text = f"""Best Model: {model_key}

Best-Fit Parameters:
  • Amplitude (c₀):          {params.c0:.2f}
  • Peak time (t₀):          {params.t0:.4f} ms
  • Intrinsic width (γ):     {params.gamma:.4f} ms
  • Width parameter (ζ):     {params.zeta:.6f} ms
  • Scattering τ(1 GHz):     {params.tau_1ghz:.4f} ms
  • Scattering index (α):    {params.alpha:.2f}
  • DM refinement (ΔDM):     {params.delta_dm:.6f} pc/cm³
"""
    
    # Add goodness of fit if provided
    if goodness_of_fit:
        param_text += f"""
Goodness of Fit:
  • χ²/dof:     {goodness_of_fit.get('chi2_reduced', 'N/A')}
  • R²:         {goodness_of_fit.get('r_squared', 'N/A')}
  • Quality:    {goodness_of_fit.get('quality_flag', 'N/A')}
"""
    
    # Add processing info if provided
    if processing_info:
        t_factor = processing_info.get('t_factor', 'N/A')
        f_factor = processing_info.get('f_factor', 'N/A')
        likelihood = processing_info.get('likelihood', 'N/A')
        method = processing_info.get('fitting_method', 'N/A')
        
        param_text += f"""
Processing:
  • Downsampling: {t_factor}× (time), {f_factor}× (freq)
  • Likelihood: {likelihood}
  • Fitting method: {method}
"""
    
    if scale_model and scale_factor != 1.0:
        param_text += f"""
Visualization:
  • Model scaled by {scale_factor:.3f}× for display
"""
    
    ax5.text(
        0.05, 0.95,
        param_text,
        transform=ax5.transAxes,
        fontsize=10,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    )
    
    # Overall title
    fig.suptitle(
        f'{burst_name} - Scattering Analysis Diagnostics',
        fontsize=14,
        fontweight='bold',
        y=0.995
    )
    
    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Saved diagnostic plot to: {output_path}")
    
    # Show if requested
    if show:
        plt.show()
    
    return fig
