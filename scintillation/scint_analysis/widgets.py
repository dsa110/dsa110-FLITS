"""
Interactive widgets for scintillation analysis notebooks.

This module provides reusable ipywidgets-based interfaces for manual 
interactive analysis tasks, including window selection and ACF fitting.

Requirements
------------
- ipywidgets
- ipympl (for matplotlib widget backend: `%matplotlib widget`)
- matplotlib
- lmfit

Example Usage
-------------
>>> from scint_analysis import widgets, pipeline, config
>>> cfg = config.load_config("configs/bursts/burst_dsa.yaml")
>>> pipe = pipeline.ScintillationAnalysis(cfg)
>>> pipe.prepare_data()
>>> widgets.interactive_window_selector(pipe, cfg_path="configs/bursts/burst_dsa.yaml")

"""
from __future__ import annotations

import yaml
import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from lmfit.models import ConstantModel

try:
    from ipywidgets import (
        IntSlider, IntRangeSlider, FloatSlider, FloatRangeSlider, FloatText,
        Dropdown, Checkbox, VBox, HBox, Button, Output, Textarea, Label
    )
    from IPython.display import display
    _HAS_WIDGETS = True
except ImportError:
    _HAS_WIDGETS = False
    logging.warning("ipywidgets not available. Interactive widgets will not work.")

from .analysis import (
    lorentzian_component,
    gaussian_component,
    lorentzian_generalised,
    power_law_model,
)

__all__ = [
    "interactive_window_selector",
    "acf_fitter_dashboard",
]


# ==============================================================================
# Interactive Window Selector
# ==============================================================================

def interactive_window_selector(
    pipeline_obj,
    config_path: str | Path,
    initial_on_frac: tuple[float, float] = (0.5, 0.75),
    initial_off_frac: tuple[float, float] = (0.0, 0.25),
) -> VBox:
    """
    Launch an interactive widget for selecting on-pulse and off-pulse windows.
    
    This widget uses ipywidgets with matplotlib (ipympl backend) to provide
    a flicker-free interface for manually selecting burst and noise regions
    in the time series. Selected windows are saved back to the YAML config file.
    
    Parameters
    ----------
    pipeline_obj : ScintillationAnalysis
        Initialized pipeline object with `masked_spectrum` already loaded
        via `pipeline_obj.prepare_data()`.
    config_path : str or Path
        Path to the burst-specific YAML configuration file. Window selections
        will be written to the 'analysis.rfi_masking' section.
    initial_on_frac : tuple of float, optional
        Initial on-pulse window as fraction of total bins (start, end).
        Default is (0.5, 0.75) meaning middle 50-75% of data.
    initial_off_frac : tuple of float, optional
        Initial off-pulse window as fraction of total bins (start, end).
        Default is (0.0, 0.25) meaning first 25% of data.
    
    Returns
    -------
    VBox
        ipywidgets container. Use `display()` in Jupyter to show it.
    
    Notes
    -----
    Requires `%matplotlib widget` to be set in the notebook for interactive
    matplotlib figures (ipympl backend).
    
    The widget saves two keys to the YAML file:
    - `analysis.rfi_masking.manual_burst_window`: [start_bin, end_bin]
    - `analysis.rfi_masking.manual_noise_window`: [start_bin, end_bin]
    
    Examples
    --------
    >>> pipe = ScintillationAnalysis(config)
    >>> pipe.prepare_data()
    >>> widget = interactive_window_selector(pipe, "configs/bursts/freya_dsa.yaml")
    >>> display(widget)  # Or just call the function if in notebook
    """
    if not _HAS_WIDGETS:
        raise ImportError("ipywidgets is required for interactive_window_selector")
    
    if pipeline_obj.masked_spectrum is None:
        raise ValueError("Pipeline masked_spectrum is None. Call prepare_data() first.")
    
    config_path = Path(config_path)
    
    # Helper: persist slider choices back to YAML
    def _update_rfi_config(cfg_path: Path, key: str, val: Any) -> str:
        """Update a single key in the YAML config file."""
        try:
            with open(cfg_path, 'r') as fh:
                data = yaml.safe_load(fh) or {}
            data.setdefault('analysis', {}).setdefault('rfi_masking', {})[key] = val
            with open(cfg_path, 'w') as fh:
                yaml.dump(data, fh, default_flow_style=False, sort_keys=False)
            return f"Success! Updated '{key}' → {val}."
        except Exception as exc:
            return f"YAML update failed: {exc}"
    
    # Extract time series data
    spec = pipeline_obj.masked_spectrum
    time_series = spec.get_profile().filled(0)
    time_axis = spec.times
    max_bin = len(time_axis) - 1
    total_dur = time_axis[-1] - time_axis[0]
    centre_t = time_axis[np.argmax(time_series)]
    
    # Calculate initial window positions
    on_start = int(initial_on_frac[0] * max_bin)
    on_end = int(initial_on_frac[1] * max_bin)
    off_start = int(initial_off_frac[0] * max_bin)
    off_end = int(initial_off_frac[1] * max_bin)
    
    # Create widgets
    on_slider = IntRangeSlider(
        value=[on_start, on_end], min=0, max=max_bin, step=1,
        description='On-pulse bins:', layout={'width': '95%'}
    )
    off_slider = IntRangeSlider(
        value=[off_start, off_end], min=0, max=max_bin, step=1,
        description='Off-pulse bins:', layout={'width': '95%'}
    )
    zoom_slider = FloatSlider(
        value=total_dur, min=0.01*total_dur, max=total_dur,
        step=total_dur/200, description='Zoom width (s):',
        readout_format='.3f', layout={'width': '95%'}
    )
    save_btn = Button(
        description="Save windows to YAML",
        button_style='primary',
        icon='save'
    )
    status_lbl = Label("Adjust sliders to select regions.")
    plot_box = Output()  # container for the ipympl figure
    
    # Build figure once
    with plot_box:
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.canvas.header_visible = False
        display(fig.canvas)
        
        (ln,) = ax.plot(time_axis, time_series, color='navy', label='Profile')
        
        # Initial spans
        on_span = ax.axvspan(
            time_axis[on_slider.value[0]],
            time_axis[on_slider.value[1]],
            0, 1, color='cyan', alpha=0.4, label='On-pulse'
        )
        off_span = ax.axvspan(
            time_axis[off_slider.value[0]],
            time_axis[off_slider.value[1]],
            0, 1, color='orange', alpha=0.4, label='Off-pulse (noise)'
        )
        
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Power")
        ax.set_title("On/Off-pulse Window Selector")
        ax.grid(True, linestyle=':')
        ax.legend()
        ax.set_xlim(centre_t - zoom_slider.value/2, centre_t + zoom_slider.value/2)
    
    # Artist update callback
    def _update(_=None):
        """Update the plot when sliders change."""
        on_x0 = time_axis[on_slider.value[0]]
        on_x1 = time_axis[on_slider.value[1]]
        off_x0 = time_axis[off_slider.value[0]]
        off_x1 = time_axis[off_slider.value[1]]
        
        on_span.set_x(on_x0)
        on_span.set_width(on_x1 - on_x0)
        
        off_span.set_x(off_x0)
        off_span.set_width(off_x1 - off_x0)
        
        ax.set_xlim(centre_t - zoom_slider.value/2, centre_t + zoom_slider.value/2)
        fig.canvas.draw_idle()
    
    # YAML save callback
    def _save(_):
        """Save the current window selections to the YAML config."""
        on_win = list(on_slider.value)
        off_win = list(off_slider.value)
        m1 = _update_rfi_config(config_path, 'manual_burst_window', on_win)
        m2 = _update_rfi_config(config_path, 'manual_noise_window', off_win)
        status_lbl.value = f"{m1}  |  {m2}"
        
        # Also update the pipeline's in-memory config
        pipeline_obj.config['analysis']['rfi_masking']['manual_burst_window'] = on_win
        pipeline_obj.config['analysis']['rfi_masking']['manual_noise_window'] = off_win
    
    # Wire callbacks
    for w in (on_slider, off_slider, zoom_slider):
        w.observe(_update, names='value')
    save_btn.on_click(_save)
    
    # Initial draw
    _update()
    
    # Build and return the dashboard
    dashboard = VBox([
        zoom_slider,
        on_slider,
        off_slider,
        plot_box,
        HBox([save_btn, status_lbl])
    ])
    
    # Auto-display if in notebook
    display(dashboard)
    return dashboard


# ==============================================================================
# ACF Fitter Dashboard
# ==============================================================================

# Default model configuration
DEFAULT_MODEL_CONFIG = {
    "Lorentzian": dict(
        func=lorentzian_component,
        prefix="l_",
        param_names=["gamma", "m"]
    ),
    "Gaussian": dict(
        func=gaussian_component,
        prefix="g_",
        param_names=["sigma", "m"]
    ),
    "Gen-Lorentz": dict(
        func=lorentzian_generalised,
        prefix="lg_",
        param_names=["gamma", "alpha", "m"]
    ),
    "Power-Law": dict(
        func=power_law_model,
        prefix="p_",
        param_names=["c", "n"]
    ),
}


def acf_fitter_dashboard(
    acf_results: Dict[str, Any],
    config_path: str | Path,
    model_config: Optional[Dict] = None,
    lower_width_limit: float = 0.06,
) -> VBox:
    """
    Launch an interactive dashboard for manual ACF fitting with 1-3 components.
    
    This widget provides a comprehensive interface for:
    - Selecting sub-bands
    - Choosing 1-3 model components (Lorentzian, Gaussian, Gen-Lorentz, Power-law)
    - Adjusting initial parameter guesses with sliders
    - Fitting composite models with lmfit
    - Visualizing fits with live matplotlib plots (ipympl)
    - Saving fit results to YAML config
    
    Parameters
    ----------
    acf_results : dict
        Dictionary containing ACF data from the pipeline, typically loaded from
        a pickled file. Must contain:
        - 'subband_acfs': list of ACF arrays
        - 'subband_lags_mhz': list of lag arrays (MHz)
        - 'subband_center_freqs_mhz': list of center frequencies
        - 'subband_acfs_err': (optional) list of error arrays
    config_path : str or Path
        Path to burst-specific YAML config where fit results will be saved.
    model_config : dict, optional
        Custom model configuration dictionary. If None, uses DEFAULT_MODEL_CONFIG.
        Format: {name: {'func': callable, 'prefix': str, 'param_names': list}}
    lower_width_limit : float, optional
        Hard lower bound for gamma/sigma parameters. Default is 0.06 MHz.
    
    Returns
    -------
    VBox
        The complete dashboard widget. Call `display()` to show it.
    
    Notes
    -----
    Requires `%matplotlib widget` for interactive plots (ipympl backend).
    
    The dashboard saves fit results to the YAML config under:
    `analysis.stored_fits.subband_{i}.{model_name}`
    
    Each fit entry contains:
    - reference_frequency_mhz: Center frequency of the sub-band
    - best_fit_params: Dictionary of parameter values and errors
    - redchi: Reduced chi-squared
    - bic: Bayesian Information Criterion
    - fit_range_mhz: Lag range used for fitting
    
    Examples
    --------
    >>> import pickle
    >>> with open("data/cache/freya_acf_results.pkl", "rb") as f:
    ...     acf_data = pickle.load(f)
    >>> widget = acf_fitter_dashboard(acf_data, "configs/bursts/freya_dsa.yaml")
    >>> display(widget)
    """
    if not _HAS_WIDGETS:
        raise ImportError("ipywidgets is required for acf_fitter_dashboard")
    
    config_path = Path(config_path)
    MODEL_CONFIG = model_config or DEFAULT_MODEL_CONFIG
    LOWER_WIDTH = lower_width_limit
    
    # Extract data
    num_subbands = len(acf_results["subband_acfs"])
    logging.info(f"Loaded ACFs for {num_subbands} sub-bands.")
    
    # Storage for fit results
    all_fits = {}
    
    # =========================================================================
    # Create Widgets
    # =========================================================================
    
    subband_slider = IntSlider(
        0, 0, num_subbands - 1, 1, description="Sub-band:"
    )
    
    prim_dd = Dropdown(
        options=list(MODEL_CONFIG.keys()),
        value="Lorentzian",
        description="Model 1:"
    )
    add2_cb = Checkbox(False, description="+2nd")
    sec_dd = Dropdown(
        options=list(MODEL_CONFIG.keys()),
        value="Gaussian",
        description="Model 2:",
        disabled=True
    )
    add3_cb = Checkbox(False, description="+3rd")
    thr_dd = Dropdown(
        options=list(MODEL_CONFIG.keys()),
        value="Power-Law",
        description="Model 3:",
        disabled=True
    )
    
    # Enable/disable secondary model dropdowns
    add2_cb.observe(
        lambda ch: setattr(sec_dd, "disabled", not ch["new"]),
        names="value"
    )
    add3_cb.observe(
        lambda ch: setattr(thr_dd, "disabled", not ch["new"]),
        names="value"
    )
    
    # Fit range controls
    max_lag = float(max(np.max(np.abs(l)) for l in acf_results["subband_lags_mhz"]))
    fit_min_txt = FloatText(-max_lag, description="Min Lag:")
    fit_max_txt = FloatText(max_lag, description="Max Lag:")
    fit_rng_slider = FloatRangeSlider(
        value=(-max_lag, max_lag),
        min=-max_lag,
        max=max_lag,
        step=0.1,
        description="Fit Range Slider:"
    )
    
    ignore_err_cb = Checkbox(False, description="Ignore σ")
    
    # Action buttons
    fit_btn = Button(
        description="Perform Fit",
        button_style="success",
        icon="cogs"
    )
    print_btn = Button(
        description="Print Fits",
        button_style="info",
        icon="print"
    )
    save_btn = Button(
        description="Save→YAML",
        button_style="warning",
        icon="save"
    )
    
    # Output areas
    param_box = VBox([])
    plot_out = Output()
    results_tb = Textarea(layout={"width": "98%", "height": "200px"})
    status_lbl = Label()
    stats_lbl = Label("Fit Stats: N/A")
    
    # =========================================================================
    # Parameter Widget Factory
    # =========================================================================
    
    def _slider(desc: str, kind: str, val: float = 0.5):
        """Create appropriate slider/text widget for parameter type."""
        if kind == "width":
            return FloatSlider(
                value=max(val, LOWER_WIDTH),
                min=LOWER_WIDTH,
                max=15,
                step=0.01,
                description=desc
            )
        if kind == "m":
            return FloatSlider(
                value=val, min=0, max=4, step=0.01, description=desc
            )
        if kind == "c":
            return FloatSlider(
                value=0.0, min=0., max=0.5, step=0.001, description=desc
            )
        if kind == "alpha":
            return FloatText(value=5/3, description=desc)
        if kind == "n":
            return FloatSlider(
                value=-2, min=-6, max=-1, step=0.1, description=desc
            )
        return FloatText(value=val, description=desc)
    
    def _make_widgets(model_key: str, idx: int) -> list:
        """Generate parameter widgets for a specific model component."""
        cfg = MODEL_CONFIG[model_key]
        prefix = cfg['prefix']
        names = cfg["param_names"]
        widgets = []
        i = 0
        
        while i < len(names):
            nm = names[i]
            if nm.startswith(("gamma", "sigma")):
                w_width = _slider(f"{prefix}{idx}_{nm}", "width")
                w_width.observe(_on_visual_change, names="value")
                row = [w_width]
                
                # Check if next parameter is 'm' and group them
                if i + 1 < len(names) and names[i + 1].startswith("m"):
                    w_m = _slider(f"{prefix}{idx}_{names[i+1]}", "m")
                    w_m.observe(_on_visual_change, names="value")
                    row.append(w_m)
                    i += 1
                widgets.append(HBox(row))
                
            elif nm == "alpha":
                w_alpha = _slider(f"{prefix}{idx}_{nm}", "alpha")
                w_alpha.observe(_on_visual_change, names="value")
                widgets.append(w_alpha)
            elif nm.startswith("m"):
                w_m = _slider(f"{prefix}{idx}_{nm}", "m")
                w_m.observe(_on_visual_change, names="value")
                widgets.append(w_m)
            elif nm == "c":
                w_c = _slider(f"{prefix}{idx}_{nm}", "c")
                w_c.observe(_on_visual_change, names="value")
                widgets.append(w_c)
            elif nm == "n":
                w_n = _slider(f"{prefix}{idx}_{nm}", "n")
                w_n.observe(_on_visual_change, names="value")
                widgets.append(w_n)
            i += 1
        
        return widgets
    
    def _refresh_param_box(*_):
        """Rebuild parameter widgets based on selected models."""
        children = _make_widgets(prim_dd.value, 1)
        if add2_cb.value:
            children += _make_widgets(sec_dd.value, 2)
        if add3_cb.value:
            children += _make_widgets(thr_dd.value, 3)
        param_box.children = tuple(children)
        _draw_plot(initial_only=True)
    
    # Wire model selection to param box refresh
    for w in (prim_dd, sec_dd, thr_dd, add2_cb, add3_cb):
        w.observe(_refresh_param_box, names="value")
    
    # =========================================================================
    # Helper Functions
    # =========================================================================
    
    def _param_vals() -> list:
        """Extract current values from all parameter widgets."""
        vals = []
        for child in param_box.children:
            if isinstance(child, HBox):
                vals += [wid.value for wid in child.children]
            else:
                vals.append(child.value)
        return vals
    
    def _auto_bounds(params):
        """Set automatic parameter bounds based on parameter type."""
        for name, par in params.items():
            if any(k in name for k in ("gamma", "sigma")):
                par.set(min=LOWER_WIDTH)
            elif name.endswith("m") or name.endswith("_m"):
                par.set(min=0)
            elif "alpha" in name:
                par.set(min=0.1, max=6)
    
    # =========================================================================
    # Plot Area (create once, update in-place)
    # =========================================================================
    
    with plot_out:
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.canvas.header_visible = False
        display(fig.canvas)
    
    def _draw_plot(initial_only: bool = False, fit_res=None):
        """Refresh the live figure; avoids flicker by re-using canvas."""
        sb = subband_slider.value
        lags = acf_results["subband_lags_mhz"][sb]
        acf = acf_results["subband_acfs"][sb]
        errs = acf_results.get("subband_acfs_err", [None]*num_subbands)[sb]
        
        m0, m1 = fit_rng_slider.value
        mask = (lags >= m0) & (lags <= m1) & (lags != 0)
        x_fit, y_fit = lags[mask], acf[mask]
        
        ax.clear()
        ax.errorbar(
            lags, acf,
            yerr=None if (errs is None or ignore_err_cb.value) else errs,
            fmt=".", ms=3, capsize=3, color="lightgray",
            label="All data", zorder=1
        )
        ax.plot(x_fit, y_fit, '-', color="purple", alpha=0.6,
                label="Fit range", zorder=2)
        
        # Initial guess preview
        if initial_only and x_fit.size:
            vals, idx, guess = _param_vals(), 0, np.zeros_like(x_fit)
            sequence = [prim_dd.value]
            if add2_cb.value:
                sequence.append(sec_dd.value)
            if add3_cb.value:
                sequence.append(thr_dd.value)
            
            for key in sequence:
                cfg = MODEL_CONFIG[key]
                npar = len(cfg["param_names"])
                guess += cfg["func"](x_fit, *vals[idx:idx+npar])
                idx += npar
            ax.plot(x_fit, guess, color="pink", label="Initial guess")
        
        # Best fit
        if fit_res is not None:
            ax.plot(x_fit, fit_res.best_fit, "k-", lw=2.5,
                   label="Best Composite Fit")
            for prefix, comp in fit_res.eval_components(x=x_fit).items():
                if 'tpl' in prefix:
                    ax.plot(x_fit, comp, 'm--', lw=2,
                           label='Noise Template Fit')
        
        ax.set_ylim(np.min(y_fit)-0.05, np.max(y_fit)+0.05)
        ax.set_xlim(m0 - 0.05*(m1-m0), m1 + 0.05*(m1-m0))
        ax.set_xlabel("Δν (MHz)")
        ax.set_ylabel("C_I")
        ax.grid(ls=":")
        ax.legend(loc="upper right")
        fig.canvas.draw_idle()
    
    def _on_visual_change(_=None):
        """Trigger plot update when parameters change."""
        _draw_plot(initial_only=True)
    
    # =========================================================================
    # Lag Range Sync (textbox ↔ slider)
    # =========================================================================
    
    def _sync_text_to_slider(_):
        """Update slider when textboxes change."""
        if fit_min_txt.value < fit_max_txt.value:
            fit_rng_slider.value = (fit_min_txt.value, fit_max_txt.value)
    
    def _sync_slider_to_text(ch):
        """Update textboxes when slider changes."""
        fit_min_txt.value = round(ch["new"][0], 2)
        fit_max_txt.value = round(ch["new"][1], 2)
    
    fit_min_txt.observe(_sync_text_to_slider, names="value")
    fit_max_txt.observe(_sync_text_to_slider, names="value")
    fit_rng_slider.observe(_sync_slider_to_text, names="value")
    fit_rng_slider.observe(_on_visual_change, names="value")
    fit_min_txt.observe(_on_visual_change, names="value")
    fit_max_txt.observe(_on_visual_change, names="value")
    subband_slider.observe(_on_visual_change, names="value")
    
    # =========================================================================
    # Fit Callback
    # =========================================================================
    
    def _on_fit(_):
        """Execute the fit with current parameters and models."""
        sb = subband_slider.value
        lags = acf_results["subband_lags_mhz"][sb]
        acf = acf_results["subband_acfs"][sb]
        errs = acf_results.get("subband_acfs_err", [None]*num_subbands)[sb]
        
        m0, m1 = fit_rng_slider.value
        mask = (lags >= m0) & (lags <= m1) & (lags != 0)
        x, y = lags[mask], acf[mask]
        wts = None if (ignore_err_cb.value or errs is None) else \
              1.0 / np.maximum(errs[mask], 1e-9)
        
        # Build composite model
        comps = [prim_dd.value]
        if add2_cb.value:
            comps.append(sec_dd.value)
        if add3_cb.value:
            comps.append(thr_dd.value)
        
        signal_model, params, vals, idx = None, None, _param_vals(), 0
        
        for i, key in enumerate(comps, start=1):
            cfg = MODEL_CONFIG[key]
            m = Model(cfg["func"], prefix=f"{cfg['prefix']}{i}_")
            p = m.make_params()
            n = len(cfg["param_names"])
            
            for val, pname in zip(vals[idx:idx+n], p.keys()):
                p[pname].set(value=val)
            idx += n
            
            _auto_bounds(p)
            
            if signal_model is None:
                signal_model, params = m, p
            else:
                signal_model = signal_model + m
                params.update(p)
        
        # Add constant offset
        const_model = ConstantModel(prefix='c_')
        if params is None:
            params = const_model.make_params(c=0.0)
        else:
            params.update(const_model.make_params(c=0.0))
        
        composite_model = (signal_model + const_model) if signal_model is not None \
                         else const_model
        
        # Perform fit
        result = composite_model.fit(y, params, x=x, weights=wts)
        all_fits[(sb, tuple(comps))] = result
        
        # Update stats
        mean_sigma = np.nan if wts is None else float(1/np.mean(wts))
        stats_lbl.value = (f"χ²ᵣ = {result.redchi:.3f}   |   "
                          f"BIC = {result.bic:.2f}   |   "
                          f"⟨σ⟩ = {mean_sigma:.4f}")
        
        _draw_plot(fit_res=result)
    
    # =========================================================================
    # Print / Save Callbacks
    # =========================================================================
    
    def _on_print(_):
        """Print all stored fits to the textarea."""
        out = []
        for (sb, comps), res in all_fits.items():
            cf = acf_results["subband_center_freqs_mhz"][sb]
            out.append(f"--- Sub-band {sb} @ {cf:.2f} MHz :: {' + '.join(comps)}")
            for n, p in res.params.items():
                serr = p.stderr if p.stderr is not None else 0
                out.append(f"  {n}: {p.value:.4g} ± {serr:.2g}")
            out.append("")
        results_tb.value = "\n".join(out) if out else "No fits yet."
    
    def _on_save(_):
        """Save the current fit to the YAML config."""
        sb = subband_slider.value
        comps = [prim_dd.value]
        if add2_cb.value:
            comps.append(sec_dd.value)
        if add3_cb.value:
            comps.append(thr_dd.value)
        
        key = (sb, tuple(comps))
        if key not in all_fits:
            status_lbl.value = "Run a fit first."
            return
        
        res = all_fits[key]
        tag = "+".join(comps)
        fit_range_mhz = list(fit_rng_slider.value)
        params_to_save = {
            n: {"value": float(p.value), "stderr": float(p.stderr or 0.)}
            for n, p in res.params.items()
        }
        
        entry = dict(
            reference_frequency_mhz=float(acf_results["subband_center_freqs_mhz"][sb]),
            best_fit_params=params_to_save,
            redchi=float(res.redchi),
            bic=float(res.bic),
            fit_range_mhz=fit_range_mhz,
        )
        
        cfg = yaml.safe_load(open(config_path)) or {}
        cfg.setdefault("analysis", {}) \
           .setdefault("stored_fits", {}) \
           .setdefault(f"subband_{sb}", {})[tag] = entry
        
        yaml.safe_dump(cfg, open(config_path, "w"), sort_keys=False)
        status_lbl.value = f"Saved under subband_{sb} :: {tag}"
    
    # =========================================================================
    # Wire Callbacks
    # =========================================================================
    
    fit_btn.on_click(_on_fit)
    print_btn.on_click(_on_print)
    save_btn.on_click(_on_save)
    
    # =========================================================================
    # Build Dashboard Layout
    # =========================================================================
    
    fit_range_box = VBox([
        Label("Fit Lag Range (MHz):"),
        HBox([fit_min_txt, fit_max_txt]),
        fit_rng_slider,
    ])
    
    dashboard = VBox([
        HBox([subband_slider, prim_dd, add2_cb, sec_dd, add3_cb, thr_dd]),
        fit_range_box,
        Label(r"Component Parameters γ, σ, m, α, n"),
        param_box,
        ignore_err_cb,
        HBox([fit_btn, print_btn, save_btn]),
        status_lbl,
        stats_lbl,
        plot_out,
        results_tb,
    ])
    
    # Initialize parameter widgets
    _refresh_param_box()
    
    # Auto-display
    display(dashboard)
    return dashboard
