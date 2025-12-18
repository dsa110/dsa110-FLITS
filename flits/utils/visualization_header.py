import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# ==========================================
# 1. Design Configuration (Colors & Styles)
# ==========================================
# A refined, professional palette
C_BG = "#F4F6F7"        # Very subtle light gray background for the header strip
C_TEXT_PRIMARY = "#333333" # Nearly black for main text
C_TEXT_SECONDARY = "#777777" # Medium gray for supporting info
C_HIGHLIGHT_BLUE = "#0056b3" # A professional deep blue for selection
C_STATUS_RED = "#d9534f"   # A muted, professional red for FAIL
C_STATUS_GREEN = "#28a745" # A muted green for SUCCESS
C_DIVIDER = "#E0E0E0"      # Subtle gray for vertical lines

# Standard font properties for consistency
FONT_SANS = 'DejaVu Sans' # Or 'Helvetica', 'Arial' if available on your system
KW_TITLE = dict(fontname=FONT_SANS, fontsize=9, color=C_TEXT_SECONDARY, weight='bold', ha='left', va='top')
KW_BODY = dict(fontname=FONT_SANS, fontsize=10, color=C_TEXT_PRIMARY, ha='left', va='top')

def setup_header_axis(ax, show_divider=True):
    """Cleans the axis and adds a right-side divider line."""
    ax.axis('off')
    # Set limits to 0-1 for easy relative positioning
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    if show_divider:
        # Add a thin vertical line on the right edge
        line = mpatches.ConnectionPatch(xyA=(1, 0.1), xyB=(1, 0.9), 
                                        coordsA=ax.transAxes, coordsB=ax.transAxes,
                                        color=C_DIVIDER, linewidth=1)
        ax.add_artist(line)

def create_custom_header(fig, gs_header, burst_name, results):
    """
    Populates the 4-column header using the GridSpec provided.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The parent figure.
    gs_header : matplotlib.gridspec.GridSpecFromSubplotSpec
        The gridspec for the header row (should have 4 columns).
    burst_name : str
        Name of the burst.
    results : dict
        Dictionary containing fit results usually produced by the pipeline.
    """
    
    # --- Background Patch ---
    # Add a single rectangle behind all axes to unify them.
    # We add it to the figure instance, behind the axes (zorder=-1).
    # Coordinates are approximate relative to the GS location, but for simplicity
    # we can just let the background be white or rely on the figure background if needed.
    # The user's code used fig.transFigure which is tricky with sub-gridspecs.
    # We'll skip the rectangle for now or add it to the specific subplot axes if needed,
    # but the user's snippet adds it to the whole figure area defined by coordinates.
    # Given we are integrating into a larger figure, we can just style the axes.
    
    # --- Panel 1: Identity ---
    ax1 = fig.add_subplot(gs_header[0])
    setup_header_axis(ax1)
    
    ax1.text(0.05, 0.9, "OBSERVATION CONTEXT", **KW_TITLE)
    
    # Parse burst name if possible, otherwise use raw
    # Try to split if standard formatted name
    parts = burst_name.split('_')
    if len(parts) > 1:
        hero_name = parts[0]
        sub_name = "_".join(parts[1:])
    else:
        hero_name = burst_name
        sub_name = ""

    ax1.text(0.05, 0.65, f"FRB {hero_name}", fontname=FONT_SANS, fontsize=16, weight='bold', color=C_TEXT_PRIMARY, va='center')
    ax1.text(0.05, 0.45, sub_name, fontname=FONT_SANS, fontsize=10, weight='medium', color=C_TEXT_PRIMARY, va='center')
    
    # Secondary Content
    ax1.text(0.05, 0.25, "Observatory: CHIME/FRB" if "chime" in str(burst_name).lower() else "Observatory: DSA-110", 
             fontname=FONT_SANS, fontsize=10, color=C_TEXT_SECONDARY, va='center')


    # --- Panel 2: Model Selection ---
    ax2 = fig.add_subplot(gs_header[1])
    setup_header_axis(ax2)
    
    ax2.text(0.05, 0.9, "MODEL SELECTION", **KW_TITLE)
    
    best_key = results.get("best_key", "M3")
    
    # Construct model text dynamically
    # Ideally we'd have logZ for all models but we might only have the best one or a list.
    # For now, we highlight the best one.
    
    model_labels = []
    # If we have model comparison stats, use them. Otherwise just show best.
    # The user's example hardcoded values. We will format the BEST model generically.
    
    model_text = r"$\rightarrow \bf{\color{" + C_HIGHLIGHT_BLUE + "}{\mathsf{" + best_key + ": Selected}}}$" + "\n"
    # Placeholder for others if not available
    model_text += r"$\quad \color{" + C_TEXT_SECONDARY + "}{\mathsf{(Other models compared)}}$"
    
    ax2.text(0.05, 0.65, model_text, fontsize=11, ha='left', va='top', linespacing=1.6)


    # --- Panel 3: Evaluation ---
    ax3 = fig.add_subplot(gs_header[2])
    setup_header_axis(ax3)
    
    ax3.text(0.05, 0.9, "FIT EVALUATION", **KW_TITLE)
    
    # Determine status
    gof = results.get("goodness_of_fit", {})
    chi2_red = gof.get("chi2_reduced", 0.0)
    r2 = gof.get("r_squared", 0.0)
    
    # Simple logic: Fail if R2 < 0 or Chi2 > 5 (loose threshold)
    is_fail = (r2 < 0.0) or (chi2_red > 5.0)
    
    status_text = "FAIL" if is_fail else "SUCCESS"
    status_color = C_STATUS_RED if is_fail else C_STATUS_GREEN
    
    ax3.text(0.05, 0.65, "Status:", **KW_BODY)
    ax3.text(0.35, 0.65, status_text, fontname=FONT_SANS, fontsize=14, weight='bold', color=status_color, va='top')
    
    metrics_text = (
        r"$\mathsf{\chi_\nu^2 = " + f"{chi2_red:.2f}" + "}$" + "\n" +
        r"$\mathsf{R^2 = " + f"{r2:.3f}" + "}$"
    )
    ax3.text(0.05, 0.4, metrics_text, fontsize=11, color=C_TEXT_PRIMARY, ha='left', va='top', linespacing=1.5)


    # --- Panel 4: Parameters ---
    ax4 = fig.add_subplot(gs_header[3])
    setup_header_axis(ax4, show_divider=False)
    
    ax4.text(0.05, 0.9, "BEST FIT PARAMETERS", **KW_TITLE)
    
    # Format parameters
    best_params = results.get("best_params", None)
    if best_params:
        # We need a way to get uncertainties if available.
        # The params object might just be values.
        # If we have specific formatted strings in results, use those using regex or something.
        # Or just print values if uncertainties missing.
        
        # Check if we have formatted params in results (added in previous step?)
        # results['param_summary'] might be useful if we added it.
        # Otherwise, just show generic list.
        
        # Let's try to extract from keys
        p_str = r"$ \begin{aligned} "
        
        # Common params
        try:
            # Try to grab from best_params object
            c0 = getattr(best_params, 'c0', 0)
            t0 = getattr(best_params, 't0', 0)
            tau = getattr(best_params, 'tau_1ghz', 0)
            
            p_str += r"\mathsf{c_0} &\mathsf{= " + f"{c0:.1f}" + r"} \\ "
            p_str += r"\mathsf{t_0} &\mathsf{= " + f"{t0:.3f}" + r"} \\ "
            if tau > 0:
                p_str += r"\mathsf{\tau} &\mathsf{= " + f"{tau:.3f}" + r"} \\ "
            
        except:
            pass
            
        p_str += r"\end{aligned} $"
        
        # If we have the detailed summary table logic from `repoting.py`, we could use that.
        # For now, use the constructed string.
        ax4.text(0.05, 0.75, p_str, fontsize=10, color=C_TEXT_PRIMARY, ha='left', va='top')
    else:
        ax4.text(0.05, 0.5, "No parameters available", **KW_BODY)
