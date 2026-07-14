import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters, Minimizer, report_fit
from tqdm import tqdm

# --- Lorentzian Model Definitions ---
def lorentzian_model_1_comp(x, gamma1, m1, c1):
    """A single Lorentzian model with a constant offset."""
    return (m1**2 / (1 + (x / gamma1)**2)) + c1

def lorentzian_model_2_comp(x, gamma1, m1, gamma2, m2, c2):
    """A model of two Lorentzians with a shared constant offset."""
    lor1 = m1**2 / (1 + (x / gamma1)**2)
    lor2 = m2**2 / (1 + (x / gamma2)**2)
    return lor1 + lor2 + c2

# --- Helper Function for Fitting a Single ACF ---
def _fit_lorentzian_models_to_acf(acf, lags_mhz, fit_lagrange_mhz=0.5):
    """
    Fits both 1- and 2-component Lorentzian models to a single ACF.
    
    Returns:
        dict: A dictionary containing the lmfit result objects for both fits.
    """
    fit_results = {}
    fit_mask = np.abs(lags_mhz) <= fit_lagrange_mhz
    
    # Fit 1-Component Model
    try:
        model1 = Model(lorentzian_model_1_comp)
        params1 = model1.make_params(gamma1=0.05, m1=0.8, c1=0.0)
        params1['gamma1'].set(min=1e-6)
        params1['m1'].set(min=0)
        fit_results['fit_1_comp'] = model1.fit(acf[fit_mask], params1, x=lags_mhz[fit_mask])
    except Exception as e:
        print(f"Warning: 1-comp fit failed: {e}")
        fit_results['fit_1_comp'] = None

    # Fit 2-Component Model
    try:
        model2 = Model(lorentzian_model_2_comp)
        params2 = model2.make_params(gamma1=0.01, m1=0.5, gamma2=0.1, m2=0.5, c2=0.0)
        params2['gamma1'].set(min=1e-6)
        params2['gamma2'].set(min=1e-6)
        params2['m1'].set(min=0)
        params2['m2'].set(min=0)
        fit_results['fit_2_comp'] = model2.fit(acf[fit_mask], params2, x=lags_mhz[fit_mask])
    except Exception as e:
        print(f"Warning: 2-comp fit failed: {e}")
        fit_results['fit_2_comp'] = None
        
    return fit_results

# --- Helper Function for Model Selection ---
def _select_overall_best_model(all_subband_fits):
    """
    Determines the best overall model (1 vs 2 components) by comparing
    BIC values across all sub-bands and picking the majority winner.
    """
    votes = {1: 0, 2: 0}
    for fits in all_subband_fits:
        fit1 = fits.get('fit_1_comp')
        fit2 = fits.get('fit_2_comp')
        if fit1 and fit2:
            if fit1.bic < fit2.bic:
                votes[1] += 1
            else:
                votes[2] += 1
        elif fit1: # Only fit 1 succeeded
            votes[1] += 1
        elif fit2: # Only fit 2 succeeded
            votes[2] += 1
    
    # Return the model choice with the most votes
    return 1 if votes[1] >= votes[2] else 2

# --- Helper Functions for Plotting ---
def _plot_subband_fits(acf_results, all_subband_fits):
    """Generates a stacked plot of sub-band ACFs and their model fits."""
    num_subbands = len(acf_results['subband_acfs'])
    plt.figure(figsize=(8, 10))
    cmap = plt.get_cmap('plasma')
    
    for i in range(num_subbands):
        rgba = cmap(i / (num_subbands - 1))
        offset = i * 1.5
        lags = acf_results['subband_lags_mhz'][i]
        acf = acf_results['subband_acfs'][i]
        fits = all_subband_fits[i]
        
        plt.plot(lags, acf + offset, color=rgba, label=f"{acf_results['subband_center_freqs_mhz'][i]:.1f} MHz")
        
        # Plot the preferred fit (or best available)
        fit_to_plot = fits.get('fit_2_comp') or fits.get('fit_1_comp')
        if fit_to_plot:
            plt.plot(lags, fit_to_plot.eval(x=lags) + offset, 'k--', alpha=0.7)

    plt.yticks([(i * 1.5) for i in range(num_subbands)], [f"{cf:.1f}" for cf in acf_results['subband_center_freqs_mhz']])
    plt.ylabel("Center Frequency (MHz)")
    plt.xlabel("Frequency Lag (MHz)")
    plt.title("Sub-band ACF Fits")
    plt.xlim(-0.5, 0.5)
    plt.grid(True, alpha=0.2)
    plt.show()

def _plot_power_law_fit(frequencies, data_dict, fit_result, component_name):
    """Plots a single power-law fit."""
    plt.figure(figsize=(8, 6))
    
    # Plot data points with error bars
    plt.errorbar(frequencies, data_dict['bandwidths'], yerr=data_dict['errors'],
                 fmt='o', capsize=5, label=f'{component_name} Data')

    # Plot the fitted model
    fit_params = fit_result.params
    c, n = fit_params['c'].value, fit_params['n'].value
    freq_model = np.linspace(min(frequencies), max(frequencies), 100)
    scint_model = c * (freq_model ** n)
    plt.plot(freq_model, scint_model, 'r--', label=f'Fit: $\\Delta\\nu \\propto \\nu^{{{n:.2f}}}$')

    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Decorrelation Bandwidth (MHz)")
    plt.title(f"Power-Law Fit for {component_name} Component")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()

# --- Main Orchestrator Function ---
def analyze_scintillation_from_acfs(acf_results, reference_frequency_mhz=600.0, show_diagnostic_plots=True):
    """
    Analyzes a set of ACFs to derive scintillation parameters. This is a generic,
    instrument-agnostic replacement for the original `fit_subband_acfs`.

    Args:
        acf_results (dict): A dictionary containing ACF data, must include:
            'subband_acfs', 'subband_lags_mhz', 'subband_center_freqs_mhz'.
        reference_frequency_mhz (float): The frequency to which results are scaled.
        show_diagnostic_plots (bool): If True, generates and displays plots.

    Returns:
        dict: A dictionary containing the final derived scintillation parameters.
    """
    # 1. Fit both 1- and 2-component models to every sub-band ACF
    print("Fitting Lorentzian models to all sub-band ACFs...")
    all_subband_fits = [_fit_lorentzian_models_to_acf(acf, lags) for acf, lags in 
                        zip(acf_results['subband_acfs'], acf_results['subband_lags_mhz'])]

    # 2. Select the best overall model using BIC comparison
    best_model_choice = _select_overall_best_model(all_subband_fits)
    print(f"Model selection complete. Best overall model: {best_model_choice} component(s).")
    
    # 3. Extract physical parameters based on the best model choice
    final_params = {'narrow_comp': [], 'broad_comp': []} # For 2-comp model
    single_comp_params = [] # For 1-comp model
    
    for fits in all_subband_fits:
        fit_obj = fits.get(f'fit_{best_model_choice}_comp')
        if not fit_obj:
            if best_model_choice == 1: single_comp_params.append({})
            else: final_params['narrow_comp'].append({}); final_params['broad_comp'].append({})
            continue

        p = fit_obj.params
        if best_model_choice == 1:
            single_comp_params.append({'bw': p['gamma1'].value, 'mod': p['m1'].value, 'bw_err': p['gamma1'].stderr})
        else:
            # Sort the two components by bandwidth (gamma)
            gammas = [p['gamma1'].value, p['gamma2'].value]
            mods = [p['m1'].value, p['m2'].value]
            errs = [p['gamma1'].stderr, p['gamma2'].stderr]
            
            sort_indices = np.argsort(gammas)
            narrow_idx, broad_idx = sort_indices[0], sort_indices[1]
            
            final_params['narrow_comp'].append({'bw': gammas[narrow_idx], 'mod': mods[narrow_idx], 'bw_err': errs[narrow_idx]})
            final_params['broad_comp'].append({'bw': gammas[broad_idx], 'mod': mods[broad_idx], 'bw_err': errs[broad_idx]})

    # 4. Perform Power-Law fitting for each component
    final_results = {'best_model_choice': best_model_choice, 'components': {}}
    components_to_fit = []
    if best_model_choice == 1:
        components_to_fit.append(('scint_scale', single_comp_params))
    else:
        components_to_fit.append(('narrow_component', final_params['narrow_comp']))
        components_to_fit.append(('broad_component', final_params['broad_comp']))

    for name, params_list in components_to_fit:
        # Collate data for fitting
        freqs = np.array(acf_results['subband_center_freqs_mhz'])
        bws = np.array([p.get('bw', np.nan) for p in params_list])
        bw_errs = np.array([p.get('bw_err', np.nan) for p in params_list])

        # Filter out failed fits (NaNs)
        valid_mask = ~np.isnan(bws) & ~np.isnan(bw_errs)
        if np.sum(valid_mask) < 2:
            print(f"Skipping power-law fit for {name}: not enough valid data points.")
            continue
            
        # Perform the fit
        power_law_model = Model(lambda x, c, n: c * (x**n))
        fit_result = power_law_model.fit(bws[valid_mask], x=freqs[valid_mask], weights=1.0/bw_errs[valid_mask], c=1, n=4)
        
        # Calculate bandwidth at reference frequency
        c, n = fit_result.params['c'].value, fit_result.params['n'].value
        bw_at_ref = c * (reference_frequency_mhz ** n)

        final_results['components'][name] = {
            'power_law_fit_params': fit_result.params.valuesdict(),
            'scaling_index': n,
            'bw_at_ref_mhz': bw_at_ref
        }
        
        # Plotting
        if show_diagnostic_plots:
             _plot_power_law_fit(freqs[valid_mask], {'bandwidths': bws[valid_mask], 'errors': bw_errs[valid_mask]}, fit_result, name)

    # 5. Diagnostic plot of sub-band fits
    if show_diagnostic_plots:
        _plot_subband_fits(acf_results, all_subband_fits)

    return final_results
