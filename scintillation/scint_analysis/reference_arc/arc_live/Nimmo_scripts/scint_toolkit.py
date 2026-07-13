import numpy as np
import matplotlib.pyplot as plt

import lmfit
from lmfit.models import ConstantModel
from scipy.optimize import curve_fit
from scipy.stats import linregress

# Linear function for fitting in log-log space
def linear_model(x, a, b):
    return a * x + b

def lorentzian_model_func(x, amp, fwhm):
    """
    A Lorentzian function parameterized by peak amplitude and FWHM.
    This matches the model in Nimmo et al. 2025, where amp = m^2.
    The center is assumed to be at x=0.
    """
    return amp / (1 + (x / (fwhm / 2))**2)

def lorentzian_with_offset_func(x, amp, fwhm, c):
    """
    A Lorentzian function parameterized by peak amplitude, FWHM,
    and a constant vertical offset 'c'.
    """
    return (amp / (1 + (x / (fwhm / 2))**2)) + c

def shift(arr, num, fill_value=np.nan):
    """Shifts an array by a number of bins, filling with a value."""
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def autocorr(x, norm=True, crop=False):
    """
    Computes the autocorrelation of a 1D array with proper NaN handling.
    """
    x_mean = np.nanmean(x)
    if np.nanstd(x) == 0: return np.zeros_like(x) # Return zeros if there is no variance
    x = x - x_mean
    result = np.zeros(len(x))
    for lag in range(len(x)):
        # Calculate the correlation for a given lag
        result[lag] = np.nansum(x * shift(x, lag))
    
    if norm and result[0] != 0:
        # Normalize by the variance (the zero-lag value)
        result /= result[0]
    
    # The result is symmetric, so we can make it perfectly so
    result = (result + result[::-1]) / 2
    
    if crop:
        # Return only the second half (positive lags)
        return result[len(x)//2:]
    return result

        
class ScintillationAnalyzer:
    """
    A toolkit for performing scintillation analysis on FRB dynamic spectra.
    """
    # __init__, _validate_config, set_on_off_spectra, apply_instrumental_correction,
    # compute_acf, run_full_band_analysis, plot_full_band_acf
    # (These methods are from the previous response and are included here for completeness)
    def __init__(self, dyn_spec, config):
        """
        Initializes the analyzer with a dynamic spectrum and configuration.
        """
        self.dyn_spec = dyn_spec
        self.config = config
        self._validate_config()
        self.freqs = np.linspace(
            self.config['freq_top_mhz'], 
            self.config['freq_bottom_mhz'], 
            self.config['nchan']
        )
        self.on_burst_spectrum = None
        self.off_burst_spectrum = None
        self.full_band_acf = None
        self.full_band_fit_result = None
        self.subband_results = None
        self.freq_dependence_fits = None

    def _validate_config(self):
        """Ensure all necessary keys are in the config dictionary."""
        required_keys = ['nchan', 'freq_top_mhz', 'freq_bottom_mhz', 'bw_mhz', 'tsamp_us']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required key in config: {key}")
        print("Configuration validated.")

    def set_on_off_spectra(self, on_pulse_bins, off_pulse_bins):
        """
        Computes the on- and off-burst spectra from the dynamic spectrum.
        """
        self.on_burst_spectrum = self.dyn_spec[:, on_pulse_bins].mean(axis=1)
        # We can use the off-burst spectrum for S/N calculations and RFI masking
        self.off_burst_spectrum = self.dyn_spec[:, off_pulse_bins].mean(axis=1)
        print("On- and off-burst spectra have been set.")
        return self
    
    def apply_rfi_mask(self, threshold_sigma=5):
        """
        Masks RFI by replacing channels with high noise with NaNs.
        This allows the autocorr function to ignore them.
        """
        if self.off_burst_spectrum is None:
            raise RuntimeError("Off-burst spectrum not set. Cannot determine RFI.")
        
        noise_mean = np.mean(self.off_burst_spectrum)
        noise_std = np.std(self.off_burst_spectrum)
        
        # Find channels where the off-burst power is too high
        rfi_indices = np.where(self.off_burst_spectrum > noise_mean + threshold_sigma * noise_std)[0]
        
        if len(rfi_indices) > 0:
            self.on_burst_spectrum[rfi_indices] = np.nan
            print(f"Masked {len(rfi_indices)} RFI channels.")
        
        return self

    def apply_instrumental_correction(self, correction_func, *args):
        """
        Applies a user-defined function to correct for instrumental effects.
        """
        self.on_burst_spectrum = correction_func(self.on_burst_spectrum, *args)
        print(f"Applied instrumental correction function: {correction_func.__name__}")
        return self

    def compute_acf(self, spectrum):
        """
        Wrapper for the autocorrelation() function.
        """
        # The paper's ACF is not normalized by (S-S_noise)^2, but rather by the variance
        # of the on-burst spectrum itself. The modulation index is recovered from the
        # amplitude of the *un-normalized* ACF. Let's compute both.
        
        if np.all(np.isnan(spectrum)):
            return None, None, None

        # The autocorr function returns the full, un-shifted ACF
        acf_full = autocorr(spectrum, norm=False)
        center_val = acf_full[0]

        if center_val == 0:
            return None, None, None

        # Use fftshift to correctly reorder the ACF around the zero lag.
        # This creates the symmetric view we want for fitting.
        acf_shifted = np.fft.fftshift(acf_full)
        
        # Use fftfreq and fftshift to generate the corresponding frequency lags.
        # This guarantees the lags and acf arrays have the exact same shape.
        chan_bw_mhz = self.config['bw_mhz'] / self.config['nchan']
        lags = np.fft.fftshift(np.fft.fftfreq(len(spectrum), d=chan_bw_mhz))
        
        return lags, acf_shifted, center_val

    def run_full_band_analysis(self, num_components, guess_fwhm_mhz, guess_amps):
        """
        Runs ACF and fitting for the full band using a dynamic lmfit composite model.

        Args:
            num_components (int): The number of Lorentzians to fit.
            guess_fwhm_mhz (list): List of initial FWHM guesses [MHz].
            guess_amps (list): List of initial amplitude guesses.
        """
        if self.on_burst_spectrum is None: raise RuntimeError("On-burst spectrum not set.")
        
        lags, acf, acf_center_val = self.compute_acf(self.on_burst_spectrum)
        
        # Dynamically build the composite model
        model = ConstantModel(prefix='const_') # Start with the constant offset
        for i in range(1, num_components + 1):
            prefix = f'c{i}_'
            comp_model = lmfit.Model(lorentzian_model_func, prefix=prefix)
            model += comp_model

        params = model.make_params()
        # Set initial guesses for the constant offset
        params['const_c'].set(value=np.median(acf), vary=True)
        
        # Set initial guesses for each Lorentzian component
        for i in range(1, num_components + 1):
            prefix = f'c{i}_'
            params[prefix + 'amp'].set(value=guess_amps[i-1], min=0)
            params[prefix + 'fwhm'].set(value=guess_fwhm_mhz[i-1], min=1e-6)

        self.full_band_fit_result = model.fit(acf, params, x=lags)
        
        print("--- Full-Band Fit Report ---")
        print(self.full_band_fit_result.fit_report(min_correl=0.5))
        return self
    
    def run_subband_analysis(self, num_subbands, num_components, guess_fwhm_mhz, fit_lag_range_mhz, subband_mode='equal_snr'):
        """
        Divides the band and runs analysis on each sub-band.

        Args:
            num_subbands (int): The number of sub-bands.
            num_components (int): The number of Lorentzians to fit.
            guess_fwhm_mhz (list): List of initial FWHM guesses [MHz].
            fit_lag_range_mhz (float): The +/- lag range [MHz] to use for fitting.
            subband_mode (str): Method for sub-banding.
                                'equal_snr' for equal S/N per band (default).
                                'equal_channels' for equal number of channels per band.
        """
        if self.on_burst_spectrum is None: raise RuntimeError("On-burst spectrum not set.")

        # --- FIX IS HERE: Select sub-banding mode ---
        if subband_mode == 'equal_snr':
            print("Using 'equal_snr' sub-banding mode.")
            snr_spectrum = (self.on_burst_spectrum - np.nanmean(self.off_burst_spectrum)) / np.nanstd(self.off_burst_spectrum)
            snr_spectrum[np.isnan(snr_spectrum) | (snr_spectrum < 0)] = 0
            cumulative_snr = np.cumsum(snr_spectrum)
            total_snr = cumulative_snr[-1]
            if total_snr == 0:
                print("Warning: No signal found for sub-band analysis.")
                self.subband_results, self.subband_acf_results = [], []
                return self
            subband_edges = [0]
            for i in range(1, num_subbands):
                subband_edges.append(np.searchsorted(cumulative_snr, i * (total_snr / num_subbands)))
            subband_edges.append(len(self.on_burst_spectrum))
        
        elif subband_mode == 'equal_channels':
            print("Using 'equal_channels' sub-banding mode.")
            num_channels = len(self.on_burst_spectrum)
            subband_edges = np.linspace(0, num_channels, num_subbands + 1, dtype=int)

        else:
            raise ValueError(f"Unknown subband_mode: {subband_mode}. Choose 'equal_snr' or 'equal_channels'.")

        # The rest of the function proceeds identically, regardless of how edges were calculated
        results, acf_results_for_plotting = [], []
        for i in range(num_subbands):
            start_chan, end_chan = subband_edges[i], subband_edges[i+1]
            if end_chan - start_chan < 50: continue

            sub_spectrum = self.on_burst_spectrum[start_chan:end_chan]
            lags, acf, acf_center_val = self.compute_acf(sub_spectrum)
            if lags is None: continue

            fit_mask = np.abs(lags) <= fit_lag_range_mhz
            lags_for_fit, acf_for_fit = lags[fit_mask], acf[fit_mask]

            model = lmfit.Model(lorentzian_with_offset_func, prefix='c1_')
            for j in range(1, num_components):
                 model += lmfit.Model(lorentzian_model_func, prefix=f'c{j+1}_')

            params = model.make_params()
            params['c1_c'].set(value=np.median(acf_for_fit))
            for j in range(num_components):
                prefix = f'c{j+1}_'
                params[prefix + 'amp'].set(value=np.max(acf_for_fit) / num_components, min=0)
                params[prefix + 'fwhm'].set(value=guess_fwhm_mhz[j], min=1e-6)

            fit_result = model.fit(acf_for_fit, params, x=lags_for_fit)
            acf_results_for_plotting.append({'lags': lags, 'acf': acf, 'fit_result': fit_result})

            if fit_result.success:
                res = {'center_freq_mhz': np.nanmean(self.freqs[start_chan:end_chan])}
                for j in range(1, num_components + 1):
                    prefix = f'c{j}_'
                    amp = fit_result.params[prefix+'amp'].value
                    amp_err = fit_result.params[prefix+'amp'].stderr if fit_result.params[prefix+'amp'].stderr is not None else 0
                    fwhm = fit_result.params[prefix+'fwhm'].value
                    fwhm_err = fit_result.params[prefix+'fwhm'].stderr if fit_result.params[prefix+'fwhm'].stderr is not None else 0
                    mod_index = np.sqrt(amp / acf_center_val) if amp > 0 and acf_center_val > 0 else 0
                    mod_index_err = mod_index * (0.5 * amp_err / amp) if amp > 0 and amp_err > 0 else 0
                    res[f'mod_index_comp{j}'] = mod_index
                    res[f'mod_index_comp{j}_err'] = mod_index_err
                    res[f'fwhm_mhz_comp{j}'] = fwhm
                    res[f'fwhm_mhz_comp{j}_err'] = fwhm_err
                results.append(res)
        
        self.subband_results = results
        self.subband_acf_results = acf_results_for_plotting
        print(f"Sub-band analysis complete. Found results for {len(results)}/{num_subbands} sub-bands.")
        return self

    def fit_frequency_dependence(self, num_components): # Pass num_components explicitly
        if not self.subband_results: raise RuntimeError("Sub-band analysis has not been run.")
            
        self.freq_dependence_fits = {}
        for i in range(1, num_components + 1): # Use the passed argument
            freqs, fwhms, fwhm_errs = [], [], []
            for res in self.subband_results:
                if res.get(f'fwhm_mhz_comp{i}_err', 0) > 0:
                    freqs.append(res['center_freq_mhz'])
                    fwhms.append(res[f'fwhm_mhz_comp{i}'])
                    fwhm_errs.append(res[f'fwhm_mhz_comp{i}_err'])
            
            if len(freqs) < 2:
                print(f"Warning: Not enough data points to fit frequency dependence for component {i}.")
                continue

            log_freqs, log_fwhms = np.log10(freqs), np.log10(fwhms)
            log_fwhm_errs = np.array(fwhm_errs) / (np.array(fwhms) * np.log(10))

            try:
                popt, pcov = curve_fit(linear_model, log_freqs, log_fwhms, sigma=log_fwhm_errs, absolute_sigma=True)
                perr = np.sqrt(np.diag(pcov))
                self.freq_dependence_fits[f'comp{i}'] = {'alpha': popt[0], 'alpha_err': perr[0], 'log10_A': popt[1]}
                print(f"--- Frequency Dependence Fit: Component {i} ---")
                print(f"  Scintillation index (alpha) = {popt[0]:.2f} +/- {perr[0]:.2f}")
            except Exception as e:
                print(f"Could not perform frequency dependence fit for component {i}: {e}")
        return self

    def plot_subband_acf_fits(self, lag_extent_mhz=0.2):
        """
        Plots the ACF and the multi-component fit for each sub-band.
        (UPDATED with manual symmetric construction for perfect centering)
        """
        if not self.subband_acf_results:
            print("No sub-band ACF results to plot.")
            return

        num_plots = len(self.subband_acf_results)
        if num_plots == 0: return
        
        num_cols = 4
        num_rows = (num_plots + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 3), 
                                 sharex=True, sharey=True, squeeze=False)
        axes = axes.flatten()

        for i, res in enumerate(self.subband_acf_results):
            ax = axes[i]
            # The full, original lags and acf from the fit result
            lags, fit_result = res['lags'], res['fit_result']

            # --- Manually build a symmetric plot array ---
            
            # 1. Get the ACF for positive lags only, excluding zero.
            # We assume the full un-shifted ACF is stored in fit_result.data
            full_acf_unshifted = np.fft.ifftshift(fit_result.data)
            max_lag_idx = len(full_acf_unshifted) // 2
            
            # Get ACF and lags for positive side (lag 1, 2, 3...)
            positive_acf = full_acf_unshifted[1:max_lag_idx]
            chan_bw_mhz = self.config['bw_mhz'] / self.config['nchan']
            positive_lags = (np.arange(len(positive_acf)) + 1) * chan_bw_mhz
            
            # 2. Build the symmetric arrays for plotting, which have a gap at zero
            plot_lags = np.concatenate([-positive_lags[::-1], positive_lags])
            plot_acf = np.concatenate([positive_acf[::-1], positive_acf])

            # 3. Plot the data as a step plot, which will now be perfectly centered
            ax.step(plot_lags, plot_acf, where='mid', color='black', label='ACF Data')

            # 4. The fit is plotted over the original, continuous lags from the fit
            if fit_result and fit_result.success:
                smooth_lags = np.linspace(plot_lags[0], plot_lags[-1], 1000)
                smooth_fit = fit_result.eval(x=smooth_lags)
                ax.plot(smooth_lags, smooth_fit, 'r-', label='Total Fit', alpha=0.8)
            
            ax.axvline(0, color='grey', linestyle='--', alpha=0.7)
            
            if self.subband_results and i < len(self.subband_results):
                center_freq = self.subband_results[i]['center_freq_mhz']
                ax.set_title(f'Sub-band {i+1}\n~{center_freq:.0f} MHz')

            ax.grid(True, alpha=0.5)
            ax.set_xlim(-lag_extent_mhz, lag_extent_mhz)

        # Clean up unused axes and add labels/legend
        for ax_idx in range(num_plots, len(axes)):
            axes[ax_idx].axis('off')
        fig.supxlabel('Frequency Lag (MHz)')
        fig.supylabel('Autocorrelation')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', fontsize='small')
        plt.tight_layout(rect=[0, 0, 0.88, 1])
        plt.show()
        
    def plot_subband_results(self, num_components): # Pass num_components explicitly
        if not self.subband_results: raise RuntimeError("Cannot plot: no sub-band results.")
        
        num_plots = num_components * 2
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True, squeeze=False)
        axes = axes.flatten()
        fig.suptitle('Sub-band Scintillation Analysis', fontsize=16)

        for i in range(1, num_components + 1):
            ax_fwhm, ax_mod = axes[(i-1)*2], axes[(i-1)*2 + 1]
            
            freqs = [res['center_freq_mhz'] for res in self.subband_results]
            fwhms = [res.get(f'fwhm_mhz_comp{i}', np.nan) * 1000 for res in self.subband_results]
            fwhm_errs = [res.get(f'fwhm_mhz_comp{i}_err', 0) * 1000 for res in self.subband_results]
            mod_indices = [res.get(f'mod_index_comp{i}', np.nan) for res in self.subband_results]
            mod_errs = [res.get(f'mod_index_comp{i}_err', 0) for res in self.subband_results]

            ax_fwhm.errorbar(freqs, fwhms, yerr=fwhm_errs, fmt='o', capsize=5, label=f'Comp {i} Data')
            if self.freq_dependence_fits and f'comp{i}' in self.freq_dependence_fits:
                fit = self.freq_dependence_fits[f'comp{i}']
                alpha, alpha_err, log10_A = fit['alpha'], fit['alpha_err'], fit['log10_A']
                fit_freqs = np.linspace(self.freqs.min(), self.freqs.max(), 100)
                fit_fwhms = (10**log10_A) * (fit_freqs**alpha)
                ax_fwhm.plot(fit_freqs, fit_fwhms * 1000, 'r--', label=f'Fit ($\\alpha={alpha:.2f} \\pm {alpha_err:.2f}$)')
            
            ax_fwhm.set_ylabel(f'Bandwidth (kHz)\nComponent {i}'); ax_fwhm.legend(); ax_fwhm.grid(True)

            ax_mod.errorbar(freqs, mod_indices, yerr=mod_errs, fmt='o', capsize=5, label=f'Comp {i} Data')
            ax_mod.set_ylabel(f'Modulation Index\nComponent {i}'); ax_mod.set_ylim(0); ax_mod.legend(); ax_mod.grid(True)
        
        axes[-1].set_xlabel('Center Frequency (MHz)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()