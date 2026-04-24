# scint_pipeline.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List
# Import the refactored functions
from scint_pipeline_funcs import (
    scrunch, upchannelize, calculate_acf_2d, fit_lorentzian_acf,
    calculate_secondary_spectrum, screen_distance_from_curvature,
    fit_scint_bandwidth_freq_relation, scintillation_bandwidth_to_timescale,
    effective_velocity
    # ... other necessary functions
)

class ScintillationAnalyser:
    """
    A class to perform scintillation analysis on Fast Radio Burst dynamic spectra.

    Attributes:
        dyn_spec (np.ndarray): Dynamic spectrum (time, frequency).
        freqs_mhz (np.ndarray): Array of channel center frequencies in MHz.
        time_res_s (float): Time resolution in seconds.
        freq_res_mhz (float): Frequency resolution (channel width) in MHz.
        source_name (str): Identifier for the source/observation.
        params (Dict[str, Any]): Dictionary to store analysis parameters.
        results (Dict[str, Any]): Dictionary to store analysis results.
    """
    def __init__(
        self,
        dyn_spec: np.ndarray,
        freqs_mhz: np.ndarray,
        time_res_s: float,
        source_name: str = "FRB"
        ):
        """
        Initialize the analyser.

        Args:
            dyn_spec: Dynamic spectrum array (time, freq).
            freqs_mhz: Array of channel center frequencies in MHz.
            time_res_s: Time resolution in seconds.
            source_name: Identifier for the source/observation.
        """
        if dyn_spec.ndim != 2:
            raise ValueError("dyn_spec must be 2D (time, freq).")
        if dyn_spec.shape[1] != len(freqs_mhz):
            raise ValueError("Number of frequency channels in dyn_spec must match length of freqs_mhz.")

        self.dyn_spec = dyn_spec
        self.freqs_mhz = freqs_mhz
        self.time_res_s = time_res_s
        # Calculate frequency resolution (assuming constant channel width)
        if len(freqs_mhz) > 1:
            self.freq_res_mhz = np.abs(np.median(np.diff(freqs_mhz)))
        else:
            self.freq_res_mhz = 0 # Or requires explicit input
            print("Warning: Cannot determine frequency resolution from single channel.")
        self.source_name = source_name

        self.params = {} # Store analysis parameters used
        self.results = {} # Store results like ACF, fits, secondary spectra

        print(f"Initialized Analyser for {self.source_name}")
        print(f"Data shape (time, freq): {self.dyn_spec.shape}")
        print(f"Time resolution: {self.time_res_s:.6f} s")
        print(f"Frequency range: {self.freqs_mhz.min():.2f} - {self.freqs_mhz.max():.2f} MHz")
        print(f"Frequency resolution: {self.freq_res_mhz:.6f} MHz")


    def preprocess(
        self,
        t_scrunch: int = 1,
        f_scrunch: int = 1,
        time_range: Optional[Tuple[int, int]] = None,
        freq_range_mhz: Optional[Tuple[float, float]] = None,
        apply_upchannel: bool = False,
        upchannel_params: Dict[str, Any] = {'fftsize': 32, 'downfreq': 2, 'downtime': 1}
        ) -> None:
        """
        Preprocess the dynamic spectrum: scrunching, time/freq selection, upchannelization.

        Updates self.dyn_spec, self.freqs_mhz, self.time_res_s, self.freq_res_mhz
        and stores parameters in self.params['preprocess'].
        """
        self.params['preprocess'] = locals() # Store args
        del self.params['preprocess']['self'] # Don't store self

        current_dyn_spec = self.dyn_spec
        current_freqs = self.freqs_mhz
        current_time_res = self.time_res_s
        current_freq_res = self.freq_res_mhz

        # --- Apply Frequency Selection ---
        if freq_range_mhz is not None:
            f_min, f_max = freq_range_mhz
            freq_mask = (current_freqs >= f_min) & (current_freqs <= f_max)
            if np.sum(freq_mask) == 0:
                raise ValueError(f"No channels found in frequency range {freq_range_mhz} MHz.")
            current_dyn_spec = current_dyn_spec[:, freq_mask]
            current_freqs = current_freqs[freq_mask]
            print(f"Selected frequency range: {current_freqs.min():.2f} - {current_freqs.max():.2f} MHz")

        # --- Apply Time Selection ---
        if time_range is not None:
             t_start, t_end = time_range # Assuming indices
             if not (0 <= t_start < t_end <= current_dyn_spec.shape[0]):
                  raise ValueError(f"Invalid time range indices: {time_range} for data with {current_dyn_spec.shape[0]} time samples.")
             current_dyn_spec = current_dyn_spec[t_start:t_end, :]
             print(f"Selected time range (indices): {t_start} to {t_end}")


        # --- Apply Scrunching ---
        if t_scrunch > 1 or f_scrunch > 1:
            print(f"Scrunching by T={t_scrunch}, F={f_scrunch}")
            current_dyn_spec = scrunch(current_dyn_spec, t_scrunch, f_scrunch)
            # Update resolutions and frequency axis
            current_time_res *= t_scrunch
            current_freq_res *= f_scrunch
            # Need to recalculate freqs based on scrunching (simple averaging for centers)
            if f_scrunch > 1:
                current_freqs = current_freqs.reshape(-1, f_scrunch).mean(axis=1)
            print(f"New shape: {current_dyn_spec.shape}, New T_res: {current_time_res:.6f} s, New F_res: {current_freq_res:.6f} MHz")


        # --- Apply Upchannelization ---
        # NOTE: Upchannelization changes time/freq axes significantly.
        # It should typically be done *before* scrunching or selection,
        # or requires careful handling of axes after.
        # Current implementation assumes it runs on the *original* data (or selected subset).
        # Let's apply it here after selection but before scrunching for this example flow.
        # The output of upchannelize is (new_freq, new_time). We need to transpose.
        if apply_upchannel:
            print(f"Applying upchannelization with params: {upchannel_params}")
            # Need Intensity(freq, time) input for upchannelize function
            spec_freq_time = current_dyn_spec.T # Transpose to (freq, time)
            try:
                upchann_spec, upchan_factor = upchannelize(spec_freq_time, **upchannel_params)
                # Output is (new_freq, new_time). Transpose back to (time, freq)
                current_dyn_spec = upchann_spec.T
                # Update resolutions and axes
                # Time resolution increases: original_time_res * fftsize * downtime
                # Frequency resolution decreases: original_freq_res / upchan_factor
                # The exact mapping requires careful derivation based on upchannelize logic.
                # This needs refinement - providing approximate updates for now.
                fftsize = upchannel_params.get('fftsize', 32)
                downtime = upchannel_params.get('downtime', 1)
                # New time resolution: effective sampling interval of the blocks
                current_time_res = self.time_res_s * fftsize * downtime # Update original time_res
                # New frequency resolution: needs careful calculation based on FFT bins
                # If original channels are C, new are C * upchan_factor. Total BW approx conserved.
                original_bw = len(self.freqs_mhz) * self.freq_res_mhz
                # This assumes upchannel fills the *same* bandwidth. Needs verification.
                new_n_freq = current_dyn_spec.shape[1]
                current_freq_res = original_bw / new_n_freq if new_n_freq > 0 else 0
                # Need to recalculate the frequency centers - this is complex!
                # Placeholder: assume centered within original range for now
                current_freqs = np.linspace(self.freqs_mhz.min(), self.freqs_mhz.max(), new_n_freq)
                print(f"Upchannelized shape: {current_dyn_spec.shape}, New T_res: {current_time_res:.6f} s, New F_res: {current_freq_res:.6f} MHz")
            except ValueError as e:
                print(f"Upchannelization failed: {e}. Skipping.")


        # --- Update analyser state ---
        self.dyn_spec = current_dyn_spec
        self.freqs_mhz = current_freqs
        self.time_res_s = current_time_res
        self.freq_res_mhz = current_freq_res
        self.results['processed_dyn_spec'] = self.dyn_spec # Store processed spec

        print("Preprocessing complete.")
        print(f"Final shape (time, freq): {self.dyn_spec.shape}")
        print(f"Final Time resolution: {self.time_res_s:.6f} s")
        print(f"Final Frequency range: {self.freqs_mhz.min():.2f} - {self.freqs_mhz.max():.2f} MHz")
        print(f"Final Frequency resolution: {self.freq_res_mhz:.6f} MHz")


    def calculate_acf(self, axis: int = 1, norm: bool = True) -> None:
        """
        Calculates the 2D ACF along the specified axis (0=time, 1=frequency).

        Stores results in self.results['acf'].
        """
        if 'processed_dyn_spec' not in self.results:
            print("Run preprocess() first or use original data.")
            data_to_use = self.dyn_spec
        else:
            data_to_use = self.results['processed_dyn_spec']

        axis_name = "frequency" if axis == 1 else "time"
        print(f"Calculating ACF along axis {axis} ({axis_name})...")

        lags, avg_acf = calculate_acf_2d(data_to_use, axis=axis, norm=norm)

        self.results['acf'] = {'axis': axis, 'lags': lags, 'acf': avg_acf}
        print("ACF calculation complete.")

    def fit_acf_lorentzian(
        self,
        const_offset: bool = True
        ) -> None:
        """
        Fits a Lorentzian model to the calculated ACF (assumes ACF along freq axis).

        Stores fit results in self.results['acf_fit'].
        """
        if 'acf' not in self.results or self.results['acf']['axis'] != 1:
            raise RuntimeError("Calculate ACF along frequency axis (axis=1) first.")

        print("Fitting Lorentzian to ACF...")
        acf_data = self.results['acf']
        lags = acf_data['lags'] # Units are freq lags (MHz if freqs are MHz)
        acf = acf_data['acf']

        # Convert lags from freq units (MHz) to lag units (channels) for fitting?
        # Or fit directly in freq units. Let's use freq units.
        # Ensure lags are centered near zero if needed by fit function
        center_guess = lags[len(lags)//2] # Approx center

        params, model, fit_result = fit_lorentzian_acf(
            lags, acf, errs=None, # Add error estimation later if needed
            center_guess=center_guess,
            const_offset=const_offset
        )

        if params is not None:
            print("Lorentzian fit successful.")
            # Extract key parameters, e.g., HWHM
            # lmfit 'wid' parameter corresponds to HWHM for the standard form used.
            hwhm = params['wid'].value
            hwhm_err = params['wid'].stderr if params['wid'].stderr is not None else np.nan
            self.results['acf_fit'] = {
                'params': params,
                'model': model,
                'fit_result': fit_result,
                'scint_bandwidth_hwhm': hwhm, # Store HWHM
                'scint_bandwidth_hwhm_err': hwhm_err,
                'fit_report': fit_result.fit_report()
            }
            print(f"  Scintillation Bandwidth (HWHM): {hwhm:.4f} +/- {hwhm_err:.4f} {self.params.get('acf_freq_units', 'MHz')}") # Need units
        else:
            print("Lorentzian fit failed.")
            self.results['acf_fit'] = None

    def run_subband_analysis(
        self,
        n_subbands: int,
        fit_model: str = 'lorentzian' # or 'double_lorentzian' etc.
        ) -> None:
        """
        Divides data into subbands, calculates ACF per subband, fits scint bandwidth,
        and analyzes frequency dependence (Δν_d ∝ ν^α).

        Stores results in self.results['subband_analysis'].
        """
        if 'processed_dyn_spec' not in self.results:
            print("Run preprocess() first.")
            data_to_use = self.dyn_spec
            freqs_to_use = self.freqs_mhz
        else:
            data_to_use = self.results['processed_dyn_spec']
            freqs_to_use = self.freqs_mhz # Note: freqs might have changed in preprocess! Needs tracking.

        nt, nf = data_to_use.shape
        if nf < n_subbands:
            raise ValueError(f"Number of channels ({nf}) is less than requested subbands ({n_subbands}).")

        print(f"Performing analysis across {n_subbands} subbands...")
        sub_indices = np.array_split(np.arange(nf), n_subbands)

        subband_results = {
            'center_freq_mhz': [],
            'scint_bw_hwhm': [],
            'scint_bw_hwhm_err': [],
            'fit_params': [],
            'fit_success': []
        }

        for i, indices in enumerate(sub_indices):
            if len(indices) == 0: continue
            sub_spec = data_to_use[:, indices]
            sub_freqs = freqs_to_use[indices]
            center_freq = np.mean(sub_freqs)
            print(f"  Subband {i+1}/{n_subbands} (Freq ~ {center_freq:.2f} MHz, {len(indices)} chans)")

            # 1. Calculate ACF for the subband
            # Use freq resolution of the subband for lags
            sub_freq_res = np.abs(np.median(np.diff(sub_freqs))) if len(indices) > 1 else self.freq_res_mhz
            # Ensure calculate_acf_2d uses correct freq resolution if lags are scaled
            lags, acf = calculate_acf_2d(sub_spec, axis=1, norm=True)
            # Lags from calculate_acf_2d are currently indices/pixels. Convert to MHz.
            lags_mhz = lags * sub_freq_res

            # 2. Fit ACF (e.g., Lorentzian)
            if fit_model == 'lorentzian':
                params, _, fit_result = fit_lorentzian_acf(lags_mhz, acf, const_offset=True)
            # Add other models ('double_lorentzian', etc.) here if needed
            else:
                print(f"Warning: Unsupported fit model '{fit_model}'. Skipping fit.")
                params = None

            # 3. Store results
            subband_results['center_freq_mhz'].append(center_freq)
            if params is not None and fit_result is not None and fit_result.success:
                hwhm = params['wid'].value
                hwhm_err = params['wid'].stderr if params['wid'].stderr is not None else np.nan
                subband_results['scint_bw_hwhm'].append(hwhm)
                subband_results['scint_bw_hwhm_err'].append(hwhm_err)
                subband_results['fit_params'].append(params)
                subband_results['fit_success'].append(True)
                print(f"    Fit OK. HWHM: {hwhm:.4f} +/- {hwhm_err:.4f} MHz")
            else:
                subband_results['scint_bw_hwhm'].append(np.nan)
                subband_results['scint_bw_hwhm_err'].append(np.nan)
                subband_results['fit_params'].append(None)
                subband_results['fit_success'].append(False)
                print(f"    Fit Failed.")


        # Convert lists to arrays
        for key in subband_results:
            if key != 'fit_params': # Keep params as list of objects
                subband_results[key] = np.array(subband_results[key])

        self.results['subband_analysis'] = subband_results

        # 4. Fit frequency scaling relation Δν_d ∝ ν^α
        print("\nFitting scintillation bandwidth vs. frequency (Δν_d ∝ ν^α)...")
        valid_mask = subband_results['fit_success'] & np.isfinite(subband_results['scint_bw_hwhm'])
        if np.sum(valid_mask) >= 2:
            freqs_fit = subband_results['center_freq_mhz'][valid_mask]
            bw_fit = subband_results['scint_bw_hwhm'][valid_mask]
            bw_err_fit = subband_results['scint_bw_hwhm_err'][valid_mask]
            # Use errors if available and finite, otherwise basic fit
            use_errors = np.all(np.isfinite(bw_err_fit)) and np.all(bw_err_fit > 0)

            params_pl, model_pl, fit_result_pl = fit_scint_bandwidth_freq_relation(
                freqs_fit, bw_fit, errs=bw_err_fit if use_errors else None
            )

            if params_pl is not None:
                print("Power law fit successful.")
                alpha = params_pl['index'].value
                alpha_err = params_pl['index'].stderr if params_pl['index'].stderr is not None else np.nan
                amp = params_pl['amp'].value
                amp_err = params_pl['amp'].stderr if params_pl['amp'].stderr is not None else np.nan
                subband_results['power_law_fit'] = {
                    'params': params_pl,
                    'model': model_pl,
                    'fit_result': fit_result_pl,
                    'alpha': alpha,
                    'alpha_err': alpha_err,
                    'amplitude': amp,
                    'amplitude_err': amp_err,
                    'fit_report': fit_result_pl.fit_report()
                }
                print(f"  alpha = {alpha:.2f} +/- {alpha_err:.2f}")
                print(f"  Fit: Δν_d = ({amp:.2e}) * ν_MHz^({alpha:.2f})")
            else:
                print("Power law fit failed.")
                subband_results['power_law_fit'] = None
        else:
            print("Not enough valid points to fit power law.")
            subband_results['power_law_fit'] = None


    def calculate_secondary(self) -> None:
        """
        Calculates the secondary spectrum.

        Stores results in self.results['secondary_spectrum'].
        """
        if 'processed_dyn_spec' not in self.results:
            print("Run preprocess() first.")
            data_to_use = self.dyn_spec
        else:
            data_to_use = self.results['processed_dyn_spec']

        print("Calculating secondary spectrum...")
        freq_res_hz = self.freq_res_mhz * 1e6 # Convert MHz to Hz

        sec_spec, fd_axis, tau_axis = calculate_secondary_spectrum(
            data_to_use, self.time_res_s, freq_res_hz
        )

        self.results['secondary_spectrum'] = {
            'spec': sec_spec,
            'fd_hz': fd_axis,    # Doppler freq axis (Hz)
            'tau_us': tau_axis * 1e6 # Delay axis (microseconds)
        }
        print("Secondary spectrum calculation complete.")

    # --- Placeholder for Arc Fitting ---
    def fit_secondary_arc(self, *args, **kwargs) -> None:
        """ Fits parabolic arc(s) to the secondary spectrum."""
        if 'secondary_spectrum' not in self.results:
            raise RuntimeError("Calculate secondary spectrum first.")
        print("Placeholder: Secondary spectrum arc fitting not implemented yet.")
        # TODO: Implement arc detection and fitting (e.g., using Hough transform or direct fitting)
        # Store curvature results in self.results['secondary_fit'] = {'curvature': value, 'curvature_err': err}
        self.results['secondary_fit'] = None # Placeholder

    # --- Derivations ---
    def derive_timescale_from_bandwidth(self, freq_mhz: Optional[float] = None) -> None:
        """ Estimate timescale τ_d from ACF bandwidth Δν_d at reference frequency."""
        if 'acf_fit' not in self.results or self.results['acf_fit'] is None:
            print("Warning: Cannot estimate timescale. Run fit_acf_lorentzian first.")
            return
        if freq_mhz is None:
            # Use center frequency of the band used for the main ACF fit
            # This assumes the main ACF was calculated over the full (or preprocessed) band
            freq_mhz = np.mean(self.freqs_mhz) # Use current mean freq

        delta_nu_d_mhz = self.results['acf_fit']['scint_bandwidth_hwhm']
        delta_nu_d_hz = delta_nu_d_mhz * 1e6
        alpha = 4.0 # Assume Kolmogorov for now, or get from fit if available
        if 'subband_analysis' in self.results and self.results['subband_analysis']['power_law_fit']:
            alpha = self.results['subband_analysis']['power_law_fit']['alpha']

        tau_d_s = scintillation_bandwidth_to_timescale(delta_nu_d_hz, freq_mhz, alpha)
        tau_d_ms = tau_d_s * 1000.0

        self.results['derived_timescale_ms'] = tau_d_ms
        print(f"Derived scintillation timescale τ_d ≈ {tau_d_ms:.4f} ms (at {freq_mhz:.1f} MHz, assuming α={alpha:.2f})")


    def derive_screen_distance(self, source_dist_mpc: Optional[float] = None) -> None:
        """ Estimate screen distance D_L or D_eff from secondary arc curvature."""
        if 'secondary_fit' not in self.results or self.results['secondary_fit'] is None:
            print("Warning: Cannot estimate screen distance. Run fit_secondary_arc first.")
            return
        if 'curvature' not in self.results['secondary_fit']:
            print("Warning: Arc curvature measurement not found in secondary_fit results.")
            return

        curvature = self.results['secondary_fit']['curvature'] # Units: s^3 assumed
        # Use center frequency of the band for calculation
        freq_ghz = np.mean(self.freqs_mhz) / 1000.0

        dist_pc = screen_distance_from_curvature(curvature, freq_ghz, source_dist_mpc)

        if source_dist_mpc is not None:
            self.results['derived_lens_distance_pc'] = dist_pc
            print(f"Derived Lens distance D_L ≈ {dist_pc:.2f} pc (assuming D_S = {source_dist_mpc} Mpc)")
        else:
            self.results['derived_effective_distance_pc'] = dist_pc
            print(f"Derived Effective distance D_eff ≈ {dist_pc:.2f} pc")


    # --- Plotting Methods (Examples) ---
    def plot_dynamic_spectrum(self, processed: bool = True, **kwargs):
        """ Plots the dynamic spectrum. """
        if processed and 'processed_dyn_spec' in self.results:
            spec = self.results['processed_dyn_spec']
            title = f"{self.source_name} Processed Dynamic Spectrum"
            ylabel = f"Frequency ({self.params.get('acf_freq_units', 'MHz')})" # Need units
        else:
            spec = self.dyn_spec
            title = f"{self.source_name} Original Dynamic Spectrum"
            ylabel = "Frequency (MHz)" # Original freqs are MHz

        t_vec = np.arange(spec.shape[0]) * self.time_res_s
        f_vec = self.freqs_mhz

        plt.figure(figsize=kwargs.get('figsize', (10, 5)))
        plt.imshow(spec.T, aspect='auto', origin='lower',
                   extent=[t_vec[0], t_vec[-1], f_vec[0], f_vec[-1]],
                   **kwargs)
        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)
        plt.colorbar(label="Intensity (Arb. Units)")
        plt.title(title)
        plt.tight_layout()
        plt.show()


    def plot_acf(self, **kwargs):
        """ Plots the calculated ACF. """
        if 'acf' not in self.results:
            print("No ACF calculated yet.")
            return

        acf_data = self.results['acf']
        lags = acf_data['lags']
        acf = acf_data['acf']
        axis = acf_data['axis']
        axis_name = "Frequency Lag (MHz)" if axis == 1 else "Time Lag (samples?)" # TODO: Need units for time lag

        plt.figure(figsize=kwargs.get('figsize', (8, 5)))
        plt.plot(lags, acf, **kwargs)
        plt.xlabel(axis_name)
        plt.ylabel("Autocorrelation")
        plt.title(f"{self.source_name} Averaged ACF ({'Freq Axis' if axis==1 else 'Time Axis'})")
        plt.grid(True)

        # Overplot fit if available
        if 'acf_fit' in self.results and self.results['acf_fit'] is not None and axis == 1:
            fit_result = self.results['acf_fit']['fit_result']
            plt.plot(lags, fit_result.best_fit, 'r--', label=f'Lorentzian Fit (HWHM={fit_result.params["wid"].value:.3f} MHz)')
            plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_subband_analysis(self, **kwargs):
        """ Plots the scintillation bandwidth vs frequency and the power law fit."""
        if 'subband_analysis' not in self.results:
            print("No subband analysis performed yet.")
            return

        res = self.results['subband_analysis']
        freqs = res['center_freq_mhz']
        bw = res['scint_bw_hwhm']
        bw_err = res['scint_bw_hwhm_err']
        success = res['fit_success']

        plt.figure(figsize=kwargs.get('figsize', (8, 5)))
        plt.errorbar(freqs[success], bw[success], yerr=bw_err[success], fmt='o', label='Successful Fits', capsize=3)
        if np.any(~success):
            plt.plot(freqs[~success], bw[~success], 'x', color='red', label='Failed Fits')

        # Plot power law fit if available
        if res['power_law_fit'] is not None:
            pl_fit = res['power_law_fit']
            alpha = pl_fit['alpha']
            amp = pl_fit['amplitude']
            fit_freqs = np.linspace(freqs.min(), freqs.max(), 100)
            fit_bw = amp * (fit_freqs ** alpha)
            plt.plot(fit_freqs, fit_bw, 'r--', label=f'Fit: $α = {alpha:.2f} \pm {pl_fit["alpha_err"]:.2f}$')

        plt.xlabel("Center Frequency (MHz)")
        plt.ylabel("Scintillation Bandwidth HWHM (MHz)")
        plt.title(f"{self.source_name} Scintillation Bandwidth vs. Frequency")
        plt.yscale('log') # Often plotted log-log
        plt.xscale('log')
        plt.grid(True, which='both')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_secondary_spectrum(self, **kwargs):
        """ Plots the secondary spectrum."""
        if 'secondary_spectrum' not in self.results:
            print("No secondary spectrum calculated yet.")
            return

        sec = self.results['secondary_spectrum']
        spec = sec['spec']
        fd = sec['fd_hz']
        tau = sec['tau_us']

        # Determine intensity scaling (e.g., log scale, limits)
        vmin = kwargs.pop('vmin', np.percentile(spec, 5))
        vmax = kwargs.pop('vmax', np.percentile(spec, 99.5))
        norm = kwargs.pop('norm', matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))

        plt.figure(figsize=kwargs.get('figsize', (8, 6)))
        plt.imshow(spec, aspect='auto', origin='lower',
                   extent=[fd[0], fd[-1], tau[0], tau[-1]],
                   norm=norm, cmap=kwargs.get('cmap', 'viridis'), **kwargs)
        plt.xlabel("Doppler Frequency $f_D$ (Hz)")
        plt.ylabel("Delay $τ$ (µs)")
        plt.colorbar(label="Secondary Spectrum Power (Arb. Units)")
        plt.title(f"{self.source_name} Secondary Spectrum")
        # Optionally limit axes for better arc visibility
        # plt.xlim(-fd_lim, fd_lim)
        # plt.ylim(-tau_lim, tau_lim)
        plt.tight_layout()
        plt.show()