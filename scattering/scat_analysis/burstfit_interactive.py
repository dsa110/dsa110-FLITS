"""
burstfit_interactive.py
=======================

Interactive widgets for manual initial guess refinement before MCMC.
Allows real-time visualization of model vs data with adjustable parameters.

Overview
--------
This module provides an interactive Jupyter widget (`InitialGuessWidget`) that
enables users to manually refine FRB scattering model parameters before running
expensive MCMC fits. The widget displays:

- Real-time dynamic spectrum comparison (Data vs Model)
- Residual visualization (Data - Model)
- Time profile overlay
- Chi-squared goodness-of-fit metric

Key Features
------------
1. **Interactive Sliders**: Adjust all model parameters with immediate visual feedback
2. **Auto-Optimize**: Uses scipy's L-BFGS-B optimizer to refine from current values
3. **Accept & Continue**: Save parameters for use in downstream MCMC fitting

Auto-Optimize Algorithm
-----------------------
The "Auto-Optimize" button runs scipy.optimize.minimize with the L-BFGS-B method:

- **Objective**: Minimize negative log-likelihood (equivalent to chi-squared for Gaussian noise)
- **Method**: L-BFGS-B (Limited-memory BFGS with box constraints)
- **Starting Point**: Current slider values (user's manual adjustment)
- **Bounds**: Built from `build_priors()` with 1.5× scale around current values
- **Convergence**: Up to 500 iterations, ftol=1e-9

The log-likelihood assumes independent Gaussian noise per frequency channel:

    log L = -0.5 * sum_ij [(d_ij - m_ij) / sigma_i]^2

where d is data, m is model, and sigma is per-channel noise (estimated from off-pulse).

Why L-BFGS-B works well here:
- Quasi-Newton method that approximates the Hessian (curvature) efficiently
- Box constraints prevent unphysical parameter values (negative tau, etc.)
- Very fast for smooth, continuous likelihood surfaces like this one
- Starting from a reasonable manual guess ensures convergence to global minimum

Example Usage
-------------
>>> from scat_analysis.burstfit_interactive import InitialGuessWidget
>>> widget = InitialGuessWidget(dataset, model_key="M3")
>>> display(widget.create_widget())
>>> # ... user adjusts sliders, clicks Auto-Optimize, clicks Accept ...
>>> optimized_params = widget.get_params()
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from IPython.display import display
import ipywidgets as widgets
from .burstfit import FRBModel, FRBParams


class InitialGuessWidget:
    """
    Interactive Jupyter widget for manually refining FRB model initial guesses.
    
    This widget provides a visual interface for adjusting scattering model parameters
    before running computationally expensive MCMC fits. It displays the data and model
    side-by-side with real-time updates as sliders are adjusted.
    
    The widget is particularly useful when:
    - Automated initial guesses fail or converge to local minima
    - The burst has unusual structure (multiple components, weak scattering)
    - You want to develop intuition about parameter sensitivities
    - MCMC keeps failing or giving unreasonable results
    
    Attributes
    ----------
    dataset : object
        Dataset with .data, .time, .freq, .df_MHz attributes
    model_key : str
        Model variant to use: "M0" (Gaussian), "M1" (+intrinsic width),
        "M2" (+scattering), "M3" (full model with all parameters)
    model : FRBModel
        The forward model instance for generating dynamic spectra
    params : FRBParams
        Current parameter values (updated by sliders)
    optimized_params : FRBParams or None
        Final accepted parameters (set when user clicks "Accept & Continue")
    
    Methods
    -------
    create_widget()
        Build and return the interactive ipywidgets interface
    get_params()
        Return the optimized parameters (or current if not yet accepted)
    
    Examples
    --------
    Basic usage in a Jupyter notebook:
    
    >>> from scat_analysis.burstfit_interactive import InitialGuessWidget
    >>> from scat_analysis.burstfit_pipeline import BurstDataset
    >>> 
    >>> # Load your data
    >>> dataset = BurstDataset("burst.npy", "output/", telescope=telcfg, ...)
    >>> 
    >>> # Create widget
    >>> widget = InitialGuessWidget(dataset, model_key="M3")
    >>> display(widget.create_widget())
    >>> 
    >>> # After adjusting and accepting...
    >>> params = widget.get_params()
    >>> print(f"Optimized t0 = {params.t0:.3f} ms")
    """
    
    def __init__(self, dataset, model_key="M3", initial_params=None):
        """
        Initialize the interactive widget.
        
        Parameters
        ----------
        dataset : object
            Dataset object containing the burst data. Must have attributes:
            - data : ndarray, shape (n_freq, n_time) - dynamic spectrum
            - time : ndarray, shape (n_time,) - time axis in ms
            - freq : ndarray, shape (n_freq,) - frequency axis in GHz
            - df_MHz : float - channel bandwidth in MHz (optional)
            - dm_init : float - initial DM correction (optional, default 0)
            
        model_key : str, default "M3"
            Which model variant to use:
            - "M0": Simple Gaussian pulse (c0, t0, delta_dm)
            - "M1": Gaussian + intrinsic width (+ gamma, zeta)
            - "M2": Gaussian + scattering (+ tau_1ghz)
            - "M3": Full model with all parameters (+ alpha)
            
        initial_params : FRBParams, optional
            Starting parameter values. If None, uses data-driven heuristics
            to estimate reasonable starting values from the data itself.
            
        Notes
        -----
        The widget creates its own FRBModel instance from the dataset, so
        modifications to the model won't affect the original dataset.
        """
        self.dataset = dataset
        self.model_key = model_key
        
        # Create FRBModel instance
        self.model = FRBModel(
            data=dataset.data,
            time=dataset.time,
            freq=dataset.freq,
            dm_init=getattr(dataset, 'dm_init', 0.0),
            df_MHz=getattr(dataset, 'df_MHz', None)
        )
        
        # Initialize parameters
        if initial_params is None:
            self.params = self._get_data_driven_guess()
        else:
            self.params = initial_params
        
        # Storage for final optimized params
        self.optimized_params = None
        
    def _get_data_driven_guess(self):
        """
        Generate intelligent initial parameter guesses from data statistics.
        
        This method analyzes the observed dynamic spectrum to estimate reasonable
        starting values for each model parameter. While not as precise as a full
        fit, these heuristics typically get within the right order of magnitude.
        
        Heuristics Used
        ---------------
        **c0 (amplitude)**:
            1. Generate test model with c0=100
            2. Compare peak of data time-profile to model time-profile
            3. Scale c0 to match (with 80% factor to leave headroom)
        
        **t0 (arrival time)**:
            Peak position of frequency-summed time profile
        
        **gamma (log-width parameter)**:
            Estimated from FWHM of time profile: gamma ≈ log10(FWHM in ms)
            Clipped to range [-3, 2]
        
        **zeta (spectral width)**:
            Coefficient of variation (std/mean) of the frequency spectrum.
            Clipped to range [0.1, 2.0]
        
        **tau_1ghz (scattering timescale at 1 GHz)**:
            Estimated from asymmetry of the time profile:
            - If tail/rise ratio > 1.5: estimate from time to half-max decay
            - Otherwise: use 5% of total time window
            Always clipped to minimum 0.01 ms
        
        **alpha (scattering index)**:
            Fixed at 4.0 (Kolmogorov turbulence prediction)
        
        **delta_dm (DM correction)**:
            Fixed at 0.0 (assume pre-dedispersed)
        
        Returns
        -------
        FRBParams
            Parameter object with data-driven initial guesses
        
        Notes
        -----
        These estimates are intentionally approximate. The widget's sliders
        and Auto-Optimize feature allow refinement from these starting points.
        For weak or noisy bursts, manual adjustment is often necessary.
        """
        # Time profile
        prof = np.nansum(self.dataset.data, axis=0)
        if np.all(prof == 0):
            prof = np.ones_like(prof)
        
        # Peak position
        t0 = self.dataset.time[np.argmax(prof)]
        
        # Amplitude - calibrate by matching profile peaks
        # Generate a test model and scale c0 to match data
        data_prof_max = np.max(prof)
        
        # Start with a test c0 and measure the ratio
        test_c0 = 100.0
        test_params = FRBParams(
            c0=test_c0, t0=t0, gamma=-1.0, zeta=0.5,
            tau_1ghz=0.1, alpha=4.0, delta_dm=0.0
        )
        test_model = self.model(test_params, self.model_key)
        test_prof_max = np.max(np.sum(test_model, axis=0))
        
        # Scale c0 to match data profile
        if test_prof_max > 0:
            c0 = test_c0 * (data_prof_max / test_prof_max) * 0.8  # 80% to leave room
        else:
            c0 = 100.0
        
        # Width estimate from profile FWHM  
        peak_val = np.max(prof)
        half_max = peak_val / 2
        above_half = prof > half_max
        dt_ms = self.dataset.time[1] - self.dataset.time[0]
        if np.any(above_half):
            width_samples = np.sum(above_half)
            width_ms = width_samples * dt_ms
            # gamma relates to log(width), typical range -2 to 2
            gamma = np.clip(np.log10(max(width_ms, 1e-4)), -3, 2)
        else:
            gamma = -1.0
        
        # Spectral width - check if there's frequency structure
        spec = np.nansum(self.dataset.data, axis=1)
        spec_var = np.std(spec) / (np.mean(spec) + 1e-10)
        zeta = max(0.1, min(spec_var, 2.0))
        
        # Scattering - check for tail in time profile (asymmetry)
        peak_idx = np.argmax(prof)
        n_tail = min(20, len(prof) - peak_idx - 1)
        n_rise = min(20, peak_idx)
        if n_tail > 5 and n_rise > 5:
            tail_mean = np.mean(prof[peak_idx+1:peak_idx+n_tail+1])
            rise_mean = np.mean(prof[peak_idx-n_rise:peak_idx])
            # If tail is stronger than rise, there's scattering
            asymmetry = tail_mean / (rise_mean + 1e-10)
            if asymmetry > 1.5:
                # Estimate tau from time to half-max after peak
                decay = prof[peak_idx:]
                half_decay_idx = np.argmax(decay < peak_val * 0.5)
                if half_decay_idx > 0:
                    tau_1ghz = half_decay_idx * dt_ms * 0.7  # Rough conversion
                else:
                    tau_1ghz = 0.5 * dt_ms * n_tail
            else:
                tau_1ghz = 0.05 * (self.dataset.time[-1] - self.dataset.time[0])
        else:
            tau_1ghz = 0.05 * (self.dataset.time[-1] - self.dataset.time[0])
        
        # Ensure tau is positive and reasonable
        tau_1ghz = max(0.01, tau_1ghz)
        
        # Alpha (scattering index) - standard value
        alpha = 4.0
        
        # DM correction
        delta_dm = 0.0
        
        return FRBParams(
            c0=c0,
            t0=t0,
            gamma=gamma,
            zeta=zeta,
            tau_1ghz=tau_1ghz,
            alpha=alpha,
            delta_dm=delta_dm
        )
    
    def create_widget(self):
        """
        Create interactive ipywidgets interface for Jupyter notebook.
        
        Builds a complete interactive UI with sliders, buttons, and real-time
        visualization of the model fit quality.
        
        UI Components
        -------------
        **Sliders** (left panel):
            - c0: Amplitude (arbitrary units, scaled to data)
            - t0: Arrival time at reference frequency (ms)
            - gamma: Log-width parameter (dimensionless)
            - zeta: Spectral width parameter (ms)
            - tau_1ghz: Scattering timescale at 1 GHz (ms)
            - alpha: Scattering spectral index (dimensionless)
        
        **Buttons**:
            - "Auto-Optimize": Runs L-BFGS-B optimization from current values
            - "Accept & Continue": Saves current parameters for downstream use
        
        **Plots** (right panel, 2×2 grid):
            - Top-left: Data dynamic spectrum
            - Top-right: Model dynamic spectrum
            - Bottom-left: Residual (Data - Model) with chi-squared
            - Bottom-right: Overlaid time profiles
        
        Returns
        -------
        ipywidgets.HBox
            Widget container ready for display() in Jupyter
        
        Examples
        --------
        >>> widget = InitialGuessWidget(dataset, model_key="M3")
        >>> display(widget.create_widget())
        
        Interaction Workflow
        --------------------
        1. Adjust sliders to approximately match the data
        2. Click "Auto-Optimize" to refine (uses L-BFGS-B)
        3. Fine-tune with sliders if needed
        4. Click "Accept & Continue" to save parameters
        5. Call widget.get_params() to retrieve the result
        """
        # Get reasonable ranges for sliders based on data
        t_range = self.dataset.time[-1] - self.dataset.time[0]
        dt_ms = self.dataset.time[1] - self.dataset.time[0]
        c_max = np.max(self.dataset.data) * self.dataset.data.shape[0]  # Scale by n_freq
        
        # Create sliders with refined step sizes and bounds
        # Step sizes are chosen to be ~1/200 of the range for fine control
        style = {'description_width': '120px'}
        layout = widgets.Layout(width='500px')
        
        # Amplitude: max at 2× initial or 1.5× data max, fine steps
        c0_max = max(c_max * 1.5, self.params.c0 * 2)
        c0_step = max(c0_max / 2000, 0.0001)  # ~2000 steps across range
        
        # Scattering: max at 2× initial or 20% of time range, very fine steps
        tau_max = max(t_range * 0.2, self.params.tau_1ghz * 2, 0.5)
        tau_step = max(tau_max / 2000, dt_ms / 20, 0.0001)
        
        # Time bounds
        t_min, t_max = self.dataset.time[0], self.dataset.time[-1]
        t0_step = dt_ms / 20  # 20× finer than time resolution
        
        # Gamma and zeta bounds
        gamma_min, gamma_max = -3.0, 2.0
        zeta_min, zeta_max = 0.001, 2.0
        alpha_min, alpha_max = 2.0, 6.0
        
        # Style for range labels
        range_style = {'font-size': '10px', 'color': '#666', 'margin-left': '5px'}
        
        sliders = {
            'c0': widgets.FloatSlider(
                value=self.params.c0,
                min=0.0,
                max=c0_max,
                step=c0_step,
                description=f'c0 [0, {c0_max:.1f}]:',
                style=style,
                layout=layout,
                readout_format='.4f'
            ),
            't0': widgets.FloatSlider(
                value=self.params.t0,
                min=t_min,
                max=t_max,
                step=t0_step,
                description=f't0 [{t_min:.2f}, {t_max:.2f}]:',
                style=style,
                layout=layout,
                readout_format='.5f'
            ),
            'gamma': widgets.FloatSlider(
                value=self.params.gamma,
                min=gamma_min,
                max=gamma_max,
                step=0.001,  # Very fine step for spectral index
                description=f'γ [{gamma_min}, {gamma_max}]:',
                style=style,
                layout=layout,
                readout_format='.4f'
            ),
            'zeta': widgets.FloatSlider(
                value=self.params.zeta,
                min=zeta_min,
                max=zeta_max,
                step=0.001,  # Very fine step for pulse width
                description=f'ζ [{zeta_min}, {zeta_max}]:',
                style=style,
                layout=layout,
                readout_format='.4f'
            ),
            'tau_1ghz': widgets.FloatSlider(
                value=self.params.tau_1ghz,
                min=0.0,
                max=tau_max,
                step=tau_step,
                description=f'τ₁GHz [0, {tau_max:.2f}]:',
                style=style,
                layout=layout,
                readout_format='.5f'
            ),
            'alpha': widgets.FloatSlider(
                value=getattr(self.params, 'alpha', 4.0),
                min=alpha_min,
                max=alpha_max,
                step=0.01,  # Finer step for scattering index
                description=f'α [{alpha_min}, {alpha_max}]:',
                style=style,
                layout=layout,
                readout_format='.3f'
            ),
        }
        
        # Output widget for plots
        output = widgets.Output()
        
        # Buttons
        optimize_btn = widgets.Button(
            description='Auto-Optimize',
            button_style='success',
            tooltip='Run scipy optimization to refine current guess'
        )
        
        accept_btn = widgets.Button(
            description='Accept & Continue',
            button_style='primary',
            tooltip='Accept current parameters as initial guess for MCMC'
        )
        
        # Status text
        status = widgets.HTML(value='<b>Status:</b> Adjust sliders to match data')
        
        def update_plot(*args):
            """
            Update plot when any slider changes.
            
            Regenerates the model dynamic spectrum from current slider values
            and updates all four plot panels:
            - Data (unchanged, for reference)
            - Model (regenerated from current parameters)
            - Residual (Data - Model)
            - Time profiles (overlaid)
            
            Also computes and displays chi-squared metric:
                χ² = sum(residual²) / sum(data²)
            
            Lower χ² indicates better fit (0 = perfect match).
            """
            # Get current parameter values
            current_params = FRBParams(
                c0=sliders['c0'].value,
                t0=sliders['t0'].value,
                gamma=sliders['gamma'].value,
                zeta=sliders['zeta'].value,
                tau_1ghz=sliders['tau_1ghz'].value,
                alpha=sliders['alpha'].value,
                delta_dm=0.0
            )
            
            # Generate model
            model_dyn = self.model(current_params, self.model_key)
            residual = self.dataset.data - model_dyn
            
            # Calculate goodness metrics
            chi2 = np.sum(residual**2) / np.sum(self.dataset.data**2)
            
            with output:
                output.clear_output(wait=True)
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 9))
                
                # Data
                vmin, vmax = np.percentile(self.dataset.data, [1, 99])
                im0 = axes[0,0].imshow(self.dataset.data, aspect='auto', origin='lower',
                                       extent=[self.dataset.time[0], self.dataset.time[-1],
                                              self.dataset.freq[0], self.dataset.freq[-1]],
                                       vmin=vmin, vmax=vmax, cmap='plasma')
                axes[0,0].set_title('Data', fontweight='bold')
                axes[0,0].set_ylabel('Frequency [GHz]')
                plt.colorbar(im0, ax=axes[0,0])
                
                # Model
                im1 = axes[0,1].imshow(model_dyn, aspect='auto', origin='lower',
                                       extent=[self.dataset.time[0], self.dataset.time[-1],
                                              self.dataset.freq[0], self.dataset.freq[-1]],
                                       vmin=vmin, vmax=vmax, cmap='plasma')
                axes[0,1].set_title(f'Model ({self.model_key})', fontweight='bold')
                plt.colorbar(im1, ax=axes[0,1])
                
                # Residual
                res_std = np.std(residual)
                im2 = axes[1,0].imshow(residual, aspect='auto', origin='lower',
                                       extent=[self.dataset.time[0], self.dataset.time[-1],
                                              self.dataset.freq[0], self.dataset.freq[-1]],
                                       vmin=-3*res_std, vmax=3*res_std, cmap='PuOr')
                axes[1,0].set_title(f'Residual (χ² = {chi2:.3f})', fontweight='bold')
                axes[1,0].set_xlabel('Time [ms]')
                axes[1,0].set_ylabel('Frequency [GHz]')
                plt.colorbar(im2, ax=axes[1,0])
                
                # Profiles
                time_data = np.sum(self.dataset.data, axis=0)
                time_model = np.sum(model_dyn, axis=0)
                axes[1,1].plot(self.dataset.time, time_data, 'k-', lw=1.5, alpha=0.7, label='Data')
                axes[1,1].plot(self.dataset.time, time_model, 'm-', lw=2, label='Model')
                axes[1,1].set_xlabel('Time [ms]')
                axes[1,1].set_ylabel('Intensity')
                axes[1,1].set_title('Time Profile', fontweight='bold')
                axes[1,1].legend()
                axes[1,1].grid(alpha=0.3)
                
                plt.tight_layout()
                plt.show()
        
        def on_optimize_click(b):
            """
            Run scipy optimization from current slider values.
            
            This callback implements the "Auto-Optimize" functionality, which
            uses scipy's L-BFGS-B optimizer to refine the model parameters.
            
            Algorithm Details
            -----------------
            **Starting Point**: Current slider values (user's manual adjustment)
            
            **Optimizer**: L-BFGS-B (Limited-memory BFGS with box constraints)
                - Quasi-Newton method that approximates the Hessian
                - Memory-efficient: only stores ~10 gradient history vectors
                - Handles box constraints natively (no transformations needed)
            
            **Objective Function**: Negative log-likelihood
                
                nll(θ) = -log L(θ|d) = 0.5 * Σᵢⱼ [(dᵢⱼ - mᵢⱼ(θ)) / σᵢ]²
                
                where d is data, m is model, σ is per-channel noise.
            
            **Bounds Construction**:
                - Calls build_priors(current_params, scale=1.5)
                - Creates ±1.5× range around current values for each parameter
                - Prevents unphysical values (negative tau, etc.)
            
            **Convergence Criteria**:
                - maxiter=500 (maximum function evaluations)
                - ftol=1e-9 (relative tolerance on function value)
            
            **On Success**: Updates all sliders to optimized values
            
            **On Failure**: Displays error message; user should try different
                starting point or adjust manually
            
            Parameters
            ----------
            b : ipywidgets.Button
                The button that was clicked (unused, required by callback signature)
            
            Notes
            -----
            L-BFGS-B works well here because:
            1. The likelihood surface is smooth and continuous
            2. Starting from manual guess avoids local minima
            3. Box constraints prevent unphysical parameter regions
            4. The problem is low-dimensional (6 parameters)
            """
            from scipy.optimize import minimize
            from .burstfit import build_priors
            
            # Get current values as starting point
            current_params = FRBParams(
                c0=sliders['c0'].value,
                t0=sliders['t0'].value,
                gamma=sliders['gamma'].value,
                zeta=sliders['zeta'].value,
                tau_1ghz=sliders['tau_1ghz'].value,
                alpha=sliders['alpha'].value,
                delta_dm=0.0
            )
            
            status.value = '<b>Status:</b> <span style="color:orange">Running optimization...</span>'
            
            # Build priors
            priors, _ = build_priors(current_params, scale=1.5, 
                                    abs_max={"tau_1ghz": 5e4, "zeta": 5e4},
                                    log_weight_pos=True)
            
            x0 = current_params.to_sequence(self.model_key)
            from .burstfit import FRBFitter
            bounds = [priors[n] for n in FRBFitter._ORDER[self.model_key]]
            
            def nll(theta):
                p = FRBParams.from_sequence(theta, self.model_key)
                ll = self.model.log_likelihood(p, self.model_key)
                return -ll if np.isfinite(ll) else np.inf
            
            res = minimize(nll, x0, method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 500, 'ftol': 1e-9})
            
            if res.success:
                opt_params = FRBParams.from_sequence(res.x, self.model_key)
                # Update sliders
                sliders['c0'].value = opt_params.c0
                sliders['t0'].value = opt_params.t0
                sliders['gamma'].value = opt_params.gamma
                sliders['zeta'].value = opt_params.zeta
                sliders['tau_1ghz'].value = opt_params.tau_1ghz
                sliders['alpha'].value = getattr(opt_params, 'alpha', 4.0)
                status.value = '<b>Status:</b> <span style="color:green">Optimization successful!</span>'
            else:
                status.value = '<b>Status:</b> <span style="color:red">Optimization failed. Try different starting values.</span>'
        
        def on_accept_click(b):
            """
            Accept current parameters and save for downstream use.
            
            Creates an FRBParams object from current slider values and stores
            it in self.optimized_params. Also prints a confirmation with the
            saved parameter values.
            
            Parameters
            ----------
            b : ipywidgets.Button
                The button that was clicked (unused, required by callback signature)
            """
            self.optimized_params = FRBParams(
                c0=sliders['c0'].value,
                t0=sliders['t0'].value,
                gamma=sliders['gamma'].value,
                zeta=sliders['zeta'].value,
                tau_1ghz=sliders['tau_1ghz'].value,
                alpha=sliders['alpha'].value,
                delta_dm=0.0
            )
            status.value = '<b>Status:</b> <span style="color:blue">Parameters accepted! Ready for MCMC.</span>'
            print("\nInitial guess parameters saved.")
            print(f"  c0: {self.optimized_params.c0:.4f}")
            print(f"  t0: {self.optimized_params.t0:.4f} ms")
            print(f"  gamma: {self.optimized_params.gamma:.4f}")
            print(f"  zeta: {self.optimized_params.zeta:.4f}")
            print(f"  tau_1ghz: {self.optimized_params.tau_1ghz:.4f} ms")
            print(f"  alpha: {self.optimized_params.alpha:.4f}")
        
        # Connect callbacks
        for slider in sliders.values():
            slider.observe(update_plot, names='value')
        
        optimize_btn.on_click(on_optimize_click)
        accept_btn.on_click(on_accept_click)
        
        # Initial plot
        update_plot()
        
        # Layout
        slider_box = widgets.VBox(list(sliders.values()))
        button_box = widgets.HBox([optimize_btn, accept_btn])
        controls = widgets.VBox([
            widgets.HTML('<h3>Initial Guess Parameter Adjustment</h3>'),
            slider_box,
            button_box,
            status
        ])
        
        return widgets.HBox([controls, output])
    
    def get_params(self):
        """
        Return the current or accepted parameters.
        
        Returns the parameters saved by "Accept & Continue" if available,
        otherwise returns the initial data-driven guess.
        
        Returns
        -------
        FRBParams
            Parameter object with current values
        
        Examples
        --------
        >>> widget = InitialGuessWidget(dataset)
        >>> display(widget.create_widget())
        >>> # ... user interacts with widget and clicks Accept ...
        >>> params = widget.get_params()
        >>> print(f"Scattering timescale: {params.tau_1ghz:.3f} ms")
        
        Notes
        -----
        To use these parameters in MCMC fitting:
        
        >>> from scat_analysis.burstfit import FRBFitter
        >>> fitter = FRBFitter(model, model_key="M3", initial_guess=widget.get_params())
        >>> fitter.run_mcmc(...)
        """
        if self.optimized_params is not None:
            return self.optimized_params
        else:
            return self.params
