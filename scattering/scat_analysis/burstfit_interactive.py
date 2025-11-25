"""
burstfit_interactive.py
=======================

Interactive widgets for manual initial guess refinement before MCMC.
Allows real-time visualization of model vs data with adjustable parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from IPython.display import display
import ipywidgets as widgets
from .burstfit import FRBModel, FRBParams


class InitialGuessWidget:
    """
    Interactive widget for manually refining FRB model initial guess.
    
    Displays data and model side-by-side with sliders for all parameters.
    Updates model in real-time as parameters change.
    """
    
    def __init__(self, dataset, model_key="M3", initial_params=None):
        """
        Parameters
        ----------
        dataset : Dataset object with .data, .time, .freq attributes
        model_key : str, default "M3"
            Which model to use (M0, M1, M2, M3)
        initial_params : FRBParams, optional
            Starting parameter values. If None, uses data-driven defaults.
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
        """Generate intelligent initial guess from data statistics."""
        # Time profile
        prof = np.nansum(self.dataset.data, axis=0)
        if np.all(prof == 0):
            prof = np.ones_like(prof)
        
        # Peak position
        t0 = self.dataset.time[np.argmax(prof)]
        
        # Amplitude (use data peak, not profile sum)
        c0 = np.percentile(self.dataset.data, 99)
        
        # Width estimate from profile FWHM
        peak_val = np.max(prof)
        half_max = peak_val / 2
        above_half = prof > half_max
        if np.any(above_half):
            width_samples = np.sum(above_half)
            width_ms = width_samples * (self.dataset.time[1] - self.dataset.time[0])
            # gamma relates to log(width), typical range -2 to 2
            gamma = np.log10(max(width_ms, 0.01))
        else:
            gamma = -1.0
        
        # Spectral width - check if there's frequency structure
        spec = np.nansum(self.dataset.data, axis=1)
        spec_var = np.std(spec) / (np.mean(spec) + 1e-10)
        zeta = max(0.1, min(spec_var, 2.0))
        
        # Scattering - check for tail in time profile
        peak_idx = np.argmax(prof)
        if peak_idx < len(prof) - 10:
            tail = prof[peak_idx+5:peak_idx+20]
            if len(tail) > 0 and np.mean(tail) > 0.1 * peak_val:
                # Significant tail detected
                tau_1ghz = 1.0
            else:
                tau_1ghz = 0.1
        else:
            tau_1ghz = 0.1
        
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
        """Create interactive ipywidgets interface for Jupyter."""
        # Get reasonable ranges for sliders based on data
        t_range = self.dataset.time[-1] - self.dataset.time[0]
        c_max = np.percentile(self.dataset.data, 99.9)
        
        # Create sliders
        style = {'description_width': '120px'}
        layout = widgets.Layout(width='500px')
        
        sliders = {
            'c0': widgets.FloatSlider(
                value=self.params.c0,
                min=0.0,
                max=c_max * 2,
                step=c_max / 100,
                description='c0 (amplitude):',
                style=style,
                layout=layout
            ),
            't0': widgets.FloatSlider(
                value=self.params.t0,
                min=self.dataset.time[0],
                max=self.dataset.time[-1],
                step=(self.dataset.time[1] - self.dataset.time[0]),
                description='t0 (arrival):',
                style=style,
                layout=layout
            ),
            'gamma': widgets.FloatSlider(
                value=self.params.gamma,
                min=-3.0,
                max=3.0,
                step=0.1,
                description='gamma (width):',
                style=style,
                layout=layout
            ),
            'zeta': widgets.FloatSlider(
                value=self.params.zeta,
                min=0.01,
                max=5.0,
                step=0.05,
                description='zeta (spec width):',
                style=style,
                layout=layout
            ),
            'tau_1ghz': widgets.FloatSlider(
                value=self.params.tau_1ghz,
                min=0.0,
                max=10.0,
                step=0.1,
                description='tau_1ghz (scatter):',
                style=style,
                layout=layout
            ),
            'alpha': widgets.FloatSlider(
                value=getattr(self.params, 'alpha', 4.0),
                min=2.0,
                max=6.0,
                step=0.1,
                description='alpha (scat index):',
                style=style,
                layout=layout
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
            """Update plot when any slider changes."""
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
                                       vmin=-3*res_std, vmax=3*res_std, cmap='coolwarm')
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
            """Run scipy optimization from current slider values."""
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
            """Accept current parameters."""
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
            print("\n✓ Initial guess parameters saved.")
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
        """Return the optimized parameters (or current if not accepted yet)."""
        if self.optimized_params is not None:
            return self.optimized_params
        else:
            return self.params
