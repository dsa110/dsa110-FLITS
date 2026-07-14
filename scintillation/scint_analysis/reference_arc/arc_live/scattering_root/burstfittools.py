import numpy as np
import emcee
from scipy.signal import fftconvolve

# --------------------------------------------------------------------------
# --- Classes and Functions for 2D Dynamic Spectrum Fitting (Original) ---
# --------------------------------------------------------------------------

class FRBModel:
    """
    A class to model a 2D Fast Radio Burst dynamic spectrum.
    """
    def __init__(self, data, time, frequencies, dm_init):
        """
        Initializes the FRBModel for 2D data.
        """
        self.data = data
        self.time = time
        self.frequencies = frequencies
        self.dm = dm_init
        self.n_channels = data.shape[0]
        self.n_time_samples = data.shape[1]
        self.noise_std = self.estimate_noise()
        self.dt = time[1] - time[0]
        self.alpha = 4.0 # Scattering index for Kolmogorov turbulence.

    def estimate_noise(self):
        """Estimates noise from off-pulse regions."""
        off_pulse_indices = np.concatenate([
            np.arange(0, self.n_time_samples // 4),
            np.arange(3 * self.n_time_samples // 4, self.n_time_samples)
        ])
        off_pulse_data = self.data[:, off_pulse_indices]
        noise_std = np.std(off_pulse_data, axis=1)
        noise_std = np.maximum(noise_std, 1e-6)
        print("Estimated 2D Noise Std Dev per channel:", noise_std)
        return noise_std

    def model(self, params, model_type='model3'):
        """Generates the 2D model dynamic spectrum."""
        c0, t0, spectral_index = params[0], params[1], params[2]
        zeta = 0.0
        tau_1GHz = 0.0

        if model_type == 'model1':
            zeta = params[3]
        elif model_type == 'model2':
            tau_1GHz = params[3]
        elif model_type == 'model3':
            zeta = params[3]
            tau_1GHz = params[4]
        
        ref_freq = self.frequencies[self.n_channels // 2]
        c_i = c0 * (self.frequencies / ref_freq)**spectral_index
        sigma_smear = (8.3e-6) * self.dm * (self.frequencies)**(-3) * (self.frequencies[0]-self.frequencies[1])
        sigma_i = np.sqrt(sigma_smear**2 + zeta**2)

        model_spec = np.zeros_like(self.data)
        for i in range(self.n_channels):
            pulse = c_i[i] * np.exp(-(self.time - t0)**2 / (2 * sigma_i[i]**2))
            if tau_1GHz > 0:
                tau_i = tau_1GHz * (self.frequencies[i] / 1.0)**(-self.alpha)
                pbf = np.zeros_like(self.time)
                pbf[self.time >= 0] = np.exp(-self.time[self.time >= 0] / tau_i)
                pbf /= np.sum(pbf) * self.dt
                S_i = fftconvolve(pulse, pbf, mode='same')
            else:
                S_i = pulse
            model_spec[i, :] = S_i
        return model_spec

    def log_likelihood(self, params, model_type):
        """Computes the log-likelihood for the 2D model."""
        model_spectrum = self.model(params, model_type)
        sigma2 = self.noise_std[:, np.newaxis]**2
        residuals = self.data - model_spectrum
        return -0.5 * np.sum(residuals**2 / sigma2 + np.log(2 * np.pi * sigma2))

    def log_prior(self, params, prior_bounds):
        """Defines priors for the 2D model parameters."""
        param_names = list(prior_bounds.keys())
        for i, name in enumerate(param_names):
            if not (prior_bounds[name][0] <= params[i] <= prior_bounds[name][1]):
                return -np.inf
        return 0.0

    def log_posterior(self, params, prior_bounds, model_type):
        """Computes the log-posterior for the 2D model."""
        lp = self.log_prior(params, prior_bounds)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(params, model_type)

# (All original helper functions for 2D fitting like run_mcmc, fit_models, etc. remain the same)
def run_mcmc(model, initial_params, prior_bounds, model_type, nsteps=500):
    """Initializes and runs the MCMC sampler for the 2D model."""
    ndim = len(initial_params)
    nwalkers = max(10 * ndim, 50)
    
    p0 = np.zeros((nwalkers, ndim))
    param_names = list(prior_bounds.keys())
    for i in range(ndim):
        p0[:, i] = np.random.uniform(prior_bounds[param_names[i]][0], prior_bounds[param_names[i]][1], nwalkers)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, model.log_posterior, args=(prior_bounds, model_type))
    sampler.run_mcmc(p0, nsteps, progress=True)
    return sampler

def compute_bic(lnL_max, k, n):
    """Computes the Bayesian Information Criterion (BIC)."""
    return k * np.log(n) - 2 * lnL_max

def fit_models(model, init_params_dict, prior_bounds, numsteps=500, models_to_fit=['model3']):
    """Fits specified 2D models and selects the best one based on BIC."""
    results = {}
    n = model.data.size
    
    all_models = ['model0', 'model1', 'model2', 'model3']
    for m in all_models:
        results[m] = {'sampler': None, 'BIC': np.inf, 'lnL_max': -np.inf, 'k': 0}

    for model_type in models_to_fit:
        print(f"\n----- Fitting {model_type} -----")
        if model_type == 'model0':
            params_list = ['c0', 't0', 'spectral_index']
        elif model_type == 'model1':
            params_list = ['c0', 't0', 'spectral_index', 'zeta']
        elif model_type == 'model2':
            params_list = ['c0', 't0', 'spectral_index', 'tau_1GHz']
        elif model_type == 'model3':
            params_list = ['c0', 't0', 'spectral_index', 'zeta', 'tau_1GHz']
        
        initial_params = [init_params_dict[p] for p in params_list]
        current_priors = {p: prior_bounds[p] for p in params_list}

        sampler = run_mcmc(model, initial_params, current_priors, model_type, nsteps=numsteps)
        
        lnL_max = np.max(sampler.get_log_prob())
        k = len(initial_params)
        bic = compute_bic(lnL_max, k, n)
        
        results[model_type] = {'sampler': sampler, 'BIC': bic, 'lnL_max': lnL_max, 'k': k}
        print(f"Finished {model_type}: BIC = {bic:.2f}")

    BICs = {m: results[m]['BIC'] for m in models_to_fit}
    best_model = min(BICs, key=BICs.get)
    print(f"\nBest model based on BIC is: {best_model}")
    
    return results, best_model

def downsample_data(data, f_factor=1, t_factor=1):   
    """Averages data array over frequency and time factors."""
    print(f'Original Data Shape: {data.shape}')
    nf_orig, nt_orig = data.shape
    nf_new = nf_orig // f_factor
    data_ds_f = np.mean(data[:nf_new * f_factor, :].reshape(nf_new, f_factor, nt_orig), axis=1)
    nt_new = nt_orig // t_factor
    data_ds_t = np.mean(data_ds_f[:, :nt_new * t_factor].reshape(nf_new, nt_new, t_factor), axis=2)
    print(f'Downsampled Data Shape: {data_ds_t.shape}')
    return data_ds_t

# ---------------------------------------------------------------
# --- NEW Classes and Functions for 1D Time Series Fitting ---
# ---------------------------------------------------------------

class FRBModel1D:
    """
    A class to model a 1D FRB time series at a single frequency.
    """
    def __init__(self, data_1d, time, frequency):
        """
        Initializes the FRBModel for 1D data.
        Args:
            data_1d (np.ndarray): 1D time series.
            time (np.ndarray): 1D array of time samples in ms.
            frequency (float): The frequency of this channel in GHz.
        """
        self.data = data_1d
        self.time = time
        self.frequency = frequency
        self.n_time_samples = data_1d.shape[0]
        self.noise_std = self.estimate_noise_1d()
        self.dt = time[1] - time[0]

    def estimate_noise_1d(self):
        """Estimates noise from off-pulse regions for a 1D time series."""
        off_pulse_indices = np.concatenate([
            np.arange(0, self.n_time_samples // 4),
            np.arange(3 * self.n_time_samples // 4, self.n_time_samples)
        ])
        noise_std = np.std(self.data[off_pulse_indices])
        return np.maximum(noise_std, 1e-6)

    def model(self, params):
        """
        Generates the 1D model time series.
        Args:
            params (list): [c0, t0, zeta, tau], where tau is the
                           scattering time at this channel's frequency.
        """
        c0, t0, zeta, tau = params
        
        # Intrinsic pulse (Gaussian)
        # Note: Intra-channel smearing is ignored as it's sub-dominant to zeta.
        pulse = c0 * np.exp(-(self.time - t0)**2 / (2 * zeta**2))
        
        # Convolve with scattering profile
        if tau > 0:
            pbf = np.zeros_like(self.time)
            pbf[self.time >= 0] = np.exp(-self.time[self.time >= 0] / tau)
            pbf /= np.sum(pbf) * self.dt # Normalize
            S_i = fftconvolve(pulse, pbf, mode='same')
        else:
            S_i = pulse
            
        return S_i

    def log_likelihood_1d(self, params):
        """Computes the log-likelihood for the 1D model."""
        model_ts = self.model(params)
        sigma2 = self.noise_std**2
        residuals = self.data - model_ts
        return -0.5 * np.sum(residuals**2 / sigma2 + np.log(2 * np.pi * sigma2))

    def log_prior_1d(self, params, prior_bounds):
        """Defines priors for the 1D model parameters."""
        param_names = ['c0', 't0', 'zeta', 'tau']
        for i, name in enumerate(param_names):
            if not (prior_bounds[name][0] <= params[i] <= prior_bounds[name][1]):
                return -np.inf
        return 0.0

    def log_posterior_1d(self, params, prior_bounds):
        """Computes the log-posterior for the 1D model."""
        lp = self.log_prior_1d(params, prior_bounds)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood_1d(params)

def fit_channel_1d(model_1d, initial_params, prior_bounds, nsteps=1000):
    """
    Initializes and runs the MCMC sampler for a single 1D time series.
    """
    ndim = len(initial_params)
    nwalkers = max(10 * ndim, 50)
    
    # Initialize walkers within the prior bounds
    p0 = np.zeros((nwalkers, ndim))
    param_names = list(prior_bounds.keys())
    for i in range(ndim):
        p0[:, i] = np.random.uniform(prior_bounds[param_names[i]][0], prior_bounds[param_names[i]][1], nwalkers)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, model_1d.log_posterior_1d, args=(prior_bounds,))
    sampler.run_mcmc(p0, nsteps, progress=True)
    return sampler
