import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import emcee
import corner

# FRBModel class
class FRBModel:
    def __init__(self, data, time, frequencies, DM_init, beta=2):
        """
        Initialize the FRBModel with data and parameters.

        Parameters:
        - data: 2D numpy array of shape (N_channels, N_time_samples)
        - time: 1D numpy array of time samples in ms
        - frequencies: 1D numpy array of frequencies in GHz
        - DM_init: Initial dispersion measure used in dedispersion
        - beta: Dispersion index (default is 2 for cold plasma)
        """
        self.data = data
        self.time = time
        self.frequencies = frequencies
        self.DM_init = DM_init
        self.beta = beta
        self.N_channels = data.shape[0]
        self.N_time_samples = data.shape[1]
        self.noise_std = self.estimate_noise()

    #def estimate_noise(self):
    #    """
    #    Estimate the noise level from the off-pulse regions of the data.
    #    """
    #    # Assuming the pulse occupies a small fraction of the data
    #    # Use median absolute deviation as a robust estimator
    #    noise_std = np.median(np.abs(self.data - np.median(self.data, axis=1, keepdims=True)), axis=1) / 0.6745
    #    # Set a minimum noise level to prevent zero or negative estimates
    #    noise_std = np.maximum(noise_std, 1e-3)
    #    return noise_std
    
    def estimate_noise(self):
        # Identify off-pulse regions (e.g., first and last quarters of the time axis)
        off_pulse_indices = np.concatenate([
            np.arange(0, self.N_time_samples // 4),
            np.arange(3 * self.N_time_samples // 4, self.N_time_samples)
        ])
        off_pulse_data = self.data[:, off_pulse_indices]
        # Estimate noise using the off-pulse data
        noise_std = np.std(off_pulse_data, axis=1)
        noise_std = np.maximum(noise_std, 1e-3)
        return noise_std


    def t_i_DM(self, DM_err):
        """
        Compute dispersion delay for each frequency channel.

        Parameters:
        - DM_err: Deviation of the burst DM from the initial DM
        """
        DM = self.DM_init + DM_err
        nu_ref = self.frequencies.max()  # Use the highest frequency as the reference
        t_i_DM = (4.15) * DM_err * ((self.frequencies) ** (-self.beta) - nu_ref ** (-self.beta))
        return t_i_DM

    def sigma_i_DM(self, DM, zeta=0):
        """
        Compute intra-channel dispersion smearing and intrinsic width.

        Parameters:
        - DM: Total dispersion measure
        - zeta: Intrinsic pulse width (only for Model 1 and Model 3)
        """
        sigma_i_DM = (1.622e-3) * DM * (self.frequencies) ** (-self.beta - 1)
        sigma_i_DM = np.sqrt(sigma_i_DM ** 2 + zeta ** 2)
        return sigma_i_DM

    def scattering_timescale(self, tau_1GHz, alpha):
        """
        Compute scattering timescale for each frequency channel.

        Parameters:
        - tau_1GHz: Scattering timescale at 1 GHz
        - alpha: Frequency-dependency index of scattering
        """
        tau_i = tau_1GHz * (self.frequencies / 1.0) ** (-alpha)
        return tau_i

    def model(self, params, model_type='model0'):
        # Extract parameters
        c0 = params[0]
        t0 = params[1]
        spectral_index = params[2]
        DM_err = params[3]
        DM = self.DM_init + DM_err

        # Calculate c_i using the spectral index model
        reference_frequency = self.frequencies[self.N_channels // 2]
        c_i = c0 * (self.frequencies / reference_frequency) ** spectral_index

        # Determine the index for additional parameters
        idx = 4
        if model_type == 'model0':
            zeta = 0
            tau_1GHz = 0
            alpha = 0
        elif model_type == 'model1':
            zeta = params[idx]
            tau_1GHz = 0
            alpha = 0
            idx += 1
        elif model_type == 'model2':
            zeta = 0
            tau_1GHz = params[idx]
            alpha = params[idx + 1]
            idx += 2
        elif model_type == 'model3':
            zeta = params[idx]
            tau_1GHz = params[idx + 1]
            alpha = params[idx + 2]
            idx += 3
        else:
            raise ValueError("Invalid model type")

        t_i_DM = self.t_i_DM(DM_err)
        sigma_i_DM = self.sigma_i_DM(DM, zeta)

        model = np.zeros((self.N_channels, self.N_time_samples))
        for i in range(self.N_channels):
            c = c_i[i]
            mu = t0 + t_i_DM[i]
            sigma = sigma_i_DM[i]
            S_i = c / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(- (self.time - mu) ** 2 / (2 * sigma ** 2))

            if tau_1GHz > 0:
                # Include scattering by convolving with exponential function
                tau_i = tau_1GHz * (self.frequencies[i] / 1.0) ** (-alpha)
                dt = self.time[1] - self.time[0]
                t_pbf = self.time
                pbf = np.zeros_like(self.time)
                pbf[t_pbf >= 0] = np.exp(- t_pbf[t_pbf >= 0] / tau_i)
                pbf /= np.sum(pbf) * dt  # Normalize PBF

                # Perform convolution using 'full' mode
                conv_result = fftconvolve(S_i, pbf, mode='full')
                conv_time = np.arange(0, len(S_i) + len(pbf) - 1) * dt + 2 * self.time[0]
                start_idx = np.searchsorted(conv_time, self.time[0])
                end_idx = start_idx + len(self.time)
                S_i = conv_result[start_idx:end_idx]

            model[i, :] = S_i
        model = np.flip(model, axis=0)
        return model

    def log_likelihood(self, params, model_type='model0'):
        """
        Compute the log-likelihood of the model given the data.

        Parameters:
        - params: Model parameters
        - model_type: One of 'model0', 'model1', 'model2', 'model3'
        """
        model_spectrum = self.model(params, model_type)
        sigma2 = self.noise_std[:, np.newaxis] ** 2
        sigma2 = np.maximum(sigma2, 1e-6)  # Prevent division by zero
        residuals = self.data - model_spectrum
        lnL = -0.5 * np.sum(residuals ** 2 / sigma2 + np.log(2 * np.pi * sigma2))
        return lnL

    def log_prior(self, params, prior_bounds, model_type='model0'):
        """
        Define priors for the model parameters.

        Parameters:
        - params: Model parameters
        - model_type: One of 'model0', 'model1', 'model2', 'model3'
        """
        c0 = params[0]
        t0 = params[1]
        spectral_index = params[2]
        DM_err = params[3]

        # Priors for c0 and spectral_index
        if c0 <= prior_bounds['c0'][0] or c0 > prior_bounds['c0'][1]:
            return -np.inf
        if not (prior_bounds['t0'][0] <= t0 <= prior_bounds['t0'][1]):
            return -np.inf
        if not (prior_bounds['spectral_index'][0] <= spectral_index <= prior_bounds['spectral_index'][1]):
            return -np.inf
        if not (prior_bounds['DM_err'][0] <= DM_err <= prior_bounds['DM_err'][1]):
            return -np.inf

        # Priors for c0 and spectral_index
        #if c0 <= 0 or c0 > 100:
        #    return -np.inf
        #if not (self.time[0] <= t0 <= self.time[-1]):
        #    return -np.inf
        #if not (-3 <= spectral_index <= 1):
        #    return -np.inf
        #if not (-0.001 <= DM_err <= 0.001):
        #    return -np.inf

        idx = 4
        # Additional priors based on model type
        if model_type == 'model1':
            zeta = params[idx]
            if not (prior_bounds['zeta'][0] <= zeta <= prior_bounds['zeta'][1]):
                return -np.inf
        elif model_type == 'model2':
            tau_1GHz = params[idx]
            alpha = params[idx + 1]
            if not (prior_bounds['tau_1GHz'][0] <= tau_1GHz <= prior_bounds['tau_1GHz'][1]):
                return -np.inf
            if not (prior_bounds['alpha'][0] <= alpha <= prior_bounds['alpha'][1]):
                return -np.inf
        elif model_type == 'model3':
            zeta = params[idx]
            tau_1GHz = params[idx + 1]
            alpha = params[idx + 2]
            if not (prior_bounds['zeta'][0] <= zeta <= prior_bounds['zeta'][1]):
                return -np.inf
            if not (prior_bounds['tau_1GHz'][0] <= tau_1GHz <= prior_bounds['tau_1GHz'][1]):
                return -np.inf
            if not (prior_bounds['alpha'][0] <= alpha <= prior_bounds['alpha'][1]):
                return -np.inf

        #idx = 4
        ## Additional priors based on model type
        #if model_type == 'model1':
        #    zeta = params[idx]
        #    if not (0 <= zeta <= 0.4):
        #        return -np.inf
        #elif model_type == 'model2':
        #    tau_1GHz = params[idx]
        #    alpha = params[idx + 1]
        #    if not (0 <= tau_1GHz <= 0.4):
        #        return -np.inf
        #    if not (4.39999 <= alpha <= 4.40001):
        #        return -np.inf
        #elif model_type == 'model3':
        #    zeta = params[idx]
        #    tau_1GHz = params[idx + 1]
        #    alpha = params[idx + 2]
        #    if not (0 <= zeta <= 0.4):
        #        return -np.inf
        #    if not (0 <= tau_1GHz <= 0.4):
        #        return -np.inf
        #    if not (4.39999 <= alpha <= 4.40001):
        #        return -np.inf
        return 0.0  # Log-prior is zero if within bounds

    def log_posterior(self, params, prior_bounds, model_type='model0'):
        """
        Compute the log-posterior probability.

        Parameters:
        - params: Model parameters
        - model_type: One of 'model0', 'model1', 'model2', 'model3'
        """
        #lp = self.log_prior(params, model_type)
        #if not np.isfinite(lp):
        #    return -np.inf
        #ll = self.log_likelihood(params, model_type)
        lp = self.log_prior(params, prior_bounds, model_type)
        if not np.isfinite(lp):
            #print(f"Log prior is not finite: {lp} for params: {params}")
            return -np.inf
        ll = self.log_likelihood(params, model_type)
        if not np.isfinite(ll):
            #print(f"Log likelihood is not finite: {ll} for params: {params}")
            return -np.inf
        return lp + ll

# Functions for MCMC and model fitting
#def run_mcmc(model, initial_params, nwalkers=None, nsteps=20000, model_type='model0'):
#    ndim = len(initial_params)
#    if nwalkers is None:
#        #nwalkers = 2 * ndim  # At least twice the number of dimensions
#        nwalkers = max(10 * ndim, 50)  # Use more walkers if needed
#    ## Increase the perturbation scale
#    #perturbation_scale = 1e-2
#    #p0 = initial_params + perturbation_scale * np.random.randn(nwalkers, ndim)
#    
#    # Set perturbation scales relative to parameter magnitudes
#    perturbation_scale = np.array(initial_params) * 0.1
#    perturbation_scale[perturbation_scale == 0] = 0.1  # Prevent zero perturbation
#    p0 = initial_params + perturbation_scale * np.random.randn(nwalkers, ndim)
#    
#    # Ensure initial positions are within prior bounds
#    p0[:, 0] = np.abs(p0[:, 0])  # c0 > 0
#    p0[:, 1] = np.clip(p0[:, 1], -10, 10)  # spectral_index between -10 and 10
#    p0[:, 2] = np.clip(p0[:, 2], model.time[0], model.time[-1])  # t0 within time range
#    p0[:, 3] = np.clip(p0[:, 3], -1000, 1000)  # DM_err within bounds
#
#    idx = 4
#    if model_type == 'model1':
#        p0[:, idx] = np.clip(p0[:, idx], 0, 100)  # zeta
#    elif model_type == 'model2':
#        p0[:, idx] = np.clip(p0[:, idx], 0, 1000)  # tau_1GHz
#        p0[:, idx + 1] = np.clip(p0[:, idx + 1], 0, 10)  # alpha
#    elif model_type == 'model3':
#        p0[:, idx] = np.clip(p0[:, idx], 0, 100)  # zeta
#        p0[:, idx + 1] = np.clip(p0[:, idx + 1], 0, 1000)  # tau_1GHz
#        p0[:, idx + 2] = np.clip(p0[:, idx + 2], 0, 10)  # alpha
#
#    sampler = emcee.EnsembleSampler(nwalkers, ndim, model.log_posterior, args=(model_type,))
#    sampler.run_mcmc(p0, nsteps, progress=True)
#    return sampler

def run_mcmc(model, initial_params, prior_bounds, nsteps=300, nwalkers=None, model_type='model0'):
    ndim = len(initial_params)
    if nwalkers is None:
        nwalkers = max(10 * ndim, 50)
    
    # Define prior bounds
    #prior_bounds = {
    #    'c0': (1e-3, 100),  # Adjusted as per prior
    #    't0': (model.time[0], model.time[-1]),
    #    'spectral_index': (-3, 1),
    #    'DM_err': (-0.001, 0.001),
    #    'zeta': (0, 0.4),
    #    'tau_1GHz': (0, 0.4),
    #    'alpha': (3.5, 4.5)
    #}
    
    # For additional parameters
    #idx = 4
    #if model_type == 'model1':
    #    prior_bounds['zeta'] = (0, 0.4)
    #elif model_type == 'model2':
    #    prior_bounds['tau_1GHz'] = (0, 0.4)
    #    prior_bounds['alpha'] = (3.5, 4.5)
    #elif model_type == 'model3':
    #    prior_bounds['zeta'] = (0, 0.4)
    #    prior_bounds['tau_1GHz'] = (0, 0.4)
    #    prior_bounds['alpha'] = (3.5, 4.5)
    
    # Initialize p0 within the prior bounds
    p0 = []
    for i in range(nwalkers):
        walker_params = []
        # c0
        c0_min, c0_max = prior_bounds['c0']
        walker_params.append(np.random.uniform(c0_min, c0_max))
        # t0
        t0_min, t0_max = prior_bounds['t0']
        walker_params.append(np.random.uniform(t0_min, t0_max))
        # spectral_index
        si_min, si_max = prior_bounds['spectral_index']
        walker_params.append(np.random.uniform(si_min, si_max))
        # DM_err
        DM_err_min, DM_err_max = prior_bounds['DM_err']
        walker_params.append(np.random.uniform(DM_err_min, DM_err_max))
        # Additional parameters
        if model_type == 'model1':
            zeta_min, zeta_max = prior_bounds['zeta']
            walker_params.append(np.random.uniform(zeta_min, zeta_max))
        elif model_type == 'model2':
            tau_min, tau_max = prior_bounds['tau_1GHz']
            alpha_min, alpha_max = prior_bounds['alpha']
            walker_params.append(np.random.uniform(tau_min, tau_max))
            walker_params.append(np.random.uniform(alpha_min, alpha_max))
        elif model_type == 'model3':
            zeta_min, zeta_max = prior_bounds['zeta']
            tau_min, tau_max = prior_bounds['tau_1GHz']
            alpha_min, alpha_max = prior_bounds['alpha']
            walker_params.append(np.random.uniform(zeta_min, zeta_max))
            walker_params.append(np.random.uniform(tau_min, tau_max))
            walker_params.append(np.random.uniform(alpha_min, alpha_max))
        p0.append(walker_params)
    p0 = np.array(p0)
    
    # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, model.log_posterior, args=(prior_bounds, model_type))
    sampler.run_mcmc(p0, nsteps, progress=True)
    return sampler

#def compute_bic(lnL_max, k, n):
#    """
#    Compute the Bayesian Information Criterion (BIC).
#
#    Parameters:
#    - lnL_max: Maximum log-likelihood
#    - k: Number of model parameters
#    - n: Number of data points
#    """
#    BIC = k * np.log(n) - 2 * lnL_max
#    return BIC

def compute_bic(lnL_max, k, n):
    """
    Compute the Bayesian Information Criterion (BIC).

    Parameters:
    - lnL_max: Maximum log-likelihood
    - k: Number of model parameters
    - n: Number of data points
    """
    BIC = -2 * lnL_max + k * np.log(n)
    return BIC

def fit_models(model, init_params, prior_bounds, numsteps=300, fit_m0=True, fit_m1=True, fit_m2=True, fit_m3=True):
    """
    Fit all models and select the best one based on BIC.

    Parameters:
    - model: FRBModel instance
    """
    results = {}
    n = model.data.size  # Total number of data points

    # Define initial parameters for each model
    #initial_c0 = np.max(np.sum(model.data, axis=1))
    #initial_spectral_index = -1.0  # Initial guess
    #initial_t0 = model.time[np.argmax(np.sum(model.data, axis=0))]
    #initial_DM_err = 0.0
    # Define initial parameters for each model
    #initial_c0 = 1.0  # Close to the true value
    #initial_spectral_index = 2.5  # Close to the true value
    #initial_t0 = 1.0 #model.time[np.argmax(np.sum(model.data, axis=0))]
    #initial_DM_err = 0.0  # Assuming we expect DM_err close to zero

    # Unpack initial parameters
    c0, t0, gamma, DM_err, zeta, tau, alpha = init_params
    
    # Estimate c0 from data (initial spectrum amplitude)
    initial_c0 = c0 #np.max(np.sum(model.data, axis=1)) / model.data.shape[1]

    # Estimate t0 from data
    initial_t0 = t0 #model.time[np.argmax(np.sum(model.data, axis=0))]

    # Set initial spectral index
    initial_spectral_index = gamma #-1.0  # Adjust based on expectations

    # Set initial DM_err
    initial_DM_err = DM_err #0.0  # If data is dedispersed

    # Set initial zeta (burst width)
    initial_zeta = zeta #0.05 # ms

    # Set initial scattering timescale
    initial_tau_1GHz = tau # ms

    # Set initial scattering index
    initial_alpha = alpha # dimensionles

    # Common initial parameters
    initial_common = [initial_c0, initial_t0, initial_spectral_index, initial_DM_err]

    prior_bounds_all = prior_bounds.copy()

    print('Check Prior Bounds: ', prior_bounds_all)

    # Initialize BIC and results
    BIC0 = np.nan
    BIC1 = np.nan
    BIC2 = np.nan
    BIC3 = np.nan

    # Model 0
    if fit_m0:
        print("Fitting Model 0")
        initial_params0 = initial_common.copy()
        sampler0 = run_mcmc(model, initial_params0, prior_bounds_all, nsteps=numsteps, model_type='model0')
        lnL_max0 = np.max(sampler0.get_log_prob())
        k0 = len(initial_params0)
        BIC0 = compute_bic(lnL_max0, k0, n)
        results['model0'] = {'sampler': sampler0, 'BIC': BIC0, 'lnL_max': lnL_max0, 'k': k0}
    else:
        results['model0'] = {'sampler': np.nan, 'BIC': BIC0, 'lnL_max': np.nan, 'k': np.nan}

    # Model 1
    if fit_m1:
        print("Fitting Model 1")
        initial_params1 = initial_common + [initial_zeta]  # Add zeta
        sampler1 = run_mcmc(model, initial_params1, prior_bounds_all, nsteps=numsteps, model_type='model1')
        lnL_max1 = np.max(sampler1.get_log_prob())
        k1 = len(initial_params1)
        BIC1 = compute_bic(lnL_max1, k1, n)
        results['model1'] = {'sampler': sampler1, 'BIC': BIC1, 'lnL_max': lnL_max1, 'k': k1}
    else:
        results['model1'] = {'sampler': np.nan, 'BIC': BIC1, 'lnL_max': np.nan, 'k': np.nan}

    # Model 2
    if fit_m2:
        print("Fitting Model 2")
        initial_params2 = initial_common + [initial_tau_1GHz, initial_alpha]  # Add tau_1GHz and alpha
        sampler2 = run_mcmc(model, initial_params2, prior_bounds_all, nsteps=numsteps, model_type='model2')
        lnL_max2 = np.max(sampler2.get_log_prob())
        k2 = len(initial_params2)
        BIC2 = compute_bic(lnL_max2, k2, n)
        results['model2'] = {'sampler': sampler2, 'BIC': BIC2, 'lnL_max': lnL_max2, 'k': k2}
    else:
        results['model2'] = {'sampler': np.nan, 'BIC': BIC2, 'lnL_max': np.nan, 'k': np.nan}


    # Model 3
    if fit_m3:
        print("Fitting Model 3")
        initial_params3 = initial_common + [initial_zeta, initial_tau_1GHz, initial_alpha]  # Add zeta, tau_1GHz, alpha
        sampler3 = run_mcmc(model, initial_params3, prior_bounds_all, nsteps=numsteps, model_type='model3')
        lnL_max3 = np.max(sampler3.get_log_prob())
        k3 = len(initial_params3)
        BIC3 = compute_bic(lnL_max3, k3, n)
        results['model3'] = {'sampler': sampler3, 'BIC': BIC3, 'lnL_max': lnL_max3, 'k': k3}
    else:
        results['model3'] = {'sampler': np.nan, 'BIC': BIC3, 'lnL_max': np.nan, 'k': np.nan}

    # Select best model based on BIC
    BICs = [BIC0, BIC1, BIC2, BIC3]
    min_BIC = np.nanmin(BICs)
    min_idx = BICs.index(min_BIC)
    model_names = ['model0', 'model1', 'model2', 'model3']
    best_model = model_names[min_idx]

    # Check if a simpler model is within 6 units of the lowest BIC
    #threshold = 6 # positive evidence against the model with the higher BIC is 2 <= Delta BIC < 6
    #for i in range(len(BICs)):
    #    if i != min_idx and (BICs[i] - min_BIC) <= 6:
    #        if results[model_names[i]]['k'] < results[best_model]['k']:
    #            best_model = model_names[i]

    print("Best model is", best_model)
    return results, best_model

def downsample_data(data, f_factor = 1, t_factor = 1):   

    # Check data shape
    print(f'Power Shape (frequency axis): {data.shape[0]}')
    print(f'Power Shape (time axis): {data.shape[1]}')

    # Downsample in frequency
    # Ensure nearest multiple is not greater than the frequency axis length
    nrst_mltpl_f = f_factor * (data.shape[0] // f_factor)
    print(f'Nearest Multiple To Downsampling Factor (frequency): {nrst_mltpl_f}')

    # Clip the frequency axis to the nearest multiple
    data_clip_f = data[:nrst_mltpl_f, :]

    # Downsample along the frequency axis (y-axis)
    data_ds_f = data_clip_f.reshape([
        nrst_mltpl_f // f_factor, f_factor,
        data_clip_f.shape[1]
    ]).mean(axis=1)

    # Downsample in time
    # Ensure nearest multiple is not greater than the time axis length
    nrst_mltpl_t = t_factor * (data_ds_f.shape[1] // t_factor)
    print(f'Nearest Multiple To Downsampling Factor (time): {nrst_mltpl_t}')

    # Clip the time axis to the nearest multiple
    data_clip_t = data_ds_f[:, :nrst_mltpl_t]

    # Downsample along the time axis (x-axis)
    data_ds_t = data_clip_t.reshape([
        data_clip_t.shape[0],  # Frequency axis remains the same
        nrst_mltpl_t // t_factor, t_factor
    ]).mean(axis=2)

    # Output the final downsampled data
    print(f'Downsampled Data Shape: {data_ds_t.shape}')

    return data_ds_t
