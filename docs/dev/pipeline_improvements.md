# FLITS Pipeline Core Methods Analysis

## Opportunities for More Sophisticated, Rigorous, and Robust Methods

This document identifies areas in the FLITS pipeline where more advanced methodologies could be implemented to improve accuracy, robustness, and scientific rigor.

---

## 1. Scattering Analysis Pipeline

### Current Implementation

**Core Physics (`scattering/scat_analysis/burstfit.py`):**

- Thin-screen scattering model with exponential tail
- Gaussian likelihood (or Student-t for robustness)
- emcee MCMC sampler
- BIC for model selection

**Model Selection (`burstfit_modelselect.py`):**

- Sequential fitting M0 → M3
- Bayesian Information Criterion (BIC)
- Max-likelihood estimation

### Opportunities for Improvement

#### 1.1 **Advanced Likelihood Models** 🔬

**Current:** Gaussian or Student-t likelihood

**Opportunity:**

```python
# Implement hierarchical Bayesian model with hyperparameters
# for systematic uncertainties

class HierarchicalLikelihood:
    """
    Account for:
    - Frequency-dependent noise correlations
    - Time-domain correlations
    - Systematic calibration uncertainties
    """
    def __init__(self, data, noise_cov_model='diagonal'):
        if noise_cov_model == 'diagonal':
            # Current: uncorrelated noise per pixel
            self.cov = np.diag(sigma**2)
        elif noise_cov_model == 'frequency_correlated':
            # NEW: Account for RFI, bandpass systematics
            self.cov = self._build_freq_corr_matrix()
        elif noise_cov_model == 'full':
            # ADVANCED: Full covariance including time correlations
            self.cov = self._build_full_cov()

    def _build_freq_corr_matrix(self):
        """Model correlations between frequency channels."""
        # Use GP kernel or empirical correlation from off-pulse
        return exponential_kernel + white_noise
```

**Benefits:**

- More accurate parameter uncertainties
- Properly accounts for systematic effects
- Reduces bias from correlated noise

**Implementation Complexity:** Medium  
**Expected Impact:** High - improved parameter constraints

---

#### 1.2 **Nested Sampling for Model Comparison** 📊

**Current:** BIC-based model selection

**Opportunity:**

```python
# Use dynesty or UltraNest for rigorous Bayesian evidence calculation

import dynesty
from dynesty import NestedSampler

def compute_evidence(model_key, data, priors):
    """
    Nested sampling provides:
    - Bayesian evidence (marginal likelihood)
    - Posterior samples as byproduct
    - Better exploration of multimodal posteriors
    """
    sampler = NestedSampler(
        loglikelihood_fn,
        prior_transform_fn,
        ndim=len(priors),
        nlive=500,  # More live points = better evidence estimate
        sample='rwalk'  # or 'slice' for difficult geometries
    )
    sampler.run_nested()

    evidence = sampler.results.logz  # ln(Z)
    evidence_err = sampler.results.logzerr

    # Bayes factor instead of BIC
    return evidence, evidence_err
```

**Benefits:**

- Rigorous model comparison via Bayes factors
- Better handles multimodal posteriors
- No reliance on asymptotic approximations (BIC)
- Evidence uncertainty quantification

**Current:** `BIC = -2*logL_max + k*ln(n)`  
**Improved:** `log(Bayes Factor) = log(Z_M2/Z_M1)`

**Implementation Complexity:** Medium  
**Expected Impact:** High - more robust model selection

---

#### 1.3 **Physical Priors from NE2001/YMW16** 🌌

**Current:** Uniform/log-uniform priors

**Opportunity:**

```python
def build_physical_priors(burst_position, freq_range):
    """
    Incorporate astrophysical knowledge:
    - NE2001/YMW16 for expected scattering
    - Galactic latitude dependence
    - Distance-dependent scattering
    """
    ra, dec = burst_position

    # Query electron density model
    tau_expected = query_ne2001(ra, dec, freq_range)
    tau_uncertainty = 0.5  # dex uncertainty

    # Log-normal prior centered on prediction
    tau_prior = LogNormal(
        mu=np.log(tau_expected),
        sigma=tau_uncertainty
    )

    # Spectral index: informed by theory + observations
    alpha_prior = Normal(mu=4.0, sigma=0.5)  # Kolmogorov=4.4, β=11/3→4.4

    return {'tau_1ghz': tau_prior, 'alpha': alpha_prior}
```

**Benefits:**

- Incorporates prior astrophysical knowledge
- Helps constrain poorly-determined parameters
- Enables detection of anomalous scattering

**Implementation Complexity:** Low-Medium  
**Expected Impact:** Medium - tighter constraints, physical consistency

---

#### 1.4 **Multi-Component Scattering Models** 🔀

**Current:** Single thin-screen model

**Opportunity:**

```python
class MultiScreenModel:
    """
    Two-screen model: Galactic + Host/IGM

    Model: I(ν,t) = S(t) * K_gal(t,ν) * K_host(t,ν)

    where:
    - K_gal: MW scattering (known from geometry)
    - K_host: Host galaxy / nearby screen (fit parameter)
    """
    def __init__(self, galaxy_tau_1ghz):
        self.tau_gal = galaxy_tau_1ghz  # Fixed from NE2001

    def __call__(self, params):
        # Galactic scattering (fixed)
        kernel_gal = self._scattering_kernel(self.tau_gal, alpha=4.4)

        # Host scattering (free parameter)
        kernel_host = self._scattering_kernel(params.tau_host, params.alpha_host)

        # Convolve intrinsic pulse with both screens
        profile = gaussian_pulse(params.t0, params.width)
        scattered = convolve(profile, kernel_gal)
        scattered = convolve(scattered, kernel_host)

        return scattered
```

**Benefits:**

- Physically motivated multi-component model
- Separates Galactic vs. extragalactic scattering
- Constraints on host galaxy environment

**Implementation Complexity:** High  
**Expected Impact:** High - new science from host scattering

---

#### 1.5 **Robust Scattering Tail Modeling** 📉

**Current:** Pure exponential tail

**Opportunity:**

```python
def generalized_scattering_kernel(t, tau, alpha, beta=2.0):
    """
    Stretched exponential or power-law tail

    Options:
    1. K(t) ∝ exp(-(t/τ)^β)  # β<1: heavy tail, β=1: exponential
    2. K(t) ∝ t^(-α) for t > t_break  # Power-law for thick screens
    3. Mixture model: exp + power-law
    """
    if model == 'stretched_exp':
        return np.exp(-(t/tau)**beta)
    elif model == 'power_law':
        return (1 + t/tau)**(-alpha)
    elif model == 'mixture':
        # Exponential for early times, power-law for late
        early = np.exp(-t/tau)
        late = (t/tau)**(-alpha)
        return np.where(t < t_break, early, late)
```

**Benefits:**

- Better fits for complex scattering geometries
- Handles thick-screen vs thin-screen regimes
- Distinguishes scattering mechanisms

**Implementation Complexity:** Medium  
**Expected Impact:** Medium - improved fits for complex bursts

---

## 2. Scintillation Analysis Pipeline

### Current Implementation

**Core:** ACF computation → model fitting → parameter extraction  
**Noise:** Empirical noise characterization  
**Fitting:** Lorentzian/Gaussian component fit

### Opportunities for Improvement

#### 2.1 **2D Scintillation Modeling** 🗺️

**Current:** 1D frequency ACF

**Opportunity:**

```python
class Scintillation2DModel:
    """
    Full 2D (ν, t) scintillation analysis

    Advantages:
    - Uses all information (not just frequency axis)
    - Constrains scintillation timescale directly
    - Detects anisotropic scintillation
    """
    def __init__(self, dynamic_spectrum):
        self.ds = dynamic_spectrum

    def compute_2d_acf(self):
        """2D autocorrelation function."""
        # FFT-based for efficiency
        ft = np.fft.fft2(self.ds - self.ds.mean())
        power = np.abs(ft)**2
        acf2d = np.fft.ifft2(power).real
        return np.fft.fftshift(acf2d)

    def fit_2d_model(self, acf2d):
        """
        Model: ACF(Δν, Δt) = A*exp(-Δν²/ν_d² - Δt²/t_d²)

        Fit parameters:
        - ν_d: decorrelation bandwidth
        - t_d: scintillation timescale
        - θ: anisotropy angle (if present)
        """
        return nu_d, t_d, theta
```

**Benefits:**

- Full use of data (not marginalizing over time)
- Direct measurement of scintillation timescale
- Anisotropy detection

**Implementation Note:** `fitting_2d.py` exists but may need enhancement

**Implementation Complexity:** Low (partially exists)  
**Expected Impact:** High - richer scintillation characterization

---

#### 2.2 **Bayesian Scintillation Inference** 🎲

**Current:** Least-squares ACF fitting

**Opportunity:**

```python
def bayesian_scint_inference(acf_data, acf_err):
    """
    MCMC sampling of scintillation parameters

    Advantages:
    - Proper uncertainty propagation
    - Handles non-Gaussian errors
    - Joint constraints on (ν_s, t_s, DM_var)
    """
    def log_prior(params):
        nu_s, t_s, dm_var = params
        # Physical priors
        if nu_s < 0 or t_s < 0 or dm_var < 0:
            return -np.inf
        # Informed priors from theory
        return -0.5*((np.log(nu_s/1e3))**2/1.0**2)  # Expected ~kHz scale

    def log_likelihood(params):
        model_acf = scintillation_model(*params)
        return -0.5*np.sum(((acf_data - model_acf)/acf_err)**2)

    # Run emcee
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(p0, nsteps)

    return sampler.get_chain(flat=True)
```

**Benefits:**

- Rigorous uncertainty quantification
- Detection of parameter degeneracies
- Marginalized constraints

**Implementation Complexity:** Medium  
**Expected Impact:** Medium-High - improved uncertainties

---

#### 2.3 **Noise Covariance Modeling** 📡

**Current:** Empirical noise estimation from off-pulse

**Opportunity:**

```python
class NoiseModel:
    """
    Sophisticated noise characterization

    Sources:
    - Radiometer noise (white)
    - RFI contamination (non-Gaussian)
    - System temperature variations
    - DM smearing (frequency-dependent)
    """
    def __init__(self, off_pulse_data):
        self.off_pulse = off_pulse_data

    def characterize_noise(self):
        # 1. Test for Gaussianity
        kurtosis = self._compute_kurtosis()
        is_gaussian = abs(kurtosis - 3) < threshold

        # 2. Frequency-dependent noise
        noise_spectrum = np.std(self.off_pulse, axis=0)

        # 3. Correlation structure
        noise_cov = np.cov(self.off_pulse.T)

        # 4. Outlier detection (RFI)
        outliers = self._detect_outliers()

        return {
            'is_gaussian': is_gaussian,
            'noise_per_channel': noise_spectrum,
            'covariance': noise_cov,
            'rfi_mask': outliers
        }
```

**Benefits:**

- More accurate ACF error bars
- RFI-robust scintillation measurements
- Handles non-Gaussian noise

**Implementation Complexity:** Medium  
**Expected Impact:** Medium - cleaner ACFs, better errors

---

## 3. DM Refinement

### Current Implementation

**Method:** Phase-coherence with bootstrap uncertainty  
**Optimization:** Quadratic peak fitting

### Opportunities for Improvement

#### 3.1 **Matched Filtering DM Estimation** 🎯

**Current:** Coherent power maximization

**Opportunity:**

```python
def matched_filter_dm(waterfall, template):
    """
    Cross-correlate with expected pulse shape

    SNR(DM) = ∫∫ [data * template(DM)] / noise

    Advantages:
    - Optimal for known pulse shapes
    - Better SNR than simple coherent power
    - Naturally weights by expected signal
    """
    snr_vs_dm = []
    for dm in dm_grid:
        dedispersed = dedisperse(waterfall, dm)
        # Cross-correlation with template
        cc = np.correlate(dedispersed.sum(axis=0), template, mode='valid')
        snr = cc.max() / noise_std
        snr_vs_dm.append(snr)

    return dm_grid[np.argmax(snr_vs_dm)]
```

**Benefits:**

- Higher SNR for faint bursts
- Less bias for complex morphologies
- Template can be empirical (from bright burst)

**Implementation Complexity:** Low-Medium  
**Expected Impact:** Medium - better DM precision for faint bursts

---

#### 3.2 **Joint DM + Scattering Fitting** 🔗

**Current:** DM optimized separately, then scattering fit

**Opportunity:**

```python
def joint_dm_scattering_inference():
    """
    Simultaneous fit of DM and scattering parameters

    Benefits:
    - Breaks degeneracy between DM smearing and scattering
    - Consistent parameter uncertainties
    - No bias from two-step procedure
    """
    def log_likelihood(dm, tau, alpha, width):
        # Build full model with exact DM
        model = build_scattered_model(dm, tau, alpha, width)
        chi2 = np.sum((data - model)**2 / noise**2)
        return -0.5 * chi2

    # Sample jointly
    result = optimize_or_sample(log_likelihood, priors)
    return dm_best, tau_best, correlations
```

**Benefits:**

- No bias from iterative DM → scattering procedure
- Proper accounting of DM/scattering degeneracy
- More realistic uncertainties

**Implementation Complexity:** Medium-High  
**Expected Impact:** High - more accurate parameters

---

## 4. Cross-Cutting Improvements

### 4.1 **Automatic Differentiation for Gradients** ⚡

**Current:** Finite differences or no gradients (pure MCMC)

**Opportunity:**

```python
import jax
import jax.numpy as jnp

@jax.jit
def forward_model_jax(params, freq, time):
    """JAX-accelerated forward model with automatic gradients."""
    # Same physics, but compiled and differentiable
    return scattered_pulse(params, freq, time)

# Use with gradient-based samplers
grad_log_prob = jax.grad(log_probability)

# Or use HMC/NUTS (much faster than emcee for complex models)
import numpyro
from numpyro.infer import MCMC, NUTS

nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=2000)
```

**Benefits:**

- 10-100x faster sampling (HMC vs random-walk)
- Gradient information aids geometry exploration
- GPU acceleration possible

**Implementation Complexity:** Medium (requires refactor to JAX)  
**Expected Impact:** Very High - massive speedup

---

### 4.2 **Simulation-Based Inference (SBI)** 🤖

**Cutting-Edge Opportunity:**

```python
from sbi import inference

def simulator(params):
    """Forward model: params → data."""
    tau, alpha, width, dm = params
    return generate_synthetic_burst(tau, alpha, width, dm)

# Train neural network to approximate posterior
posterior_nn = inference.SNPE(
    simulator=simulator,
    prior=build_priors()
)

# Learn posterior from simulations
posterior_nn.train(num_simulations=10000)

# Apply to real data
posterior_samples = posterior_nn.sample(observed_data)
```

**Benefits:**

- Amortized inference (fast after training)
- Handles complex/intractable likelihoods
- Natural for systematic studies (many bursts)

**Implementation Complexity:** High  
**Expected Impact:** Very High (if many bursts to analyze)

---

### 4.3 **Cross-Validation for Model Selection** ✅

**Current:** BIC on full dataset

**Opportunity:**

```python
def cross_validated_model_selection(data, models, n_folds=5):
    """
    K-fold cross-validation for model selection

    More robust than BIC:
    - Directly tests predictive power
    - Less sensitive to sample size
    - Guards against overfitting
    """
    scores = {model: [] for model in models}

    for fold in kfold_split(data, n_folds):
        train, test = fold
        for model in models:
            fit = model.fit(train)
            score = -fit.log_likelihood(test)  # Negative log-likelihood
            scores[model].append(score)

    # Best model = lowest mean test score
    return min(scores, key=lambda m: np.mean(scores[m]))
```

**Benefits:**

- More robust than BIC
- Direct test of generalization
- Detect overfitting

**Implementation Complexity:** Low  
**Expected Impact:** Medium - more reliable model selection

---

## 5. Implementation Priority Matrix

| Improvement                             | Complexity  | Impact    | Priority | Effort    |
| --------------------------------------- | ----------- | --------- | -------- | --------- |
| **Nested Sampling**                     | Medium      | High      | **1**    | 2-3 weeks |
| **Physical Priors (NE2001)**            | Low-Medium  | Medium    | **2**    | 1 week    |
| **2D Scintillation (enhance existing)** | Low         | High      | **3**    | 1 week    |
| **JAX + HMC/NUTS**                      | Medium      | Very High | **4**    | 3-4 weeks |
| **Hierarchical Likelihood**             | Medium      | High      | **5**    | 2 weeks   |
| **Joint DM+Scattering**                 | Medium-High | High      | **6**    | 2-3 weeks |
| **Multi-Screen Model**                  | High        | High      | **7**    | 4-6 weeks |
| **Cross-Validation**                    | Low         | Medium    | **8**    | 1 week    |

---

## 6. Quick Wins (Low-Hanging Fruit)

### 6.1 **Add Physical Priors**

- **Effort:** 1 week
- **Impact:** Medium
- Query NE2001, set informed priors on `tau` and `alpha`

### 6.2 **Enhance 2D Scintillation**

- **Effort:** 1 week
- **Impact:** High
- `fitting_2d.py` exists, integrate into main pipeline

### 6.3 **Cross-Validation**

- **Effort:** 1 week
- **Impact:** Medium
- Simple addition to model selection workflow

---

## Recommendations

**Short-term (1-2 months):**

1. Implement nested sampling for model comparison
2. Add NE2001-informed priors
3. Enhance 2D scintillation integration

**Medium-term (3-6 months):** 4. Migrate to JAX + HMC for 10-100x speedup 5. Implement hierarchical Bayesian likelihood 6. Add joint DM+scattering inference

**Long-term (6-12 months):** 7. Multi-screen scattering models 8. Simulation-based inference for systematic studies

**The pipeline is already sophisticated**, but these improvements would push it to **state-of-the-art** in FRB analysis!
