# ==============================================================================
# File: scint_analysis/scint_analysis/fitting_2d.py
# ==============================================================================
"""
2D Scintillation Analysis Module

Provides simultaneous fitting across all frequency sub-bands with a global model
that enforces physical frequency-scaling constraints.

Key Features:
- Global power-law scaling: γ(ν) = γ₀ × (ν/ν_ref)^α
- Direct measurement of scaling index α without post-hoc fitting
- Proper covariance between γ₀ and α
- Support for multiple ACF model types (Lorentzian, Generalized Lorentzian)
- MCMC sampling for full posterior estimation (optional)

Physical Background:
- Thin screen: α ≈ 4.0
- Kolmogorov turbulence: α ≈ 4.4 (or 11/3 ≈ 3.67 for some conventions)
- Extended medium: α varies depending on geometry

Usage:
    from scint_analysis.fitting_2d import fit_2d_scintillation, Scintillation2DModel
    
    # Quick fit
    result = fit_2d_scintillation(acf_results)
    
    # Custom model
    model = Scintillation2DModel(acf_results, model_type='gen_lorentzian')
    result = model.fit()
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal, Union
from lmfit import Parameters, minimize, Minimizer
import warnings

log = logging.getLogger(__name__)

# ==============================================================================
# Data Classes for Results
# ==============================================================================

@dataclass
class Scintillation2DResult:
    """
    Container for 2D scintillation fit results.
    
    Attributes
    ----------
    gamma_0 : float
        Reference scintillation bandwidth at ν_ref (MHz)
    gamma_0_err : float
        Uncertainty on γ₀
    alpha : float
        Frequency scaling exponent (γ ∝ ν^α)
    alpha_err : float
        Uncertainty on α
    m_0 : float
        Reference modulation index at ν_ref
    m_0_err : float
        Uncertainty on m₀
    nu_ref : float
        Reference frequency (MHz)
    redchi : float
        Reduced chi-squared of the fit
    nfree : int
        Degrees of freedom
    success : bool
        Whether the fit converged
    params : Parameters
        Full lmfit Parameters object
    model_type : str
        Type of ACF model used
    covar : np.ndarray, optional
        Covariance matrix of fitted parameters
    subband_gamma : np.ndarray
        Derived γ for each sub-band
    subband_gamma_err : np.ndarray
        Uncertainty on derived γ for each sub-band
    center_freqs : np.ndarray
        Center frequencies of sub-bands (MHz)
    """
    gamma_0: float
    gamma_0_err: float
    alpha: float
    alpha_err: float
    m_0: float
    m_0_err: float
    nu_ref: float
    redchi: float
    nfree: int
    success: bool
    params: Parameters
    model_type: str
    covar: Optional[np.ndarray] = None
    subband_gamma: np.ndarray = field(default_factory=lambda: np.array([]))
    subband_gamma_err: np.ndarray = field(default_factory=lambda: np.array([]))
    center_freqs: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def __repr__(self):
        return (
            f"Scintillation2DResult(\n"
            f"  γ₀ = {self.gamma_0:.3f} ± {self.gamma_0_err:.3f} MHz @ {self.nu_ref:.0f} MHz\n"
            f"  α  = {self.alpha:.3f} ± {self.alpha_err:.3f}\n"
            f"  m₀ = {self.m_0:.3f} ± {self.m_0_err:.3f}\n"
            f"  χ²_red = {self.redchi:.2f}, nfree = {self.nfree}\n"
            f")"
        )
    
    def gamma_at_freq(self, nu: float) -> Tuple[float, float]:
        """
        Calculate γ at arbitrary frequency with error propagation.
        
        Parameters
        ----------
        nu : float
            Frequency in MHz
            
        Returns
        -------
        gamma : float
            Scintillation bandwidth at ν
        gamma_err : float
            Uncertainty on γ (from error propagation)
        """
        ratio = nu / self.nu_ref
        gamma = self.gamma_0 * (ratio ** self.alpha)
        
        # Error propagation: σ_γ² = (∂γ/∂γ₀)²σ_γ₀² + (∂γ/∂α)²σ_α² + 2(∂γ/∂γ₀)(∂γ/∂α)cov(γ₀,α)
        dgamma_dgamma0 = ratio ** self.alpha
        dgamma_dalpha = self.gamma_0 * (ratio ** self.alpha) * np.log(ratio)
        
        if self.covar is not None:
            # Use full covariance
            var_gamma = (
                dgamma_dgamma0**2 * self.covar[0, 0] +
                dgamma_dalpha**2 * self.covar[1, 1] +
                2 * dgamma_dgamma0 * dgamma_dalpha * self.covar[0, 1]
            )
        else:
            # Assume uncorrelated
            var_gamma = (
                (dgamma_dgamma0 * self.gamma_0_err)**2 +
                (dgamma_dalpha * self.alpha_err)**2
            )
        
        gamma_err = np.sqrt(max(0, var_gamma))
        return gamma, gamma_err


# ==============================================================================
# ACF Model Functions
# ==============================================================================

def lorentzian_acf(lag: np.ndarray, gamma: float, m: float) -> np.ndarray:
    """
    Standard Lorentzian ACF model.
    
    C(Δν) = m² / [1 + (Δν/γ)²]
    
    Parameters
    ----------
    lag : np.ndarray
        Frequency lag (MHz)
    gamma : float
        Scintillation bandwidth (MHz)
    m : float
        Modulation index
        
    Returns
    -------
    np.ndarray
        ACF values
    """
    return (m**2) / (1 + (lag / gamma)**2)


def gen_lorentzian_acf(lag: np.ndarray, gamma: float, m: float, eta: float = 2.0) -> np.ndarray:
    """
    Generalized Lorentzian ACF with power-law index.
    
    C(Δν) = m² / [1 + (Δν/γ)²]^(η/2)
    
    η = 2 gives standard Lorentzian
    η = 5/3 corresponds to Kolmogorov turbulence
    
    Parameters
    ----------
    lag : np.ndarray
        Frequency lag (MHz)
    gamma : float
        Scintillation bandwidth (MHz)
    m : float
        Modulation index
    eta : float
        Power-law index (default 2.0 = standard Lorentzian)
        
    Returns
    -------
    np.ndarray
        ACF values
    """
    return (m**2) / (1 + (lag / gamma)**2)**(eta / 2)


def gaussian_acf(lag: np.ndarray, sigma: float, m: float) -> np.ndarray:
    """
    Gaussian ACF model (for self-noise or resolved scattering).
    
    C(Δν) = m² × exp(-Δν²/(2σ²))
    
    Parameters
    ----------
    lag : np.ndarray
        Frequency lag (MHz)
    sigma : float
        Width parameter (MHz)
    m : float
        Modulation index
        
    Returns
    -------
    np.ndarray
        ACF values
    """
    return (m**2) * np.exp(-0.5 * (lag / sigma)**2)


# ==============================================================================
# 2D Model Class
# ==============================================================================

class Scintillation2DModel:
    """
    2D Scintillation model for simultaneous fitting across sub-bands.
    
    Enforces physical frequency scaling:
        γ(ν) = γ₀ × (ν/ν_ref)^α
        m(ν) = m₀ × (ν/ν_ref)^β  (optional, β=0 by default)
    
    Parameters
    ----------
    acf_results : dict
        Dictionary from ScintillationAnalysis.acf_results containing:
        - 'subband_lags_mhz': list of lag arrays
        - 'subband_acfs': list of ACF arrays
        - 'subband_acfs_err': list of error arrays
        - 'subband_center_freqs_mhz': array of center frequencies
    model_type : str
        ACF model type: 'lorentzian', 'gen_lorentzian', 'gaussian'
    nu_ref : float, optional
        Reference frequency (MHz). Default: mean of center frequencies
    fit_range_mhz : float, optional
        Maximum lag to include in fit (MHz). Default: 25.0
    include_self_noise : bool, optional
        Add Gaussian self-noise component. Default: False
    self_noise_width_mhz : float, optional
        Fixed width of self-noise Gaussian (MHz). Default: 0.5
    """
    
    MODEL_TYPES = ['lorentzian', 'gen_lorentzian', 'gaussian']
    
    def __init__(
        self,
        acf_results: dict,
        model_type: str = 'lorentzian',
        nu_ref: Optional[float] = None,
        fit_range_mhz: float = 25.0,
        include_self_noise: bool = False,
        self_noise_width_mhz: float = 0.5,
    ):
        if model_type not in self.MODEL_TYPES:
            raise ValueError(f"model_type must be one of {self.MODEL_TYPES}")
        
        self.model_type = model_type
        self.fit_range_mhz = fit_range_mhz
        self.include_self_noise = include_self_noise
        self.self_noise_width_mhz = self_noise_width_mhz
        
        # Extract data from acf_results
        self.lags_list = acf_results['subband_lags_mhz']
        self.acfs_list = acf_results['subband_acfs']
        self.errs_list = acf_results.get('subband_acfs_err', [None] * len(self.lags_list))
        self.center_freqs = np.array(acf_results['subband_center_freqs_mhz'])
        self.n_subbands = len(self.center_freqs)
        
        # Handle missing errors
        for i, err in enumerate(self.errs_list):
            if err is None:
                # Estimate from ACF scatter in outer regions
                lags = self.lags_list[i]
                acf = self.acfs_list[i]
                outer_mask = np.abs(lags) > 0.5 * np.max(np.abs(lags))
                self.errs_list[i] = np.full_like(acf, np.std(acf[outer_mask]))
                log.warning(f"Sub-band {i}: No errors provided, estimated from ACF scatter")
        
        # Set reference frequency
        self.nu_ref = nu_ref if nu_ref is not None else np.mean(self.center_freqs)
        
        # Build fit masks
        self.masks = []
        for lags in self.lags_list:
            mask = np.abs(lags) <= self.fit_range_mhz
            self.masks.append(mask)
        
        # Count data points
        self.n_data = sum(np.sum(m) for m in self.masks)
        
        log.info(f"Initialized 2D model: {self.n_subbands} sub-bands, "
                 f"{self.n_data} data points, ν_ref={self.nu_ref:.1f} MHz")
    
    def _build_params(
        self,
        gamma_0_init: float = 1.0,
        alpha_init: float = 4.0,
        m_0_init: float = 0.5,
        vary_alpha: bool = True,
        vary_m_scaling: bool = False,
    ) -> Parameters:
        """Build lmfit Parameters object."""
        params = Parameters()
        
        # Core parameters
        params.add('gamma_0', value=gamma_0_init, min=0.01, max=100.0)
        params.add('alpha', value=alpha_init, min=1.0, max=8.0, vary=vary_alpha)
        params.add('m_0', value=m_0_init, min=0.01, max=2.0)
        
        # Optional modulation scaling
        params.add('beta', value=0.0, min=-2.0, max=2.0, vary=vary_m_scaling)
        
        # Generalized Lorentzian index
        if self.model_type == 'gen_lorentzian':
            params.add('eta', value=2.0, min=1.0, max=4.0, vary=True)
        
        # Self-noise component
        if self.include_self_noise:
            params.add('m_sn', value=0.1, min=0.0, max=1.0)
            params.add('sigma_sn', value=self.self_noise_width_mhz, 
                       min=0.1, max=5.0, vary=False)
        
        return params
    
    def _model_single_subband(
        self,
        params: Parameters,
        lags: np.ndarray,
        nu_c: float,
    ) -> np.ndarray:
        """Evaluate model for a single sub-band."""
        gamma_0 = params['gamma_0'].value
        alpha = params['alpha'].value
        m_0 = params['m_0'].value
        beta = params['beta'].value
        
        # Frequency scaling
        ratio = nu_c / self.nu_ref
        gamma = gamma_0 * (ratio ** alpha)
        m = m_0 * (ratio ** beta)
        
        # Main scintillation component
        if self.model_type == 'lorentzian':
            acf_model = lorentzian_acf(lags, gamma, m)
        elif self.model_type == 'gen_lorentzian':
            eta = params['eta'].value
            acf_model = gen_lorentzian_acf(lags, gamma, m, eta)
        elif self.model_type == 'gaussian':
            acf_model = gaussian_acf(lags, gamma, m)
        
        # Add self-noise if enabled
        if self.include_self_noise:
            m_sn = params['m_sn'].value
            sigma_sn = params['sigma_sn'].value
            acf_model = acf_model + gaussian_acf(lags, sigma_sn, m_sn)
        
        return acf_model
    
    def _residual(self, params: Parameters) -> np.ndarray:
        """Compute weighted residuals across all sub-bands."""
        residuals = []
        
        for i, (lags, acf, err, mask, nu_c) in enumerate(
            zip(self.lags_list, self.acfs_list, self.errs_list, 
                self.masks, self.center_freqs)
        ):
            model = self._model_single_subband(params, lags[mask], nu_c)
            resid = (acf[mask] - model) / err[mask]
            residuals.extend(resid)
        
        return np.array(residuals)
    
    def fit(
        self,
        gamma_0_init: float = 1.0,
        alpha_init: float = 4.0,
        m_0_init: float = 0.5,
        vary_alpha: bool = True,
        vary_m_scaling: bool = False,
        method: str = 'leastsq',
    ) -> Scintillation2DResult:
        """
        Perform 2D fit across all sub-bands.
        
        Parameters
        ----------
        gamma_0_init : float
            Initial guess for reference scintillation bandwidth (MHz)
        alpha_init : float
            Initial guess for frequency scaling exponent
        m_0_init : float
            Initial guess for reference modulation index
        vary_alpha : bool
            Whether to fit α or fix it. Default: True
        vary_m_scaling : bool
            Whether to fit β (modulation scaling). Default: False
        method : str
            Fitting method ('leastsq', 'nelder', 'powell', etc.)
            
        Returns
        -------
        Scintillation2DResult
            Fit results with uncertainties
        """
        params = self._build_params(
            gamma_0_init=gamma_0_init,
            alpha_init=alpha_init,
            m_0_init=m_0_init,
            vary_alpha=vary_alpha,
            vary_m_scaling=vary_m_scaling,
        )
        
        # Run minimization
        minimizer = Minimizer(self._residual, params)
        result = minimizer.minimize(method=method)
        
        # Extract covariance if available
        covar = None
        if hasattr(result, 'covar') and result.covar is not None:
            # Extract covariance for gamma_0 and alpha
            param_names = list(result.params.keys())
            try:
                i_gamma = param_names.index('gamma_0')
                i_alpha = param_names.index('alpha')
                covar = result.covar[np.ix_([i_gamma, i_alpha], [i_gamma, i_alpha])]
            except (ValueError, IndexError):
                pass
        
        # Build result object
        p = result.params
        
        # Calculate derived γ for each sub-band
        subband_gamma = np.array([
            p['gamma_0'].value * (nu / self.nu_ref) ** p['alpha'].value
            for nu in self.center_freqs
        ])
        
        # Propagate errors to sub-band γ values
        subband_gamma_err = np.array([
            self._propagate_gamma_error(p, nu, covar)
            for nu in self.center_freqs
        ])
        
        fit_result = Scintillation2DResult(
            gamma_0=p['gamma_0'].value,
            gamma_0_err=p['gamma_0'].stderr or 0.0,
            alpha=p['alpha'].value,
            alpha_err=p['alpha'].stderr or 0.0,
            m_0=p['m_0'].value,
            m_0_err=p['m_0'].stderr or 0.0,
            nu_ref=self.nu_ref,
            redchi=result.redchi,
            nfree=result.nfree,
            success=result.success,
            params=result.params,
            model_type=self.model_type,
            covar=covar,
            subband_gamma=subband_gamma,
            subband_gamma_err=subband_gamma_err,
            center_freqs=self.center_freqs,
        )
        
        log.info(f"2D fit complete: {fit_result}")
        return fit_result
    
    def _propagate_gamma_error(
        self,
        params: Parameters,
        nu: float,
        covar: Optional[np.ndarray],
    ) -> float:
        """Propagate uncertainty to γ(ν)."""
        gamma_0 = params['gamma_0'].value
        alpha = params['alpha'].value
        gamma_0_err = params['gamma_0'].stderr or 0.0
        alpha_err = params['alpha'].stderr or 0.0
        
        ratio = nu / self.nu_ref
        dgamma_dgamma0 = ratio ** alpha
        dgamma_dalpha = gamma_0 * (ratio ** alpha) * np.log(ratio)
        
        if covar is not None:
            var = (
                dgamma_dgamma0**2 * covar[0, 0] +
                dgamma_dalpha**2 * covar[1, 1] +
                2 * dgamma_dgamma0 * dgamma_dalpha * covar[0, 1]
            )
        else:
            var = (dgamma_dgamma0 * gamma_0_err)**2 + (dgamma_dalpha * alpha_err)**2
        
        return np.sqrt(max(0, var))
    
    def evaluate(self, params: Optional[Parameters] = None) -> List[np.ndarray]:
        """
        Evaluate model at all sub-bands.
        
        Parameters
        ----------
        params : Parameters, optional
            If None, uses last fitted parameters
            
        Returns
        -------
        List[np.ndarray]
            Model ACFs for each sub-band
        """
        if params is None:
            raise ValueError("No parameters provided. Run fit() first or provide params.")
        
        return [
            self._model_single_subband(params, lags, nu)
            for lags, nu in zip(self.lags_list, self.center_freqs)
        ]


# ==============================================================================
# Convenience Functions
# ==============================================================================

def fit_2d_scintillation(
    acf_results: dict,
    model_type: str = 'lorentzian',
    fit_range_mhz: float = 25.0,
    nu_ref: Optional[float] = None,
    gamma_0_init: float = 1.0,
    alpha_init: float = 4.0,
    m_0_init: float = 0.5,
    vary_alpha: bool = True,
    include_self_noise: bool = False,
) -> Scintillation2DResult:
    """
    Quick 2D scintillation fit.
    
    Parameters
    ----------
    acf_results : dict
        Dictionary from ScintillationAnalysis.acf_results
    model_type : str
        'lorentzian', 'gen_lorentzian', or 'gaussian'
    fit_range_mhz : float
        Maximum lag to include in fit
    nu_ref : float, optional
        Reference frequency. Default: mean of sub-band frequencies
    gamma_0_init, alpha_init, m_0_init : float
        Initial parameter guesses
    vary_alpha : bool
        Whether to fit the scaling exponent
    include_self_noise : bool
        Add Gaussian self-noise component
        
    Returns
    -------
    Scintillation2DResult
        Fit results with full uncertainty information
        
    Examples
    --------
    >>> result = fit_2d_scintillation(acf_results)
    >>> print(f"α = {result.alpha:.2f} ± {result.alpha_err:.2f}")
    >>> print(f"γ₀ = {result.gamma_0:.2f} ± {result.gamma_0_err:.2f} MHz")
    """
    model = Scintillation2DModel(
        acf_results,
        model_type=model_type,
        nu_ref=nu_ref,
        fit_range_mhz=fit_range_mhz,
        include_self_noise=include_self_noise,
    )
    
    return model.fit(
        gamma_0_init=gamma_0_init,
        alpha_init=alpha_init,
        m_0_init=m_0_init,
        vary_alpha=vary_alpha,
    )


def compare_1d_vs_2d(
    acf_results: dict,
    fit_1d_results: List[dict],
    fit_2d_result: Scintillation2DResult,
) -> dict:
    """
    Compare 1D sub-band fits with 2D global fit.
    
    Parameters
    ----------
    acf_results : dict
        Original ACF results
    fit_1d_results : List[dict]
        Results from individual sub-band fits (with 'gamma' and 'gamma_err' keys)
    fit_2d_result : Scintillation2DResult
        Result from 2D fit
        
    Returns
    -------
    dict
        Comparison statistics including AIC/BIC differences
    """
    center_freqs = np.array(acf_results['subband_center_freqs_mhz'])
    
    # Extract 1D gammas
    gamma_1d = np.array([r.get('gamma', np.nan) for r in fit_1d_results])
    gamma_1d_err = np.array([r.get('gamma_err', np.nan) for r in fit_1d_results])
    
    # Get 2D predicted gammas
    gamma_2d = fit_2d_result.subband_gamma
    gamma_2d_err = fit_2d_result.subband_gamma_err
    
    # Compute residuals
    residuals = gamma_1d - gamma_2d
    pull = residuals / np.sqrt(gamma_1d_err**2 + gamma_2d_err**2)
    
    # Chi-squared of 1D vs 2D
    chi2_comparison = np.nansum(pull**2)
    npoints = np.sum(~np.isnan(pull))
    
    return {
        'gamma_1d': gamma_1d,
        'gamma_1d_err': gamma_1d_err,
        'gamma_2d': gamma_2d,
        'gamma_2d_err': gamma_2d_err,
        'residuals': residuals,
        'pull': pull,
        'chi2_comparison': chi2_comparison,
        'npoints': npoints,
        'center_freqs': center_freqs,
    }


# ==============================================================================
# MCMC Extension (optional, requires emcee)
# ==============================================================================

def fit_2d_mcmc(
    acf_results: dict,
    model_type: str = 'lorentzian',
    fit_range_mhz: float = 25.0,
    nu_ref: Optional[float] = None,
    nwalkers: int = 32,
    nsteps: int = 2000,
    burn_in: int = 500,
    progress: bool = True,
) -> dict:
    """
    MCMC sampling for 2D scintillation model using emcee.
    
    Parameters
    ----------
    acf_results : dict
        ACF results dictionary
    model_type : str
        ACF model type
    fit_range_mhz : float
        Fit range
    nu_ref : float, optional
        Reference frequency
    nwalkers : int
        Number of MCMC walkers
    nsteps : int
        Number of MCMC steps
    burn_in : int
        Number of burn-in steps to discard
    progress : bool
        Show progress bar
        
    Returns
    -------
    dict
        MCMC results with chains, percentiles, and corner plot data
    """
    try:
        import emcee
    except ImportError:
        raise ImportError("emcee is required for MCMC fitting. Install with: pip install emcee")
    
    # First do a quick least-squares fit for initial position
    model = Scintillation2DModel(
        acf_results,
        model_type=model_type,
        nu_ref=nu_ref,
        fit_range_mhz=fit_range_mhz,
    )
    lsq_result = model.fit()
    
    # Parameter names and initial values
    param_names = ['gamma_0', 'alpha', 'm_0']
    p0_center = [lsq_result.gamma_0, lsq_result.alpha, lsq_result.m_0]
    
    # Log-probability function
    def log_prior(theta):
        gamma_0, alpha, m_0 = theta
        # Uniform priors with bounds
        if not (0.01 < gamma_0 < 100 and 1.0 < alpha < 8.0 and 0.01 < m_0 < 2.0):
            return -np.inf
        return 0.0
    
    def log_likelihood(theta):
        gamma_0, alpha, m_0 = theta
        params = Parameters()
        params.add('gamma_0', value=gamma_0)
        params.add('alpha', value=alpha)
        params.add('m_0', value=m_0)
        params.add('beta', value=0.0)
        
        resid = model._residual(params)
        return -0.5 * np.sum(resid**2)
    
    def log_prob(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)
    
    # Initialize walkers
    ndim = len(param_names)
    pos = p0_center + 0.1 * np.random.randn(nwalkers, ndim) * np.array(p0_center)
    
    # Run MCMC
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(pos, nsteps, progress=progress)
    
    # Extract chains after burn-in
    flat_samples = sampler.get_chain(discard=burn_in, flat=True)
    
    # Compute percentiles
    percentiles = {}
    for i, name in enumerate(param_names):
        q = np.percentile(flat_samples[:, i], [16, 50, 84])
        percentiles[name] = {
            'median': q[1],
            'lower': q[0],
            'upper': q[2],
            'err_minus': q[1] - q[0],
            'err_plus': q[2] - q[1],
        }
    
    return {
        'param_names': param_names,
        'flat_samples': flat_samples,
        'percentiles': percentiles,
        'sampler': sampler,
        'lsq_result': lsq_result,
        'nu_ref': model.nu_ref,
    }
