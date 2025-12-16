"""
priors_physical.py
==================

Astrophysically informed priors for scattering analysis using Galactic
electron density models (NE2001, YMW16).

This module provides functions to build prior distributions that incorporate
our physical understanding of interstellar scattering:

1. **Scattering timescale (τ)**: Log-normal prior centered on NE2001 prediction
   with ~0.5 dex uncertainty (factor of 3)
   
2. **Spectral index (α)**: Gaussian prior centered on theoretical values
   - α = 4.0 for thin screen (Gaussian spectrum)
   - α = 4.4 for Kolmogorov turbulence
   
3. **Intrinsic width (ζ)**: Prior from typical FRB durations

Usage
-----
```python
from priors_physical import build_physical_priors, get_ne2001_scattering

# Get NE2001 prediction
tau_pred, nu_scint = get_ne2001_scattering(ra=180.0, dec=45.0, dm=500.0)

# Build priors for fitting
priors = build_physical_priors(
    ra_deg=180.0,
    dec_deg=45.0,
    dm=500.0,
    freq_ghz=1.0,
)

# Use with FRBFitter
fitter = FRBFitter(model, priors["bounds"], tau_prior=priors["tau"])
```

References
----------
- NE2001: Cordes & Lazio (2002), astro-ph/0207156
- YMW16: Yao, Manchester & Wang (2017), ApJ 835:29
- Kolmogorov α=4.4: Rickett (1990), ARA&A 28:561
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "get_ne2001_scattering",
    "build_physical_priors",
    "PhysicalPriors",
    "TURBULENCE_INDICES",
]

# Theoretical spectral indices for different turbulence spectra
TURBULENCE_INDICES = {
    "kolmogorov": 4.4,       # β = 11/3, α = 2β/(β-2) ≈ 4.4
    "thin_screen": 4.0,      # Gaussian angular spectrum
    "square_law": 4.0,       # Structure function ∝ r²
    "thick_medium": 3.5,     # Extended scattering region
}


@dataclass
class PhysicalPriors:
    """Container for physical prior specifications.
    
    Attributes
    ----------
    tau_lognormal : tuple
        (mu, sigma) for log-normal prior on τ (in log10 space: τ ~ 10^N(μ,σ²))
    alpha_gaussian : tuple
        (mu, sigma) for Gaussian prior on α
    bounds : dict
        Parameter bounds for uniform components of priors
    ne2001_tau_1ghz : float
        NE2001 prediction for τ at 1 GHz (ms)
    ne2001_nu_scint : float
        NE2001 prediction for scintillation bandwidth (MHz)
    """
    tau_lognormal: Tuple[float, float]
    alpha_gaussian: Tuple[float, float]
    bounds: Dict[str, Tuple[float, float]]
    ne2001_tau_1ghz: float
    ne2001_nu_scint: float
    
    def __repr__(self) -> str:
        return (
            f"PhysicalPriors(\n"
            f"  τ ~ LogNormal(μ={self.tau_lognormal[0]:.2f}, σ={self.tau_lognormal[1]:.2f}) [log10]\n"
            f"  α ~ Normal(μ={self.alpha_gaussian[0]:.2f}, σ={self.alpha_gaussian[1]:.2f})\n"
            f"  NE2001 prediction: τ(1GHz) = {self.ne2001_tau_1ghz:.3f} ms\n"
            f")"
        )


def get_ne2001_scattering(
    ra_deg: float,
    dec_deg: float,
    dm: float,
    freq_mhz: float = 1000.0,
    alpha: float = 4.0,
) -> Tuple[float, float]:
    """Query NE2001 for expected scattering at a sky position.
    
    Parameters
    ----------
    ra_deg : float
        Right ascension in degrees
    dec_deg : float
        Declination in degrees
    dm : float
        Dispersion measure (pc cm⁻³)
    freq_mhz : float
        Frequency in MHz (default 1000 = 1 GHz)
    alpha : float
        Frequency scaling index (default 4.0)
        
    Returns
    -------
    tau_ms : float
        Scattering timescale at freq_mhz (ms)
    nu_scint_khz : float
        Scintillation bandwidth at freq_mhz (kHz)
        
    Notes
    -----
    Requires the `mwprop` package: pip install mwprop
    Falls back to empirical relation if NE2001 unavailable.
    """
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        from mwprop.ne2001p.NE2001 import ne2001
        
        # Convert to Galactic coordinates
        coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame='icrs')
        l_deg = coord.galactic.l.deg
        b_deg = coord.galactic.b.deg
        
        # Query NE2001 (ndir=-1 means use DM to infer distance)
        Dk, Dv, Du, Dd = ne2001(
            ldeg=l_deg,
            bdeg=b_deg,
            dmd=dm,
            ndir=-1,
            classic=False,
            dmd_only=False,
        )
        
        # τ at 1 GHz from NE2001
        tau_1ghz_ms = Dv['TAU']  # Already in ms at 1 GHz
        
        # Scale to requested frequency
        tau_ms = tau_1ghz_ms * (1000.0 / freq_mhz) ** alpha
        
        # Scintillation bandwidth: Δν = 1/(2πτ)
        nu_scint_khz = 1.0 / (2 * np.pi * tau_ms * 1e-3) / 1e3
        
        return tau_ms, nu_scint_khz
        
    except ImportError:
        log.warning(
            "mwprop not available; using empirical DM-τ relation. "
            "Install with: pip install mwprop"
        )
        return _empirical_dm_tau_relation(dm, freq_mhz, alpha)
    except Exception as e:
        log.warning(f"NE2001 query failed: {e}. Using empirical relation.")
        return _empirical_dm_tau_relation(dm, freq_mhz, alpha)


def _empirical_dm_tau_relation(
    dm: float,
    freq_mhz: float = 1000.0,
    alpha: float = 4.0,
) -> Tuple[float, float]:
    """Empirical DM-τ relation from Bhat et al. (2004).
    
    log₁₀(τ/ms) = -6.46 + 1.07 log₁₀(DM) - 3.86 log₁₀(ν/GHz)
    
    Valid for pulsars; FRBs may have additional host/IGM contribution.
    """
    # Bhat (2004) relation at 1 GHz
    log_tau = -6.46 + 1.07 * np.log10(dm)
    tau_1ghz_ms = 10 ** log_tau
    
    # Scale to frequency
    tau_ms = tau_1ghz_ms * (1000.0 / freq_mhz) ** alpha
    
    # Scintillation bandwidth
    nu_scint_khz = 1.0 / (2 * np.pi * tau_ms * 1e-3) / 1e3
    
    return tau_ms, nu_scint_khz


def build_physical_priors(
    ra_deg: float,
    dec_deg: float,
    dm: float,
    freq_ghz: float = 1.0,
    tau_uncertainty_dex: float = 0.5,
    alpha_mean: float = 4.0,
    alpha_std: float = 0.5,
    turbulence_model: str = "thin_screen",
    allow_host_scattering: bool = True,
) -> PhysicalPriors:
    """Build astrophysically motivated priors for scattering parameters.
    
    Parameters
    ----------
    ra_deg : float
        Right ascension in degrees
    dec_deg : float
        Declination in degrees  
    dm : float
        Dispersion measure (pc cm⁻³)
    freq_ghz : float
        Reference frequency in GHz
    tau_uncertainty_dex : float
        Uncertainty on τ in dex (0.5 = factor of 3)
    alpha_mean : float
        Mean of Gaussian prior on α
    alpha_std : float
        Standard deviation of Gaussian prior on α
    turbulence_model : str
        Override alpha_mean with theoretical value:
        "kolmogorov", "thin_screen", "thick_medium", or None
    allow_host_scattering : bool
        If True, allow τ to exceed NE2001 prediction (host contribution)
        
    Returns
    -------
    PhysicalPriors
        Prior specifications for use with FRBFitter
    """
    # Get NE2001 prediction
    freq_mhz = freq_ghz * 1000.0
    tau_pred_ms, nu_scint_khz = get_ne2001_scattering(ra_deg, dec_deg, dm, freq_mhz)
    
    # Scale τ to 1 GHz for consistency with fitting
    tau_1ghz_pred = tau_pred_ms * (freq_mhz / 1000.0) ** alpha_mean
    
    # Log-normal prior on τ: centered on NE2001 prediction
    # In log10 space: τ ~ 10^N(log10(τ_pred), σ²)
    log10_tau_mean = np.log10(max(tau_1ghz_pred, 1e-6))  # Guard against zero
    log10_tau_std = tau_uncertainty_dex
    
    # Allow host/extragalactic contribution by widening upper bound
    if allow_host_scattering:
        log10_tau_std = max(log10_tau_std, 1.0)  # At least 1 dex to allow host
    
    tau_lognormal = (log10_tau_mean, log10_tau_std)
    
    # Alpha prior
    if turbulence_model is not None and turbulence_model in TURBULENCE_INDICES:
        alpha_mean = TURBULENCE_INDICES[turbulence_model]
    
    alpha_gaussian = (alpha_mean, alpha_std)
    
    # Parameter bounds (uniform components)
    # These should be wide enough to not truncate the physical priors
    tau_lo = 10 ** (log10_tau_mean - 3 * log10_tau_std)
    tau_hi = 10 ** (log10_tau_mean + 3 * log10_tau_std)
    
    bounds = {
        "c0": (0.01, 100.0),
        "t0": (-10.0, 50.0),  # ms, relative to data start
        "gamma": (-5.0, 5.0),  # spectral index
        "zeta": (0.001, 10.0),  # ms, intrinsic width
        "tau_1ghz": (max(1e-6, tau_lo), max(100.0, tau_hi)),  # ms
        "alpha": (max(2.0, alpha_mean - 3*alpha_std), 
                  min(6.0, alpha_mean + 3*alpha_std)),
        "delta_dm": (-5.0, 5.0),  # pc cm⁻³
    }
    
    log.info(
        f"Built physical priors:\n"
        f"  Position: ({ra_deg:.2f}, {dec_deg:.2f}) deg\n"
        f"  DM: {dm:.1f} pc/cm³\n"
        f"  NE2001 τ(1GHz): {tau_1ghz_pred:.4f} ms\n"
        f"  τ prior: log10(τ) ~ N({log10_tau_mean:.2f}, {log10_tau_std:.2f}²)\n"
        f"  α prior: N({alpha_mean:.2f}, {alpha_std:.2f}²)"
    )
    
    return PhysicalPriors(
        tau_lognormal=tau_lognormal,
        alpha_gaussian=alpha_gaussian,
        bounds=bounds,
        ne2001_tau_1ghz=tau_1ghz_pred,
        ne2001_nu_scint=nu_scint_khz,
    )


def apply_physical_priors_to_fitter(
    fitter,
    physical_priors: PhysicalPriors,
) -> None:
    """Apply physical priors to an existing FRBFitter instance.
    
    Modifies the fitter in-place to use NE2001-informed priors.
    """
    # Update alpha prior
    fitter.alpha_prior = physical_priors.alpha_gaussian
    
    # Update tau prior (passed to log_prob wrapper)
    fitter.tau_prior = physical_priors.tau_lognormal
    
    # Update bounds
    fitter.priors.update(physical_priors.bounds)
    
    log.info("Applied physical priors to fitter")


def get_burst_priors_from_catalog(
    burst_name: str,
    catalog_path: str = "configs/bursts.yaml",
) -> PhysicalPriors:
    """Get physical priors for a burst from the catalog.
    
    Convenience function that looks up RA, Dec, DM from bursts.yaml.
    """
    import yaml
    from pathlib import Path
    
    # Find catalog relative to this file or use absolute path
    catalog = Path(catalog_path)
    if not catalog.is_absolute():
        # Try relative to repository root
        repo_root = Path(__file__).parent.parent.parent
        catalog = repo_root / catalog_path
    
    with open(catalog) as f:
        data = yaml.safe_load(f)
    
    bursts = data.get("bursts", {})
    if burst_name not in bursts:
        available = list(bursts.keys())
        raise ValueError(f"Burst '{burst_name}' not in catalog. Available: {available}")
    
    b = bursts[burst_name]
    
    # Get coordinates (may need to parse from string)
    ra = b.get("ra_deg", b.get("ra", 0.0))
    dec = b.get("dec_deg", b.get("dec", 0.0))
    dm = b["dm"]
    
    # Handle sexagesimal if needed
    if isinstance(ra, str):
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        coord = SkyCoord(ra, dec, frame='icrs')
        ra = coord.ra.deg
        dec = coord.dec.deg
    
    return build_physical_priors(ra_deg=ra, dec_deg=dec, dm=dm)


# Utility: log-normal prior probability
def log_prob_lognormal(x: float, mu: float, sigma: float) -> float:
    """Log probability for log-normal prior (in log10 space).
    
    x ~ 10^N(μ, σ²) → log10(x) ~ N(μ, σ²)
    
    Parameters
    ----------
    x : float
        Parameter value (must be positive)
    mu : float
        Mean of log10(x)
    sigma : float
        Standard deviation of log10(x)
        
    Returns
    -------
    log_prob : float
        Log probability density
    """
    if x <= 0:
        return -np.inf
    
    log10_x = np.log10(x)
    z = (log10_x - mu) / sigma
    
    # Normal PDF for log10(x), plus Jacobian 1/(x * ln(10))
    return -0.5 * z**2 - np.log(sigma) - np.log(x) - np.log(np.log(10))
