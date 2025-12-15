"""
dm_preprocessing.py
===================

Integrate DM estimation into the scattering analysis preprocessing pipeline.
This module provides utilities to estimate optimal DM values before running
the main scattering MCMC analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from dispersion.dmphasev2 import DMPhaseEstimator

log = logging.getLogger(__name__)

__all__ = ["estimate_dm_from_waterfall", "refine_dm_init"]


def estimate_dm_from_waterfall(
    waterfall: NDArray[np.floating],
    freqs: NDArray[np.floating],
    dt_ms: float,
    dm_catalog: float,
    dm_search_window: float = 5.0,
    dm_grid_resolution: float = 0.01,
    n_bootstrap: int = 200,
    f_cut_hz: Tuple[float, float] | None = None,
) -> dict:
    """
    Estimate optimal DM using phase-coherence method.
    
    Parameters
    ----------
    waterfall : ndarray
        2D dynamic spectrum (frequency × time)
    freqs : ndarray
        Frequency axis in GHz
    dt_ms : float
        Time resolution in milliseconds
    dm_catalog : float
        Catalog DM value (center of search range)
    dm_search_window : float
        Half-width of DM search range (pc/cm³)
    dm_grid_resolution : float
        DM grid spacing (pc/cm³)
    n_bootstrap : int
        Number of bootstrap resamples for uncertainty
    f_cut_hz : tuple or None
        Frequency cutoff range (lo, hi) in Hz for coherent power integration
        
    Returns
    -------
    dict
        Results dictionary with keys:
        - dm_best : float
            Best-fit DM value
        - dm_sigma : float
            Bootstrap uncertainty on DM
        - dm_offset : float
            Difference from catalog DM (dm_best - dm_catalog)
        - dm_curve : ndarray
            Coherent power vs DM
        - dm_grid : ndarray
            DM search grid
        - dm_curve_sigma : ndarray
            Bootstrap uncertainty on DM curve
    """
    log.info(f"Running DM estimation around catalog value {dm_catalog:.3f} pc/cm³")
    log.info(f"  Search window: ±{dm_search_window:.2f} pc/cm³")
    log.info(f"  Grid resolution: {dm_grid_resolution:.3f} pc/cm³")
    
    # Build DM search grid
    dm_min = dm_catalog - dm_search_window
    dm_max = dm_catalog + dm_search_window
    n_grid = int((dm_max - dm_min) / dm_grid_resolution) + 1
    dm_grid = np.linspace(dm_min, dm_max, n_grid)
    
    # Convert time resolution to seconds
    dt_sec = dt_ms * 1e-3
    
    # Initialize estimator
    estimator = DMPhaseEstimator(
        waterfall=waterfall,
        freqs=freqs,
        dt=dt_sec,
        dm_grid=dm_grid,
        ref="top",  # Use highest frequency as reference
        f_cut=f_cut_hz,
        n_boot=n_bootstrap,
    )
    
    # Get results
    dm_best, dm_sigma = estimator.get_dm()
    dm_offset = dm_best - dm_catalog
    
    log.info(f"  DM estimate: {dm_best:.3f} ± {dm_sigma:.3f} pc/cm³")
    log.info(f"  Offset from catalog: {dm_offset:+.3f} pc/cm³")
    
    return {
        "dm_best": float(dm_best),
        "dm_sigma": float(dm_sigma),
        "dm_offset": float(dm_offset),
        "dm_curve": estimator.dm_curve,
        "dm_grid": estimator.dm_grid,
        "dm_curve_sigma": estimator.dm_err,
        "catalog_dm": float(dm_catalog),
    }


def refine_dm_init(
    dataset,
    catalog_dm: float,
    enable_dm_estimation: bool = True,
    dm_search_window: float = 5.0,
    **kwargs,
) -> float:
    """
    Refine initial DM value for scattering analysis.
    
    If DM estimation is enabled, runs phase-coherence DM estimation.
    Otherwise, returns the catalog DM value.
    
    Parameters
    ----------
    dataset : BurstDataset
        Loaded burst dataset with data, freq, time axes
    catalog_dm : float
        Catalog DM value (from bursts.yaml)
    enable_dm_estimation : bool
        Whether to run DM estimation or use catalog value
    dm_search_window : float
        Half-width of DM search range (pc/cm³)
    **kwargs
        Additional arguments passed to estimate_dm_from_waterfall
        
    Returns
    -------
    float
        Refined DM value to use in scattering analysis
    """
    if not enable_dm_estimation:
        log.info(f"DM estimation disabled, using catalog DM: {catalog_dm:.3f} pc/cm³")
        return catalog_dm
    
    try:
        # Run DM estimation
        dm_results = estimate_dm_from_waterfall(
            waterfall=dataset.data,
            freqs=dataset.freq,
            dt_ms=dataset.dt_ms,
            dm_catalog=catalog_dm,
            dm_search_window=dm_search_window,
            **kwargs,
        )
        
        dm_refined = dm_results["dm_best"]
        dm_offset = dm_results["dm_offset"]
        
        # Only use refined DM if offset is within reasonable range
        if abs(dm_offset) > 2 * dm_search_window:
            log.warning(
                f"DM offset ({dm_offset:.3f}) exceeds 2× search window. "
                f"Using catalog DM {catalog_dm:.3f} instead."
            )
            return catalog_dm
        
        log.info(f"✓ Using refined DM: {dm_refined:.3f} pc/cm³ (offset: {dm_offset:+.3f})")
        return dm_refined
        
    except Exception as e:
        log.error(f"DM estimation failed: {e}")
        log.info(f"Falling back to catalog DM: {catalog_dm:.3f} pc/cm³")
        return catalog_dm
