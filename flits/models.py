"""Simple FRB signal model utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .params import FRBParams
from .common.constants import K_DM_MS as K_DM, DM_DELAY_MS
from scattering.scat_analysis.burstfit import FRBModel as CoreModel, FRBParams as CoreParams

class FRBModel:
    """Generates dispersed Gaussian pulses, with optional scattering tail.
    
    This implementation wraps scattering.scat_analysis.burstfit.FRBModel to provide
    a unified physics kernel across the codebase.
    """

    def __init__(self, params: FRBParams):
        self.params = params

    def simulate(
        self,
        t: NDArray[np.floating],
        freqs: NDArray[np.floating],
        *,
        tau_1ghz_override: float | None = None,
        tau_alpha_override: float | None = None,
        ref_freq_mhz: float = 1000.0,
    ) -> NDArray[np.floating]:
        """Return model intensity for times ``t`` and frequencies ``freqs``.

        Parameters
        ----------
        t : ndarray
            Time axis in milliseconds.
        freqs : ndarray
            Frequencies in MHz.
        tau_1ghz_override : float or None
            Override scattering timescale at 1 GHz (ms). If None, uses params.tau_1ghz.
        tau_alpha_override : float or None
            Override frequency scaling exponent. If None, uses params.tau_alpha.
        ref_freq_mhz : float
            Reference frequency for scaling (default 1000 MHz = 1 GHz).

        Returns
        -------
        ndarray
            Dynamic spectrum with shape (len(freqs), len(t)).
        """
        t = np.asarray(t, dtype=np.float64)
        freqs = np.asarray(freqs, dtype=np.float64)
        
        # Convert MHz to GHz for CoreModel
        freqs_ghz = freqs / 1000.0
        
        # Infer channel width for smearing calculation
        if len(freqs) > 1:
            df_mhz = abs(freqs[1] - freqs[0])
        else:
            df_mhz = 1.0 # Default if single channel

        # Adjust t0 to match CoreModel's dispersion definition
        # CoreModel delay is relative to the highest frequency in the band
        # Legacy FRBModel delay was absolute (relative to infinite frequency)
        # t0_core = t0_legacy + Delay(f_max)
        f_max_ghz = np.max(freqs_ghz)
        delay_at_max = DM_DELAY_MS * self.params.dm / (f_max_ghz**2)
        
        # Map parameters
        tau_1ghz = tau_1ghz_override if tau_1ghz_override is not None else self.params.tau_1ghz
        alpha = tau_alpha_override if tau_alpha_override is not None else self.params.tau_alpha
        
        core_p = CoreParams(
            c0=self.params.amplitude,
            t0=self.params.t0 + delay_at_max,
            gamma=0.0, # Legacy model assumes flat spectrum or amplitude per-channel handled externally
            zeta=self.params.width,
            tau_1ghz=tau_1ghz,
            alpha=alpha,
            delta_dm=self.params.dm # We treat full DM as delta from 0 for dispersion calculation
        )
        
        # Instantiate CoreModel
        # We set dm_init=0 so that _dispersion_delay uses full delta_dm
        # We could set dm_init=params.dm and delta_dm=0, but _dispersion_delay only uses delta_dm
        # However, _smearing_sigma uses dm_init.
        # So we set dm_init = params.dm to get correct smearing,
        # AND delta_dm = params.dm to get correct delay?
        # No, _dispersion_delay depends ONLY on delta_dm.
        # So delta_dm must be the full DM if we want full dispersion.
        model = CoreModel(
            time=t,
            freq=freqs_ghz,
            dm_init=self.params.dm, # For smearing
            df_MHz=df_mhz
        )
        
        # Generate spectrum
        return model(core_p, model_key="M3")

__all__ = ["FRBModel", "K_DM"]
