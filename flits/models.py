"""Simple FRB signal model utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .params import FRBParams
from .scattering import scatter_broaden, tau_per_freq

# Import the canonical dispersion constant from the shared constants module.
# K_DM_MS gives delay in ms when DM is in pc/cm³ and freq is in MHz.
from .common.constants import K_DM_MS as K_DM




class FRBModel:
    """Generates dispersed Gaussian pulses, with optional scattering tail.

    The scattering tail is modeled as an exponential decay convolved with the
    Gaussian pulse in time: I(t) = Gauss(t) ⊗ exp(-t/τ) H(t), where H(t) is
    the Heaviside step function.

    Scattering modes:
    - Disabled: tau_1ghz = 0 (default).
    - Frequency-independent: tau_1ghz > 0, tau_alpha = 0.
    - Frequency-dependent: tau_1ghz > 0, tau_alpha > 0 → τ(ν) = τ_1GHz × (1 GHz / ν)^α.
    """

    def __init__(self, params: FRBParams):
        self.params = params

    def _generate_dispersed_pulse(
        self, t: NDArray[np.floating], delays: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Generate the dispersed Gaussian pulse profiles."""
        result = []
        for delay in delays:
            shifted = t - (self.params.t0 + delay)
            gauss = self.params.amplitude * np.exp(
                -0.5 * (shifted / self.params.width) ** 2
            )
            result.append(gauss)
        return np.vstack(result)

    def _apply_scattering(
        self,
        dynspec: NDArray[np.floating],
        t: NDArray[np.floating],
        freqs: NDArray[np.floating],
        tau_1ghz: float,
        alpha: float,
        ref_freq_mhz: float,
    ) -> NDArray[np.floating]:
        """Apply scattering broadening to the dynamic spectrum."""
        if tau_1ghz <= 0.0:
            return dynspec

        if alpha > 0.0:
            # Frequency-dependent: compute τ per frequency
            tau_array: float | NDArray[np.floating] = tau_per_freq(
                tau_1ghz, freqs, alpha, ref_freq_mhz
            )
        else:
            # Frequency-independent: uniform τ
            tau_array = tau_1ghz

        return scatter_broaden(dynspec, t, tau_array, causal=True)

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

        # Use override or fallback to params
        tau_1ghz = (
            tau_1ghz_override if tau_1ghz_override is not None else self.params.tau_1ghz
        )
        alpha = (
            tau_alpha_override
            if tau_alpha_override is not None
            else self.params.tau_alpha
        )

        # Compute per-frequency dispersion delays
        delays = K_DM * self.params.dm / freqs**2  # ms

        # Build dispersed Gaussian profile
        dynspec = self._generate_dispersed_pulse(t, delays)

        # Apply scattering broadening if enabled
        return self._apply_scattering(dynspec, t, freqs, tau_1ghz, alpha, ref_freq_mhz)


__all__ = ["FRBModel", "K_DM"]
