"""Simple FRB signal model utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .params import FRBParams
from .scattering import scatter_broaden, tau_per_freq

# Dispersion constant: delay(ms) = K_DM * DM(pc/cm³) / freq(MHz)²
# Derived from: dt = 4.148808 ms × DM / f_GHz², converting GHz → MHz gives factor of 1e6
K_DM = 4.148808e6


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
        tau_1ghz = tau_1ghz_override if tau_1ghz_override is not None else self.params.tau_1ghz
        alpha = tau_alpha_override if tau_alpha_override is not None else self.params.tau_alpha

        # Compute per-frequency dispersion delays
        delays = K_DM * self.params.dm / freqs**2  # ms

        # Build dispersed Gaussian profile
        result = []
        for delay in delays:  # FIX: was incorrectly iterating over `freqs`
            shifted = t - (self.params.t0 + delay)
            gauss = self.params.amplitude * np.exp(
                -0.5 * (shifted / self.params.width) ** 2
            )
            result.append(gauss)
        dynspec = np.vstack(result)

        # Apply scattering broadening if enabled
        if tau_1ghz > 0.0:
            if alpha > 0.0:
                # Frequency-dependent: compute τ per frequency
                tau_array: float | NDArray[np.floating] = tau_per_freq(
                    tau_1ghz, freqs, alpha, ref_freq_mhz
                )
            else:
                # Frequency-independent: uniform τ
                tau_array = tau_1ghz
            dynspec = scatter_broaden(dynspec, t, tau_array, causal=True)

        return dynspec


__all__ = ["FRBModel", "K_DM"]
