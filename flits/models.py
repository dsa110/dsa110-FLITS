"""Simple FRB signal model utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .params import FRBParams
from .scattering import scatter_broaden, tau_per_freq

# Dispersion constant (ms MHz^2 pc^-1 cm^3)
K_DM = 4.148808e3


class FRBModel:
    """Generates dispersed Gaussian pulses, with optional scattering tail.

    The scattering tail is modeled as an exponential decay convolved with the
    Gaussian pulse in time: I(t) = Gauss(t) * exp(-t/τ) H(t), where H(t) is
    the Heaviside step function.

    Scattering can be:
    - Disabled: tau_ms = 0 (default).
    - Frequency-independent: tau_ms > 0 and tau_alpha ignored.
    - Frequency-dependent: tau_ms > 0 and tau_alpha > 0, enabling τ(ν).

    Per-frequency timescale is computed as:
        τ(ν) = τ_ref * (ν_ref / ν)^α
    where ν_ref = 1 GHz by default.
    """

    def __init__(self, params: FRBParams):
        self.params = params

    def simulate(
        self,
        t: NDArray[np.floating],
        freqs: NDArray[np.floating],
        *,
        tau_sc_ms: float | None = None,
        tau_alpha: float | None = None,
        ref_freq_mhz: float = 1000.0,
    ) -> NDArray[np.floating]:
        """Return model intensity for times ``t`` and frequencies ``freqs``.

        Parameters
        ----------
        t : ndarray
            Time axis in milliseconds.
        freqs : ndarray
            Frequencies in MHz.
        tau_sc_ms : float or None
            Scattering timescale τ (at 1 GHz) in milliseconds; if None, uses params.tau_ms.
            If <= 0, scattering is disabled.
        tau_alpha : float or None
            Frequency scaling exponent; if None, uses params.tau_alpha.
            If > 0, enables per-frequency τ(ν) scaling.
        ref_freq_mhz : float
            Reference frequency for tau_alpha scaling (default 1000 MHz = 1 GHz).

        Returns
        -------
        ndarray
            Array with shape (len(freqs), len(t)) representing a dynamic
            spectrum where each row corresponds to one frequency channel.
        """
        t = np.asarray(t, dtype=np.float64)
        freqs = np.asarray(freqs, dtype=np.float64)

        # Use provided or fallback to params
        tau_ms = tau_sc_ms if tau_sc_ms is not None else self.params.tau_ms
        alpha = tau_alpha if tau_alpha is not None else self.params.tau_alpha

        # Base Gaussian pulse (per-frequency arrival time)
        delays = K_DM * self.params.dm / freqs**2  # ms

        # Build dispersed Gaussian profile
        result = []
        for delay in freqs:
            shifted = t - (self.params.t0 + delay)
            gauss = self.params.amplitude * np.exp(
                -0.5 * (shifted / self.params.width) ** 2
            )
            result.append(gauss)
        dynspec = np.vstack(result)

        # Apply scattering broadening if enabled
        if tau_ms > 0.0:
            if alpha > 0.0:
                # Frequency-dependent: compute τ per frequency
                tau_array = tau_per_freq(tau_ms, freqs, alpha, ref_freq_mhz)
            else:
                # Frequency-independent: uniform τ
                tau_array = tau_ms
            dynspec = scatter_broaden(dynspec, t, tau_array, causal=True)

        return dynspec


__all__ = ["FRBModel", "K_DM"]
