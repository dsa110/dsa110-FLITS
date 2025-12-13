"""Simple FRB signal model utilities."""

from __future__ import annotations

import numpy as np

from .params import FRBParams

# Dispersion constant (ms MHz^2 pc^-1 cm^3)
K_DM = 4.148808e3


class FRBModel:
    """Generates dispersed Gaussian pulses, with optional scattering tail.

    The scattering tail is modeled as an exponential decay convolved with the
    Gaussian pulse in time: I(t) = Gauss(t) * exp(-t/τ) H(t), where H(t) is
    the Heaviside step function. Set ``tau_sc_ms`` to a positive value to
    enable scatter broadening.
    """

    def __init__(self, params: FRBParams):
        self.params = params

    def simulate(
        self,
        t: np.ndarray,
        freqs: np.ndarray,
        *,
        tau_sc_ms: float | None = None,
    ) -> np.ndarray:
        """Return model intensity for times ``t`` and frequencies ``freqs``.

        Parameters
        ----------
        t : array-like
            Time axis in milliseconds.
        freqs : array-like
            Frequencies in MHz.
        tau_sc_ms : float or None
            Scattering timescale τ in milliseconds; if provided and > 0,
            applies an exponential tail via convolution.

        Returns
        -------
        np.ndarray
            Array with shape (len(freqs), len(t)) representing a dynamic
            spectrum where each row corresponds to one frequency channel.
        """
        t = np.asarray(t)
        freqs = np.asarray(freqs)

        # Base Gaussian pulse (per-frequency arrival time)
        delays = K_DM * self.params.dm / freqs**2  # ms
        dt = float(t[1] - t[0]) if len(t) > 1 else 1.0

        # Precompute exponential tail kernel if enabled
        use_tail = tau_sc_ms is not None and tau_sc_ms > 0.0
        if use_tail:
            # Kernel defined over same time grid, causal exponential
            kernel = np.exp(-np.maximum(t - t.min(), 0.0) / tau_sc_ms)
            # Normalize kernel to preserve area
            kernel /= kernel.sum() if kernel.sum() > 0 else 1.0

        result = []
        for delay in delays:
            shifted = t - (self.params.t0 + delay)
            gauss = self.params.amplitude * np.exp(
                -0.5 * (shifted / self.params.width) ** 2
            )
            if use_tail:
                # Discrete convolution; scale by dt to preserve units
                broadened = np.convolve(gauss, kernel, mode="same") * dt
                result.append(broadened)
            else:
                result.append(gauss)
        return np.vstack(result)


__all__ = ["FRBModel", "K_DM"]
