"""Simple FRB signal model utilities."""
from __future__ import annotations

import numpy as np

from .params import FRBParams

# Dispersion constant (ms MHz^2 pc^-1 cm^3)
K_DM = 4.148808e3


class FRBModel:
    """Generates dispersed Gaussian pulses."""

    def __init__(self, params: FRBParams):
        self.params = params

    def simulate(self, t: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Return model intensity for times ``t`` and frequencies ``freqs``.

        Parameters
        ----------
        t : array-like
            Time axis in milliseconds.
        freqs : array-like
            Frequencies in MHz.

        Returns
        -------
        np.ndarray
            Array with shape (len(freqs), len(t)) representing a dynamic
            spectrum where each row corresponds to one frequency channel.
        """
        t = np.asarray(t)
        freqs = np.asarray(freqs)

        delays = K_DM * self.params.dm / freqs**2  # ms
        result = []
        for delay in delays:
            shifted = t - (self.params.t0 + delay)
            pulse = self.params.amplitude * np.exp(
                -0.5 * (shifted / self.params.width) ** 2
            )
            result.append(pulse)
        return np.vstack(result)


__all__ = ["FRBModel", "K_DM"]
