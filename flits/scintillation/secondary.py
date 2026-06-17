"""Secondary (conjugate) spectrum of a dynamic spectrum.

Migrated from the legacy `scint_pipeline_funcs` module. The secondary spectrum
is the squared magnitude of the 2D FFT of the dynamic spectrum, used to search
for scintillation arcs in the Doppler-delay plane.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.fft import fftfreq, fftshift

__all__ = ["calculate_secondary_spectrum"]


def calculate_secondary_spectrum(
    dyn_spec: NDArray[np.floating],
    time_res_s: float,
    freq_res_hz: float,
    subtract_mean: bool = True,
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Compute the secondary spectrum and its Doppler/delay axes.

    Parameters
    ----------
    dyn_spec : ndarray
        2D dynamic spectrum with shape ``(ntime, nfreq)``.
    time_res_s : float
        Time resolution (sampling interval) in seconds.
    freq_res_hz : float
        Frequency resolution (channel width) in Hz.
    subtract_mean : bool, optional
        Subtract the per-channel mean before the FFT. Default True.

    Returns
    -------
    secondary_spec : ndarray
        ``|fftshift(fft2(dyn_spec))|**2``, with zero Doppler/delay at the centre.
    fd_axis : ndarray
        Conjugate (Doppler) frequency axis in Hz.
    tau_axis : ndarray
        Conjugate (delay) time axis in seconds.

    Raises
    ------
    ValueError
        If ``dyn_spec`` is not 2D.

    Notes
    -----
    Non-finite samples are replaced by the array median before transforming.
    """
    dyn_spec = np.asarray(dyn_spec, dtype=np.float64)
    if dyn_spec.ndim != 2:
        raise ValueError("Input dynamic spectrum must be 2D (time, freq).")

    nt, nf = dyn_spec.shape
    proc_spec = dyn_spec.copy()

    if np.any(~np.isfinite(proc_spec)):
        print(
            "Warning: Non-finite values found in dynamic spectrum. "
            "Replacing with median."
        )
        proc_spec[~np.isfinite(proc_spec)] = np.nanmedian(proc_spec)

    if subtract_mean:
        proc_spec = proc_spec - np.mean(proc_spec, axis=0, keepdims=True)

    fft_res = np.fft.fft2(proc_spec)
    secondary_spec = np.abs(fftshift(fft_res)) ** 2

    tau_axis = fftshift(fftfreq(nf, d=freq_res_hz))  # delay (s)
    fd_axis = fftshift(fftfreq(nt, d=time_res_s))  # Doppler (Hz)

    return secondary_spec, fd_axis, tau_axis
