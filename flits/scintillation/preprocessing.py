"""Dynamic-spectrum preprocessing for scintillation analysis.

Block-averaging ("scrunching") and FFT-based upchannelization of FRB dynamic
spectra, migrated from the legacy `scint_pipeline_funcs` module. Time-axis
block averaging is delegated to :func:`flits.common.utils.downsample_time` so
FLITS keeps a single canonical down-sampler.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from flits.common.utils import downsample_time

__all__ = ["scrunch", "upchannelize"]


def scrunch(
    data: NDArray[np.floating],
    t_scrunch: int = 1,
    f_scrunch: int = 1,
) -> NDArray[np.floating]:
    """Block-average a dynamic spectrum in time and frequency.

    Parameters
    ----------
    data : ndarray
        Dynamic spectrum with shape ``(ntime, nfreq)``.
    t_scrunch : int, optional
        Integer block size along the time axis. Default 1 (no averaging).
    f_scrunch : int, optional
        Integer block size along the frequency axis. Default 1 (no averaging).

    Returns
    -------
    ndarray
        Scrunched dynamic spectrum with shape
        ``(ntime // t_scrunch, nfreq // f_scrunch)``.

    Raises
    ------
    ValueError
        If ``data`` is not 2D, or if either scrunch factor is ``< 1`` or does
        not evenly divide its dimension.

    Notes
    -----
    The time-axis averaging is delegated to
    :func:`flits.common.utils.downsample_time` (which operates on a
    ``(nfreq, ntime)`` array), keeping one down-sampling implementation across
    FLITS. The result is identical to a single
    ``reshape(nt // t, t, nf // f, f).mean((1, 3))``.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"data must be 2D (ntime, nfreq); got shape {data.shape}.")
    nt, nf = data.shape
    if t_scrunch < 1 or f_scrunch < 1:
        raise ValueError("Scrunch factors must be >= 1.")
    if nt % t_scrunch != 0:
        raise ValueError(
            f"Time dimension ({nt}) not divisible by t_scrunch ({t_scrunch})."
        )
    if nf % f_scrunch != 0:
        raise ValueError(
            f"Frequency dimension ({nf}) not divisible by f_scrunch ({f_scrunch})."
        )

    # Average frequency blocks first: (nt, nf) -> (nt, nf // f_scrunch).
    if f_scrunch > 1:
        data = data.reshape(nt, nf // f_scrunch, f_scrunch).mean(axis=2)

    # Reuse the canonical time down-sampler (it expects (nfreq, ntime), so we
    # transpose in and back out).
    if t_scrunch > 1:
        data = downsample_time(data.T, t_scrunch).T

    return np.ascontiguousarray(data)


def upchannelize(
    intensity: NDArray[np.floating],
    fftsize: int = 32,
    downfreq: int = 2,
    downtime: int = 1,
) -> Tuple[NDArray[np.floating], int]:
    """Upchannelize an intensity dynamic spectrum via a per-block FFT.

    Splits the time axis into blocks of length ``fftsize``, takes the power
    spectrum of each block, optionally averages over ``downtime`` consecutive
    blocks, and averages the FFT bins down by ``downfreq``. Each original
    channel therefore becomes ``fftsize // downfreq`` upchannelized channels.

    Parameters
    ----------
    intensity : ndarray
        Dynamic spectrum intensity (Stokes I) with shape ``(nfreq, ntime)``.
    fftsize : int, optional
        FFT length along the time axis. Default 32.
    downfreq : int, optional
        Down-sampling factor applied to the FFT bins. Must divide ``fftsize``.
        Default 2.
    downtime : int, optional
        Number of consecutive FFT blocks to average. Default 1.

    Returns
    -------
    upchann_spec : ndarray
        Upchannelized power spectrum with shape ``(nfreq * upchan_factor,
        nblock)``, channel-major in the order
        ``[chan0_up0, chan0_up1, ..., chan1_up0, ...]``.
    upchan_factor : int
        Effective upchannelization factor, ``fftsize // downfreq``.

    Raises
    ------
    ValueError
        If ``intensity`` is not 2D, if ``ntime`` is not divisible by
        ``fftsize * downtime``, or if ``fftsize`` is not divisible by
        ``downfreq``.

    Notes
    -----
    Power is ``|FFT|**2`` of the (real) intensity blocks. This mirrors the
    legacy behaviour, which used a full (complex) FFT on intensity data; the
    interpretation of the output frequency axis therefore mixes the original
    channelization with the FFT bins and should be handled downstream.
    """
    intensity = np.asarray(intensity, dtype=np.float64)
    if intensity.ndim != 2:
        raise ValueError("Input intensity array must be 2D (frequency, time).")

    nchan, nsamp = intensity.shape
    if nsamp % (fftsize * downtime) != 0:
        raise ValueError(
            f"Time samples ({nsamp}) not divisible by fftsize*downtime "
            f"({fftsize * downtime})."
        )
    if fftsize % downfreq != 0:
        raise ValueError(
            f"fftsize ({fftsize}) must be divisible by downfreq ({downfreq})."
        )

    upchan_factor = fftsize // downfreq
    nchan_up = nchan * upchan_factor
    nblock = nsamp // (fftsize * downtime)

    # (nchan, nblock, downtime, fftsize) blocks for the time-axis FFT.
    wfall_block = intensity.reshape(nchan, nblock, downtime, fftsize)
    upchann_spec = np.zeros((nchan_up, nblock), dtype=np.float64)

    for i in range(nblock):
        block_fft = np.fft.fft(wfall_block[:, i, :, :], axis=-1)
        block_power = np.abs(block_fft) ** 2  # (nchan, downtime, fftsize)

        if downtime > 1:
            block_power = block_power.mean(axis=1)  # (nchan, fftsize)
        else:
            block_power = block_power[:, 0, :]

        # Average the fftsize bins down by downfreq -> upchan_factor bins.
        power_reshaped = block_power.reshape(nchan, upchan_factor, downfreq)
        upchann_spec[:, i] = power_reshaped.mean(axis=2).ravel()

    return upchann_spec, upchan_factor
