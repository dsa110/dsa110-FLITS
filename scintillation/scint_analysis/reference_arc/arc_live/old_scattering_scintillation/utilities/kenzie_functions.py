# -*- coding: utf-8 -*-
"""
Collection of utility functions for FRB analysis, focusing on data loading,
preprocessing (dedispersion, RFI flagging, derippling), ACF calculation,
and interaction with CHIME databases and fitburst models.

Originally developed by Kenzie Nimmo (hence the filename).
"""

import sys
import os
import json
from copy import deepcopy

import numpy as np
from lmfit import Model, Minimizer # Note: minimize, Parameters, fit_report not used directly here
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
from scipy.interpolate import make_lsq_spline, interp2d
from scipy import signal

# Imports from CHIME-specific or related packages
try:
    from baseband_analysis.core.signal import get_main_peak_lim, tiedbeam_baseband_to_power
    from baseband_analysis.core.bbdata import BBData
    from baseband_analysis.analysis.snr import get_snr, get_profile
    from baseband_analysis.core.sampling import scrunch
    from baseband_analysis.core.dedispersion import coherent_dedisp, incoherent_dedisp
    from baseband_analysis.analysis.polarization import get_burst_envelope
    import chime_frb_api
    import chime_frb_constants as const
    import fitburst as fb
    # from pfb_tools import DeconvolvePFB # Commented out in original - PFB flattening used elsewhere?
except ImportError as e:
    print(f"Warning: Could not import one or more required libraries ({e}). "
          "Some functions may not work.")
    # Define dummy classes/functions if needed for script to load without errors
    class BBData:
        @staticmethod
        def from_file(filepath):
            raise NotImplementedError("BBData not available.")
        def __setitem__(self, key, value):
            pass
        def __getitem__(self, key):
            raise NotImplementedError("BBData not available.")
        def keys(self):
            return []
        @property
        def index_map(self):
            # Provide minimal structure if accessed directly
            return {'freq': {'id': np.array([]), 'centre': np.array([])}}

# --- CHIME FRB Master API Setup ---
# Attempt to connect and authorize with the CHIME FRB Master database
try:
    master = chime_frb_api.frb_master.FRBMaster(base_url = "https://frb.chimenet.ca/frb-master")
    master.API.authorize()
    auth = {"Authorization": master.API.access_token}
    print("Successfully authorized CHIME FRB Master API.")
except Exception as e:
    print(f"Warning: Could not connect or authorize CHIME FRB Master API: {e}")
    master = None
    auth = None

# --- Constants ---
# Copied from scint_functions.py for consistency if used here
FREQ_TOP_MHZ = 800.1953125
FREQ_BOTTOM_MHZ = 400.1953125
TOTAL_CHANNELS = 1024

# --- Data Loading ---

def get_data(event):
    """
    Constructs the expected file path for CHIME baseband data based on event ID
    and date information extracted from the event metadata.

    Note: This function relies on a specific, hardcoded directory structure
          at `/arc/projects/chime_frb/data/chime/baseband/processed/`.
          It might need modification if the data location changes.

    Parameters
    ----------
    event : dict
        Event dictionary, typically obtained from the CHIME FRB Master API,
        containing 'id' and 'measured_parameters'.

    Returns
    -------
    str
        The constructed file path to the singlebeam HDF5 file.

    Raises
    ------
    ValueError
        If the event date cannot be found or is invalid within the
        'measured_parameters' list under the 'realtime' pipeline.
    KeyError
        If the 'id' key is missing from the event dictionary.
    """
    event_date = None
    event_id = event['id'] # Get event ID early to raise KeyError if missing

    # Search for the event date from the 'realtime' pipeline parameters
    for par in event.get("measured_parameters", []):
        pipeline_info = par.get("pipeline", {})
        if pipeline_info.get("name") == "realtime":
            datetime_str = par.get("datetime", "")
            if datetime_str:
                try:
                    event_date_parts = datetime_str.split(" ")[0].split("-")
                    if len(event_date_parts) == 3:
                        event_date = event_date_parts
                        break # Found the date, exit loop
                except Exception:
                    pass # Ignore errors during parsing, continue searching

    if not event_date:
        raise ValueError(f"Could not find valid event date for event {event_id} "
                         "in 'realtime' measured parameters.")

    # Construct the path using the hardcoded structure
    # Original path included "Run_pre2025", this is removed in the cleaned version
    # based on the commented-out line in the original code. Adjust if needed.
    data_path = (
        f"/arc/projects/chime_frb/data/chime/baseband/processed/"
        f"{event_date[0]}/{event_date[1]}/{event_date[2]}/astro_"
        f"{event_id}/singlebeam_{event_id}.h5"
    )

    # Check if the file actually exists (optional but recommended)
    # if not os.path.exists(data_path):
    #     print(f"Warning: Constructed data path does not exist: {data_path}")

    return data_path

# --- Data Preprocessing ---

def deripple(ds, offpulse):
    """
    Removes frequency-dependent baseline ripple and normalizes by standard deviation.

    Subtracts the mean and divides by the standard deviation of the off-pulse
    data for each frequency channel and polarization. Modifies `offpulse` in place
    by subtracting its mean.

    Parameters
    ----------
    ds : np.ndarray
        Dynamic spectrum data array. Shape [freq, pol, time] or [freq, time].
    offpulse : np.ndarray
        Off-pulse data array used for calculating statistics. Must have the
        same frequency and polarization dimensions as `ds`. Modified in place.

    Returns
    -------
    ds_final : np.ndarray
        Derippled and normalized dynamic spectrum, same shape as `ds`.
    """
    ds_final = np.zeros_like(ds)

    if ds.ndim == 3: # Shape [freq, pol, time]
        if ds.shape[:2] != offpulse.shape[:2]:
             raise ValueError("Shape mismatch: ds and offpulse must match in freq and pol dimensions.")
        for chan in range(offpulse.shape[0]):
            for pol in range(offpulse.shape[1]):
                off_std = np.std(offpulse[chan, pol, :])
                if off_std != 0:
                    off_mean = np.mean(offpulse[chan, pol, :])
                    # Modify offpulse in place
                    offpulse[chan, pol, :] -= off_mean
                    # Apply correction to ds
                    ds_final[chan, pol, :] = (ds[chan, pol, :] - off_mean) / off_std
                else:
                    # Handle zero standard deviation case (e.g., constant offpulse)
                    ds_final[chan, pol, :] = ds[chan, pol, :] - np.mean(offpulse[chan, pol, :])

    elif ds.ndim == 2: # Shape [freq, time]
        if ds.shape[0] != offpulse.shape[0]:
             raise ValueError("Shape mismatch: ds and offpulse must match in freq dimension.")
        for chan in range(offpulse.shape[0]):
            off_std = np.std(offpulse[chan, :])
            if off_std != 0:
                off_mean = np.mean(offpulse[chan, :])
                # Modify offpulse in place
                offpulse[chan, :] -= off_mean
                # Apply correction to ds
                ds_final[chan, :] = (ds[chan, :] - off_mean) / off_std
            else:
                ds_final[chan, :] = ds[chan, :] - np.mean(offpulse[chan, :])
    else:
        raise ValueError("Input ds must be 2D or 3D.")

    return ds_final

def fill_missing_chans(ds, bbdata):
    """
    Fills gaps in a frequency channel array based on BBData index map.

    Creates a new array covering the full standard CHIME band (1024 channels)
    and places the input data `ds` into the appropriate channels according
    to the frequency IDs in `bbdata`. Missing channels are left as masked.
    Also recalculates a full, evenly spaced frequency array.

    Parameters
    ----------
    ds : np.ndarray or np.ma.MaskedArray
        Input data array with potentially missing frequency channels.
        Expected shape [freq_subset, pol, time] or [freq_subset, time].
    bbdata : BBData object
        Baseband data object containing the index map for frequencies.

    Returns
    -------
    data_masked : np.ma.MaskedArray
        Data array expanded to 1024 frequency channels, with missing
        channels masked. Shape [1024, pol, time] or [1024, time].
    new_freqs : np.ndarray
        Linearly spaced frequency array (MHz) covering the full band (1024 channels).
    new_freq_id : np.ndarray
        Array of channel IDs from 0 to 1023.
    """
    original_shape = ds.shape
    nchan_in = original_shape[0]
    other_dims = original_shape[1:] # e.g., (pol, time) or (time,)

    # Determine output shape
    output_shape = (TOTAL_CHANNELS,) + other_dims

    # Initialize output array with zeros (will be masked later)
    # Use complex type if input is complex, otherwise float
    dtype = ds.dtype if np.iscomplexobj(ds) else np.float64
    new_data = np.zeros(output_shape, dtype=dtype)

    # Get frequency info from bbdata
    try:
        freq_map = bbdata.index_map["freq"]
        freq_id_in = freq_map["id"]
        freqs_in = freq_map["centre"]
    except (KeyError, AttributeError) as e:
        raise ValueError(f"Could not access frequency index map in bbdata: {e}")

    if len(freq_id_in) != nchan_in:
         raise ValueError(f"Mismatch between ds freq channels ({nchan_in}) and "
                          f"bbdata freq IDs ({len(freq_id_in)}).")

    # Place input data into the full 1024-channel array
    for i, chan_id in enumerate(freq_id_in):
        if 0 <= chan_id < TOTAL_CHANNELS:
            new_data[chan_id, ...] = ds[i, ...]
        else:
            print(f"Warning: Input channel ID {chan_id} is outside the valid range [0, {TOTAL_CHANNELS-1}). Skipping.")

    # Mask the channels that were not filled (still zero)
    data_masked = np.ma.masked_where(new_data == 0, new_data)

    # --- Recalculate Full Frequency Axis ---
    new_freq_id = np.arange(TOTAL_CHANNELS)

    # Estimate frequency resolution and range from input frequencies/IDs
    # This assumes the input channels are somewhat contiguous
    valid_indices = np.where(np.diff(freq_id_in) > 0)[0]
    if len(valid_indices) >= 1:
        idx0 = valid_indices[0]
        num_chan_diff = int(freq_id_in[idx0 + 1] - freq_id_in[idx0])
        freq_diff = np.abs(freqs_in[idx0 + 1] - freqs_in[idx0])
        f_res = freq_diff / float(num_chan_diff) if num_chan_diff > 0 else freq_diff
    elif len(freqs_in) > 1:
         f_res = np.abs(freqs_in[1] - freqs_in[0]) # Fallback
    else:
         f_res = (FREQ_TOP_MHZ - FREQ_BOTTOM_MHZ) / TOTAL_CHANNELS # Default

    # Estimate band edges based on the first/last input channel and resolution
    # This logic seems slightly complex and might be simplified if
    # FREQ_TOP_MHZ and FREQ_BOTTOM_MHZ are reliable band edges.
    # Using the predefined constants directly for simplicity:
    new_freqs = np.linspace(FREQ_BOTTOM_MHZ, FREQ_TOP_MHZ, TOTAL_CHANNELS)[::-1] # High to low freq

    # --- Original frequency calculation logic (kept for reference) ---
    # if freq_id_in[0] == 0:
    #     fmax_est = freqs_in[0]
    # else:
    #     # Estimate freq of channel 0 based on freq of first channel present
    #     fmax_est = freqs_in[0] + (f_res * freq_id_in[0]) # Assuming linear spacing upwards
    # if freq_id_in[-1] == TOTAL_CHANNELS - 1:
    #     fmin_est = freqs_in[-1]
    # else:
    #     # Estimate freq of last channel based on freq of last channel present
    #     fmin_est = freqs_in[-1] - (f_res * (TOTAL_CHANNELS - 1 - freq_id_in[-1]))
    # new_freqs_orig_logic = np.linspace(fmin_est, fmax_est, TOTAL_CHANNELS)
    # print(f"Recalculated freqs range: {new_freqs_orig_logic[-1]:.4f} - {new_freqs_orig_logic[0]:.4f} MHz")
    # print(f"Using standard range:   {new_freqs[-1]:.4f} - {new_freqs[0]:.4f} MHz")
    # --- End original logic ---

    return data_masked, new_freqs, new_freq_id


def upchannel(wfall, freq_id, fftsize=32, downfreq=2):
    """
    Upchannelizes baseband voltage data using an FFT-based method.

    Note: This function appears identical to the `upchannel` function in
          `scint_functions.py`. Consider consolidating to avoid redundancy.

    Parameters and Returns: See `scint_functions.upchannel` docstring.
    """
    # --- Implementation copied from scint_functions.py ---
    # (Assuming the version in scint_functions is the intended one)

    # Input validation
    if wfall.ndim != 3: raise ValueError("Input wfall must be 3D (freq, pol, time)")
    if freq_id.ndim != 1: raise ValueError("Input freq_id must be 1D")
    if wfall.shape[0] != len(freq_id): raise ValueError("Dimension mismatch: wfall.shape[0] != len(freq_id)")
    if fftsize <= 0 or not isinstance(fftsize, int): raise ValueError("fftsize must be positive integer")
    if downfreq <= 0 or not isinstance(downfreq, int): raise ValueError("downfreq must be positive integer")
    if fftsize % downfreq != 0: raise ValueError("fftsize must be divisible by downfreq")

    wfall_proc = np.swapaxes(wfall, 0, 1)
    wfall_proc = np.swapaxes(wfall_proc, 1, 2)
    npol, nsamp, nchan_in = wfall_proc.shape

    downtime = 1
    upchan = fftsize // downfreq
    nblock = nsamp // (fftsize * downtime)
    nchan_up = nchan_in * upchan

    f_upchan_bandtot = np.linspace(FREQ_TOP_MHZ, FREQ_BOTTOM_MHZ, upchan * TOTAL_CHANNELS)

    spec = np.zeros((npol, nblock, nchan_up), dtype=np.complex64)
    chan_id_upchan_map = np.zeros((nchan_in, upchan), dtype=int)

    for pol in range(npol):
        for bi in range(nblock):
            time_start = bi * fftsize * downtime
            time_end = time_start + fftsize
            for chidx_in in range(nchan_in):
                ts_segment = wfall_proc[pol, time_start:time_end, chidx_in].copy()
                ft = np.fft.fft(ts_segment) # Use numpy's fft
                ft_shifted = np.fft.fftshift(ft) # Use numpy's fftshift
                ft_downsampled = ft_shifted.reshape(upchan, downfreq).mean(axis=1)
                spec_chan_start = chidx_in * upchan
                spec_chan_end = spec_chan_start + upchan
                spec[pol, bi, spec_chan_start:spec_chan_end] = ft_downsampled
                if pol == 0 and bi == 0:
                    original_chan_id = freq_id[chidx_in]
                    chan_id_upchan_map[chidx_in, :] = np.arange(
                        upchan * original_chan_id, upchan * original_chan_id + upchan, 1
                    )

    chan_id_upchan_final = chan_id_upchan_map.ravel()
    valid_indices = chan_id_upchan_final < len(f_upchan_bandtot)
    chan_id_upchan_final = chan_id_upchan_final[valid_indices]
    f_upchan_final = f_upchan_bandtot[chan_id_upchan_final]

    if not np.all(valid_indices): print("Warning: Some upchannel IDs out of bounds.")

    return spec, f_upchan_final, chan_id_upchan_final


def upchannel_fast(wfall, freq_id, fftsize=32, downfreq=2):
    """
    Optimized version of the upchannelization function using vectorized operations.

    Performs the same logic as `upchannel` but aims for better performance
    by avoiding nested Python loops where possible.

    Parameters
    ----------
    wfall : np.ndarray
        Input complex voltage data array. Expected shape [freq, pol, time].
    freq_id : np.ndarray
        1D array containing the original frequency channel IDs.
    fftsize : int, optional
        Size of the FFT window. Default is 32.
    downfreq : int, optional
        Downsampling factor after FFT shift. Default is 2.

    Returns
    -------
    spec : np.ndarray
        Upchannelized complex voltage data. Shape [pol, nblock, nchan_up].
    f_upchan_final : np.ndarray
        1D array of central frequencies (MHz) for the upchannelized channels.
    chan_id_upchan_final : np.ndarray
        1D array of integer channel IDs for the upchannelized channels.
    """
    # Input validation (same as upchannel)
    if wfall.ndim != 3: raise ValueError("Input wfall must be 3D (freq, pol, time)")
    if freq_id.ndim != 1: raise ValueError("Input freq_id must be 1D")
    if wfall.shape[0] != len(freq_id): raise ValueError("Dimension mismatch: wfall.shape[0] != len(freq_id)")
    if fftsize <= 0 or not isinstance(fftsize, int): raise ValueError("fftsize must be positive integer")
    if downfreq <= 0 or not isinstance(downfreq, int): raise ValueError("downfreq must be positive integer")
    if fftsize % downfreq != 0: raise ValueError("fftsize must be divisible by downfreq")

    # Swap axes: [freq, pol, time] -> [pol, time, freq]
    wfall_proc = np.swapaxes(wfall, 0, 1)
    wfall_proc = np.swapaxes(wfall_proc, 1, 2)

    npol, nsamp, nchan_in = wfall_proc.shape

    # Parameters
    downtime = 1 # No time averaging
    upchan = fftsize // downfreq
    nblock = nsamp // (fftsize * downtime)
    nchan_up = nchan_in * upchan

    # --- Frequency and Channel ID Calculation (Vectorized) ---
    f_upchan_bandtot = np.linspace(FREQ_TOP_MHZ, FREQ_BOTTOM_MHZ, upchan * TOTAL_CHANNELS)
    # Create the mapping from input channel index to upchannel IDs directly
    # freq_id[:, None] -> [[id0], [id1], ...] ; shape (nchan_in, 1)
    # np.arange(upchan) -> [0, 1, ..., upchan-1] ; shape (upchan,)
    # Broadcasting: (freq_id[:, None] * upchan) + np.arange(upchan)
    # Result shape (nchan_in, upchan)
    chan_id_upchan_map = (freq_id[:, None] * upchan) + np.arange(upchan)
    chan_id_upchan_final = chan_id_upchan_map.ravel() # Flatten to 1D

    # Select corresponding frequencies
    valid_indices = chan_id_upchan_final < len(f_upchan_bandtot)
    chan_id_upchan_final = chan_id_upchan_final[valid_indices]
    f_upchan_final = f_upchan_bandtot[chan_id_upchan_final]
    if not np.all(valid_indices): print("Warning: Some upchannel IDs out of bounds.")


    # --- Upchannelization (Vectorized) ---
    spec = np.zeros((npol, nblock, nchan_up), dtype=np.complex64)

    # Truncate input data to full blocks
    valid_nsamp = nblock * fftsize * downtime
    wfall_trunc = wfall_proc[:, :valid_nsamp, :] # Shape [pol, valid_nsamp, nchan_in]

    for pol in range(npol):
        # Reshape for block processing: [valid_nsamp, nchan_in] -> [nblock, fftsize, nchan_in]
        pol_data_reshaped = wfall_trunc[pol].reshape(nblock, fftsize, nchan_in)

        # Perform FFT along the time axis (axis=1)
        fft_result = np.fft.fft(pol_data_reshaped, axis=1)

        # Shift frequencies
        fft_shifted = np.fft.fftshift(fft_result, axes=1)

        # Downsample in frequency (upchannel dimension)
        # Reshape to group downfreq bins: [nblock, fftsize, nchan_in] -> [nblock, upchan, downfreq, nchan_in]
        # Then take the mean over the downfreq axis (axis=2)
        downsampled = fft_shifted.reshape(nblock, upchan, downfreq, nchan_in).mean(axis=2)
        # Result shape: [nblock, upchan, nchan_in]

        # Reorganize output to match desired shape [nblock, nchan_up]
        # Transpose: [nblock, upchan, nchan_in] -> [nblock, nchan_in, upchan]
        transposed = downsampled.transpose(0, 2, 1)
        # Reshape: [nblock, nchan_in, upchan] -> [nblock, nchan_in * upchan]
        spec[pol] = transposed.reshape(nblock, nchan_up)

    return spec, f_upchan_final, chan_id_upchan_final


def fftsize16_functions(name='ziggy'):
    """
    Returns hardcoded PFB window models for fftsize=16.

    These appear to be precomputed models related to the Polyphase Filter Bank
    response for specific configurations ('ziggy', 'eve', 'richard').

    Parameters
    ----------
    name : str, optional
        Name of the model to return ('ziggy', 'eve', or 'richard').
        Default is 'ziggy'.

    Returns
    -------
    model : np.ndarray
        1D array of the PFB model tiled across 1024 channels.
        Length is 16 * 1024 = 16384.
    """
    model_single = None
    if name == 'ziggy':
        model_single = np.array([
            0.63338375, 0.71153545, 0.84496403, 0.99040335, 1.11110604,
            1.19043183, 1.23157179, 1.24731326, 1.24810541, 1.23464894,
            1.19744003, 1.12338078, 1.0074048 , 0.86338407, 0.72602457,
            0.63890308
        ])
    elif name == 'eve':
        model_single = np.array([
            0.52225748, 0.58330915, 0.6868705, 0.80121821, 0.89386546,
            0.95477358, 0.98662733, 0.99942558, 0.99988676, 0.98905127,
            0.95874124, 0.90094667, 0.81113021, 0.6999944, 0.59367968,
            0.52614263
        ])
    elif name == 'richard':
        # Assumes DeconvolvePFB is available and Q=16 corresponds to fftsize=16
        try:
            from pfb_tools import DeconvolvePFB
            # Summing weights across time taps? Check DeconvolvePFB documentation.
            model_single = DeconvolvePFB(Q=16).Wt2.sum(axis=1)
            # Roll to potentially align the peak or shape
            model_single = np.roll(model_single, 8)
        except ImportError:
             raise ImportError("Cannot generate 'richard' model: pfb_tools.DeconvolvePFB not found.")
        except AttributeError:
             raise AttributeError("Cannot generate 'richard' model: DeconvolvePFB structure might have changed.")

    else:
        raise ValueError(f"Unknown model name: {name}. Choose 'ziggy', 'eve', or 'richard'.")

    # Tile the single model across the standard 1024 channels
    model_full = np.tile(model_single, TOTAL_CHANNELS)
    return model_full


# --- ACF Calculation and Fitting ---

def shift(v, i, nchan):
    """
    Helper function to circularly shift a 1D array `v` by `i` positions,
    padding with zeros for ACF calculation.

    Note: Identical to the `shift` function in `scint_functions.py`.

    Parameters and Returns: See `scint_functions.shift` docstring.
    """
    # --- Implementation copied from scint_functions.py ---
    n = len(v)
    r = np.zeros(3 * n)
    start_index = int(i + nchan - 1)
    end_index = start_index + n
    v_start, v_end = 0, n
    if start_index < 0: v_start = -start_index; start_index = 0
    if end_index > 3 * n: v_end = n - (end_index - 3 * n); end_index = 3 * n
    if start_index < end_index and v_start < v_end:
        r[start_index:end_index] = v[v_start:v_end]
    return r

def autocorr(spec, v=None, zerolag=False, maxlag=None, offspec_mean=None, freq=None):
    """
    Calculates the Auto-Correlation Function (ACF) of a 1D spectrum.

    Handles masking. Normalization depends on `offspec_mean`. Includes an
    unused `freq` parameter.

    Note: Very similar to `autocorr` in `scint_functions.py`, but includes
          an unused `freq` parameter and slightly different mean subtraction logic.
          Consider consolidating.

    Parameters
    ----------
    spec : np.ndarray or np.ma.MaskedArray
        1D input spectrum.
    v : np.ndarray, optional
        Mask array (1 for valid, 0 for masked).
    zerolag : bool, optional
        Include zero-lag if True. Default False.
    maxlag : int, optional
        Maximum lag in channels. Default None (all lags).
    offspec_mean : float, optional
        Mean of off-burst spectrum for normalization. Default None.
    freq : np.ndarray, optional
        Frequency array (MHz). Currently unused in the function logic,
        except for a commented-out alternative mean calculation.

    Returns
    -------
    ACF : np.ndarray
        The calculated auto-correlation function for positive lags.
    """
    nchan = len(spec)

    # Handle input spectrum and mask
    if isinstance(spec, np.ma.MaskedArray):
        mask = ~spec.mask
        x = spec.data
    else:
        x = np.copy(spec)
        mask = v.astype(bool) if v is not None else np.ones(nchan, dtype=bool)

    x[~mask] = np.nan # Use NaNs for masked values

    # Calculate mean of valid points
    x_mean_valid = np.nanmean(x)
    if np.isnan(x_mean_valid):
        print("Warning: Mean is NaN.")
        out_len = maxlag if maxlag is not None else (nchan if zerolag else nchan - 1)
        return np.zeros(out_len) * np.nan

    # --- Alternative Mean Calculation (Commented out in original) ---
    # if freq is not None:
    #     # Example power-law model for mean based on frequency
    #     xmean_model = 5417.46963982 * freq[mask]**-1.5
    #     # xmean_model = 1.03055693e+09*freq[mask]**-3.451
    #     print('Using frequency-dependent mean model (method 2)')
    #     # How to incorporate this model mean isn't fully clear from original.
    #     # Does it replace x_mean_valid? Or used differently?
    #     # Sticking to simple mean for now.
    #     pass
    # --- End Alternative ---

    # Determine normalization denominator
    if offspec_mean is None:
        denom = x_mean_valid**2
    else:
        denom = (x_mean_valid - offspec_mean)**2

    if denom == 0:
        print("Warning: Normalization denominator is zero.")
        out_len = maxlag if maxlag is not None else (nchan if zerolag else nchan - 1)
        return np.zeros(out_len) * np.nan

    # Subtract mean from valid data points
    x_meansub = np.copy(x)
    x_meansub[mask] -= x_mean_valid

    # ACF Calculation setup
    if maxlag is None:
        num_lags_calc = nchan
    else:
        maxlag = min(int(maxlag), nchan)
        num_lags_calc = maxlag

    if zerolag:
        acf_len = num_lags_calc
        lag_start_idx = 0
    else:
        if num_lags_calc == 0: return np.array([])
        acf_len = num_lags_calc - 1
        lag_start_idx = 1

    ACF = np.zeros(acf_len)
    mask_float = mask.astype(float)

    # ACF Loop
    for i in tqdm(range(lag_start_idx, num_lags_calc), desc="Calculating ACF (kenzie)", leave=False):
        shifted_mask = shift(mask_float, i, nchan)
        shifted_x = shift(x_meansub, i, nchan)
        unshifted_mask = shift(mask_float, 0, nchan)
        unshifted_x = shift(x_meansub, 0, nchan)

        overlap_mask = unshifted_mask * shifted_mask
        numerator = np.nansum(unshifted_x * shifted_x * overlap_mask)
        num_overlap_points = np.sum(overlap_mask)

        if num_overlap_points > 0:
            acf_index = i - lag_start_idx
            # Normalization applied here
            ACF[acf_index] = numerator / (num_overlap_points * denom)
        else:
            ACF[acf_index] = np.nan

    # Original code had a check `if i > 1` inside the loop when zerolag=False,
    # which seemed redundant given lag_start_idx is already 1. Removed this.
    # Original code also assigned to ACF[i-1] when zerolag=False, adjusted to ACF[i-lag_start_idx].

    return ACF


def autocorr_m(x, v=None, zerolag=False, maxlag=None):
    """
    Calculates the ACF using a different normalization (correlation coefficient style).

    Normalizes by the product of the standard deviations of the overlapping segments.

    Parameters
    ----------
    x : np.ndarray
        1D input spectrum.
    v : np.ndarray, optional
        Mask array (1 for valid, 0 for masked).
    zerolag : bool, optional
        Include zero-lag if True. Default False.
    maxlag : int, optional
        Maximum lag in channels. Default None (all lags).

    Returns
    -------
    ACF : np.ndarray
        The calculated auto-correlation function (positive lags).
    """
    nchan = len(x)
    if v is None:
        v = np.ones(nchan)
    x = x.copy()
    mask = v.astype(bool)
    x[~mask] = np.nan # Use NaNs for masked values

    # Subtract mean from valid points
    x_mean_valid = np.nanmean(x)
    if np.isnan(x_mean_valid): x_mean_valid = 0 # Handle case where all are masked
    x[mask] -= x_mean_valid

    # ACF Calculation setup
    if maxlag is None:
        num_lags_calc = nchan
    else:
        maxlag = min(int(maxlag), nchan)
        num_lags_calc = maxlag

    if zerolag:
        acf_len = num_lags_calc
        lag_start_idx = 0
    else:
        if num_lags_calc == 0: return np.array([])
        acf_len = num_lags_calc - 1
        lag_start_idx = 1

    ACF = np.zeros(acf_len)
    mask_float = mask.astype(float)

    # ACF Loop
    for i in tqdm(range(lag_start_idx, num_lags_calc), desc="Calculating ACF (m-norm)", leave=False):
        shifted_mask = shift(mask_float, i, nchan)
        shifted_x = shift(x, i, nchan) # Shift mean-subtracted data
        unshifted_mask = shift(mask_float, 0, nchan)
        unshifted_x = shift(x, 0, nchan)

        overlap_mask = unshifted_mask * shifted_mask

        # Numerator: Sum of product over overlap
        numerator = np.nansum(unshifted_x * shifted_x * overlap_mask)

        # Denominator: sqrt(sum(x_unshifted^2 * overlap) * sum(x_shifted^2 * overlap))
        sum_sq_unshifted = np.nansum(unshifted_x**2 * overlap_mask)
        sum_sq_shifted = np.nansum(shifted_x**2 * overlap_mask)

        denom = np.sqrt(sum_sq_unshifted * sum_sq_shifted)

        acf_index = i - lag_start_idx
        if denom != 0:
            ACF[acf_index] = numerator / denom
        else:
            ACF[acf_index] = np.nan # Or 0 if no overlap or zero variance

    return ACF


def lorentz(x, gamma, m, c):
    """ Lorentzian model definition used by lmfit. """
    # Original form: m**2 / (1+(x/gamma)**2) + c
    # Alternative form sometimes used: (y0 * gamma**2) / (x**2 + gamma**2) + c
    # This implementation uses the m^2 form.
    return m**2 / (1 + (x / gamma)**2) + c

def doublelorentz(x, gamma1, m1, gamma2, m2, c):
    """ Double Lorentzian model definition used by lmfit. """
    return (m1**2 / (1 + (x / gamma1)**2)) + \
           (m2**2 / (1 + (x / gamma2)**2)) + c

def scint_freq_relation(v, c, n):
    """ Power-law model for scintillation bandwidth vs frequency. """
    # Note: Identical to scint_functions.scint_freq_relation
    v_safe = np.maximum(v, 1e-9)
    return c * (v_safe**n)

def acf_scint_plot(ds, freq_ids, freqs, time_range, lagrange_for_fit=10.,
                   diagnostic_plots=True, maxlag=None, offspec_mean=None):
    """
    Calculates ACF, fits Lorentzian, plots results.

    Note: This appears identical to `acf_scint_plot` in `scint_functions.py`.
          Consider consolidating.

    Parameters and Returns: See `scint_functions.acf_scint_plot` docstring.
    """
    # --- Implementation copied from scint_functions.py ---
    if ds.ndim not in [1, 2]: raise ValueError("Input ds must be 1D or 2D.")
    if freqs.shape != freq_ids.shape or freqs.ndim != 1: raise ValueError("freqs/freq_ids shape mismatch")
    if ds.shape[0] != len(freqs): raise ValueError("Frequency dimension mismatch")

    valid_indices = np.where(np.diff(freq_ids) > 0)[0]
    if len(valid_indices) < 1:
        f_res = np.abs(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.390625 / TOTAL_CHANNELS
    else:
        idx0 = valid_indices[0]
        num_chan_diff = int(freq_ids[idx0 + 1] - freq_ids[idx0])
        freq_diff = np.abs(freqs[idx0 + 1] - freqs[idx0])
        f_res = freq_diff / float(num_chan_diff) if num_chan_diff > 0 else freq_diff
    print(f"Frequency resolution: {f_res:.5f} MHz/channel")
    if f_res <= 0: return None, None, None

    spec_1d = np.ma.mean(ds[:, int(time_range[0]):int(time_range[1])], axis=1) if ds.ndim == 2 else np.ma.masked_array(ds)
    spec_1d = np.ma.masked_where((spec_1d == 0) | np.isnan(spec_1d), spec_1d)
    mask_1d = ~spec_1d.mask

    maxlag_bin = int(maxlag / f_res) if maxlag is not None else None
    if maxlag_bin is not None and maxlag_bin <= 0: maxlag_bin = 1

    acf_pos = autocorr(spec_1d, v=mask_1d, zerolag=False, maxlag=maxlag_bin, offspec_mean=offspec_mean)

    if len(acf_pos) == 0 or np.all(np.isnan(acf_pos)): return None, None, None

    acf_full = np.concatenate((acf_pos[::-1], acf_pos))
    lags_pos_bins = np.arange(1, len(acf_pos) + 1)
    lags_mhz = np.concatenate((-lags_pos_bins[::-1], lags_pos_bins)) * f_res

    fit_result = None
    try:
        # Use lorentz model defined above (includes constant c)
        gmodel = Model(lorentz)
        fit_mask = (lags_mhz >= -lagrange_for_fit) & (lags_mhz <= lagrange_for_fit)
        lags_for_fit = lags_mhz[fit_mask]
        acf_for_fit = acf_full[fit_mask]
        if len(lags_for_fit) < 3: raise ValueError("Not enough points for fit.")

        initial_gamma = lagrange_for_fit / 4.0
        initial_m = np.sqrt(np.max(acf_for_fit)) if np.max(acf_for_fit) > 0 else 1.0
        initial_c = np.min(acf_for_fit)

        # Fit using the 'lorentz' function which includes 'c'
        fit_result = gmodel.fit(acf_for_fit, x=lags_for_fit,
                                gamma=initial_gamma, m=initial_m, c=initial_c)
        if diagnostic_plots: from lmfit import fit_report; print(fit_report(fit_result)) # Import here if needed

    except Exception as e:
        print(f"Could not fit Lorentzian: {e}")

    if diagnostic_plots:
        plt.figure(figsize=(10, 6))
        plt.plot(lags_mhz, acf_full, drawstyle='steps-mid', color='k', lw=1.0, label=f"ACF (res: {f_res:.3f} MHz)")
        if fit_result:
            gamma_fit = fit_result.params['gamma'].value
            plt.plot(lags_mhz, fit_result.eval(x=lags_mhz), color='orange', label=f'Fit ($\\gamma$={gamma_fit:.3f} MHz)')
            plot_xlim = max(np.abs(gamma_fit) * 10, lagrange_for_fit * 1.5)
            plt.xlim(-plot_xlim, plot_xlim)
        else:
            plot_xlim_default = maxlag if maxlag is not None else (lags_mhz[-1] if len(lags_mhz)>0 else 1.0)
            plt.xlim(-plot_xlim_default, plot_xlim_default)
        plt.xlabel("Frequency Lag [MHz]"); plt.ylabel("Normalized ACF"); plt.title("ACF")
        plt.grid(True, linestyle=':', alpha=0.6); plt.legend(); plt.tight_layout(); plt.show()

    return acf_full, lags_mhz, fit_result


# --- Burst Property Calculation ---

def get_burst_envelope_kn(power, thres=5, pad=0.0, diagnostic_plots=False):
    """
    Determines the time limits of a burst signal in a power profile.

    Iteratively identifies the main peak above a threshold, masks it, and
    repeats until only noise floor remains (or all data is masked).
    The final limits encompass all identified peaks.

    Parameters
    ----------
    power : np.ndarray or tuple
        Input power data. If 2D [freq, time], it's averaged over frequency.
        If 1D [time], it's used directly. Tuple input might be legacy? Assume ndarray.
    thres : float, optional
        Signal-to-noise threshold used to identify peaks relative to the
        noise floor standard deviation. Default is 5.
    pad : float, optional
        Fractional padding to add to the start and end of the determined limits.
        E.g., pad=0.1 adds 10% of the burst duration to each side. Default is 0.0.
    diagnostic_plots : bool or str, optional
        If True, show diagnostic plot. If str, save plot to that path. Default False.

    Returns
    -------
    lims : np.ndarray
        Array containing [start_bin, end_bin] of the burst envelope.
    """
    # Get 1D power profile
    if isinstance(power, tuple): # Handle legacy tuple input?
        print("Warning: tuple input for power in get_burst_envelope_kn is deprecated. Assuming ndarray.")
        # Need clarification on how tuple power was structured. Assume first element?
        power_arr = np.asarray(power[0]) if len(power)>0 else np.array([])
    else:
        power_arr = np.asarray(power)

    if power_arr.ndim == 0:
         raise ValueError("Input power cannot be scalar.")
    elif power_arr.ndim >= 2:
        # Assuming [freq, time] or [pol, freq, time], average over all but last axis
        avg_axes = tuple(range(power_arr.ndim - 1))
        prof = np.nanmean(power_arr, axis=avg_axes)
    else: # 1D
        prof = power_arr.copy()

    if len(prof) == 0:
        print("Warning: Power profile is empty.")
        return np.array([0, 0])

    # Normalize profile by median and std dev of the noise floor
    floor = prof.copy()
    median_floor = np.nanmedian(floor)
    floor -= median_floor # Subtract median
    std_floor = np.nanstd(floor)
    if std_floor == 0: # Handle constant data
        print("Warning: Standard deviation of floor is zero.")
        # If std dev is 0, treat any deviation from median as significant?
        # Or assume no burst? For now, return full range if std is 0.
        return np.array([0, len(prof)])

    prof_norm = (prof - median_floor) / std_floor
    floor_norm = floor / std_floor

    # Iteratively mask peaks above threshold
    while True:
        # Find main peak in the *current* floor
        # Use baseband_analysis function if available
        try:
            peak_t0, peak_t1 = get_main_peak_lim(floor_norm, floor_level=thres)
        except NameError:
            # Basic fallback if get_main_peak_lim is not imported
            print("Warning: get_main_peak_lim not found. Using basic peak finding.")
            above_thresh = np.where(floor_norm > thres)[0]
            if len(above_thresh) == 0: break
            peak_t0, peak_t1 = above_thresh[0], above_thresh[-1] + 1 # Inclusive range

        # Check if the found peak covers the entire remaining floor
        if (peak_t1 - peak_t0) >= len(floor_norm[~np.isnan(floor_norm)]):
            # Avoid infinite loop if threshold is too low / noise is high
            if np.all(np.isnan(floor_norm)): break # All masked
            # If peak covers everything left, assume it's the last part
            floor_norm[peak_t0:peak_t1] = np.nan # Mask it and break
            break

        # Mask the identified peak
        floor_norm[peak_t0:peak_t1] = np.nan

        # Check if any points remain above threshold
        if not np.any(floor_norm > thres):
            break
        # Check if all points are masked
        if np.all(np.isnan(floor_norm)):
            break

        # Re-normalize based on remaining floor (optional, original didn't do this)
        # median_rem = np.nanmedian(floor_norm)
        # std_rem = np.nanstd(floor_norm)
        # if std_rem > 0: floor_norm = (floor_norm - median_rem) / std_rem

    # Find the overall limits from the masked regions
    is_masked = np.isnan(floor_norm)
    if not np.any(is_masked): # No peaks found above threshold
        print("Warning: No significant burst detected above threshold.")
        # Return center +/- small window? Or [0, 0]? Or full range?
        # Returning [0, 0] might be problematic downstream. Return full range?
        lims = np.array([0, len(prof)])
    else:
        try:
            lims = np.array([np.min(np.where(is_masked)), np.max(np.where(is_masked)) + 1])
        except ValueError: # Should not happen if np.any(is_masked) is true
             lims = np.array([0, len(prof)]) # Fallback


    # Apply padding
    duration = lims[1] - lims[0]
    pad_bins = int(duration * pad)
    lims[0] = max(0, lims[0] - pad_bins)
    lims[1] = min(len(prof), lims[1] + pad_bins)

    # --- Diagnostic Plotting ---
    if diagnostic_plots:
        plt.figure(figsize=(10, 5))
        time_axis = np.arange(len(prof)) # Simple bin number axis
        plt.plot(time_axis, prof_norm, label='Normalized Profile')
        # Plot original floor (before masking) for comparison
        plt.plot(time_axis, (prof - median_floor) / std_floor, alpha=0.5, label='Original Floor (Normalized)')
        # Plot the final masked floor
        plt.plot(time_axis, floor_norm, linestyle=':', color='grey', label='Final Floor (Masked Peaks)')
        plt.axhline(thres, color='r', linestyle='--', label=f'Threshold ({thres}$\sigma$)')
        plt.axvline(lims[0], c="k", ls="--", label='Burst Limits')
        plt.axvline(lims[1], c="k", ls="--")
        plt.xlabel("Time [bins]")
        plt.ylabel("Normalized Intensity (S/N)")
        plt.title("Burst Envelope Detection")
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.tight_layout()

        if isinstance(diagnostic_plots, str):
            try:
                plt.savefig(diagnostic_plots) # Save plot to specified path
                print(f"Saved diagnostic plot to: {diagnostic_plots}")
            except Exception as e:
                print(f"Error saving diagnostic plot: {e}")
            plt.close() # Close plot after saving
        else:
            plt.show() # Show plot interactively

    return lims.astype(int) # Return integer bin limits


# --- Main Data Processing Function ---

def data_dedisp_derip_filled_masked(event_id, dm, downsample_factor=32,
                                    interactive=True, off=False, file=None,
                                    zap_extra=True, diagnostic_plot=None,
                                    time_range=None):
    """
    Main processing pipeline for a single event.

    Loads data, calculates SNR, dedisperses, identifies valid channels/times,
    fills missing channels, optionally performs extra RFI flagging, and
    determines on-burst or off-burst time windows.

    Parameters
    ----------
    event_id : int or str
        ID of the FRB event.
    dm : float
        Dispersion measure (pc cm^-3) to use for dedispersion.
    downsample_factor : int, optional
        Time downsampling factor used for SNR calculation and potentially
        scrunching plots. Default is 32.
    interactive : bool, optional
        If True, prompts user to define time ranges via input. If False,
        uses automatic burst envelope detection. Default is True.
    off : bool, optional
        If True, select the off-burst region *before* the main burst.
        If False, select the region encompassing the burst. Default is False.
    file : str, optional
        Direct path to the baseband HDF5 file. If None, attempts to
        construct the path using `get_data`. Default is None.
    zap_extra : bool, optional
        If True, performs an additional RFI flagging step based on channel
        spectrum statistics after initial processing. Default is True.
    diagnostic_plot : str, optional
        Directory path to save diagnostic plots (e.g., profiles). If None,
        plots might be shown interactively depending on `interactive`. Default None.
    time_range : list or tuple, optional
        Explicitly define the time range [start_bin, end_bin] in *downsampled*
        units to keep. Overrides interactive/automatic selection. Default None.

    Returns
    -------
    data_final : np.ma.MaskedArray
        Processed complex voltage data, dedispersed, filled, masked, and
        time-trimmed. Shape [1024, pol, time_trimmed].
    freqs_final : np.ndarray
        Frequency array (MHz) for the 1024 channels.
    freq_id_final : np.ndarray
        Frequency channel IDs (0-1023).
    """
    print(f"\n--- Processing Event {event_id} (DM={dm}) ---")

    # --- Load Data ---
    if file is None:
        if master is None:
             raise ConnectionError("CHIME FRB Master API not available to fetch event info for get_data.")
        try:
            print("Fetching event metadata...")
            event = master.events.get_event(event_id)
            frb_filepath = get_data(event)
            print(f"Constructed data path: {frb_filepath}")
        except Exception as e:
            raise RuntimeError(f"Could not get event metadata or construct file path: {e}")
    else:
        frb_filepath = file
        print(f"Using provided data path: {frb_filepath}")

    try:
        print("Loading baseband data...")
        frb_bbdata = BBData.from_file(frb_filepath)
    except Exception as e:
        raise IOError(f"Could not load BBData from file {frb_filepath}: {e}")

    # --- Initial SNR and Power Calculation ---
    # Calculate power if not already present
    if "tiedbeam_power" not in list(frb_bbdata.keys()):
        print("Calculating tiedbeam power...")
        try:
            # time_downsample_factor=1 here means power is calculated at native res
            tiedbeam_baseband_to_power(
                frb_bbdata, time_downsample_factor=1, dm=dm, dedisperse=True, time_shift=False
            )
        except Exception as e:
             print(f"Warning: tiedbeam_baseband_to_power failed: {e}. Continuing without it.")

    # Calculate SNR and get valid channel/time masks
    print(f"Calculating SNR (downsample={downsample_factor})...")
    try:
        # Assuming get_snr returns: snr, peak_dm, peak_time, spec, prof, valid_chans, valid_times
        snr_results = get_snr(frb_bbdata, DM=dm, diagnostic_plots=False, # Disable internal plots
                              return_full=True, downsample=downsample_factor,
                              DM_range=None, spectrum_lim=False)
        valid_channels_mask = snr_results[5] # Boolean mask for valid channels
        valid_time_bins_native = snr_results[6] # Time range [start, end] in native bins
        print(f"Initial valid time range (native bins): {valid_time_bins_native}")
        print(f"Number of initial valid channels: {np.sum(valid_channels_mask)}")
    except NameError:
        raise RuntimeError("baseband_analysis.analysis.snr.get_snr not available.")
    except Exception as e:
        print(f"Warning: get_snr failed: {e}. Attempting to proceed with default masks.")
        # Fallback: assume all channels/times are initially valid
        try:
            nchan_bb = frb_bbdata['tiedbeam_baseband'].shape[0]
            nsamp_bb = frb_bbdata['tiedbeam_baseband'].shape[2]
            valid_channels_mask = np.ones(nchan_bb, dtype=bool)
            valid_time_bins_native = [0, nsamp_bb]
        except KeyError:
            raise RuntimeError("Cannot determine data shape: 'tiedbeam_baseband' not in BBData.")


    # --- Dedispersion ---
    print(f"Applying coherent dedispersion (DM={dm})...")
    if dm != 0:
        try:
            # coherent_dedisp modifies bbdata in place
            coherent_dedisp(frb_bbdata, dm, time_shift=False, write=True) # write=True adds dedispersed data to bbdata
            # Access the dedispersed data (assuming key 'tiedbeam_baseband_dedispersed')
            data_dedisp = frb_bbdata['tiedbeam_baseband_dedispersed'] # Shape [freq, pol, time]
            freq_id_dedisp = frb_bbdata.index_map['freq']['id']
            # freqs_dedisp = frb_bbdata.index_map['freq']['centre'] # Freqs needed later
        except NameError:
             raise RuntimeError("baseband_analysis.core.dedispersion.coherent_dedisp not available.")
        except KeyError:
             raise RuntimeError("Could not find 'tiedbeam_baseband_dedispersed' after coherent_dedisp.")
        except Exception as e:
             raise RuntimeError(f"Coherent dedispersion failed: {e}")
    else:
        print("DM is 0, skipping coherent dedispersion.")
        try:
            data_dedisp = frb_bbdata['tiedbeam_baseband']
            freq_id_dedisp = frb_bbdata.index_map['freq']['id']
        except KeyError:
             raise RuntimeError("Could not find 'tiedbeam_baseband' in BBData.")

    # --- Apply Initial Masks ---
    print("Applying initial channel and time masks...")
    # Ensure mask length matches data frequency dimension
    if len(valid_channels_mask) != data_dedisp.shape[0]:
        print(f"Warning: Channel mask length ({len(valid_channels_mask)}) mismatch with data "
               f"({data_dedisp.shape[0]}). Attempting resize/broadcast if possible, or using full mask.")
        # Basic fallback: assume all channels are valid if mask doesn't match
        if len(valid_channels_mask) < data_dedisp.shape[0]:
            valid_channels_mask = np.ones(data_dedisp.shape[0], dtype=bool)
        else:
            valid_channels_mask = valid_channels_mask[:data_dedisp.shape[0]]


    # Create masked array
    data_masked_tmp = np.ma.masked_array(data_dedisp, mask=False) # Start with no mask
    # Apply channel mask (mask=True where invalid)
    # Expand mask dimensions: [freq] -> [freq, 1, 1] to broadcast
    channel_mask_3d = ~valid_channels_mask[:, np.newaxis, np.newaxis]
    data_masked_tmp.mask = np.logical_or(data_masked_tmp.mask, channel_mask_3d)

    # Apply time mask (trim data)
    t_start = int(valid_time_bins_native[0])
    t_end = int(valid_time_bins_native[1])
    t_start = max(0, t_start) # Ensure start is not negative
    t_end = min(data_masked_tmp.shape[2], t_end) # Ensure end is within bounds
    if t_start >= t_end:
         raise ValueError(f"Initial valid time range is empty or invalid: [{t_start}, {t_end}]")

    data_masked_tmp = data_masked_tmp[:, :, t_start:t_end]
    print(f"Data trimmed to native time bins: [{t_start}, {t_end}]")

    # --- Determine Time Window for Analysis (On/Off Burst) ---
    print("Determining analysis time window...")
    # Calculate power profile *after* initial masking and trimming
    power_trimmed = np.abs(data_masked_tmp)**2
    I_trimmed = np.ma.sum(power_trimmed, axis=1) # Sum over polarization -> [freq, time]
    # Scrunch in time for faster processing / plotting
    I_scr = scrunch(I_trimmed, tscrunch=downsample_factor, fscrunch=1) # [freq, time_scr]

    # Define time range in *downsampled* bins
    if time_range is not None:
        # Use user-provided range
        start_bin_ds, end_bin_ds = int(time_range[0]), int(time_range[1])
        print(f"Using provided downsampled time range: [{start_bin_ds}, {end_bin_ds}]")
    elif interactive:
        # Interactive selection
        plt.close('all')
        profile_scr = np.ma.mean(I_scr, axis=0) # Average over freq -> [time_scr]
        plt.plot(profile_scr.filled(np.nanmedian(profile_scr))) # Plot filled profile
        plt.xlabel(f"Time [bins x {downsample_factor}]")
        plt.ylabel("Intensity [arb.]")
        plt.title("Select Time Range to Keep")
        plt.grid(True)
        plt.show(block=False) # Show non-blocking plot
        answer = input(f"Define downsampled time bin range to keep (e.g., '100,{len(profile_scr)-100}'): ")
        plt.close() # Close plot after input
        try:
            start_bin_ds, end_bin_ds = map(int, answer.split(','))
        except Exception as e:
            raise ValueError(f"Invalid input format for time range: {e}")
    else:
        # Automatic selection using get_burst_envelope_kn
        print("Using automatic burst envelope detection...")
        # Use the *trimmed* power data (native time resolution)
        # Pass the full power array [freq, pol, time]
        try:
            lims_native = get_burst_envelope_kn(power_trimmed.filled(0), # Use filled power
                                                 thres=6, pad=0.1, # Add 10% padding
                                                 diagnostic_plots=(diagnostic_plot is not None))
            print(f"Detected burst limits (native bins): {lims_native}")
        except Exception as e:
            print(f"Warning: get_burst_envelope_kn failed: {e}. Using full trimmed range.")
            lims_native = [0, power_trimmed.shape[-1]]


        # Define range based on 'off' flag
        if off:
            # Select region *before* the burst start (with margin)
            margin_native = 10000 # Margin in native bins
            end_native = max(0, lims_native[0] - margin_native)
            start_native = 0
            print(f"Selecting OFF-burst range (native bins): [{start_native}, {end_native}]")
        else:
            # Select region around the burst (with margin)
            margin_native = 20000
            start_native = max(0, lims_native[0] - margin_native)
            end_native = min(power_trimmed.shape[-1], lims_native[1] + margin_native)
            print(f"Selecting ON-burst range (native bins): [{start_native}, {end_native}]")

        # Convert native bin range to downsampled bin range (approximate)
        start_bin_ds = start_native // downsample_factor
        end_bin_ds = end_native // downsample_factor
        # Ensure at least one downsampled bin is selected
        if start_bin_ds >= end_bin_ds: end_bin_ds = start_bin_ds + 1

    # Ensure final downsampled range is valid
    num_ds_bins = I_scr.shape[1]
    start_bin_ds = max(0, start_bin_ds)
    end_bin_ds = min(num_ds_bins, end_bin_ds)
    if start_bin_ds >= end_bin_ds:
        raise ValueError(f"Final downsampled time range is empty: [{start_bin_ds}, {end_bin_ds}]")
    print(f"Final selected downsampled time range: [{start_bin_ds}, {end_bin_ds}]")

    # Convert downsampled bin range back to native bin range for final data slicing
    start_bin_final = start_bin_ds * downsample_factor
    end_bin_final = end_bin_ds * downsample_factor
    # Ensure final native range is within the bounds of the *trimmed* data
    start_bin_final = max(0, start_bin_final)
    end_bin_final = min(data_masked_tmp.shape[2], end_bin_final)
    print(f"Final selected native time range: [{start_bin_final}, {end_bin_final}]")


    # --- Fill Missing Channels and Final Slice ---
    print("Filling missing channels to 1024...")
    # Pass the *trimmed* data to fill_missing_chans
    # Note: fill_missing_chans expects bbdata object for original freq map
    data_filled, freqs_final, freq_id_final = fill_missing_chans(
        data_masked_tmp, frb_bbdata
    )
    # data_filled shape: [1024, pol, time_trimmed]

    # Apply the final time slice determined above
    data_filled_sliced = data_filled[:, :, start_bin_final:end_bin_final]
    print(f"Final data shape: {data_filled_sliced.shape}")


    # --- Optional Extra RFI Zapping ---
    if zap_extra:
        print("Performing extra RFI zapping...")
        # Calculate channel spectrum on the filled, time-sliced data
        power_final = np.abs(data_filled_sliced)**2 # [1024, pol, time_final]
        # Sum over pol and time
        chan_spectrum = np.ma.sum(np.ma.sum(power_final, axis=1), axis=-1) # [1024]

        # Normalize using robust stats (median/MAD)
        try:
            spec_median = np.ma.median(chan_spectrum)
            spec_mad = median_abs_deviation(chan_spectrum.compressed(), scale='normal') # Use compressed for MAD
            if spec_mad == 0: spec_mad = np.ma.std(chan_spectrum) # Fallback
            if spec_mad == 0: spec_mad = 1.0 # Avoid division by zero

            chan_spectrum_snr = (chan_spectrum - spec_median) / spec_mad

            # Identify channels with low power (negative SNR outliers)
            # Original condition: (chan_spectrum_snr < -1) * (chan_spectrum > 0)
            # Using a slightly more standard threshold, e.g., -3 sigma
            rfi_mask_extra = (chan_spectrum_snr < -3.0) & (~chan_spectrum.mask)

            num_extra_zapped = np.sum(rfi_mask_extra)
            print(f"Zapping {num_extra_zapped} additional channels based on low power.")

            # Apply mask (mask=True where RFI identified)
            # Expand mask dimensions: [1024] -> [1024, 1, 1]
            rfi_mask_3d = rfi_mask_extra[:, np.newaxis, np.newaxis]
            # Combine with existing mask
            data_final = np.ma.masked_where(
                np.logical_or(data_filled_sliced.mask, rfi_mask_3d),
                data_filled_sliced.data # Apply to underlying data
            )
        except ImportError:
            print("Warning: Cannot perform extra zapping, scipy.stats.median_abs_deviation not found.")
            data_final = data_filled_sliced # Use data before this step
        except Exception as e:
            print(f"Warning: Extra RFI zapping failed: {e}")
            data_final = data_filled_sliced
    else:
        data_final = data_filled_sliced # Use data without extra zapping

    # --- Final Diagnostic Plot (Profile) ---
    print("Generating final time profile plot...")
    power_final_plot = np.abs(data_final)**2
    I_final = np.ma.sum(power_final_plot, axis=1) # Sum pol -> [1024, time_final]

    # Profile at native resolution
    prof_native = np.ma.mean(I_final, axis=0) # Mean freq -> [time_final]
    time_axis_native = np.linspace(0, len(prof_native) * 2.56e-3, len(prof_native)) # Time in ms

    # Profile scrunched
    # Need to handle potential non-integer division for scrunching final data
    current_ds_factor = downsample_factor # Or recalculate based on final length?
    prof_scr = scrunch(I_final, tscrunch=current_ds_factor, fscrunch=1)
    prof_scr_mean = np.ma.mean(prof_scr, axis=0)
    time_axis_scr = np.linspace(0, len(prof_scr_mean) * 2.56e-3 * current_ds_factor, len(prof_scr_mean))

    plt.close('all')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_axis_native, prof_native.filled(np.nan), color='k', alpha=0.5, label='Native Res Profile')
    ax.plot(time_axis_scr, prof_scr_mean.filled(np.nan), color='r', label=f'Scrunched x{current_ds_factor}')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Intensity [arb.]')
    ax.set_title(f'Event {event_id} - Final Time Profile ({ "Off" if off else "On"}-Burst Window)')
    ax.legend()
    ax.grid(True, alpha=0.5)
    plt.tight_layout()

    if diagnostic_plot is not None:
        plot_filename = f'{diagnostic_plot}/{"off" if off else "on"}burst_prof_evt{event_id}.png'
        try:
            plt.savefig(plot_filename, format='png')
            print(f"Saved final profile plot to: {plot_filename}")
        except Exception as e:
            print(f"Error saving profile plot: {e}")
        plt.close(fig)
    elif not interactive: # Show plot if not interactive and no save path
        plt.show()
    # If interactive, plot was likely shown earlier or user doesn't need this one shown again.


    # --- Return Results ---
    # Ensure frequencies are high to low
    if freqs_final[0] < freqs_final[-1]:
        print("Flipping frequency axis to be high-to-low.")
        freqs_final = freqs_final[::-1]
        # Flip data and IDs accordingly
        data_final = data_final[::-1, :, :]
        freq_id_final = freq_id_final[::-1] # Should still be 0-1023 but reversed

    print(f"--- Finished Processing Event {event_id} ---")
    return data_final, freqs_final, freq_id_final


def extra_flag(com_vol):
    """
    Performs additional RFI flagging based on channel spectrum statistics.

    Identifies channels where the total power (summed over time and pol)
    is significantly lower than the median (< -1 sigma using MAD).

    Note: This logic seems similar to the `zap_extra` part within
          `data_dedisp_derip_filled_masked`. Consider consolidation.

    Parameters
    ----------
    com_vol : np.ndarray or np.ma.MaskedArray
        Input complex voltage array [freq, pol, time].

    Returns
    -------
    data_masked : np.ma.MaskedArray
        Input array with additional channels masked based on low power.
    """
    if not isinstance(com_vol, np.ma.MaskedArray):
        data_masked = np.ma.masked_array(com_vol) # Ensure it's masked
    else:
        data_masked = com_vol.copy() # Work on a copy

    # Calculate channel spectrum
    chan_spectrum = np.ma.sum(np.ma.sum(np.abs(data_masked)**2, axis=1), axis=-1)

    # Normalize using robust stats
    try:
        spec_median = np.ma.median(chan_spectrum)
        # Use compressed array (only valid points) for MAD calculation
        spec_mad = median_abs_deviation(chan_spectrum.compressed(), scale='normal')
        if spec_mad == 0: spec_mad = np.ma.std(chan_spectrum) # Fallback
        if spec_mad == 0: spec_mad = 1.0 # Avoid division by zero

        chan_spectrum_snr = (chan_spectrum - spec_median) / spec_mad

        # Identify low power channels (original threshold: < -1 sigma)
        miss_chan_mask = (chan_spectrum_snr < -1.0) & (~chan_spectrum.mask)

        num_flagged = np.sum(miss_chan_mask)
        if num_flagged > 0:
            print(f"Extra flagging: Masking {num_flagged} channels with power < -1 sigma.")
            # Apply mask (expand dims: [freq] -> [freq, 1, 1])
            data_masked.mask = np.logical_or(data_masked.mask, miss_chan_mask[:, np.newaxis, np.newaxis])

    except ImportError:
         print("Warning: Cannot perform extra flagging, scipy.stats.median_abs_deviation not found.")
    except Exception as e:
         print(f"Warning: Extra flagging failed: {e}")

    return data_masked


# --- Model Fitting Functions (Simple Models) ---

def gaus(x, a, x0, sigma, c):
    """ Gaussian function definition. """
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + c

def scatt_tail(t, tau_scatt, t0, t1, sigma, a):
    """
    Models a scattering tail by convolving a Gaussian with a one-sided exponential.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    tau_scatt : float
        Scattering timescale (decay time of exponential).
    t0 : float
        Center of the Gaussian component.
    t1 : float
        Start time of the exponential decay (relative to t).
    sigma : float
        Standard deviation (width) of the Gaussian component.
    a : float
        Amplitude scaling factor.

    Returns
    -------
    np.ndarray
        The convolved scattering tail model.
    """
    # Define the Gaussian component
    gaussian_component = gaus(t, 1, t0, sigma, 0) # Use normalized amplitude

    # Define the one-sided exponential decay
    # Ensure decay only happens for t >= t1
    exponential_decay = np.zeros_like(t)
    time_indices_after_t1 = np.where(t >= t1)[0]
    if len(time_indices_after_t1) > 0:
        # Apply decay relative to t1
        exponential_decay[time_indices_after_t1] = np.exp(-(t[time_indices_after_t1] - t1) / tau_scatt)

    # Convolve the two components
    # 'same' mode returns output with same shape as first input (t)
    # 'direct' method computes convolution directly
    convolved_signal = signal.convolve(gaussian_component, exponential_decay,
                                       mode='same', method='direct')

    # Scale by amplitude 'a'
    return a * convolved_signal


# --- PFB Flattening (Requires pfb_tools) ---

def fit_n_flat(ds, t_lims, Q):
    """
    Attempts to flatten the PFB bandpass shape using the DeconvolvePFB tool.

    Requires the `pfb_tools` package. Calculates weights based on off-pulse
    variance and applies the flattening algorithm.

    Parameters
    ----------
    ds : np.ndarray
        Input dynamic spectrum [freq, time]. Assumes frequency axis corresponds
        to PFB channels (length must be multiple of Q).
    t_lims : tuple or list
        Time bin limits [start_bin, end_bin] defining the on-pulse region.
        Used to define the off-pulse region for weight calculation.
    Q : int
        PFB factor (related to fftsize, e.g., 16).

    Returns
    -------
    ds_flatten : np.ma.MaskedArray
        The flattened dynamic spectrum.
    spec_flat : np.ma.MaskedArray
        The time-averaged spectrum of the on-pulse region after flattening.
    off_spec_flat : np.ma.MaskedArray
        The time-averaged spectrum of the off-pulse region after flattening.

    Raises
    ------
    ImportError
        If `pfb_tools.DeconvolvePFB` cannot be imported.
    ValueError
        If input dimensions are incorrect.
    """
    try:
        from pfb_tools import DeconvolvePFB
    except ImportError:
        raise ImportError("Function fit_n_flat requires the 'pfb_tools' package.")

    if ds.ndim != 2:
        raise ValueError("Input ds must be 2D [freq, time].")
    if ds.shape[0] % Q != 0:
        raise ValueError(f"Number of frequency channels ({ds.shape[0]}) must be divisible by Q ({Q}).")

    nchan, ntime = ds.shape
    nchan_coarse = nchan // Q

    # Reshape data for DeconvolvePFB: [nchan_coarse, Q, ntime]
    ds_reshape = ds.reshape(nchan_coarse, Q, ntime)

    # Define off-pulse data for weight calculation
    off_indices = np.r_[0:t_lims[0], t_lims[1]:ntime]
    offdata = ds[:, off_indices] # Shape [nchan, ntime_off]

    # Calculate weights (inverse variance of off-pulse) per channel
    weights = np.zeros_like(ds) # Shape [nchan, ntime]
    off_var = np.var(offdata, axis=1) # Variance per channel [nchan]

    # Avoid division by zero for channels with zero variance
    valid_var_mask = off_var != 0
    weights[valid_var_mask, :] = 1.0 / off_var[valid_var_mask, np.newaxis]

    # Reshape weights to match ds_reshape: [nchan_coarse, Q, ntime]
    weights_reshape = weights.reshape(nchan_coarse, Q, ntime)

    # Handle NaNs before passing to flatten (replace with 0, weights will be 0 too)
    ds_nonan = np.nan_to_num(ds_reshape)
    weights_nonan = np.nan_to_num(weights_reshape)

    # Perform the flattening using DeconvolvePFB
    # This function returns flattened data `a` and potentially other info `b`
    print("Applying PFB flattening...")
    try:
        # Check expected signature of DeconvolvePFB.flatten if possible
        a, b = DeconvolvePFB(Q=Q).flatten(x=ds_nonan, Ni=weights_nonan)
        # `a` should have shape [nchan_coarse, Q, ntime]
    except Exception as e:
        raise RuntimeError(f"DeconvolvePFB().flatten failed: {e}")

    # Check output shape
    if a.shape != ds_nonan.shape:
         print(f"Warning: Flattened data shape {a.shape} differs from input {ds_nonan.shape}.")
         # Attempt to reshape or handle mismatch if possible, otherwise raise error
         raise RuntimeError("Shape mismatch after flattening.")


    # Reconstruct the full flattened dynamic spectrum [nchan, ntime]
    # Original code used concatenation, direct reshape is simpler:
    ds_flatten_raw = a.reshape(nchan, ntime)

    # Mask channels that were originally NaN or had zero variance (weight=0)
    # Also mask outputs that became zero during flattening (might indicate issues)
    final_mask = (weights == 0) | (ds_flatten_raw == 0) | np.isnan(ds_flatten_raw)
    ds_flatten = np.ma.masked_where(final_mask, ds_flatten_raw)

    # Calculate average spectra for on-pulse and off-pulse regions
    spec_flat = np.ma.mean(ds_flatten[:, t_lims[0]:t_lims[1]], axis=1)
    off_spec_flat = np.ma.mean(ds_flatten[:, off_indices], axis=1)

    return ds_flatten, spec_flat, off_spec_flat


# --- Simulation and Model Handling ---

def fakefrb(ds_noise, fb_model, data_I):
    """
    Injects a fitburst model into noise data.

    Scales the fitburst model by the peak intensity of the real data profile
    and adds it to the provided noise dynamic spectrum.

    Parameters
    ----------
    ds_noise : np.ndarray or np.ma.MaskedArray
        Complex voltage noise data [freq, pol, time].
    fb_model : np.ndarray
        Fitburst intensity model [freq, time].
    data_I : np.ndarray or np.ma.MaskedArray
        Real burst intensity data [freq, time] used for scaling.

    Returns
    -------
    fake_frb_ds : np.ndarray
        Complex voltage array representing noise + injected scaled model.
        Shape [freq, pol, time], same as ds_noise.
    """
    nfreq_noise, npol_noise, ntime_noise = ds_noise.shape
    nfreq_model, ntime_model = fb_model.shape
    nfreq_data, ntime_data = data_I.shape # Assuming data_I is [freq, time]

    if nfreq_noise != nfreq_model or nfreq_noise != nfreq_data:
        raise ValueError("Frequency dimension mismatch between noise, model, and data_I.")

    # --- Time Alignment and Padding/Truncation ---
    # Ensure fb_model has the same time duration as ds_noise
    if ntime_model < ntime_noise:
        # Pad model with zeros if shorter
        pad_width = ntime_noise - ntime_model
        # Pad axis 1 (time) after the existing data
        fb_model_aligned = np.pad(fb_model, ((0, 0), (0, pad_width)), mode='constant')
    elif ntime_model > ntime_noise:
        # Truncate model if longer
        print(f"Warning: Fitburst model duration ({ntime_model}) longer than noise ({ntime_noise}). Truncating model.")
        fb_model_aligned = fb_model[:, :ntime_noise]
    else:
        fb_model_aligned = fb_model # Durations match

    # --- Scaling ---
    # Scale model by peak of the real data profile
    data_prof = np.nanmean(data_I, axis=0) # Profile of real data [time]
    peak_data_intensity = np.nanmax(data_prof)

    # Avoid scaling by zero or NaN
    if np.isnan(peak_data_intensity) or peak_data_intensity == 0:
        print("Warning: Peak data intensity is zero or NaN. Using scale factor 1.")
        scale_factor = 1.0
    else:
        scale_factor = peak_data_intensity

    # Scale the aligned intensity model
    # Add small epsilon to avoid potential sqrt(0) if model has zeros
    scaled_model_intensity = fb_model_aligned * scale_factor
    # Convert scaled intensity model to complex voltage model (sqrt, arbitrary phase)
    # Assume equal power in both polarizations for the model injection
    scaled_model_voltage = np.sqrt(scaled_model_intensity / 2.0 + 1e-12) # Shape [freq, time]

    # --- Injection ---
    # Create the output array
    fake_frb_ds = np.zeros_like(ds_noise, dtype=np.complex64)

    # Generate random phases for the model voltage component
    random_phase = np.exp(1j * 2 * np.pi * np.random.rand(nfreq_noise, ntime_noise))
    model_voltage_complex = scaled_model_voltage * random_phase # Shape [freq, time]

    # Add model voltage to noise voltage for each polarization
    # Ensure noise is complex
    noise_complex = ds_noise.astype(np.complex64, copy=True)

    # Add model (broadcast phase across polarizations?) or use separate phases?
    # Original code used noise components scaled by model, which seems unusual.
    # Let's add the complex model voltage directly to the noise.
    fake_frb_ds[:, 0, :] = noise_complex[:, 0, :] + model_voltage_complex
    fake_frb_ds[:, 1, :] = noise_complex[:, 1, :] + model_voltage_complex # Same model voltage, different noise

    # --- Original Injection Logic (kept for reference) ---
    # Seems to use noise components multiplied by the *intensity* model, then scaled.
    # This might not correctly represent adding a coherent signal to noise.
    # r1 = np.random.normal(loc=np.nanmean(ds_noise), scale=np.nanstd(ds_noise), size=ds_noise.shape)
    # r2 = r1[:, 1, :]
    # r1 = r1[:, 0, :]
    # r3 = ds_noise[:, 0, :] # Using actual noise components
    # r4 = ds_noise[:, 1, :]
    # # Apply mask if noise is masked
    # if isinstance(ds_noise, np.ma.MaskedArray):
    #     r1 = np.ma.array(r1, mask=r3.mask)
    #     r2 = np.ma.array(r2, mask=r4.mask)
    # p0 = fb_model_aligned * r1 # Intensity model * noise component?
    # p1 = fb_model_aligned * r2
    # fake_frb_ds[:, 0, :] = p0 * scale_factor + r2 # Add different noise component back?
    # fake_frb_ds[:, 1, :] = p1 * scale_factor + r1
    # --- End Original Logic ---

    # Apply mask from original noise if it was masked
    if isinstance(ds_noise, np.ma.MaskedArray):
        fake_frb_ds = np.ma.masked_array(fake_frb_ds, mask=ds_noise.mask)

    return fake_frb_ds


def fitburst_model_to_ds(fitburst_json, downsamp=1):
    """
    Loads fitburst model parameters from a JSON file and computes the
    model dynamic spectrum using the fitburst library.

    Parameters
    ----------
    fitburst_json : str
        Path to the fitburst JSON file.
    downsamp : int, optional
        Time downsampling factor assumed when the fitburst model was created.
        Used to determine the time axis. Default is 1.

    Returns
    -------
    model : np.ndarray
        Computed fitburst model dynamic spectrum [freq, time].
    times : np.ndarray
        Time axis (seconds) corresponding to the model's time dimension.

    Raises
    ------
    ImportError
        If the `fitburst` library cannot be imported.
    FileNotFoundError
        If the `fitburst_json` file does not exist.
    KeyError or Exception
        If the JSON structure is unexpected or model computation fails.
    """
    if 'fb' not in sys.modules:
        raise ImportError("fitburst library is required for fitburst_model_to_ds.")

    # Load data from JSON file
    try:
        with open(fitburst_json, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Fitburst JSON file not found: {fitburst_json}")
    except json.JSONDecodeError:
        raise ValueError(f"Could not decode JSON from file: {fitburst_json}")

    # Extract necessary parameters
    try:
        params = data["model_parameters"]
        fit_stats = data["fit_statistics"]
        numtime = fit_stats['num_time']
        numfreq = fit_stats['num_freq']
        # Determine number of components (handle single vs multiple)
        amp = params["amplitude"]
        num_components = len(amp) if isinstance(amp, list) else 1
    except KeyError as e:
        raise KeyError(f"Missing expected key in fitburst JSON: {e}")

    # Create frequency and time axes for the modeler
    # Assumes standard CHIME frequency range
    freqs = np.linspace(const.FREQ_TOP_MHZ, const.FREQ_BOTTOM_MHZ, num=numfreq)
    # Time axis depends on number of time bins and original sample rate/downsampling
    native_bin_duration_s = 2.56e-6 # CHIME native time resolution
    model_bin_duration_s = native_bin_duration_s * downsamp
    times = np.linspace(0., numtime * model_bin_duration_s, num=numtime, endpoint=False)

    # Initialize fitburst modeler
    model_obj = fb.analysis.model.SpectrumModeler(
        freqs,
        times,
        # Assuming model DM is the incoherent one? Check fitburst usage.
        dm_incoherent = params["dm"][0] if isinstance(params["dm"], list) else params["dm"],
        # Upsampling factors (usually 1 if model is already at desired res)
        factor_freq_upsample = 1,
        factor_time_upsample = 1,
        # Assume model was computed on already dedispersed data?
        is_dedispersed = True,
        verbose = False,
        num_components = num_components,
    )

    # Update modeler with parameters from JSON
    # Need deepcopy as modeler might modify params? Original code used it.
    model_obj.update_parameters(deepcopy(params))

    # Compute the model dynamic spectrum
    model_ds = model_obj.compute_model() # Shape [numfreq, numtime]

    return model_ds, times


def convert_scatscin(value, scint=False, scatt=False):
    """
    Converts between scattering time (ms) and scintillation bandwidth (kHz)
    using the relationship: delta_nu_d = 1 / (2 * pi * tau_d).

    Parameters
    ----------
    value : float
        Input value (either scattering time in ms or scint bandwidth in kHz).
    scint : bool, optional
        If True, assumes input `value` is scintillation bandwidth (kHz) and
        returns scattering time (ms). Default False.
    scatt : bool, optional
        If True, assumes input `value` is scattering time (ms) and returns
        scintillation bandwidth (kHz). Default False.

    Returns
    -------
    float
        The converted value (scattering time in ms or scint bandwidth in kHz).

    Raises
    ------
    ValueError
        If neither or both `scint` and `scatt` are True.
    """
    if not scint and not scatt:
        raise ValueError("Specify input type: set either scint=True or scatt=True.")
    if scint and scatt:
        raise ValueError("Specify only one input type: set either scint=True or scatt=True.")

    if scint:
        # Input is scint bw in kHz, convert to Hz
        delta_nu_d_hz = value * 1000.0
        # Calculate tau_d in seconds
        tau_d_s = 1.0 / (2.0 * np.pi * delta_nu_d_hz)
        # Convert tau_d to ms
        return tau_d_s * 1000.0
    else: # scatt is True
        # Input is scattering time in ms, convert to s
        tau_d_s = value / 1000.0
        # Calculate delta_nu_d in Hz
        delta_nu_d_hz = 1.0 / (2.0 * np.pi * tau_d_s)
        # Convert delta_nu_d to kHz
        return delta_nu_d_hz / 1000.0


def get_event_info(event_id):
    """
    Retrieves basic event information (date, SNR, RA, Dec) from the
    CHIME FRB Master database for a given event ID.

    Parameters
    ----------
    event_id : int or str
        ID of the FRB event.

    Returns
    -------
    tuple
        (event_date, event_snr, event_ra, event_dec)
        event_date : list of str [year, month, day]
        event_snr : float
        event_ra : float (degrees)
        event_dec : float (degrees)

    Raises
    ------
    ConnectionError
        If the CHIME FRB Master API is not available.
    RuntimeError
        If event metadata cannot be fetched or parsed.
    """
    if master is None:
        raise ConnectionError("CHIME FRB Master API connection not established.")

    try:
        event = master.events.get_event(event_id)
    except Exception as e:
        raise RuntimeError(f"Could not fetch event {event_id} from API: {e}")

    event_date = None
    event_snr = None
    event_ra = None
    event_dec = None

    # Find parameters from the 'realtime' pipeline
    for par in event.get("measured_parameters", []):
        if par.get("pipeline", {}).get("name") == "realtime":
            try:
                event_date = par["datetime"].split(" ")[0].split("-")
                event_snr = float(par["snr"])
                event_ra = float(par["ra"])
                event_dec = float(par["dec"])
                # Found all info, break loop
                if len(event_date) == 3: break
                else: event_date = None # Reset if date format invalid
            except (KeyError, ValueError, TypeError, IndexError) as e:
                # Ignore if parameters are missing or have wrong type/format
                print(f"Warning: Could not parse realtime parameters for event {event_id}: {e}")
                event_date = event_snr = event_ra = event_dec = None # Reset on error

    if not all([event_date, event_snr is not None, event_ra is not None, event_dec is not None]):
        raise RuntimeError(f"Could not extract all required info (date, snr, ra, dec) "
                         f"from realtime parameters for event {event_id}.")

    return event_date, event_snr, event_ra, event_dec


def fit_spline(spec, num_splines=50, k=3):
    """
    Fits a smoothing spline to a 1D spectrum.

    Uses `scipy.interpolate.make_lsq_spline` to fit a spline with knots
    distributed across the valid data range.

    Parameters
    ----------
    spec : np.ma.MaskedArray
        1D input spectrum with potentially masked values.
    num_splines : int, optional
        Approximate number of spline segments (determines number of knots).
        Default is 50.
    k : int, optional
        Degree of the spline. Default is 3 (cubic spline).

    Returns
    -------
    spec_smooth : np.ma.MaskedArray
        Smoothed spectrum, with the same mask as the input `spec`.
    """
    if not isinstance(spec, np.ma.MaskedArray):
        # Convert to masked array if not already
        spec = np.ma.masked_invalid(spec) # Masks NaN and inf

    if spec.mask is np.ma.nomask or np.all(spec.mask):
        print("Warning: Spectrum is fully masked or has no mask. Cannot fit spline.")
        return spec # Return original or fully masked spec

    # Get indices and values of valid data points
    xs = np.arange(len(spec))
    valid_mask = ~spec.mask
    xs_valid = xs[valid_mask]
    ys_valid = spec.data[valid_mask]

    if len(xs_valid) <= k:
        print(f"Warning: Not enough valid points ({len(xs_valid)}) to fit spline of degree {k}.")
        return spec # Return original spec

    # Define knot locations
    # Place knots roughly evenly across the valid data range
    # Ensure first and last valid points are included for boundary conditions
    if num_splines >= len(xs_valid) - 2:
        # Reduce knots if too many requested for the number of points
        num_internal_knots = max(0, len(xs_valid) - k - 1)
        print(f"Warning: Reducing number of spline knots to {num_internal_knots+2}.")
    else:
         num_internal_knots = num_splines

    # Select internal knot locations from valid points (excluding endpoints)
    if num_internal_knots > 0:
        knot_indices = np.linspace(0, len(xs_valid) - 1, num_internal_knots + 2, dtype=int)[1:-1]
        internal_knots = xs_valid[knot_indices]
    else:
        internal_knots = np.array([])


    # Construct full knot vector with clamped ends (required by make_lsq_spline)
    # Repeat start/end points k+1 times
    knots = np.r_[(xs_valid[0],) * (k + 1),
                  internal_knots,
                  (xs_valid[-1],) * (k + 1)]

    # Fit the LSQ spline
    try:
        spline = make_lsq_spline(xs_valid, ys_valid, knots, k=k)
    except ValueError as e:
        print(f"Error fitting spline: {e}. Knot vector might be invalid.")
        # Common issue: knots not strictly increasing if data points coincide
        # Check knot vector:
        print("Knots:", knots)
        return spec # Return original spec on error


    # Evaluate the spline at all valid points
    ys_smooth = spline(xs_valid)

    # Create the output smoothed spectrum with the original mask
    spec_smooth = np.ma.masked_array(np.zeros_like(spec.data), mask=spec.mask)
    spec_smooth.data[valid_mask] = ys_smooth

    return spec_smooth


def acf_per_subband(spec, freqs, freqids, num_subbands=2, savefig='./acf_per_freq.pdf',
                    plot_fit=True, maxlag=None, snsubband=False, offspec=None):
    """
    Divides a spectrum into subbands and calculates the ACF for each.

    Note: This appears identical to `acf_per_subband` in `scint_functions.py`.
          Consider consolidating.

    Parameters and Returns: See `scint_functions.acf_per_subband` docstring.
    """
    # --- Implementation copied from scint_functions.py ---
    spec = np.ma.masked_where((spec == 0) | np.isnan(spec) | (spec.mask if hasattr(spec, 'mask') else False), spec, copy=True)
    if offspec is not None: offspec = np.ma.masked_where(spec.mask, offspec, copy=True)

    nchan = len(spec)
    mask_bool = ~spec.mask

    all_acfs, all_lags, all_fcents = [], [], []
    sub_sn, sub_mask_count, spec_lens = [], [], []
    sub_scint_results = [] # To store fit results if plot_fit is True

    total_valid_flux = np.sum(spec.data[mask_bool]) if snsubband else 0
    if snsubband and total_valid_flux <= 0: snsubband = False # Fallback

    flux_per_subband = total_valid_flux / float(num_subbands) if snsubband else 0
    start_idx = 0

    for sub in range(num_subbands):
        end_idx = nchan
        if snsubband:
            if sub == num_subbands - 1: end_idx = nchan
            else:
                current_flux = 0; idx = start_idx
                while current_flux < flux_per_subband and idx < nchan:
                    if mask_bool[idx]: current_flux += spec.data[idx]
                    idx += 1
                end_idx = idx
        else:
            end_idx = start_idx + (nchan // num_subbands)
            if sub == num_subbands - 1: end_idx = nchan

        end_idx = min(end_idx, nchan)
        if start_idx >= end_idx: continue

        spec_sub = spec[start_idx:end_idx]
        freqs_sub = freqs[start_idx:end_idx]
        freqids_sub = freqids[start_idx:end_idx]
        offspec_sub = offspec[start_idx:end_idx] if offspec is not None else None

        sub_len = len(spec_sub)
        spec_lens.append(sub_len)
        sub_mask_count.append(np.sum(spec_sub.mask))
        sub_sn.append(np.sum(spec_sub.data[~spec_sub.mask]))

        fcent_sub = freqs_sub[~spec_sub.mask].mean() if np.any(~spec_sub.mask) else np.nan
        all_fcents.append(fcent_sub)
        print(f"Subband {sub+1}: Freqs {freqs_sub[0]:.2f}-{freqs_sub[-1]:.2f} MHz (Cent: {fcent_sub:.2f} MHz)")

        lagrange_fit_sub = 5.0
        if maxlag is not None and maxlag < 10.0: lagrange_fit_sub = maxlag / 2.0

        # Use acf_scint_plot from this file (kenzie_functions)
        acf_result = acf_scint_plot(spec_sub, freqids_sub, freqs_sub, [0, 1],
                                    lagrange_for_fit=lagrange_fit_sub,
                                    diagnostic_plots=False,
                                    maxlag=maxlag,
                                    offspec_mean=np.ma.mean(offspec_sub) if offspec_sub is not None else None)

        if acf_result is None or acf_result[0] is None:
            all_acfs.append(np.array([np.nan])); all_lags.append(np.array([np.nan]))
            if plot_fit: sub_scint_results.append(None)
        else:
            all_acfs.append(acf_result[0]); all_lags.append(acf_result[1])
            if plot_fit: sub_scint_results.append(acf_result[2]) # Store lmfit result

        start_idx = end_idx

    # Plotting Overlaid ACFs
    if savefig is not None and len(all_acfs) > 0:
        plt.figure(figsize=(10, 8)); cmap = matplotlib.cm.get_cmap('plasma')
        max_val, min_val = 0, 0
        for i in range(len(all_fcents)):
            if len(all_acfs[i]) > 1:
                rgba = cmap(i / float(len(all_fcents)))
                offset = 1.0 * i
                plt.plot(all_lags[i], all_acfs[i] + offset, drawstyle='steps-mid', color=rgba, lw=1.5, alpha=0.8, label=f'{all_fcents[i]:.1f} MHz')
                max_val = max(max_val, np.nanmax(all_acfs[i] + offset))
                min_val = min(min_val, np.nanmin(all_acfs[i] + offset))
                # Plot fit if requested and available
                if plot_fit and sub_scint_results[i] is not None:
                     plt.plot(all_lags[i], sub_scint_results[i].eval(x=all_lags[i]) + offset, color='k', lw=0.8, alpha=0.7)

        plt.xlabel('Frequency Lag [MHz]'); plt.ylabel('ACF + Offset')
        plt.title(f'ACF per Subband ({num_subbands} subbands)')
        if maxlag: plt.xlim(-maxlag, maxlag)
        elif len(all_lags) > 0 and len(all_lags[0]) > 1: plt.xlim(all_lags[0][0], all_lags[0][-1])
        plt.ylim(min_val - 0.5, max_val + 0.5)
        plt.legend(loc='upper left', fontsize='small'); plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout(); plt.savefig(savefig, format='pdf'); plt.close()

    # Plot Scintillation Bandwidth vs Frequency
    if plot_fit and savefig is not None:
        sub_scint_bw = []
        sub_cent_valid = []
        for i, res in enumerate(sub_scint_results):
             if res is not None and 'gamma' in res.params:
                sub_scint_bw.append(np.abs(res.params['gamma'].value))
                sub_cent_valid.append(all_fcents[i])
             # else: # Append NaN or skip? Skipping for now.
             #      sub_scint_bw.append(np.nan)

        if len(sub_cent_valid) > 0:
            plt.figure()
            plt.scatter(sub_cent_valid, sub_scint_bw, marker='x', color='k', label='Fitted $\gamma$')
            # Optional: Fit power law to these points
            # try:
            #      scint_model = Model(scint_freq_relation)
            #      # Need errors for proper fitting
            #      fit_res_scint = scint_model.fit(sub_scint_bw, v=sub_cent_valid, c=0.1, n=4)
            #      plt.plot(freqs, fit_res_scint.eval(v=freqs), color='r', label=f'Fit $\\nu^{{{fit_res_scint.params["n"]:.1f}}}$')
            # except Exception as e: print(f"Could not fit power law to scint bw: {e}")
            plt.xlabel('Frequency [MHz]'); plt.ylabel('Scintillation Bandwidth [MHz]')
            plt.title('Scintillation Bandwidth vs Frequency'); plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend(); plt.tight_layout()
            plt.savefig(savefig.replace('.pdf', '_scintbw.pdf'), format='pdf'); plt.close()
        else:
             print("No valid scintillation bandwidths found from fits to plot.")


    return all_acfs, all_fcents, all_lags, sub_sn, sub_mask_count, spec_lens


def make_fitburst_mask(fitburst_json, fitburst_downsamp_factor, data_I):
    """
    Creates a time-frequency mask based on a fitburst model.

    Loads a fitburst model, interpolates it to the native time resolution
    of the data, aligns it in time with the data profile peak, and creates
    a binary mask where the model intensity is above 1% of its peak.

    Parameters
    ----------
    fitburst_json : str
        Path to the fitburst model JSON file.
    fitburst_downsamp_factor : int
        Time downsampling factor used when creating the fitburst model.
    data_I : np.ndarray or np.ma.MaskedArray
        Intensity dynamic spectrum of the burst [freq, time] at native
        time resolution (e.g., 2.56 us). Used for alignment.

    Returns
    -------
    mask : np.ndarray
        Binary mask [freq, time] with the same shape as `data_I`.
        Value is 1 where model is > 1% of peak, 0 otherwise.
        (Note: Original returned scaled model, this returns the mask itself).
    """
    # --- Load and Prepare Fitburst Model ---
    try:
        mod, times_mod = fitburst_model_to_ds(fitburst_json, downsamp=fitburst_downsamp_factor)
        # mod shape [freq, time_mod], times_mod is time axis for model
    except Exception as e:
        raise RuntimeError(f"Could not load or compute fitburst model: {e}")

    nfreq_data, ntime_data = data_I.shape
    nfreq_mod, ntime_mod = mod.shape

    if nfreq_data != nfreq_mod:
        # Attempt interpolation if frequency axes differ but cover same range
        print("Warning: Frequency dimension mismatch between data and model. Attempting interpolation.")
        freqs_data = np.linspace(FREQ_TOP_MHZ, FREQ_BOTTOM_MHZ, nfreq_data) # Assuming standard range
        freqs_mod = np.linspace(FREQ_TOP_MHZ, FREQ_BOTTOM_MHZ, nfreq_mod)
        # Need to implement 2D interpolation if this is required.
        # For now, raise error.
        raise ValueError("Frequency dimension mismatch requires interpolation (not implemented here).")


    # --- Interpolate Model to Native Time Resolution ---
    native_bin_duration_s = 2.56e-6
    data_time_axis = np.arange(ntime_data) * native_bin_duration_s

    # Check if model is already at native resolution
    model_bin_duration_s = times_mod[1] - times_mod[0] if len(times_mod) > 1 else (native_bin_duration_s * fitburst_downsamp_factor)
    if np.isclose(model_bin_duration_s, native_bin_duration_s):
        print("Model already at native time resolution.")
        mod_native = mod
        times_native = times_mod
    else:
        print(f"Interpolating fitburst model to {native_bin_duration_s*1e6:.2f} us resolution...")
        # Create interpolation function (linear should be sufficient)
        # interp2d expects x, y, z where x=time, y=freq
        freqs_mod_axis = np.linspace(0, nfreq_mod - 1, nfreq_mod) # Use index for y-axis
        try:
            interp_func = interp2d(times_mod, freqs_mod_axis, mod, kind='linear',
                                   bounds_error=False, fill_value=0.0)
            # Evaluate on the data's time axis and model's freq index axis
            mod_native = interp_func(data_time_axis, freqs_mod_axis)
            times_native = data_time_axis
        except Exception as e:
            raise RuntimeError(f"2D interpolation of fitburst model failed: {e}")

    # Ensure interpolated model has correct shape [nfreq, ntime_data]
    if mod_native.shape != (nfreq_data, ntime_data):
        # This might happen due to interp2d behavior or axis definitions
        print(f"Warning: Interpolated model shape {mod_native.shape} doesn't match data shape {(nfreq_data, ntime_data)}. Attempting reshape/transpose.")
        # Add checks and potential fixes here if needed. For now, assume it matches.
        pass


    # --- Align Model in Time ---
    print("Aligning model peak with data peak...")
    data_prof = np.nanmean(data_I, axis=0)
    model_prof = np.nanmean(mod_native, axis=0)

    peak_idx_data = np.nanargmax(data_prof)
    peak_idx_model = np.nanargmax(model_prof)

    time_shift = peak_idx_data - peak_idx_model
    print(f"Required time shift: {time_shift} bins.")

    mod_aligned = np.roll(mod_native, time_shift, axis=1)

    # --- Create Mask ---
    print("Creating mask (model > 1% of peak)...")
    peak_model_val = np.nanmax(mod_aligned)
    if np.isnan(peak_model_val) or peak_model_val <= 0:
        print("Warning: Model peak is zero or NaN. Cannot create mask.")
        mask = np.zeros_like(data_I, dtype=bool) # Return all-zero mask
    else:
        threshold = peak_model_val * 0.01
        mask = mod_aligned > threshold

    # --- Optional Plotting (from original) ---
    # plt.figure()
    # plt.plot(data_time_axis * 1000, data_prof, label='Data Profile')
    # # Scale aligned model for plotting comparison (using data offset seems odd here)
    # # offset = np.mean(data_prof[0:100]) # Baseline offset?
    # # scale_mod_plot = (mod_aligned / peak_model_val) * (np.nanmax(data_prof) - offset) + offset
    # scale_mod_plot = mod_aligned * (np.nanmax(data_prof) / peak_model_val) # Simple scaling
    # plt.plot(times_native * 1000, np.nanmean(scale_mod_plot, axis=0), label='Aligned Scaled Model Profile')
    # plt.xlabel("Time [ms]")
    # plt.ylabel("Intensity")
    # plt.legend()
    # plt.show()
    # --- End Plotting ---


    # Return the boolean mask (True where model > threshold)
    return mask.astype(bool)


def apply_fbmask_to_data(mask, data):
    """
    Applies a time-frequency mask to complex voltage data.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask [freq, time]. True indicates region to keep.
    data : np.ndarray or np.ma.MaskedArray
        Complex voltage data [freq, pol, time].

    Returns
    -------
    data_mod : np.ma.MaskedArray
        Masked complex voltage data. Original mask (if any) is combined
        with the new mask. Data outside the mask is set to zero and masked.
    """
    if data.ndim != 3:
        raise ValueError("Input data must be 3D [freq, pol, time].")
    if mask.ndim != 2:
        raise ValueError("Input mask must be 2D [freq, time].")
    if mask.shape != (data.shape[0], data.shape[2]):
        raise ValueError("Shape mismatch between mask [freq, time] and data [freq, pol, time].")

    # Expand mask to match data dimensions: [freq, time] -> [freq, 1, time]
    mask_3d = mask[:, np.newaxis, :]

    # Create output masked array
    if isinstance(data, np.ma.MaskedArray):
        # Combine new mask with existing mask
        combined_mask = np.logical_or(data.mask, ~mask_3d) # Mask where original OR ~new_mask
        data_mod = np.ma.masked_where(combined_mask, data.data)
    else:
        # Apply new mask
        data_mod = np.ma.masked_where(~mask_3d, data) # Mask where ~new_mask is True

    return data_mod

