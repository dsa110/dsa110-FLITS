import importlib
import numpy as np
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
import matplotlib.patches as patches

from matplotlib import rcParams
rcParams['mathtext.fontset'] = 'dejavuserif'
rcParams['mathtext.fallback'] = 'cm'
rcParams['font.serif'] = ['cmr10']
rcParams['font.size'] = 24
rcParams['axes.formatter.use_mathtext'] = True
rcParams['axes.unicode_minus'] = True
rcParams['mathtext.fontset'] = 'cm'
#rcParams['text.usetex'] = True

import astropy.units as u
import astropy.constants as const
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.modeling.models import Gaussian2D
from astropy.visualization import quantity_support
from astropy.visualization import wcsaxes
from astropy.wcs import WCS
from astropy.coordinates import AltAz
from astropy.coordinates import SkyOffsetFrame
import astropy.constants as const
from astropy.table import Table

# Assume these are defined elsewhere in your script
from baseband_analysis.core.bbdata import BBData
from baseband_analysis.core.dedispersion import delay_across_the_band
from baseband_analysis.core.bbdata import BBData
from baseband_analysis.analysis.snr import get_snr
from baseband_analysis.core.dedispersion import incoherent_dedisp, coherent_dedisp, get_freq

# Dispersion constant in MHz^2 pc^-1 cm^3 s

from numpy.typing import NDArray
import numpy as np

def downsample_time(data, t_factor):
    """
    Block-average by integer factor along the time axis.
    
    Works on either
      • 1D array of shape (ntime,)
      • 2D array of shape (nfreq, ntime)
    
    Parameters
    ----------
    data
        Input time series or spectrogram.
    t_factor
        Integer factor ≥1 by which to downsample time.
    
    Returns
    -------
    downsampled
        If input is 1D of length nt, returns 1D of length floor(nt/t_factor).
        If input is 2D (nf, nt), returns 2D of shape (nf, floor(nt/t_factor)).
    
    Raises
    ------
    ValueError
        If `t_factor < 1` or input is not 1D/2D.
    """
    if t_factor < 1:
        raise ValueError(f"t_factor must be ≥1, got {t_factor}")
    
    arr = np.asarray(data)
    
    # Handle 1D time series
    if arr.ndim == 1:
        nt = arr.shape[0]
        nt_trim = nt - (nt % t_factor)
        # reshape into (ntime_out, t_factor) then average
        return arr[:nt_trim].reshape(nt_trim // t_factor, t_factor).mean(axis=1)
    
    # Handle 2D spectrogram-like input
    elif arr.ndim == 2:
        nfreq, nt = arr.shape
        nt_trim = nt - (nt % t_factor)
        # reshape into (nfreq, ntime_out, t_factor) then average over last axis
        blocks = arr[:, :nt_trim].reshape(nfreq, nt_trim // t_factor, t_factor)
        return blocks.mean(axis=2)
    
    else:
        raise ValueError(
            f"Unsupported array shape {arr.shape}; expected 1D or 2D."
        )


def measure_fwhm(timeseries, time_resolution, t_factor):
    """
    Measures the Full Width at Half Maximum (FWHM) of a pulse.

    This function assumes the timeseries has had its baseline subtracted
    (i.e., the noise level is around zero).

    Parameters
    ----------
    timeseries : np.ndarray
        A 1D array representing the time series of the pulse.
    time_resolution : float
        The time duration of a single bin/sample in the timeseries (e.g., in ms).

    Returns
    -------
    float
        The FWHM of the pulse in the same units as time_resolution.
        Returns np.nan if the FWHM cannot be determined.
    """
    try:
        # Downsample the timeseries
        timeseries = downsample_time(timeseries, t_factor = t_factor)
        time_resoution = time_resolution * t_factor
        
        # Find the peak value and its index
        peak_val = np.max(timeseries)
        peak_idx = np.argmax(timeseries)

        # Calculate the half maximum value
        half_max = peak_val / 2.5

        # Find all indices where the timeseries is greater than half max
        above_indices = np.where(timeseries > half_max)[0]

        # --- Edge Case Checks ---
        if not above_indices.any():
            # Pulse never crosses the half-maximum level
            return np.nan
        
        # Check if pulse is truncated at the start or end of the window
        if above_indices[0] == 0 or above_indices[-1] == len(timeseries) - 1:
            print("Warning: Pulse may be truncated. FWHM could be inaccurate.")
            # Fallback to the simple method for truncated pulses
            width_in_bins = len(above_indices)
            return width_in_bins * time_resolution

        # --- Interpolate Rising Edge ---
        # Find the point on the curve just before and after crossing the half-max line
        idx_after_rise = above_indices[0]
        idx_before_rise = idx_after_rise - 1
        val_before_rise = timeseries[idx_before_rise]
        val_after_rise = timeseries[idx_after_rise]
        
        # Linearly interpolate to find the precise time (in bins) of the crossing
        t_rise = idx_before_rise + (half_max - val_before_rise) / (val_after_rise - val_before_rise)

        # --- Interpolate Falling Edge ---
        idx_before_fall = above_indices[-1]
        idx_after_fall = idx_before_fall + 1
        val_before_fall = timeseries[idx_before_fall]
        val_after_fall = timeseries[idx_after_fall]
        
        t_fall = idx_before_fall + (half_max - val_before_fall) / (val_after_fall - val_before_fall)

        # Calculate the width in number of bins
        width_in_bins = t_fall - t_rise

        # Convert width to time units
        fwhm = width_in_bins * time_resolution
        
        return fwhm

    except IndexError:
        # This can happen if the pulse is right at the edge (e.g., peak is the last bin).
        print("Could not measure FWHM: Pulse is at the edge of the time window.")
        return np.nan
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred during FWHM measurement: {e}")
        return np.nan

def calculate_dm_timing_error(dDM, f_obs, f_ref, K_DM = 4.148808e3):
    """
    Calculates the timing error due to DM uncertainty.

    Parameters
    ----------
    dDM : float
        The uncertainty in the Dispersion Measure (pc/cm^3).
    f_obs : astropy.units.Quantity
        The central observing frequency in MHz.
    f_ref : astropy.units.Quantity
        The reference frequency in MHz.

    Returns
    -------
    astropy.units.Quantity
        The timing error in milliseconds.
    """
    # Calculate the time shift in seconds
    time_shift = K_DM * dDM * (1 / f_obs.value**2 - 1 / f_ref.value**2) * u.s
    
    # Return the absolute value in milliseconds
    return np.abs(time_shift.to(u.ms))

def clean_and_serialize_dict(burst_dict):
    """
    Converts a dictionary containing astropy objects into a
    JSON-serializable dictionary.
    """
    clean_dict = {}
    for key, value in burst_dict.items():
        if isinstance(value, u.Quantity):
            clean_dict[key] = value.value
        elif isinstance(value, Time):
            clean_dict[key] = value.iso
        else:
            clean_dict[key] = value
    return clean_dict

def append_to_json(new_data_dict, filename):
    """
    Reads a JSON file containing a list of dictionaries, appends a new
    dictionary to the list, and writes it back to the file.

    Parameters
    ----------
    new_data_dict : dict
        The new dictionary to append. It can contain astropy objects.
    filename : str
        The path to the JSON file.
    """
    # First, clean the new data to make it serializable
    clean_new_data = clean_and_serialize_dict(new_data_dict)
    
    # Check if the file exists and read its content
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        try:
            with open(filename, 'r') as f:
                data_list = json.load(f)
            # Ensure the loaded data is a list
            if not isinstance(data_list, list):
                print(f"Error: JSON file '{filename}' does not contain a list.")
                # Start with a new list containing the new data
                data_list = [clean_new_data]
            else:
                # Append the new dictionary
                data_list.append(clean_new_data)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from '{filename}'. Starting a new file.")
            data_list = [clean_new_data]
    else:
        # If the file doesn't exist or is empty, start a new list
        data_list = [clean_new_data]

    # Write the updated list back to the file
    with open(filename, 'w') as f:
        json.dump(data_list, f, indent=4)
        
    print(f"Successfully appended data to {filename}")

