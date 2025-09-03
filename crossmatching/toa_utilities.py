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

from flits.common.utils import (
    downsample_time,
    calculate_dm_timing_error,
    clean_and_serialize_dict,
    append_to_json,
)

# Assume these are defined elsewhere in your script
from baseband_analysis.core.bbdata import BBData
from baseband_analysis.core.dedispersion import delay_across_the_band
from baseband_analysis.core.bbdata import BBData
from baseband_analysis.analysis.snr import get_snr
from baseband_analysis.core.dedispersion import incoherent_dedisp, coherent_dedisp, get_freq

# Dispersion constant in MHz^2 pc^-1 cm^3 s

from numpy.typing import NDArray
import numpy as np


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


