# -*- coding: utf-8 -*-
"""
FRB scintillation analysis functions.
"""

import sys
import os
import json
import math
from copy import deepcopy

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal
from scipy.stats import median_abs_deviation
from scipy.interpolate import make_lsq_spline, interp2d
from scipy.fft import fft, fftshift
from lmfit import Model, Minimizer, Parameters, fit_report, report_fit

# Optional Imports (handle gracefully if not found)
try:
    # Import necessary functions from baseband_analysis
    from baseband_analysis.core.signal import get_main_peak_lim, tiedbeam_baseband_to_power, get_spectrum_lim
    from baseband_analysis.core.bbdata import BBData
    from baseband_analysis.analysis.snr import get_snr, get_profile
    from baseband_analysis.core.sampling import scrunch
    from baseband_analysis.core.dedispersion import coherent_dedisp, incoherent_dedisp # Keep both for potential future use/reference
    from baseband_analysis.analysis.polarization import get_burst_envelope
    BASEBAND_ANALYSIS_AVAILABLE = True
except ImportError:
    print("Warning: 'baseband_analysis' package not found. Some functionality will be limited.")
    BASEBAND_ANALYSIS_AVAILABLE = False
    # Define dummy classes/functions if needed for script loading
    class BBData: pass
    def get_main_peak_lim(*args, **kwargs): return [0, 100] # Dummy
    def get_spectrum_lim(*args, **kwargs): return [0, 1024] # Dummy
    def tiedbeam_baseband_to_power(*args, **kwargs): pass
    def get_snr(*args, **kwargs): return (0, 0, 0, None, None, np.ones(1024, dtype=bool), [0, 1000]) # Dummy
    def scrunch(wfall, tscrunch, fscrunch): # Basic scrunch needed internally
        if wfall.ndim != 2: raise ValueError("Dummy scrunch needs 2D input.")
        nchan, nbins = wfall.shape
        if tscrunch > 1:
            remainder_t = nbins % tscrunch
            if remainder_t != 0: wfall = wfall[:, : nbins - remainder_t]
            wfall = np.nanmean(wfall.reshape(nchan, nbins // tscrunch, tscrunch), axis=2)
        if fscrunch > 1:
            remainder_f = nchan % fscrunch
            if remainder_f != 0: raise ValueError("Dummy scrunch chan mismatch.")
            if wfall.shape[1] == 0: return np.array([]) # Handle empty time axis after tscrunch
            wfall = np.nanmean(wfall.reshape(nchan // fscrunch, fscrunch, wfall.shape[1]), axis=1)
        return wfall
    def coherent_dedisp(*args, **kwargs): pass # Dummy implementation
    def incoherent_dedisp(*args, **kwargs): return (np.zeros((1024, 2, 100)), np.linspace(400,800,1024), np.arange(1024)) # Dummy
    def get_burst_envelope(*args, **kwargs): return [10, 90] # Dummy

try:
    # Import CHIME API and constants if available
    import chime_frb_api
    import chime_frb_constants as const
    CHIME_API_AVAILABLE = True
except ImportError:
    print("Warning: 'chime_frb_api' or 'chime_frb_constants' not found. API/Constant features disabled.")
    CHIME_API_AVAILABLE = False
    # Define dummy const if needed for basic operation
    class const: FREQ_TOP_MHZ = 800.1953125; FREQ_BOTTOM_MHZ = 400.1953125

try:
    # Import fitburst if available
    import fitburst as fb
    FITBURST_AVAILABLE = True
except ImportError:
    print("Warning: 'fitburst' package not found. Fitburst model features disabled.")
    FITBURST_AVAILABLE = False

# --- Constants ---
FREQ_TOP_MHZ = getattr(const, 'FREQ_TOP_MHZ', 800.1953125)
FREQ_BOTTOM_MHZ = getattr(const, 'FREQ_BOTTOM_MHZ', 400.1953125)
TOTAL_CHANNELS = 1024
NATIVE_BIN_DURATION_S = 2.56e-6

# --- Helper Functions (Low-level utilities, models) ---

def _shift(v, i, nchan):
    """
    Helper function to circularly shift a 1D array `v` by `i` positions,
    padding with zeros for ACF calculation. (Internal use)

    Parameters
    ----------
    v : np.ndarray
        1D array to shift.
    i : int
        Shift amount (can be positive or negative).
    nchan : int
        Original number of channels (length of `v`), used for calculating
        the effective shift index within the padded array.

    Returns
    -------
    r : np.ndarray
        Zero-padded array of size 3*n containing `v` shifted by `i`.
    """
    n = len(v)
    r = np.zeros(3 * n, dtype=v.dtype) # Match dtype
    start_index = int(i + nchan - 1)
    end_index = start_index + n
    v_start, v_end = 0, n
    # Adjust indices for wrapping/padding
    if start_index < 0: v_start = -start_index; start_index = 0
    if end_index > 3 * n: v_end = n - (end_index - 3 * n); end_index = 3 * n
    # Copy only if valid range exists
    if start_index < end_index and v_start < v_end:
        r[start_index:end_index] = v[v_start:v_end]
    return r

# --- Model Definitions ---
def lorentz_w_c(x, gamma1, m1, c):
    """ Lorentzian function with constant offset 'c'. """
    # Ensure gamma is not zero to avoid division error
    gamma1_safe = np.where(gamma1 == 0, 1e-9, gamma1)
    return m1**2 / (1 + (x / gamma1_safe)**2) + c

def doublelorentz_w_c(x, gamma1, m1, gamma2, m2, c):
    """ Sum of two Lorentzian functions with a shared constant offset 'c'. """
    gamma1_safe = np.where(gamma1 == 0, 1e-9, gamma1)
    gamma2_safe = np.where(gamma2 == 0, 1e-9, gamma2)
    return (m1**2 / (1 + (x / gamma1_safe)**2)) + (m2**2 / (1 + (x / gamma2_safe)**2)) + c

def triplelorentz(x, gamma1, m1, gamma2, m2, gamma3, m3):
    """ Sum of three Lorentzian functions without constant offset. """
    gamma1_safe = np.where(gamma1 == 0, 1e-9, gamma1)
    gamma2_safe = np.where(gamma2 == 0, 1e-9, gamma2)
    gamma3_safe = np.where(gamma3 == 0, 1e-9, gamma3)
    return (m1**2 / (1 + (x / gamma1_safe)**2)) + \
           (m2**2 / (1 + (x / gamma2_safe)**2)) + \
           (m3**2 / (1 + (x / gamma3_safe)**2))

def gaus(x, a, x0, sigma, c):
    """ Gaussian function definition. """
    # Ensure sigma is not zero
    sigma_safe = np.where(sigma == 0, 1e-9, sigma)
    return a * np.exp(-(x - x0)**2 / (2 * sigma_safe**2)) + c

def scint_freq_relation(v, c, n):
    """ Power-law model for scintillation bandwidth: bw = c * v^n. """
    # Ensure base v is positive for potentially non-integer exponent n
    v_safe = np.maximum(v, 1e-9)
    return c * (v_safe**n)

def lin(x, grad, c):
    """ Linear model: y = grad * x + c. """
    return grad * x + c

# --- Minimizer Functions for lmfit ---
# These functions define the residual (data - model) / error

def lorentz_withc_min(params, x, y, err):
    """ Residual function for lorentz_w_c model. """
    # Access parameter values safely
    gamma1 = params['gamma1'].value
    m1 = params['m1'].value
    c = params['c'].value
    model = lorentz_w_c(x, gamma1, m1, c)
    # Avoid division by zero error if err is zero
    err_safe = np.where(err == 0, 1.0, err)
    return (model - y) / err_safe

def doublelorentz_withc_min(params, x, y, err):
    """ Residual function for doublelorentz_w_c model. """
    gamma1 = params['gamma1'].value; m1 = params['m1'].value
    gamma2 = params['gamma2'].value; m2 = params['m2'].value
    c = params['c'].value
    model = doublelorentz_w_c(x, gamma1, m1, gamma2, m2, c)
    err_safe = np.where(err == 0, 1.0, err)
    return (model - y) / err_safe

def triplelorentz_min(params, x, y, err):
    """ Residual function for triplelorentz model (no offset 'c'). """
    gamma1 = params['gamma1'].value; m1 = params['m1'].value
    gamma2 = params['gamma2'].value; m2 = params['m2'].value
    gamma3 = params['gamma3'].value; m3 = params['m3'].value
    model = triplelorentz(x, gamma1, m1, gamma2, m2, gamma3, m3)
    err_safe = np.where(err == 0, 1.0, err)
    return (model - y) / err_safe

def scint_freq_relation_min(params, x, y, err):
    """ Residual function for scint_freq_relation model. """
    c = params['c'].value; n = params['n'].value
    model = scint_freq_relation(x, c, n)
    err_safe = np.where(err == 0, 1.0, err)
    return (model - y) / err_safe

def linmin(params, x, y, errs):
    """ Residual function for linear model. """
    grad = params['grad'].value; c = params['c'].value
    model = lin(x, grad, c)
    errs_safe = np.where(errs == 0, 1.0, errs)
    return (model - y) / errs_safe

# --- Physical Parameter Estimations ---
def res(lens_dist, lda, scat_lens):
    """
    Estimates the physical resolution of a thin scattering screen.

    Parameters
    ----------
    lens_dist : float
        Distance between source and scattering lens [kpc].
    lda : float
        Wavelength of observation [m].
    scat_lens : float
        Scattering timescale imparted by the screen [ms].

    Returns
    -------
    float
        Physical resolution of the lens [km]. Returns NaN if inputs are invalid.
    """
    if lens_dist <= 0 or lda <= 0 or scat_lens <= 0: return np.nan
    # Use scipy constants for physical values
    parsec_m = getattr(cons, 'parsec', 3.085677581491367e16) # Use default if const not avail
    c_light = getattr(cons, 'c', 299792458.0)
    lens_dist_m = lens_dist * parsec_m * 1000.0
    scat_lens_s = scat_lens / 1000.0
    # Avoid division by zero or sqrt of negative
    term_under_sqrt = lens_dist_m / (4.0 * c_light * scat_lens_s)
    if term_under_sqrt < 0: return np.nan
    resolution_m = (lda / np.pi) * np.sqrt(term_under_sqrt)
    return resolution_m / 1000.0 # Convert m to km

def emission_size(phys_res, mod_ind):
    """
    Estimates the physical size of the emission region based on lens
    resolution and scintillation modulation index.

    Assumes Gaussian source brightness distribution and relates modulation
    index to the ratio of source size to lens resolution.

    Parameters
    ----------
    phys_res : float
        Physical resolution of the lens [km] (e.g., from `res` function).
    mod_ind : float
        Modulation index of scintillation (e.g., sqrt(ACF[0])).
        Should be > 0 and ideally <= 1.

    Returns
    -------
    float
        Estimated physical size (sigma of Gaussian) of the emission region [km].
        Returns NaN if inputs are invalid or mod_ind > 1.
    """
    if phys_res <= 0 or mod_ind <= 0: return np.nan
    # Avoid division by zero if mod_ind is very small
    mod_ind_safe = max(mod_ind, 1e-9)
    term_inside_sqrt = (1.0 / mod_ind_safe**2) - 1.0
    if term_inside_sqrt < 0: return np.nan # Happens if mod_ind > 1
    return phys_res * np.sqrt(term_inside_sqrt)

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
        Returns np.inf if conversion involves division by zero.

    Raises
    ------
    ValueError
        If neither or both `scint` and `scatt` are True.
    """
    if not scint and not scatt: raise ValueError("Specify input type: scint=True or scatt=True.")
    if scint and scatt: raise ValueError("Specify only one input type.")
    if scint: # Input is scint bw [kHz] -> output scat time [ms]
        delta_nu_d_hz = value * 1000.0
        if delta_nu_d_hz == 0: return np.inf # Avoid division by zero
        tau_d_s = 1.0 / (2.0 * np.pi * delta_nu_d_hz)
        return tau_d_s * 1000.0
    else: # Input is scat time [ms] -> output scint bw [kHz]
        tau_d_s = value / 1000.0
        if tau_d_s == 0: return np.inf # Avoid division by zero
        delta_nu_d_hz = 1.0 / (2.0 * np.pi * tau_d_s)
        return delta_nu_d_hz / 1000.0

# --- Main Processing Class ---

class ScintillationProcessor:
    """
    Class to manage the scintillation analysis pipeline for an FRB event.

    Encapsulates data loading, preprocessing, upchannelization, normalization,
    ACF calculation, and subband analysis steps.

    Attributes
    ----------
    event_id : str
        FRB event identifier.
    dm : float
        Dispersion measure (pc cm^-3).
    baseband_file : str or None
        Path to the input baseband file.
    output_dir : str or None
        Directory for saving results.
    bbdata : BBData object or None
        Loaded baseband data object.
    raw_data, freqs, freq_ids : np.ndarray or None
        Raw data and corresponding frequency info.
    processed_data, processed_freqs, processed_freq_ids : np.ndarray or None
        Data after preprocessing (dedispersion, masking, slicing).
    upchan_data, upchan_freqs, upchan_freq_ids : np.ndarray or None
        Data after upchannelization.
    upchan_fftsize, upchan_downfreq : int or None
        Parameters used for upchannelization.
    scallop_model, scallop_rfi_inds : np.ndarray or None
        Calculated scallop model and flagged RFI channel indices.
    spec_on, spec_peak, spec_off : np.ma.MaskedArray or None
        Calculated normalized spectra.
    acf_on, acf_peak, acf_lags_mhz : np.ndarray or None
        Calculated ACFs and corresponding lags.
    subband_results : dict
        Dictionary storing results from subband analysis.
    master_api : FRBMaster object or None
        Connection to the CHIME FRB Master API.
    auth : dict or None
        Authorization token for CHIME API.
    """
    # Class attributes for default values or constants if needed
    DEFAULT_FFTSIZE = 32
    DEFAULT_DOWNFREQ = 1
    SET_DM = 'Not Set'

    def __init__(self, event_id, dm, baseband_file=None, output_dir=None):
        """
        Initialize the processor for a specific event.

        Parameters
        ----------
        event_id : int or str
            FRB event identifier.
        dm : float
            Dispersion measure (pc cm^-3).
        baseband_file : str, optional
            Direct path to the baseband HDF5 file. If None, attempts to
            find it using CHIME API and event_id. Default None.
        output_dir : str, optional
            Directory to save plots and results. If None, plots are shown
            interactively or not saved. Default None.
        """
        # Input validation
        if not isinstance(event_id, (int, str)) or not event_id:
             raise ValueError("Event ID must be a non-empty string or integer.")
        if not isinstance(dm, (int, float)):
             raise ValueError("DM must be a number.")

        self.event_id = str(event_id) # Ensure string representation
        self.dm = float(dm)
        self.dm_set = False
        self.baseband_file = baseband_file
        self.output_dir = output_dir
        if self.output_dir:
            try:
                os.makedirs(self.output_dir, exist_ok=True)
            except OSError as e:
                print(f"Warning: Could not create output directory {self.output_dir}: {e}")
                self.output_dir = None # Disable saving if dir creation fails

        self.master_api = None
        self.auth = None
        self._connect_chime_api() # Attempt connection

        # Data attributes initialized to None
        self.bbdata = None
        self.raw_data = None
        self.freqs = None
        self.freq_ids = None
        self.processed_data = None
        self.processed_freqs = None
        self.processed_freq_ids = None
        self.upchan_data = None
        self.upchan_freqs = None
        self.upchan_freq_ids = None
        self.upchan_fftsize = None
        self.upchan_downfreq = None
        self.scallop_model = None
        self.scallop_rfi_inds = None
        self.spec_on = None
        self.spec_peak = None
        self.spec_off = None
        self.acf_on = None
        self.acf_peak = None
        self.acf_lags_mhz = None
        self.subband_results = {}

        print(f"Initialized ScintillationProcessor for Event {self.event_id}, DM {self.dm}")

    def _connect_chime_api(self):
        """ Attempt to connect to CHIME/FRB Master API. """
        if CHIME_API_AVAILABLE:
            # Check if already connected
            if self.master_api is not None: return
            try:
                # Add timeout? Retry logic?
                self.master_api = chime_frb_api.frb_master.FRBMaster(base_url="https://frb.chimenet.ca/frb-master")
                self.master_api.API.authorize()
                self.auth = {"Authorization": self.master_api.API.access_token}
                print("Successfully connected and authorized CHIME FRB Master API.")
            except Exception as e:
                print(f"Warning: Could not connect/authorize CHIME FRB Master API: {e}")
                self.master_api = None
                self.auth = None
        else:
            print("CHIME FRB API libraries not available.")

    def _get_event_metadata(self):
        """ Fetch event metadata using the CHIME/FRB Master API. """
        if self.master_api is None:
            # Attempt to connect if not already connected
            self._connect_chime_api()
            if self.master_api is None:
                 raise ConnectionError("CHIME FRB Master API connection failed or not available.")
        try:
            print(f"Fetching metadata for event {self.event_id}...")
            event_data = self.master_api.events.get_event(self.event_id)
            return event_data
        except Exception as e:
            raise RuntimeError(f"Could not fetch event metadata for {self.event_id}: {e}")

    def _construct_data_path(self):
        """ Construct data path if not provided directly. """
        if self.baseband_file:
            print(f"Using provided baseband file: {self.baseband_file}")
            if not os.path.exists(self.baseband_file):
                 print(f"Warning: Provided file does not exist: {self.baseband_file}")
            return self.baseband_file

        print("Constructing data path using CHIME API metadata...")
        event_meta = self._get_event_metadata()
        event_date = None
        # Look for date in realtime parameters
        for par in event_meta.get("measured_parameters", []):
            if par.get("pipeline", {}).get("name") == "realtime":
                dt_str = par.get("datetime", "")
                if dt_str:
                    try:
                        date_parts = dt_str.split(" ")[0].split("-")
                        if len(date_parts) == 3: event_date = date_parts; break
                    except Exception: pass # Ignore parsing errors
        if not event_date:
            raise ValueError(f"Could not find valid event date in 'realtime' parameters for event {self.event_id}.")

        # Hardcoded path structure - **MODIFY IF NEEDED**
        # Consider making this configurable or adding checks
        data_path = (
            f"/arc/projects/chime_frb/data/chime/baseband/processed/"
            f"{event_date[0]}/{event_date[1]}/{event_date[2]}/astro_"
            f"{event_meta['id']}/singlebeam_{event_meta['id']}.h5"
        )
        print(f"Constructed data path: {data_path}")
        if not os.path.exists(data_path):
             print(f"Warning: Constructed data path does not exist.")
        self.baseband_file = data_path # Store constructed path
        return data_path

    def load_data(self):
        """
        Loads baseband data from the specified file path using BBData.

        Populates `self.bbdata`, `self.raw_data`, `self.freqs`, `self.freq_ids`.

        Raises
        ------
        ImportError
            If `baseband_analysis` package is not available.
        IOError
            If the file cannot be found or loaded.
        KeyError
            If essential keys ('tiedbeam_baseband', 'freq') are missing.
        """
        if not BASEBAND_ANALYSIS_AVAILABLE:
            raise ImportError("Cannot load data: 'baseband_analysis' package not available.")
        if self.bbdata is not None:
            print("Data already loaded.")
            return

        filepath = self._construct_data_path()
        try:
            print(f"Loading BBData from: {filepath}")
            self.bbdata = BBData.from_file(filepath)

            # Store raw data and frequency info immediately after loading
            baseband_key = 'tiedbeam_baseband'
            if baseband_key not in self.bbdata.keys():
                raise KeyError(f"'{baseband_key}' key not found in loaded BBData object from {filepath}.")
            
            self.raw_data = self.bbdata[baseband_key][:] # Make a copy? Or view? Using view for now.
            self.freqs = self.bbdata.index_map['freq']['centre']
            self.freq_ids = self.bbdata.index_map['freq']['id']
            print(f"Raw data loaded. Shape: {self.raw_data.shape}")

        except KeyError as e:
             raise IOError(f"Failed to load BBData or find required key '{e}' from {filepath}.")
        except FileNotFoundError:
             raise IOError(f"Baseband file not found at {filepath}.")
        except Exception as e:
            # Catch other potential loading errors (HDF5 issues, etc.)
            raise IOError(f"Failed to load BBData from {filepath}: {e}")

    def preprocess_data(self, downsample_factor=32, interactive=False,
                        select_off_burst=False, time_range_ds=None, zap_extra=True,
                        spec_lims=None, min_duration_native=None):
        """
        Performs preprocessing: SNR, dedispersion, masking, time windowing, filling.

        Applies coherent dedispersion in-place to `bbdata['tiedbeam_baseband']`.
        Determines valid channels and time ranges, applies masks, selects the
        desired on-burst or off-burst window, fills missing frequency channels
        up to 1024, performs optional extra RFI flagging, and applies frequency limits.

        Parameters
        ----------
        downsample_factor : int, optional
            Time downsampling for SNR calc and time window selection. Default 32.
        interactive : bool, optional
            Prompt user for time window if True (not recommended in scripts). Default False.
        select_off_burst : bool, optional
            Select off-burst window if True. Default False.
        time_range_ds : tuple, optional
            Specify time window [start, end] in downsampled bins. Overrides interactive/auto.
        zap_extra : bool, optional
            Perform extra RFI flagging based on channel power. Default True.
        spec_lims : tuple, optional
            Specify frequency channel limits [start_chan, end_chan] (absolute 0-1023 indices)
            to keep. If None, attempts to determine automatically using `get_spectrum_lim`.
            Default None.
        min_duration_native : int, optional
            Minimum required duration (in native time bins) for the selected
            time window. Primarily used for ensuring sufficient off-burst data
            for scallop model generation. Default None.

        Returns
        -------
        np.ma.MaskedArray
            Processed data [freq_lim, pol, time_window]. Also stored in `self.processed_data`.
            `self.processed_freqs` and `self.processed_freq_ids` are also populated.

        Raises
        ------
        ImportError
            If `baseband_analysis` is not available.
        RuntimeError
            If critical steps like dedispersion fail.
        ValueError
            If time/frequency ranges become invalid or empty.
        """
        if self.bbdata is None:
            self.load_data() # Ensure data is loaded
        if not BASEBAND_ANALYSIS_AVAILABLE:
             raise ImportError("Cannot preprocess: 'baseband_analysis' not available.")

        print("\n--- Starting Data Preprocessing ---")
        bbdata = self.bbdata # Local reference
        baseband_key = 'tiedbeam_baseband'

        # 1. Initial SNR and Power Calculation
        # Calculate power if not present, useful for SNR and envelope finding
        if "tiedbeam_power" not in bbdata.keys():
            print("Calculating tiedbeam power...")
            try:
                # Calculate power at native resolution before dedispersion for SNR
                tiedbeam_baseband_to_power(bbdata, time_downsample_factor=1, dm=self.dm,
                                           dedisperse=True, time_shift=False)
            except Exception as e: print(f"Warning: tiedbeam_power calculation failed: {e}")

        print(f"Calculating SNR (downsample={downsample_factor})...")
        try:
            snr_results = get_snr(bbdata, DM=self.dm, diagnostic_plots=False,
                                  return_full=True, downsample=downsample_factor)
            valid_channels_mask = snr_results[5]
            valid_time_bins_native = snr_results[6]
            print(f"Initial valid time range (native bins): {valid_time_bins_native}")
            print(f"Number of initial valid channels: {np.sum(valid_channels_mask)}")
        except Exception as e:
            print(f"Warning: get_snr failed: {e}. Using defaults.")
            if baseband_key not in bbdata.keys(): raise KeyError("Cannot find baseband data for fallback.")
            nchan_bb = bbdata[baseband_key].shape[0]; nsamp_bb = bbdata[baseband_key].shape[2]
            valid_channels_mask = np.ones(nchan_bb, dtype=bool); valid_time_bins_native = [0, nsamp_bb]
        
        
        data_dedisp = bbdata[baseband_key] # This is now the coherently dedispersed complex data
    
        # 2. Dedispersion
        print(f"Checking dedispersion status (Target DM={self.dm})...")
        current_dm = ScintillationProcessor.SET_DM # Get current DM, default to 0 if not set
        print(f"Current DM attribute found: {current_dm}")

        # Check if dedispersion is needed (target DM is different from current DM)
        # Use a tolerance for floating point comparison
        if self.dm != 0 and isinstance(current_dm, str):
            print(f"Current DM {current_dm} and self.dm {self.dm} don't agree within 1 pc cm-3")
            print(f"Applying coherent dedispersion (DM={self.dm})...")
            try:
                # coherent_dedisp modifies bbdata[baseband_key] in place when write=True
                coherent_dedisp(bbdata, self.dm, time_shift=False, write=True)
                print(f"Coherent dedispersion applied in-place to '{baseband_key}'.")
                data_dedisp, freq, freq_id = incoherent_dedisp(bbdata, self.dm, fill_wfall=False)
                print(f"Incoherent dedispersion applied.")
                ScintillationProcessor.SET_DM = self.dm
                print(f"DM reset to {ScintillationProcessor.SET_DM} pc cm-3")
            except Exception as e:
                raise RuntimeError(f"Coherent dedispersion failed: {e}")
        elif self.dm == 0 and current_dm != 0:
             # This case might require 're-dispersing' or loading original data if needed.
             # For now, assume if target is 0, we use whatever is currently in bbdata.
             print("Target DM is 0, using current data (which might be dispersed).")
        else:
            print(f"Data already dedispersed to target DM ({self.dm}). Skipping coherent dedispersion.")

        # Access the (potentially modified) baseband data
        if baseband_key not in bbdata.keys():
             raise KeyError(f"Cannot find '{baseband_key}' data after dedispersion step.")

        # 3. Apply Initial Masks (Channels and Time)
        print("Applying initial channel and time masks...")
        if len(valid_channels_mask) != data_dedisp.shape[0]:
            print(f"Warning: Channel mask length ({len(valid_channels_mask)}) mismatch with data "
                  f"({data_dedisp.shape[0]}). Using full mask.")
            valid_channels_mask = np.ones(data_dedisp.shape[0], dtype=bool)

        # Create masked array or update mask if already masked
        if isinstance(data_dedisp, np.ma.MaskedArray):
            # If data_dedisp is already masked (e.g., from loading), combine masks
            data_masked_tmp = data_dedisp
            print("Input data is already masked. Combining masks.")
        else:
            # Create a new masked array if input is plain numpy array
            data_masked_tmp = np.ma.masked_array(data_dedisp, mask=False)

        # Apply channel mask (mask=True where invalid)
        # Ensure mask is broadcastable to data shape [freq, pol, time]
        channel_mask_3d = ~valid_channels_mask[:, np.newaxis, np.newaxis]
        data_masked_tmp.mask = np.logical_or(data_masked_tmp.mask, channel_mask_3d)

        # Apply time mask (trim data based on SNR results)
        t_start = max(0, int(valid_time_bins_native[0]))
        t_end = min(data_masked_tmp.shape[-1], int(valid_time_bins_native[1])) # Use shape[-1] for time axis
        if t_start >= t_end: raise ValueError("Initial valid time range determined by SNR is empty.")
        # Slice the data and the mask simultaneously
        data_masked_tmp = data_masked_tmp[..., t_start:t_end] # Ellipsis for freq, pol
        print(f"Data trimmed by SNR time limits to native bins: [{t_start}, {t_end}]")

        # 4. Determine Analysis Time Window (On/Off Burst Selection)
        print("Determining analysis time window (on/off burst selection)...")
        power_trimmed = np.abs(data_masked_tmp)**2
        pol_axis = 1 if data_masked_tmp.ndim == 3 else None
        if pol_axis is not None: I_trimmed = np.ma.sum(power_trimmed, axis=pol_axis)
        else: I_trimmed = power_trimmed
        if I_trimmed.ndim != 2: raise ValueError("Intensity array I_trimmed is not 2D.")
        if I_trimmed.shape[1] == 0: raise ValueError("Data has zero time samples after initial trimming.")
        I_scr = scrunch(I_trimmed, tscrunch=downsample_factor, fscrunch=1)
        num_ds_bins = I_scr.shape[1] if I_scr.ndim == 2 and I_scr.shape[1]>0 else 0
        if num_ds_bins == 0: raise ValueError("Data has zero time samples after scrunching.")

        start_bin_ds, end_bin_ds = self._determine_time_window_ds(
            power_trimmed, num_ds_bins, downsample_factor, interactive,
            select_off_burst, time_range_ds, min_duration_native # <-- Pass it here
        )

        # Convert selected downsampled range back to native bins for final slice
        start_bin_final = start_bin_ds * downsample_factor
        end_bin_final = end_bin_ds * downsample_factor
        start_bin_final = max(0, start_bin_final)
        end_bin_final = min(data_masked_tmp.shape[-1], end_bin_final)
        if start_bin_final >= end_bin_final:
             raise ValueError(f"Final native time range is empty: [{start_bin_final}, {end_bin_final}]")
        print(f"Final selected native time range for analysis: [{start_bin_final}, {end_bin_final}]")

        # 5. Fill Missing Channels (using data before final time slice)
        print("Filling missing channels to 1024...")
        # Pass the data *before* the final time slice but *after* initial SNR trim
        data_filled, freqs_filled, freq_ids_filled = self._fill_missing_chans(
            data_masked_tmp, bbdata # bbdata needed for original freq map
        )
        # data_filled shape: [1024, pol, time_after_snr_trim]

        # 6. Apply Final Time Slice to Filled Data
        # Slice the filled data using the final native bin range
        data_filled_sliced = data_filled[..., start_bin_final:end_bin_final] # Ellipsis for freq, pol
        print(f"Data sliced to final time window. Shape: {data_filled_sliced.shape}")

        # 7. Optional Extra RFI Zapping on the final sliced data
        if zap_extra:
            print("Performing extra RFI zapping...")
            data_final = self._extra_flag(data_filled_sliced)
        else:
            data_final = data_filled_sliced

        # 8. Apply Frequency Limits (Spectrum Lim) to the final time-sliced data
        if spec_lims is None:
            print("Determining frequency limits...")
            try:
                power_final = np.abs(data_final)**2
                # Use get_spectrum_lim on the final processed, time-sliced power
                spec_lims = get_spectrum_lim(freq_ids_filled, power_final, diagnostic_plots=False)
                print(f"Determined frequency limits (channel indices): {spec_lims}")
            except Exception as e:
                print(f"Warning: Could not determine spectrum limits: {e}. Using full band.")
                spec_lims = [0, TOTAL_CHANNELS]
        else:
            # Validate user-provided limits
            if not (isinstance(spec_lims, (list, tuple)) and len(spec_lims) == 2):
                print("Warning: Invalid spec_lims format. Using full band.")
                spec_lims = [0, TOTAL_CHANNELS]
            print(f"Using provided frequency limits: {spec_lims}")

        f_start, f_end = int(spec_lims[0]), int(spec_lims[1])
        f_start = max(0, f_start)
        f_end = min(TOTAL_CHANNELS, f_end) # Ensure f_end is within 0-1024 range
        if f_start >= f_end:
             raise ValueError(f"Frequency limits are invalid or empty: [{f_start}, {f_end}]")

        # Slice final data and frequency arrays based on the 1024-channel grid
        # Ensure slicing uses the correct axis (axis 0 for frequency)
        self.processed_data = data_final[f_start:f_end, :, :]
        self.processed_freqs = freqs_filled[f_start:f_end]
        self.processed_freq_ids = freq_ids_filled[f_start:f_end] # These are the absolute IDs (0-1023)
        print(f"Data sliced to frequency limits [{f_start}, {f_end}]. Final shape: {self.processed_data.shape}")
        if len(self.processed_freqs) > 0:
             print(f"Final frequency range: {self.processed_freqs[-1]:.2f} - {self.processed_freqs[0]:.2f} MHz")
        else:
             print("Warning: Final frequency range is empty after slicing.")

        # 9. Plot Final Profile of the fully processed data
        if self.processed_data.shape[-1] > 0: # Check if time dimension is not empty
            self._plot_final_profile(self.processed_data, downsample_factor, select_off_burst)
        else:
            print("Skipping final profile plot: No time samples remain after processing.")

        print("--- Finished Data Preprocessing ---")
        return self.processed_data

    def _determine_time_window_ds(self, power_native, num_ds_bins, ds_factor,
                                 interactive, select_off_burst, time_range_ds,
                                 min_duration_native=None):
        """ Helper to determine the time window in downsampled bins. """
        if time_range_ds is not None:
            # User provided downsampled range - primarily for on-burst selection
            start_bin_ds, end_bin_ds = int(time_range_ds[0]), int(time_range_ds[1])
            print(f"Using provided downsampled time range: [{start_bin_ds}, {end_bin_ds}]")
            # Convert back to native for reference, but ds bins drive selection here
            start_native = start_bin_ds * ds_factor
            end_native = end_bin_ds * ds_factor
        elif interactive:
            # Ensure power_native is 2D+ before summing
            if power_native.ndim < 2: raise ValueError("_determine_time_window_ds needs power_native with time axis")
            # Sum pol if present -> [freq, time]
            pol_axis = 1 if power_native.ndim == 3 else None
            if pol_axis is not None: I_native = np.ma.sum(power_native, axis=pol_axis)
            else: I_native = power_native
            profile_scr = np.ma.mean(scrunch(I_native, ds_factor, 1), axis=0)
            plt.close('all'); plt.plot(profile_scr.filled(np.nanmedian(profile_scr)))
            plt.title("Select Time Range to Keep"); plt.grid(True); plt.show(block=False)
            answer = input(f"Define downsampled time bin range (e.g., '100,{num_ds_bins-100}'): ")
            plt.close()
            try: start_bin_ds, end_bin_ds = map(int, answer.split(','))
            except Exception as e: raise ValueError(f"Invalid input format: {e}")
            start_native = start_bin_ds * ds_factor
            end_native = end_bin_ds * ds_factor
        else:
            # Automatic selection
            print("Using automatic burst envelope detection...")
            try:
                # Detect envelope on the input power_native
                lims_native = self._get_burst_envelope(power_native.filled(0), thres=6, pad=0.1)
                print(f"Detected burst limits (native bins): {lims_native}")
            except Exception as e:
                print(f"Warning: _get_burst_envelope failed: {e}. Using full range as burst limit.")
                # If envelope fails, assume full range is potentially bursty for off-burst selection
                lims_native = [0, power_native.shape[-1]]

            if select_off_burst:
                # --- Improved Logic for Off-Burst Window ---
                print("Selecting OFF-burst range...")
                burst_start, burst_end = lims_native[0], lims_native[1]
                total_duration_native = power_native.shape[-1]

                # Define potential off-burst regions
                region1_start, region1_end = 0, burst_start
                region2_start, region2_end = burst_end, total_duration_native

                # Calculate durations
                duration1 = region1_end - region1_start
                duration2 = region2_end - region2_start

                # Check if minimum duration is required
                min_dur = min_duration_native if min_duration_native is not None else 1

                # Prioritize region 1 (before burst) if long enough
                if duration1 >= min_dur:
                    print(f"  Found sufficient off-burst data before burst ({duration1} bins).")
                    # Take the latest possible block of required duration from region 1
                    start_native = max(0, region1_end - min_dur) if min_duration_native else 0
                    end_native = region1_end
                # Else, try region 2 (after burst) if long enough
                elif duration2 >= min_dur:
                    print(f"  Found sufficient off-burst data after burst ({duration2} bins).")
                    # Take the earliest possible block of required duration from region 2
                    start_native = region2_start
                    end_native = min(total_duration_native, region2_start + min_dur) if min_duration_native else total_duration_native
                # Else, take the longest available region (even if shorter than min_dur)
                # This might still fail later if it's shorter than fftsize, but we try.
                elif duration1 > duration2:
                    print(f"  Warning: Pre-burst off-burst ({duration1} bins) is shorter than required ({min_dur}). Using it anyway.")
                    start_native, end_native = region1_start, region1_end
                elif duration2 > 0:
                    print(f"  Warning: Post-burst off-burst ({duration2} bins) is shorter than required ({min_dur}). Using it anyway.")
                    start_native, end_native = region2_start, region2_end
                else:
                    # No off-burst region found at all
                    raise ValueError("Cannot find any off-burst data outside the detected envelope.")

                print(f"  Selected native off-burst window: [{start_native}, {end_native}]")

            else: # On-Burst Window (add margin around detected limits)
                margin_native = 20000
                start_native = max(0, lims_native[0] - margin_native)
                end_native = min(power_native.shape[-1], lims_native[1] + margin_native)
                print(f"Selecting ON-burst range (native bins): [{start_native}, {end_native}]")

            # Convert final selected native range to downsampled range
            start_bin_ds = start_native // ds_factor
            end_bin_ds = end_native // ds_factor
            # Ensure end > start, minimum 1 bin wide
            if start_bin_ds >= end_bin_ds: end_bin_ds = start_bin_ds + 1

        # Final validation of downsampled range against available bins
        start_bin_ds = max(0, start_bin_ds)
        end_bin_ds = min(num_ds_bins, end_bin_ds) if num_ds_bins > 0 else start_bin_ds + 1
        if start_bin_ds >= end_bin_ds:
            print(f"Warning: Final downsampled time range empty/invalid: [{start_bin_ds}, {end_bin_ds}]. Using single bin.")
            end_bin_ds = start_bin_ds + 1
        print(f"Selected downsampled time range: [{start_bin_ds}, {end_bin_ds}]")
        return start_bin_ds, end_bin_ds

    def _fill_missing_chans(self, ds_in, bbdata):
        """ Helper to fill missing channels based on bbdata index map. """
        nchan_in = ds_in.shape[0]
        other_dims = ds_in.shape[1:] # e.g., (pol, time) or (time,)
        output_shape = (TOTAL_CHANNELS,) + other_dims
        # Match dtype (complex or float)
        dtype = ds_in.dtype if np.iscomplexobj(ds_in) else np.float64
        new_data = np.zeros(output_shape, dtype=dtype)

        try:
            # Get frequency IDs corresponding to the input data `ds_in`
            freq_map = bbdata.index_map["freq"]
            freq_id_in = freq_map["id"]
            if len(freq_id_in) != nchan_in:
                print(f"Warning: Mismatch in _fill_missing_chans. ds_in shape {ds_in.shape}, "
                    f"bbdata freq map len {len(freq_id_in)}. Attempting to use first {nchan_in} IDs.")
                if len(freq_id_in) < nchan_in: raise ValueError("Not enough freq IDs in bbdata map.")
                freq_id_in = freq_id_in[:nchan_in]

        except Exception as e: raise ValueError(f"Could not access frequency map in bbdata: {e}")

        # Initialize the final mask for the output array (same shape as new_data)
        # Start with everything masked (True means masked)
        final_mask_full = np.ones(output_shape, dtype=bool)

        # Place input data into the full 1024-channel array
        # Also, unmask the channels that are being filled
        # And transfer the original mask (if any) for the filled data points
        original_mask = None
        if isinstance(ds_in, np.ma.MaskedArray):
            original_mask = ds_in.mask

        for i, chan_id in enumerate(freq_id_in):
            abs_chan_id = int(chan_id) # Ensure integer
            if 0 <= abs_chan_id < TOTAL_CHANNELS:
                # Copy data
                new_data[abs_chan_id, ...] = ds_in[i, ...]
                # Unmask this channel in the final mask
                final_mask_full[abs_chan_id, ...] = False
                # If there was an original mask, apply it to the corresponding slice
                if original_mask is not None and not np.ma.is_masked(original_mask) and original_mask.shape[0] == nchan_in:
                     # Apply the original mask for this specific channel to the final mask
                     # Ensure broadcasting works for pol/time dimensions
                     original_channel_mask = original_mask[i, ...] # Shape (pol, time) or (time,)
                     # Combine with existing mask for this channel (logical OR)
                     final_mask_full[abs_chan_id, ...] = np.logical_or(
                         final_mask_full[abs_chan_id, ...],
                         original_channel_mask
                     )
            else:
                print(f"Warning: Input channel ID {abs_chan_id} out of range [0, {TOTAL_CHANNELS-1}). Skipping.")

        # Create the final masked array using the correctly shaped mask
        data_masked = np.ma.masked_array(new_data, mask=final_mask_full)
        # --- End Corrected Mask Handling ---

        # Create the standard full frequency axis
        new_freq_id_abs = np.arange(TOTAL_CHANNELS)
        new_freqs_abs = np.linspace(FREQ_BOTTOM_MHZ, FREQ_TOP_MHZ, TOTAL_CHANNELS)[::-1] # High to low

        return data_masked, new_freqs_abs, new_freq_id_abs

    def _extra_flag(self, data_in):
        """ Helper for extra RFI flagging based on low channel power. """
        # Ensure input is masked array
        if not isinstance(data_in, np.ma.MaskedArray):
             data_masked = np.ma.masked_array(data_in)
        else:
             data_masked = data_in.copy() # Work on copy

        # Calculate channel spectrum (sum power over pol and time)
        # Ensure correct axes are summed based on ndim
        if data_masked.ndim == 3: # [freq, pol, time]
             chan_spectrum = np.ma.sum(np.ma.sum(np.abs(data_masked)**2, axis=1), axis=-1)
        elif data_masked.ndim == 2: # [freq, time] - No pol?
            print("Warning: _extra_flag received 2D data. Assuming no polarization.")
            chan_spectrum = np.ma.sum(np.abs(data_masked)**2, axis=-1)
        else:
            print("Warning: _extra_flag received data with unexpected dimensions. Skipping.")
            return data_masked

        try:
            # Use only valid (unmasked) points for stats
            valid_spec = chan_spectrum.compressed() # Get 1D array of unmasked values
            if len(valid_spec) < 2: # Need at least 2 points for stats
                print("Warning: Not enough valid points for extra flagging stats. Skipping.")
                return data_masked

            spec_median = np.median(valid_spec) # Use median on valid points
            spec_mad = median_abs_deviation(valid_spec, scale='normal')

            # Handle zero MAD case robustly
            if spec_mad == 0 or np.isnan(spec_mad):
                 spec_mad = np.std(valid_spec) # Fallback to std dev
            if spec_mad == 0 or np.isnan(spec_mad):
                 spec_mad = 1.0 # Avoid division by zero if still zero

            # Calculate SNR relative to median/MAD for each channel
            chan_spectrum_snr = (chan_spectrum - spec_median) / spec_mad

            # Identify low power channels (e.g., < -3 sigma) that are not already masked
            rfi_mask_extra = (chan_spectrum_snr < -3.0) & (~chan_spectrum.mask)

            num_flagged = np.sum(rfi_mask_extra)
            if num_flagged > 0:
                print(f"Extra flagging: Masking {num_flagged} channels based on low power.")
                # Apply mask (expand dims to match data_masked)
                mask_shape = rfi_mask_extra.shape + (1,) * (data_masked.ndim - 1)
                rfi_mask_nd = rfi_mask_extra.reshape(mask_shape)
                # Combine with existing mask
                data_masked.mask = np.logical_or(data_masked.mask, rfi_mask_nd)

        except ImportError:
             print("Warning: Cannot perform extra flagging, scipy.stats required.")
        except Exception as e:
             print(f"Warning: Extra flagging failed: {e}")
        return data_masked

    def _get_burst_envelope(self, power_arr, thres=5, pad=0.0):
        """ Helper to find burst envelope using baseband_analysis or fallback. """
        # Ensure power_arr is at least 1D
        if power_arr.ndim == 0: raise ValueError("_get_burst_envelope needs at least 1D power array.")

        # Average over all dimensions except the last (time)
        if power_arr.ndim >= 2:
            avg_axes = tuple(range(power_arr.ndim - 1))
            prof = np.nanmean(power_arr, axis=avg_axes)
        else:
             prof = power_arr.copy()

        if len(prof) == 0: return np.array([0, 0]) # Handle empty profile

        lims_raw = None # Initialize
        # --- Use baseband_analysis.signal.get_main_peak_lim if available ---
        if BASEBAND_ANALYSIS_AVAILABLE:
            try:
                # Normalize profile robustly before finding peak
                median = np.nanmedian(prof)
                std = np.nanstd(prof)
                # Check for invalid std dev (constant profile or all NaNs)
                if std == 0 or np.isnan(std): return np.array([0, len(prof)])
                prof_norm = (prof - median) / std
                # Find limits based on threshold
                lims_raw = get_main_peak_lim(prof_norm, floor_level=thres)
                # --- Ensure lims_raw is a numpy array ---
                lims_raw = np.array(lims_raw)
                # ---
            except Exception as e:
                print(f"Warning: baseband_analysis.get_main_peak_lim failed ({e}). Using basic thresholding.")
                lims_raw = None # Signal fallback

        # --- Fallback if baseband_analysis not available OR get_main_peak_lim failed ---
        if lims_raw is None:
            print("Warning: Using basic thresholding for burst envelope.")
            median = np.nanmedian(prof); std = np.nanstd(prof)
            if std == 0 or np.isnan(std): return np.array([0, len(prof)]) # No variation
            above_thresh = np.where(prof > median + thres * std)[0]
            if len(above_thresh) == 0: lims_raw = np.array([0, 0]) # No peak found
            else: lims_raw = np.array([above_thresh[0], above_thresh[-1] + 1])

        # Handle case where no peak was found (lims=[0,0] or similar)
        if lims_raw[1] <= lims_raw[0]:
            print("Warning: No significant burst detected by envelope finder. Returning full range.")
            lims = np.array([0, len(prof)]) # Return full time range
        else:
            lims = lims_raw # Use the detected limits

        # Apply padding to the determined limits
        duration = lims[1] - lims[0]
        pad_bins = int(duration * pad)
        lims[0] = max(0, lims[0] - pad_bins)
        lims[1] = min(len(prof), lims[1] + pad_bins)
        # Return integer bin indices as a numpy array
        return lims.astype(int) 

    def _plot_final_profile(self, data_final, ds_factor, off_burst_flag):
        """ Helper to plot the final time profile after all processing. """
        if data_final.size == 0 or data_final.shape[-1] == 0:
            print("Skipping final profile plot: Data is empty.")
            return

        print("Generating final time profile plot...")
        try:
            power_final = np.abs(data_final)**2
            # Sum polarization if present
            pol_axis = 1 if data_final.ndim == 3 else None
            if pol_axis is not None: I_final = np.ma.sum(power_final, axis=pol_axis)
            else: I_final = power_final # Assumes [freq, time]

            # Calculate native resolution profile
            prof_native = np.ma.mean(I_final, axis=0) # Mean over freq -> [time]
            time_axis_native = np.arange(len(prof_native)) * NATIVE_BIN_DURATION_S * 1000 # ms

            # Calculate scrunched profile
            prof_scr_mean = np.array([])
            time_axis_scr = np.array([])
            if I_final.ndim == 2 and I_final.shape[1] > 0: # Check if I_final is 2D and has time samples
                prof_scr = scrunch(I_final, tscrunch=ds_factor, fscrunch=1)
                if prof_scr.size > 0:
                    prof_scr_mean = np.ma.mean(prof_scr, axis=0)
                    time_axis_scr = np.arange(len(prof_scr_mean)) * NATIVE_BIN_DURATION_S * ds_factor * 1000 # ms

            # Plotting
            plt.close('all'); fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(time_axis_native, prof_native.filled(np.nan), color='k', alpha=0.5, label='Native Res')
            if prof_scr_mean.size > 0:
                ax.plot(time_axis_scr, prof_scr_mean.filled(np.nan), color='r', label=f'Scrunched x{ds_factor}')
            ax.set_xlabel('Time [ms]'); ax.set_ylabel('Intensity [arb.]')
            ax.set_title(f'Evt {self.event_id} - Final Profile ({ "Off" if off_burst_flag else "On"}-Burst Window)')
            ax.legend(); ax.grid(True, alpha=0.5); plt.tight_layout()

            # Save or show plot
            if self.output_dir:
                fname = f'{self.output_dir}/{"off" if off_burst_flag else "on"}burst_prof_evt{self.event_id}.png'
                try: plt.savefig(fname); print(f"Saved final profile plot: {fname}")
                except Exception as e: print(f"Error saving plot: {e}")
                plt.close(fig)
            elif matplotlib.get_backend() != 'agg': plt.show() # Show if interactive
            else: plt.close(fig) # Close otherwise

        except Exception as e:
            print(f"Error generating final profile plot: {e}")
            plt.close('all') # Ensure plot is closed on error


    # --- Upchannelization ---
    def upchannelize(self, data=None, freq_ids=None, fftsize=32, downfreq=2, use_fast=True):
        """
        Upchannelizes the provided data (defaults to self.processed_data).

        Performs FFT-based upchannelization on complex voltage data.

        Parameters
        ----------
        data : np.ma.MaskedArray or np.ndarray, optional
            Complex voltage data [freq, pol, time]. Defaults to `self.processed_data`.
        freq_ids : np.ndarray, optional
            Absolute frequency IDs (0-1023 range) corresponding to the frequency
            axis of `data`. Defaults to `self.processed_freq_ids`.
        fftsize : int, optional
            FFT size for the time-domain transform. Default 32.
        downfreq : int, optional
            Factor by which to average frequency channels after FFT shift. Default 2.
        use_fast : bool, optional
            Use the vectorized `_upchannel_fast` implementation. Default True.

        Returns
        -------
        tuple
            (upchan_data, upchan_freqs, upchan_freq_ids) containing:
            - upchan_data : np.ndarray [pol, nblock, nfreq_up] - Upchannelized complex data.
            - upchan_freqs : np.ndarray [nfreq_up] - Frequencies (MHz) of upchannelized channels.
            - upchan_freq_ids : np.ndarray [nfreq_up] - Absolute IDs of upchannelized channels.
            These are also stored in instance attributes.

        Raises
        ------
        ValueError
            If input data is invalid, empty, or parameters are incorrect.
        RuntimeError
            If the upchannelization calculation fails.
        """
        if data is None: data = self.processed_data
        if freq_ids is None: freq_ids = self.processed_freq_ids # Use IDs corresponding to processed data
        if data is None: raise ValueError("No processed data available to upchannelize.")
        if data.size == 0: raise ValueError("Processed data is empty, cannot upchannelize.")

        # Validate fftsize and downfreq
        if not isinstance(fftsize, int) or fftsize <= 0: raise ValueError("fftsize must be a positive integer.")
        if not isinstance(downfreq, int) or downfreq <= 0: raise ValueError("downfreq must be a positive integer.")
        if fftsize % downfreq != 0: raise ValueError("fftsize must be divisible by downfreq.")
        # Check data length vs fftsize
        if data.shape[-1] < fftsize:
            raise ValueError(f"Data length ({data.shape[-1]}) is less than fftsize ({fftsize}). Cannot upchannelize.")


        print(f"\n--- Upchannelizing (fftsize={fftsize}, downfreq={downfreq}) ---")
        self.upchan_fftsize = fftsize
        self.upchan_downfreq = downfreq

        # Select implementation
        upchan_func = self._upchannel_fast if use_fast else self._upchannel_loop
        impl_name = "fast (vectorized)" if use_fast else "original (loop-based)"
        print(f"Using {impl_name} upchannel implementation.")

        try:
            # Ensure data is complex before passing
            if not np.iscomplexobj(data):
                print("Warning: Data for upchannelization is not complex. Casting to complex.")
                # Use astype on the underlying data if masked
                if isinstance(data, np.ma.MaskedArray):
                    complex_data = data.data.astype(np.complex64)
                    data = np.ma.masked_array(complex_data, mask=data.mask)
                else:
                    data = data.astype(np.complex64)

            # Call the selected upchannelization function
            self.upchan_data, self.upchan_freqs, self.upchan_freq_ids = upchan_func(
                data, freq_ids, fftsize, downfreq
            )
            print(f"Upchannelized data shape: {self.upchan_data.shape}")
            if len(self.upchan_freqs) > 0:
                 print(f"Upchannelized freq range: {self.upchan_freqs[-1]:.3f} - {self.upchan_freqs[0]:.3f} MHz")
            else:
                 print("Warning: Upchannelized frequency array is empty.")
            return self.upchan_data, self.upchan_freqs, self.upchan_freq_ids
        except Exception as e:
            # Add more specific error context if possible
            print(f"ERROR during upchannelization with {impl_name} implementation.")
            raise RuntimeError(f"Upchannelization failed: {e}")


    def _upchannel_fast(self, wfall, freq_id, fftsize=32, downfreq=2):
        """ Vectorized upchannelization implementation. """
        # Assumes input validation done in calling function
        wfall_proc = np.swapaxes(wfall, 0, 1); wfall_proc = np.swapaxes(wfall_proc, 1, 2)
        npol, nsamp, nchan_in = wfall_proc.shape
        upchan = fftsize // downfreq
        nblock = nsamp // fftsize
        nchan_up = nchan_in * upchan

        if nblock == 0: return np.array([]).reshape(npol, 0, nchan_up), np.array([]), np.array([])

        # Calculate full frequency axis and channel ID map
        f_upchan_bandtot = np.linspace(FREQ_TOP_MHZ, FREQ_BOTTOM_MHZ, upchan * TOTAL_CHANNELS)
        freq_id_int = freq_id.astype(int)
        # Create mapping for *input* channels to *absolute* upchannel IDs (0 to 1024*upchan - 1)
        chan_id_map_abs = (freq_id_int[:, None] * upchan) + np.arange(upchan)
        # Flatten to get the absolute IDs corresponding to the output frequency axis
        chan_id_final_abs = chan_id_map_abs.ravel()

        # Filter IDs and Frequencies that are within the valid total range
        valid_ids_mask = (chan_id_final_abs >= 0) & (chan_id_final_abs < len(f_upchan_bandtot))
        chan_id_final_abs_valid = chan_id_final_abs[valid_ids_mask]
        f_final_valid = f_upchan_bandtot[chan_id_final_abs_valid]
        if not np.all(valid_ids_mask): print("Warning: Some absolute upchannel IDs were out of bounds.")

        # Perform FFT and reshaping
        spec = np.zeros((npol, nblock, nchan_up), dtype=np.complex64)
        valid_nsamp = nblock * fftsize
        wfall_trunc = wfall_proc[:, :valid_nsamp, :]

        for pol in range(npol):
            reshaped = wfall_trunc[pol].reshape(nblock, fftsize, nchan_in)
            fft_res = np.fft.fft(reshaped, axis=1)
            fft_shifted = np.fft.fftshift(fft_res, axes=1)
            downsampled = fft_shifted.reshape(nblock, upchan, downfreq, nchan_in).mean(axis=2)
            transposed = downsampled.transpose(0, 2, 1)
            spec[pol] = transposed.reshape(nblock, nchan_up) # Shape [npol, nblock, nchan_in * upchan]

        # Select the valid columns in spec corresponding to valid_ids_mask
        # The spec array columns directly correspond to the flattened chan_id_map_abs
        spec_valid = spec[:, :, valid_ids_mask]

        return spec_valid, f_final_valid, chan_id_final_abs_valid

    def _upchannel_loop(self, wfall, freq_id, fftsize=32, downfreq=2):
        """ Original loop-based upchannelization implementation. """
        # Assumes input validation done in calling function
        wfall_proc = np.swapaxes(wfall, 0, 1); wfall_proc = np.swapaxes(wfall_proc, 1, 2)
        npol, nsamp, nchan_in = wfall_proc.shape
        upchan = fftsize // downfreq; nblock = nsamp // fftsize; nchan_up = nchan_in * upchan

        if nblock == 0: return np.array([]).reshape(npol, 0, nchan_up), np.array([]), np.array([])

        f_upchan_bandtot = np.linspace(FREQ_TOP_MHZ, FREQ_BOTTOM_MHZ, upchan * TOTAL_CHANNELS)
        spec = np.zeros((npol, nblock, nchan_up), dtype=np.complex64)
        chan_id_map_abs = np.zeros((nchan_in, upchan), dtype=int) # Store absolute IDs
        freq_id_int = freq_id.astype(int)

        for pol in range(npol):
            for bi in range(nblock):
                t_start = bi * fftsize; t_end = t_start + fftsize
                for chidx_in in range(nchan_in):
                    ts = wfall_proc[pol, t_start:t_end, chidx_in].copy()
                    ft = np.fft.fft(ts); ft_shifted = np.fft.fftshift(ft)
                    ft_down = ft_shifted.reshape(upchan, downfreq).mean(axis=1)
                    # Calculate absolute start index in the full 1024*upchan grid
                    abs_freq_id = freq_id_int[chidx_in]
                    # Calculate index within the output `spec` array (0 to nchan_up-1)
                    spec_start_rel = chidx_in * upchan
                    spec_end_rel = spec_start_rel + upchan
                    spec[pol, bi, spec_start_rel:spec_end_rel] = ft_down
                    # Store absolute IDs only once
                    if pol == 0 and bi == 0:
                        chan_id_map_abs[chidx_in, :] = np.arange(upchan*abs_freq_id, upchan*abs_freq_id + upchan)

        chan_id_final_abs = chan_id_map_abs.ravel()
        valid_ids_mask = (chan_id_final_abs >= 0) & (chan_id_final_abs < len(f_upchan_bandtot))
        chan_id_final_abs_valid = chan_id_final_abs[valid_ids_mask]
        f_final_valid = f_upchan_bandtot[chan_id_final_abs_valid]
        if not np.all(valid_ids_mask): print("Warning: Some absolute upchannel IDs out of bounds.")

        # Select valid columns from spec
        spec_valid = spec[:, :, valid_ids_mask]

        return spec_valid, f_final_valid, chan_id_final_abs_valid


    # --- Scallop Modeling and Normalization ---
    def make_scallop_model(self, off_burst_data=None, fftsize=None, downfreq=None,
                          manual_off_burst_range_native=None,
                          interactive_off_burst_fallback=True):
        """
        Creates the scallop (instrumental bandpass ripple) model using
        off-burst upchannelized data.

        If `off_burst_data` is not provided:
        1. If `manual_off_burst_range_native` is given, uses that range.
        2. If `manual_off_burst_range_native` is None, attempts automatic off-burst selection.
        3. If automatic selection fails AND `interactive_off_burst_fallback` is True,
           plots a profile and prompts the user for a native time range..

        Parameters
        ----------
        off_burst_data : np.ndarray, optional
            Upchannelized complex off-burst data [pol, block, upchan_freq].
            If None, attempts to generate it automatically.
        fftsize : int, optional
            FFT size used for upchannelization. Defaults to `self.upchan_fftsize`.
        downfreq : int, optional
            Downsampling factor used. Defaults to `self.upchan_downfreq`.
        manual_off_burst_range_native : tuple, optional
            Specify the off-burst window [start_bin, end_bin] in *native* time bins.
            Overrides automatic off-burst selection if `off_burst_data` is None.
            Default None.
        interactive_off_burst_fallback : bool, optional
            If True and automatic off-burst selection fails, prompt the user
            interactively for the range. Default True.

        Returns
        -------
        tuple
            (model, rfi_indices) containing:
            - model : np.ndarray [nfreq_up] - The scallop model, tiled to match on-burst data length.
            - rfi_indices : np.ndarray - Indices of channels flagged as RFI during model creation.
            These are also stored in `self.scallop_model` and `self.scallop_rfi_inds`.

        Raises
        ------
        ValueError
            If parameters are missing or off-burst data is invalid.
        RuntimeError
            If automatic generation of off-burst data fails.
        """
        # Ensure we're using the correct off-burst data generation logic
        # if off_burst_data is None
        if fftsize is None: fftsize = self.upchan_fftsize
        if downfreq is None: downfreq = self.upchan_downfreq
        if fftsize is None or downfreq is None:
             raise ValueError("fftsize/downfreq needed for scallop model.")

        if off_burst_data is None:
            print("Off-burst data not provided, generating now...")
            generated_off_burst_data = None
            off_burst_native_range_to_use = manual_off_burst_range_native

            # --- Try Manual Range First ---
            if off_burst_native_range_to_use is not None:
                print(f"Using manually specified native off-burst range: {off_burst_native_range_to_use}")
                if (off_burst_native_range_to_use[1] - off_burst_native_range_to_use[0]) < fftsize:
                     raise ValueError(f"Manual off-burst range duration insufficient for fftsize={fftsize}.")
                # Proceed to generate data using this manual range
            else:
                # --- Try Automatic Selection ---
                print("Attempting automatic off-burst selection...")
                try:
                    # Run preprocess with select_off_burst=True and min_duration
                    if self.bbdata is None: self.load_data()
                    temp_processor_off = ScintillationProcessor(self.event_id, self.dm, ScintillationProcessor.SET_DM, self.baseband_file)
                    temp_processor_off.bbdata = self.bbdata
                    # Call preprocess with auto selection (time_range_ds=None)
                    data_off_auto = temp_processor_off.preprocess_data(
                        select_off_burst=True, interactive=False, time_range_ds=None,
                        zap_extra=False, spec_lims=[0, TOTAL_CHANNELS],
                        min_duration_native=fftsize
                    )
                    # If preprocess succeeded, store the range it used (optional, for info)
                    # Note: preprocess_data doesn't directly return the range it selected.
                    # We'd need to modify it or infer from data_off_auto shape if needed.
                    print("Automatic off-burst selection successful.")
                    # Proceed to upchannelize data_off_auto
                    generated_off_burst_data = data_off_auto # Use the data generated by auto selection
                    del temp_processor_off

                except ValueError as e_auto: # Catch errors specifically from auto-selection failure
                    print(f"Automatic off-burst selection failed: {e_auto}")
                    if not interactive_off_burst_fallback:
                         # If interaction is disabled, raise the error
                        raise RuntimeError(f"Off-burst data required but automatic selection failed and interaction is disabled: {e_auto}")
                      
                    else:
                        # --- Interactive Fallback ---
                        print("Attempting interactive off-burst selection...")
                        try:
                            # Plot a helpful profile - use initially loaded raw data profile?
                            # Or profile from the temporary processor *before* time selection failed?
                            # Let's plot the profile from the initial SNR step if possible.
                            # This requires access to the original bbdata and potentially re-running parts.
                            # Simpler: Plot profile of *full* available data after initial SNR time cut.
                            print("Plotting profile of available data range for selection...")
                            if self.bbdata is None: self.load_data()
                            # Get full time range after initial SNR cut from a temp processor
                            temp_proc_prof = ScintillationProcessor(self.event_id, self.dm, ScintillationProcessor.SET_DM, self.baseband_file)
                            temp_proc_prof.bbdata = self.bbdata
                            # Run only the initial steps of preprocess to get data_masked_tmp
                            temp_proc_prof._initial_snr_and_dedisp() # Need to create this helper or replicate logic
                            full_data_after_snr_cut = temp_proc_prof.data_masked_tmp # Assume this attribute exists after helper
                            full_native_time_bins = full_data_after_snr_cut.shape[-1]

                            pol_axis = 1 if full_data_after_snr_cut.ndim == 3 else None
                            if pol_axis is not None: I_full = np.ma.sum(np.abs(full_data_after_snr_cut)**2, axis=pol_axis)
                            else: I_full = np.abs(full_data_after_snr_cut)**2
                            profile_full = np.ma.mean(I_full, axis=0) # Mean over freq
                            time_axis_full = np.arange(len(profile_full))

                            plt.close('all'); plt.figure(figsize=(12,5))
                            plt.plot(time_axis_full, profile_full.filled(np.nanmedian(profile_full)))
                            plt.xlabel(f"Native Time Bins (Total Available: {full_native_time_bins})")
                            plt.ylabel("Intensity (arb.)")
                            plt.title(f"Select OFF-BURST Range (Native Bins) for Event {self.event_id}\n(Ensure duration >= {fftsize} bins)")
                            plt.grid(True); plt.tight_layout(); plt.show(block=False)

                            while off_burst_native_range_to_use is None:
                                answer = input(f"Enter desired OFF-BURST range [start_bin, end_bin] (e.g., '0,{fftsize*2}'): ")
                                plt.close() # Close plot after input
                                try:
                                    start_bin, end_bin = map(int, answer.split(','))
                                    if 0 <= start_bin < end_bin <= full_native_time_bins:
                                        if (end_bin - start_bin) >= fftsize:
                                            off_burst_native_range_to_use = (start_bin, end_bin)
                                            print(f"Using manually entered native range: {off_burst_native_range_to_use}")
                                        else:
                                            print(f"Error: Duration ({end_bin - start_bin}) must be >= fftsize ({fftsize}). Try again.")
                                    else:
                                        print(f"Error: Invalid range. Must be within [0, {full_native_time_bins}] and start < end. Try again.")
                                except Exception as e_input:
                                    print(f"Error parsing input: {e_input}. Please use format 'start,end'. Try again.")
                            del temp_proc_prof # Clean up

                        except Exception as e_interact:
                            plt.close('all')
                            raise RuntimeError(f"Failed during interactive off-burst selection: {e_interact}")
            # --- End of Automatic/Interactive Logic ---

            # --- Generate Off-Burst Data using the Determined Range ---
            if generated_off_burst_data is None: # Only if not already generated by successful auto-selection
                if off_burst_native_range_to_use is None:
                    # Should not happen if logic above is correct, but as safety check
                    raise RuntimeError("Off-burst range could not be determined.")

                print(f"Preprocessing manually selected off-burst range: {off_burst_native_range_to_use}")
                # Need to run preprocess again, but selecting *only* the desired native range
                # This is tricky as preprocess works with downsampled ranges.
                # Option 1: Modify preprocess to accept native range (cleaner but more invasive).
                # Option 2: Load data, slice manually, then pass to preprocess (less efficient).
                # Option 3: Approximate ds range and pass to preprocess (as done before). Let's stick with this for now.
                ds_factor_approx = 32 # Use default ds factor for conversion
                start_ds = off_burst_native_range_to_use[0] // ds_factor_approx
                end_ds = off_burst_native_range_to_use[1] // ds_factor_approx
                if start_ds >= end_ds: end_ds = start_ds + 1
                off_burst_time_range_ds = (start_ds, end_ds)
                
                temp_processor_off = ScintillationProcessor(self.event_id, self.dm, ScintillationProcessor.SET_DM, self.baseband_file)
                temp_processor_off.bbdata = self.bbdata
                data_off = temp_processor_off.preprocess_data(
                    select_off_burst=True, # Keep True for logging consistency
                    time_range_ds=off_burst_time_range_ds, # Pass the *target* range
                    interactive=False, zap_extra=False, spec_lims=[0, TOTAL_CHANNELS],
                    min_duration_native=fftsize # Still useful for validation within preprocess
                )
                # We need to ensure the *output* data_off actually corresponds *exactly*
                # to the native range requested. Preprocess might slightly alter it.
                # A more robust way is needed if exact native range slicing is critical *before* upchannel.
                # For now, assume preprocess output with target ds range is sufficient.

                print("Upchannelizing selected off-burst data...")
                generated_off_burst_data, _, _ = temp_processor_off.upchannelize(
                    data=data_off,
                    freq_ids=temp_processor_off.processed_freq_ids,
                    fftsize=fftsize, downfreq=downfreq
                )
                if generated_off_burst_data.size == 0:
                     raise ValueError("Generated off-burst upchannelized data is empty.")
                del temp_processor_off

            # Use the generated data
            off_burst_data = generated_off_burst_data
            # --- End of Off-Burst Data Generation ---

        print("\n--- Creating Scallop Model ---")
        if off_burst_data.ndim != 3: raise ValueError("Input off_burst_data must be 3D")
        if off_burst_data.shape[1] == 0: raise ValueError("Off-burst data has zero time blocks.")

        # Calculate noise spectrum from off-burst data
        noise_power = np.abs(off_burst_data**2)
        I_noise = np.mean(noise_power, axis=0).T # Avg pol -> [freq, block]
        spec_noise = np.nanmean(I_noise, axis=1) # Avg block -> [freq]

        # RFI flagging using robust stats
        try:
            valid_noise = spec_noise[~np.isnan(spec_noise)]
            if len(valid_noise) == 0: raise ValueError("Noise spectrum is all NaN.")
            noise_median = np.median(valid_noise) # Use median of valid points
            noise_mad = median_abs_deviation(valid_noise, scale='normal')
            if noise_mad == 0 or np.isnan(noise_mad): noise_mad = np.std(valid_noise)
            if noise_mad == 0 or np.isnan(noise_mad): noise_mad = 1.0
            spec_noise_norm = (spec_noise - noise_median) / noise_mad
        except Exception as e: # Fallback to mean/std
            print(f"Warning: Robust stats failed ({e}), using mean/std.")
            noise_mean = np.nanmean(spec_noise); noise_std = np.nanstd(spec_noise)
            if noise_std == 0 or np.isnan(noise_std): noise_std = 1.0
            spec_noise_norm = (spec_noise - noise_mean) / noise_std

        # Ensure spec_noise_norm has same shape for where() call
        inds_rfi = np.array([], dtype=int)
        if spec_noise_norm.shape == spec_noise.shape:
             inds_rfi = np.where(np.abs(spec_noise_norm) > 3)[0] # 3-sigma RFI
        else: print("Warning: Could not normalize noise spectrum, skipping RFI flagging.")

        # Create model by averaging over PFB shape within coarse channels
        spec_noise_masked = np.ma.masked_where(np.isnan(spec_noise), spec_noise)
        spec_noise_masked[inds_rfi] = np.ma.masked # Mask RFI
        upchan = fftsize // downfreq
        nchan_up_total = len(spec_noise_masked)
        nchan_coarse = nchan_up_total // upchan

        if nchan_coarse == 0: raise ValueError("Not enough channels for scallop model.")
        # Truncate if not divisible (should match upchannelization output length)
        if nchan_up_total % upchan != 0:
            print("Warning: Scallop noise spectrum length not divisible by upchan factor. Truncating.")
            nchan_up_total = nchan_coarse * upchan
            spec_noise_masked = spec_noise_masked[:nchan_up_total]

        spec_noise_masked_reshape = spec_noise_masked.reshape(nchan_coarse, upchan)
        model_scallop_single = np.ma.mean(spec_noise_masked_reshape, axis=0)

        # Check if model is valid before tiling
        if model_scallop_single.count() > 0: # Check if any unmasked values
            model_data = model_scallop_single.filled(np.nanmedian(model_scallop_single)) # Fill with median
            model = np.tile(model_data, nchan_coarse)
            # Adjust length if truncated
            if len(model) != len(spec_noise): model = model[:len(spec_noise)]
            model = np.nan_to_num(model, nan=1.0) # Replace any remaining NaNs with 1
        else:
            print("Warning: Scallop model could not be computed. Using flat model.")
            model = np.ones(nchan_up_total) # Fallback to flat model

        # Store the model corresponding to the *on-burst* upchannel data length
        target_len = self.upchan_data.shape[2]
        if len(model) != target_len:
            print(f"Warning: Generated scallop model length ({len(model)}) differs from target ({target_len}). Resizing.")
            if len(model) > target_len: self.scallop_model = model[:target_len]
            elif len(model) > 0: self.scallop_model = np.tile(model, int(np.ceil(target_len/len(model))))[:target_len]
            else: self.scallop_model = np.ones(target_len) # Should not happen if checks above work
        else:
             self.scallop_model = model

        # Store RFI indices relevant to the target length
        self.scallop_rfi_inds = inds_rfi[inds_rfi < target_len]
        print(f"Scallop model created. Flagged {len(self.scallop_rfi_inds)} RFI channels within target range.")
        return self.scallop_model, self.scallop_rfi_inds

    def _initial_snr_and_dedisp(self):
        """ Helper to run only initial SNR/Dedisp steps."""
        if self.bbdata is None: self.load_data()
        bbdata = self.bbdata
        baseband_key = 'tiedbeam_baseband'
        try:
            snr_results = get_snr(bbdata, DM=self.dm, diagnostic_plots=False, return_full=True, downsample=32)
            valid_time_bins_native = snr_results[6]
        except Exception: valid_time_bins_native = [0, bbdata[baseband_key].shape[-1]]

        #data_dedisp = bbdata[baseband_key]
        
        # Dedisperse (make sure bbdata is modified if needed)
        if self.dm != 0 and isinstance(current_dm, str):
            print(f"Current DM {current_dm} and self.dm {self.dm} don't agree within 1 pc cm-3")
            print(f"Applying coherent dedispersion (DM={self.dm})...")
            # coherent_dedisp modifies bbdata[baseband_key] in place when write=True
            coherent_dedisp(bbdata, self.dm, time_shift=False, write=True)
            print(f"Coherent dedispersion applied in-place to '{baseband_key}'.")
            data_dedisp, freq, freq_id = incoherent_dedisp(bbdata, self.dm, fill_wfall=False)

        t_start = max(0, int(valid_time_bins_native[0]))
        t_end = min(data_dedisp.shape[-1], int(valid_time_bins_native[1]))
        # Store the data available *after* initial SNR time cut but *before* on/off selection
        # Make a copy to avoid modifying the main bbdata further if called multiple times
        self.data_masked_tmp = np.ma.masked_array(data_dedisp[..., t_start:t_end].copy(), mask=False)
        print(f"(Helper: Data available for plotting profile has shape {self.data_masked_tmp.shape})")
    
    def calculate_normalized_spectra(self, on_burst_lims_native=None):
        """
        Normalizes the upchannelized data using the scallop model and calculates
        on-burst, peak, and off-burst spectra relative to the local baseline.

        Applies the scallop model, then normalizes each frequency channel by
        subtracting the mean and dividing by the standard deviation of a preceding
        off-burst region (in upchannelized blocks). Calculates the average spectrum
        over the on-burst blocks, the spectrum at the single brightest block, and
        the average spectrum over the off-burst blocks used for normalization.

        Parameters
        ----------
        on_burst_lims_native : tuple, optional
             DEPRECATED. On-burst limits are now determined automatically from
             the upchannelized data profile.

        Returns
        -------
        tuple
            (spec_on, spec_peak, spec_off) containing the normalized spectra:
            - spec_on : np.ma.MaskedArray [nfreq_up]
            - spec_peak : np.ma.MaskedArray [nfreq_up]
            - spec_off : np.ma.MaskedArray [nfreq_up]
            These are also stored in instance attributes.

        Raises
        ------
        ValueError
            If upchannelized data or scallop model are missing or invalid.
        """
        # Ensure we're using self attributes correctly
        if self.upchan_data is None or self.upchan_data.size == 0: raise ValueError("Upchan data empty.")
        if self.scallop_model is None: raise ValueError("Scallop model needed.")
        if len(self.scallop_model) != self.upchan_data.shape[2]: raise ValueError("Scallop model length mismatch.")

        print("\n--- Calculating Normalized Spectra ---")
        up_data = self.upchan_data; model = self.scallop_model
        npol, nblock, nfreq_up = up_data.shape

        I_upchan = np.mean(np.abs(up_data**2), axis=0).T # Avg pol -> [freq, block]
        model_safe = np.where(model <= 0, np.nan, model)
        I_upchan_norm = I_upchan / model_safe[:, np.newaxis]

        # Determine block limits
        prof_norm_upchan = np.nanmean(I_upchan_norm, axis=0)
        if np.all(np.isnan(prof_norm_upchan)): lims_block = [0, nblock]
        else:
            median_prof = np.nanmedian(prof_norm_upchan); std_prof = np.nanstd(prof_norm_upchan)
            if std_prof > 0 and not np.isnan(std_prof):
                prof_snr = (prof_norm_upchan - median_prof) / std_prof; thresh = 3.0
                above = np.where(prof_snr > thresh)[0]
                if len(above) > 0:
                    lims_raw = [above[0], above[-1] + 1]; pad = max(1, int(0.1*(lims_raw[1]-lims_raw[0])))
                    lims_block = [max(0, lims_raw[0]-pad), min(nblock, lims_raw[1]+pad)]
                else: mid = nblock//2; width = max(1, nblock//20); lims_block = [max(0,mid-width), min(nblock, mid+width+1)]
            else: lims_block = [0, nblock]
        print(f"Determined upchannelized block limits: {lims_block}")

        # Per-channel normalization
        off_start = 0; off_end = min(max(1, lims_block[0]), nblock)
        print(f"Using off-burst blocks: [{off_start}, {off_end}]")
        if off_start >= off_end:
            print("Warning: No valid off-burst blocks. Skipping per-channel normalization.")
            I_norm_final = I_upchan_norm
        else:
            I_norm_final = np.zeros_like(I_upchan_norm)*np.nan; off_std = np.zeros(nfreq_up)*np.nan
            for f in range(nfreq_up):
                I_off = I_upchan_norm[f, off_start:off_end]
                if np.all(np.isnan(I_off)): continue
                mean_off = np.nanmean(I_off); std_off = np.nanstd(I_off)
                if not np.isnan(mean_off):
                    if std_off > 0 and not np.isnan(std_off):
                        I_norm_final[f,:] = (I_upchan_norm[f,:]-mean_off)/std_off; off_std[f]=std_off
                    else: I_norm_final[f,:] = I_upchan_norm[f,:]-mean_off; off_std[f]=0.0
            mask_bad = np.isnan(off_std) | (off_std <= 0); I_norm_final[mask_bad,:]=np.nan

        # Apply RFI mask
        if self.scallop_rfi_inds is not None:
            valid_rfi = self.scallop_rfi_inds[self.scallop_rfi_inds < nfreq_up]
            if len(valid_rfi)>0: print(f"Applying {len(valid_rfi)} RFI indices."); I_norm_final[valid_rfi,:]=np.nan

        I_final_masked = np.ma.masked_invalid(I_norm_final)

        # Calculate spectra
        on_s = min(lims_block[0], I_final_masked.shape[1]); on_e = min(lims_block[1], I_final_masked.shape[1])
        off_s = min(off_start, I_final_masked.shape[1]); off_e = min(off_end, I_final_masked.shape[1])
        self.spec_on = np.ma.mean(I_final_masked[:, on_s:on_e], axis=1) if on_s<on_e else np.ma.masked_all(nfreq_up)
        self.spec_off = np.ma.mean(I_final_masked[:, off_s:off_e], axis=1) if off_s<off_e else np.ma.masked_all(nfreq_up)
        if on_s < on_e:
            prof_on = np.ma.mean(I_final_masked[:, on_s:on_e], axis=0)
            if prof_on.count()>0: peak_idx = on_s + np.ma.argmax(prof_on); self.spec_peak = I_final_masked[:, peak_idx]
            else: self.spec_peak = np.ma.masked_all(nfreq_up)
        else: self.spec_peak = np.ma.masked_all(nfreq_up)
        print("Normalized spectra calculated (on, peak, off).")

        # Plot spectra
        if self.output_dir or matplotlib.get_backend() != 'agg': self._plot_norm_spectra()
        return self.spec_on, self.spec_peak, self.spec_off

    def _plot_norm_spectra(self):
        """ Plots the normalized spectra. """
        plt.close('all'); fig,ax=plt.subplots(2,1,sharex=True, figsize=(12,8))
        freq_axis = self.upchan_freqs if self.upchan_freqs is not None else np.arange(len(self.spec_on))
        ax[0].plot(freq_axis, self.spec_on.filled(np.nan), 'k', alpha=0.7, lw=0.8, label='On Burst')
        ax[0].plot(freq_axis, self.spec_off.filled(np.nan), 'grey', alpha=0.7, lw=0.8, label='Off Burst')
        ax[1].plot(freq_axis, self.spec_peak.filled(np.nan), 'r', alpha=0.7, lw=0.8, label='Peak')
        ax[1].plot(freq_axis, self.spec_off.filled(np.nan), 'grey', alpha=0.7, lw=0.8, label='Off Burst')
        ax[0].legend(); ax[1].legend(); ax[1].set_xlabel('Frequency [MHz]')
        ax[0].set_ylabel('Norm. Intensity'); ax[1].set_ylabel('Norm. Intensity')
        ax[0].set_title(f'Event {self.event_id} - Normalized Spectra'); ax[0].grid(True, alpha=0.3); ax[1].grid(True, alpha=0.3)
        valid_y = np.concatenate([arr.compressed() for arr in [self.spec_on, self.spec_off, self.spec_peak] if hasattr(arr,'count') and arr.count() > 0])
        if len(valid_y)>0: ymin,ymax=np.nanpercentile(valid_y,[1,99]); yrange=max(0.1,ymax-ymin); lim=(ymin-0.1*yrange,ymax+0.1*yrange); ax[0].set_ylim(lim); ax[1].set_ylim(lim)
        plt.tight_layout()
        if self.output_dir: fname=f"{self.output_dir}/normalized_spectra_evt{self.event_id}.png"; plt.savefig(fname); print(f"Saved spectra plot: {fname}"); plt.close(fig)
        elif matplotlib.get_backend() != 'agg': plt.show()
        else: plt.close(fig)


    # --- ACF Calculation ---
    def calculate_acf(self, spec=None, maxlag=20.0, use_peak_spec=True, plot=True):
        """
        Calculates the Auto-Correlation Function (ACF) of a specified spectrum.

        Parameters
        ----------
        spec : np.ma.MaskedArray, optional
            Spectrum to analyze [upchan_freq]. Defaults to `self.spec_peak`
            if `use_peak_spec` is True, else `self.spec_on`.
        maxlag : float, optional
            Maximum frequency lag [MHz] for ACF calculation. Default 20.0.
        use_peak_spec : bool, optional
            If spec is None, use `self.spec_peak` if True, else use `self.spec_on`. Default True.
        plot : bool, optional
            Generate diagnostic plots of the ACF. Default True.

        Returns
        -------
        tuple
            (acf, lags_mhz) containing:
            - acf : np.ndarray - ACF values for positive and negative lags (excluding zero).
            - lags_mhz : np.ndarray - Corresponding frequency lags in MHz.
            Returns empty arrays if calculation fails. Results are also stored
            in `self.acf_peak`/`self.acf_on` and `self.acf_lags_mhz`.
        """
        if spec is None:
            spec = self.spec_peak if use_peak_spec else self.spec_on
            spec_name = "Peak" if use_peak_spec else "On-Burst"
            if spec is None: raise ValueError(f"{spec_name} spectrum not calculated yet.")
        else: spec_name = "Provided"
        if self.upchan_freqs is None or len(self.upchan_freqs) < 2: raise ValueError("Upchannel freqs needed.")
        if not hasattr(spec, 'count') or spec.count() == 0: print(f"Warning: Spectrum '{spec_name}' empty/masked."); return np.array([]), np.array([])

        print(f"\n--- Calculating ACF for {spec_name} Spectrum (maxlag={maxlag} MHz) ---")
        f_res_upchan = np.abs(np.diff(self.upchan_freqs)).mean()
        if f_res_upchan <= 0 or np.isnan(f_res_upchan): raise ValueError("Invalid freq resolution.")
        print(f"Upchannelized Freq Resolution: {f_res_upchan:.5f} MHz/channel")
        maxlag_bin = int(maxlag / f_res_upchan) if maxlag is not None else None

        acf_pos = self._autocorr(spec, maxlag=maxlag_bin, offspec_mean=0.0)
        if len(acf_pos) == 0 or np.all(np.isnan(acf_pos)): print("ACF calc failed."); return np.array([]), np.array([])

        acf_full = np.concatenate((acf_pos[::-1], acf_pos))
        lags_pos_bins = np.arange(1, len(acf_pos) + 1)
        lags_mhz = np.concatenate((-lags_pos_bins[::-1], lags_pos_bins)) * f_res_upchan

        if use_peak_spec and spec_name != "Provided": self.acf_peak = acf_full; self.acf_lags_mhz = lags_mhz
        elif not use_peak_spec and spec_name != "Provided": self.acf_on = acf_full
        if self.acf_lags_mhz is None: self.acf_lags_mhz = lags_mhz
        if plot and (self.output_dir or matplotlib.get_backend() != 'agg'): self._plot_acf(acf_full, lags_mhz, spec_name, maxlag)
        return acf_full, lags_mhz

    def _autocorr(self, spec, maxlag=None, offspec_mean=0.0):
        """
        Internal ACF calculation helper.

        Calculates ACF for positive lags (excluding zero lag). Assumes input
        spectrum `spec` is already appropriately mean-subtracted. Normalizes
        by the variance of the input spectrum.

        Parameters
        ----------
        spec : np.ma.MaskedArray or np.ndarray
            1D input spectrum (should be mean-subtracted).
        maxlag : int, optional
            Maximum lag in channels (bins). Defaults to N-1.
        offspec_mean : float, optional
            DEPRECATED/UNUSED here. Normalization uses variance of input `spec`.

        Returns
        -------
        np.ndarray
            ACF for positive lags (lag 1 to maxlag). Returns empty array on error.
        """
        nchan = len(spec)
        if isinstance(spec, np.ma.MaskedArray): mask = ~spec.mask; x = spec.data.copy()
        else: x = np.copy(spec); mask = np.ones(nchan, dtype=bool)
        nan_mask = np.isnan(x); mask &= (~nan_mask); x[~mask] = np.nan
        if np.sum(mask) == 0: return np.array([])

        denom = np.nanvar(x)
        if denom == 0 or np.isnan(denom): return np.array([])
        x_meansub = x # Assume input is already mean-subtracted

        num_lags_calc = min(maxlag, nchan) if maxlag is not None else nchan
        acf_len = num_lags_calc - 1
        if acf_len <= 0: return np.array([])
        ACF = np.zeros(acf_len); mask_float = mask.astype(float)

        lag_iterator = range(1, num_lags_calc)
        if 'tqdm' in sys.modules: lag_iterator = tqdm(lag_iterator, desc=f"Calculating ACF", leave=False, ncols=80)
        for i in lag_iterator:
            shifted_mask = _shift(mask_float, i, nchan); shifted_x = _shift(x_meansub, i, nchan)
            unshifted_mask = _shift(mask_float, 0, nchan); unshifted_x = _shift(x_meansub, 0, nchan)
            overlap_mask = unshifted_mask * shifted_mask
            numerator = np.nansum(unshifted_x * shifted_x * overlap_mask)
            num_overlap = np.sum(overlap_mask)
            ACF[i-1] = numerator / (num_overlap * denom) if num_overlap > 0 else np.nan
        return ACF

    def _plot_acf(self, acf, lags_mhz, label, maxlag_plot=None):
        """ Helper to plot ACF. """
        if len(lags_mhz) == 0 or len(acf) == 0: 
            return
        
        plt.close('all')
        
        fig,ax=plt.subplots(3,1, figsize=(10, 9), sharex=True)
        fig.suptitle(f'Event {self.event_id} - ACF ({label})')
        f_res = np.abs(np.diff(lags_mhz)).mean() if len(lags_mhz)>1 else 0
        ax[0].plot(lags_mhz, acf, drawstyle='steps-mid', color='orange', label=f'{label} (res={f_res*1000:.1f} kHz)')
        ax[0].set_ylabel('ACF'); ax[0].legend(); ax[0].grid(True, alpha=0.3)
        
        if maxlag_plot: 
            ax[0].set_xlim(-maxlag_plot, maxlag_plot)
        else: 
            ax[0].set_xlim(lags_mhz.min(), lags_mhz.max())
            
        ax[1].plot(lags_mhz, acf, drawstyle='steps-mid', color='orange')
        ax[1].set_xlim(-3, 3)
        ax[1].set_ylabel('ACF (Zoom 1)')
        ax[1].grid(True, alpha=0.3)
        ax[2].plot(lags_mhz, acf, drawstyle='steps-mid', color='orange')
        ax[2].set_xlim(-0.5, 0.5)
        ax[2].set_ylabel('ACF (Zoom 2)')
        ax[2].grid(True, alpha=0.3)
        ax[2].set_xlabel('Frequency Lag [MHz]')
        valid_acf = acf[~np.isnan(acf)]
        ymin,ymax=np.min(valid_acf),np.max(valid_acf) if len(valid_acf)>0 else (-1,1)
        yrange=max(0.1,ymax-ymin)
        lim=(ymin-0.1*yrange,ymax+0.1*yrange)
        ax[0].set_ylim(lim)
        ax[1].set_ylim(lim)
        ax[2].set_ylim(lim)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if self.output_dir: 
            fname=f"{self.output_dir}/ACF_{label}_evt{self.event_id}.png"
            plt.savefig(fname)
            print(f"Saved ACF plot: {fname}")
            plt.close(fig)
        elif matplotlib.get_backend() != 'agg': 
            plt.show()
        else: 
            plt.close(fig)


    # --- Subband Analysis ---
    def analyze_subbands(self, num_subbands=8, maxlag_sub=5.0, snsubband=False,
                         use_peak_spec=True, fit_scint_bw=True, plot=True):
        """
        Divides the spectrum into subbands and analyzes the ACF for each.

        Parameters
        ----------
        num_subbands : int, optional
            Number of subbands to divide the spectrum into. Default 8.
        maxlag_sub : float, optional
            Maximum ACF lag [MHz] for subband analysis and plotting. Default 5.0.
        snsubband : bool, optional
            If True, divide into subbands based on equal signal-to-noise (flux)
            rather than equal number of channels. Default False.
        use_peak_spec : bool, optional
            Use the peak spectrum (`self.spec_peak`) for analysis if True,
            otherwise use the on-burst average (`self.spec_on`). Default True.
        fit_scint_bw : bool, optional
            Attempt to fit a Lorentzian model to each subband ACF to estimate
            the scintillation bandwidth. Default True.
        plot : bool, optional
            Generate summary plots (overlaid ACFs, scint bw vs freq). Default True.

        Returns
        -------
        dict
            Dictionary containing subband results keyed by subband index:
            'acfs': list of ACF arrays
            'lags': list of lag arrays (MHz)
            'fcents': list of subband center frequencies (MHz)
            'sn': list of total flux in subband
            'mask_counts': list of masked channel counts
            'lengths': list of subband channel lengths
            'fits': list of lmfit ModelResult objects (or None if fit failed/skipped)
            This dictionary is also stored in `self.subband_results[spec_name]`.
            Returns empty dict if analysis cannot proceed.
        """
        spec = self.spec_peak if use_peak_spec else self.spec_on; spec_name = "Peak" if use_peak_spec else "On-Burst"
        if spec is None or spec.count() == 0: print(f"Warning: {spec_name} spectrum empty. Skipping subbands."); return {}
        if self.upchan_freqs is None or len(self.upchan_freqs) < 2: print("Warning: Upchan freqs needed. Skipping subbands."); return {}

        print(f"\n--- Analyzing Subbands ({num_subbands}) for {spec_name} Spectrum ---")
        freqs = self.upchan_freqs; freqids = self.upchan_freq_ids; offspec_norm = self.spec_off
        results = self._acf_per_subband(spec, freqs, freqids, offspec_norm, num_subbands, maxlag_sub, snsubband, fit_scint_bw, plot)
        self.subband_results[spec_name] = results
        return results

    def _acf_per_subband(self, spec, freqs, freqids, offspec, num_subbands,
                         maxlag, snsubband, fit_scint_bw, plot):
        """ Helper for subband ACF analysis and plotting. """
        nchan = len(spec)
        if not isinstance(spec, np.ma.MaskedArray): spec = np.ma.masked_invalid(spec)
        mask_bool = ~spec.mask
        results = {'acfs': [], 'lags': [], 'fcents': [], 'sn': [], 'mask_counts': [], 'lengths': [], 'fits': []}
        total_valid_flux = np.sum(spec.data[mask_bool]) if snsubband else 0
        if snsubband and total_valid_flux <= 0: snsubband = False
        flux_per_subband = total_valid_flux / float(num_subbands) if snsubband else 0
        start_idx = 0

        for sub in range(num_subbands):
            end_idx = nchan
            if snsubband: # Determine end index by flux
                if sub == num_subbands - 1: end_idx = nchan
                else:
                    current_flux = 0; idx = start_idx
                    while current_flux < flux_per_subband and idx < nchan - (num_subbands - 1 - sub):
                        if mask_bool[idx]: current_flux += spec.data[idx]
                        idx += 1
                    end_idx = idx
            else: # Determine end index by channel count
                end_idx = start_idx + (nchan - start_idx) // (num_subbands - sub)
                if sub == num_subbands - 1: end_idx = nchan
            end_idx = min(end_idx, nchan)
            if start_idx >= nchan or start_idx >= end_idx: continue

            spec_sub = spec[start_idx:end_idx]; freqs_sub = freqs[start_idx:end_idx]; freqids_sub = freqids[start_idx:end_idx]
            offspec_sub = offspec[start_idx:end_idx] if offspec is not None else None
            results['lengths'].append(len(spec_sub)); results['mask_counts'].append(np.sum(spec_sub.mask)); results['sn'].append(np.sum(spec_sub.data[~spec_sub.mask]))
            valid_freqs_sub = freqs_sub[~spec_sub.mask]; fcent_sub = np.mean(valid_freqs_sub) if len(valid_freqs_sub)>0 else np.nan; results['fcents'].append(fcent_sub)
            print(f"Subband {sub+1}: Freq {fcent_sub:.1f} MHz, Chans {start_idx}-{end_idx-1}")

            if spec_sub.count() < 3: acf_sub, lags_sub = None, None
            else: acf_sub, lags_sub = self.calculate_acf(spec=spec_sub, maxlag=maxlag, plot=False) # Call internal ACF method

            if acf_sub is None or len(acf_sub)==0: results['acfs'].append(np.array([np.nan])); results['lags'].append(np.array([np.nan])); results['fits'].append(None)
            else:
                results['acfs'].append(acf_sub); results['lags'].append(lags_sub)
                fit_res_sub = None
                if fit_scint_bw: # Fit Lorentzian
                    try:
                        f_res_sub = np.abs(np.diff(lags_sub)).mean() if len(lags_sub)>1 else 0; lagrange_fit = max(f_res_sub*3, min(maxlag/2.0, 0.5))
                        fit_mask = (lags_sub >= -lagrange_fit) & (lags_sub <= lagrange_fit)
                        if np.sum(fit_mask) >= 3:
                            gmodel = Model(lorentz_w_c); acf_fit = acf_sub[fit_mask]; lags_fit = lags_sub[fit_mask]
                            init_g = lagrange_fit/4.0; init_m_sq=np.nanmax(acf_fit); init_m=np.sqrt(max(0,init_m_sq)) if not np.isnan(init_m_sq) else 1.0; init_c=np.nanmin(acf_fit) if not np.all(np.isnan(acf_fit)) else 0.0
                            params=gmodel.make_params(gamma=init_g,m=init_m,c=init_c); params['gamma'].set(min=f_res_sub/10.0,max=maxlag*2); params['m'].set(min=0)
                            fit_res_sub=gmodel.fit(acf_fit,params=params,x=lags_fit,weights=1.0/np.sqrt(np.clip(np.abs(acf_fit),1e-6,None)))
                        else: print("  Skipping fit: not enough points.")
                    except Exception as e: print(f"  Subband fit failed: {e}")
                results['fits'].append(fit_res_sub)
            start_idx = end_idx

        if plot and (self.output_dir or matplotlib.get_backend() != 'agg'):
            self._plot_subband_acfs(results, num_subbands, maxlag)
            if fit_scint_bw: self._plot_subband_scintbw(results, self.upchan_freqs)
        return results

    def _plot_subband_acfs(self, results, num_subbands, maxlag_plot):
        """ Helper to plot overlaid subband ACFs. """
        if len(results['fcents']) == 0: return
        plt.close('all'); fig = plt.figure(figsize=(10, 8)); cmap = matplotlib.cm.get_cmap('plasma'); max_val, min_val = -np.inf, np.inf; n_subs_plot = len(results['fcents'])
        for i in range(n_subs_plot):
            acf_i = results['acfs'][i]; lags_i = results['lags'][i]
            if lags_i is not None and acf_i is not None and len(lags_i) > 0 and len(acf_i) > 0 and not np.all(np.isnan(acf_i)):
                rgba = cmap(i / float(n_subs_plot)); offset = 1.0 * i
                plt.plot(lags_i, acf_i + offset, drawstyle='steps-mid', color=rgba, lw=1.5, alpha=0.8, label=f"{results['fcents'][i]:.1f} MHz")
                max_val = max(max_val, np.nanmax(acf_i + offset)); min_val = min(min_val, np.nanmin(acf_i + offset))
                fit_res = results['fits'][i]
                if fit_res is not None:
                    valid_lags_mask = ~np.isnan(lags_i)
                    if np.any(valid_lags_mask): lags_eval = lags_i[valid_lags_mask]; fit_curve = fit_res.eval(x=lags_eval); plt.plot(lags_eval, fit_curve + offset, 'k', lw=0.8, alpha=0.7)
        plt.xlabel('Frequency Lag [MHz]'); plt.ylabel('ACF + Offset'); plt.title(f'Event {self.event_id} - ACF per Subband ({num_subbands})')
        if maxlag_plot: plt.xlim(-maxlag_plot, maxlag_plot)
        elif len(results['lags']) > 0 and results['lags'][0] is not None and len(results['lags'][0]) > 0: all_lags = np.concatenate([l for l in results['lags'] if l is not None and len(l)>0]); plt.xlim(np.nanmin(all_lags), np.nanmax(all_lags))
        if np.isfinite(min_val) and np.isfinite(max_val): plt.ylim(min_val-0.5, max_val+0.5)
        else: plt.ylim(-1, n_subs_plot+1)
        plt.legend(loc='upper left', fontsize='small'); plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout()
        if self.output_dir: fname=f"{self.output_dir}/subband_acfs_evt{self.event_id}.png"; plt.savefig(fname); print(f"Saved subband ACF plot: {fname}"); plt.close(fig)
        elif matplotlib.get_backend() != 'agg': plt.show()
        else: plt.close(fig)

    def _plot_subband_scintbw(self, results, full_freq_axis):
        """ Helper to plot fitted scintillation bandwidth vs frequency. """
        bw_vals, bw_errs, cent_freqs = [], [], []
        if self.upchan_freqs is not None and len(self.upchan_freqs) > 1: 
            f_res_upchan = np.abs(np.diff(self.upchan_freqs)).mean()
        else: 
            f_res_upchan = 0
        for i, fit_res in enumerate(results['fits']):
            if fit_res is not None and fit_res.params is not None and 'gamma1' in fit_res.params:
                gamma = fit_res.params['gamma1']
                if gamma.value is not None and not np.isnan(gamma.value): 
                    bw_vals.append(np.abs(gamma.value))
                    bw_errs.append(gamma.stderr if gamma.stderr is not None and not np.isnan(gamma.stderr) else 0)
                    cent_freqs.append(results['fcents'][i])
        if len(cent_freqs) == 0: 
            print("No valid subband fits to plot scint bw.")
            return
        
        bw_vals=np.array(bw_vals)
        bw_errs=np.array(bw_errs)
        cent_freqs=np.array(cent_freqs)
        valid_fcent = ~np.isnan(cent_freqs)
        bw_vals=bw_vals[valid_fcent]
        bw_errs=bw_errs[valid_fcent]
        cent_freqs=cent_freqs[valid_fcent]
        
        if len(cent_freqs)==0: 
            return
        
        good_chans = (np.array(results['lengths'])-np.array(results['mask_counts']))[valid_fcent]
        bw_vals_safe=np.where(bw_vals<=0,1e-9,bw_vals)
        
        if f_res_upchan>0: 
            N_scintles=1+0.2*((good_chans*f_res_upchan)/bw_vals_safe)
            add_uncert=bw_vals/(2*np.sqrt(np.maximum(1,N_scintles)))
        else: 
            add_uncert=np.zeros_like(bw_vals)
            
        total_err = np.sqrt(bw_errs**2 + add_uncert**2)
        fit_pl = None
        
        try: # Fit power law
            if len(cent_freqs)>2: pl_model=Model(scint_freq_relation)
            params=pl_model.make_params(c=0.1,n=4.0)
            params['n'].set(min=0,max=10)
            weights=1.0/np.maximum(1e-9,total_err)
            mini=Minimizer(scint_freq_relation_min,params,fcn_args=(cent_freqs,bw_vals,total_err))
            fit_pl=mini.minimize(method='leastsq')
            print("\nPower Law Fit:")
            report_fit(fit_pl)
        except Exception as e: 
            print(f"Could not fit power law: {e}")
        
        plt.close('all')
        plt.figure()
        plt.errorbar(cent_freqs,bw_vals*1000,yerr=total_err*1000,fmt='o',color='k',markersize=4,capsize=3,label='Subband Fits')
        
        if fit_pl and fit_pl.success: 
            n_val=fit_pl.params['n'].value
            n_err=fit_pl.params['n'].stderr
            label_pl=f'Fit $\\nu^{{{n_val:.1f} \\pm {n_err:.1f}}}$' if n_err else f'Fit $\\nu^{{{n_val:.1f}}}$'
            plot_freqs=np.linspace(min(full_freq_axis),max(full_freq_axis),200)
            plt.plot(plot_freqs,scint_freq_relation(plot_freqs,fit_pl.params['c'],n_val)*1000,color='r',label=label_pl)
        
        if f_res_upchan>0: 
            plt.axhline(f_res_upchan*1000,color='grey',ls='--',label=f'Freq Res ({f_res_upchan*1000:.1f} kHz)')
            
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('Scint BW [kHz]')
        plt.title(f'Event {self.event_id} - Scint BW vs Freq')
        plt.yscale('log'); plt.grid(True,alpha=0.5)
        plt.legend()
        plt.tight_layout()
        
        if self.output_dir: 
            fname=f"{self.output_dir}/scintbw_vs_freq_evt{self.event_id}.png"
            plt.savefig(fname)
            print(f"Saved scint bw plot: {fname}")
            plt.close()
        
        elif matplotlib.get_backend() != 'agg': 
            plt.show()
        else: 
            plt.close()


    # --- Pipeline Execution ---
    def run_full_pipeline(self, fftsize=None, downfreq=1, # Upchannel params
                         num_subbands=8, maxlag_acf=20.0, maxlag_sub=5.0, # Analysis params
                         interactive_time=False, zap_extra=True, # Preprocessing params
                         spec_lims=None, # Freq limits
                         use_peak_spec_acf=True, use_peak_spec_subband=True, # Which spec to analyze
                         manual_off_burst_range_native=None):
        """
        Runs the full analysis pipeline: load, preprocess, upchannel, normalize, ACF, subbands.

        Orchestrates the main analysis steps in sequence.

        Parameters
        ----------
        fftsize : int, optional
            FFT size for upchannelization. If None, attempts to determine from data.
        downfreq : int, optional
            Downsampling factor for upchannelization. Default 1.
        num_subbands : int, optional
            Number of subbands for analysis. Default 8.
        maxlag_acf : float, optional
            Max lag [MHz] for full ACF calculation. Default 20.0.
        maxlag_sub : float, optional
            Max lag [MHz] for subband ACF calculation. Default 5.0.
        interactive_time : bool, optional
            Use interactive time selection in preprocessing (Not recommended). Default False.
        zap_extra : bool, optional
            Perform extra RFI flagging during preprocessing. Default True.
        spec_lims : tuple, optional
            Frequency channel limits [start, end] (absolute 0-1023 indices). Default None (auto-detect).
        use_peak_spec_acf : bool, optional
            Calculate full ACF on peak spectrum instead of on-burst average. Default True.
        use_peak_spec_subband : bool, optional
            Analyze subbands on peak spectrum instead of on-burst average. Default True.
        manual_off_burst_range_native : tuple, optional
            Specify the off-burst window [start_bin, end_bin] in *native* time bins.
            If provided, overrides automatic off-burst selection for scallop model.
            Default None.

        Returns
        -------
        dict
            Dictionary containing key results (e.g., 'acf_full', 'acf_lags_mhz',
            'subbands') or an 'error' key if the pipeline failed.
        """
        print(f"\n=== Running Full Scintillation Pipeline for Event {self.event_id} ===")
        results = {}
        try:
            # 1. Preprocess ON-burst data (Same as before)
            self.preprocess_data(interactive=interactive_time, select_off_burst=False,
                                 zap_extra=zap_extra, spec_lims=spec_lims)
            if self.processed_data is None or self.processed_data.size == 0:
                 raise ValueError("Preprocessing resulted in empty data. Cannot proceed.")
            
            auto_fftsize = False # Determine fftsize if needed
            if fftsize is None:
                auto_fftsize = True; print("Determining optimal fftsize...")
                try:
                    power_proc = np.abs(self.processed_data)**2; pol_axis = 1 if self.processed_data.ndim==3 else None
                    I_proc = np.ma.sum(power_proc, axis=pol_axis) if pol_axis is not None else power_proc
                    prof_nat = np.ma.mean(I_proc, axis=0)
                    lims_env = self._get_burst_envelope(prof_nat.filled(np.nanmedian(prof_nat)), thres=5, pad=0)
                    dur = lims_env[1]-lims_env[0]; dur = max(1, dur) # Ensure positive duration
                    poss_fft = np.array([2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384])
                    fftsize = poss_fft[np.argmin(np.abs(poss_fft - dur))]
                    fftsize = min(fftsize, self.processed_data.shape[-1]); fftsize = max(2, fftsize)
                    print(f"Est. duration ~{dur} bins. Using fftsize={fftsize}")
                    if downfreq != 1 and auto_fftsize: print(f"Warning: Auto-fftsize used. Setting downfreq=1."); downfreq = 1
                except Exception as e: print(f"Warning: Auto-fftsize failed ({e}). Using default."); fftsize=self.DEFAULT_FFTSIZE; downfreq=self.DEFAULT_DOWNFREQ

            self.upchannelize(fftsize=fftsize, downfreq=downfreq) # Upchannel on-burst
            if self.upchan_data is None or self.upchan_data.size == 0: raise ValueError("Upchannelization failed.")

            self.make_scallop_model(off_burst_data=None, # Trigger auto-generation or use manual range
                                    fftsize=fftsize,
                                    downfreq=downfreq,
                                    manual_off_burst_range_native=manual_off_burst_range_native
                                   )
            if self.scallop_model is None:
                 raise ValueError("Failed to generate scallop model. Cannot proceed.")

            self.calculate_normalized_spectra() # Normalize & calc spectra
            if self.spec_on is None or self.spec_peak is None: raise ValueError("Normalized spectra calculation failed.")

            acf, lags = self.calculate_acf(maxlag=maxlag_acf, use_peak_spec=use_peak_spec_acf) # Full ACF
            results['acf_full'] = acf; results['acf_lags_mhz'] = lags; results['acf_spec_type'] = 'Peak' if use_peak_spec_acf else 'On-Burst'

            sub_results = self.analyze_subbands(num_subbands=num_subbands, maxlag_sub=maxlag_sub, use_peak_spec=use_peak_spec_subband, fit_scint_bw=True) # Subbands
            results['subbands'] = sub_results; results['subband_spec_type'] = 'Peak' if use_peak_spec_subband else 'On-Burst'

            print(f"\n=== Pipeline Finished Successfully for Event {self.event_id} ===")
        except Exception as e:
            print(f"\n*** Pipeline ERROR for Event {self.event_id}: {e} ***"); import traceback; traceback.print_exc(); results['error'] = str(e)
        return results


