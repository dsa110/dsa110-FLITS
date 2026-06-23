# -*- coding: utf-8 -*-
"""
FRB preprocessing functions.
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

class BBProcessor:
    """
    Class to manage the baseband processing pipeline for an FRB event.

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
    spec_on, spec_peak, spec_off : np.ma.MaskedArray or None
        Calculated normalized spectra.
    master_api : FRBMaster object or None
        Connection to the CHIME FRB Master API.
    auth : dict or None
        Authorization token for CHIME API.
    """
    # Class attributes for default values or constants if needed
    DEFAULT_DOWNFREQ = 1

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

        print(f"Initialized BBProcessor for Event {self.event_id}, DM {self.dm}")

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

        # Check if dedispersion is needed (target DM is different from current DM)
        # Use a tolerance for floating point comparison
        if self.dm != 0:
            print(f"Applying coherent dedispersion (DM={self.dm})...")
            try:
                # coherent_dedisp modifies bbdata[baseband_key] in place when write=True
                coherent_dedisp(bbdata, self.dm, time_shift=False, write=True)
                print(f"Coherent dedispersion applied in-place to '{baseband_key}'.")
                data_dedisp, freq, freq_id = incoherent_dedisp(bbdata, self.dm, fill_wfall=False)
                print(f"Incoherent dedispersion applied.")
            except Exception as e:
                raise RuntimeError(f"Coherent dedispersion failed: {e}")
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
        t_start = 0 #max(0, int(valid_time_bins_native[0]))
        t_end = data_masked_tmp.shape[-1] #min(data_masked_tmp.shape[-1], int(valid_time_bins_native[1])) # Use shape[-1] for time axis
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

        #f_start, f_end = int(spec_lims[0]), int(spec_lims[1])
        f_start, f_end = int(0), int(1024)
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
            plt.close('all')
            #mean = np.nanmean(scrunch(I_native, ds_factor, 1))
            #std = np.nanstd(scrunch(I_native, ds_factor, 1))
            #plt.imshow(scrunch(I_native, ds_factor, 1), vmin = mean - 3*std, vmax = mean + 3*std, aspect='auto')
            plt.plot(profile_scr.filled(np.nanmedian(profile_scr)))
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
        if self.dm != 0:
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


    # --- Pipeline Execution ---
    def run_full_pipeline(self, downfreq=1, # Upchannel params
                         interactive_time=False, zap_extra=True, # Preprocessing params
                         spec_lims=None, # Freq limits
                         ):
        """
        Runs the full analysis pipeline: load, preprocess, upchannel, normalize, ACF, subbands.

        Orchestrates the main analysis steps in sequence.

        Parameters
        ----------
        fftsize : int, optional
            FFT size for upchannelization. If None, attempts to determine from data.
        downfreq : int, optional
            Downsampling factor for upchannelization. Default 1.
        interactive_time : bool, optional
            Use interactive time selection in preprocessing (Not recommended). Default False.
        zap_extra : bool, optional
            Perform extra RFI flagging during preprocessing. Default True.
        spec_lims : tuple, optional
            Frequency channel limits [start, end] (absolute 0-1023 indices). Default None (auto-detect).

        Returns
        -------
        dict
            Dictionary containing key results (e.g., 'acf_full', 'acf_lags_mhz',
            'subbands') or an 'error' key if the pipeline failed.
        """
        print(f"\n=== Running Preprocessing Pipeline for Event {self.event_id} ===")
        # 1. Preprocess ON-burst data (Same as before)
        self.preprocess_data(interactive=interactive_time, select_off_burst=False,
                             zap_extra=zap_extra, spec_lims=spec_lims)
        if self.processed_data is None or self.processed_data.size == 0:
             raise ValueError("Preprocessing resulted in empty data. Cannot proceed.")
                
        power_proc = np.abs(self.processed_data)**2; pol_axis = 1 if self.processed_data.ndim==3 else None
        I_proc = np.ma.sum(power_proc, axis=pol_axis) if pol_axis is not None else power_proc
        prof_nat = np.ma.mean(I_proc, axis=0)
        lims_env = self._get_burst_envelope(prof_nat.filled(np.nanmedian(prof_nat)), thres=5, pad=0)
        dur = lims_env[1]-lims_env[0]; dur = max(1, dur) # Ensure positive duration
            
        return I_proc, prof_nat, lims_env, dur


