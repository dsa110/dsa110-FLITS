# ==============================================================================
# File: scint_analysis/scint_analysis/core.py
# ==============================================================================
import numpy as np
import logging
from typing import Union
from astropy.stats import sigma_clip
from scipy.ndimage import label
from scipy.stats import median_abs_deviation
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set up a logger for this module
log = logging.getLogger(__name__)

class DynamicSpectrum:
    """
    A class to represent and interact with a dynamic spectrum (freq, time).
    """
    def __init__(self, power_2d, frequencies_mhz, time_axis_s):
        """
        Initializes the DynamicSpectrum object.

        Args:
            power_2d (np.ndarray): 2D array of shape (num_channels, num_timesteps).
            frequencies_mhz (np.ndarray): 1D array of frequencies in MHz.
            time_axis_s (np.ndarray): 1D array of time values in seconds.
        """
        log.info("Initializing DynamicSpectrum object.")
        
        # --- Data Validation using Assertions ---
        assert isinstance(power_2d, np.ndarray), "power_2d must be a NumPy array."
        assert power_2d.ndim == 2, f"power_2d must be 2-dimensional, but got {power_2d.ndim} dimensions."
        assert power_2d.shape[0] == len(frequencies_mhz), "Frequency axis length does not match power_2d shape."
        assert power_2d.shape[1] == len(time_axis_s), "Time axis length does not match power_2d shape."

        self.power = np.ma.masked_invalid(power_2d, copy=True) # Work with masked arrays internally
        self.frequencies = frequencies_mhz
        self.times = time_axis_s
        
        # Immediately after you assign self.power, self.frequencies, self.times
        if self.frequencies[0] > self.frequencies[-1]:
            self.frequencies = self.frequencies[::-1]
            self.power       = self.power[::-1, :]         # flip channel axis

        
        log.info(f"Spectrum shape: ({self.num_channels}, {self.num_timesteps})")

    @classmethod
    def from_numpy_file(cls, filepath):
        """
        Class method to load a dynamic spectrum from a generic .npz file.
        
        The .npz file must contain keys: 'power_2d', 'frequencies_mhz', 'times_s'.
        """
        log.info(f"Loading DynamicSpectrum from file: {filepath}")
        try:
            with np.load(filepath) as data:
                power = np.flip(data['power_2d'], axis=0)
                freqs = data['frequencies_mhz']
                times = data['times_s']
                return cls(power, freqs, times)
        except FileNotFoundError:
            log.error(f"File not found: {filepath}")
            raise
        except KeyError as e:
            log.error(f"Missing required key in {filepath}: {e}")
            raise
            
    # --- Properties for easy metadata access ---
    @property
    def num_channels(self):
        return self.power.shape[0]

    @property
    def num_timesteps(self):
        return self.power.shape[1]

    @property
    def channel_width_mhz(self):
        return np.abs(np.mean(np.diff(self.frequencies)))

    @property
    def time_resolution_s(self):
        return np.abs(np.mean(np.diff(self.times)))
        
    # --- Core Methods ---
    def get_profile(self, time_window_bins=None):
        """
        Returns the 1D frequency-averaged time series.
        
        Args:
            time_window_bins (tuple): A tuple (start_bin, end_bin).
        """
        log.debug("Calculating frequency-averaged profile.")
        if time_window_bins is not None:
            start, end = time_window_bins
            return np.ma.mean(self.power[:, start:end], axis=0)
        else:
            return np.ma.mean(self.power, axis=0)
        
    def get_spectrum(self, time_window_bins):
        """
        Returns a 1D time-averaged spectrum from a specified window.
        
        Args:
            time_window_bins (tuple): A tuple (start_bin, end_bin).
        """
        start, end = time_window_bins
        log.debug(f"Calculating time-averaged spectrum from bins {start} to {end}.")
        return np.ma.mean(self.power[:, start:end], axis=1)

    
    def find_burst_envelope(self, thres=5, downsample_factor=8, padding_factor=0.0):
        """
        Finds the full time envelope containing ALL signal above a given S/N threshold.
        Uses sigma-clipping for robust noise estimation and can apply padding.
        """
        log.info(f"Finding full signal envelope with S/N threshold > {thres} (downsample ×{downsample_factor}).")

        prof = self.get_profile().compressed()
        if downsample_factor > 1:
            n = prof.size - (prof.size % downsample_factor)
            if n == 0:
                 log.warning("Not enough data to downsample. Using full resolution profile.")
            else:
                prof = prof[:n].reshape(-1, downsample_factor).mean(axis=1)

        # --- Robust noise estimation using sigma-clipping ---
        filtered_prof = sigma_clip(prof, sigma=3, maxiters=5, masked=True)
        med = np.ma.median(filtered_prof)
        std = np.ma.std(filtered_prof)

        if std is np.ma.masked or std == 0:
            log.warning("Zero-variance profile in noise region; returning empty envelope.")
            return [0, 0]

        snr = (prof - med) / std
        mask = snr > thres
        if not np.any(mask):
            log.warning("No burst envelope found above threshold.")
            return [0, 0]

        signal_indices = np.where(mask)[0]
        first_bin_ds = signal_indices.min()
        last_bin_ds = signal_indices.max()

        # --- Apply padding to widen burst envelope ---
        if padding_factor > 0:
            duration_ds = last_bin_ds - first_bin_ds
            padding_ds = int(duration_ds * padding_factor)
            log.info(f"Applying {padding_factor*100:.1f}% padding ({padding_ds} downsampled bins) to each side.")

            first_bin_ds -= padding_ds
            last_bin_ds += padding_ds

            # Ensure we don't go out of bounds
            first_bin_ds = max(0, first_bin_ds)
            last_bin_ds = min(len(prof) - 1, last_bin_ds)

        # --- Final calculation using the (potentially padded) bins ---
        final_lims = [int(first_bin_ds * downsample_factor),
                      int((last_bin_ds + 1) * downsample_factor) - 1]

        log.info(f"Full signal envelope found between bins {final_lims[0]} and {final_lims[1]}.")
        return final_lims

    def mask_rfi(self, config):
        """
        Applies RFI masking and returns a new, cleaned DynamicSpectrum object.
        Includes a toggle for time-domain RFI flagging.
        """
        log.info("Applying RFI masking.")
        rfi_config = config.get('analysis', {}).get('rfi_masking', {})
        
        ds_factor = rfi_config.get('rfi_downsample_factor', 8)
        log.info(f"Using time downsampling factor of {ds_factor} for RFI statistical checks.")
        
        remainder = self.num_timesteps % ds_factor
        power_for_stats = self.power[:, :-remainder] if remainder > 0 else self.power
        power_ds = np.ma.mean(power_for_stats.reshape(self.num_channels, -1, ds_factor), axis=2)

        burst_lims = self.find_burst_envelope(thres=rfi_config.get('find_burst_thres', 5))
        
        # Symmetric noise window
        if rfi_config.get('use_symmetric_noise_window', False):
            on_burst_duration = burst_lims[1] - burst_lims[0]
            off_burst_end = burst_lims[0]
            off_burst_start = off_burst_end - on_burst_duration
            log.info(f"Using symmetric noise window of duration {on_burst_duration} bins.")
        else:
            # Original logic using a fixed buffer
            off_burst_end = burst_lims[0] - rfi_config.get('off_burst_buffer', 100)
            off_burst_start = 0

        # Calculate window in terms of downsampled bins
        off_burst_start_ds = off_burst_start // ds_factor
        off_burst_end_ds = off_burst_end // ds_factor

        # Ensure the window is valid
        if off_burst_start_ds < 0:
            log.warning(f"Calculated noise window start is before data start. Clipping to 0.")
            off_burst_start_ds = 0
        
        assert off_burst_end_ds > off_burst_start_ds, "Not enough off-burst data to perform RFI masking."
        log.info(f"Using downsampled noise statistics from bins {off_burst_start_ds} to {off_burst_end_ds}.")
        
        noise_data_ds = power_ds[:, off_burst_start_ds:off_burst_end_ds]
        
        # Frequency Domain Flagging
        channel_mask = np.zeros(self.num_channels, dtype=bool)
        for _ in tqdm(range(5), desc="Iterative RFI Masking in Frequency Domain"):
            masked_noise = np.ma.masked_array(noise_data_ds, mask=np.tile(channel_mask, (noise_data_ds.shape[1], 1)).T)
            means = np.ma.mean(masked_noise, axis=1)
            stds = np.ma.std(masked_noise, axis=1)
            if np.ma.is_masked(np.ma.std(means)) or np.ma.std(stds) == 0: continue
            snr_means = (means - np.ma.median(means)) / np.ma.std(means)
            snr_stds = (stds - np.ma.median(stds)) / np.ma.std(stds)
            newly_flagged = Union[(np.abs(snr_means) > rfi_config.get('freq_threshold_sigma', 5.0)), \
                            (np.abs(snr_stds) > rfi_config.get('freq_threshold_sigma', 5.0))]
            if not np.any(newly_flagged): break
            channel_mask |= newly_flagged
        
        log.info(f"Masked {np.sum(channel_mask)} channels based on frequency-domain stats.")

        # Time Domain Flagging
        if rfi_config.get('enable_time_domain_flagging', True):
            log.info("Performing time-domain RFI flagging.")
            freq_masked_power_ds = np.ma.masked_array(power_ds, mask=np.tile(channel_mask, (power_ds.shape[1], 1)).T)
            time_series_ds = np.ma.mean(freq_masked_power_ds, axis=0)
            ts_mad = median_abs_deviation(time_series_ds.compressed(), nan_policy='omit')
            if ts_mad == 0: ts_mad = 1e-9
            ts_median = np.ma.median(time_series_ds)
            robust_z = 0.6745 * (time_series_ds - ts_median) / ts_mad
            time_mask_ds = np.abs(robust_z) > rfi_config.get('time_threshold_sigma', 7.0)
            
            time_mask = np.repeat(time_mask_ds, ds_factor)
            if len(time_mask) < self.num_timesteps:
                time_mask = np.pad(time_mask, (0, self.num_timesteps - len(time_mask)), 'constant')
            log.info(f"Masked {np.sum(time_mask)} time steps based on time-domain stats.")
        else:
            log.info("Skipping time-domain RFI flagging as per configuration.")
            time_mask = np.zeros(self.num_timesteps, dtype=bool)

        # Combine masks and create new object
        
        # 1. Promote existing mask (if any) to full Boolean array
        if self.power.mask is np.ma.nomask or np.isscalar(self.power.mask):
            base_mask = np.zeros_like(self.power.data, dtype=bool)
        else:
            base_mask = self.power.mask.copy()          # keeps dtype=bool

        # 2. Broadcast new masks to (n_chan, n_time)
        chan_mask_full = np.broadcast_to(channel_mask[:, None], self.power.shape)
        time_mask_full = np.broadcast_to(time_mask[None, :], self.power.shape)

        # 3. Combine
        final_mask = Union[base_mask, chan_mask_full, time_mask_full]

        # 4. Build a *new* DynamicSpectrum so callers get a fresh object
        new_power = np.ma.MaskedArray(self.power.data.copy(), mask=final_mask)
        return DynamicSpectrum(new_power, self.frequencies.copy(), self.times.copy())

    def subtract_poly_baseline(self, off_pulse_spectrum, poly_order=1):
        """
        Fits a polynomial to an off-pulse spectrum and subtracts this baseline
        from the dynamic spectrum.

        Returns:
            DynamicSpectrum: A new DynamicSpectrum object with the baseline subtracted.
            np.ndarray: The 1D baseline model that was subtracted. ### NEW RETURN VALUE ###
        """
        log.info(f"Performing order-{poly_order} polynomial baseline subtraction using off-pulse spectrum.")
        
        # Use only valid (unmasked) data from the off-pulse spectrum to fit the baseline
        valid_mask = ~off_pulse_spectrum.mask
        if np.sum(valid_mask) < poly_order + 2:
            log.warning("Not enough valid data in off-pulse spectrum to fit baseline. Skipping subtraction.")
            return self # Return the original object if fit is not possible

        # Fit a single, stable polynomial to the time-averaged off-pulse bandpass
        coeffs = np.polyfit(
            self.frequencies[valid_mask],
            off_pulse_spectrum.compressed(), # Use .compressed() to get only valid data
            poly_order
        )
        
        # Evaluate this single baseline model across all frequencies
        baseline_model = np.poly1d(coeffs)(self.frequencies)
        
        # Subtract this 1D baseline model from the entire 2D data block.
        # NumPy broadcasting subtracts the 1D array from every column (each time step).
        new_power_data = self.power.data - baseline_model[:, np.newaxis]
        
        # Create the new object, preserving the original RFI mask
        new_power = np.ma.MaskedArray(new_power_data, mask=self.power.mask)
        
        log.info("Baseline subtraction complete.")
        return DynamicSpectrum(new_power, self.frequencies.copy(), self.times.copy()), baseline_model
    
    def __repr__(self):
        return (f"<DynamicSpectrum ({self.num_channels} channels x {self.num_timesteps} timesteps, "
                f"{self.frequencies.min():.1f}-{self.frequencies.max():.1f} MHz)>")

class ACF:
    """
    Container for a 1-D spectral autocorrelation function.
    """

    def __init__(self, acf_data, lags_mhz, acf_err=None):
        # --- validate & coerce to NumPy arrays
        acf_data = np.asarray(acf_data, dtype=float)
        lags_mhz = np.asarray(lags_mhz, dtype=float)
        if acf_err is not None:
            acf_err = np.asarray(acf_err, dtype=float)

        if acf_data.ndim != 1:
            raise ValueError("ACF data must be 1-D.")
        if len(acf_data) != len(lags_mhz):
            raise ValueError("acf_data and lags_mhz must have the same length.")
        if acf_err is not None and len(acf_err) != len(acf_data):
            raise ValueError("acf_err must match acf_data length.")

        self.acf  = acf_data
        self.lags = lags_mhz
        self.err  = acf_err          # may be None

    # optional convenience: length & repr
    def __len__(self):
        return self.acf.size

    def __repr__(self):
        return f"<ACF ({len(self)} points)>"
