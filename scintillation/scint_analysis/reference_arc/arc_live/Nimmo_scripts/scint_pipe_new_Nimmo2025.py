import h5py
import sys

from scintillation_funcs_new_Nimmo2025 import upchannel as upchan
from scintillation_funcs_new_Nimmo2025 import make_scallop_model, acf_scint_plot, acf_per_subband
from scintillation_funcs_new_Nimmo2025 import lorentz_withc_min, lorentz_w_c, doublelorentz_withc_min, doublelorentz_w_c 
from scintillation_funcs_new_Nimmo2025 import scint_freq_relation, scint_freq_relation_min
from scintillation_funcs_new_Nimmo2025 import data_dedisp_derip_filled_masked, get_burst_envelope, get_data

from mwprop.ne2001p import *
from mwprop.ne2001p.NE2001 import ne2001
from astropy.coordinates import SkyCoord
from astropy import units as u

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json 

from lmfit import minimize, Parameters, fit_report, Model, Minimizer, report_fit

from baseband_analysis.core.signal import get_main_peak_lim, get_spectrum_lim
from baseband_analysis.core.bbdata import BBData

import matplotlib.gridspec as gridspec

import scipy.constants as cons

import matplotlib as mpl

mpl.rcParams['font.size'] = 15
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['legend.fontsize'] = 15
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['xtick.major.pad']='6'
mpl.rcParams['ytick.major.pad']='6'

import chime_frb_api

master = chime_frb_api.frb_master.FRBMaster(base_url = "https://frb.chimenet.ca/frb-master")
master.API.authorize()
auth = {"Authorization": master.API.access_token}


from scipy.fftpack import fft, ifft, fftshift
import matplotlib.cm
import os


def plot_acf_diagnostic_plot(fftsize, on_spectrum, off_spectrum, freqs, time_series, time_series_limits, acf, lags, event_id, outdir, downfreq):
    """
    Generates a 6-panel diagnostic plot for burst time series, spectra, and autocorrelation function (ACF).
    
    Args:
    - fftsize (int): The FFT size used in the upchannelisation
    - downfreq (float): downfreq used for the upchannelisation
    - on_spectrum (array): The on-burst spectrum data.
    - off_spectrum (array): The off-burst spectrum data.
    - freqs (array): The freqency array [in MHz]. 
    - time_series (array): The time array [in bins].
    - time_series_limits (tuple): Indices for the on and off burst regions (start_idx, stop_idx).
    - acf (array): The autocorrelation function (ACF) values.
    - lags (array): The lag values corresponding to the ACF.
    - event_id (str): CHIME event ID (for labelling the plot).
    - outdir (str): The directory where the plot should be saved.
    
    
    Returns: None, saved the plot as f"{outdir}/{event_id}_diagnostic_plot_fftsize_{fftsize}.png"
    """
    
    f_res = 0.39101 / (fftsize // downfreq)  # MHz
    
    # Create a 2x3 grid for subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    
    # Time Series Plot (Top-left)
    start_idx, stop_idx = time_series_limits
    time_axis = np.arange(len(time_series))
    
    # Normalizing time series
    normalized_time_series = time_series / np.max(time_series)
    
    # Plot normalized time series
    axes[0, 0].plot(time_axis / len(time_axis), normalized_time_series, label='Normalized Time Series')
    axes[0, 0].axvline(start_idx / len(time_axis), color='r', linestyle='--', label='On Burst Region')
    axes[0, 0].axvline(stop_idx / len(time_axis), color='r', linestyle='--')
    axes[0, 0].legend(fontsize=12)
    axes[0, 0].set_xlabel('Fraction of Time Series', size=15)
    axes[0, 0].set_ylabel('Normalized Amplitude', size=15)
    axes[0, 0].set_title(f'Normalized Burst Profile (FFTSIZE={fftsize})', size=18)
    
    # Bottom-left: Burst Spectrum
    axes[1, 0].plot(freqs, on_spectrum, label='On-Burst Spectrum', color='blue', alpha=0.7)
    axes[1, 0].plot(freqs, off_spectrum, label='Off-Burst Spectrum', color='orange', alpha=0.7)
    axes[1, 0].legend(fontsize=12)
    axes[1, 0].set_xlabel('Frequency (MHz)', size=15)
    axes[1, 0].set_ylabel('Bandpass Corrected SNR', size=15)
    axes[1, 0].set_title('Burst Spectrum', size=18)
    
    # ACF Plots with Different Zoom Levels (Top-right and Bottom-right)
    max_lag = np.max(lags)
    zoom_levels = [max_lag, max_lag / 2, max_lag / 4, max_lag / 8]
    
    for i, zoom in enumerate(zoom_levels):
        ax = axes[i // 2, (i % 2) + 1]
        mask = np.abs(lags) <= zoom
        ax.plot(lags, acf, )
        ax.set_xlabel('Lag (MHz)', size=15)
        ax.set_ylabel('Fourier ACF', size=15)
        ax.set_title(f'Fourier ACF Zoom: ± {zoom:.2f} (MHz)', size=18)
        ax.set_xlim(-zoom, zoom)
    
    plt.suptitle(f'CHIME: {event_id} fftsize: {fftsize} Freq_res_mhz: {f_res}', size=25)
    plt.tight_layout()
    
    # Define file name
    file_name = f"{event_id}_diagnostic_plot_fftsize_{fftsize}.png"
    
    # Full file path
    file_path = os.path.join(outdir, file_name)
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.show()


def compute_acf_fourier(signal, frequencies, max_lag_mhz = 75, normalize=True):
    """
    Compute the autocorrelation function (ACF) of a signal using the Fourier method, with an option to limit the lag range.
    The spike at 0 lag is removed, and the ACF is (approx) normalized such that the (remaining) peak is ~ the modulation index.
    
    You should not use this for model fitting. The overall shape of the ACF is correct, but the normalization is slightly wrong, as well as
    scintillation bandwidth. We use this for diagnostic purposes, and do the brute force ACF later in the pipeline which is more correct. 
    
    Parameters:
        signal (np.ndarray): Burst spectrum. Can be a masked numpy array.
        frequencies (np.ndarray): Frequency axis corresponding to the signal (in units of MHz).
        max_lag_mhz (float, optional): Maximum lag in MHz to return. Default is 75 MHz.
        
    Returns:
        (np.ndarray, np.ndarray) = (acf, lags): Autocorrelation function and corresponding lag axis in MHz.
    """
    
    
    # Check if the signal is a masked array and fill masked values with zeros
    if np.ma.is_masked(signal):
        signal = signal.filled(0)
        
    # Fill NaNs with zeros
    signal = np.nan_to_num(signal)

    # Calculate frequency resolution
    freq_resolution = np.abs(frequencies[1] - frequencies[0])
    max_lag_samples = int(max_lag_mhz / freq_resolution)
    
    # Subtract the mean to remove DC component
    signal_centered = signal - np.mean(signal)
    
    # Compute FFT of the spectrum
    spectrum_fft = fft(signal_centered)
    
    # Compute the power spectrum
    power_spectrum = np.abs(spectrum_fft)**2
    
    # Inverse FFT to get the autocorrelation
    acf_full = ifft(power_spectrum).real
    
    
    # Shift zero lag to the center for symmetric plotting
    acf_shifted = fftshift(acf_full)
    
    # Create lag axis in MHz
    num_points = len(signal)
    lags_mhz = np.arange(-num_points//2, num_points//2) * freq_resolution
    
    # Remove zero lag value
    zero_lag_index = np.where(lags_mhz == 0)[0][0]
    acf_shifted_no_zero = np.delete(acf_shifted, zero_lag_index)
    lags_mhz_no_zero = np.delete(lags_mhz, zero_lag_index)


    # Normalize the ACF
    if normalize:
        # Replace 0s with NaNs
        acf_shifted_no_zero = acf_shifted_no_zero / np.nanmean(signal)**2
        acf_shifted_no_zero = acf_shifted_no_zero / (2*len(signal)) 

    # Limit to specified max lag range
    lag_mask = np.abs(lags_mhz_no_zero) <= max_lag_mhz
    acf_limited = acf_shifted_no_zero[lag_mask]
    lags_limited = lags_mhz_no_zero[lag_mask]
    
    return acf_limited, lags_limited


def save_spectrum_data(diagnostic_plots_direc, event_id, fftsize_list, on_spectrum_list, off_spectrum_list, 
                       peak_spectrum_list, freq_list, time_series_list, overwrite_npz=False):
    '''
    Save or update spectrum data in a .npz file.

    This function saves spectra data to a compressed NumPy file (.npz) located at 
    '{diagnostic_plots_direc}/{event_id}_spectrum_data.npz'. If the file already exists, it either 
    overwrites the file (if `overwrite_npz` is True) or appends new data, avoiding duplicate `fftsize` values.

    Parameters:
    -----------
    diagnostic_plots_direc : str
        Directory where the .npz file will be saved.
    event_id : str
        CHIME event ID, used to name the .npz file.
    fftsize_list : list of int
        List of FFT sizes for upchannelisation.
    on_spectrum_list : list of arrays
        List of on burst spectra (at the corresponding upchan factors: fftsizes)
    off_spectrum_list : list of arrays
        List of off burst spectra (at the corresponding upchan factors: fftsizes)
    peak_spectrum_list : list of arrays
        List of peak burst spectra (at the corresponding upchan factors: fftsizes)
    freq_list : list of arrays
        List of frequency arrays corresponding to each FFT size.
    time_series_list : list of arrays
        List of time arrays corresponding to each FFT size.
    overwrite_npz : bool, optional
        If True, overwrites the existing .npz file. If False (default), appends new data while avoiding
        duplicate `fftsize` values.

    Returns:
    --------
    None
        Saves the data to the specified .npz file.
    '''
    # Construct the file path
    npz_file = os.path.join(diagnostic_plots_direc, f'{event_id}_spectrum_data.npz')
    
    # Check if the file already exists
    if os.path.exists(npz_file):
        # If file exists and overwrite_npz is False, only add new data
        if not overwrite_npz:
            # Load existing data
            existing_data = np.load(npz_file, allow_pickle=True)
            
            # Extract existing fftsizes to avoid duplication
            existing_fftsizes = existing_data['fftsize_list'].tolist()
            
            # Find the indices of new data that does not have a matching fftsize
            new_indices = [i for i, fft in enumerate(fftsize_list) if fft not in existing_fftsizes]
            
            # Filter the new data based on the indices
            new_fftsizes = [fftsize_list[i] for i in new_indices]
            new_on_spectrum = [on_spectrum_list[i] for i in new_indices]
            new_off_spectrum = [off_spectrum_list[i] for i in new_indices]
            new_peak_spectrum = [peak_spectrum_list[i] for i in new_indices]
            new_freq = [freq_list[i] for i in new_indices]
            new_time_series = [time_series_list[i] for i in new_indices]
            
            # Append the new data to the existing data
            existing_fftsizes.extend(new_fftsizes)
            existing_on_spectrum = existing_data['on_spectrum_list'].tolist() + new_on_spectrum
            existing_off_spectrum = existing_data['off_spectrum_list'].tolist() + new_off_spectrum
            existing_peak_spectrum = existing_data['peak_spectrum_list'].tolist() + new_peak_spectrum
            existing_freq = existing_data['freq_list'].tolist() + new_freq
            existing_time_series = existing_data['time_series_list'].tolist() + new_time_series
            
            # Save the updated data with object arrays
            np.savez(npz_file, fftsize_list=existing_fftsizes, 
                     on_spectrum_list = np.array(existing_on_spectrum, dtype=object),
                     off_spectrum_list = np.array(existing_off_spectrum, dtype=object),
                     peak_spectrum_list = np.array(existing_peak_spectrum, dtype=object),
                     time_series_list = np.array(existing_time_series, dtype=object),
                     freq_list = np.array(existing_freq, dtype=object))
            
        else:
            # If overwrite_npz is True, replace all data with object arrays
            np.savez(npz_file, fftsize_list=fftsize_list, 
                     on_spectrum_list = np.array(on_spectrum_list, dtype=object),
                     off_spectrum_list = np.array(off_spectrum_list, dtype=object),
                     peak_spectrum_list=np.array(peak_spectrum_list, dtype=object),
                     time_series_list = np.array(time_series_list, dtype=object),
                     freq_list = np.array(freq_list, dtype=object))
    
    else:
        # If the file doesn't exist, simply create it with object arrays
        np.savez(npz_file, fftsize_list=fftsize_list, 
                 on_spectrum_list = np.array(on_spectrum_list, dtype=object),
                 off_spectrum_list = np.array(off_spectrum_list, dtype=object),
                 peak_spectrum_list = np.array(peak_spectrum_list, dtype=object),
                time_series_list = np.array(time_series_list, dtype=object),
                 freq_list = np.array(freq_list, dtype=object))


def load_spectrum_data(diagnostic_plots_direc, event_id):
    '''
    Load spectra data from a .npz file.

    This function loads data from a compressed NumPy file (.npz) located at
    '{diagnostic_plots_direc}/{event_id}_spectrum_data.npz' and returns the stored spectral analysis arrays.

    Parameters:
    -----------
    diagnostic_plots_direc : str
        Directory where the .npz file is stored.
    event_id : str
        CHIME event ID, used to locate the .npz file.

    Returns:
    --------
    tuple
        A tuple containing the following:
        - fftsize_list (ndarray): Array of FFT sizes used for upchannelisation.
        - on_spectrum_list (ndarray): Array of on burst spectra (at the corresponding upchan factors: fftsize_list)
        - off_spectrum_list (ndarray): Array of off burst spectra (at the corresponding upchan factors: fftsize_list)
        - peak_spectrum_list (ndarray): Array of peak burst spectra (at the corresponding upchan factors: fftsize_list)
        - freq_list (ndarray): Array of frequency arrays.
        - time_series_list (ndarray): Array of time arrays.

    Raises:
    -------
    FileNotFoundError
        If the specified .npz file does not exist.
    '''
    # Construct the file path
    npz_file = os.path.join(diagnostic_plots_direc, f'{event_id}_spectrum_data.npz')

    print(f' Loading data from {npz_file}')

    # Check if the file exists
    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"The file {npz_file} does not exist.")

    # Load the data from the .npz file
    data = np.load(npz_file, allow_pickle=True)
    print("Keys in the .npz file:", list(data.keys()))

    # Extract the arrays directly
    fftsize_list = data['fftsize_list']
    on_spectrum_list = data['on_spectrum_list']
    off_spectrum_list = data['off_spectrum_list']
    peak_spectrum_list = data['peak_spectrum_list']
    time_series_list = data['time_series_list']
    freq_list = data['freq_list']

    # Return the data as a tuple of np arrays
    return fftsize_list, on_spectrum_list, off_spectrum_list, peak_spectrum_list, freq_list, time_series_list



def make_scint_input(event_id, dm, diagnostic_plots_direc=None, fftsize=[], downfreq=1, speclims=None, overwrite_npz=False, n_fft=5, maxlag_mhz=10, fft_min=32):
    """
    Processes FRB data for scintillation analysis by dedispersing, masking, upchannelizing, and correcting for the upchannelisation scalloping artefact. This function creates spectra, and 
    writes them to a .npz file at '{diagnostic_plots_direc}/{event_id}_spectrum_data.npz'. Additonally, it creates diagnostic plots for each fftsize showing the
    spectrum, burst time-series, and Fourier ACF's. 
    
    Parameters:
    ----------
    event_id : str
        CHIME event ID
    dm : float
        Dispersion measure of the burst [pc/cc].
    diagnostic_plots_direc : str, optional
        Directory for saving diagnostic plots. Defaults to None.
    fftsize : list, optional
        List of FFT sizes for upchannelization. If empty (default), it is determined based on burst width. 
    downfreq : int, optional
        Downsampling factor for frequency reduction during upchannelization. Defaults to 1.
    speclims : tuple, optional
        Predefined frequency limits (min, max) for the burst extent in channels. If None (default) it will compute the burst extent in frequency.
    overwrite_npz : bool, optional
        If False, appends new FFT size spectrums to the .npz file if they do not already exist.
        If True, overwrites the .npz. Defaults to False.
    n_fft : int, optional
        Number of different FFT sizes to consider, ranging between fftsize = 8 and the nearest power of 2
        to fftsize = 1/burst width. Defaults to 5.
    maxlag_mhz : float, optional
        Maximum lag in MHz up to which the Fourier ACF is calculated. Defaults to 10 MHz.
    fft_min : int, optional
        Minimum FFT size to consider for upchannelization. Defaults to 32.
    
    Returns:
    -------
    None
        Saves an .npz file at '{diagnostic_plots_direc}/{event_id}_spectrum_data.npz'. The file contains:
        - `fftsize_list`: List of FFT sizes used.
        - `on_spectrum_list`: List of on-burst spectra.
        - `off_spectrum_list`: List of off-burst spectra.
        - `peak_spectrum_list`: List of peak-burst spectra.
        - `time_series_list`: List of time series data.
    """
    
    # Set directory for diagnostic plots if provided
    direc = diagnostic_plots_direc if diagnostic_plots_direc is not None else None
    
    # Load dedispersed and masked data, with different downsample factors for on and off-burst data
    if direc is None:
        data, freq, freqid = data_dedisp_derip_filled_masked(event_id, dm, downsample_factor=128, interactive=False)
        data_off, freq_off, freqid_off = data_dedisp_derip_filled_masked(event_id, dm, downsample_factor=128, interactive=False, off=True)
    
    else:
        data, freq, freqid = data_dedisp_derip_filled_masked(event_id, dm, downsample_factor=128, interactive=False, diagnostic_plot=direc)
        data_off, freq_off, freqid_off = data_dedisp_derip_filled_masked(event_id, dm, downsample_factor=128, interactive=False, off=True, diagnostic_plot=direc)
    
    # Determine burst frequency extent if not provided
    if speclims is None:
        print('*** Determining the burst extent in frequency ***')
        power = np.abs(data) ** 2  # Compute power spectrum
        try:
            spect_lim = get_spectrum_lim(freqid, power, diagnostic_plots=True)  # Determine spectral limits
            plt.show()
        except:
            print('Spec lim determination not working, defaulting to full band')
            spect_lim=[0,1024]

        print(spect_lim)
    else:
        spect_lim = speclims
    
    # Apply frequency limits to data
    data = data[spect_lim[0]:spect_lim[1]]
    freq = freq[spect_lim[0]:spect_lim[1]]
    freqid = freqid[spect_lim[0]:spect_lim[1]]
    data_off = data_off[spect_lim[0]:spect_lim[1]]
    freq_off = freq_off[spect_lim[0]:spect_lim[1]]
    freqid_off = freqid_off[spect_lim[0]:spect_lim[1]]
    
    print('*** Determining the burst width ***')
    power = np.abs(data) ** 2  # Compute power spectrum
    I = np.nansum(power,axis=1)
    prof = np.nanmean(I,axis=0)
    power_off = np.abs(data_off) **2
    I_off = np.nansum(power_off,axis=1)
    prof_off = np.nanmean(I_off,axis=0)

    prof-=np.nanmean(prof_off)
    prof/=np.nanstd(prof_off)

    for downsamp_fact in 2 ** np.arange(9):
        SNr = prof.copy()
        indx = SNr.size // downsamp_fact
        SNr = SNr[: SNr.size // downsamp_fact * downsamp_fact]
        SNr = np.nansum(
            SNr.reshape([int(SNr.size / downsamp_fact), downsamp_fact]), axis=-1
        ) / np.sqrt(downsamp_fact)
    
        if np.nanmax(SNr) > 20:
            power = power[..., : power.shape[-1] // downsamp_fact * downsamp_fact]
            power = np.nanmean(
                power.reshape(list(power.shape[:-1]) + [power.shape[-1] // downsamp_fact, downsamp_fact]),
                axis=-1,
            )
            break

    print(
        f"downsampling data by factor of {downsamp_fact} to {downsamp_fact * 2.56e-3} ms time resolution"
    )
    
    try:
        lims = get_burst_envelope(power, thres=6, pad=0, diagnostic_plots=False)  # Determine burst envelope
        lims = downsamp_fact * np.array(lims)
    except:
        print('Could not determine burst limits')
   


    # Determine optimal FFT size if not provided
    if len(fftsize)==0:
        possible_fftsizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        deltat = lims[1] - lims[0]
        fftsize = possible_fftsizes[np.argmin(np.abs((possible_fftsizes - deltat)))]
        downfreq = 1
        print('*** Determining fftsize using the width of the burst ***')

        # fftsize is the maxium that we search until
        # Generate log-spaced values and round to the nearest power of 2

        n = min(n_fft, int(np.log2(fftsize)) - 3 + 1)  # Ensure n does not exceed available values
        logspace_values = np.logspace(np.log2(fft_min), np.log2(fftsize), num=n, base=2)
        fftsizes_array = np.unique(np.round(logspace_values).astype(int))  # Remove duplicates
        fftsizes_array = np.unique(np.array([size if size in possible_fftsizes else possible_fftsizes[np.argmin(np.abs(possible_fftsizes - size))] for size in fftsizes_array]))

    else:
        fftsizes_array = fftsize


    # Lists to store results for each fftsize
    fftsize_list  = []
    on_spectrum_list = [] 
    off_spectrum_list = []
    peak_spectrum_list = []
    freq_list = []
    time_series_list = []
    
    # upchan_lims are saved temporarly but not written to the .npz file
    upchan_lims_list = []
    

    # Upchannelization process per fftsize
    print(f"*** We will upchannelise {event_id} to the following fftsizes: {fftsizes_array} ***")
    for fftsize in fftsizes_array:
        if fftsize!=1:
            print(f'*** Upchannelising to fftsize {fftsize}, downfreq {downfreq} ***')
            data_dedisp_masked_upchan_ds = upchan(data, freqid, fftsize=fftsize, downfreq=downfreq)
            noise_dedisp_masked_upchan_ds = upchan(data_off, freqid, fftsize=fftsize, downfreq=downfreq) # for scallop model
            model_ds, inds = make_scallop_model(noise_dedisp_masked_upchan_ds[0], fftsize, downfreq)
   
        
            # Compute power and determine main peak limits
            power = np.abs(data_dedisp_masked_upchan_ds[0]) ** 2
            I_upchan_ds = np.sum(power, axis=0).T

            power_off = np.abs(noise_dedisp_masked_upchan_ds[0]) ** 2
            I_upchan_ds_off = np.sum(power_off,axis=0).T

            prof_upchan = np.nansum(I_upchan_ds,axis=0)
            plt.close('all')
            plt.plot(prof_upchan)
            upchan_lims = get_main_peak_lim(I_upchan_ds, normalize_profile=True)
            plt.axvline(upchan_lims[0])
            plt.axvline(upchan_lims[1])
            plt.savefig(diagnostic_plots_direc+'/upchan_prof_fftsize%s.png'%fftsize, format='png')
            plt.show()

            # Ensure limits are within valid range
            upchan_lims = list(upchan_lims)
            upchan_lims[0] = max(upchan_lims[0], 0)
            upchan_lims[1] = min(upchan_lims[1], I_upchan_ds.shape[1] - 1)
            upchan_lims_list.append(upchan_lims)
        
            # Now is the spectrum normalisation, and scalloping correction
            # Define off-burst region for background subtraction
            offburst_begin = 0
            offburst_end = upchan_lims[0] - 1
            if offburst_end <= 0:
                offburst_end = 1
            print(f'Offburst_end: {offburst_end}')
        
            # Correct for scalllops
            I_upchan_corrected = np.zeros_like(I_upchan_ds)
            I_upchan_corrected_off = np.zeros_like(I_upchan_ds_off)
            for time_bin in range(I_upchan_ds.shape[1]):
                I_upchan_corrected[:, time_bin] = I_upchan_ds[:, time_bin] / model_ds
            for time_bin in range(I_upchan_ds_off.shape[1]):
                I_upchan_corrected_off[:,time_bin] = I_upchan_ds_off[:, time_bin] / model_ds
            
            # Bandpass correct  
            #first check if the off burst data is useful
            #if offburst_end <=1:
            #    Ioff_spec=I_upchan_corrected[:,0]
            #else:
            #    Ioff_spec = np.nanmean(I_upchan_corrected[:,offburst_begin:offburst_end])
           
            #    print("no bandpass correction applied since off burst data was not useful")
            print(I_upchan_corrected_off.shape[1])
            if I_upchan_corrected_off.shape[1] < 4:
                for time_bin in range(I_upchan_corrected.shape[1]):
                    I_upchan_corrected[:,time_bin] = I_upchan_corrected[:,time_bin] - np.nanmean(I_upchan_corrected_off[:,0])
                    if time_bin < I_upchan_corrected_off.shape[1]:
                        I_upchan_corrected_off[:, time_bin] = I_upchan_corrected_off[:, time_bin] - np.nanmean(I_upchan_corrected_off[:,0])
                    I_upchan_corrected[:,time_bin] = I_upchan_corrected[:,time_bin] / np.nanstd(I_upchan_corrected_off[:,0])
                    if time_bin < I_upchan_corrected_off.shape[1]:
                        I_upchan_corrected_off[:, time_bin] = I_upchan_corrected_off[:,time_bin] / np.nanstd(I_upchan_corrected_off[:,0])
            else:
                for freq_chan in range(I_upchan_corrected.shape[0]):
                    Ioff = I_upchan_corrected_off[freq_chan,:].data 
                    I_upchan_corrected[freq_chan,:] = I_upchan_corrected[freq_chan,:] - np.nanmean(Ioff)
                    I_upchan_corrected_off[freq_chan, :] = I_upchan_corrected_off[freq_chan, :] - np.nanmean(Ioff)

                    Ioff-=np.nanmean(Ioff)
                    if np.nanstd(Ioff) != 0:
                        I_upchan_corrected[freq_chan,:] = I_upchan_corrected[freq_chan,:] / np.nanstd(Ioff)
                        I_upchan_corrected_off[freq_chan,:] = I_upchan_corrected_off[freq_chan,:] / np.nanstd(Ioff)
            
        else:
            print(f'*** Computing spectrum and ACF without upchannelisation ***')
            power = np.abs(data)**2
            I_ds = np.nansum(power,axis=1)
            lims = get_main_peak_lim(I_ds, normalize_profile=True)
            #off burst
            power_off = np.abs(data_off)**2
            I_ds_off = np.nansum(power,axis=1)

            #ensure limits are within valid range
            lims = list(lims)
            lims[0] = max(lims[0],0)
            lims[1] = min(lims[1],I_ds.shape[1]-1)
            lims=[20000,20250]
            upchan_lims_list.append(lims)
            upchan_lims = lims
            
            plt.plot(np.nanmean(I_ds,axis=0))
            plt.axvline(upchan_lims[0])
            plt.axvline(upchan_lims[1])
            plt.xlim(upchan_lims[0]-3000, upchan_lims[1]+3000)
            plt.show()

            #spectrum normalisation
            #Define off-burst region 
            offburst_begin = 0
            offburst_end = lims[0]-1
            if offburst_end <=0:
                offburst_end=1
            print(f'Offburst_end: {offburst_end}')

            #bandpass correct
            I_upchan_corrected = np.zeros_like(I_ds)
            for freq_chan in range(I_upchan_corrected.shape[0]):
                try:
                    #if offburst_end <=1:
                    #    Ioff = I_ds[freq_chan,0].copy()
                    #else:
                    #    Ioff = I_ds[freq_chan,offburst_begin:offburst_end].copy()

                    Ioff = I_ds_off[freq_chan, :].copy()
                    I_upchan_corrected[freq_chan,:] = I_ds[freq_chan,:].copy()
                    I_upchan_corrected[freq_chan,:] -= np.nanmean(Ioff)
                    Ioff-=np.nanmean(Ioff)
                    I_upchan_corrected[freq_chan,:] = I_upchan_corrected[freq_chan,:] / np.nanstd(Ioff)
                except:
                    print('fully masked channel')
            I_upchan_corrected_off = I_upchan_corrected[:,offburst_begin:offburst_end]

        plt.plot(np.nanmean(I_upchan_corrected_off,axis=1))
        plt.show()

        spec_off = np.nanmean(I_upchan_corrected_off, axis=1).copy()
        spec_off[np.isnan(spec_off)]=0
        spec_off=np.ma.masked_where(spec_off==0,spec_off)

        print(f' np.nanmean(spec_off) { np.nanmean(spec_off)}')
        calib_off=spec_off-np.nanmean(spec_off)
        calib_off/=np.std(calib_off) 
        newinds=np.where(np.abs(calib_off)>3)[0]

        if fftsize!=1:
            I_upchan_corrected[inds,:]=0
             

        I_upchan_corrected[newinds,:]=0
        I_upchan_corrected_off[newinds,:]=0
        I_upchan_corrected = np.ma.masked_where(I_upchan_corrected==0,I_upchan_corrected)
        I_upchan_corrected_off = np.ma.masked_where(I_upchan_corrected_off==0, I_upchan_corrected_off)

        # on burst spectrum
        spec_upchan_corr=np.nanmean(I_upchan_corrected[:,upchan_lims[0]:upchan_lims[1]],axis=1)
        spec_upchan_corr[np.isnan(spec_upchan_corr)]=0
        spec_upchan_corr=np.ma.masked_where(spec_upchan_corr==0,spec_upchan_corr)

        # peak burst spectrum
        if fftsize!=1:
            prof = np.nanmean(I_upchan_ds,axis=0)
        else:
            prof = np.nanmean(I_ds,axis=0)

        peak=np.argmax(prof)
        spec_peak_upchan_corr=I_upchan_corrected[:,peak]
        spec_peak_upchan_corr[np.isnan(spec_peak_upchan_corr)]=0
        spec_peak_upchan_corr=np.ma.masked_where(spec_peak_upchan_corr==0,spec_peak_upchan_corr)

        # off burst spectrum
        #if upchan_lims[1]<=1:
        #    spec_fake_upchan_corr = I_upchan_corrected[:,0]
        #else:
        #    spec_fake_upchan_corr = np.nanmean(I_upchan_corrected[:,0:upchan_lims[0]],axis=1)
        spec_fake_upchan_corr = np.nanmean(I_upchan_corrected_off,axis=1)
        spec_fake_upchan_corr[np.isnan(spec_fake_upchan_corr)]=0
        spec_fake_upchan_corr=np.ma.masked_where(spec_fake_upchan_corr==0,spec_fake_upchan_corr)
        
            
            
        # Save spectrums, freqs, time series to lists
        fftsize_list.append(fftsize)
        on_spectrum_list.append(spec_upchan_corr)
        off_spectrum_list.append(spec_fake_upchan_corr)
        peak_spectrum_list.append(spec_peak_upchan_corr)
        if fftsize!=1:
            freq_list.append(data_dedisp_masked_upchan_ds[1])
            time_series_list.append(np.nanmean(I_upchan_ds,axis=0))
        else:
            freq_list.append(freq)
            time_series_list.append(np.nanmean(I_ds,axis=0))
        
        
    # write out to file    
    save_spectrum_data(diagnostic_plots_direc, event_id, fftsize_list, on_spectrum_list, off_spectrum_list, 
                       peak_spectrum_list, freq_list, time_series_list, overwrite_npz=False)  
    
    npz_file = os.path.join(diagnostic_plots_direc, f'{event_id}_spectrum_data.npz')
    print(f'*** Writing spectra to {npz_file}***')
    
    # compute the ACF's 
    # diagnostic plot
    outdir = diagnostic_plots_direc

    
    for idx, fftsize in enumerate(fftsize_list):
        signal = on_spectrum_list[idx]
        frequencies = freq_list[idx]
        acf, lags = compute_acf_fourier(signal, frequencies, max_lag_mhz = maxlag_mhz, normalize=True)
        
        on_spectrum = signal 
        off_spectrum = off_spectrum_list[idx]
        freqs = frequencies
        time_series = time_series_list[idx]
        time_series_limits = upchan_lims_list[idx]
        
        plot_acf_diagnostic_plot(fftsize, on_spectrum, off_spectrum, freqs, time_series, time_series_limits, acf, lags, event_id, outdir, downfreq)
    

    print('Done')
    return


def save_acf_results(outdir, event_id, fftsize, acf_res, acf_peak_res, acf_subs, 
                     acf_peak_subs, fcents_subs, lags_subs, submask, 
                     submask_peak, spec_lens, spec_lens_peak):
    """
    Saves autocorrelation function (ACF) results to a compressed .npz file.

    Parameters:
        outdir (str): Path to the output directory where the results will be saved.
        event_id (str): CHIME event ID
        fftsize (int): fftsize used for upchannelisation
        acf_res (array-like): acf_res[0] is the full band ACF, acf_res[1] is the corresponding lags
        acf_peak_res (array-like): acf_peak_res is the same as acf_res except for the peak burst spectrum
        acf_subs (array-like): ACFs per subband
        acf_peak_subs (array-like): ACFs per subband for peak burst spectrum
        fcents_subs (array-like): Central frequencies of the subbands in MHz.
        lags_subs (array-like): Lag values corresponding to the subband ACFs.
        submask (array-like): Boolean mask for subbands
        submask_peak (array-like): Boolean mask for subbands in peak burst spectrum
        spec_lens (array-like): Lengths of the subband spectra used in ACF computation
        spec_lens_peak (array-like): Lengths of the subband spectra used for peak burst ACF computation

    Saves:
        A compressed .npz file named `{event_id}_fftsize_{fftsize}.npz` in `outdir`.
        The file contains the following arrays:
        - `acf_res_0`:  The full spectrum ACF.
        - `acf_peak_res_0`: The full spectrum ACF of the burst peak
        - `acf_peak_res_1`: Lags for full ACF 
        - `acf_subs`: Subband ACFs
        - `acf_peak_subs`: Subband ACFs for burst peak.
        - `fcents_subs`: Frequency centers for subbands
        - `lags_subs`: Lag values for subband ACFs
        - `submask`: Fraction of channels masked per subband for the full burst spectrum
        - `submask_peak`: Fraction of channels masked per subband for the burst peak spectrum
        - `spec_lens`: Subband lengths
        - `spec_lens_peak`: Subband lengths for the burst peak spectrum

    Prints:
        A confirmation message indicating the file has been saved.
    """

    # Construct the filename
    filename = f"{outdir}/{event_id}_fftsize_{fftsize}.npz"
    
    
    # Save variables to .npz file
    np.savez_compressed(filename, 
                        acf_res_0=acf_res[0],
                        acf_peak_res_0=acf_peak_res[0],
                        acf_peak_res_1=acf_peak_res[1],
                        acf_subs=acf_subs,
                        acf_peak_subs=acf_peak_subs,
                        fcents_subs=fcents_subs,
                        lags_subs=lags_subs,
                        submask=submask,
                        submask_peak=submask_peak,
                        spec_lens=spec_lens,
                        spec_lens_peak=spec_lens_peak)   
    

    print(f"Saved results to {filename}")



def run_scint_pipe(event_id, fftsize, downfreq,diagnostic_plots_direc, subbands=8, peak_only=False, maxlag=20, maxlag_subs=10, snsubband=False):
    """
    Loads in spectra from an .npz file at '{diagnostic_plots_direc}/{event_id}_spectrum_data.npz'. 
    Computes the brute force ACF applying normalization, such that the height of the ACF around zero lag is the modulation index squared (see Nimmo et al. 2025 for a description).
    
    
    Parameters:
    - event_id (str): CHIME event ID
    - fftsize (int): FFT size for upchannelisation to use (this must correspond to one existing in the spectra npz file).
    - downfreq (int): Frequency downsampling factor for upchannelisation.
    - diagnostic_plots_direc (str): Directory path to save diagnostic plots. If None, no plots are saved.
    - subbands (int, optional): Number of subbands for ACF analysis. Default is 8.
    - peak_only (bool, optional): If True, computes only peak burst ACF. Default is False, which will compute both full burst and peak burst ACF. 
    - maxlag (int, optional): Maximum lag for full band ACF calculation. Default is 20 MHz.
    - maxlag_subs (int, optional): Maximum lag for subband ACF. Default is 10 MHz.
    - snsubband (bool, optional): If True, enables signal-to-noise based subbanding. Default is False, which will do equal frequency width subbands.
    
    Returns:
    - None: ACF data is written to a compressed .npz file with the filename format:
      '{event_id}_fftsize_{fftsize}.npz'.
      The saved data includes:
        - acf_full: Full-band ACF.
        - acf_peak: Full-band peak burst ACF.
        - freq_lags: Frequency lags corresponding to the full-band ACFs.
        - acf_subs: Subband ACFs.
        - acf_peak_subs: Peak burst subband ACFs.
        - fcents_subs: Central frequencies of subbands [MHz].
        - lags_subs: Lags for subband ACFs [MHz].
        - submask: Fraction of channels masked per subband (RFI flagging).
        - submask_peak: Fraction of channels masked per subband in the peak burst spectrum (RFI flagging).
        - speclens: Lengths of subbands.
        - speclens_peak: Lengths of subbands in the peak burst spectrum.
    """

    if diagnostic_plots_direc == None:
        diagnostic_plots_direc = ''

    print(maxlag_subs)
    print("*** Loading burst spectra ***")
    

    # Load in the spectra: on-burst, peak burst and off-burst 
    # Loop over and extract the values for our fftsize
    
    fftsize_list, on_spectrum_list, off_spectrum_list, peak_spectrum_list, freq_list, time_series_list = load_spectrum_data(diagnostic_plots_direc, event_id)
    
    
    for idx, my_fftsize in enumerate(fftsize_list):
        if my_fftsize == fftsize:
            spec_upchan_corr = np.array(on_spectrum_list[idx], dtype=np.float64)
            spec_peak_upchan_corr = np.array(peak_spectrum_list[idx], dtype=np.float64)
            spec_fake_upchan_corr = np.array(off_spectrum_list[idx], dtype=np.float64)
            freqs = np.array(freq_list[idx], dtype=np.float64)
            #apply masks
            spec_upchan_corr = np.ma.masked_where(spec_upchan_corr==0, spec_upchan_corr)
            spec_peak_upchan_corr = np.ma.masked_where(spec_peak_upchan_corr==0, spec_peak_upchan_corr)
            spec_fake_upchan_corr = np.ma.masked_where(spec_fake_upchan_corr==0, spec_fake_upchan_corr)
    
    
    # Generate the freq_ids
    freq_ids = np.arange(len(freqs))
    
    print(f'type( spec_fake_upchan_corr) {type(spec_fake_upchan_corr)}')
    fake_mean = np.nanmean(spec_fake_upchan_corr)

    # Generate diagnostic plots if requested
    if diagnostic_plots_direc != '':
        plt.close('all')
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(freqs, spec_upchan_corr, color='k', alpha=0.5, label='on')
        ax[0].plot(freqs, spec_fake_upchan_corr, color='k', label='off')
        ax[1].plot(freqs, spec_peak_upchan_corr, color='k', alpha=0.5, label='peak')
        ax[1].plot(freqs, spec_fake_upchan_corr, color='k', label='off')
        ax[0].legend()
        ax[1].legend()
        ax[1].set_xlabel('Freq [MHz]')
        ax[0].set_ylabel('Intensity')
        ax[1].set_ylabel('Intensity')
        plt.savefig(diagnostic_plots_direc + '/%s_upchan_spec_fftsize%s_downfreq%s.png'%(event_id,fftsize,downfreq), format='png')
    
    # Compute ACFs
    acf_res = acf_scint_plot(spec_upchan_corr, freq_ids, freqs, [0, 0], maxlag=maxlag, offspec_mean=fake_mean) if not peak_only else [0]
    acf_peak_res = acf_scint_plot(spec_peak_upchan_corr, freq_ids, freqs, [0, 0], maxlag=maxlag, offspec_mean=fake_mean)
    
    # Compute subband ACFs if required
    if not peak_only:
        acf_subs, fcents_subs, lags_subs, submask, spec_lens = multi_sub_acf(
            spec_upchan_corr, freqs, freq_ids, spec_fake_upchan_corr, fftsize, downfreq, diagnostic_plot=diagnostic_plots_direc, numsubs=subbands, maxlag=maxlag_subs, snsubband=snsubband)
    else:
        acf_subs, submask, spec_lens = [], [], []
    
    acf_peak_subs, fcents_subs, lags_subs, submask_peak, spec_lens_peak = multi_sub_acf(
        spec_peak_upchan_corr, freqs, freq_ids, spec_fake_upchan_corr, fftsize, downfreq, diagnostic_plot=diagnostic_plots_direc, numsubs=subbands, filename_add='peak', maxlag=maxlag_subs, snsubband=snsubband)

    outdir = diagnostic_plots_direc
    save_acf_results(outdir, event_id, fftsize, acf_res, acf_peak_res, acf_subs, 
                     acf_peak_subs, fcents_subs, lags_subs, submask, 
                     submask_peak, spec_lens, spec_lens_peak)
    
    
    return 



def multi_sub_acf(spec_upchan_corr,freqs,freq_ids,spec_fake_upchan_corr,fftsize,downfreq,diagnostic_plot=None, numsubs=8, filename_add=None, maxlag=10,snsubband=False):
    """
    Computes the autocorrelation function (ACF) for {numsubs} subbands in the input spectrum {spec_upchan_corr}

    This function divides the input spectrum into subbands, calculates the ACF for each subband, 
    and optionally generates diagnostic plots to visualize the ACF across frequency lags.

    Parameters:
    ----------
    spec_upchan_corr : array-like
        The upchannelized and scallop corrected spectrum.
    freqs : array-like
        The frequency values corresponding to the spectrum in MHz.
    freq_ids : array-like
        The indices of the frequencies in the spectrum.
    spec_fake_upchan_corr : array-like
        Off burst spectrum used for noise estimation, and normalization.
    fftsize : int
        The FFT size used in upchannelising the spectrum.
    downfreq : int
         The Frequency downsampling factor used in upchannelising the spectrum.
    diagnostic_plot : str, optional
        Path to save diagnostic plots. If None, no plots are saved.
    numsubs : int, optional (default=8)
        Number of subbands to divide the spectrum into for ACF computation.
    filename_add : str, optional
        Additional filename identifier for saved diagnostic plots.
    maxlag : int, optional (default=10)
        Maximum lag value for computing the ACF in MHz.
    snsubband : bool, optional (default=False)
        Whether to use the signal-to-noise ratio per subband (True) or to use equal frequency extent subbands (False)

    Returns:
    -------
    acfs : list of arrays
        A list containing the ACF values for each subband.
    fcents : list of floats
        The central frequency of each subband in MHz.
    lags : list of arrays
        The frequency lag values corresponding to the ACFs.
    submask : array-like
        Fraction of channels masked per subband.
    spec_lens : list
        The lengths of spectra in each subband.

    Notes:
    -----
    - If `diagnostic_plot` is provided, the function generates and saves plots of the ACF 
      for different frequency lags (0.3 MHz and 1.5 MHz) for visualization.
    - The ACFs are computed using the `acf_per_subband` function.
    """
   
    
    acfs,fcents,lags, subsn, submask,spec_lens=acf_per_subband(spec_upchan_corr,freqs,freq_ids,num_subbands=numsubs,plot_fit=False,savefig=None,maxlag=maxlag,offspec=spec_fake_upchan_corr,snsubband=snsubband)
    cmap = matplotlib.cm.get_cmap('plasma')
    if diagnostic_plot:
        plt.close('all')
        for i in range(len(fcents)):
            rgba = cmap(i/len(fcents))
            plt.plot(lags[len(fcents)-i-1],acfs[len(fcents)-i-1]+(1.0*i),drawstyle='steps-mid',color=rgba,linewidth=2,alpha=0.7,label='%.2f MHz'%fcents[len(fcents)-i-1])
            plt.xlim(-0.3,0.3)
            plt.xlabel('Freq Lag [MHz]')
            plt.ylabel('ACF power [offset + 1]')

        if filename_add:
            plt.savefig(diagnostic_plot+'/ACF_%s_per_subband_0.3MHz_fftsize%s_downfreq%s.png'%(filename_add,fftsize,downfreq), format='png')
        else:
            plt.savefig(diagnostic_plot+'/ACF_per_subband_0.3MHz_fftsize%s_downfreq%s.png'%(fftsize,downfreq), format='png')

        plt.close('all')
        for i in range(len(fcents)):
            rgba = cmap(i/len(fcents))
            plt.plot(lags[len(fcents)-i-1],acfs[len(fcents)-i-1]+(1.0*i),drawstyle='steps-mid',color=rgba,linewidth=2,alpha=0.7,label='%.2f MHz'%fcents[len(fcents)-i-1])
            plt.xlim(-1*maxlag,maxlag)
            plt.xlabel('Freq Lag [MHz]')
            plt.ylabel('ACF power [offset + 1]')

        if filename_add:
            plt.savefig(diagnostic_plot+'/ACF_%s_per_subband_1.5MHz_fftsize%s_downfreq%s.png'%(filename_add,fftsize,downfreq), format='png')
        else:
            plt.savefig(diagnostic_plot+'/ACF_per_subband_1.5MHz_fftsize%s_downfreq%s.png'%(fftsize,downfreq), format='png')

    return acfs, fcents, lags, submask, spec_lens


def ne2001_scat(event_id):
    """
    Estimates the scattering timescale and scintillation bandwidth for a given CHIME/FRB event
    using the NE2001 electron density model.

    Parameters:
    -----------
    event_id : int or str
        CHIME/FRB event ID

    Returns:
    --------
    ne2001_scatt_chime : astropy.Quantity
        The estimated scattering timescale at CHIME frequencies (600 MHz), in milliseconds.
    ne2001_scint_chime : astropy.Quantity
        The estimated scintillation bandwidth at CHIME frequencies (600 MHz), in kHz.

    Notes:
    ------
    - The function retrieves the FRB's sky position (RA, Dec) from the CHIME/FRB API.
    - The NE2001 model is used to estimate the scattering timescale at 1 GHz.
    - The timescale is scaled to CHIME's observing frequency (~600 MHz) assuming a frequency scaling of ν⁻⁴.
    - The scintillation bandwidth is computed as (2π * τ_scatt)⁻¹.
    """
    
    master = chime_frb_api.frb_master.FRBMaster(base_url = "https://frb.chimenet.ca/frb-master")
    master.API.authorize()
    auth = {"Authorization": master.API.access_token}
    event = master.events.get_event(event_id)

    for par in event['measured_parameters']:
        if par['pipeline']['name']=='baseband':
            ra=par['ra']
            dec=par['dec']
    
    pos = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    Dk,Dv,Du,Dd = ne2001(ldeg=pos.galactic.l.value,bdeg=pos.galactic.b.value,dmd=20,ndir=-1,classic=False,dmd_only=False)
    ne2001_scatt = Dv['TAU']*u.ms #at 1GHz


    #scale to CHIME
    ne2001_scatt_chime = ne2001_scatt * (1/0.6)**4

    #convert to scint bw
    ne2001_scint_chime = 1/(2*np.pi*ne2001_scatt_chime)

    return ne2001_scatt_chime, ne2001_scint_chime.to('kHz')


def load_acf_results(outdir, event_id, fftsize):
    """
    Loads ACF analysis results from a compressed .npz file. Expected format is '{event_id}_fftsize_{fftsize}.npz'.
    
    Parameters:
        outdir (str): Output directory path.
        event_id (str): CHIME/FRB event ID
        fftsize (int): FFT size used in upchannelisation.
        
    Returns:
        - acf_full: Full-band ACF.
        - acf_peak: Full-band peak burst ACF.
        - freq_lags: Frequency lags corresponding to the full-band ACFs.
        - acf_subs: Subband ACFs.
        - acf_peak_subs: Peak burst subband ACFs.
        - fcents_subs: Central frequencies of subbands [MHz].
        - lags_subs: Lags for subband ACFs [MHz].
        - submask: Fraction of channels masked per subband (RFI flagging).
        - submask_peak: Fraction of channels masked per subband in the peak burst spectrum (RFI flagging).
        - speclens: Lengths of subbands.
        - speclens_peak: Lengths of subbands in the peak burst spectrum.
   
    """
    # Construct the filename
    filename = f"{outdir}/{event_id}_fftsize_{fftsize}.npz"
    
    print(f' Loading data from {filename}')
    
    # Load the data from the .npz file
    data = np.load(filename, allow_pickle=True)
    
    # Rename and return the variables
    acf_full = data['acf_res_0']
    acf_peak = data['acf_peak_res_0']
    freq_lags = data['acf_peak_res_1']
    acf_subs = data['acf_subs']
    acf_peak_subs = data['acf_peak_subs']
    fcents_subs = data['fcents_subs']
    lags_subs = data['lags_subs']
    submask = data['submask']
    submask_peak = data['submask_peak']
    spec_lens = data['spec_lens']
    speclens_peak = data['spec_lens_peak']
    

    return acf_full, acf_peak, freq_lags, acf_subs, acf_peak_subs, fcents_subs, lags_subs, submask, submask_peak, spec_lens, speclens_peak


def calculate_acf_errors(acf):
    """
    Compute the error estimates for an autocorrelation function (ACF) using the standard method 
    based on cumulative variance.
    
    The errors are computed using the variance formula:
        var_f[1:] = (1 + 2 * cumulative sum of squared ACF values)
    and the final errors are obtained as the square root of this variance. The function 
    returns a symmetric array of errors by mirroring the computed values.

    Parameters
    ----------
    acf : array-like
        The autocorrelation function (ACF) values.

    Returns
    -------
    f_errors_full : numpy.ndarray
        The symmetric error estimates for the ACF, computed based on the variance of 
        the second half of the ACF.
    
    """
    # Take the second half of the ACF
    acf_half = acf[len(acf) // 2:]
    
    # Initialize the variance for each element
    var_f = np.ones(len(acf_half)) / len(acf_half)
    
    # Compute the cumulative sum of squared ACF values and update var_f
    var_f[1:] *= 1 + 2 * np.cumsum(acf_half[:-1] ** 2)
    
    # Compute the errors as the square root of var_f
    f_errors = np.sqrt(var_f)
    
    # Concatenate the reversed errors with the original errors
    f_errors_full = np.concatenate((f_errors[::-1], f_errors))
    
    return f_errors_full


def bic_comparison(results_dict, fcents_subs, event_id, outdir, fftsize, save=False):
    """
    Plots the Bayesian Information Criterion (BIC) for different lmfit models across subbands.
    Determines the best model per subband and identifies the overall best model.
    
    Parameters:
        results_dict (dict): A dictionary containing 'lm_fitting_objects', a list of subband fit results.
        fcents_subs (list): A list of central frequencies corresponding to each subband (MHz).
        event_id (str): CHIME/FRB event ID
        fftsize (int): The upchannelization fft factor used for this data set
        outdir (str): Path to outdir to store files
        save (bool, optional): If True (default), saves the plot.
        
    returns: int, the best model IE 2 if the best model shows 2 lorentzians
    """
    lmfit_objects = results_dict['lm_fitting_objects']
    num_subbands = len(lmfit_objects)
    cmap = matplotlib.cm.get_cmap('plasma')
    
    model_labels = ["1 Lorentzian", "2 Lorentzians"]
    model_indices = [1, 2]
    best_model_counts = {1: 0, 2: 0}
    
    plt.figure(figsize=(6, 5))
    
    for i, (subband, fcent) in enumerate(zip(lmfit_objects, fcents_subs)):
        rgba = cmap(i / num_subbands)
        if 1 in subband and 2 in subband:  # Ensure both models exist
            bic1 = subband[1].bic
            bic2 = subband[2].bic
            best_model = 1 if bic1 < bic2 else 2
            best_model_counts[best_model] += 1
            plt.scatter(1, bic1, color=rgba)
            plt.scatter(2, bic2, color=rgba)
            plt.text(best_model, min(bic1, bic2), f'✓', ha='center', fontsize=12, fontweight='bold', color='black')
    
    overall_best_model = max(best_model_counts, key=best_model_counts.get)
    plt.axvline(x=overall_best_model, linestyle="--", color="k", alpha=0.7, label=f"Overall Best Model: {model_labels[overall_best_model - 1]}")
    
    plt.xticks(model_indices, model_labels)
    plt.xlabel("Model")
    plt.ylabel("BIC Value")
    plt.title(f"{event_id} fftsize {fftsize}\nBIC Comparison of Models Across Subbands")
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(fcents_subs)
    cbar = plt.colorbar(sm, label="Subband Central Frequency (MHz)")
    plt.legend()
    
    
    if save == True:
        filename = outdir +f'/{event_id}_fftsize_{fftsize}_aic_bic_compare.png'
        plt.savefig(filename)
    
    plt.show()
    
    return overall_best_model


def fit_subband_acfs(event_id, outdir, fftsize, downfreq, lagrange_for_fits=None, offset=1, save_plots=True, xlim=0.5, print_outputs=True, ignore_subband_idx=None, peak=False):
    """
    Fits subband auto-correlation functions (ACFs) to Lorentzian models and computes scintillation bandwidths. It then performs a powerlaw fit to the scintilation
    bandwidths vs center freq of the subband. 
    
    It makes several diagnostic plots showing the ACFs, and their 1, and 2 lorentzian fit. As well as a BIC comparison plot, powerlaw plot, modulation vs subband plot.

    
    Final fitting results are saved to a .json with the output directory with the filename format 
    '<event_id>_scint_pipeline_results.json'
    
    
    Parameters:
    -----------
    event_id : str
        CHIME/FRB event ID
    outdir : str
        Directory path where ACF data is saved, and where to save files.
    fftsize : int
        FFT size used in the upchannelisation (must match the ACF computed and saved in outdir).
    downfreq : int
        Frequency downsampling factor used in upchannelisation
    lagrange_for_fits : list, optional
        List of lag ranges (in MHz) for fitting. Default is [0.3] * numsubs if not provided.
    offset : float, optional
        Vertical offset for plotting multiple subbands. Default is 1.
    save_plots : bool, optional
        Whether to save the generated plots. Default is True.
    xlim : float, optional
        The limit for the x-axis on the plots. Default is 0.5 MHz.
    print_outputs : bool, optional
        Whether to print diagnostic, model fitting outputs. Default is True.
    ignore_subband_idx : list, optional
        Indices of subbands to ignore during power law fitting. Default is None.
    peak : bool, optional
        Whether to use peak data instead of subband data for the ACF. Default is False.

    Returns:
    --------
    results_dict, best_model, power_law_fits_results_list, y600_list, y600_list_err
    
    results_dict: Dict, The fitted Lorentzian components, central frequencies of subbands (fcents), and the acf_offsets
    
    best_model: Which Lorentzian model is prefered 
    
    power_law_fits_results_list: results of the power law fit to the central frequencies vs scintillation bandwitdhs from the lorentzians
    
    y600_list: list of the scintillation as measured by the powerlaw scaled to 600 MHz 
    
    y600_list_err: list of the errors for the scintillation as measured by the powerlaw  
    """

    
    # Load in ACF data 
    acf_full, acf_peak, freq_lags, acf_subs, acf_peak_subs, fcents_subs, lags_subs, submask, submask_peak, spec_lens, speclens_peak = load_acf_results(outdir, event_id, fftsize)
    numsubs = len(acf_subs)
    fcents = fcents_subs
    

    # Create a figure for the plots
    fig = plt.figure(figsize=(5,7))
    
    # Set up the colormap for plotting
    cmap = matplotlib.cm.get_cmap('plasma')
    
    # Initialize lagrange_for_fits to default value of 1 MHz if not provided
    if lagrange_for_fits is None:
        lagrange_for_fits = [0.3] * numsubs

    # Frequency resolution based on FFT size and downsampling
    f_res = 0.39101 / (fftsize // downfreq)

   # initalize results dict for fitting 
    results_dict = {'1_lorenz' : {}, '2_lorenz' : {}, "f_cents": [], "acfs_offset": []

}
    results_dict['lm_fitting_objects'] = []
    # initalize outputs for the models
    for key in list(results_dict.keys()):
        if '_lorenz' in key:
            results_dict[key] = {"sub_scint_1": [], "sub_scint_uncert_1": [], "mods1": [], "mods1_uncert": [], "add_un1": [], 'c1': [], 'c1_uncert': []}
            
            

            if int(key[0]) == 2:
                results_dict[key] = {"sub_scint_1": [], "sub_scint_uncert_1": [], "mods1": [], "add_un1": [],  "add_un2": [],
                                      "mods1_uncert": [], "sub_scint_2": [], "sub_scint_uncert_2": [], 
                                      "mods2": [], "mods2_uncert": []}

    # loop over and fit for lorentzians for each subband ACF
    print(f'len(fcents) {len(fcents)}')
    print('\n\n\n\n\n')
    print(f'Order of fcents {fcents}')
    for i in range(len(fcents)):
        print(f'i : {i} fcent: {fcents[i]} total_offset {float(offset)*i}')
        
        
        lagrange_for_fit = lagrange_for_fits[i]
        rgba = cmap(i / len(fcents))
        if peak is False:
            acf = acf_subs[i]
            lag = lags_subs[i]
        else:
            acf = acf_peak_subs[i]
            lag = lags_subs[i]
        
        results_dict["acfs_offset"].append(acf + (float(offset) * i))
        results_dict["f_cents"].append(fcents[i]) # same order 

        acf_fit = acf[int(len(acf) / 2.) - int(lagrange_for_fit / f_res):int(len(acf) / 2.) + int(lagrange_for_fit / f_res)]
        lag_fit = lag[int(len(acf) / 2.) - int(lagrange_for_fit / f_res):int(len(acf) / 2.) + int(lagrange_for_fit / f_res)]
        acf_fit_errors = calculate_acf_errors(acf_fit) 

        plt.plot(lag, acf + (float(offset) * i), drawstyle='steps-mid', color=rgba, linewidth=2, alpha=0.7, label=f'{fcents[i]:.2f} MHz')
        
        
        # this part here should be changed for n lorentzian fittings 
        fit_results = {} # stores the lmfit outputs
        for numlorentz, model_func, param_names in zip(
            [1, 2], [lorentz_withc_min, doublelorentz_withc_min],
            [["gamma1", "m1"], ["gamma1", "m1", "gamma2", "m2"]]
        ):
            params = Parameters()
            params.add('gamma1', value=lagrange_for_fit, min=0.00001, max=100)
            params.add('m1', value=1, min=0, max=100)
            if numlorentz == 2:
                params.add('gamma2', value=lagrange_for_fit/10, min=0.00001, max=100)
                params.add('m2', value=1, min=0, max=100)
            params.add('c', value=0, min=-100, max=100)

            try:
                fit_min = Minimizer(model_func, params, fcn_args=(lag_fit, acf_fit, np.sqrt(acf_fit_errors)))
                result = fit_min.minimize()
                fit_results[numlorentz] = result # this is an lmfit object
            except Exception as e:
                print(f'Fitting failed: {e}')
                fit_results[numlorentz] = None
                
        # add in the lm_fit_obect to the dict
        results_dict['lm_fitting_objects'].append(fit_results)
        
        
        
        # Store results for 1 Lorentzian
        result_1 = fit_results[1]
        if result_1:
            label = None
            if i == 0:
                label = 'Single'
            plt.plot(lag, lorentz_w_c(lag, result_1.params['gamma1'], result_1.params['m1'], result_1.params['c']) + (float(offset) * i), color='red', linestyle='-.', linewidth=1, label=label)
            results_dict['1_lorenz']["sub_scint_1"].append(np.abs(result_1.params['gamma1'].value))
            results_dict['1_lorenz']["sub_scint_uncert_1"].append(abs(result_1.params['gamma1'].stderr))
            results_dict['1_lorenz']["mods1"].append(np.abs(result_1.params['m1'].value))
            results_dict['1_lorenz']["mods1_uncert"].append(abs(result_1.params['m1'].stderr))
            results_dict['1_lorenz']['c1'].append(result_1.params['c'].value)
            results_dict['1_lorenz']['c1_uncert'].append(result_1.params['c'].stderr)
            
            
            scints = np.array([np.abs(result_1.params['gamma1'].value)])
            modinds = np.array([np.abs(result_1.params['m1'].value)])
            errs = np.array([abs(result_1.params['gamma1'].stderr)])
            moderrs = np.array([abs(result_1.params['m1'].stderr)])
            inds = np.argsort(scints)
            
            
            # Compute additional uncertainty from low number of scintles
            sub_scint1 = scints[inds][0]
            if peak==False:
                good_chans = np.array(spec_lens) - np.array(submask)
            if peak==True:
                good_chans = np.array(speclens_peak) - np.array(submask_peak)

            N = 1 + 0.2 * ((np.flip(good_chans) * f_res) / sub_scint1)
            add_un = sub_scint1 / (2 * np.sqrt(N))
            
            results_dict['1_lorenz']["add_un1"].append(add_un)
            
            
        else:
            results_dict['1_lorenz']["sub_scint_1"].append(0)
            results_dict['1_lorenz']["sub_scint_uncert_1"].append(0)
            results_dict['1_lorenz']["mods1"].append(0)
            results_dict['1_lorenz']["mods1_uncert"].append(0)
            results_dict['1_lorenz']["add_un1"].append(0)
            results_dict['1_lorenz']['c1'].append(0)
            results_dict['1_lorenz']['c1_uncert'].append(0)


        # Store results for 2 Lorentzians
        result_2 = fit_results[2]
        if result_2:
            label = None
            if i == 0:
                label = 'Double'
            plt.plot(lag, doublelorentz_w_c(lag, result_2.params['gamma1'], result_2.params['m1'], result_2.params['gamma2'], result_2.params['m2'], result_2.params['c']) + (float(offset) * i), color='k', linestyle='--', linewidth=1, label=label)
            scints = np.array([np.abs(result_2.params['gamma1'].value), np.abs(result_2.params['gamma2'].value)])
            modinds = np.array([np.abs(result_2.params['m1'].value), np.abs(result_2.params['m2'].value)])
            
            gamma1_stderr = result_2.params['gamma1'].stderr
            gamma2_stderr = result_2.params['gamma2'].stderr

            # Check if stderr is None and replace it with np.nan
            errs = np.array([abs(gamma1_stderr) if gamma1_stderr is not None else np.nan,
                             abs(gamma2_stderr) if gamma2_stderr is not None else np.nan])
           # errs = np.array([abs(result_2.params['gamma1'].stderr), abs(result_2.params['gamma2'].stderr)])
            moderrs = np.array([abs(result_2.params['m1'].stderr), abs(result_2.params['m2'].stderr)])

            inds = np.argsort(scints)
            results_dict['2_lorenz']["sub_scint_1"].append(scints[inds][0])
            results_dict['2_lorenz']["sub_scint_2"].append(scints[inds][1])
            
            results_dict['2_lorenz']["sub_scint_uncert_1"].append(errs[inds][0])
            results_dict['2_lorenz']["sub_scint_uncert_2"].append(errs[inds][1])
            
            
            results_dict['2_lorenz']["mods1"].append(modinds[inds][0])
            results_dict['2_lorenz']["mods1_uncert"].append(moderrs[inds][0])
            results_dict['2_lorenz']["mods2"].append(modinds[inds][1])
            results_dict['2_lorenz']["mods2_uncert"].append(moderrs[inds][1])
            
            # Compute additional uncertainty from low number of scintles
            sub_scint2 = scints[inds][1]
            sub_scint1 = scints[inds][0]
            if peak==False:
                good_chans = np.array(spec_lens) - np.array(submask)
            if peak==True:
                good_chans = np.array(speclens_peak) - np.array(submask_peak)
                
            N2 = 1 + 0.2 * ((np.flip(good_chans) * f_res) / sub_scint2)
            add_un2 = sub_scint2 / (2 * np.sqrt(N2))

        
            N = 1 + 0.2 * ((np.flip(good_chans) * f_res) / sub_scint1)
            add_un = sub_scint1 / (2 * np.sqrt(N))
            
            results_dict['2_lorenz']["add_un1"].append(add_un)
            results_dict['2_lorenz']["add_un2"].append(add_un2)

        else:
            results_dict['2_lorenz']["sub_scint_1"].append(0)
            results_dict['2_lorenz']["sub_scint_2"].append(0)
            results_dict['2_lorenz']["sub_scint_uncert_1"].append(0)
            results_dict['2_lorenz']["sub_scint_uncert_2"].append(0)
            results_dict['2_lorenz']["mods1"].append(0)
            results_dict['2_lorenz']["mods1_uncert"].append(0)
            results_dict['2_lorenz']["mods2"].append(0)
            results_dict['2_lorenz']["mods2_uncert"].append(0)
            results_dict['2_lorenz']["add_un1"].append(0)
            results_dict['2_lorenz']["add_un2"].append(0)
            

    
        
    # Plot the subband frequencies
    acfs_offset = results_dict["acfs_offset"]
    plt.yticks([x[0] for x in acfs_offset], ['%.1f' % x for x in (fcents)])
    plt.xlim(-1 * xlim, xlim)
    plt.xlabel('Freq lag [MHz]')
    plt.ylabel('Subband freq [MHz]')
    fig.tight_layout(pad=1)

    # Save the diagnostic plot if requested
    if save_plots == True:
        plt.savefig(outdir + '/subband_fits_fftsize%s_downfreq%s.png' % (fftsize, downfreq), format='png')
    else:
        plt.show()


    # Now we plot the AIC, BIC 
    save = save_plots
    best_model = bic_comparison(results_dict, fcents_subs, event_id, outdir, fftsize, save=save)


    #print the expectation for Galactic scattering and scintillation from NE2001
    try:
        mw_scat,mw_scint=ne2001_scat(event_id)
        print('*** Expected MW scatt @ 600MHz: %s ***'%mw_scat)
        print('*** Expected MW scint @ 600MHz: %s ***'%mw_scint)
    except:
        print("could not compute NE2001 comparison")
        mw_scat = None
        mw_scint = None
    # Now we fit for powerlaw fitting 
    power_law_fits_results_list, y600_list, y600_list_err = perform_powerlaw_fit(event_id, fftsize, downfreq, outdir, results_dict, best_model, save_plot=save, ignore_subband_idx=ignore_subband_idx)

    #plot the modulation indices
    mi, mi_err = plot_mod_inds(event_id, fftsize, downfreq, outdir, results_dict, best_model, save_plot=save)

    #save the final measurements to a json file
    save_scint_results(event_id, power_law_fits_results_list, y600_list, y600_list_err, mi, mi_err, mw_scat, mw_scint, outdir)
        
    return results_dict, best_model, power_law_fits_results_list, y600_list, y600_list_err
    

def plot_power_law_fit(event_id, fftsize, downfreq, outdir, model_name, f_cents, lmfit_params, sub_scint, sub_scint_uncert, add_un, save_plot=True):
    """
    Plots the power-law fit to scintillation bandwidth vs central frequency and overlays the fitted model with error bars.

    Parameters:
    event_id (str): CHIME/FRB event ID
    fftsize (int): The FFT size used in the upchannelisation (must match the ACF you wrote to the npz file).
    downfreq (float): The downfreq value used in the upchannelisation (must match the ACF you wrote to the npz file).
    outdir (str): The directory where the plot will be saved.
    model_name (str): Name of the model used for fitting (e.g., 'Lorentzian 1').
    f_cents (array-like): Array of central frequencies (in MHz).
    lmfit_params (lmfit.Parameters): Parameters obtained from the lmfit model fitting of the power law (includes 'c' and 'n').
    sub_scint (array-like): Measured scintillation values per subband (in kHz).
    sub_scint_uncert (array-like): Uncertainties in the measured scintillation values per subband (in kHz).
    add_un (array-like): Additional uncertainties to be added in quadrature (in kHz).
    save_plot (bool, optional): Whether to save the plot. Defaults to True.

    Returns:
    tuple: A tuple containing:
        - y_600 (float): The fitted value of the scintillation bandwidth at 600 MHz (in kHz).
        - sigma_y_600 (float): The uncertainty in the scintillation bandwidth at 600 MHz (in kHz).

    """
    # Compute total error
    total_uncert = np.sqrt(sub_scint_uncert**2 + add_un**2)
    
    # Extract parameters and their uncertainties
    c = lmfit_params.params['c'].value
    n = lmfit_params.params['n'].value
    sigma_c = lmfit_params.params['c'].stderr
    sigma_n = lmfit_params.params['n'].stderr
    
    
    print(f'fcent order is {f_cents}')

    
    # Compute fitted model
    x_axis_model = np.linspace(np.min(f_cents), np.max(f_cents), 100)
    fitted_y = c * ( x_axis_model ** n)
    
    # Plot data with error bars
    plt.errorbar(f_cents, np.array(sub_scint)*1000, yerr=np.array(total_uncert)*1000, fmt='o', label='Data', capsize=5)
    
    # Compute and report value at x = 600
    y_600 = c * (600 ** n) *1000
    

    # use residuals of our scintillation bw measurements on the power law fit to determine the uncertainties
    fit_vals = c * (np.array(f_cents) ** n)
    resids = np.array(sub_scint) - fit_vals
    sigma_y_600 = np.nanstd(resids) * 1000

    # Overlay fitted model
    plt.plot(x_axis_model, fitted_y*1000, 'r--', 
             label=f'Fit: $y = c \cdot x^n$\nAt 600 MHz: $y = {y_600:.3f} \pm {sigma_y_600:.3f}$ kHz\n$n = {n:.2f} \pm {sigma_n:.2f}$')

                         
    # Labels and legend
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Scintillation Bandwidth (kHz)')
    plt.legend()
    
    title = f'CHIME {event_id} {model_name} {fftsize} Lorentzian Powerlaw'
    plt.title(title)
    if save_plot == True:
        plt.savefig(f'{outdir}/{event_id}_{model_name}_Lorentzian_Powerlaw_fftsize{fftsize}_downfreq{downfreq}.png')
    plt.show()
    
    return y_600, sigma_y_600


def perform_powerlaw_fit(event_id, fftsize, downfreq, outdir, result_dict, best_model,  save_plot=True, print_report=True, ignore_subband_idx= None):
    """
    Perform power-law fitting to scintillation data, including optional ignoring of subbands and plotting the results. 
    
    This function will loop over the number of Lorentzians indicated by 'best_model', and peform a powerlaw fiting for each 
    Lorentzian. 
    

    Args:
        event_id (str): The CHIME/FRB event ID
        fftsize (int): The FFT size used in the upchannelisation.
        downfreq (float): The downfreq value used in the upchannelisation.
        outdir (str): The output directory where the plots and results will be saved.
        result_dict (dict): A dictionary containing the Lorentzian model parameters and frequencies for the fitting process.
        best_model (int): The best model to use for fitting (indicates how many lorentzians to consider: 1 or 2 is supported).
        save_plot (bool, optional): Whether to save the plot of the power-law fit. Default is True.
        print_report (bool, optional): Whether to print the fitting report. Default is True.
        ignore_subband_idx (list, optional): A list of indices to exclude from the fitting process. Default is None.

    Returns:
        tuple: A tuple containing three elements:
            - power_law_fits_lm_results_list (list): A list of the lmfit parameter results of the power-law fitting for each lorentzian result.
            - y600_list (list): A list of the scintillation measurements at 600 MHz for each lorentzian.
            - y600_list_err (list): A list of the errors on the scintillation measurements at 600 MHz for each lorentzian.

    Example:
        >>> perform_powerlaw_fit(event_id='event123', fftsize=2048, downfreq=1, outdir='./output/', result_dict=my_result_dict, best_model=1)

    """
    
    
    
    f_cents = result_dict['f_cents']
    

    
    for key in (result_dict.keys()):
        if str(best_model) in key:
            best_lorentzian_params  = result_dict[key]
            
    params = Parameters()
    params.add('n', value=2, min=0, max=10)
    params.add('c', value=0.1)
    
    # we want to loop over and fit for n powerlaws 
    # our model is i+1 not i 
    
    power_law_fits_lm_results_list = [] # init 
    y600_list = [] # scintillation at 600 Mhz
    y600_list_err = [] #error on the scintillation measurement

    for i in range(int(best_model)): 
        f_cents = result_dict['f_cents']
        i+=1
        
        sub_scint = best_lorentzian_params[f'sub_scint_{i}']
        sub_scint_uncert = best_lorentzian_params[f'sub_scint_uncert_{i}']
        add_un = best_lorentzian_params[f'add_un{i}'][i-1] 
        
        
        # Apply filtering if ignore_subband_idx is provided
                # Create a mask to exclude ignored subbands
        if ignore_subband_idx is not None:
            
            # Ensure f_cents is a NumPy array
            f_cents = np.array(f_cents)
            sub_scint = np.array(sub_scint)
            sub_scint_uncert = np.array(sub_scint_uncert)
            add_un = np.asarray(add_un)
            
            ignore_subband_idx = np.array(ignore_subband_idx)
            all_indices = np.arange(len(f_cents))
            mask = ~np.isin(all_indices, ignore_subband_idx)  # Keep everything *not* in ignore_subband_idx
            
            print(mask)
            
            f_cents = f_cents[mask]
            sub_scint = sub_scint[mask]
            sub_scint_uncert = sub_scint_uncert[mask]
            add_un = add_un[mask]
            

    
        fit_min = Minimizer(scint_freq_relation_min, params, fcn_args=(f_cents, sub_scint, np.sqrt(np.array(sub_scint_uncert) ** 2 + add_un ** 2)))
        result_scint_i = fit_min.minimize()
        
        # plot 
        y600, y600_err = plot_power_law_fit(event_id, fftsize, downfreq, outdir, i, f_cents, result_scint_i, sub_scint, np.asarray(sub_scint_uncert), np.asarray(add_un), save_plot=save_plot)
        if print_report ==True:
            print(report_fit(result_scint_i))
        
        power_law_fits_lm_results_list.append(result_scint_i)
        y600_list.append(y600)
        y600_list_err.append(y600_err)
        
    return power_law_fits_lm_results_list, y600_list, y600_list_err
        
        
def plot_mod_inds(event_id, fftsize, downfreq, outdir, result_dict, best_model, save_plot=True):
    """
    Plot and optionally save the Lorentzian modulation indices for each subband in a given model.

    This function retrieves the modulation indices (MI) and their uncertainties from the provided Lorentzian model parameters, 
    calculates the average and standard deviation of the MI for each subband, and generates error bar plots for the MI as a 
    function of frequency. The function also saves the plots to the specified output directory if requested.

    Args:
        event_id (str): CHIME/FRB event ID.
        fftsize (int): The FFT size used in the upchannelisation.
        downfreq (float): The downfreq value used in the upchannelisation.
        outdir (str): The directory where the generated plots will be saved.
        result_dict (dict): A dictionary containing the Lorentzian model parameters, including the modulation indices and their uncertainties.
        best_model (int): The number of Lorentzians to process.
        save_plot (bool, optional): Whether to save the generated plots. Default is True.

    Returns:
        tuple: A tuple containing two lists:
            - mi_avg (list): The average modulation index for each subband.
            - mi_std (list): The standard deviation of the modulation index for each subband.

    """
    f_cents = result_dict['f_cents']
    
    for key in (result_dict.keys()):
        if str(best_model) in key:
            best_lorentzian_params  = result_dict[key] 

    mi_avg = []
    mi_std = []
    alpha=[1,0.3]
    for i in range(best_model):
        i+=1
        mod_ind = best_lorentzian_params[f'mods{i}']
        mi_avg.append(np.nanmean(mod_ind))
        mi_std.append(np.nanstd(mod_ind))

        mod_ind_uncert = best_lorentzian_params[f'mods{i}_uncert']
        

        plt.errorbar(f_cents,mod_ind,yerr=mod_ind_uncert,marker='x', color='k', alpha=alpha[i-1], label='%s Lorentz'%(i))
    plt.title("%s %s Lorentzian Modulation Indices"%(event_id,i))
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Modulation index')
    plt.ylim(0,3)
    plt.axhline(1,linestyle='--',color='k',alpha=0.5)
    plt.legend()
    if save_plot == True:
        plt.savefig(f'{outdir}/{event_id}_{i}_Lorentzian_modind_fftsize{fftsize}_downfreq{downfreq}.png')

    return mi_avg, mi_std 

def save_scint_results(event_id, power_law_fits, scint_600, scint_600_err, modind, modind_err, mw_scatt, mw_scint, outdir):
    """
    Save the results of the scintillation analysis to a JSON file.
    
    The results are saved as a JSON file in the specified output directory with the filename format 
          '<event_id>_scint_pipeline_results.json'

    This function compiles the results:
        - the power law fit parameters with uncertainties
        - the scintillation bandwidth at 600MHz with uncertainty
        - the modulation index with uncertainty
        - the NE2001 predictions for scattering and scintillation
    and saves as a JSON file in the specified output directory, outdir. 
    
    Args:
        event_id (str): CHIME/FRB event ID
        power_law_fits (list): A list of fit results containing parameters for each Lorentzians power-law fit.
        scint_600 (list): A list of scintillation values at 600 MHz for each Lorentzian.
        scint_600_err (list): A list of uncertainties associated with the scintillation values at 600 MHz for each Lorenzian.
        modind (list): A list of modulation indices for each subband.
        modind_err (list): A list of uncertainties associated with the modulation indices for each subband.
        mw_scatt (float): The value of the scattering parameter at 600 MHz from the NE2001 model.
        mw_scint (float): The value of the scintillation parameter at 600 MHz from the NE2001 model.
        outdir (str): The directory where the results JSON file will be saved.

    
    Returns: None
    - The results are saved as a JSON file in the specified output directory with the filename format 
          '<event_id>_scint_pipeline_results.json'.

    """

    final_results = {}
    for i in range(len(power_law_fits)):
        final_results["c_%s"%(i+1)] = power_law_fits[i].params['c'].value
        final_results["n_%s"%(i+1)] = power_law_fits[i].params['n'].value
        final_results["sigma_c_%s"%(i+1)] = power_law_fits[i].params['c'].stderr
        final_results["sigma_n_%s"%(i+1)] = power_law_fits[i].params['n'].stderr
        final_results["sbw_kHz_600MHz_%s"%(i+1)] = scint_600[i]
        final_results["sigma sbw_600MHz_%s"%(i+1)] = scint_600_err[i]
        final_results["mod_ind_%s"%(i+1)] = modind[i]
        final_results["sigma_mod_ind_%s"%(i+1)] = modind_err[i]

    try:
        final_results['NE2001_scatt_600MHz_ms'] = mw_scatt.value
        final_results['NE2001_scint_600MHz_kHz'] = mw_scint.value
    except:
        final_results['NE2001_scatt_600MHz_ms'] = 0
        final_results['NE2001_scint_600MHz_kHz'] = 0
        

    with open(outdir+'/%s_scint_pipeline_results.json'%event_id, 'w') as f:
        json.dump(final_results, f)

    return 
    
    

