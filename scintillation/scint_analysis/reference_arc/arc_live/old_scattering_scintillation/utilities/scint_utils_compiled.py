### Created 2024-09-16
### Maxwell A. Fine 
# This is helpful functions for scintillation analysis stuff, if you are reading this good luck
# The starter code for this project was sourced from Dr. Kenzi Nimmo's repository:  https://github.com/KenzieNimmo/FRB20221022A_scintillation)
import os
import sys
import json
import corner
import pickle
import importlib
import numpy as np

from tqdm import tqdm
from lmfit import minimize, Parameters, fit_report, Model, Minimizer, report_fit

from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM, Planck15
import astropy.units as u

import scipy.constants as cons
from scipy.fft import fft, fft2, fftshift
from scipy.signal import savgol_filter, resample, correlate


def bandpasscorr(initrow, off_pulse_idxs):
    """
    effective bandpass correction (don't need for scintillation analysis)
    """
    row = (initrow - np.mean(initrow[off_pulse_idxs])) / np.std(initrow[off_pulse_idxs])
    return row

def bandpasscorr_channel(arr, off_pulse_idxs):
    arr = [bandpasscorr(row, off_pulse_idxs) for row in arr]
    arr = np.asarray(arr)
    arr = np.nan_to_num(arr)
    return arr

# Functions for modeling the ACF of the spectrum
# For Burst with scintillation, the decorrelation (aka scintillation) bandwidth is given by
# fitting a Lorentzian to the auto-correlation function (ACF), the half-width at half-max (hwhm) is the decorrelation bandwidth.
# wiki on the Lorentzian function https://en.wikipedia.org/wiki/Cauchy_distribution
def lorentz(x, gamma1, m1):
    """
    Computes the Lorentzian function.

    Parameters:
    x (float): The variable for which the Lorentzian is evaluated.
    gamma1 (float): The half-width at half-maximum (HWHM) parameter.
    m1 (float): The peak amplitude of the Lorentzian function.

    Returns:
    float: The value of the Lorentzian function at x.
    """
    return m1**2 / (1 + (x / gamma1) ** 2)


def lorentz_w_c(x, gamma1, m1, c):
    """
    Computes the Lorentzian function with an additional constant offset.

    Parameters:
    x (float): The variable for which the Lorentzian is evaluated.
    gamma1 (float): The half-width at half-maximum (HWHM) parameter.
    m1 (float): The peak amplitude of the Lorentzian function.
    c (float): The constant offset added to the Lorentzian function.

    Returns:
    float: The value of the Lorentzian function at x with the constant offset.
    """
    return m1**2 / (1 + (x / gamma1) ** 2) + c
    
def doublelorentz_w_c(x, gamma1, m1, gamma2, m2, c):
    """
    Computes the sum of two Lorentzian functions with an additional constant offset.

    Parameters:
    x (float): The variable for which the Lorentzian functions are evaluated.
    gamma1 (float): The half-width at half-maximum (HWHM) parameter for the first Lorentzian.
    m1 (float): The peak amplitude of the first Lorentzian function.
    gamma2 (float): The half-width at half-maximum (HWHM) parameter for the second Lorentzian.
    m2 (float): The peak amplitude of the second Lorentzian function.
    c (float): The constant offset added to the sum of the Lorentzian functions.

    Returns:
    float: The value of the sum of the two Lorentzian functions at x with the constant offset.
    """
    return m1**2 / (1 + (x / gamma1) ** 2) + m2**2 / (1 + (x / gamma2) ** 2) + c


def triplelorentz(x, gamma1, m1, gamma2, m2, gamma3, m3):
    """
    Computes the sum of three Lorentzian functions.

    Parameters:
    x (float): The variable for which the Lorentzian functions are evaluated.
    gamma1 (float): The half-width at half-maximum (HWHM) parameter for the first Lorentzian.
    m1 (float): The peak amplitude of the first Lorentzian function.
    gamma2 (float): The half-width at half-maximum (HWHM) parameter for the second Lorentzian.
    m2 (float): The peak amplitude of the second Lorentzian function.
    gamma3 (float): The half-width at half-maximum (HWHM) parameter for the third Lorentzian.
    m3 (float): The peak amplitude of the third Lorentzian function.

    Returns:
    float: The value of the sum of the three Lorentzian functions at x.
    """
    return m1**2 / (1 + (x / gamma1) ** 2) + m2**2 / (1 + (x / gamma2) ** 2) + m3**2 / (1 + (x / gamma3) ** 2)


# Calculating Residuals for Lorentzian functions
def lorentz_withc_min(params, x, y, err):
    """
    Computes the residuals for a Lorentzian function with an additional constant offset.

    This function calculates the difference between the observed values `y` and the values predicted
    by a Lorentzian function with parameters `gamma1`, `m1`, and `c`, normalized by the error values `err`.

    Parameters:
    params (dict): A dictionary containing the parameters of the Lorentzian function. The dictionary should
                   have keys 'gamma1', 'm1', and 'c', each with an object that has a `value` attribute.
    x (array-like): The independent variable values at which the Lorentzian function is evaluated.
    y (array-like): The observed values to be compared against the Lorentzian function predictions.
    err (array-like): The error values for each observed data point, used to normalize the residuals.

    Returns:
    array-like: The normalized residuals between the observed values and the Lorentzian function predictions.
    """
    gamma1 = params['gamma1'].value
    m1 = params['m1'].value
    c = params['c'].value
    
    modulo = m1**2 / (1 + (x / gamma1) ** 2) + c
    # TODO max, make this a function call to the model instead of writing the model twice
    return (modulo - y) / err


def doublelorentz_withc_min(params, x, y, err):
    """
    Computes the residuals for the sum of two Lorentzian functions with an additional constant offset.

    This function calculates the difference between the observed values `y` and the values predicted
    by the sum of two Lorentzian functions with parameters `gamma1`, `m1`, `gamma2`, `m2`, and an offset `c`,
    normalized by the error values `err`.

    Parameters:
    params (dict): A dictionary containing the parameters of the Lorentzian functions. The dictionary should
                   have keys 'gamma1', 'm1', 'gamma2', 'm2', and 'c', each with an object that has a `value` attribute.
    x (array-like): The independent variable values at which the Lorentzian functions are evaluated.
    y (array-like): The observed values to be compared against the Lorentzian function predictions.
    err (array-like): The error values for each observed data point, used to normalize the residuals.

    Returns:
    array-like: The normalized residuals between the observed values and the sum of the two Lorentzian functions
                with the constant offset.
    """
    gamma1 = params['gamma1'].value
    m1 = params['m1'].value
    gamma2 = params['gamma2'].value
    m2 = params['m2'].value
    c = params['c'].value
    
    modulo = m1**2 / (1 + (x / gamma1) ** 2) + m2**2 / (1 + (x / gamma2) ** 2) + c
    # TODO max, make this a function call to the model instead of writing the model twice
    return (modulo - y) / err


def triplelorentz_min(params, x, y, err):
    """
    Computes the residuals for the sum of three Lorentzian functions.

    This function calculates the difference between the observed values `y` and the values predicted
    by the sum of three Lorentzian functions with parameters `gamma1`, `m1`, `gamma2`, `m2`, `gamma3`, and `m3`,
    normalized by the error values `err`.

    Parameters:
    params (dict): A dictionary containing the parameters of the Lorentzian functions. The dictionary should
                   have keys 'gamma1', 'm1', 'gamma2', 'm2', 'gamma3', and 'm3', each with an object that has a `value` attribute.
    x (array-like): The independent variable values at which the Lorentzian functions are evaluated.
    y (array-like): The observed values to be compared against the Lorentzian function predictions.
    err (array-like): The error values for each observed data point, used to normalize the residuals.

    Returns:
    array-like: The normalized residuals between the observed values and the sum of the three Lorentzian functions.
    """
    gamma1 = params['gamma1'].value
    m1 = params['m1'].value
    gamma2 = params['gamma2'].value
    m2 = params['m2'].value
    gamma3 = params['gamma3'].value
    m3 = params['m3'].value
    
    modulo = m1**2 / (1 + (x / gamma1) ** 2) + m2**2 / (1 + (x / gamma2) ** 2) + m3**2 / (1 + (x / gamma3) ** 2)
    # TODO max, make this a function call to the model instead of writing the model twice
    return (modulo - y) / err


def shift(v, i, nchan):
    """                                                                                                                                                            
    function v by a shift i                                                                                                                                        
    nchan is the number of frequency channels (to account for negative lag)                                                                                        
    """
    n = len(v)
    r = np.zeros(3*n)
    i+=nchan-1 #to account for negative lag                                                                                                                        
    i = int(i)
    r[i:i+n] = v
    return r

def autocorr(spec, v=None,zerolag=False,maxlag=None,offspec_mean=None,freq=None):
    """
    x is the 1D array you want to autocorrelate
    v is the array of 1s and 0s representing a mask where 1 is no mask, and 0 is mask
    zerolag = True will keep the zero lag noise spike, otherwise it won't compute the zero lag
    maxlag = None will compute the ACF for the entire length of x
    maxlag = bin_number will compute the ACF for lags up to x[bin_number]
    
    """
    nchan=len(spec)
    if v is None:
        v = np.ones_like(spec)
        
    
    x = np.copy(spec)
    xmean=np.nanmean(x[v!=0])
    if freq is not None:
        xmean=5417.46963982*freq[v!=0]**-1.5
        #xmean=1.03055693e+09*freq[v!=0]**-3.451
        print('doing method 2')

    if offspec_mean is None:
        denom = xmean**2
    else:
        denom = (xmean - offspec_mean)**2


    x[v!=0] -= xmean#x[v!=0].mean()
    if maxlag==None:
        ACF = np.zeros_like(x)
    else:
        ACF = np.zeros_like(x)[:int(maxlag)]
    
    #x[v!=0] /= (xmean-offspec_mean)

    for i in tqdm(range(len(ACF))):
        if zerolag == False:
                if i>1:
                        m = shift(v,0,nchan)*shift(v,i,nchan)

                        ACF[i-1] = np.nansum(shift(x,0,nchan)*shift(x, i,nchan)*m) / (np.sum(m)*denom)
                        #ACF[i-1]=np.nansum(shift(x,0,nchan)*shift(x,i,nchan)*m) / np.sum(m)
        else:
                m = shift(v,0,nchan)*shift(v,i,nchan)
                ACF[i] = np.nansum(shift(x,0,nchan)*shift(x, i,nchan)*m) / (np.sum(m)*denom)

    return ACF

def autocorr_midx(x, v=None,zerolag=False,maxlag=None):
    """
    x is the 1D array you want to autocorrelate
    v is the array of 1s and 0s representing a mask where 1 is no mask, and 0 is mask
    zerolag = True will keep the zero lag noise spike, otherwise it won't compute the zero lag
    maxlag = None will compute the ACF for the entire length of x
    maxlag = bin_number will compute the ACF for lags up to x[bin_number]
    """
    nchan=len(x)
    if v is None:
        v = np.ones_like(x)
    x = x.copy()
    x[v!=0] -= x[v!=0].mean()
    if maxlag==None:
        ACF = np.zeros_like(x)
    else:
        ACF = np.zeros_like(x)[:int(maxlag)]
    #print(maxlag)
    #print('acf length', len(ACF))
    for i in tqdm(range(len(ACF))):
        if zerolag == False:
                if i>1:
                        m = shift(v,0,nchan)*shift(v,i,nchan)
                        ACF[i-1] = np.sum(shift(x,0,nchan)*shift(x, i,nchan)*m)/np.sqrt(np.sum(shift(x, 0, nchan)**2*m)*np.sum(shift(x, i, nchan)**2*m))
        else:
                m = shift(v,0,nchan)*shift(v,i,nchan)
                ACF[i] = np.sum(shift(x,0,nchan)*shift(x, i,nchan)*m)/np.sqrt(np.sum(shift(x, 0, nchan)**2*m)*np.sum(shift(x, i, nchan)**2*m))
            

    return ACF

# Loading in data, and correcting for CHIME artifcats
# TODO max fine, clean up the loading in data functions, and add doc strings
#def get_data(event):
#     for par in event["measured_parameters"]:
#          if par["pipeline"]["name"] == "realtime":
#              event_date = par["datetime"].split(" ")[0].split("-")
#     data_path = "/arc/projects/chime_frb/data/chime/baseband/processed/" + \
#         event_date[0] + "/" + \
#         event_date[1] + "/" + \
#         event_date[2] + "/astro_" + \
#         str(event["id"]) + "/singlebeam_" + str(event["id"]) +".h5"
#     return data_path
    
def deripple(ds, offpulse):
    ds_final = np.zeros_like(ds)
    if len(ds.shape)==3:
        for chan in range(offpulse.shape[0]):
            for pol in range(2):
                if np.std(offpulse[chan,pol,:])!=0:
                    ds_final[chan,pol,:]=ds[chan,pol,:]-np.mean(offpulse[chan,pol,:])
                    offpulse[chan,pol,:]-=np.mean(offpulse[chan,pol,:])
                    ds_final[chan,pol,:]=ds_final[chan,pol,:]/np.std(offpulse[chan,pol,:])
    if len(ds.shape)==2:
        for chan in range(offpulse.shape[0]):
            if np.std(offpulse[chan,:])!=0:
                ds_final[chan,:]=ds[chan,:]-np.mean(offpulse[chan,:])
                offpulse[chan,:]-=np.mean(offpulse[chan,:])
                ds_final[chan,:]=ds_final[chan,:]/np.std(offpulse[chan,:])
    return ds_final


def fill_missing_chans(ds,bbdata):
    """
    ds shape [freq<1024,pol,time]
    bbdata object
    """
    new_data = np.zeros([1024,ds.shape[1],ds.shape[2]],dtype=np.complex64)
    
    freq_id = bbdata.index_map["freq"]["id"]
    freqs = bbdata.index_map["freq"]["centre"]
    
    for chan in np.arange(1024):
        if chan in freq_id:
            new_data[chan,:,:]=ds[np.where(freq_id==chan),:,:]
    
 
    
    data_masked=np.ma.masked_where(new_data==0,new_data)
    new_freq_id = np.arange(1024)
    
    f_res=np.abs((freqs[1]-freqs[0])/(freq_id[1]-freq_id[0]))
    if freq_id[0]==0:
        fmax=freqs[0]
    else:
        fmax = freqs[0]+(f_res*(freq_id[0]+1))
    if freq_id[-1]==1023:
        fmin=freqs[-1]
    else:
        fmin = freqs[-1] - (f_res*(1023-freq_id[-1]))

    new_freqs = np.linspace(fmin,fmax,1024)
    
    
    return data_masked, new_freqs, new_freq_id


def upchannel_fast(wfall, freq_id, fftsize=32, downfreq=2):
    
    # swap axes ordering to (pol,time,chan)
    wfall = np.swapaxes(wfall, 0, 1)
    wfall = np.swapaxes(wfall, 1, 2)
    
    npol, nsamp, nchan = wfall.shape
    upchan = fftsize // downfreq
    downtime = 1  # Hard-coded as per original code
    nblock = nsamp // (fftsize * downtime)
    
    # Precompute chan_id_upchan
    chan_id_upchan = (freq_id[:, None] * upchan + np.arange(upchan)).ravel()
    
    # Precompute frequency array
    freq_top_mhz = 800.1953125 
    freq_bottom_mhz = 400.1953125
    f_upchan_bandtot = np.linspace(freq_top_mhz, freq_bottom_mhz, upchan * 1024)
    
    # Initialize output array
    spec = np.zeros((npol, nblock, nchan * upchan), dtype=np.complex64)
    
    for pol in range(npol):
        # Truncate to valid blocks and reshape
        pol_data = wfall[pol, :nblock * fftsize, :]
        reshaped = pol_data.reshape(nblock, fftsize, nchan)
        
        # Compute FFT and shift frequencies
        fft = np.fft.fft(reshaped, axis=1)
        fft_shifted = np.fft.fftshift(fft, axes=1)
        
        # Downsample in frequency
        downsampled = fft_shifted.reshape(nblock, upchan, downfreq, nchan).mean(axis=2)
        
        # Organize output correctly
        transposed = downsampled.transpose(0, 2, 1)
        spec[pol] = transposed.reshape(nblock, nchan * upchan)
    
    return spec, f_upchan_bandtot[chan_id_upchan], chan_id_upchan


def upchannel(wfall, freq_id, fftsize=32, downfreq=2):
    """Upchannelize a dynamic spectrum.

    Performs the CHIME upchannelization on a dynamic spectrum,
    average every 3 time samples (hard-coded) and every `downfreq`
    frequency channels after upchannelization.

    Parameters
    ----------
    wfall : np.ndarray
        Dynamic spectrum to process.
    freq_id : np.1darray
        frequency channel ids
    fftsize : int
        FFT step-size.
    downfreq : int
        Downsampling factor in frequency.

    Returns
    -------
    upchan : np.ndarray[:, nfreq]
        Array of upchannelization frequencies, ordered high to low
        (order will change later!).
    """
    # swap axes ordering to (pol,time,chan)
    wfall = np.swapaxes(wfall, 0, 1)
    wfall = np.swapaxes(wfall, 1, 2)

    # set downtime to 1 => no averaging over complex numbers!!!
    downtime = 1

    npol, nsamp, nchan = wfall.shape

    # upchannelization factor (16 by default)
    upchan = fftsize // downfreq

    # number of blocks
    nblock = nsamp // (fftsize * downtime)

    # initialise array for spectrum
    spec = np.zeros((npol, nblock, nchan * upchan), dtype=np.complex64)

    # iterate over blocks and perform the upchannelization
    count = 0
    chan_id_upchan = np.zeros((nchan * upchan), dtype=int)

    #CHIME band
    freq_top_mhz = 800.1953125 
    freq_bottom_mhz = 400.1953125
    f_upchan_bandtot = np.linspace(
        freq_top_mhz, freq_bottom_mhz, upchan * 1024
    )
    for pol in range(npol):
        for bi in range(nblock):
            for chidx in range(nchan):
                # cut out the correct timestream section
                ts = wfall[pol, bi * fftsize : bi * fftsize + fftsize, chidx].copy()
                # perform a FFT
                ft = fftshift(fft(ts))
                # downsample in frequency
                ft = ft.reshape(upchan, downfreq).mean(axis=1).copy()

                spec[pol, bi, chidx * upchan : chidx * upchan + upchan] = ft

                chan_id_upchan[chidx * upchan : chidx * upchan + upchan] = np.arange(
                    upchan * freq_id[chidx], upchan * freq_id[chidx] + upchan, 1
                )

                count += 1

    return spec, f_upchan_bandtot[chan_id_upchan], chan_id_upchan


def data_dedisp_derip_filled_masked(event_id, dm, downsample_factor=32, interactive=True, off=False, file=None):
    """
    given an CHIME event_id and dm, 
    output the data ( RFI zapped, missing channels filled, derippled, coherent dedispersed), the frequencies and frequency channel IDs
    
    """
    
    #read into bbdata object
    event = master.events.get_event(event_id)
    if file is None:
        FRB_data=get_data(event)
    else:
        FRB_data = file


    frb_bbdata = BBData.from_file(FRB_data) #baseband data
    
    if "tiedbeam_power" not in list(frb_bbdata.keys()):
        tiedbeam_baseband_to_power(
            frb_bbdata, time_downsample_factor=1, dm=dm, dedisperse=True, time_shift=False
        )

    output=get_snr(frb_bbdata,DM=dm,diagnostic_plots=True,return_full=True,downsample=downsample_factor,DM_range=None,spectrum_lim=False)

    
    # plt.savefig('./snr.png')
    # plt.close()
    # plt.plot(np.nanmean(output[2][:,744:778], axis=1))
    # plt.savefig('./spec.png')
    # exit()
    #dedisperse
    if dm!=0:
        coherent_dedisp(frb_bbdata, dm, time_shift=False,write=True)
        data_dedisp,freq, freq_id=incoherent_dedisp(frb_bbdata,dm,fill_wfall=False)
    else:
        frb_bbdata['tiedbeam_baseband']
    #identify off burst region to use
    # power=np.abs(data_dedisp)**2
    # I = np.sum(power,axis=1)
    # Iscr=scrunch(I,tscrunch=downsample_factor,fscrunch=1)
    Iscr = output[2]
    
    if interactive==True:
        plt.close('all')
        plt.plot(np.nanmean(Iscr,axis=0))
        plt.show()
        plt.savefig('./temp.png')
       
        answer=input('Please define the bin range to use for the off burst statistics (beginbin,endbin): ')
        answer = answer.split(',')
    else:
        #nanind=np.argwhere(np.isnan(Iscr[-1,:]))[0][0]
        #st_tbin, end_tbin = get_main_peak_lim(Iscr[:,:nanind+500],diagnostic_plots=False,normalize_profile=True)
        st_tbin, end_tbin = get_main_peak_lim(Iscr,diagnostic_plots=False,normalize_profile=True)
        lim=np.array([st_tbin, end_tbin])
        answer=[0,lim[0]]
        
    
    #get rid of invalid channels as determined by get_snr
    valid_channels=output[5]
    data_dedisp[~valid_channels] = 0
    
    #let's now figure out what data we want to keep
    power=np.abs(data_dedisp)**2
    I = np.sum(power,axis=1)
    Iscr=scrunch(I,tscrunch=downsample_factor,fscrunch=1)
    nanind=np.argwhere(np.isnan(Iscr[-1,:]))[0][0]
    Iscr=Iscr[:,:nanind]
    
    if interactive==True:
        # plt.imshow(Iscr,aspect='auto')
        # plt.show()
        plt.close('all')
        plt.plot(np.nanmean(Iscr,axis=0))
        plt.show()
        plt.savefig('./temp.png')
        answer=input('Please define the time bin range to keep (beginbin,endbin): ')
        answer = answer.split(',')
    else:
        answer=[0,Iscr.shape[1]]
            
    data_dedisp_masked, freqs, freq_id = fill_missing_chans(data_dedisp[:,:,int(answer[0])*downsample_factor:int(answer[1])*downsample_factor],frb_bbdata)
    

    
    #now let's try additional RFI zapping
    chan_spectrum = np.nansum( np.nansum(np.abs(data_dedisp_masked)**2,axis=1),axis=-1)
    chan_spectrum_snr = (chan_spectrum - np.nanmedian(chan_spectrum))
    chan_spectrum_snr /= (1.4826*median_abs_deviation(chan_spectrum,nan_policy='omit'))
    miss_chan_mask = np.where( (chan_spectrum_snr < -1) *(chan_spectrum > 0) )
    data_dedisp_masked[miss_chan_mask,:,:] = 0
    
    data_dedisp_masked = np.ma.masked_where(data_dedisp_masked==0,data_dedisp_masked)
    
    freqs=np.flip(freqs)
    
    return data_dedisp_masked, freqs, freq_id


# Accounting for scallops
import numpy as np

def make_scallop_model(off_data, fftsize, downfreq):
    """
    Create a scallop correction model based on off-burst complex voltage data.

    This function generates a model to correct for frequency-dependent variations 
    (scalloping) in radio astronomy data. It processes the off-burst complex voltage 
    data to identify and correct these variations.

    Parameters:
    -----------
    off_data : ndarray
        A complex voltage array with shape (polarization, time, frequency) containing 
        off-burst data. The data should be in the form of complex numbers.
    fftsize : int
        The size of the FFT used for upchannelisation.
    downfreq : int
        The down-sampling factor in the frequency domain.

    Returns:
    --------
    model : ndarray
        The scallop correction model averaged over the reshaped frequency segments.
    spec_noise_masked_corr : ndarray
        The corrected spectrum with the scalloping effect minimized.
    inds : ndarray
        Indices of the outliers in the normalized spectrum that were masked.
    """
    # Calculate the noise power from the complex voltage data
    noise_power = np.abs(off_data**2)
    
    # Average over the polarization axis and transpose
    I_noise = np.mean(noise_power, axis=0).T
    
    # Compute the mean spectrum by averaging over time
    spec_noise = np.nanmean(I_noise, axis=1)
    
    # Normalize the spectrum by subtracting the mean and dividing by the standard deviation
    noise_mean = np.mean(spec_noise)
    noise_std = np.std(spec_noise)
    spec_noise_norm = (spec_noise - noise_mean) / noise_std
    
    # Identify outliers where the normalized spectrum is greater than 3 standard deviations
    inds = np.where(np.abs(spec_noise_norm) > 3)[0]
    
    # Set the outlier values to zero
    spec_noise[inds] = 0
    
    # Mask the zeroed values
    spec_noise_masked = np.ma.masked_where(spec_noise == 0, spec_noise)
    
    # Reshape the masked spectrum for averaging based on FFT size and down-sampling factor
    spec_noise_masked_reshape = spec_noise_masked.reshape(
        len(spec_noise_masked) // (fftsize // downfreq), 
        (fftsize // downfreq)
    )
    
    # Create the scallop model by averaging over the reshaped frequency segments
    model_scallop = np.nanmean(spec_noise_masked_reshape, axis=0)
    
    # Tile the model to match the original data size in the frequency domain
    model = np.tile(model_scallop, I_noise.shape[0] // (fftsize // downfreq))
    
    # Correct the original masked spectrum by dividing by the scallop model
    spec_noise_masked_corr = spec_noise_masked / model
    
    # Mask the corrected spectrum where the values are zero
    spec_noise_masked_corr = np.ma.masked_where(spec_noise_masked_corr == 0, spec_noise_masked_corr)
    
    return model, spec_noise_masked_corr, inds

#ACF stuff
def acf_scint_plot(ds, freq_ids, freqs, time_range, lagrange_for_fit=10., diagnostic_plots=True, maxlag=None, offspec_mean=None):
    """
    Computes and plots the Autocorrelation Function (ACF) of a given spectrum or 2D dataset (frequency vs. time).
    
    Parameters:
    ds (ndarray): A 2D array [freq, time] or a 1D spectrum. Represents the data from which to compute the ACF.
    freq_ids (array-like): Array of frequency channel numbers corresponding to 'ds'.
    freqs (array-like): Array of central frequencies in MHz corresponding to 'ds'.
    time_range (list): A two-element list [begin_bin, end_bin] specifying the time range for computing the burst spectrum. Only needed if 'ds' is 2D.
    lagrange_for_fit (float, optional): The range in MHz used to define the lag range for fitting the Lorentzian. Default is 10 MHz.
    diagnostic_plots (bool, optional): If True, generates diagnostic plots during the process. Default is True.
    maxlag (float, optional): Maximum lag in MHz up to which the ACF is computed. If None, the ACF is computed for the entire range.
    offspec_mean (float, optional): Optional mean value used in the autocorrelation computation.

    Returns:
    acf (ndarray): The computed ACF.
    lags (ndarray): The lag values corresponding to the ACF.
    result (ModelResult or None): The result of the Lorentzian fit to the ACF, or None if the fit failed.

    Notes:
    - This function optionally fits a Lorentzian to the ACF to measure scintillation bandwidth.
    - Generates plots if 'diagnostic_plots' is set to True.
    """
    
    # Determine the frequency resolution in MHz
    num_chan_diff = int(np.abs(freq_ids[1] - freq_ids[0]))
    f_res = np.abs(freqs[1] - freqs[0]) / float(num_chan_diff)
    print("Frequency resolution is %.5f MHz" % f_res)

    # Compute the spectrum; if 'ds' is 2D, average over the specified time range
    if ds.ndim == 1:
        spec = ds
    else:
        spec = np.mean(ds[:, time_range[0]:time_range[1]], axis=1)
        
    # Handle masking; mask elements where 'ds' equals zero if no mask is present
    try:
        if ds.ndim == 1:
            mask = np.abs(np.array(ds.mask, dtype=int) - 1)
        else:
            mask = np.abs(np.array(ds.mask[:, 0], dtype=int) - 1)
    except:
        print('Masking where the array = 0')
        ds = np.ma.masked_where(ds == 0, ds)
        if ds.ndim == 1:
            mask = np.abs(np.array(ds.mask, dtype=int) - 1)
        else:
            mask = np.abs(np.array(ds.mask[:, 0], dtype=int) - 1)

    # Determine the maximum lag in bins if specified
    if maxlag is None:
        maxlag_bin = None
    else:
        maxlag_bin = int(maxlag / f_res)
        
    # Compute the autocorrelation function (ACF)
    acf = autocorr(spec, v=mask, zerolag=False, maxlag=maxlag_bin, offspec_mean=offspec_mean)
    height = acf[0]
    lags = np.arange(len(acf)) + 1
    
    # Symmetrize the ACF and lag arrays
    acf = acf[1:]
    lags = lags[1:]
    acf = np.concatenate((acf[::-1], acf))
    lags = np.concatenate((-1 * lags[::-1], lags)) * f_res
    
    # Generate diagnostic plots if requested
    if diagnostic_plots:
        plt.plot(lags, acf, drawstyle='steps-mid', color='k', linewidth=0.5)
        plt.show()
    
        plt.plot(lags, acf, drawstyle='steps-mid', color='k', linewidth=0.5, label="%.5f MHz" % f_res)
    
    # Fit a Lorentzian to measure scintillation bandwidth
    try:
        gmodel = Model(lorentz)
        acf_for_fit = acf[int(len(acf) / 2.) - int(lagrange_for_fit / f_res):int(len(acf) / 2.) + int(lagrange_for_fit / f_res)]
        lags_for_fit = lags[int(len(acf) / 2.) - int(lagrange_for_fit / f_res):int(len(acf) / 2.) + int(lagrange_for_fit / f_res)]
        result = gmodel.fit(acf_for_fit, x=lags_for_fit, gamma=0.001, m=1, c=0)
        
        if diagnostic_plots:
            plt.plot(lags, lorentz(lags, result.params['gamma'], result.params['m'], result.params['c']),
                     color='orange', label='scint bw = %.2f MHz' % result.params['gamma'].value)
            plt.xlim(-np.abs(result.params['gamma'].value) * 10, np.abs(result.params['gamma'].value) * 10)
            plt.ylim(-0.2, height + 0.05)
            plt.legend()
            plt.show()
        
        return acf, lags, result
    except:
        print("Could not fit a Lorentzian")
        if diagnostic_plots:
            plt.legend()
            plt.show()
        
        return acf, lags