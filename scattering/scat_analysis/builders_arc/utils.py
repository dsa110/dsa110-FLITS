import os
import sys
import numpy as np
from lmfit import minimize, Parameters, fit_report, Model
from tqdm import tqdm
import matplotlib.pyplot as plt
#from pfb_tools import DeconvolvePFB
from scipy.stats import median_abs_deviation
from scipy.interpolate import make_lsq_spline
import matplotlib
from scipy import signal

from baseband_analysis.core.signal import get_main_peak_lim, tiedbeam_baseband_to_power
from baseband_analysis.core.bbdata import BBData
from baseband_analysis.analysis.snr import get_snr, get_profile
from baseband_analysis.core.sampling import scrunch
from baseband_analysis.core.dedispersion import coherent_dedisp, incoherent_dedisp
from baseband_analysis.analysis.polarization import get_burst_envelope

import chime_frb_api
master = chime_frb_api.frb_master.FRBMaster(base_url = "https://frb.chimenet.ca/frb-master")
master.API.authorize()
auth = {"Authorization": master.API.access_token}


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

def get_burst_envelope_kn(
    power: tuple, thres: float = 5, pad: float = 0.0, diagnostic_plots: bool = False):

    """
    Get indices of power floor (noiselike to within 3 sigma).

    Parameters
    ----------
    power: tuple.
       Power from which to deriv the burst profile.

    thres: float. Default is 5.
       Threshold for power floor.

    pad: float. Default is 0.
       Set a pad around the burst limits.

    diagnostic_plots: boolean. Default is False.
       Indicate whether to generate diagnostic plots.

    Returns
    -------
    lims: list of floats.
       Burst lower and upper limits.
    """

    # Get the power and power floor
    prof = get_profile(power)
    floor = prof.copy()
    prof -= np.nanmedian(floor)
    floor -= np.nanmedian(floor)
    prof /= np.nanstd(floor)
    floor /= np.nanstd(floor)
    #floor /= np.nanmedian(abs(floor-np.nanmedian(floor)))
    while True:
        peak_t0, peak_t1 = get_main_peak_lim(floor, floor_level=thres)
        if (peak_t1 - peak_t0) == floor.size:
            break
        floor[peak_t0:peak_t1] = np.nan
        #floor -= np.nanmedian(floor)
        #floor /= np.nanstd(floor)
        #         floor /= np.nanmedian(abs(floor-np.nanmedian(floor)))
        idx = floor > thres  # Identify bins larger than 3 sigma
        floor[idx] = np.nan
        if len(idx[idx]) == 0:  # If no bins larger than 3 sigma
            break
        if len(floor[~np.isnan(floor)]) == 0:  # All bins larger than 3 sigma
            break
    idx = np.isnan(floor)
    try:
        lims = np.array([np.argwhere(idx == True).min(), np.argwhere(idx == True).max()])
    except:
        lims=[0,len(floor)]
        
    if lims[0] - ((lims[1] - lims[0]) * pad) > 0:
        lims[0] -= (lims[1] - lims[0]) * pad

    if lims[1] + ((lims[1] - lims[0]) * pad) < floor.size:
        lims[1] += (lims[1] - lims[0]) * pad

    # Generate diagnostic plots
    if diagnostic_plots:

        plt.plot(prof)
        plt.plot(floor)
        plt.axvline(lims[0], c="k", ls="--")
        plt.axvline(lims[1], c="k", ls="--")
        plt.xlabel("Time [bins]")
        plt.ylabel("S/N")

    if isinstance(diagnostic_plots, bool):
        plt.show()

    # Save the plot
    else:
        plot_name = "burst_envelope_limits.png"
        plt.savefig(os.path.join(diagnostic_plots, plot_name))
        plt.close("all")

    # Return the burst limits
    return lims