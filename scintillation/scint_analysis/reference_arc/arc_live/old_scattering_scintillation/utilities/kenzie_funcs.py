import sys
#sys.path.insert(0, "/arc/home/knimmo/local_software/fitburst_morphology/baseband-analysis/")
#sys.path.insert(0, "/arc/home/knimmo/local_software/fitburst_morphology/baseband-analysis/fitburst/")
#sys.path.insert(0, "/arc/home/knimmo/scripts/")
import os

import numpy as np
from lmfit import minimize, Parameters, fit_report, Model
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update(
    {
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    }
)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['image.origin'] = 'lower'
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Tahoma', 'Verdana', 'Lucida Grande', 'DejaVu Sans']

cmap = matplotlib.cm.get_cmap('magma_r')
colors = ["white", "white", "white"]+[cmap(i/100) for i in range(100)]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["lime","navajowhite", "dodgerblue"])

import seaborn as sns

sns.set_context('talk') 
sns.set(font_scale=1.8)
sns.set_palette('colorblind')
sns.set_style('ticks')

#from pfb_tools import DeconvolvePFB
from scipy.stats import median_abs_deviation
from scipy.interpolate import make_lsq_spline
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

import json
from copy import deepcopy
import chime_frb_constants as const
import fitburst as fb
from scipy.interpolate import interp2d


# def get_data(event):
#      for par in event["measured_parameters"]:
#           if par["pipeline"]["name"] == "realtime":
#               event_date = par["datetime"].split(" ")[0].split("-")
#      data_path = "/arc/projects/chime_frb/data/chime/baseband/processed/" + \
#          event_date[0] + "/" + \
#          event_date[1] + "/" + \
#          event_date[2] + "/astro_" +
#          str(event["id"]) + "/Run_pre2025"+ "/singlebeam_" + str(event["id"]) +".h5"
#      return data_path


def get_data(event):
    event_date = None

    for par in event.get("measured_parameters", []):
        if par.get("pipeline", {}).get("name") == "realtime":
            event_date = par.get("datetime", "").split(" ")[0].split("-")
            break  # Stop iteration once the date is found

    if not event_date or len(event_date) != 3:
        raise ValueError("Invalid or missing event date.")

    data_path = (
        f"/arc/projects/chime_frb/data/chime/baseband/processed/"
        f"{event_date[0]}/{event_date[1]}/{event_date[2]}/astro_"
        #f"{event['id']}/Run_pre2025/singlebeam_{event['id']}.h5"
        f"{event['id']}/singlebeam_{event['id']}.h5"
    )
    
    return data_path
    
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

def fftsize16_functions(name='ziggy'):
    if name=='ziggy':
        model = np.array([ 0.63338375,  0.71153545,  0.84496403,  0.99040335,  1.11110604,
        1.19043183,  1.23157179,  1.24731326,  1.24810541,  1.23464894,
        1.19744003,  1.12338078,  1.0074048 ,  0.86338407,  0.72602457,
        0.63890308])
    if name=='eve':
        model = np.array([0.52225748, 0.58330915, 0.6868705, 0.80121821,
                          0.89386546, 0.95477358, 0.98662733, 0.99942558,
                          0.99988676, 0.98905127, 0.95874124, 0.90094667,
                          0.81113021, 0.6999944, 0.59367968, 0.52614263])
    if name=='richard':
        model = DeconvolvePFB(Q=16).Wt2.sum(axis=1)
        model = np.roll(model, 8)
        
        
        
    model = np.tile(model,1024)
    return model

def acf_scint_plot(ds,freq_ids,freqs,time_range,lagrange_for_fit=10.,diagnostic_plots=True, maxlag=None, offspec_mean=None):
    """
    ds is either 2D array [freq,time] or 1d spectrum
    freq_ids and freqs are the frequency channel numbers and central frequency in MHz mapping ds
    time_range is [begin_bin,end_bin] within which the burst spectrum is computed. Only needed if ds is 2d.
    lagrange_for_fit is the lag range in MHz used to define the lag range out to which the final lorentzian will be fit
    diagnostic_plots will produce plots while running
    maxlag is the maximum lag in MHz to compute the ACF out to
    """
    
    #figure out the frequency resolution
    num_chan_diff = int(np.abs(freq_ids[1]-freq_ids[0]))
    f_res = np.abs(freqs[1]-freqs[0])/float(num_chan_diff)
    print("Frequency resolution is %.5f MHz"%f_res)

    #make the spectrum
    if ds.ndim == 1:
        spec = ds
    else:
        spec=np.mean(ds[:,time_range[0]:time_range[1]],axis=1)
        
    try:
        if ds.ndim == 1:
            mask =  np.abs(np.array(ds.mask,dtype=int)-1)
        else:
            mask=np.abs(np.array(ds.mask[:,0],dtype=int)-1)
    except:
        print('masking where the array = 0')
        ds = np.ma.masked_where(ds==0,ds)
        if ds.ndim == 1:
            mask =  np.abs(np.array(ds.mask,dtype=int)-1)
        else:
            mask=np.abs(np.array(ds.mask[:,0],dtype=int)-1)

    #ACF
    if maxlag ==None:
        maxlag_bin=None
    else:
        maxlag_bin=int(maxlag/f_res)
        
    acf=autocorr(spec, v=mask,zerolag=False,maxlag=maxlag_bin,offspec_mean=offspec_mean,freq=None)
    height=acf[0]
    lags=np.arange(len(acf))+1
    
    acf=acf[1:]
    lags=lags[1:]
    acf=np.concatenate((acf[::-1],acf))
    lags=np.concatenate((-1*lags[::-1],lags))*f_res
    
    #plotting
    if diagnostic_plots==True:
        plt.plot(lags,acf,drawstyle='steps-mid',color='k',linewidth=0.5)
        plt.show()
    
        plt.plot(lags,acf,drawstyle='steps-mid',color='k',linewidth=0.5,label="%.5f MHz"%f_res)
    #fit lorentzian to measure scintillation bandwidth
    try:
        gmodel = Model(lorentz)
        acf_for_fit = acf[int(len(acf)/2.)-int(lagrange_for_fit/f_res):int(len(acf)/2.)+int(lagrange_for_fit/f_res)]
        lags_for_fit = lags[int(len(acf)/2.)-int(lagrange_for_fit/f_res):int(len(acf)/2.)+int(lagrange_for_fit/f_res)]
        result = gmodel.fit(acf_for_fit, x=lags_for_fit, gamma=0.001, m=1, c=0)
        if diagnostic_plots == True:
            plt.plot(lags,lorentz(lags,result.params['gamma'],result.params['m'],result.params['c']),color='orange',label='scint bw = %.2f MHz'%result.params['gamma'].value)
            plt.xlim(-np.abs(result.params['gamma'].value)*10,np.abs(result.params['gamma'].value)*10)
            plt.ylim(-0.2,height+0.05)
            plt.legend()
            plt.show()
    
        return acf, lags, result
    except:
        print("Could not fit a Lorentzian")
        if diagnostic_plots==True:
            plt.legend()
            plt.show()
        
        return acf, lags


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

def autocorr_m(x, v=None,zerolag=False,maxlag=None):
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

def lorentz(x,gamma,m, c):
        #return (y0*gamma**2)/(((x)**2)+gamma**2)+c
        return m**2 / (1+(x/gamma)**2) + c

def doublelorentz(x,gamma1,m1, gamma2,m2,c):
        return m1**2 / (1+(x/gamma1)**2) + m2**2 / (1+(x/gamma2)**2) + c
    
def scint_freq_relation(v,c,n):
    return c*(1/v**n)
   
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

def data_dedisp_derip_filled_masked(event_id, dm, downsample_factor=32, interactive=True, off=False, file=None,zap_extra=True,diagnostic_plot=None, time_range=None):
    """
    given an event_id and dm, 
    output the data ( RFI zapped, missing channels filled, derippled, coherent dedispersed), the frequencies and frequency channel IDs
    
    """
    #read into bbdata object
    #File = None # Max edited
    #master = chime_frb_api.frb_master.FRBMaster(base_url = "https://frb.chimenet.ca/frb-master")
    #master.API.authorize()
    #auth = {"Authorization": master.API.access_token} 
    #event = master.events.get_event(event_id)
    if file is None:
        FRB_data=get_data(event)
    else:
        FRB_data = file


    frb_bbdata = BBData.from_file(FRB_data)
    
    if "tiedbeam_power" not in list(frb_bbdata.keys()):
        tiedbeam_baseband_to_power(
            frb_bbdata, time_downsample_factor=1, dm=dm, dedisperse=True, time_shift=False
        )

    output=get_snr(frb_bbdata,DM=dm,diagnostic_plots=True,return_full=True,downsample=downsample_factor,DM_range=None,spectrum_lim=False)

    #dedisperse
    if dm!=0:
        coherent_dedisp(frb_bbdata, dm, time_shift=False,write=True)
        data_dedisp,freq, freq_id=incoherent_dedisp(frb_bbdata,dm,fill_wfall=False)
    else:
        frb_bbdata['tiedbeam_baseband']
   
    """
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
    """ 
    
    #get rid of invalid channels as determined by get_snr
    valid_channels=output[5]
    data_dedisp[~valid_channels] = 0
    data_dedisp = np.ma.masked_where(data_dedisp==0,data_dedisp)

    #keep only valid times as determined by get_snr
    tr = output[6]
    #data_dedisp = data_dedisp[:,:,int(tr[0])*downsample_factor:int(tr[1])*downsample_factor]
    data_dedisp = data_dedisp[:,:,tr[0]:tr[1]]

    #let's now figure out what data we want to keep
    power=np.abs(data_dedisp)**2
    I = np.nansum(power,axis=1)
    Iscr=scrunch(I,tscrunch=downsample_factor,fscrunch=1)
    try:
        nanind=np.argwhere(np.isnan(Iscr[-1,:]))[0][0]
        Iscr=Iscr[:,:nanind]
    except:
        print('No NaNs to mask')

    if time_range is not None:
        answer=time_range
    else:
        if interactive==True:
            plt.close('all')
            plt.plot(np.nanmean(Iscr,axis=0))
            plt.show()
            plt.savefig('./temp.png')
            answer=input('Please define the time bin range to keep (beginbin,endbin): ')
            answer = answer.split(',')
            answer = [int(i) for i in answer]
            
            if off==True:
                answer_off=[0,(answer[0]-10000)//downsample_factor]
        else:
            plt.close('all')
            lims=get_burst_envelope_kn(power, thres=6, pad=0, diagnostic_plots=diagnostic_plot)
            print(lims)
            answer=[(lims[0]-20000)//downsample_factor,(lims[1]+20000)//downsample_factor]
            
            if off==True:
                answer=[0,(lims[0]-10000)//downsample_factor]

    data_dedisp_masked, freqs, freq_id = fill_missing_chans(data_dedisp[:,:,int(answer[0])*downsample_factor:int(answer[1])*downsample_factor],frb_bbdata) 

    if zap_extra==True:
        #now let's try additional RFI zapping
        chan_spectrum = np.nansum( np.nansum(np.abs(data_dedisp_masked)**2,axis=1),axis=-1)
        chan_spectrum_snr = (chan_spectrum - np.nanmedian(chan_spectrum))
        chan_spectrum_snr /= (1.4826*median_abs_deviation(chan_spectrum,nan_policy='omit'))
        miss_chan_mask = np.where( (chan_spectrum_snr < -1) *(chan_spectrum > 0) )
        data_dedisp_masked[miss_chan_mask,:,:] = 0
    
        data_dedisp_masked = np.ma.masked_where(data_dedisp_masked==0,data_dedisp_masked)

    power=np.abs(data_dedisp_masked)**2
    I = np.nansum(power,axis=1)
    Iscr = scrunch(I, tscrunch=downsample_factor,fscrunch=1)
  
    #time axis
    prof = np.nanmean(I,axis=0)
    tax=np.linspace(0,len(prof),len(prof))*2.56e-3 
    prof_scr = np.nanmean(Iscr,axis=0)
    tax_scr = np.linspace(0,len(prof_scr), len(prof_scr))*2.56e-3 * downsample_factor
    
    plt.close('all')
    plt.plot(tax_scr,prof_scr,color='r')
    plt.plot(tax,prof,alpha=0.4,color='k')
    plt.xlabel('Time [ms]')
    plt.ylabel('Intensity [arb.]')
    if diagnostic_plot==None:   
        plt.show()
    else:
        if off:
            plt.savefig(diagnostic_plot+'/offburst_prof.png',format='png')
        else:   
            plt.savefig(diagnostic_plot+'/onburst_prof.png',format='png')
    
    freqs=np.flip(freqs)
    
    return data_dedisp_masked, freqs, freq_id


def extra_flag(com_vol):
    """
    com_vol is the complex voltage array [freq,pol,time]
    """
    #now let's try additional RFI zapping
    chan_spectrum = np.nansum( np.nansum(np.abs(com_vol)**2,axis=1),axis=-1)
    chan_spectrum_snr = (chan_spectrum - np.nanmedian(chan_spectrum))
    chan_spectrum_snr /= (1.4826*median_abs_deviation(chan_spectrum,nan_policy='omit'))
    miss_chan_mask = np.where( (chan_spectrum_snr < -1) *(chan_spectrum > 0) )
    com_vol[miss_chan_mask,:,:] = 0
    
    data_masked = np.ma.masked_where(com_vol==0,com_vol)
    return data_masked

def gaus(x,a,x0,sigma,c):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def scatt_tail(t, tau_scatt,t0,t1,sigma,a):
    #convolve gaussian function with a one-sided exponential
    return a*signal.convolve(gaus(t,1,t0,sigma,0),np.exp(-(t-t1)/tau_scatt),mode='same',method='direct')

def fit_n_flat(ds, t_lims, Q):
    """
    ds is [freq,time]
    t_lims is the on pulse time bin limits (tuple)
    Q is the fftsize
    """
    #reshape the array to [freq, subfreq, time]
    ds_reshape = ds.reshape(ds.shape[0]//Q,Q,ds.shape[-1])
    offdata = np.concatenate((ds[:,:t_lims[0]],ds[:,t_lims[1]:]),axis=1)
    
    #make the weights
    weights = np.zeros_like(ds)
    for chan in range(offdata.shape[0]):
        varchan=np.var(offdata[chan,:])
        if varchan!=0:
            weights[chan,:] = np.zeros(weights.shape[-1]) + 1./varchan
            
    weights = weights.reshape(ds.shape[0]//Q,Q,weights.shape[-1])   
    
    #get rid of nans
    ds_reshape[np.isnan(ds_reshape)]=0
    weights[np.isnan(weights)]=0
    
    #do the flattening
    #problem with this function is that it introduces sharp spikes in the spectrum
    a,b = DeconvolvePFB(Q=Q).flatten(x=ds_reshape,Ni=weights)
    
    
    a[np.isnan(a)]=0
    
    for i in range(a.shape[0]):
        if i==0:
            ds_flatten = a[i,:,:]
        else:
            ds_flatten = np.concatenate((ds_flatten,a[i,:,:]),axis=0)
            
    ds_flatten = np.ma.masked_where(ds_flatten==0,ds_flatten)
    
    spec_flat = np.nanmean(ds_flatten[:,t_lims[0]:t_lims[1]],axis=1)
    spec_flat=np.ma.masked_where(spec_flat==0,spec_flat)
    
    off_spec_flat = np.nanmean(np.concatenate((ds_flatten[:,0:t_lims[0]],ds_flatten[:,t_lims[1]:]),axis=1),axis=1)
    off_spec_flat=np.ma.masked_where(off_spec_flat==0,off_spec_flat)
    
    return ds_flatten, spec_flat, off_spec_flat

def fakefrb(ds_noise, fb_model, data_I):
    """
    ds_noise is complex voltages [freq,pol,time]
    fb_model is the fitburst model [freq,time]
    data_I is the dynamic spectrum containing the burst [freq,time]
    """
   

    if fb_model.shape[1] < ds_noise.shape[-1]: 
        # extra = (ds_noise.shape[-1]-(ds_noise.shape[-1]-fb_model.shape[1]))
        # ds_noise = ds_noise[:,:,0:extra]
        extra_t = ds_noise.shape[-1]-fb_model.shape[1]
        fb_model = np.concatenate((fb_model,np.zeros((fb_model.shape[0],extra_t))), axis=1)
    elif ds_noise.shape[-1] < fb_model.shape[1]:
        fb_model = fb_model[:,:ds_noise.shape[-1]]
        # fb_prof=np.nanmean(fb_model,axis=0)
        # if np.argmax(fb_prof)-ds_noise.shape[-1]//2 < 0 :
        #     beg = 0
        # else:
        #     beg = np.argmax(fb_prof)-ds_noise.shape[-1]//2
        # fb_model = fb_model[:,beg:beg+ds_noise.shape[-1]]

    # if ds_noise.shape[-1] < fb_model.shape[1]:
    #     print('Need to make sure the fitburst model has a shorter time length (or equal) to your noise')
        #ds_noise = ds_noise[:,:,:-(fb_model.shape[1]-ds_noise.shape[-1])]
       
    fake_frb_model = np.zeros_like(ds_noise)
    
    r1=np.random.normal(loc=np.nanmean(ds_noise),scale=np.nanstd(ds_noise),size=ds_noise.shape)
    r2=r1[:,1,:]
    r1=r1[:,0,:]

    r3=ds_noise[:,0,:]#np.random.normal(loc=np.nanmean(ds_noise),scale=np.nanstd(ds_noise),size=fb_model.shape)
    r4=ds_noise[:,1,:]#np.random.normal(loc=np.nanmean(ds_noise),scale=np.nanstd(ds_noise),size=fb_model.shape)
    r1=np.ma.array(r1,mask=r3.mask)
    r2=np.ma.array(r2,mask=r4.mask)

    plt.imshow(r1,aspect='auto')
    plt.show()
    plt.imshow(fb_model,aspect='auto')
    plt.show()

    p0=fb_model * r1#ds_noise[:,0,:] * fb_model
    p1=fb_model * r2#ds_noise[:,1,:] * fb_model
    real_off = np.mean(np.nanmean(data_I,axis=0))
    
    fake_frb_model[:,0,:] = (p0)*np.max(np.nanmean(data_I,axis=0)) + r2# + (ds_noise[:,0,:])
    fake_frb_model[:,1,:] = (p1)*np.max(np.nanmean(data_I,axis=0)) + r1# + (ds_noise[:,1,:])#ds_noise[:,1,:] * fb_model + (ds_noise[:,1,:])
        
    
    return fake_frb_model


def fitburst_model_to_ds(fitburst_json,downsamp=1):
    data = json.load(open(fitburst_json, "r"))
    params = data["model_parameters"]
    numtime=data['fit_statistics']['num_time']
    numfreq=data['fit_statistics']['num_freq']
    num_components=len(params["amplitude"])
    new_params = deepcopy(params)
    freqs = np.linspace(const.FREQ_TOP_MHZ, const.FREQ_BOTTOM_MHZ, num = numfreq)
    times = np.linspace(0.,numtime*downsamp*2.56e-6, num = numtime)
    model_obj = fb.analysis.model.SpectrumModeler(
                freqs,
                times,
                dm_incoherent = params["dm"][0],
                factor_freq_upsample = 1,
                factor_time_upsample = 1,
                is_dedispersed = True,
                verbose = False,
                num_components = num_components,
    )

    model_obj.update_parameters(new_params)
    model = model_obj.compute_model()
    
    return model,times


def convert_scatscin(value, scint=False, scatt=False):
    """
    scatt in ms
    scint in kHz
    """
    if scint==False and scatt==False:
        print('Please provide a scintillation bandwidth or scattering time as input')
        exit()
    if scint==True and scatt==True:
        print('Please provide either a scintillation bandwidth or scattering time as input')
        exit()
    new=1/(2*np.pi*value)
    if scint==True:
        return new
        #print('The scattering time is {} ms'.format(new))
    if scatt==True:
        #print('The scintillation bandwidth is {} kHz'.format(new))
        return new
    
def get_event_info(event_id):
    event = master.events.get_event(event_id)
    for par in event["measured_parameters"]:
        if par["pipeline"]["name"] == "realtime":
            event_date = par["datetime"].split(" ")[0].split("-")
            event_snr = par["snr"]
            event_ra = par["ra"]
            event_dec = par["dec"]
    return event_date, event_snr, event_ra, event_dec


def fit_spline(spec, num_splines=50, k=3):
    xs = np.arange(len(spec))
    xs = xs[~spec.mask]
    ts = xs[1:-1][::(len(xs)-2)//num_splines]
    ts = np.r_[(xs[0],)*(k+1),
            ts,
            (xs[-1],)*(k+1)]
    ys = spec[~spec.mask]
    
    spline = make_lsq_spline(xs, ys, ts, k=k)
    spec_smooth = spec*0.
    spec_smooth[~spec.mask] = spline(xs)
    return spec_smooth




def acf_per_subband(spec,freqs,freqids,num_subbands=2,savefig='./acf_per_freq.pdf',plot_fit=True,maxlag=None,snsubband=False,offspec=None):
  
    plt.close()
    spec[np.isnan(spec)]=0
    
    sub_cent=[]
    sub_scint=[]
    
    sub_len = len(spec)//num_subbands
    acfs=[]
    lags=[]
    fcents=[]
    sub_sn=[]
    sub_mask=[]
    spec_lens=[]
    mask = np.abs(np.array(spec.mask, dtype='bool')-1)
    tot=np.sum(spec.data*mask)
    for sub in range(num_subbands):
        if snsubband is False:
            beg = sub*sub_len
            end = (sub+1)*sub_len
            if end>(len(spec)-1):
                end=-1
            subtot=np.sum(spec.data[beg:end])
    
        else:
            if sub==0:
                beg = 0
            else:
                beg = end
            
            i=beg-1
            subtot=0
            while subtot < (tot/float(num_subbands)):
                i+=1
                subtot+=(spec.data[i]*mask[i])
            
            end = i 
    
        sub_sn.append(subtot)
        sub_mask.append(np.sum(spec.mask[beg:end]))
        if end!=-1:
            spec_lens.append(end-beg)
        else:
            spec_lens.append(len(spec)-beg)

        #temp remove
        #edges=[0,181995,297685,363285,417014,445766,471781,495622,524207]
        #end=len(spec)-1-edges[sub]
        #beg=len(spec)-1-edges[sub+1]

        print("beg,end",beg,end)
        if 5 >= maxlag:
            lagrange=maxlag
        else:
            lagrange=5
            
        if offspec is not None:
            acf = acf_scint_plot(spec[beg:end],freqids[beg:end],freqs[beg:end],[0,0],lagrange_for_fit=lagrange,diagnostic_plots=False,maxlag=maxlag,offspec_mean=np.nanmean(offspec[beg:end]))
        else:
            acf = acf_scint_plot(spec[beg:end],freqids[beg:end],freqs[beg:end],[0,0],lagrange_for_fit=lagrange,diagnostic_plots=False,maxlag=maxlag)
            
        acfs.append(acf[0])
        lags.append(acf[1])
        cmap = matplotlib.cm.get_cmap('plasma')
        rgba = cmap(sub/num_subbands)
        
        plt.plot(acf[1],acf[0]+(1*sub),drawstyle='steps-mid',color=rgba,linewidth=1,alpha=1,label='%.2f MHz'%(freqs[beg]+((freqs[end]-freqs[beg])/2)))
        fcents.append(freqs[beg]+((freqs[end]-freqs[beg])/2))
        if plot_fit==True:
            try:
                plt.plot(acf[1],lorentz(acf[1],acf[2].params['gamma'],acf[2].params['m'],acf[2].params['c']) +(1*sub),color='k',linewidth=0.5)
                sub_cent.append((freqs[beg]+((freqs[end]-freqs[beg])/2)))
                sub_scint.append(np.abs(acf[2].params['gamma']))
            except:
                sub_cent.append((freqs[beg]+((freqs[end]-freqs[beg])/2)))
                sub_scint.append(0)
    
    plt.xlim(-maxlag,maxlag)
    plt.ylim(-1,1+(sub))
    plt.xlabel('Freq lag [MHz]')
    plt.legend(loc='upper left')
    plt.savefig(savefig,format='pdf')
    
    if plot_fit==True:
        plt.close()
        plt.scatter(sub_cent,sub_scint,marker='x',color='k')
        plt.plot(freqs,sub_scint[-1]*(freqs/sub_cent[-1])**4,color='r')
        plt.xlabel('Freq [MHz]')
        plt.ylabel('Scint bw [MHz]')
        plt.savefig(savefig[:-4]+'_scintbw.pdf',format='pdf')
    
    return acfs,fcents,lags, sub_sn, sub_mask, spec_lens

def scint_freq_relation(x,c,n):
    return c*(x)**n

def make_scallop_model(off_data, fftsize, downfreq):
    """
    off_data is a complex voltage array containing off burst data, shape pol, time, freq
    fftsize and downfreq are the factors used for upchannelisation
    """
    #use off burst data to make scallop model
    noise_power = np.abs(off_data**2)
    I_noise = np.mean(noise_power,axis=0).T
    spec_noise = np.nanmean(I_noise,axis=1)
    noise_mean=np.mean(spec_noise)
    noise_std = np.std(spec_noise)
    spec_noise_norm=spec_noise-noise_mean
    spec_noise_norm=spec_noise_norm/noise_std
    inds=np.where(np.abs(spec_noise_norm) > 3)[0]
    spec_noise[inds]=0
    spec_noise_masked=np.ma.masked_where(spec_noise==0,spec_noise)
    spec_noise_masked_reshape = spec_noise_masked.reshape(len(spec_noise_masked)//(fftsize//downfreq),(fftsize//downfreq))
    model_scallop = np.nanmean(spec_noise_masked_reshape,axis=0)
    model = np.tile(model_scallop,I_noise.shape[0]//(fftsize//downfreq))
    spec_noise_masked_corr = spec_noise_masked/model
    spec_noise_masked_corr=np.ma.masked_where(spec_noise_masked_corr==0,spec_noise_masked_corr)
    return model, spec_noise_masked_corr, inds


def make_fitburst_mask(fitburst_json, fitburst_downsamp_factor, data_I):
    """
    fitburst_json is a string containing the path to the fitburst file
    fitburst_downsamp_factor is an integer downsample factor which was used to make the fitburst model (this really should be in the json file somewhere).
    data_I is the dynamic spectrum of the burst, shape [freq, time]
    """
    #read in fitburst model
    mod,times=fitburst_model_to_ds(fitburst_json,downsamp=fitburst_downsamp_factor)
    #make a time series for the model at 2.56e-6 resolution
    newts=np.linspace(0,mod.shape[1]*fitburst_downsamp_factor*2.56e-6,mod.shape[1]*fitburst_downsamp_factor)
    #interpolate the model to 2.56e-6 time resolution
    times=np.linspace(0,mod.shape[1]*fitburst_downsamp_factor*2.56e-6,mod.shape[1])
    fbmod=interp2d(times,np.linspace(0,1024,1024),mod)
    mod_2us = fbmod(newts,np.linspace(0,1024,1024))
    #add zeros to the model to make sure it has the same size as the data
    mod_2us_full = np.concatenate((mod_2us,np.zeros((data_I.shape[0],data_I.shape[1]-mod_2us.shape[1]))),axis=1)
    time_full = np.linspace(0,mod_2us_full.shape[1]*2.56e-6,mod_2us_full.shape[1]) 
    #make into profiles 
    mod_2us_full_prof = np.nanmean(mod_2us_full,axis=0)
    dataprof=np.nanmean(data_I,axis=0)
    mod_2us_full=np.roll(mod_2us_full,np.argmax(dataprof)-np.argmax(mod_2us_full_prof), axis=1)
    
    #plot
    mod_2us_full_prof = np.nanmean(mod_2us_full,axis=0)
    datatimes=np.linspace(0,data_I.shape[1]*2.56e-6,data_I.shape[1])
    
    offset = np.mean(np.nanmean(data_I,axis=0)[0:100])
    
    mod_2us_full_prof+=offset
    
    scale_mod = (mod_2us_full/np.max(mod_2us_full_prof)) * (np.max(np.nanmean(data_I,axis=0)-offset)) + offset
    
    plt.plot(datatimes,np.nanmean(data_I,axis=0))
    plt.plot(time_full,np.nanmean(scale_mod,axis=0))
    plt.show()


    return scale_mod - offset

def apply_fbmask_to_data(mod, data):
    #make a mask out of the fitburst model
    peak_model = np.max(mod)
    modmask_inds = np.where(mod > peak_model*0.01) #above 1% of the flux is kept
    mask = np.zeros_like(mod)
    mask[modmask_inds]=1

    data_mod=np.zeros_like(data)
    for pol in range(2):
        data_mod[:,pol,:] = data[:,pol,:]*mask 
        
    return data_mod