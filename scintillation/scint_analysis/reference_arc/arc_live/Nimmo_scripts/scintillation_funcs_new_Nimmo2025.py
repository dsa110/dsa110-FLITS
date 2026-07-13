from scipy.fft import fft, fftshift
import numpy as np 
from lmfit import minimize, Parameters, fit_report, Model
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import math
import scipy.constants as cons
import os
from scipy.stats import median_abs_deviation

from baseband_analysis.core.bbdata import BBData
from baseband_analysis.analysis.snr import get_snr, get_profile
from baseband_analysis.core.signal import get_main_peak_lim, tiedbeam_baseband_to_power
from baseband_analysis.core.dedispersion import coherent_dedisp, incoherent_dedisp
from baseband_analysis.core.sampling import scrunch

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
    fftsize : intx
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

def make_scallop_model(off_data, fftsize, downfreq):
    """
    off_data is a complex voltage array containing off burst data, shape: pol, time, freq
    fftsize and downfreq are the factors used for upchannelisation

    Returns: the scallop model and the indices of the off burst data with high spikes for flagging later
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
    return model, inds


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
        gmodel = Model(lorentz_w_c)
        acf_for_fit = acf[int(len(acf)/2.)-int(lagrange_for_fit/f_res):int(len(acf)/2.)+int(lagrange_for_fit/f_res)]
        lags_for_fit = lags[int(len(acf)/2.)-int(lagrange_for_fit/f_res):int(len(acf)/2.)+int(lagrange_for_fit/f_res)]
        result = gmodel.fit(acf_for_fit, x=lags_for_fit, gamma=0.001, m=1, c=0)
        if diagnostic_plots == True:
            plt.plot(lags,lorentz_w_c(lags,result.params['gamma'],result.params['m'],result.params['c']),color='orange',label='scint bw = %.2f MHz'%result.params['gamma'].value)
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
    
    print("Offspec mean: ", offspec_mean)
    
    x = np.copy(spec)
    xmean=np.nanmean(x[v!=0])

    if offspec_mean is None:
        denom = xmean**2
    else:
        denom = (xmean - offspec_mean)**2
       

    x[v!=0] -= xmean#x[v!=0].mean()
    if maxlag==None:
        ACF = np.zeros_like(x)
    else:
        ACF = np.zeros_like(x)[:int(maxlag)]

    for i in tqdm(range(len(ACF))):
        if zerolag == False:
                if i>1:
                        m = shift(v,0,nchan)*shift(v,i,nchan)
                        ACF[i-1] = np.nansum(shift(x,0,nchan)*shift(x, i,nchan)*m) / (np.sum(m)*denom)
        else:
                m = shift(v,0,nchan)*shift(v,i,nchan)
                ACF[i] = np.nansum(shift(x,0,nchan)*shift(x, i,nchan)*m) / (np.sum(m)*denom)

    return ACF


def doublelorentz_w_c(x,gamma1,m1,gamma2,m2,c):
        return m1**2 / (1+(x/gamma1)**2) + m2**2 / (1+(x/gamma2)**2) +c
    
def lorentz_w_c(x,gamma1,m1,c):
        return m1**2 / (1+(x/gamma1)**2) + c
    
def triplelorentz(x,gamma1,m1,gamma2,m2,gamma3,m3):
        return m1**2 / (1+(x/gamma1)**2) + m2**2 / (1+(x/gamma2)**2) + m3**2 / (1+(x/gamma3)**2) 
    
def lorentz(x,gamma1,m1):
        return m1**2 / (1+(x/gamma1)**2)
    
def lorentz_withc_min(params,x,y,err):
        gamma1 = params['gamma1'].value
        m1 = params['m1'].value
        c = params['c'].value
        
        modulo= m1**2 / (1+(x/gamma1)**2) +c
        return (modulo-y)/err
    
def doublelorentz_withc_min(params,x,y,err):
        gamma1 = params['gamma1'].value
        m1 = params['m1'].value
        gamma2 = params['gamma2'].value
        m2 = params['m2'].value
        c = params['c'].value
        
        modulo= m1**2 / (1+(x/gamma1)**2) + m2**2 / (1+(x/gamma2)**2) +c
        return (modulo-y)/err
    
    
def triplelorentz_min(params,x,y,err):
        gamma1 = params['gamma1'].value
        m1 = params['m1'].value
        gamma2 = params['gamma2'].value
        m2 = params['m2'].value
        gamma3 = params['gamma3'].value
        m3 = params['m3'].value
        
        modulo= m1**2 / (1+(x/gamma1)**2) + m2**2 / (1+(x/gamma2)**2) + m3**2 / (1+(x/gamma3)**2)
        return (modulo-y)/err 



def acf_per_subband(spec,freqs,freqids,num_subbands=2,savefig='./acf_per_freq.pdf',plot_fit=True,maxlag=None,snsubband=False,offspec=None):
  
    plt.close()
    
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
    if savefig!=None: 
        plt.xlim(-maxlag,maxlag)
        plt.ylim(-1,1+(sub))
        plt.xlabel('Freq lag [MHz]')
        plt.legend(loc='upper left')
        plt.savefig(savefig,format='pdf')
    else:
        plt.close('all')
    
    if plot_fit==True and savefig!=None:
        plt.close()
        plt.scatter(sub_cent,sub_scint,marker='x',color='k')
        plt.plot(freqs,sub_scint[-1]*(freqs/sub_cent[-1])**4,color='r')
        plt.xlabel('Freq [MHz]')
        plt.ylabel('Scint bw [MHz]')
        plt.savefig(savefig[:-4]+'_scintbw.pdf',format='pdf')
        
    # Ensure all ACFs (and corresponding lags) have the same length
    acf_lengths = [len(acf) for acf in acfs]
    min_len = min(acf_lengths)
    f_res = np.abs(freqs[1]-freqs[0])
    
    if len(set(acf_lengths)) > 1:
        print(f"Warning: Center-trimming ACFs to minimum common length {min_len//2*f_res} MHz")
        acfs_trimmed = []
        lags_trimmed = []
        for acf, lag in zip(acfs, lags):
            diff = len(acf) - min_len
            start = diff // 2
            end = start + min_len
            acfs_trimmed.append(acf[start:end])
            lags_trimmed.append(lag[start:end])
        acfs = acfs_trimmed
        lags = lags_trimmed
    
    return acfs,fcents,lags, sub_sn, sub_mask, spec_lens

def scint_freq_relation(v,c,n):
    return c*(v**n)

def scint_freq_relation_min(params,x,y,err):
    c = params['c'].value
    n = params['n'].value
        
    modulo=c*(x**n)
    return (modulo-y)/err


def res(lens_dist,lda,scat_lens):
    """
    Give lens distance, lens_dist, between source and lens in kpc
    Give wavelength of observations, lda, in m
    Give scattering timescale imparted by the screen, scat_lens, in ms

    Returns: physical resolution of lens in km
    """

    lens_dist_m = lens_dist * cons.parsec * 1000
    scat_lens_s = scat_lens / 1000.

    #previously had a 2* factor in here which I think was wrong
    return ((lda/np.pi) * np.sqrt(lens_dist_m/(4*cons.c * scat_lens_s))) / 1000

def emission_size(phys_res,mod_ind):
    """
    physical resolution of the lens in km
    modulation index mod_ind (you can measure this using the ACF or the standard dev of the spectra divided by the mean).

    returns: physical emission size in km
    
    """
    sigma = np.sqrt((1/(float(mod_ind)**2) - 1)/4.)
    return sigma * phys_res

def data_dedisp_derip_filled_masked(event_id, dm, downsample_factor=32, interactive=False, off=False, file=None,zap_extra=True,diagnostic_plot=None, time_range=None):
    """
    given an event_id and dm, 
    output the data ( RFI zapped, missing channels filled, derippled, coherent dedispersed), the frequencies and frequency channel IDs
    
    """
    import chime_frb_api
    #read into bbdata object
    master = chime_frb_api.frb_master.FRBMaster(base_url = "https://frb.chimenet.ca/frb-master")
    master.API.authorize()
    auth = {"Authorization": master.API.access_token} 
    event = master.events.get_event(event_id)
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
    
    #get rid of invalid channels as determined by get_snr
    valid_channels=output[5]
    data_dedisp[~valid_channels] = 0
    data_dedisp = np.ma.masked_where(data_dedisp==0,data_dedisp)

    #keep only valid times as determined by get_snr
    tr = output[6]
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
        else:
            plt.close('all')
            lims=get_burst_envelope_kn(power, thres=6, pad=0, diagnostic_plots=diagnostic_plot)
            if lims[0]==lims[1]:
                lims=get_burst_envelope_kn(power, thres=6, pad=0, diagnostic_plots=diagnostic_plot, downsample_factor=128)
                lims = np.array(lims)*128
            print(lims)
            answer=[(lims[0]-20000)//downsample_factor,(lims[1]+20000)//downsample_factor]
        if off==True:
            answer=[0,(lims[0]-5000)//downsample_factor]

    plt.plot(np.nanmean(Iscr,axis=0))
    plt.axvline(answer[0])
    plt.axvline(answer[1])
    plt.show()

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

def get_burst_envelope(
        power: tuple, thres: float = 5, pad: float = 0.0, diagnostic_plots: bool = False, downsample_factor: int = 1):

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
    if downsample_factor!=1:
        I = np.nansum(power,axis=1)
        Iscr=scrunch(I,tscrunch=downsample_factor,fscrunch=1)
        prof = np.nanmean(Iscr,axis=0)

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
    if diagnostic_plots is not None:

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
        plt.savefig(os.path.join('./', plot_name))
        plt.close("all")

    # Return the burst limits
    return lims



def get_burst_envelope_kn(
        power: tuple, thres: float = 5, pad: float = 0.0, diagnostic_plots: bool = False, downsample_factor: int = 1):

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
    if downsample_factor!=1:
        I = np.nansum(power,axis=1)
        Iscr=scrunch(I,tscrunch=downsample_factor,fscrunch=1)
        prof = np.nanmean(Iscr,axis=0)

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
    if diagnostic_plots is not None:

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
        plt.savefig(os.path.join('./', plot_name))
        plt.close("all")

    # Return the burst limits
    return lims

def get_data(event):
    for par in event["measured_parameters"]:
        if par["pipeline"]["name"] == "realtime":
            event_date = par["datetime"].split(" ")[0].split("-")
            
        data_path = "/arc/projects/chime_frb/data/chime/baseband/processed/" + \
        event_date[0] + "/" + \
        event_date[1] + "/" + \
        event_date[2] + "/astro_" + \
        str(event["id"]) + "/singlebeam_" + str(event["id"]) +".h5"
         
    return data_path


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
