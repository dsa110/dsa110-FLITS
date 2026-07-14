from scipy.fft import fft, fftshift
import numpy as np 
from lmfit import minimize, Parameters, fit_report, Model
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import math
import scipy.constants as cons


def upchannel_nopol(wfall, freq_id, fftsize=16, downfreq=1):
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
    #wfall = np.swapaxes(wfall, 0, 1)
    #wfall = np.swapaxes(wfall, 1, 2)

    # set downtime to 1 => no averaging over complex numbers!!!
    downtime = 1

    nsamp, nchan = wfall.shape

    # upchannelization factor (16 by default)
    upchan = fftsize // downfreq

    # number of blocks
    nblock = nsamp // (fftsize * downtime)

    # initialise array for spectrum
    spec = np.zeros((nblock, nchan * upchan), dtype=np.complex64)

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
                ts = wfall[bi * fftsize : bi * fftsize + fftsize, chidx].copy()
                # perform a FFT
                ft = fftshift(fft(ts))
                # downsample in frequency
                ft = ft.reshape(upchan, downfreq).mean(axis=1).copy()

                spec[bi, chidx * upchan : chidx * upchan + upchan] = ft

                chan_id_upchan[chidx * upchan : chidx * upchan + upchan] = np.arange(
                    upchan * freq_id[chidx], upchan * freq_id[chidx] + upchan, 1
                )

                count += 1

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
    #wfall = np.swapaxes(wfall, 0, 1)
    #wfall = np.swapaxes(wfall, 1, 2)

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

def scrunch(wfall, tscrunch, fscrunch):
    """Return a rebinned array, useful for waterfall plots.

    Parameters
    ----------
    wfall: ndarray (2D)
        Array to be rebinned
    tscrunch: int
        scrunching factor along first (e.g. time) axis
    fscrunch: int
        scrunching factor along second (e.g. frequency) axis

    Returns
    -------
    rebinned_array: array

    """
    nbins = wfall.shape[-1]
    remainder = nbins % tscrunch
    wfall = wfall[..., : nbins - remainder]
    wfall = np.nanmean(
        wfall.reshape(wfall.shape[:-1] + (nbins // tscrunch, tscrunch)), axis=-1
    )
    nchan = wfall.shape[0]
    if nchan % fscrunch != 0:
        raise ValueError("Number of channel not an integer factor of fscrunch.")
    wfall = np.nanmean(
        wfall.reshape((nchan // fscrunch, fscrunch) + wfall.shape[1:]), axis=1
    )
    return wfall


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

def scint_freq_relation(v,c,n):
    return c*(v**n)

def scint_freq_relation_min(params,x,y,err):
    c = params['c'].value
    n = params['n'].value
        
    modulo=c*(x**n)
    return (modulo-y)/err

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they 
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))


def lin(x,grad,c):
    return grad*x + c

def linmin(params, x, y, errs):
    grad = params['grad'].value
    c = params['c'].value

    modelo = grad*x + c
    return (modelo - y)/errs

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
