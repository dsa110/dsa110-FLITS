import h5py
import sys

sys.path.insert(0, "/arc/home/jfaber/baseband_morphologies/chime_dsa_codetections/scintillation/utilities/")
#from scint_funcs import upchannel_fast as upchan
from scint_funcs import make_scallop_model, acf_scint_plot, scrunch, acf_per_subband, weighted_avg_and_std
from scint_funcs import lorentz_withc_min, lorentz_w_c, doublelorentz_withc_min, doublelorentz_w_c,triplelorentz_min, triplelorentz, lorentz, lin, linmin
from scint_funcs import scint_freq_relation, scint_freq_relation_min
from scint_funcs import emission_size, res

#from mwprop.ne2001p import *
#from mwprop.ne2001p.NE2001 import ne2001
from astropy.coordinates import SkyCoord
from astropy import units as u

import sys 
sys.path.insert(0, "/arc/home/jfaber/baseband_morphologies/chime_dsa_codetections/scintillation/utilities/")
from kenzie_funcs import data_dedisp_derip_filled_masked, get_burst_envelope_kn, get_data
from kenzie_funcs import upchannel_fast as upchan

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

import numpy as np

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


def make_scint_input(eventname, event_id, dm, file, diagnostic_plots_direc=None, fftsize=None, downfreq=1,speclims=None, interactive=False):


    if diagnostic_plots_direc==None:
        direc=None
    else:
        direc=diagnostic_plots_direc

    if direc==None: 
        data, freq, freqid = data_dedisp_derip_filled_masked(event_id, dm, file=file, downsample_factor=32,interactive=interactive)
        data_off, freq_off, freqid_off = data_dedisp_derip_filled_masked(event_id, dm, file=file, downsample_factor=8,interactive=interactive,off=True)
    else:
        data, freq, freqid = data_dedisp_derip_filled_masked(event_id, dm, file=file, downsample_factor=32,interactive=interactive,diagnostic_plot=direc)
        data_off, freq_off, freqid_off = data_dedisp_derip_filled_masked(event_id, dm, file=file, downsample_factor=8,interactive=interactive,off=True, diagnostic_plot=direc)

    if speclims is None:
        print('*** Determining the burst extent in frequency ***')
        power=np.abs(data)**2
        spect_lim = get_spectrum_lim(freqid, power, diagnostic_plots=True)
        print(spect_lim)
        plt.show()
    else:
        spect_lim=speclims

    data = data[spect_lim[0] : spect_lim[1]]
    freq = freq[spect_lim[0] : spect_lim[1]]
    freqid = freqid[spect_lim[0] : spect_lim[1]]
    data_off = data_off[spect_lim[0] : spect_lim[1]]
    freq_off = freq_off[spect_lim[0] : spect_lim[1]]
    freqid_off = freqid_off[spect_lim[0] : spect_lim[1]]

    print('*** Determining the burst width ***')
    power=np.abs(data)**2
    #I = np.nansum(power,axis=1)
    try:
        lims=get_burst_envelope_kn(power, thres=6, pad=0, diagnostic_plots=False)
    except:
        print('Could not determine burst lims')
        if fftsize is None:
            print('please provide an fftsize as input')
            exit()
    if fftsize is None:
        possible_fftsizes=[2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384]
        deltat = (lims[1]-lims[0])
        fftsize = possible_fftsizes[np.argmin(np.abs((possible_fftsizes - deltat)))]
        downfreq = 1
        print('*** determining fftsize using the width of the burst ***')

    #upchannelise
    print('*** Upchannelising to fftsize %s, downfreq %s ***'%(fftsize,downfreq))

    if fftsize!=0:

        data_dedisp_masked_upchan_ds = upchan(data,freqid,fftsize=fftsize,downfreq=downfreq)
        #upchannel the noise to model the scalloping
        noise_dedisp_masked_upchan_ds = upchan(data_off,freqid,fftsize=fftsize,downfreq=downfreq)

        model_ds, inds = make_scallop_model(noise_dedisp_masked_upchan_ds[0], fftsize, downfreq)

    #let's plot the upchannelised burst
    power = np.abs(data_dedisp_masked_upchan_ds[0]**2)
    I_upchan_ds = np.sum(power,axis=0).T

    upchan_lims = get_main_peak_lim(I_upchan_ds,normalize_profile=True)
    if upchan_lims[0] < 0:
        upchan_lims[0]=0
    if upchan_lims[1]> I_upchan_ds.shape[1]:
        upchan_lims[1]=I_upchan_ds.shape[1]-1

    print(upchan_lims)
    plt.plot(np.nanmean(I_upchan_ds,axis=0), drawstyle='steps-mid',color='k')
    plt.axvline(upchan_lims[0])
    plt.axvline(upchan_lims[1])
    plt.savefig(direc+f'/{eventname}_upchan_onburst_prof.png',format='png')

    return I_upchan_ds, data_dedisp_masked_upchan_ds[1], data_dedisp_masked_upchan_ds[2], model_ds, inds, upchan_lims,fftsize, downfreq





def run_scint_pipe(eventname, I_upchan_ds, freqs, freq_ids, model_ds, inds, upchan_lims,fftsize,downfreq,diagnostic_plot=None,subbands=8,peak_only=False,maxlag=20,maxlag_subs=10,snsubband=False):
    """
    """

    print(maxlag_subs)
    print("*** Normalising per channel and creating burst spectra***")
    offburst_begin = 0
    offburst_end = upchan_lims[0]-1
    if offburst_end <=0:
        offburst_end=1

    print(offburst_end)

    I_upchan_corrected = np.zeros_like(I_upchan_ds)
    for time_bin in range(I_upchan_ds.shape[1]):
        I_upchan_corrected[:,time_bin] = I_upchan_ds[:,time_bin]/model_ds


    for freq_chan in range(I_upchan_corrected.shape[0]):
        Ioff=I_upchan_corrected[freq_chan,offburst_begin:offburst_end]
        I_upchan_corrected[freq_chan,:] = I_upchan_corrected[freq_chan,:] - np.nanmean(Ioff)
        Ioff-=np.nanmean(Ioff)
        I_upchan_corrected[freq_chan,:] = I_upchan_corrected[freq_chan,:] / np.nanstd(Ioff)

    #let's find any outlying RFI spikes in the off burst data and flag from our data
    if offburst_end-offburst_begin <=1:
        spec_off = I_upchan_corrected[:,offburst_begin]
    else:
        spec_off = np.nanmean(I_upchan_corrected[:,offburst_begin:offburst_end],axis=1)
    spec_off[np.isnan(spec_off)]=0
    spec_off=np.ma.masked_where(spec_off==0,spec_off)

    spec_off = spec_off.copy()
    print(spec_off)
    print(np.nanmean(spec_off))
    calib_off=spec_off-np.nanmean(spec_off)
    calib_off/=np.std(calib_off)
    newinds=np.where(np.abs(calib_off)>3)[0]

    I_upchan_corrected[inds,:]=0
    I_upchan_corrected[newinds,:]=0
    I_upchan_corrected = np.ma.masked_where(I_upchan_corrected==0,I_upchan_corrected)

    #on burst spectrum
    spec_upchan_corr=np.nanmean(I_upchan_corrected[:,upchan_lims[0]:upchan_lims[1]],axis=1)
    spec_upchan_corr[np.isnan(spec_upchan_corr)]=0
    spec_upchan_corr=np.ma.masked_where(spec_upchan_corr==0,spec_upchan_corr)

    #peak burst spectrum
    prof = np.nanmean(I_upchan_ds,axis=0)
    peak=np.argmax(prof)
    spec_peak_upchan_corr=np.nanmean(I_upchan_corrected[:,peak:peak+1],axis=1)
    spec_peak_upchan_corr[np.isnan(spec_peak_upchan_corr)]=0
    spec_peak_upchan_corr=np.ma.masked_where(spec_peak_upchan_corr==0,spec_peak_upchan_corr)

    #off burst spectrum
    spec_fake_upchan_corr = np.nanmean(I_upchan_corrected[:,0:upchan_lims[0]],axis=1)
    spec_fake_upchan_corr[np.isnan(spec_fake_upchan_corr)]=0
    spec_fake_upchan_corr=np.ma.masked_where(spec_fake_upchan_corr==0,spec_fake_upchan_corr)

    if diagnostic_plot:
        fig,ax=plt.subplots(2,1,sharex=True)
        ax[0].plot(freqs,spec_upchan_corr,color='k',alpha=0.5,label='on')
        ax[0].plot(freqs,spec_fake_upchan_corr,color='k',label='off')
        ax[1].plot(freqs,spec_peak_upchan_corr,color='k',alpha=0.5,label='peak')
        ax[1].plot(freqs,spec_fake_upchan_corr,color='k',label='off')
        ax[0].legend()
        ax[1].legend()
        ax[1].set_xlabel('Freq [MHz]')
        ax[0].set_ylabel('Intensity')
        ax[1].set_ylabel('Intensity')
        plt.savefig(diagnostic_plot+f'/{eventname}_upchan_spec.png',format='png')


    if peak_only == False:
        acf_res = acf_scint_plot(spec_upchan_corr,freq_ids,freqs,[0,0],lagrange_for_fit=0.01,diagnostic_plots=False,maxlag=maxlag,offspec_mean=(np.nanmean(spec_fake_upchan_corr)))

        if diagnostic_plot:
            fig,ax=plt.subplots(3,1)
            ax[0].plot(acf_res[1],acf_res[0],color='orange',label='on')
            ax[1].plot(acf_res[1],acf_res[0],color='orange')
            ax[1].set_xlim(-3,3)
            ax[2].plot(acf_res[1],acf_res[0],color='orange')
            ax[2].set_xlim(-0.3,0.3)
            ax[1].axvline(-0.39,color='k',linestyle='--')
            ax[1].axvline(0.39,color='k',linestyle='--')
            ax[1].set_ylabel('ACF')
            ax[0].set_ylabel('ACF')
            ax[2].set_ylabel('ACF')
            ax[2].set_xlabel('Freq [MHz]')
            plt.savefig(diagnostic_plot+f'/{eventname}_ACF_onburst_fftsize%s_downfreq%s.png'%(fftsize,downfreq),format='png')
    else:
        acf_res = [0]


    acf_peak_res = acf_scint_plot(spec_peak_upchan_corr,freq_ids,freqs,[0,0],lagrange_for_fit=0.01,diagnostic_plots=False,maxlag=maxlag,offspec_mean=(np.nanmean(spec_fake_upchan_corr)))

    if diagnostic_plot:
        fig,ax=plt.subplots(3,1)
        ax[0].plot(acf_peak_res[1],acf_peak_res[0],color='orange',label='peak')
        ax[1].plot(acf_peak_res[1],acf_peak_res[0],color='orange')
        ax[1].set_xlim(-3,3)
        ax[2].plot(acf_peak_res[1],acf_peak_res[0],color='orange')
        ax[2].set_xlim(-0.3,0.3)
        ax[1].axvline(-0.39,color='k',linestyle='--')
        ax[1].axvline(0.39,color='k',linestyle='--')
        ax[1].set_ylabel('ACF')
        ax[0].set_ylabel('ACF')
        ax[2].set_ylabel('ACF')
        ax[2].set_xlabel('Freq [MHz]')

        plt.savefig(diagnostic_plot+f'/{eventname}_ACF_peakburst_fftsize%s_downfreq%s.png'%(fftsize,downfreq),format='png') 
    
    acf_peak_res=[0,1]

    if peak_only == False:
        acf_subs, fcents_subs, lags_subs, submask, spec_lens = multi_sub_acf(eventname, spec_upchan_corr,freqs,freq_ids,spec_fake_upchan_corr,fftsize,downfreq,diagnostic_plot=diagnostic_plot,numsubs=subbands,maxlag=maxlag_subs, snsubband=snsubband)
    else:
        submask=[]
        spec_lens=[]
        acf_subs=[]

    acf_peak_subs, fcents_subs, lags_subs, submask_peak, spec_lens_peak = multi_sub_acf(eventname, spec_peak_upchan_corr,freqs,freq_ids,spec_fake_upchan_corr,fftsize,downfreq,diagnostic_plot=diagnostic_plot,numsubs=subbands,filename_add = 'peak',maxlag=maxlag_subs,snsubband=snsubband)
    
    if diagnostic_plot:
        eid=diagnostic_plot.split('_')[-1].split('/')[0]
        np.savez(diagnostic_plot+f'/{eventname}_acf_%s_fftsize%s_downfreq%s.npz'%(eid,fftsize,downfreq), onburstacf=acf_res[0], peakburstacf=acf_peak_res[0], freq_lags = acf_peak_res[1],
                    sub_acfs = acf_subs, sub_acfs_peak = acf_peak_subs, sub_fcents = fcents_subs, sub_lags = lags_subs)
    else:
        eid=diagnostic_plot.split('_')[-1].split('/')[0]
        np.savez(f'/arc/home/jfaber/baseband_morphologies/chime_dsa_codetections/scintillation/{eventname}_acf_%s_fftsize%s_downfreq%s.npz'%(eid,fftsize,downfreq), onburstacf=acf_res[0], peakburstacf=acf_peak_res[0], freq_lags = acf_peak_res[1],
                    sub_acfs = acf_subs, sub_fcents = fcents_subs, sub_lags = lags_subs)

    #fit_subband_acfs(acf_subs, fcents_subs, lags_subs, fftsize, downfreq, submask, freq, offset=1,diagnostic_plot=diagnostic_plot, numsubs=subbands,)

    return acf_res[0], acf_peak_res[0], acf_peak_res[1], acf_subs, acf_peak_subs, fcents_subs, lags_subs, submask, submask_peak, spec_lens, spec_lens_peak


def multi_sub_acf(eventname, spec_upchan_corr,freqs,freq_ids,spec_fake_upchan_corr,fftsize,downfreq,diagnostic_plot=None, numsubs=8, filename_add=None, maxlag=10,snsubband=False):
    """

    """
    
    acfs,fcents,lags, subsn, submask,spec_lens=acf_per_subband(spec_upchan_corr,freqs,freq_ids,num_subbands=numsubs,plot_fit=True,maxlag=maxlag,offspec=spec_fake_upchan_corr,snsubband=snsubband)

    cmap = matplotlib.cm.get_cmap('plasma')
    if diagnostic_plot:
        plt.close('all')
        for i in range(len(fcents)):
            rgba = cmap(i/len(fcents))
            plt.plot(lags[len(fcents)-i-1],acfs[len(fcents)-i-1]+(1.0*i),drawstyle='steps-mid',color=rgba,linewidth=2,alpha=0.7,label='%.2f MHz'%fcents[len(fcents)-i-1])
            plt.xlim(-0.3,0.3)

        if filename_add:
            plt.savefig(diagnostic_plot+f'/{eventname}_ACF_%s_per_subband_0.3MHz_fftsize%s_downfreq%s.png'%(filename_add,fftsize,downfreq), format='png')
        else:
            plt.savefig(diagnostic_plot+f'/{eventname}_ACF_per_subband_0.3MHz_fftsize%s_downfreq%s.png'%(fftsize,downfreq), format='png')

        plt.close('all')
        for i in range(len(fcents)):
            rgba = cmap(i/len(fcents))
            plt.plot(lags[len(fcents)-i-1],acfs[len(fcents)-i-1]+(1.0*i),drawstyle='steps-mid',color=rgba,linewidth=2,alpha=0.7,label='%.2f MHz'%fcents[len(fcents)-i-1])
            plt.xlim(-1.5,1.5)

        if filename_add:
            plt.savefig(diagnostic_plot+f'/{eventname}_ACF_%s_per_subband_1.5MHz_fftsize%s_downfreq%s.png'%(filename_add,fftsize,downfreq), format='png')
        else:
            plt.savefig(diagnostic_plot+f'/{eventname}_ACF_per_subband_1.5MHz_fftsize%s_downfreq%s.png'%(fftsize,downfreq), format='png')

    return acfs, fcents, lags, submask, spec_lens


def ne2001_scat(event_id):
    """
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


def fit_subband_acfs(eventname, acfs, fcents, lags, fftsize, downfreq,submask,spec_lens,freqs,numsubs=8, numlorentz=1, lagrange_for_fits=None,offset=1,diagnostic_plot=None, xlim=0.5):
    """
    """
    fig = plt.figure(figsize=(5,7))

    cmap = matplotlib.cm.get_cmap('plasma')
    if lagrange_for_fits is None:
        lagrange_for_fits=[1]*numsubs

    f_res=0.39101/(fftsize//downfreq)

    sub_scint=[]
    sub_scint_uncert=[]
    if numlorentz==2:
        sub_scint2=[]
        sub_scint_uncert2=[]
    f_cents=[]

    if numlorentz > 2:
        print("Cannot yet fit more than 2 lorentzians, change the numlorentz to 1 or 2")
        return

    plt.xlabel('Frequency lag [MHz]')
    plt.ylabel('Subband frequency [MHz]')
    acfs_offset=[]
    mods1=[]
    mods1_uncert=[]
    if numlorentz==2:
        mods2=[]
        mods2_uncert=[]


    for i in range(len(fcents)):
        
        # Offsetting sub-banded ACFs in vertical plot
        plot_offset = (float(offset)*i)
        plot_amp = 10 #increase ACF amp to improve visualization
        
        lagrange_for_fit=lagrange_for_fits[i]
        rgba = cmap(i/len(fcents))
        acf=acfs[len(fcents)-i-1]
        lag=lags[len(fcents)-i-1]
        acf_fit=acf[int(len(acf)/2.)-int(lagrange_for_fit/f_res):int(len(acf)/2.)+int(lagrange_for_fit/f_res)]
        lag_fit=lag[int(len(acf)/2.)-int(lagrange_for_fit/f_res):int(len(acf)/2.)+int(lagrange_for_fit/f_res)]
        
        plt.plot(lags[len(fcents)-i-1],acfs[len(fcents)-i-1]*plot_amp+plot_offset,drawstyle='steps-mid',color=rgba,linewidth=2,alpha=0.7,label='%.2f MHz'%fcents[len(fcents)-i-1])

        #compute the ACF errors
        acf_half = acf[len(acf)//2:]
        var_f = np.ones(len(acf_half)) / (len(acf_half))
        var_f[1:] *= 1 + 2 * np.cumsum(acf_half[0:-1] ** 2)
        f_errors = np.sqrt(var_f)

        f_errors_full = np.concatenate((f_errors[::-1],f_errors))
        acf_fit_errors=f_errors_full[int(len(acf)/2.)-int(lagrange_for_fit/f_res):int(len(acf)/2.)+int(lagrange_for_fit/f_res)]


        fit=True
        try:
            params = Parameters()
            params.add('gamma1', value = 1, min= 0.00001, max = 100)
            params.add('m1', value = 1, min = -100, max = 100)
            if numlorentz==2:
                params.add('gamma2', value = 0.2, min= 0.00001, max = 100)
                params.add('m2', value = 1, min = -100, max = 100)
            params.add('c', value = 0, min = -100, max = 100)

            if numlorentz==1:
                fit_min = Minimizer(lorentz_withc_min, params, fcn_args=(lag_fit,acf_fit,np.sqrt(acf_fit_errors)))
            elif numlorentz==2:
                fit_min = Minimizer(doublelorentz_withc_min, params, fcn_args=(lag_fit,acf_fit,np.sqrt(acf_fit_errors)))
        
            result_subacf = fit_min.minimize()
            print(report_fit(result_subacf))
        except:
            print('could not fit')
            fit=False


        if fit==True:
            if numlorentz==1:
                bestfit_lorentz_w_c = lorentz_w_c(lags[len(fcents)-i-1],result_subacf.params['gamma1'],result_subacf.params['m1'],result_subacf.params['c'])
                plt.plot(lags[len(fcents)-i-1],bestfit_lorentz_w_c*plot_amp+plot_offset,color='k',linewidth=1)
                sub_scint.append(result_subacf.params['gamma1'])
                mods1.append(result_subacf.params['m1'])
                #try:
                sub_scint_uncert.append(np.abs(result_subacf.params['gamma1'].stderr))
                #except:
                #    sub_scint_uncert.append(0.)
                mods1_uncert.append(np.abs(result_subacf.params['m1'].stderr))

            elif numlorentz==2:
                bestfit_doublelorentz_w_c = doublelorentz_w_c(lags[len(fcents)-i-1],result_subacf.params['gamma1'],result_subacf.params['m1'],result_subacf.params['gamma2'],result_subacf.params['m2'],result_subacf.params['c'])
                plt.plot(lags[len(fcents)-i-1],bestfit_doublelorentz_w_c*plot_amp+plot_offset,color='k',linewidth=1)
                
                try:
                    scints=np.array([np.abs(result_subacf.params['gamma1']),np.abs(result_subacf.params['gamma2'])])
                    modinds=np.array([np.abs(result_subacf.params['m1']),np.abs(result_subacf.params['m2'])])
                    inds=np.argsort(scints)
                    scints=scints[inds]
                    modinds=modinds[inds]
                    errs = np.array([np.abs(result_subacf.params['gamma1'].stderr),np.abs(result_subacf.params['gamma2'].stderr)])
                    sub_scint.append(scints[0])
                    sub_scint2.append(scints[1])
                    mods1.append(modinds[0])
                    mods2.append(modinds[1])
                    errs=errs[inds]
                    sub_scint_uncert.append(errs[0])
                    sub_scint_uncert2.append(errs[1])
                    moderrs = np.array([np.abs(result_subacf.params['m1'].stderr),np.abs(result_subacf.params['m2'].stderr)])
                    moderrs=moderrs[inds]
                    mods1_uncert.append(moderrs[0])
                    mods2_uncert.append(moderrs[1])
                except:
                    scints=np.array([np.nan, np.nan])
                    modinds=np.array([np.nan, np.nan])
                    sub_scint.append(np.nan)
                    sub_scint2.append(np.nan)
                    mods1.append(np.nan)
                    mods2.append(np.nan)
                    sub_scint_uncert.append(np.nan)
                    sub_scint_uncert2.append(np.nan)
                    mods1_uncert.append(np.nan)
                    mods2_uncert.append(np.nan)
                    
        else:
            sub_scint.append(0)
            sub_scint_uncert.append(0)
            if numlorentz==2:
                sub_scint2.append(0)
                sub_scint_uncert2.append(0) 
        
        acfs_offset.append(acfs[len(fcents)-i-1]+(float(offset)*i))
        
        f_cents.append(fcents[len(fcents)-i-1])

        #plt.axvline(lagrange_for_fit,ymin=0+(0.125*i),ymax=0+(0.125*i)+0.125,color='green',linestyle='--',alpha=0.5)
        #plt.axvline(-1*lagrange_for_fit,ymin=0+(0.125*i),ymax=0+(0.125*i)+0.125,color='green',linestyle='--', alpha=0.5)
    
    ticks = [x[0] for x in acfs_offset]
    tickvals = ['%.1f'%x for x in np.flip(fcents)]
    plt.yticks(ticks,tickvals)
    plt.ylim(ticks[0]-1, ticks[-1]+1)
    plt.xlim(-1*xlim, xlim)
    fig.tight_layout(pad=1)
    if diagnostic_plot:
        plt.show()
        plt.savefig(diagnostic_plot+f'/{eventname}_subband_fits_fftsize%s_downfreq%s.png'%(fftsize,downfreq),format='png')
    else:
        plt.show()

    # #additional uncertainty from the low number of scintles (1/sqrt(N) where N is the approximate number of scintles)
    if numlorentz==2:
        good_chans=np.array(spec_lens)-np.array(submask)
        N2 = 1 + 0.2*((np.flip(good_chans)*f_res) / sub_scint2)
        add_un2 = sub_scint2/(2*np.sqrt(N2))

    good_chans=np.array(spec_lens)-np.array(submask)
    N = 1 + 0.2*((np.flip(good_chans)*f_res) / sub_scint)
    add_un = sub_scint/(2*np.sqrt(N))

    params = Parameters()
    params.add('n', value = 2, min = 0, max = 10)
    params.add('c', value = 0.1) 
    #fit the scint bws
    # check for nans
    sub_scint_uncert = np.array(sub_scint_uncert)
    sub_scint_uncert[np.isnan(sub_scint_uncert)] = 0
    fit_min = Minimizer(scint_freq_relation_min, params, fcn_args=(f_cents,sub_scint,np.sqrt(sub_scint_uncert**2 + add_un**2)))
    result_scint1 = fit_min.minimize()
    print(report_fit(result_scint1))

    if numlorentz==2:
        sub_scint_uncert2 = np.array(sub_scint_uncert2)
        sub_scint_uncert2[np.isnan(sub_scint_uncert2)] = 0.
        fit_min = Minimizer(scint_freq_relation_min, params, fcn_args=(f_cents,sub_scint2,np.sqrt(sub_scint_uncert2**2 + add_un2**2)))
        result_scint2 = fit_min.minimize()
        print(report_fit(result_scint2))

    plt.close('all')
    plt.errorbar(f_cents,1000*np.array(sub_scint),yerr=1000*np.sqrt(np.array(sub_scint_uncert)**2 + add_un**2),fmt='o',color='k',markersize=3)
    onescint=1000*scint_freq_relation(freqs,result_scint1.params['c'].value,result_scint1.params['n'].value)[np.argmin(np.abs(freqs-600))]
    plt.plot(freqs,1000*scint_freq_relation(freqs,result_scint1.params['c'].value,result_scint1.params['n'].value),color='r',label=r'$\nu^{%.1f \pm %.1f}$'%(result_scint1.params['n'].value,result_scint1.params['n'].stderr))
    plt.axhline(f_res*1000, label='freq res')
    plt.axhline(onescint,color='green',label='%.2f kHz'%onescint)
    print(f_res)
    
    plt.xlabel('Freq [MHz]')
    plt.ylabel('Scint BW [kHz]')
    plt.legend(fontsize='large')
    if diagnostic_plot:
        plt.show()
        plt.savefig(diagnostic_plot+f'/{eventname}_scintbw_vs_freq_fftsize%s_downfreq%s.png'%(fftsize,downfreq),format='png')
    else:
        plt.show()

    plt.close('all')
    plt.errorbar(f_cents,np.array(mods1),yerr=np.array(mods1_uncert),fmt='o',color='k',markersize=3)
    plt.axhline(1,linestyle='--',color='k',alpha=0.5)
    plt.xlabel('Freq [MHz]')
    plt.ylabel('Modulation index')
    plt.ylim(0,3)
    if diagnostic_plot:
        plt.show()
        plt.savefig(diagnostic_plot+f'/{eventname}_modind_vs_freq_fftsize%s_downfreq%s.png'%(fftsize,downfreq),format='png')
    else:
        plt.show()
    
    return f_cents, 1000*np.array(sub_scint), 1000*np.sqrt(np.array(sub_scint_uncert)**2 + add_un**2), onescint