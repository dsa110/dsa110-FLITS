from tqdm import tqdm
import numpy as np

def bandpasscorr(initrow):
    """
    effective bandpass correction (don't need for scintillation analysis)
    """
    row = (initrow - np.mean(initrow)) / np.std(initrow)
    return row

def bandpasscorr_channel(arr):
    arr = [bandpasscorr(row) for row in arr]
    arr = np.asarray(arr)
    arr = np.nan_to_num(arr)
    return arr

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

def autocorr(x, v=None,zerolag=True,maxlag=None):
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

def doublelorentz(x,gamma1,m1,gamma2,m2,c):
    return m1**2 / (1+(x/gamma1)**2) + m2**2 / (1+(x/gamma2)**2) + c
 
def lorentz(x,gamma1,m1,c):
    return m1**2 / (1+(x/gamma1)**2) +c