import numpy as np
from scipy.ndimage import gaussian_filter1d
import glob
import os

def check_file(file_path):
    try:
        data = np.load(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
        
    data = np.nan_to_num(data)
    n_ch, n_t = data.shape
    
    # 1. Bandpass correction (robust)
    q_t = n_t // 8
    off_pulse = np.r_[0:q_t, -q_t:0]
    mu = np.mean(data[:, off_pulse], axis=1, keepdims=True)
    sig = np.std(data[:, off_pulse], axis=1, keepdims=True)
    sig[sig < 1e-9] = 1.0
    data = (data - mu) / sig
    
    # 2. Center burst
    prof = np.sum(data, axis=0)
    prof_smooth = gaussian_filter1d(prof, sigma=10)
    burst_idx = np.argmax(prof_smooth)
    data = np.roll(data, n_t // 2 - burst_idx, axis=1)
    
    # 3. Check centroid of tail
    # Lower frequency should have a later centroid
    lo_prof = np.sum(data[:n_ch//4, :], axis=0)
    hi_prof = np.sum(data[-n_ch//4:, :], axis=0)
    
    def get_centroid(p):
        p = p - np.percentile(p, 10)
        p = np.maximum(p, 0)
        p_norm = p / (np.sum(p) + 1e-12)
        t = np.arange(len(p))
        return np.sum(t * p_norm)
    
    c_lo = get_centroid(lo_prof)
    c_hi = get_centroid(hi_prof)
    
    # Low frequency arrives later, so larger centroid
    if c_lo > c_hi:
        res = "Ascending (Low @ 0)"
    else:
        res = "Descending (High @ 0)"
    
    print(f"{os.path.basename(file_path):50s} | C_LO: {c_lo:7.1f} | C_HI: {c_hi:7.1f} | {res}")
    return res

print(f"{'Filename':50s} | {'C_LO':7s} | {'C_HI':7s} | Result")
print("-" * 80)

chime_files = glob.glob("data/chime/*.npy")
for f in sorted(chime_files):
    check_file(f)

dsa_files = glob.glob("data/dsa/*.npy")
for f in sorted(dsa_files):
    check_file(f)
