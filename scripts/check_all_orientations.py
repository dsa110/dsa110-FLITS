import numpy as np
from scipy.ndimage import gaussian_filter1d

def check_file(file_path):
    data = np.load(file_path)
    n_ch, n_t = data.shape
    
    # 1. Bandpass correction
    q_t = n_t // 4
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
    
    # 3. Check width of bottom vs top
    lo_prof = np.sum(data[:n_ch//4, :], axis=0)
    hi_prof = np.sum(data[-n_ch//4:, :], axis=0)
    
    def get_width(p):
        p = p - np.percentile(p, 10)
        p = np.maximum(p, 0)
        p /= np.sum(p)
        t = np.arange(len(p))
        mu = np.sum(t * p)
        var = np.sum((t - mu)**2 * p)
        return np.sqrt(var)
    
    w_lo = get_width(lo_prof)
    w_hi = get_width(hi_prof)
    
    print(f"File: {file_path}")
    print(f"Width Lo (index 0 region): {w_lo:.2f}")
    print(f"Width Hi (index -1 region): {w_hi:.2f}")
    
    if w_lo > w_hi:
        print("RESULT: Ascending (Correct)")
        return True
    else:
        print("RESULT: Descending (FLIPPED)")
        return False

# Check Wilhelm
check_file("data/chime/wilhelm_chime_I_602_3809_32000b_cntr_bpc.npy")

# Check Casey for comparison
check_file("data/chime/casey_chime_I_491_2085_32000b_cntr_bpc.npy")
