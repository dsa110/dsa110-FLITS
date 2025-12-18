import numpy as np

def check_orientation_by_centroid(file_path):
    data = np.load(file_path)
    n_ch = data.shape[0]
    
    # Bandpass correction to see the burst clearly
    q_t = data.shape[1] // 4
    off_pulse = np.r_[0:q_t, -q_t:0]
    mu = np.mean(data[:, off_pulse], axis=1, keepdims=True)
    sig = np.std(data[:, off_pulse], axis=1, keepdims=True)
    sig[sig < 1e-9] = 1.0
    data = (data - mu) / sig
    
    # Slices
    lo_slice = data[:n_ch//4, :]
    hi_slice = data[-n_ch//4:, :]
    
    # Profiles
    p_lo = np.sum(lo_slice, axis=0)
    p_hi = np.sum(hi_slice, axis=0)
    
    # Simple peak-based tail check
    # Lower frequency should peak LATER and have a SLOWER decay
    peak_lo = np.argmax(p_lo)
    peak_hi = np.argmax(p_hi)
    
    print(f"Peak index (bottom channels): {peak_lo}")
    print(f"Peak index (top channels): {peak_hi}")
    
    # For a scattered burst:
    # If peak_lo > peak_hi, bottom is lower frequency (it arrives later due to scatter/DM)
    # If peak_lo < peak_hi, bottom is higher frequency.
    
    if peak_lo > peak_hi:
        return "Ascending (Index 0 is Low Freq)"
    elif peak_lo < peak_hi:
        return "Descending (Index 0 is High Freq)"
    else:
        # Check tail area
        area_lo = np.sum(p_lo[peak_lo:]) / np.max(p_lo)
        area_hi = np.sum(p_hi[peak_hi:]) / np.max(p_hi)
        print(f"Tail area ratio (lo/hi): {area_lo/area_hi:.3f}")
        if area_lo > area_hi:
             return "Ascending (Index 0 is Low Freq)"
        else:
             return "Descending (Index 0 is High Freq)"

file = "data/chime/wilhelm_chime_I_602_3809_32000b_cntr_bpc.npy"
print(f"File: {file}")
print(check_orientation_by_centroid(file))
