import numpy as np
import os

def audit_file(file_path):
    data = np.load(file_path)
    n_ch, n_t = data.shape
    
    # Robust BP
    q = n_t // 8
    off = np.r_[0:q, -q:0]
    mu = np.nanmean(data[:, off], axis=1, keepdims=True)
    sig = np.nanstd(data[:, off], axis=1, keepdims=True)
    sig[sig == 0] = 1.0
    data_corr = (data - mu) / sig
    
    centroids = []
    valid_bands = []
    
    for i in range(4):
        band = data_corr[i*n_ch//4:(i+1)*n_ch//4, :]
        prof = np.nanmean(band, axis=0)
        # Check if band has signal
        if np.max(prof) - np.min(prof) < 0.2: # Stricter threshold
            continue
            
        from scipy.ndimage import gaussian_filter1d
        prof_s = gaussian_filter1d(prof, sigma=50)
        peak = np.argmax(prof_s)
        
        w = 1000
        p_window = prof_s[max(0, peak-w):min(n_t, peak+w)]
        p_window = p_window - np.min(p_window)
        if np.sum(p_window) > 0:
            c = np.sum(np.arange(len(p_window)) * p_window) / np.sum(p_window)
            c_abs = c + max(0, peak-w)
            centroids.append(c_abs)
            valid_bands.append(i)
            
    if len(valid_bands) < 2:
        return None, "Insufficient Signal"
        
    slope = np.polyfit(valid_bands, centroids, 1)[0]
    res = "Ascending" if slope < 0 else "Descending"
    return slope, res

def run_global_audit():
    dirs = ["data/chime", "data/dsa"]
    for d in dirs:
        print(f"\n--- AUDITING {d} ---")
        files = [f for f in os.listdir(d) if f.endswith(".npy")]
        print(f"{'Filename':<50} | {'Slope':>10} | {'Status':<10}")
        print("-" * 75)
        for f in files:
            path = os.path.join(d, f)
            slope, status = audit_file(path)
            if slope is None:
                print(f"{f:<50} | {'N/A':>10} | {status:<10}")
            else:
                print(f"{f:<50} | {slope:>10.2f} | {status:<10}")

if __name__ == "__main__":
    run_global_audit()
