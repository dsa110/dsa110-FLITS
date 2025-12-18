import numpy as np
from scipy.ndimage import gaussian_filter1d
import glob
import os

def check_orientation_windowed(file_path):
    try:
        data = np.load(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return
        
    data_proc = np.nan_to_num(data)
    n_ch, n_t = data_proc.shape
    
    # Pre-process
    q_t = n_t // 8
    off_pulse = np.r_[0:q_t, -q_t:0]
    mu = np.mean(data_proc[:, off_pulse], axis=1, keepdims=True)
    sig = np.std(data_proc[:, off_pulse], axis=1, keepdims=True)
    sig[sig < 1e-9] = 1.0
    data_proc = (data_proc - mu) / sig
    
    # 1. Find burst timing
    prof = np.sum(data_proc, axis=0)
    prof_smooth = gaussian_filter1d(prof, sigma=10)
    burst_idx = np.argmax(prof_smooth)
    
    # 2. Centroids in window
    window = 1000 if n_t > 5000 else 200
    w_start = max(0, burst_idx - window)
    w_end = min(n_t, burst_idx + 2*window)
    
    centroids = []
    for i in range(4):
        band = data_proc[i*n_ch//4:(i+1)*n_ch//4, w_start:w_end]
        p = np.sum(band, axis=0)
        p = p - np.percentile(p, 10)
        p = np.maximum(p, 0)
        p_norm = p / (np.sum(p) + 1e-12)
        c = np.sum(np.arange(len(p)) * p_norm)
        centroids.append(c)
    
    slope = np.polyfit(np.arange(4), centroids, 1)[0]
    
    # Ascending (Correct): centroids[0] > centroids[3] -> negative slope
    # Descending (Flipped): centroids[0] < centroids[3] -> positive slope
    
    res = "Ascending" if slope < 0 else "FLIPPED"
    print(f"{os.path.basename(file_path):50s} | Slope: {slope:7.2f} | {res}")
    return res

print(f"{'Filename':50s} | {'Slope':7s} | Status")
print("-" * 75)

all_files = sorted(glob.glob("data/chime/*.npy") + glob.glob("data/dsa/*.npy"))
for f in all_files:
    check_orientation_windowed(f)
