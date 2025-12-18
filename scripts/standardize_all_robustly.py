import numpy as np
from scipy.ndimage import gaussian_filter1d
import glob
import os

def standardize_file(file_path):
    try:
        data = np.load(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return
        
    data_proc = np.nan_to_num(data)
    n_ch, n_t = data_proc.shape
    
    # Pre-process for orientation check
    q_t = n_t // 8
    off_pulse = np.r_[0:q_t, -q_t:0]
    mu = np.mean(data_proc[:, off_pulse], axis=1, keepdims=True)
    sig = np.std(data_proc[:, off_pulse], axis=1, keepdims=True)
    sig[sig < 1e-9] = 1.0
    data_proc = (data_proc - mu) / sig
    
    # 1. Find burst timing (robustly)
    prof = np.sum(data_proc, axis=0)
    prof_smooth = gaussian_filter1d(prof, sigma=10)
    burst_idx = np.argmax(prof_smooth)
    
    # 2. Slice into 4 bands and get centroids in a window around burst
    window = 1000 if n_t > 5000 else 200 # CHIME vs DSA
    w_start = max(0, burst_idx - window)
    w_end = min(n_t, burst_idx + 2*window) # larger window for tail
    
    centroids = []
    freq_indices = np.linspace(0, n_ch, 5, dtype=int)
    for i in range(4):
        band = data_proc[freq_indices[i]:freq_indices[i+1], w_start:w_end]
        p = np.sum(band, axis=0)
        p = p - np.percentile(p, 10)
        p = np.maximum(p, 0)
        p_norm = p / (np.sum(p) + 1e-12)
        c = np.sum(np.arange(len(p)) * p_norm)
        centroids.append(c)
    
    # 3. Fit slope to centroids
    # If slope is positive, centroid increases with frequency channel index.
    # Since low freq arrives later (scattering), positive slope = Higher index is lower freq.
    # We want Ascending: lower index is lower freq. So we want NEGATIVE slope.
    
    slope = np.polyfit(np.arange(4), centroids, 1)[0]
    
    print(f"{os.path.basename(file_path):50s} | Slope: {slope:6.2f}", end=" | ")
    
    if slope < -0.5: # Clear negative slope -> Ascending
        print("Ascending (Correct)")
        # No flip needed
    elif slope > 0.5: # Clear positive slope -> Descending
        print("Descending -> FLIPPING")
        data_flipped = np.flip(data, axis=0)
        np.save(file_path, data_flipped)
    else:
        # Check edges specifically if slope is inconclusive
        if centroids[0] > centroids[-1]:
            print("Inconclusive slope, but C[0] > C[-1] -> Ascending")
        else:
            print("Inconclusive slope, but C[0] < C[-1] -> FLIPPING")
            data_flipped = np.flip(data, axis=0)
            np.save(file_path, data_flipped)

# Standardize ALL files
print("Standardizing CHIME files...")
chime_files = glob.glob("data/chime/*.npy")
for f in sorted(chime_files):
    standardize_file(f)

print("\nStandardizing DSA files...")
dsa_files = glob.glob("data/dsa/*.npy")
for f in sorted(dsa_files):
    standardize_file(f)
