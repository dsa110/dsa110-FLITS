import numpy as np
import glob
from pathlib import Path

def check_orientation_detailed(data):
    n_ch = data.shape[0]
    n_t = data.shape[1]
    times = np.arange(n_t)
    
    centroids = []
    for i in range(n_ch):
        # Subtract mean of off-pulse
        ch_data = data[i] - np.mean(data[i][:n_t//4])
        # Only use channels with significant signal
        if np.max(ch_data) > 5 * np.std(data[i][:n_t//4]):
            centroid = np.sum(times * ch_data) / np.sum(ch_data)
            centroids.append(centroid)
        else:
            centroids.append(np.nan)
    
    centroids = np.array(centroids)
    valid = ~np.isnan(centroids)
    if np.sum(valid) < 10:
        return "Unknown", 0
    
    idx = np.arange(n_ch)[valid]
    p = np.polyfit(idx, centroids[valid], 1)
    
    # Scattering tail makes centroid LATER at LOWER frequencies.
    # If p[0] is positive: higher index has LATER centroid -> higher index is LOWER frequency. -> Descending.
    # If p[0] is negative: higher index has EARLIER centroid -> higher index is HIGHER frequency. -> Ascending.
    
    orient = "Descending" if p[0] > 0 else "Ascending"
    return orient, p[0]

files = ["data/chime/casey_chime_I_491_2085_32000b_cntr_bpc.npy", "data/chime/freya_chime_I_912_4067_32000b_cntr_bpc.npy"]
for f in files:
    data = np.load(f)
    orient, slope = check_orientation_detailed(data)
    print(f"{f}: {orient} (slope={slope:.4f})")
