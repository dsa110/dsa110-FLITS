import numpy as np
import glob
from pathlib import Path

def check_orientation(data):
    # Sum over time to get frequency profile
    prof = np.sum(data, axis=1)
    # This might not be enough. Let's look at the scattering tail.
    # We'll take the centroid of each channel.
    n_ch = data.shape[0]
    n_t = data.shape[1]
    times = np.arange(n_t)
    
    centroids = []
    for i in range(n_ch):
        ch_data = data[i] - np.mean(data[i][:n_t//4])
        if np.max(ch_data) > 3 * np.std(ch_data[:n_t//4]):
            centroid = np.sum(times * ch_data) / np.sum(ch_data)
            centroids.append(centroid)
        else:
            centroids.append(np.nan)
    
    centroids = np.array(centroids)
    valid = ~np.isnan(centroids)
    if np.sum(valid) < 10:
        return "Unknown"
    
    # Fit a line to centroids vs channel index
    idx = np.arange(n_ch)[valid]
    p = np.polyfit(idx, centroids[valid], 1)
    # p[0] is the slope (dt/d_chan_idx)
    # Scattering: low freq has LATER centroid.
    # If slope is positive: as index increases, centroid increases (gets later).
    # So higher index = lower frequency. -> Descending!
    # If slope is negative: as index increases, centroid decreases (gets earlier).
    # So higher index = higher frequency. -> Ascending!
    
    if p[0] > 0:
        return "Descending"
    else:
        return "Ascending"

files = glob.glob("data/chime/*.npy")
for f in files:
    try:
        data = np.load(f)
        orient = check_orientation(data)
        print(f"{f}: {orient}")
    except Exception as e:
        print(f"{f}: Error {e}")
