import numpy as np
import matplotlib.pyplot as plt

file = "data/chime/wilhelm_chime_I_602_3809_32000b_cntr_bpc.npy"
data = np.load(file)
n_ch, n_t = data.shape

# Simple BP-corrected profiles
q = n_t // 8
off = np.r_[0:q, -q:0]
mu = np.nanmean(data[:, off], axis=1, keepdims=True)
sig = np.nanstd(data[:, off], axis=1, keepdims=True)
sig[sig == 0] = 1.0
data_corr = (data - mu) / sig

# Check scattering width
# Find peak in 4 bands
res = []
for i in range(4):
    band = data_corr[i*n_ch//4:(i+1)*n_ch//4, :]
    prof = np.nanmean(band, axis=0)
    # Filter to reduce noise
    from scipy.ndimage import gaussian_filter1d
    prof = gaussian_filter1d(prof, sigma=50)
    peak_idx = np.argmax(prof)
    
    # Calculate "width" as FWHM or just look at it
    h = np.max(prof)
    # Width after peak
    tail = prof[peak_idx:]
    width = np.sum(tail > (h * 0.5))
    res.append((i, peak_idx, width))

print("Band Results (Index, PeakIdx, TailWidthSamps):")
for r in res:
    print(f"Band {r[0]}: Peak={r[1]}, Width={r[2]}")

if res[0][1] > res[3][1]:
    print("Arrival Time: Band 0 (Bottom) arrives LATER than Band 3 (Top)")
    print("Conclusion: INDEX 0 IS LOW FREQUENCY (Ascending)")
else:
    print("Arrival Time: Band 0 (Bottom) arrives EARLIER than Band 3 (Top)")
    print("Conclusion: INDEX 0 IS HIGH FREQUENCY (Descending)")

if res[0][2] > res[3][2]:
    print("Scattering: Band 0 (Bottom) has LARGER width than Band 3 (Top)")
    print("Observation: INDEX 0 IS LOW FREQUENCY (Ascending)")
else:
    print("Scattering: Band 0 (Bottom) has SMALLER width than Band 3 (Top)")
    print("Observation: INDEX 0 IS HIGH FREQUENCY (Descending)")
