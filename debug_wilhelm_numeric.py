import numpy as np
from scipy.ndimage import gaussian_filter1d

def get_width_metric(profile):
    p = profile - np.nanpercentile(profile, 10)
    p = np.maximum(p, 0)
    p /= np.sum(p)
    t = np.arange(len(p))
    mu = np.sum(t * p)
    var = np.sum((t - mu)**2 * p)
    return np.sqrt(var)

file = "data/chime/wilhelm_chime_I_602_3809_32000b_cntr_bpc.npy"
data = np.load(file)
n_ch = data.shape[0]
q = n_ch // 4

bottom_profile = np.nansum(data[:q, :], axis=1) # Summing over time? No, need sum over freq in slices
# Wait, I want profile for bottom channels (freq range 1) and top channels (freq range 2)
# Profile is flux vs time.
bottom_profile = np.nansum(data[:q, :], axis=0)
top_profile = np.nansum(data[-q:, :], axis=0)

w_bottom = get_width_metric(bottom_profile)
w_top = get_width_metric(top_profile)

print(f"Width Metric (Bottom channels - Index 0-256): {w_bottom:.2f}")
print(f"Width Metric (Top channels - Index 768-1024): {w_top:.2f}")

if w_bottom > w_top:
    print("ORIENTATION: Ascending (Low frequency at index 0, more scattering)")
else:
    print("ORIENTATION: Descending (High frequency at index 0, less scattering)")
