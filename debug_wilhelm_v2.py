import numpy as np
import matplotlib.pyplot as plt

file = "data/chime/wilhelm_chime_I_602_3809_32000b_cntr_bpc.npy"
data = np.load(file)
n_ch = data.shape[0]

# Bandpass correction for visibility
q = data.shape[1] // 8
off = np.r_[0:q, -q:0]
mu = np.nanmean(data[:, off], axis=1, keepdims=True)
sig = np.nanstd(data[:, off], axis=1, keepdims=True)
data_corr = (data - mu) / sig

# Compare top and bottom bands
lo_band = np.nanmean(data_corr[:n_ch//4, :], axis=0)
hi_band = np.nanmean(data_corr[-n_ch//4:, :], axis=0)

plt.figure(figsize=(12, 6))
plt.plot(lo_band, label="Rows 0 to 255 (should be Low Freq = More Scatter)")
plt.plot(hi_band, label="Rows 768 to 1023 (should be High Freq = Less Scatter)")
plt.legend()
plt.title(f"Wilhelm Orientation Check\nFile: {file}")
plt.savefig("wilhelm_check_v2.png")
print("Saved wilhelm_check_v2.png")
