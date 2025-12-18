import numpy as np
import matplotlib.pyplot as plt

file = "data/chime/wilhelm_chime_I_602_3809_32000b_cntr_bpc.npy"
data = np.load(file)
n_ch, n_t = data.shape

# Simple bandpass: median of each channel
bp = np.nanmedian(data, axis=1)
# Normalize channels where possible
data_norm = np.zeros_like(data)
for i in range(n_ch):
    if bp[i] > 0:
        data_norm[i] = data[i] / bp[i]

# Find burst peak time
prof = np.sum(data_norm, axis=0)
best_t = np.argmax(prof)
window = 1000
start = max(0, best_t - window)
end = min(n_t, best_t + window)

lo_band = np.sum(data_norm[:n_ch//4, start:end], axis=0)
hi_band = np.sum(data_norm[-n_ch//4:, start:end], axis=0)

plt.figure(figsize=(12, 6))
plt.plot(lo_band, label="Rows 0-255 (should be 400-500 MHz)")
plt.plot(hi_band, label="Rows 768-1023 (should be 700-800 MHz)")
plt.legend()
plt.title(f"Wilhelm Orientation Check (Windowed)\nFile: {file}")
plt.savefig("wilhelm_check_v3.png")
print("Saved wilhelm_check_v3.png")
