import numpy as np
import matplotlib.pyplot as plt
from scattering.scat_analysis.burstfit_init import estimate_scattering_from_tail

file = "data/chime/wilhelm_chime_I_602_3809_32000b_cntr_bpc.npy"
data = np.load(file)
print(f"Shape: {data.shape}")

# Average of top 10% channels vs bottom 10%
n_ch = data.shape[0]
q = n_ch // 10
bottom_profile = np.nansum(data[:q, :], axis=0)
top_profile = np.nansum(data[-q:, :], axis=0)

plt.figure(figsize=(10, 6))
plt.plot(bottom_profile, label="First 10% Channels (Index 0 is here)")
plt.plot(top_profile, label="Last 10% Channels (Index -1 is here)")
plt.legend()
plt.title("Wilhelm Orientation Check: Lower indexes vs Higher indexes")
plt.savefig("wilhelm_check.png")
print("Saved wilhelm_check.png")
