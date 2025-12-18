import numpy as np
import matplotlib.pyplot as plt

def debug_plot(npy_path, output_path):
    data = np.load(npy_path)
    # Downsample slightly for visibility
    if data.shape[1] > 2000:
        data = data[:, ::int(data.shape[1]/1000)]
    
    # 2D plot with raw indices
    plt.figure(figsize=(10, 8))
    plt.imshow(data, aspect='auto', origin='lower')
    plt.title(f"Raw indices (origin='lower') for {npy_path}")
    plt.xlabel("Raw Time Samples")
    plt.ylabel("Raw Frequency Channel Index")
    plt.colorbar(label="Intensity")
    plt.savefig(output_path)
    plt.close()

debug_plot("data/chime/casey_chime_I_491_2085_32000b_cntr_bpc.npy", "casey_raw_check.png")
debug_plot("data/chime/freya_chime_I_912_4067_32000b_cntr_bpc.npy", "freya_raw_check.png")
print("Plots saved: casey_raw_check.png, freya_raw_check.png")
