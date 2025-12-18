import numpy as np
import glob
import matplotlib.pyplot as plt

def get_centroids(data):
    n_ch, n_t = data.shape
    times = np.arange(n_t)
    cs = []
    for i in range(n_ch):
        ch = data[i] - np.mean(data[i][:n_t//4])
        # Use more robust threshold
        if np.max(ch) > 5 * np.std(data[i][:n_t//4]):
            # Use only samples near the peak to avoid RFI tail bias
            peak = np.argmax(ch)
            w = 500
            start, end = max(0, peak-w), min(n_t, peak+2*w)
            sub_ch = ch[start:end]
            sub_t = times[start:end]
            cs.append(np.sum(sub_t * sub_ch) / np.sum(sub_ch))
        else:
            cs.append(np.nan)
    return np.array(cs)

files = glob.glob("data/**/*.npy", recursive=True)
plt.figure(figsize=(15, 10))
for f in files:
    if "backup" in f: continue
    data = np.load(f)
    cs = get_centroids(data)
    valid = ~np.isnan(cs)
    if np.sum(valid) > 10:
        idx = np.arange(len(cs))[valid]
        plt.plot(idx, cs[valid] - np.nanmedian(cs), label=f.split('/')[-1])
        p = np.polyfit(idx, cs[valid], 1)
        print(f"{f}: slope={p[0]:.4f}")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("Centroid Trend (Burst Centroid - Median) vs Channel Index")
plt.xlabel("Channel Index")
plt.ylabel("Relative Centroid [Samples]")
plt.savefig("centroid_trends.png")
print("Saved centroid_trends.png")
