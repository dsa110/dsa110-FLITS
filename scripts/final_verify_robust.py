import numpy as np
import glob

def check_orient_robust(data):
    n_ch, n_t = data.shape
    times = np.arange(n_t)
    cs = []
    for i in range(n_ch):
        ch = data[i] - np.mean(data[i][:n_t//4])
        if np.max(ch) > 4 * np.std(data[i][:n_t//4]):
            peak = np.argmax(ch)
            w = 500
            start, end = max(0, peak-w), min(n_t, peak+2*w)
            sub_ch = ch[start:end]
            sub_t = times[start:end]
            cs.append(np.sum(sub_t * sub_ch) / np.sum(sub_ch))
        else:
            cs.append(np.nan)
    cs = np.array(cs)
    v = ~np.isnan(cs)
    if np.sum(v) < 10: return "Unknown"
    p = np.polyfit(np.arange(n_ch)[v], cs[v], 1)
    return "Descending" if p[0] > 0 else "Ascending"

files = glob.glob("data/**/*.npy", recursive=True)
results = {}
for f in files:
    if "backup" in f: continue
    try: results[f] = check_orient_robust(np.load(f))
    except: results[f] = "Error"

for f, res in results.items():
    print(f"{f}: {res}")

all_asc = all(r == "Ascending" for r in results.values() if r not in ["Unknown", "Error"])
print(f"\nFinal Check Summary:")
print(f"Total files: {len(results)}")
print(f"All Ascending: {all_asc}")
