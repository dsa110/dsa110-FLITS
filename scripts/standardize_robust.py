import numpy as np
import glob
from pathlib import Path
import os

def check_orientation_robust(data):
    """
    More robust centroid calculation using windowing around peak.
    """
    n_ch, n_t = data.shape
    times = np.arange(n_t)
    cs = []
    for i in range(n_ch):
        ch = data[i] - np.mean(data[i][:n_t//4])
        if np.max(ch) > 4 * np.std(data[i][:n_t//4]):
            peak = np.argmax(ch)
            # Use window to avoid RFI interference
            w = 500
            start, end = max(0, peak-w), min(n_t, peak+2*w)
            sub_ch = ch[start:end]
            sub_t = times[start:end]
            val = np.sum(sub_t * sub_ch) / np.sum(sub_ch)
            cs.append(val)
        else:
            cs.append(np.nan)
    
    cs = np.array(cs)
    valid = ~np.isnan(cs)
    if np.sum(valid) < 10:
        return "Unknown", 0
    
    idx = np.arange(n_ch)[valid]
    p = np.polyfit(idx, cs[valid], 1)
    
    # Positive slope: higher index has LATER centroid -> higher index is LOWER frequency (Descending)
    # Negative slope: higher index has EARLIER centroid -> higher index is HIGHER frequency (Ascending)
    orient = "Descending" if p[0] > 0 else "Ascending"
    return orient, p[0]

def standardize_all():
    files = glob.glob("data/**/*.npy", recursive=True)
    print(f"Standardizing {len(files)} files to Ascending order...")
    
    for f in files:
        if "backup" in f: continue
        try:
            data = np.load(f)
            orient, slope = check_orientation_robust(data)
            
            if orient == "Descending":
                print(f"[FLIPPING] {f} (slope={slope:.4f})")
                new_data = np.flip(data, axis=0)
                np.save(f, new_data)
            elif orient == "Ascending":
                print(f"[OK] {f} (slope={slope:.4f})")
            else:
                print(f"[SKIP] {f} (Orientation Unknown)")
                
        except Exception as e:
            print(f"[ERROR] {f}: {e}")

if __name__ == "__main__":
    standardize_all()
