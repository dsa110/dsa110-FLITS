
import numpy as np
from pathlib import Path

def check_orientation():
    files = sorted(Path("data/chime").glob("*.npy"))
    print(f"Checking {len(files)} files for frequency orientation...")
    
    for f in files:
        try:
            data = np.load(f)
            # Simple heuristic: CHIME bursts are usually dedispersed to ~400 MHz (bottom of band) or ~800 MHz (top).
            # But the raw data (before dedispersion correction in pipeline) contains dispersed signal.
            # More simply: The background noise levels or dropped channels might give a clue.
            # Or we can just check if the signal is at the "bottom" (index 0) or "top" (index -1) of the array 
            # relative to the dedispersion sweep. 
            
            # Actually, the user stated for Casey: "flux heavily concentrated within 700-800 MHz" in the corrupted fit,
            # meaning the signal was at the array indices corresponding to 800 MHz in the model.
            # If the model is Ascending (0=400, -1=800), and the signal was at 800 MHz (Index -1), 
            # then the signal was at Index -1.
            # If the data was Descending (0=800, -1=400), then signal at ~400 MHz would be at Index -1.
            
            # Let's just print the index of the peak for each file to see if they are consistent.
            # If all files have peak at similar relative frequency indices, they likely share the same orientation.
            
            # Collapse to frequency profile
            prof = np.sum(np.nan_to_num(data), axis=1)
            peak_idx = np.argmax(prof)
            total_ch = len(prof)
            rel_pos = peak_idx / total_ch
            
            print(f"{f.name}: Peak at index {peak_idx}/{total_ch} ({rel_pos:.2f})")
            
        except Exception as e:
            print(f"Failed to load {f.name}: {e}")

if __name__ == "__main__":
    check_orientation()
