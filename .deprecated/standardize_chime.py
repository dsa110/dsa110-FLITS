
import numpy as np
from pathlib import Path
import shutil

def standardize_data():
    files = sorted(Path("data/chime").glob("*.npy"))
    print(f"Checking {len(files)} files for standardization...")
    
    flipped_count = 0
    skipped_count = 0
    
    for f in files:
        if f.is_dir(): continue
        try:
            data = np.load(f)
            
            # Diagnostic: Peak location
            prof = np.sum(np.nan_to_num(data), axis=1)
            peak_idx = np.argmax(prof)
            total_ch = len(prof)
            rel_pos = peak_idx / total_ch
            
            print(f"{f.name}: Peak @ {peak_idx}/{total_ch} ({rel_pos:.2f}) -> ", end="")
            
            # Heuristic: If peak is in lower half (Ascending), flip to Descending (High High)
            if rel_pos < 0.5:
                print("FLIPPING (Ascending -> Descending)")
                
                # Flip vertically
                data_flipped = np.flipud(data)
                
                # Save
                np.save(f, data_flipped)
                flipped_count += 1
            else:
                print("OK (Descending)")
                skipped_count += 1
                
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nSummary: Flipped {flipped_count}, Skipped {skipped_count}, Total {len(files)}")

if __name__ == "__main__":
    standardize_data()
