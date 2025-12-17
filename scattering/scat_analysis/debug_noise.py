
import sys
import numpy as np
import matplotlib.pyplot as plt
from scattering.scat_analysis.burstfit_pipeline import BurstDataset
from scattering.scat_analysis.burstfit import FRBModel

# Mock TelescopeConfig
class MockTelescope:
    def __init__(self):
        self.n_ch_raw = 16384 # Approx for CHIME
        self.df_MHz_raw = 400.0 / 16384
        self.dt_ms_raw = 0.98304
        self.f_min_GHz = 0.400
        self.f_max_GHz = 0.800

def diagnose_noise(inpath, frb_name="freya"):
    print(f"Diagnosing noise for {inpath}...")
    
    # Minimal Setup
    telescope = MockTelescope()
    
    # Initialize Dataset
    dataset = BurstDataset(
        inpath, 
        outpath="dummy_output",
        name=frb_name,
        telescope=telescope,
        f_factor=64, 
        t_factor=24,
        flip_freq=True
    )
    
    # Load data by calling preprocess explicitly if needed, or accessing property
    # BurstDataset usually loads on init or property access
    data = dataset.data
    # data = dataset.data
    data = dataset.data
    # weights = dataset.weights # Not available directly
    
    nt, nf = data.shape
    print(f"Data Loaded. Shape: {data.shape}")
    
    # Identify OFF-pulse region
    # Simple strategy: use first 25% and last 25% of time samples
    # Assuming burst is centered
    t_start_off = 0
    t_end_off_1 = nt // 4
    
    t_start_off_2 = int(0.75 * nt)
    t_end_off_2 = nt
    
    off_data_1 = data[t_start_off:t_end_off_1, :]
    off_data_2 = data[t_start_off_2:t_end_off_2, :]
    off_data = np.concatenate([off_data_1, off_data_2], axis=0)
    
    print(f"Off-pulse samples: {off_data.size}")
    
    # Statistics
    median_val = np.median(off_data)
    mean_val = np.mean(off_data)
    std_val = np.std(off_data)
    mad_std = 1.4826 * np.median(np.abs(off_data - median_val))
    
    print(f"\n--- Off-Pulse Statistics ---")
    print(f"Mean:   {mean_val:.4f}")
    print(f"Median: {median_val:.4f}")
    print(f"Std (Classic): {std_val:.4f}")
    print(f"Std (MAD):     {mad_std:.4f}")
    
    # Pipeline Noise Model
    # BurstDataset typically bandpass corrects data to be S/N units?
    # If so, noise_std should be ~1.0. Let's check.
    
    model = FRBModel(
        time=dataset.time,
        freq=dataset.freq,
        data=dataset.data,
        df_MHz=dataset.df_MHz
    )
    noise_est = model.noise_std
    
    print(f"\n--- Pipeline Noise Model ---")
    if isinstance(noise_est, np.ndarray):
        print(f"Model noise_std mean:  {np.mean(noise_est):.4f}")
    else:
        print(f"Model noise_std value: {noise_est:.4f}")
        
    # Check consistency
    actual_noise = mad_std
    model_noise = np.mean(noise_est)
    
    ratio = actual_noise / model_noise
    print(f"\nRATIO (Actual / Model) = {ratio:.2f}")
    
    if ratio > 1.2:
        print("❌ FAIL: Actual noise is >20% higher than model estimate.")
        print("   This causes Chi-squared to be inflated by factor of R^2.")
        print(f"   Expected Chi-sq ~ {ratio**2:.2f}")
    elif ratio < 0.8:
        print("⚠️ Warning: Model overestimates noise.")
    else:
        print("✅ Noise estimate is accurate.")
        
    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        diagnose_noise(sys.argv[1])
    else:
        print("Usage: python debug_noise.py <npy_file>")
