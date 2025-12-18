
import os
import sys
import numpy as np
import yaml
import json
import logging
from pathlib import Path

# Ensure local imports work
sys.path.insert(0, os.getcwd())

from scintillation.scint_analysis.pipeline import ScintillationAnalysis

def create_synthetic_burst(filename):
    """Creates a synthetic burst .npz file."""
    print(f"Creating synthetic data at {filename}...")
    n_freq = 64
    n_time = 512
    freqs = np.linspace(1300, 1400, n_freq) # 100 MHz bandwidth
    times = np.linspace(0, 0.1, n_time)     # 100 ms duration
    
    # Background noise
    np.random.seed(42)
    power = np.random.normal(10, 1, (n_freq, n_time))
    
    # Add burst at center: Gaussian in time, scint structure in freq
    t_idx = n_time // 2
    f_idx = np.arange(n_freq)
    
    # Time profile
    time_profile = np.exp(-0.5 * ((np.arange(n_time) - t_idx) / 20) ** 2)
    
    # Frequency structure (simulated scintillation)
    freq_structure = 1.0 + 0.8 * np.sin(2 * np.pi * f_idx / 10)**2
    
    signal = 50 * np.outer(freq_structure, time_profile)
    power += signal
    
    np.savez(filename, power_2d=power, frequencies_mhz=freqs, times_s=times)

def create_temp_config(config_path, data_path):
    """Creates a temporary YAML config."""
    print(f"Creating config at {config_path}...")
    config = {
        'burst_id': 'synthetic_test_burst',
        'input_data_path': str(data_path),
        'telescope': 'dsa',
        'analysis': {
            'rfi_masking': {
                'find_burst_thres': 5.0,
                'rfi_downsample_factor': 4,
                # Explicit windows to ensure it finds the signal
                'manual_burst_window': [200, 312], 
                'manual_noise_window': [0, 100]
            },
            'acf': {
                'num_subbands': 4,
                'max_lag_mhz': 50.0,
                'use_snr_subbanding': False,
                'enable_intra_pulse_analysis': False # Keep it simple
            },
            'fitting': {
                'fit_lagrange_mhz': 20.0,
                'reference_frequency_mhz': 1350.0,
                # Force simple model for robust test
                'force_model': 'lorentzian_component' 
            },
            'noise': {
                'disable': True # Simplify for test
            },
            'baseline_subtraction': {
                'enable': True,
                'poly_order': 1
            }
        },
        'pipeline_options': {
            'log_level': 'INFO',
            'save_intermediate_steps': False,
            'force_recalc': True,
            'halt_after_acf': False
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

def main():
    # Setup paths
    base_dir = Path(os.getcwd())
    data_dir = base_dir / "scintillation" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    data_path = data_dir / "synthetic_test.npz"
    config_path = base_dir / "synthetic_test_config.yaml"
    
    # 1. Create Data
    create_synthetic_burst(data_path)
    
    # 2. Create Config
    create_temp_config(config_path, data_path)
    
    # 3. Run Pipeline
    print("\n--- Running Pipeline ---")
    try:
        # Load config dictionary
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
            
        # Initialize and run
        scint_pipeline = ScintillationAnalysis(loaded_config)
        scint_pipeline.run()
        
        # 4. Verify Results
        results = scint_pipeline.final_results
        
        if not results:
            print("\n❌ FAILURE: Pipeline finished but produced no results.")
            sys.exit(1)
            
        print("\n✅ Pipeline execution completed.")
        
        # Check structure
        print("Keys in results:", results.keys())
        
        if 'components' not in results:
             print("❌ FAILURE: 'components' missing from results.")
             sys.exit(1)
             
        # Check for our forced model
        comp_data = results['components'].get('scint_scale') or results['components'].get('component_1')
        if not comp_data:
             print("❌ FAILURE: No component data found.")
             sys.exit(1)
             
        measurements = comp_data.get('subband_measurements', [])
        print(f"Found {len(measurements)} sub-band measurements.")
        
        if len(measurements) == 0:
             print("❌ FAILURE: No sub-band measurements extracted.")
             sys.exit(1)

        # Check values
        first_meas = measurements[0]
        print("Sample Measurement:", json.dumps(first_meas, indent=2))
        
        if 'bw' in first_meas and 'mod' in first_meas:
            print(f"\n✅ SUCCESS: Scintillation parameters extracted!")
            print(f"   Bandwidth: {first_meas['bw']:.4f} MHz")
            print(f"   Modulation Index: {first_meas['mod']:.4f}")
        else:
             print("❌ FAILURE: Bandwidth or Modulation Index missing.")
             sys.exit(1)

    except Exception as e:
        print(f"\n❌ CRITICAL EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        if config_path.exists():
            os.remove(config_path)
        if data_path.exists():
            os.remove(data_path)
        print("\nCleanup complete.")

if __name__ == "__main__":
    main()
