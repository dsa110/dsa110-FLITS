
import os
import logging
from pathlib import Path
from scattering.scat_analysis.pipeline.core import BurstPipeline
from scattering.scat_analysis.config_utils import TelescopeConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def run_fitting():
    # Common parameters
    dm_init = 912.4
    out_dir = Path("analysis_results/freya_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. DSA-110 Fitting
    print("\n" + "="*50)
    print("Starting DSA-110 Fitting for Freya")
    print("="*50)
    
    dsa_file = Path("data/dsa/freya_dsa_I_912_4_2500b_cntr_bpc.npy")
    
    dsa_config = TelescopeConfig(
        name="dsa",
        df_MHz_raw=0.03051757812,
        dt_ms_raw=0.032768,
        f_min_GHz=1.31125,
        f_max_GHz=1.49875
    )
    
    if dsa_file.exists():
        pipeline_dsa = BurstPipeline(
            inpath=dsa_file,
            outpath=out_dir / "dsa",
            name="freya_dsa",
            dm_init=dm_init,
            telescope=dsa_config,
            # Pipeline options
            steps=1000,          # Longer run
            f_factor=16,        # Downsample freq (DSA has high res)
            t_factor=16,        # Downsample time
            nproc=4,            # Use 4 cores
            model_scan=True,    # Scan M0-M3
            refine_dm=True,     # Enable DM refinement
            dm_search_window=10.0, # Wider window
            dm_n_bootstrap=50,   # Faster
            walker_width_frac=0.1, # Help walkers expand
            yes=True,           # Skip confirmation
            plot=True
        )
        try:
            pipeline_dsa.run_full()
            print("DSA fitting completed successfully.")
        except Exception as e:
            print(f"DSA fitting failed: {e}")
    else:
        print(f"DSA data file not found at {dsa_file}")

    # 2. CHIME Fitting
    print("\n" + "="*50)
    print("Starting CHIME Fitting for Freya")
    print("="*50)
    
    chime_file = Path("data/chime/freya_chime_I_912_4067_32000b_cntr_bpc.npy")
    
    chime_config = TelescopeConfig(
        name="chime",
        df_MHz_raw=0.390625,
        dt_ms_raw=0.00256,
        f_min_GHz=0.40019,
        f_max_GHz=0.80019
    )
    
    if chime_file.exists():
        pipeline_chime = BurstPipeline(
            inpath=chime_file,
            outpath=out_dir / "chime",
            name="freya_chime",
            dm_init=dm_init,
            telescope=chime_config,
            # Pipeline options
            steps=1000,          # Longer run
            f_factor=1,         # CHIME data usually already downsampled or low res
            t_factor=8,         # Downsample time (0.00256ms is very fine)
            nproc=4,
            model_scan=True,
            refine_dm=False,    # Disable DM refinement for speed
            walker_width_frac=0.1,
            yes=True,
            plot=True
        )
        try:
            pipeline_chime.run_full()
            print("CHIME fitting completed successfully.")
        except Exception as e:
            print(f"CHIME fitting failed: {e}")
    else:
        print(f"CHIME data file not found at {chime_file}")

if __name__ == "__main__":
    run_fitting()
