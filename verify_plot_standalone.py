import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.insert(0, os.getcwd())

try:
    from scattering.scat_analysis.burstfit_pipeline import BurstDataset, create_fit_summary_plot
    from scattering.scat_analysis.burstfit import FRBParams, FRBModel
    from scattering.scat_analysis.config_utils import TelescopeConfig
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

# Paths
data_path = "data/chime/casey_chime_I_491_2085_32000b_cntr_bpc.npy"
results_path = "results/bursts/casey/casey_chime_I_491_2085_32000b_cntr_bpc_fit_results.json"

if not os.path.exists(data_path) or not os.path.exists(results_path):
    print("Files not found. Check paths.")
    sys.exit(1)

# Load results
with open(results_path, 'r') as f:
    res_dict = json.load(f)

# Reconstruct objects
best_key = res_dict["best_model"]
p_vals = res_dict["best_params"]

# Use the required args for __init__
p_obj = FRBParams(c0=p_vals["c0"], t0=p_vals["t0"], gamma=p_vals["gamma"])
for k, v in p_vals.items():
    setattr(p_obj, k, v)

# Telescope config
tel = TelescopeConfig(
    name="chime",
    df_MHz_raw=0.390625,
    dt_ms_raw=2.56e-3,
    f_min_GHz=0.40019,
    f_max_GHz=0.80019
)

# Dataset
print("Loading dataset...")
dataset = BurstDataset(data_path, "data/chime", name="casey", telescope=tel, t_factor=4, f_factor=32)

# Build results for plotter
print("Reconstructing results for plotter...")
plot_results = {
    "best_key": best_key,
    "best_params": p_obj,
    "param_names": res_dict["param_names"],
    "goodness_of_fit": res_dict["goodness_of_fit"],
    "model_instance": dataset.model,
    "flat_chain": np.random.normal(0, 0.01, size=(500, len(res_dict["param_names"]))), # Mock chain for UNC
    "diagnostics": {
        "subband_2d": ("tau_1ghz", [(0.02, 0.005), (0.018, 0.004), (0.022, 0.006)], None), # Mock subband
        "dm_check": (np.linspace(-0.1, 0.1, 10), np.sin(np.linspace(0, 3, 10)) + 10), # Mock DM
        "influence": np.random.uniform(0, 1, size=len(dataset.freq)) # Mock influence
    }
}

# Run plotter
print("Generating plot...")
try:
    create_fit_summary_plot(dataset, plot_results, save=True, show=False)
    print(f"Plot saved to: {dataset.outpath / (dataset.name + '_enhanced_diagnostics.png')}")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
