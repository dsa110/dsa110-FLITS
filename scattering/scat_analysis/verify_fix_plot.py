
import json
import numpy as np
from pathlib import Path
from scattering.scat_analysis.burstfit_pipeline import BurstDataset, create_sixteen_panel_plot, create_four_panel_plot
from scattering.scat_analysis.burstfit import FRBParams
import matplotlib.pyplot as plt

def verify_plot():
    # Paths
    data_path = "data/chime/casey_chime_I_491_2085_32000b_cntr_bpc.npy"
    results_path = "scattering/scat_process/casey_chime_I_491_2085_32000b_cntr_bpc_fit_results.json"
    out_dir = "scattering/scat_process"
    
    # 1. Load Data (Correctly oriented: flip_freq=False)
    print(f"Loading data from {data_path}...")
    # Mock telescope config needed for dataset
    from scattering.scat_analysis.config_utils import load_telescope_block
    tel_cfg = load_telescope_block("scattering/configs/telescopes.yaml", "chime")
    
    dataset = BurstDataset(
        inpath=data_path,
        outpath=out_dir,
        name="casey_fixed",
        telescope=tel_cfg,
        f_factor=32,
        t_factor=4,
        flip_freq=False  # KEY FIX
    )
    
    # 2. Load Results
    print(f"Loading results from {results_path}...")
    with open(results_path, 'r') as f:
        res_json = json.load(f)
        
    # Reconstruct robust results dict
    best_p_dict = res_json['best_params']
    # Ensure all required keys for FRBParams
    best_p = FRBParams(**best_p_dict)
    
    results = {
        "best_key": "M3",
        "best_params": best_p,
        "model_instance": dataset.model,
        "goodness_of_fit": res_json.get("goodness_of_fit"),
        "param_names": res_json.get("param_names", []),
        "chain_stats": {},
        "flat_chain": np.zeros((1, len(res_json.get("param_names", [])))), # Dummy chain
        "diagnostics": {} # Dummy diagnostics
    }
    
    # 3. Plot
    print("Generating plots...")
    create_four_panel_plot(dataset, results, save=True, show=False)
    # create_sixteen_panel_plot requires more data (chain etc) which we might lack in JSON reconstruction without full pickle
    # But try anyway for resid histogram check?
    # The JSON only has summary stats, not the full chain. 
    # create_sixteen_panel_plot uses flat_chain for scatter plots. 
    # We can skip 16-panel or use dummy chain.
    try:
        create_sixteen_panel_plot(dataset, results, save=True, show=False)
        print("16-panel plot generated.")
    except Exception as e:
        print(f"Skipping 16-panel plot (requires full chain): {e}")

if __name__ == "__main__":
    verify_plot()
