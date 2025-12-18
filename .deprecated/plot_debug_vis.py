
import numpy as np
import matplotlib.pyplot as plt
import json
from scattering.scat_analysis.burstfit_pipeline import BurstDataset
from scattering.scat_analysis.burstfit import FRBModel, FRBParams

def plot_debug(fit_json, data_npy, frb_name="freya"):
    # Load Results
    with open(fit_json) as f:
        res = json.load(f)
    print("Loaded JSON.")

    p_dict = res["best_params"]
    # Reconstruct params - careful with naming if using dataclass keys
    # JSON has "tau_1ghz", dataclass matches.
    params = FRBParams(**p_dict)
    model_key = res["best_model"]
    
    print("Params:", params)
    
    # Load Data
    # Manually mimic pipeline load
    # We need to know f_factor/t_factor used!
    # They are not explicitly in JSON metadata usually? 
    # Ah, I know I ran with f=64, t=24.
    
    # MOCK TELESCOPE AGIAIN or just use raw load
    class MockTel:
        n_ch_raw = 16384
        df_MHz_raw = 400.0/16384
        dt_ms_raw = 0.98304
        f_min_GHz = 0.400
        f_max_GHz = 0.800
        
    ds = BurstDataset(
        data_npy, 
        outpath=".", 
        name=frb_name, 
        telescope=MockTel(),
        f_factor=64, 
        t_factor=24, 
        flip_freq=True # Assuming this matches pipeline default
    )
    
    # Generate Model
    model_gen = FRBModel(
        time=ds.time, freq=ds.freq, data=ds.data, df_MHz=ds.df_MHz
    )
    
    model_arr = model_gen(params, model_key)
    print("Model generated. Shape:", model_arr.shape)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Check extent
    ext = [ds.time[0], ds.time[-1], ds.freq[0], ds.freq[-1]]
    
    # vmin/vmax for consistent contrast
    vmin, vmax = np.percentile(ds.data, [1, 99])
    
    im0 = axes[0].imshow(ds.data, aspect='auto', origin='lower', extent=ext, vmin=vmin, vmax=vmax)
    axes[0].set_title("Data")
    
    im1 = axes[1].imshow(model_arr, aspect='auto', origin='lower', extent=ext, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Model {model_key}")
    
    resid = ds.data - model_arr
    im2 = axes[2].imshow(resid, aspect='auto', origin='lower', extent=ext, vmin=vmin, vmax=vmax)
    axes[2].set_title("Residuals")
    
    plt.colorbar(im0, ax=axes[0])
    plt.colorbar(im1, ax=axes[1])
    plt.colorbar(im2, ax=axes[2])
    
    out_png = "debug_model_vis.png"
    plt.savefig(out_png)
    print(f"Saved {out_png}")
    
if __name__ == "__main__":
    plot_debug(
        "results/bursts/freya_data_driven/freya_fit_results.json",
        "data/chime/freya_chime_I_912_4067_32000b_cntr_bpc.npy"
    )
