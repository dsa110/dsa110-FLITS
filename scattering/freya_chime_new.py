# -*- coding: utf-8 -*-
# Auto-generated from freya_chime_new.ipynb on 2025-10-27T22:52:05
# Markdown cells preserved as commented blocks; code cells separated with "# %%".

# %%
# ------------------------------------------------------------------
# 0. Imports and Setup
# ------------------------------------------------------------------
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    import chainconsumer, seaborn, emcee, arviz
except:
    print('Installing Non-Default Packages...')
    os.system('pip install seaborn')
    os.system('pip install emcee')
    os.system('pip install chainconsumer')
    os.system('pip install arviz')

# It's good practice to manage your python path this way
# Create a 'scat_analysis' directory for your code if you haven't already
# and place this notebook outside of it.
# e.g., /path/to/project/notebook.ipynb
#       /path/to/project/scat_analysis/__init__.py
#       /path/to/project/scat_analysis/burstfit_pipeline.py
#       ...
# This makes imports clean and explicit.
# If your project root is not in the path, uncomment the following line:
# sys.path.insert(0, '/path/to/your/project/root')

# Use ipython magic for interactive development
%load_ext autoreload
%autoreload 2

# --- Core Pipeline Import ---
from scat_analysis.burstfit_pipeline import BurstPipeline

# --- Interactive Post-processing Imports ---
# These are for optional, interactive analysis after the main run.
from scat_analysis.burstfit_corner import (
    quick_chain_check,
    get_clean_samples,
    make_beautiful_corner,
    make_beautiful_corner_wide
)
from scat_analysis.burstfit import FRBParams

# %%
# ------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------
# --- Locate data file ---
burst_name = "freya" # Name of the burst to analyze
data_dir = Path("/Users/jakobfaber/Documents/research/caltech/ovro/dsa110/chime_dsa_codetections/FLITS/data/chime")
#Path("/arc/home/jfaber/baseband_morphologies/chime_dsa_codetections/data/CHIME_bursts/dmphase")
data_dir_parent = data_dir.parent
plot_dir = Path("/Users/jakobfaber/Documents/research/caltech/ovro/dsa110/chime_dsa_codetections/FLITS/scattering/plots")
#Path("/arc/home/jfaber/baseband_morphologies/chime_dsa_codetections/FLITS/scattering/plots")

# Use pathlib for robust path handling
try:
    data_path = next(data_dir.glob(f"*{burst_name}*.npy"))
    print(f"Found data file: {data_path}")
except StopIteration:
    raise FileNotFoundError(f"No .npy file containing '{burst_name}' found in {data_dir}")

# --- Set Telescope and Run Parameters ---
# These parameters will be passed directly to the pipeline
pipeline_config = {
    "telescope": "chime",
    "telcfg_path": "configs/telescopes.yaml",
    "sampcfg_path": "configs/sampler.yaml",
    "steps": 1000,          # Total MCMC steps for the final run
    "f_factor": 8,        # Downsampling factor in frequency
    "t_factor": 512,          # Downsampling factor in time
    "center_burst": True,
    "outer_trim": 0.2,
    "smooth_ms": 0.1,
    "nproc": 16,             # Number of processes for multiprocessing
    "yes": True,            # Auto-confirm pool creation
}

# %%
# --- Set Initial Dispersion Measure ---
dm_initial = 0.0 # pc cm^-3

# ------------------------------------------------------------------
# 2. Build and Run the Pipeline
# ------------------------------------------------------------------
# The pipeline is instantiated with all configuration parameters.
# The `with` statement ensures the multiprocessing pool is handled correctly.
pipe = BurstPipeline(
    name=burst_name,
    inpath=data_path,
    outpath=plot_dir,
    dm_init=dm_initial,
    **pipeline_config
)

# %%
# This single call now performs all the steps:
# - Data loading and preprocessing
# - Finding an optimized initial guess
# - Running the model selection scan (or a direct fit)
# - Processing the MCMC chains
# - Running all diagnostics (sub-band, influence, etc.)
# - Calculating goodness-of-fit
# - Generating and saving the 16-panel summary plot
results = pipe.run_full(
    model_scan=True,      # Perform BIC scan over models M0-M3
    model_keys=["M3"],
    diagnostics=True,     # Run all post-fit diagnostic checks
    plot=True,            # Generate and save the summary plot
    save=True,            # Save output figures
    show=True            # Do not block execution with plt.show()
    
)

# %%
# The main results are in the returned dictionary. Let's look at them.
print("\n--- Pipeline Run Summary ---")
print(f"Best model found: {results['best_key']}")
print(f"Reduced Chi-squared: {results['goodness_of_fit']['chi2_reduced']:.2f}")
print("Best-fit parameters (from highest-likelihood sample):")
print(results['best_params'])



# ------------------------------------------------------------------
# 3. Interactive Post-Fit Analysis (Optional)
# ------------------------------------------------------------------
# The main pipeline has already produced a full analysis. The following
# steps are useful for interactively assessing convergence and creating
# custom plots like a detailed corner plot.

sampler = results["sampler"]
best_p = results["best_params"]
param_names = results["param_names"]

# --- FIX: Detach the sampler from the now-closed pool ---
# By setting the pool to None, subsequent calls will run in serial mode.
sampler.pool = None 

# --- Interactively extend the chain until convergence ---
print("\n--- Interactive Chain Convergence Check ---")
max_extra_chunks, chunk_size = 2, 1000 
chunks_added = 0
while not quick_chain_check(sampler):
    if chunks_added >= max_extra_chunks:
        print(f"Reached max extra steps ({max_extra_chunks * chunk_size}); proceeding.")
        break
    print(f"\nChain not fully converged. Running for {chunk_size} more steps...")
    # This call will now work correctly
    sampler.run_mcmc(None, chunk_size, progress=True)
    chunks_added += 1

# --- Generate a high-quality corner plot with the final chain ---
print("\n--- Generating Final Corner Plot ---")
final_clean_samples = get_clean_samples(sampler, param_names, verbose=True)

#fig_corner = make_beautiful_corner(
#    final_clean_samples,
#    param_names,
#    best_params=best_p,
#    title=f"Posterior for {results['best_key']} ({final_clean_samples.shape[0]} samples)"
#)
#
## Save and display the final corner plot
#corner_path = data_path.with_name(f"{data_path.stem}_corner.pdf")
#fig_corner.savefig(corner_path, dpi=200, bbox_inches="tight")
#print(f"Saved corner plot to: {corner_path}")
#plt.show()
#
#fig_corner = make_beautiful_corner_wide(
#    final_clean_samples,
#    param_names,
#    best_params=best_p,
#    title=f"Posterior for {results['best_key']} ({final_clean_samples.shape[0]} samples)"
#)

# Save and display the final corner plot
#corner_path = data_path.with_name(f"{plot_dir}/{burst_name}_scat_corner.pdf")
#fig_corner.savefig(corner_path)
#print(f"Saved corner plot to: {corner_path}")
#plt.show()

# The 4-panel plot is now generated automatically by the pipeline's
# `create_sixteen_panel_plot` or `create_four_panel_plot` functions,
# so the manual plotting code from the old notebook is no longer needed here.

print("\nAnalysis complete.")

# %%
fig_corner = make_beautiful_corner(
    final_clean_samples,
    param_names,
    best_params=best_p,
    title=f"Posterior for {results['best_key']} ({final_clean_samples.shape[0]} samples)"
)

# Save and display the final corner plot
corner_path = os.path.join(f"{plot_dir}/{burst_name}_scat_corner.pdf")
fig_corner.savefig(corner_path)
print(f"Saved corner plot to: {corner_path}")
plt.show()

# The 4-panel plot is now generated automatically by the pipeline's
# `create_sixteen_panel_plot` or `create_four_panel_plot` functions,
# so the manual plotting code from the old notebook is no longer needed here.

print("\nAnalysis complete.")

# %%

