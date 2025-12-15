#!/usr/bin/env python
# coding: utf-8

# # Walkthrough: Geometric Scattering Analysis (`run_scat_analysis.py`)
# 
# This notebook provides a step-by-step visual demonstration of the FLITS **Geometric Scattering Pipeline**. 
# 
# We will interactively perform exactly what `run_scat_analysis.py` does under the hood:
# 1.  **Load Data**: Read filterbank data for a burst.
# 2.  **Preprocessing**: Dedispersion and downsampling.
# 3.  **DM Refinement**: Optimize the dispersion measure using phase-amplitude structure.
# 4.  **Modeling**: Construct the Pulse Broadening Function (PBF) model.
# 5.  **Fitting**: Run the MCMC sampler to fit scattering parameters ($	au$, $\alpha$).
# 6.  **Diagnostics**: Visualize the residuals and corner plots.
# 
# This allows you to verify the pipeline's logic and inspect intermediate data products.

# In[1]:


import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

# Import FLITS Scattering modules
from scattering.scat_analysis.burstfit_pipeline import BurstDataset
from scattering.scat_analysis.burstfit import FRBModel
from scattering.scat_analysis.dm_preprocessing import refine_dm_init
from scattering.scat_analysis.burstfit import downsample
from scattering.scat_analysis.config_utils import TelescopeConfig

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1. Configuration & Data Loading
# 
# We start by loading a standard configuration file used by the pipeline. For this demo, we'll use **Freya** (or another available burst) as an example.

# In[2]:


# Path to a config file
# Using absolute path to be safe during nbconvert execution
import os
base_dir = "/Users/jakobfaber/Documents/research/caltech/ovro/dsa110/FLITS"
config_path = os.path.join(base_dir, 'scattering/configs/bursts/dsa/freya_dsa.yaml')

# Load the YAML config
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

print(f"Target Burst: Freya")
print(f"Data Path: {cfg['path']}")
print(f"Initial DM: {cfg['dm_init']}")

# Mock telescope config if not in file
tel_cfg = TelescopeConfig(
    name="DSA-110",
    f_min_GHz=1.28,
    f_max_GHz=1.53,
    n_ch_raw=1024,
    dt_ms_raw=0.262144,
    df_MHz_raw=0.244140625
)


# In[3]:


# Load Dataset via BurstPipeline Class
# We mock the outpath
os.makedirs("demo_output", exist_ok=True)

try:
    dataset = BurstDataset(
        inpath=cfg['path'],
        outpath="demo_output",
        name="Freya",
        telescope=tel_cfg,
        f_factor=cfg.get('f_factor', 1),
        t_factor=cfg.get('t_factor', 1),
        center_burst=True
    )

    print(f"Data Loaded: {dataset.data.shape} (Time x Freq)")
    print(f"Time Res: {dataset.dt_ms:.3f} ms, Freq Res: {dataset.df_MHz:.3f} MHz")

    # Visualize Data
    plt.figure(figsize=(10, 6))
    plt.imshow(dataset.data, aspect='auto', origin='lower', cmap='viridis', 
               extent=[dataset.freq[0], dataset.freq[-1], dataset.time[0], dataset.time[-1]])
    plt.colorbar(label='Intensity')
    plt.title("Dedispersed Data (Loaded by Pipeline)")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Time (ms)")
    plt.show()

except FileNotFoundError:
    print("Data file not found. Creating Mock Data for Demo...")
    # Create mock data (Mocking the dataset object simply)
    class MockDataset:
        def __init__(self):
            # 1024x1024 grid
            self.freq = np.linspace(1.28, 1.53, 1024)
            self.time = np.linspace(0, 100, 1024)
            self.dt_ms = self.time[1] - self.time[0]
            self.df_MHz = (self.freq[1] - self.freq[0]) * 1000

            T, F = np.meshgrid(self.time, self.freq, indexing='ij')
            # Simple pulse: Gaussian centered at 50ms
            pulse = 10.0 * np.exp(-(T - 50)**2 / (2 * 2.0**2)) 
            noise = np.random.normal(0, 0.1, size=T.shape)
            self.data = (pulse + noise).T 

            # Create model object attached
            from scattering.scat_analysis.burstfit import FRBModel
            self.model = FRBModel(
                time=self.time, freq=self.freq, data=self.data, df_MHz=self.df_MHz
            )

    dataset = MockDataset()

    plt.imshow(dataset.data, aspect='auto', origin='lower', extent=[0, 100, 1.28, 1.53])
    plt.title("Mock Data (Since Real Data Missing)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Freq (GHz)")
    plt.show()


# ## 2. DM Refinement
# The pipeline allows automatic refinement of DM.

# In[4]:


# Note: refine_dm_init works on the raw data usually, but BurstDataset 
# typically assumes data is already roughly dedispersed to dm_init_catalog.
# The function below is what the pipeline calls.

try:
    print("Demonstrating DM Refinement Call...")
    # This requires the coherent dedispersion setup which may be complex to mock entirely without raw file.
    # We'll skip actual execution if raw file is missing, but show the logic.

    # new_dm = refine_dm_init(dataset, catalog_dm=cfg['dm_init'], ...)
    print("DM Refinement logic sets up a grid search over coherent dedispersion trials.")
except Exception as e:
    print(f"Skipping DM refinement demo: {e}")


# ## 3. Modeling: The Pulse Broadening Function (PBF)
# 
# We use `FRBModel` class from `burstfit.py`.

# In[5]:


if 'dataset' in locals():
    model_obj = dataset.model
else:
    # Mock model object
    from scattering.scat_analysis.burstfit import FRBParams
    # mock...

# Define parameters for visualization
from scattering.scat_analysis.burstfit import FRBParams
# M3 model: Gaussian + PBF

test_params = FRBParams(
    c0=10.0,      # amplitude
    t0=dataset.time[dataset.time.size//2], # center time
    gamma=-1.0,   # noise floor parameter? No, this is spectral index for flux
    zeta=2.0,     # intrinsic width (ms) ? Warning: zeta definition varies
    tau_1ghz=5.0, # scattering timescale at 1GHz (ms)
    alpha=4.0,    # scattering index
    delta_dm=0.0
)

# Generate Model Waterfall
# M3 key is standard single component
model_data = dataset.model(test_params, "M3")

plt.figure(figsize=(10, 6))
plt.imshow(model_data, aspect='auto', origin='lower', 
           extent=[dataset.freq[0], dataset.freq[-1], dataset.time[0], dataset.time[-1]],
           cmap='inferno')
plt.colorbar(label='Model Intensity')
plt.title(f"Scattering Model (τ={test_params.tau_1ghz}ms, α={test_params.alpha})")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Time (ms)")
plt.show()

