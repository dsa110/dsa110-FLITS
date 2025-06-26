# Set path for module imports
import sys
# replace the path below with the absolute path to your `scattering/` folder
pkg_root = "/arc/home/jfaber/baseband_morphologies/chime_dsa_codetections/FLITS/scintillation"
sys.path.insert(0, pkg_root)

#%load_ext autoreload
#%autoreload 2

# Run this cell to import necessary libraries
import json
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt

# --- Bokeh Imports for Jupyter (Corrected) ---
from bokeh.io import output_notebook, show
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, Slider, Div
from bokeh.plotting import figure

# --- Your Pipeline's Imports ---
# Make sure your scint_analysis package is importable
# (You may need to add its path using sys.path.insert)
try:
    from scint_analysis import config, pipeline, plotting
    from scint_analysis.analysis import lorentzian_model_3_comp
    from scint_analysis.core import ACF
except ImportError as e:
    logging.error(f"Could not import scint_analysis. Make sure it's in your Python path. {e}")

## --- 1. Configuration ---
## Set up basic logging to see pipeline output in the notebook
#logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
#
## Define the path to the configuration file for the analysis run
#BURST_CONFIG_PATH = '/arc/home/jfaber/baseband_morphologies/chime_dsa_codetections/FLITS/scintillation/configs/bursts/casey_dsa.yaml'
#
## Load the merged configuration from the YAML files
#try:
#    analysis_config = config.load_config(BURST_CONFIG_PATH)
#    print("--- Loaded Configuration ---")
#    print(json.dumps(analysis_config, indent=2))
#except Exception as e:
#    logging.error(f"Failed to load configuration: {e}")
#    # Stop execution if config fails
#    raise
#
## --- 2. Initialize and Run the Pipeline ---
#print("\n--- Initializing and Running Scintillation Pipeline ---")
## Create an instance of the main pipeline controller
#scint_pipeline = pipeline.ScintillationAnalysis(analysis_config)
#
## This single .run() call executes all the steps in the correct order:
## - Loads and masks data
## - Characterizes off-pulse noise (NEW)
## - Calculates ACFs using the noise model for normalization
#scint_pipeline.run()
#print("--- Pipeline Execution Finished ---")

# Run this cell to load the data for one sub-band
SUBBAND_INDEX = 0 # Choose which sub-band to analyze (0, 1, 2...)
ACF_RESULTS_PATH = '/arc/home/jfaber/baseband_morphologies/chime_dsa_codetections/FLITS/scintillation/data/cache/casey/casey_acf_results.pkl' # Adjust path if needed

try:
    with open(ACF_RESULTS_PATH, 'rb') as f:
        acf_results = pickle.load(f)
    logging.info(f"Successfully loaded ACF results from {ACF_RESULTS_PATH}")
except FileNotFoundError:
    logging.error(f"ERROR: ACF results file not found at {ACF_RESULTS_PATH}. Please run the main pipeline first.")
    
# Create a clean ACF object for the selected sub-band
lags = acf_results['subband_lags_mhz'][SUBBAND_INDEX]
data = acf_results['subband_acfs'][SUBBAND_INDEX]
# Use the robustly calculated errors for chi-squared calculation
errors = np.sqrt(acf_results['subband_acfs_err'][SUBBAND_INDEX]**2) if 'subband_acfs_err' in acf_results else np.ones_like(data)

acf_obj = ACF(acf_data=data, lags_mhz=lags, acf_err=errors)

# Run this cell to define the application logic
#def make_document(doc):
#    """
#    This function is a self-contained Bokeh application.
#    It takes a Bokeh document `doc` and adds the interactive plot to it.
#    """
#    # --- Define Model and Parameters for Sliders ---
model_func = lorentzian_model_3_comp
param_names = ['gamma1', 'm1', 'gamma2', 'm2', 'gamma3', 'm3', 'c3']

# Use initial guesses from your pipeline config as the starting point
p0 =           [0.05,  0.5,  0.2,   0.4,  0.8,   0.3,  0.0] 
lower_bounds = [1e-4,  0,    1e-4,  0,    1e-4,  0,    -0.2]
upper_bounds = [1.0,   1.5,  2.0,   1.5,  5.0,   1.5,  0.2]

# --- Prepare Data Sources ---
x_lags = acf_obj.lags
y_data = acf_obj.acf
y_model_init = model_func(x_lags, *p0)

source_data = ColumnDataSource(data=dict(x=x_lags, y=y_data))
source_model = ColumnDataSource(data=dict(x=x_lags, y=y_model_init))
source_resid = ColumnDataSource(data=dict(x=x_lags, y=(y_data - y_model_init)))

# --- Set up Plots ---
plot = figure(height=400, width=700, title=f"Interactive ACF Fit (Sub-band {SUBBAND_INDEX})", x_axis_label="Frequency Lag (MHz)")
plot.circle('x', 'y', source=source_data, legend_label="ACF Data", color="navy", alpha=0.6)
plot.line('x', 'y', source=source_model, legend_label="Interactive Model", color="crimson", line_width=2)
plot.legend.location = "top_right"

plot_residual = figure(height=200, width=700, title="Residuals", x_range=plot.x_range, x_axis_label="Frequency Lag (MHz)")
plot_residual.line('x', 'y', source=source_resid, line_color="black")
plot_residual.line(x_lags, np.zeros_like(x_lags), line_dash='dashed', color='gray')

# --- Set up Widgets ---
sliders = [Slider(title=name, value=val, start=low, end=high, step=(high-low)/200)
           for name, val, low, high in zip(param_names, p0, lower_bounds, upper_bounds)]
gof_div = Div(text="Reduced Chi-Squared: N/A", width=300, style={'font-size': '1.1em', 'font-weight': 'bold'})

# --- Define the Callback Function ---
def update_fit(attr, old, new):
    p = [s.value for s in sliders]
    new_y_model = model_func(x_lags, *p)
    source_model.data['y'] = new_y_model

    new_resid = source_data.data['y'] - new_y_model
    source_resid.data['y'] = new_resid

    err = np.maximum(acf_obj.err, 1e-9)
    chisqr = np.sum((new_resid / err)**2)
    dof = len(y_data) - len(p)
    if dof > 0:
        redchi = chisqr / dof
        gof_div.text = f"<b>Reduced Chi-Squared: {redchi:.3f}</b>"

for w in sliders:
    w.on_change('value', update_fit)

# Trigger the initial GoF calculation
update_fit(None, None, None)

# --- Assemble Layout and Add to Document ---
inputs = column(sliders)
plots = column(plot, plot_residual)
layout = row(inputs, plots, gof_div)

doc.add_root(layout)
doc.title = "ACF Explorer"
    
# Use curdoc() for standalone apps instead of doc.add_root()
curdoc().add_root(layout)
curdoc().title = "ACF Explorer"
    
## Run this cell to display the interactive plot below
#
## This tells Bokeh to generate output in the notebook
#output_notebook()
#
## This runs make_document as an application and embeds it
#show(make_document)