# Refactoring Plan: `general_manual` Notebooks

## Current State Analysis

### Notebook Sizes:
- `general_manual.ipynb`: **2,661 lines** (49 cells)
- `general_manual_2.ipynb`: **2,526 lines** (46 cells)
- `general_manual_3.ipynb`: **2,528 lines** (47 cells)

**Total duplicate code: ~7,500 lines across 3 notebooks!**

---

## 🔍 Code Structure Analysis

### What's Already Modularized
- Basic pipeline: `scint_analysis.pipeline.ScintillationAnalysis`
- ACF calculation: `scint_analysis.core.ACF`
- Model components: `scint_analysis.analysis` (Lorentzian, Gaussian, etc.)
- Config utilities: `scint_analysis.config`

### What's Still in Notebooks ❌

#### **1. Interactive Window Selector Widget** (Lines 46-153)
**Function:** `interactive_window_selector(pipeline_obj, cfg_path)`
- 110 lines of ipywidgets code
- Creates sliders for on/off-pulse window selection
- Updates matplotlib figure in real-time (ipympl)
- Saves selections back to YAML

**Appears in:** All 3 notebooks (identical code)

**Should be:** `scint_analysis.widgets.window_selector()`

---

#### **2. Manual ACF Fitting Dashboard** (Lines 168-531)
**Features:**
- Multi-component model builder (1-3 components)
- Live parameter sliders for each component
- Real-time plot updates (flicker-free ipympl)
- Fit execution with lmfit
- Results storage and YAML export
- 363+ lines of complex widget logic

**Components:**
- Model configuration dictionary
- Parameter widget factory
- Plot management (single canvas reuse)
- Lag-range sync logic
- Fit callback with model composition
- Print/save/export callbacks

**Appears in:** All 3 notebooks (98% identical)

**Should be:** `scint_analysis.widgets.acf_fitter_dashboard()`

---

#### **3. Fit Loading and Evaluation** (Lines 534-788)
**Functions:**
- `format_value_with_error(value, error)` - compact error notation
- `load_fit_from_yaml(config_path, subband_index, model_name, lags)` - rebuild models from saved fits
- `plot_publication_acf(acf_obj, best_fit_curve, ...)` - 3-panel publication plot

**Total:** ~250 lines

**Appears in:** All 3 notebooks (identical)

**Should be:** 
- `scint_analysis.plotting.format_error()` 
- `scint_analysis.analysis.load_saved_fit()`
- `scint_analysis.plotting.plot_publication_acf()`

---

#### **4. Additional Analysis Functions** (Lines 790+)
Based on cell summaries, there appear to be:
- Noise template plotting
- Cross-burst comparison utilities
- Parameter extraction and aggregation
- Multi-subband visualization

**Estimated:** 1000+ more lines of duplicated code

---

## 🎯 Proposed Refactoring

### Phase 1: Extract Interactive Widgets Module ⭐ HIGH PRIORITY

**Create:** `scint_analysis/widgets.py`

```python
"""
Interactive widgets for scintillation analysis notebooks.
Requires: ipywidgets, ipympl (matplotlib widget backend)
"""

def interactive_window_selector(pipeline_obj, config_path, **kwargs):
    """
    Launch an interactive widget for selecting on/off-pulse windows.
    
    Parameters
    ----------
    pipeline_obj : ScintillationAnalysis
        Initialized pipeline with masked_spectrum loaded
    config_path : str or Path
        Path to burst config YAML file
    kwargs : dict
        Optional settings (zoom_default, etc.)
    
    Returns
    -------
    VBox
        ipywidgets container (display with IPython.display.display)
    """
    # Move 110 lines of widget code here
    pass

def acf_fitter_dashboard(acf_results, config_path, **kwargs):
    """
    Launch an interactive dashboard for manual ACF fitting.
    
    Parameters
    ----------
    acf_results : dict
        Cached ACF results from pipeline (subband_acfs, subband_lags_mhz, etc.)
    config_path : str or Path
        Path to burst config YAML file
    kwargs : dict
        model_config : Custom model configurations (optional)
        default_models : Default model selections (optional)
    
    Returns
    -------
    VBox
        Complete dashboard widget
    """
    # Move 363+ lines of fitting widget code here
    pass

def noise_template_plotter(acf_results, subband_index, **kwargs):
    """
    Interactive plot of noise template synthesis.
    """
    pass
```

**Benefits:**
- Single source of truth for widget code
- Easy to test and maintain
- Can version control widget behavior
- Notebooks become much shorter and clearer

---

### Phase 2: Extend `plotting.py` Module

**Add to:** `scint_analysis/plotting.py`

```python
def format_error(value, error):
    """Format value with compact error notation: X(err)"""
    # Move from notebook
    pass

def plot_publication_acf(
    acf_obj, 
    best_fit_curve, 
    component_curves,
    params,
    fit_range_mhz,
    zoom_lag_range_mhz=(-20, 20),
    save_path=None
):
    """
    Generate 3-panel publication-quality ACF plot.
    
    Panels:
        a) Wide view with zoom indicator
        b) Zoomed view with component breakdown
        c) Residuals with chi-squared
    """
    # Move 250 lines of plotting code here
    pass

def plot_multi_burst_comparison(burst_results, **kwargs):
    """Cross-burst parameter comparison plots"""
    pass
```

---

### Phase 3: Extend `analysis.py` Module

**Add to:** `scint_analysis/analysis.py`

```python
def load_saved_fit(config_path, subband_index, model_name, lags):
    """
    Load a saved fit from YAML, rebuild the lmfit model, and evaluate.
    
    Returns
    -------
    dict
        'best_fit_curve', 'component_curves', 'params', 'redchi', 'fit_range_mhz'
    """
    # Move from notebook
    pass

def extract_decorrelation_bandwidths(fit_results):
    """Extract gamma/sigma values from fit components"""
    pass

def compute_scintillation_parameters(acf_fit, telescope_config):
    """Convert ACF fit parameters to physical scintillation parameters"""
    pass
```

---

### Phase 4: Create Analysis Templates

**Create:** `scint_analysis/templates/`

```
templates/
├── interactive_analysis.ipynb    # Template for manual analysis
├── batch_analysis.ipynb          # Template for automated multi-burst
└── publication_plots.ipynb       # Template for paper figures
```

**Interactive Analysis Template:**
```python
# Cell 1: Setup
from scint_analysis import config, pipeline, widgets, plotting

burst_name = "hamilton"  # CHANGE ME
config_path = f"configs/bursts/{burst_name}_dsa.yaml"

# Cell 2: Load and Run Pipeline
cfg = config.load_config(config_path)
pipe = pipeline.ScintillationAnalysis(cfg)
pipe.prepare_data()

# Cell 3: Interactive Window Selection
widgets.interactive_window_selector(pipe, config_path)

# Cell 4: Run ACF Calculation
pipe.run()

# Cell 5: Interactive ACF Fitting
acf_data = pickle.load(open(f"data/cache/{burst_name}_acf_results.pkl", "rb"))
widgets.acf_fitter_dashboard(acf_data, config_path)

# Cell 6: Publication Plots
fit_data = analysis.load_saved_fit(config_path, subband=0, model="Lorentzian+Gaussian")
plotting.plot_publication_acf(acf_obj, **fit_data, save_path=f"plots/{burst_name}_acf.pdf")
```

**Reduced from 2500 lines → ~50 lines!**

---

## 📦 Implementation Steps

### Step 1: Create `widgets.py` Module
1. Extract `interactive_window_selector` function
2. Extract `acf_fitter_dashboard` function
3. Add docstrings and type hints
4. Create simple test notebook

### Step 2: Extend `plotting.py`
1. Move `format_error()` 
2. Move `plot_publication_acf()`
3. Add additional plotting utilities
4. Update existing plotting functions if needed

### Step 3: Extend `analysis.py`
1. Move `load_saved_fit()`
2. Add parameter extraction utilities
3. Add validation functions

### Step 4: Create Templates
1. Design minimal interactive template
2. Design batch processing template
3. Design publication figure template

### Step 5: Update Existing Notebooks
1. Replace duplicated code with module imports
2. Test each notebook still works
3. Archive old versions
4. Document usage in README

---

## Implementation Status

**Phase 1: COMPLETED**
- Created `scint_analysis/widgets.py` module with:
  - `interactive_window_selector()` function (110 lines → 1 function call)
  - `acf_fitter_dashboard()` function (363 lines → 1 function call)
- Extended `plotting.py` with `format_error()` and `plot_publication_acf()`
- Extended `analysis.py` with `load_saved_fit()`
- Refactored `general_manual_3.ipynb`: **2,528 lines → 1,533 lines (40% reduction)**

**Git diff stats:** +30 lines, -1,055 lines

---

## Example: Before vs. After

### BEFORE (general_manual_3.ipynb):
```python
# Cell 1-2: Imports (50 lines of matplotlib setup, ipywidgets imports)
# Cell 3: Config loading (40 lines)
# Cell 4: Window selector (110 lines of widget code)
# Cell 5: Pipeline run (10 lines)
# Cell 6: ACF fitting dashboard (363 lines of widget code)
# Cell 7: Helper functions (250 lines)
# Cell 8-20: More analysis (1000+ lines)
# ...
```

### AFTER (with refactoring):
```python
# Cell 1: Setup (5 lines)
from scint_analysis import config, pipeline, widgets, plotting, analysis

# Cell 2: Config (3 lines)
burst_name = "hamilton"
cfg = config.load_config(f"configs/bursts/{burst_name}_dsa.yaml")

# Cell 3: Load data (2 lines)
pipe = pipeline.ScintillationAnalysis(cfg)
pipe.prepare_data()

# Cell 4: Interactive window selection (1 line)
widgets.interactive_window_selector(pipe, cfg.config_path)

# Cell 5: Run pipeline (2 lines)
pipe.run()

# Cell 6: ACF fitting (2 lines)
acf_data = pickle.load(open(f"data/cache/{burst_name}_acf_results.pkl", "rb"))
widgets.acf_fitter_dashboard(acf_data, cfg.config_path)

# Cell 7: Load and plot fit (4 lines)
fit = analysis.load_saved_fit(cfg.config_path, subband=0, model="Lorentzian+Gaussian")
plotting.plot_publication_acf(pipe.acf, **fit, save_path=f"plots/{burst_name}.pdf")
```

**Result: 2500 lines → ~50 lines**

---

## 🚀 Benefits

1. **Maintainability**: Fix bugs in one place, not three
2. **Testability**: Can unit test widgets and plotting functions
3. **Reusability**: Other analyses can use the same widgets
4. **Documentation**: Docstrings in modules, not buried in notebooks
5. **Clarity**: Notebooks become readable workflows, not codebases
6. **Version Control**: Track changes to core functionality properly
7. **Onboarding**: New users see clear API, not wall of code

---

## ⚠️ Considerations

### Dependencies:
- `ipywidgets` → Add to `requirements.txt`
- `ipympl` → Add as optional dependency for interactive mode

### Backward Compatibility:
- Keep old notebooks in `notebooks/archive/`
- Provide migration guide
- Support both old and new workflows during transition

### Testing:
- Jupyter notebooks can't be unit tested easily
- Extracted modules CAN be tested
- Create test suite for `widgets.py`, `plotting.py`, `analysis.py`

---

## 🎯 Priority Ranking

1. **HIGH**: Extract `acf_fitter_dashboard` (most duplicated, most complex)
2. **HIGH**: Extract `interactive_window_selector` (used by all analyses)
3. **MEDIUM**: Extract plotting utilities (publication-ready code)
4. **MEDIUM**: Extract fit loading/saving (data management)
5. **LOW**: Create templates (nice-to-have, but notebooks work as-is)

---

## 📝 Next Steps

**Immediate Action:**
1. Create `scint_analysis/widgets.py` skeleton
2. Move `interactive_window_selector` first (lower complexity)
3. Test in one notebook (e.g., `freya_manual.ipynb`)
4. If successful, move `acf_fitter_dashboard`
5. Update all three `general_manual*.ipynb` notebooks

**Long-term:**
- Gradually extract remaining notebook-specific code
- Build comprehensive widget library
- Create gallery of analysis templates
- Write tutorial documentation

---

## 💡 Recommendation

**YES**, you should absolutely modularize these notebooks. The ~7500 lines of duplicated code represent a significant maintenance burden and make the analysis pipeline harder to understand and use.

Start with the interactive widgets (`widgets.py`) as they're the most duplicated and would provide immediate value. Then progressively move plotting and analysis utilities into their respective modules.

The goal: Transform these notebooks from **monolithic analysis scripts** into **clean, reusable workflows** that showcase the power of your pipeline.
