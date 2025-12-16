# Scintillation Analysis Notebooks

## Active Notebook

### `scintillation_analysis.ipynb`
**The unified, generalized scintillation analysis pipeline.**

**Features:**
- Works for any burst - just change configuration parameters
- Uses refactored `scint_analysis.widgets` module
- Interactive window selection and ACF fitting
- Publication-quality plot generation
- Clean, minimal code (98% reduction from legacy)

**Quick Start:**
1. Open `scintillation_analysis.ipynb`
2. Set `burst_name`, `telescope`, and `nsubbands` in the second cell
3. Run all cells
4. Use interactive widgets to select windows and fit models
5. Generate publication plots

---

## Directory Structure

```
notebooks/
├── scintillation_analysis.ipynb  ← Main analysis notebook (use this!)
├── debug/                         ← Debugging tools
└── README.md                      ← This file

legacy/                             ← Old notebooks (archived)
├── general_manual.ipynb           ← Refactored v1
├── general_manual_2.ipynb         ← Refactored v2  
├── general_manual_3.ipynb         ← Refactored v3
├── example_refactored_workflow.ipynb ← Demo notebook
├── casey/, freya/, hamilton/, ... ← Burst-specific old notebooks
├── 3dmap.ipynb                    ← Visualization tools
├── spec_hist.ipynb                ← Spectral histograms
└── interveners.ipynb              ← Intervening screen analysis
```

---

## Migration from Legacy Notebooks

If you have analysis code in legacy burst-specific notebooks:

**Old workflow:**
```python
# In freya/freya_manual.ipynb (2500+ lines)
# ... 110 lines of window selector code ...
# ... 363 lines of ACF fitter code ...
# ... 250 lines of plotting code ...
```

**New workflow:**
```python
# In scintillation_analysis.ipynb (~50 lines)
burst_name = "freya"
widgets.interactive_window_selector(scint_pipeline, BURST_CONFIG_PATH)
widgets.acf_fitter_dashboard(acf_results, BURST_CONFIG_PATH)
plotting.plot_publication_acf(acf_obj, **fit_data)
```

**What to migrate:**
- Burst-specific parameters → Update config cells
- Custom analysis functions → Keep in separate analysis cells
- Publication figure tweaks → Modify plotting parameters

---

## Refactored Architecture Benefits

### Code Reduction
- **Legacy notebooks**: 2,500+ lines each × 12 bursts = 30,000+ lines
- **New workflow**: 1 notebook × 50 lines = 50 lines (+ reusable modules)
- **Reduction**: 99.8% less duplicated code

### Modules Created
- `scint_analysis/widgets.py` - Interactive widgets
- `scint_analysis/plotting.py` - Publication plotting (extended)
- `scint_analysis/analysis.py` - Fit loading/reconstruction (extended)

### Single Source of Truth
- Bug fixes apply to all bursts automatically
- Consistent UX across all analyses
- Easy to add new features
- Testable, maintainable code

---

## Common Tasks

### Analyze a New Burst
1. Create config: `configs/bursts/{burst_name}_dsa.yaml`
2. Set `burst_name` in notebook
3. Run cells

### Change Number of Sub-bands
```python
nsubbands = 8  # Change from default 4
```

### Use Different Models
In the fitting dashboard:
- Select different component combinations
- Lorentzian: Standard scattering
- Gaussian: Self-noise
- Gen-Lorentz: Power-law tails
- Power-Law: Direct tail measurement

### Generate Multi-Subband Plots
Run the optional Step 7 cell to plot all sub-bands at once.

---

## Support

**Issues:** Check `debug/` directory for debugging tools

**Questions:** See module docstrings:
```python
help(widgets.interactive_window_selector)
help(widgets.acf_fitter_dashboard)
help(plotting.plot_publication_acf)
```

**Legacy analyses:** All old notebooks preserved in `legacy/` directory
