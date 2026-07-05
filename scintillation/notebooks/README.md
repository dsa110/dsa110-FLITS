# Scintillation Analysis Notes

Checked-in scintillation notebooks were removed during the generated-artifact
cleanup. Keep exploratory notebooks and rendered outputs outside git; the
maintained entrypoint is the package CLI plus reusable modules.

**Features:**
- Works for any burst - just change configuration parameters
- Uses refactored `scint_analysis.widgets` module
- Interactive window selection and ACF fitting
- Publication-quality plot generation
- Clean, minimal code (98% reduction from legacy)

**Quick Start:**

```bash
flits-scint scintillation/configs/bursts/freya_dsa.yaml
```

---

## Directory Structure

This directory is now documentation-only unless a small, intentional fixture is
added later.

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

**New workflow:** use `flits-scint` for batchable runs and import
`scintillation.scint_analysis.widgets` only in local notebooks that stay
untracked.

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
