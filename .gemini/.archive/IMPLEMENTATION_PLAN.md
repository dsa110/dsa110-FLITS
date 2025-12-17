# FLITS Immediate Priorities Implementation Plan

**Status:** In Progress  
**Lead:** AI Assistant → Jakob Faber  
**Date:** 2025-12-13

---

## ✅ Priority 1: Integrate DM Estimation (STARTED)

### Completed:

- [x] Created `scattering/scat_analysis/dm_preprocessing.py` module
- [x] Wrapper functions: `estimate_dm_from_waterfall()` and `refine_dm_init()`
- [x] Bootstrap uncertainty estimation integrated
- [x] Robust error handling with fallback to catalog DM

### Next Steps:

1. **Integrate into pipeline** - Add DM refinement to `BurstPipeline.run_full()`
2. **Add CLI flag** - `--refine-dm` option in `run_scat_analysis.py`
3. **Test on one burst** - Validate on Casey (known scattering)
4. **Update batch configs** - Add `refine_dm: true` to YAML configs

**Implementation:**

```python
# In burstfit_pipeline.py, BurstPipeline.run_full()
if self.pipeline_kwargs.get('refine_dm', False):
    from .dm_preprocessing import refine_dm_init
    self.dm_init = refine_dm_init(
        self.dataset,
        catalog_dm=self.dm_init,
        enable_dm_estimation=True,
    )
    self.dataset.model.dm_init = self.dm_init
```

---

## 🔄 Priority 2: Complete Scattering Fits (9 Bursts Remaining)

### Burst Status Table:

| Burst      | CHIME Data | DSA Data | Scattering Fit     | Priority |
| ---------- | ---------- | -------- | ------------------ | -------- |
| Casey      | ✅         | ✅       | ✅ Done (τ=0.23ms) | -        |
| Freya      | ✅         | ✅       | ✅ Done (τ=3.52ms) | -        |
| Wilhelm    | ✅         | ✅       | ✅ Done (τ=2.82ms) | -        |
| Chromatica | ✅         | ✅       | ⏳ Pending         | HIGH     |
| Hamilton   | ✅         | ✅       | ⏳ Pending         | HIGH     |
| Isha       | ✅         | ✅       | ⏳ Pending         | MEDIUM   |
| JohnDoeII  | ✅         | ✅       | ⏳ Pending         | MEDIUM   |
| Mahi       | ✅         | ✅       | ⏳ Pending         | MEDIUM   |
| Oran       | ✅         | ✅       | ⏳ Pending         | MEDIUM   |
| Phineas    | ✅         | ✅       | ⏳ Pending         | MEDIUM   |
| Whitney    | ✅         | ✅       | ⏳ Pending         | MEDIUM   |
| Zach       | ✅         | ✅       | ⏳ Pending         | MEDIUM   |

**Data Files Confirmed:**

- All 12 bursts have CHIME `.npy` files (~131 MB each)
- All 12 bursts have DSA `.npy` files (~123 MB each)
- Configs exist in `batch_configs/{chime,dsa}/`

### Execution Plan:

**Option A: Batch Processing (Recommended)**

```bash
# Run all pending bursts with DM refinement
flits-batch run data/ \
    --output results/scattering_batch_1/ \
    --db flits_results.db \
    --steps 10000 \
    --nproc 8 \
    --scattering-only \
    --bursts chromatica,hamilton,isha,johndoeii,mahi,oran,phineas,whitney,zach
```

**Option B: Individual Testing**

```bash
# Test on one burst first (Hamilton = high priority, moderate DM)
python scattering/run_scat_analysis.py \
    data/chime/hamilton_chime_I_518_8007_32000b_cntr_bpc.npy \
    --config batch_configs/chime/hamilton_chime.yaml \
    --model-scan --plot --refine-dm
```

**Option C: Parallel Execution**

```bash
# Run 3 bursts in parallel (manual tmux sessions)
# Session 1: High DM bursts (Freya-like)
# Session 2: Medium DM bursts
# Session 3: Low DM bursts
```

### Estimated Timeline:

- Per burst: ~2-4 hours (10k MCMC steps, model scan)
- 9 bursts sequential: ~27-36 hours
- 9 bursts parallel (3×): ~9-12 hours
- Batch system with queue: ~24-30 hours (conservative)

---

## ⚙️ Priority 3: Fix MCMC Walker Initialization

### Current Issue:

Lines 394-433 in `burstfit.py` - walkers sometimes start outside valid region.

**Symptoms:**

- "Probability was NaN" errors
- Walkers stuck at prior boundaries
- Poor chain mixing (Gelman-Rubin R̂ > 1.1)

**Root Cause:**

```python
# Current walker initialization (burstfit.py:402)
frac = self.walker_width_frac  # Default: 0.01 (1% of prior range)
widths.append(frac * (upper[i] - lower[i]))
```

For narrow priors or log-space parameters, 1% can be too tight.

### Proposed Fix:

**Adaptive Walker Width:**

```python
def _init_walkers(self, p0, key: str, nwalk: int):
    # ... existing code ...

    for i, n in enumerate(names):
        # ... existing value extraction ...

        if self._is_log_param(n):
            lo = max(lower[i], 1e-30)
            hi = max(upper[i], lo * (1.0 + 1e-6))
            centre.append(np.log(max(val, 1e-30)))

            # ADAPTIVE: Use larger width for log-space
            log_range = np.log(hi) - np.log(lo)
            adaptive_frac = min(frac * 10, 0.1)  # 10× wider, capped at 10%
            widths.append(adaptive_frac * log_range)
        else:
            centre.append(val)
            widths.append(frac * (upper[i] - lower[i]))
```

**Per-Parameter Tuning:**

```yaml
# In sampler.yaml
walker_widths:
  c0: 0.05 # 5% for amplitude
  t0: 0.01 # 1% for time offset
  gamma: 0.02 # 2% for spectral index
  zeta: 0.05 # 5% for intrinsic width (log-space)
  tau_1ghz: 0.1 # 10% for scattering timescale (log-space)
  alpha: 0.01 # 1% for frequency exponent
  delta_dm: 0.02 # 2% for DM offset
```

### Implementation:

1. Test adaptive width on one failing burst
2. Add `walker_widths` to `SamplerConfig` class
3. Update `_init_walkers()` to use per-parameter widths
4. Add diagnostic: print median walker acceptance rate

---

## 📚 Priority 4: Documentation

### Phase 1: Immediate (Code-Level)

- [x] Created `LEAD_DEVELOPER_ONBOARDING.md`
- [ ] Add docstrings to `dm_preprocessing.py`
- [ ] Update `bursts.yaml` comments with DM estimation notes
- [ ] Create `CHANGELOG.md` for tracking changes

### Phase 2: User Documentation (Next Week)

- [ ] Create `docs/` directory with Sphinx setup
- [ ] Tutorial: "Running Your First Burst Analysis"
- [ ] How-to: "Interpreting MCMC Diagnostics"
- [ ] FAQ: Common errors and solutions

### Phase 3: Developer Documentation

- [ ] API reference (auto-generated from docstrings)
- [ ] Architecture diagrams (pipeline flow, data flow)
- [ ] Contributing guide

---

## 🎯 Execution Order (Next 48 Hours)

### Hour 0-2: Integration & Testing

- [x] Create `dm_preprocessing.py` module
- [ ] Integrate into `burstfit_pipeline.py`
- [ ] Add `--refine-dm` CLI flag
- [ ] Test on Casey (validate against known result)

### Hour 2-4: Single Burst Validation

- [ ] Run Hamilton with DM refinement
- [ ] Check convergence (R̂ < 1.1)
- [ ] Validate against catalog DM
- [ ] Generate diagnostic plots

### Hour 4-6: Batch Setup

- [ ] Update batch configs with `refine_dm: true`
- [ ] Test batch runner on 2 bursts
- [ ] Monitor resource usage (CPU, memory)

### Hour 6-24: Batch Execution (First Round)

- [ ] Run 4 bursts: Chromatica, Hamilton, Isha, JohnDoeII
- [ ] Monitor convergence in real-time
- [ ] Save intermediate results

### Hour 24-48: Batch Execution (Second Round)

- [ ] Run remaining 5 bursts
- [ ] Aggregate results in SQLite database
- [ ] Generate summary plots
- [ ] Export LaTeX table

---

## 📊 Success Criteria

### Minimum Viable:

- [x] DM estimation module created
- [ ] 3+ new bursts fitted with R̂ < 1.1
- [ ] Results added to database
- [ ] No crashes during batch run

### Target:

- [ ] All 9 bursts completed
- [ ] DM refinement reduces χ²_red by >5%
- [ ] Publication-ready corner plots
- [ ] LaTeX table exported

### Stretch:

- [ ] Joint τ-Δν analysis completed
- [ ] MCMC walker initialization robustness improved
- [ ] Documentation drafted

---

## 🐛 Risk Mitigation

**Risk:** MCMC fails to converge on some bursts  
**Mitigation:** Manual initial guess via interactive notebook

**Risk:** DM estimation produces outliers  
**Mitigation:** Catalog DM fallback built-in

**Risk:** Batch run crashes mid-execution  
**Mitigation:** Database saves after each burst; resume capability

**Risk:** Insufficient compute time  
**Mitigation:** Reduce MCMC steps to 5k for initial fits, extend later

---

**Next Action:** Integrate DM preprocessing into pipeline and test on Hamilton.
