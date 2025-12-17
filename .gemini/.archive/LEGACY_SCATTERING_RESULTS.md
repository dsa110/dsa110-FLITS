# FLITS Scattering Analysis - Legacy Results Archive

**Purpose:** Preserve historical scattering fit results before re-running with integrated pipeline

**Date Archived:** 2025-12-13  
**Reason:** Ensure congruent analysis across all 12 bursts with new DM refinement integration

---

## Legacy Scattering Results

These results were obtained from previous analyses (found in `.archive/scattering/` notebooks).
They are preserved here for reference and comparison purposes.

### Casey (DSA-110)

- **Catalog DM:** 491.207 pc/cm³
- **Scattering timescale (τ₁GHz):** 0.227 ms
- **Frequency exponent (α):** 3.9
- **Intrinsic width (ζ):** 1.5 ms
- **Source:** `casey_dsa_new.ipynb`
- **Analysis Date:** ~2024 (exact date TBD from git history)

### Freya (CHIME)

- **Catalog DM:** 912.4 pc/cm³
- **Scattering timescale (τ₁GHz):** 3.515 ms
- **Frequency exponent (α):** 4.2
- **Intrinsic width (ζ):** 0.8 ms
- **Source:** `freya_chime_new.ipynb`, `freya_chime_new.py`
- **Analysis Date:** ~2024

### Wilhelm (CHIME)

- **Catalog DM:** 602.346 pc/cm³
- **Scattering timescale (τ₁GHz):** 2.818 ms
- **Frequency exponent (α):** 4.1
- **Intrinsic width (ζ):** 1.2 ms
- **Source:** `wilhelm_chime_new.ipynb`
- **Analysis Date:** ~2024

---

## Re-Analysis Plan (2025-12-13)

### Why Re-analyze?

1. **Congruent methodology** - All bursts analyzed with same pipeline version
2. **DM refinement** - Leverage new phase-coherence DM estimation
3. **Updated MCMC** - Current burstfit.py with all bug fixes
4. **Reproducibility** - Single workflow for entire sample
5. **Database integration** - All results in SQLite for easy querying

### New Analysis Workflow

1. Load data from `.npy` files
2. **NEW:** Refine DM using phase-coherence method
3. Run model selection (M0 → M1 → M2 → M3)
4. MCMC sampling (10k steps, convergence checks)
5. Diagnostics (sub-band, leave-one-out, ACF)
6. Store in `flits_results.db`

### Expected Differences from Legacy

- **DM values:** May differ slightly due to phase-coherence refinement
- **Scattering parameters:** Should be broadly consistent, but:
  - Better uncertainty estimates (bootstrap + MCMC)
  - Gelman-Rubin convergence guarantees (R̂ < 1.1)
  - Full diagnostic suite run
- **Model selection:** Systematic BIC comparison for all bursts

---

## Validation Strategy

Once new analysis completes, compare:

| Burst   | Legacy τ₁GHz (ms) | New τ₁GHz (ms) | Δτ (%) | Legacy α | New α | Δα  |
| ------- | ----------------- | -------------- | ------ | -------- | ----- | --- |
| Casey   | 0.227             | TBD            | TBD    | 3.9      | TBD   | TBD |
| Freya   | 3.515             | TBD            | TBD    | 4.2      | TBD   | TBD |
| Wilhelm | 2.818             | TBD            | TBD    | 4.1      | TBD   | TBD |

**Acceptance criteria:**

- Parameter values should agree within ~20% (different MCMC runs, DM refinement)
- If discrepancies > 50%, investigate:
  - DM offset impact
  - MCMC convergence
  - Model selection (did best model change?)

---

## Archived File Locations

**Legacy notebooks:**

- `.archive/scattering/casey_dsa_new.ipynb`
- `.archive/scattering/freya_chime_new.ipynb` and `.py`
- `.archive/scattering/wilhelm_chime_new.ipynb`
- `.archive/scattering/wilhelm_dsa_new.ipynb`

**Legacy config files:**
Original configs (if different from current batch_configs) should be in `.archive/`

**Legacy plots:**
Check `data/{chime,dsa}/` directories for PDFs and PNGs with timestamps

---

## Notes

- Legacy results are **scientifically valid** - they were published/used previously
- Re-analysis is for **consistency and reproducibility** across the sample
- Results will be cross-validated and documented in paper
- If new results differ significantly, we'll document why (DM refinement, updated physics, etc.)

---

**Archived by:** AI Assistant  
**Date:** 2025-12-13  
**Status:** Ready for full sample re-analysis
