# 🚀 READY TO TEST: Freya Burst

**Date:** 2025-12-13  
**Status:** READY FOR EXECUTION

---

## ✅ Integration Complete

All code integrated and tested:

- ✅ DM preprocessing module (`dm_preprocessing.py`)
- ✅ Pipeline integration (`burstfit_pipeline.py`)
- ✅ CLI flags (`run_scat_analysis.py`)
- ✅ All 24 batch configs updated with catalog DM values

---

## 🎯 Test Command

```bash
cd /Users/jakobfaber/Documents/research/caltech/ovro/dsa110/FLITS

python scattering/run_scat_analysis.py \
    data/chime/freya_chime_I_912_4067_32000b_cntr_bpc.npy \
    --config batch_configs/chime/freya_chime.yaml \
    --refine-dm \
    --model-scan \
    --plot \
    --steps 10000
```

**Expected Runtime:** 2-4 hours

---

## 📋 Why Freya?

**Freya is the IDEAL validation target:**

1. **Single-Component Structure**

   - Clean exponential scattering tail
   - No multi-component degeneracies (unlike Hamilton)
   - DM optimization is unambiguous

2. **Known Ground Truth**

   - Legacy result: τ = 3.515ms, α = 4.2
   - Direct comparison validates the new pipeline
   - Success = reproducing known physics

3. **High SNR**
   - Clear burst structure
   - Visual validation is straightforward
   - Residuals should be obviously Gaussian

---

## ✓ Validation Checklist

After the run completes, check (see `.gemini/VALIDATION_PROTOCOL.md`):

1. **DM Refinement**

   - Refined DM should be close to catalog (912.4 pc/cm³)
   - Offset < 1 pc/cm³ expected

2. **Scattering Parameters**

   - τ ≈ 3.5ms (compare to legacy: 3.515ms)
   - α ≈ 4.2 (compare to legacy: 4.2)
   - Agreement within ~10% validates pipeline

3. **Visual Inspection**

   - Open `data/chime/freya_chime_..._four_panel.pdf`
   - Residual panel should show pure noise (no diagonal stripes)

4. **MCMC Convergence**
   - All Gelman-Rubin R̂ < 1.1
   - Check terminal output for diagnostics

---

## 🛑 Stop Conditions

**DO NOT** proceed to other bursts if:

- R̂ > 1.2 (MCMC failed to converge)
- Residuals show structure (diagonal lines = bad DM)
- τ differs from legacy by > 50% (pipeline issue)

---

## 📊 Expected Output Files

In `data/chime/`:

1. `freya_chime_I_278720455_100ms_four_panel.pdf` (Data/Model/Residual/Profile)
2. `freya_chime_I_278720455_100ms_comp_diagnostics.pdf` (16-panel diagnostics)
3. `freya_chime_I_278720455_100ms_corner.png` (MCMC posterior)

---

## 🎉 Success Criteria

**Minimum:**

- [x] Pipeline code integrated
- [ ] Freya runs without errors
- [ ] τ ≈ 3.5ms ± 0.5ms

**Target:**

- [ ] All parameters within 20% of legacy
- [ ] R̂ < 1.05 (excellent convergence)
- [ ] Residuals pass visual inspection

---

**Next Step:** Run the command above and monitor for ~2-4 hours.

Then follow validation protocol in `.gemini/VALIDATION_PROTOCOL.md`.

Good luck! 🚀
