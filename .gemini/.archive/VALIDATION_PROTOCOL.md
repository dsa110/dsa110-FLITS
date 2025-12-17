# Scientific Validation Protocol: Freya Test

**Objective:** Validate the integrity of the FLITS scattering pipeline with integrated DM refinement.

---

## 🔬 Test Definition

We will analyze the **Freya** burst (CHIME data) using the updated pipeline.

**Why Freya?**

- Single-component burst (simpler than multi-component Hamilton)
- Known ground truth: τ ≈ 3.5ms, α ≈ 4.2
- High SNR for clear validation

**Command:**

```bash
python scattering/run_scat_analysis.py \
    data/chime/freya_chime_I_912_4067_32000b_cntr_bpc.npy \
    --config batch_configs/chime/freya_chime.yaml \
    --refine-dm \
    --model-scan \
    --plot \
    --save
```

---

## 🧐 Validation Checklist

### 1. DM Refinement Validity

- [ ] **Check Logs:** What was the refined DM?
  - Catalog: `912.400`
  - Refined: `_______`
  - Difference: `_______` (Is this < 1.0 pc/cm³?)
- [ ] **Physical Plausibility:** Does the refined DM maximize structure?
  - _If difference is large:_ Check if the pulse looks sharper or if it just aligned with noise.

### 2. Fit Quality (Residuals)

- [ ] **Open:** `data/chime/freya_chime_..._four_panel.pdf`
- [ ] **Residual Panel:** Is it Gaussian noise?
  - ❌ Diagonal stripes = DM error (refinement failed)
  - ❌ Vertical stripes = RFI
  - ❌ Burst "ghost" = Poor amplitude/width fit
- [ ] **Chi-Squared:** Is $\chi^2_{\nu} \approx 1.0$?

### 3. MCMC Robutness

- [ ] **Open:** `data/chime/hamilton_..._comp_diagnostics.pdf` (Bottom panels)
- [ ] **Trace Plots:** Are variables stationary (flat hairy caterpillars)?
  - ❌ Drifting up/down = Not converged (need more steps)
  - ❌ Stuck at edge = Prior bound issue
- [ ] **Correlation:** plausible correlations (e.g., $\tau$ vs $\alpha$)?

### 4. Comparison to Legacy Results

Freya has previously been fitted. We expect to **reproduce** the legacy results:

- **Scattering timescale:** τ ≈ 3.5ms (legacy: 3.515ms)
- **Frequency exponent:** α ≈ 4.2 (legacy: 4.2)
- **Model Selection:** M2 (scattering) or M3 should have lowest BIC.

### 5. Advanced Visual Analysis (Fallback)

If the automated fit looks suspicious or fails convergence, use the interactive widget for deep inspection:

1.  **Open Notebook:** Create `notebooks/validate_hamilton.ipynb`
2.  **Code Snippet:**

    ```python
    from scat_analysis.burstfit_interactive import InitialGuessWidget
    from scat_analysis.burstfit_pipeline import BurstDataset

    # Load data
    ds = BurstDataset("data/chime/hamilton...npy", "output/", telescope=telcfg, ...)

    # Launch widget
    widget = InitialGuessWidget(ds, model_key="M3")
    display(widget.create_widget())
    ```

3.  **Action:** Manually adjust sliders to see if a better fit exists that the MCMC missed.

---

## 🛑 Stop Conditions

**Do NOT proceed to batch processing if:**

1. MCMC fails to converge ($\hat{R} > 1.2$).
2. Residuals show obvious structure (diagonal lines).
3. The refined DM is wildly different (> 5 pc/cm³) from catalog without good reason.
