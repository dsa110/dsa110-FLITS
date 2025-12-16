# Data-Driven Scattering Analysis Walkthrough

## Summary

We successfully ran the FLITS scattering analysis pipeline on the `freya_chime` real data. We overcame significant issues where the fit amplitude was being driven to zero and the pipeline was crashing.

## 1. The "Zero Amplitude" Problem

**Symptom:** The pipeline ran but produced a model with amplitude $\approx 0$. The diagnostic plots showed a blank model panel.
**Diagnosis:**

1.  **Dead Channels:** The CHIME data contains ~300 "dead" channels (value=0). The `_estimate_noise` function was clipping the noise floor at $10^{-6}$, assigning a tiny non-zero noise to these channels. The model (predicting non-zero scattering tails in these channels) was penalized by a factor of $(S / 10^{-6})^2 \approx 10^{12}$ in $\chi^2$, forcing the optimizer to kill the signal amplitude.
2.  **Unit Mismatch:** `BurstDataset` was normalizing data to max=1.0, while the likelihood model and initial guess (derived from SNR) assumed standard z-score units (noise=1.0, max $\approx$ SNR). This caused the initial guess ($c_0 \approx 85$) to be interpreted as 85x brighter than the data, leading the optimizer to diverge.

**Fixes Applied:**

- **Modified `burstfit.py`:** Removed the $10^{-6}$ floor in `_estimate_noise`, allowing it to return 0.0 for dead channels. The `log_likelihood` function now correctly ignores these channels (where `noise_std < 10^{-9}`).
- **Modified `BurstDataset`:** Removed the `ds_arr / peak` normalization step. Data is now processed in consistent Signal-to-Noise (z-score) units.

## 2. Pipeline Engineering Fixes

We also fixed several crash-causing bugs:

- **CLI Telescope Config:** Fixed crash when passing `--telescope chime` as a string (added auto-loading).
- **Arguments:** Fixed `TypeError` when `outer_trim` was None.
- **JSON Serialization:** Fixed `TypeError: Object of type FRBParams is not JSON serializable` by using `dataclasses.asdict` fallback in `burstfit_pipeline.py`.

## 3. Results

The pipeline now runs to completion and saves all outputs.

**Output Files:**

- `results/bursts/freya_data_driven/freya_fit_results.json`: Contains fitted parameters.
- `results/bursts/freya_data_driven/FRB_comp_diagnostics.pdf`: Visual diagnostics.

**Scientific Note:**
The current best fit found $c_0 \approx 1.64$ and $\tau \approx 0$ (log-param $\approx -6.9$). This indicates the optimizer likely moved the pulse $t_0$ away from the burst peak (to $4.12$ ms) to escape the high $\chi^2$ of the shape mismatch.
**Recommendation:** The `burstfit_init` initial guess for $c_0$ (210) might be scaling differently than the `FRBModel` amplitude. Manually providing a lower start (e.g., `--seed_single` with $c_0 \approx 20$) or fixing $t_0$ could help the optimizer latch onto the true burst.
