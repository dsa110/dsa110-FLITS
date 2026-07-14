# Gap analysis: rescued CANFAR scintillation analysis vs current pipeline

## Executive finding

The current pipeline cannot yet be expected to reproduce the archived CANFAR-era CHIME scintillation-bandwidth measurements exactly.

The largest numerical difference is the lag convention. The notebook-exercised CANFAR recipe fits the ACF beginning at CHIME fine-channel lag 2, while the main current pipeline begins at lag 1. The independent `revalidation.py` path already supports the correct `first_lag=2` behavior, but `ScintillationAnalysis` does not use it.

There is also an important split inside the rescued originals:

- The historical notebooks and `scinttools_old.py`/`scinttools_new.py` treat Lorentzian \(\gamma\), the HWHM, as the reported \(\Delta\nu_d\). For example, the Freya notebook labels `gamma1` directly as \(\Delta\nu\): [scint_freya.ipynb:929](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/notebooks/scint_freya.ipynb:929>).
- The later `scinttools_v3.py` refactor instead reports \(2\gamma\) as FWHM: [scinttools_v3.py:286](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/code/scinttools_v3.py:286>).

Therefore, “reproduce the originals” should mean reproducing the notebook-exercised HWHM/\(\gamma\) measurements, not blindly adopting the later v3 FWHM convention.

The capture itself is strong provenance evidence: the rescued files are declared verbatim, SHA-256 covered, and traced to the h17 arc-trash rescue and live ARC home notebooks: [ORIGIN.md:1](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/ORIGIN.md:1>), [ORIGIN.md:15](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/ORIGIN.md:15>), [ORIGIN.md:19](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/ORIGIN.md:19>).

## 1. What the originals do that the current pipeline does not

### 1.1 End-to-end CHIME baseband preprocessing

The rescued chain starts earlier than the current pipeline. It can work from CHIME complex baseband, perform FFT upchannelization, form the fine-frequency grid, and then clean and normalize the result. `_upchannel` explicitly transforms each coarse-channel timestream and frequency-scrunches the FFT products: [baseband_analysis_core.py:1497](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/code/baseband_analysis_core.py:1497>).

The current `ScintillationAnalysis` starts by loading an already-packaged NumPy dynamic spectrum: [pipeline.py:148](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/pipeline.py:148>). Thus, the voltage-to-fine-channel transformation is outside the pipeline and cannot be replayed or checked from the pipeline result alone. The producer is documented separately in `DATA_PROVENANCE.md`: [DATA_PROVENANCE.md:98](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/DATA_PROVENANCE.md:98>).

### 1.2 The original cleaning sequence is materially different

The rescued baseband chain applies all of the following:

- A fixed 730–760 MHz LTE exclusion: [baseband_analysis_core.py:2246](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/code/baseband_analysis_core.py:2246>).
- Valid-time-range selection before spectral estimation: [baseband_analysis_core.py:2254](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/code/baseband_analysis_core.py:2254>).
- Channel rejection based separately on channel mean and standard deviation, optionally measured only on a supplied off-pulse mask: [baseband_analysis_core.py:2280](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/code/baseband_analysis_core.py:2280>).
- Per-channel offset subtraction and inverse-noise weighting: [baseband_analysis_core.py:2308](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/code/baseband_analysis_core.py:2308>).
- A second RFI pass using the inferred noise floor: [baseband_analysis_core.py:2353](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/code/baseband_analysis_core.py:2353>).
- Seeded replacement of missing/NaN samples using empirical off-pulse pixels: [baseband_analysis_core.py:2115](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/code/baseband_analysis_core.py:2115>).
- A periodic “scallop” model derived from off-burst data after 3σ spectral-spike rejection: [scinttools_new.py:156](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/code/scinttools_new.py:156>).

The later Wilhelm notebook also defines a direct per-channel S/N spectrum,

\[
I_{\rm norm}(\nu)=\frac{I_{\rm on}(\nu)-\mu_{\rm off}(\nu)}
{\sigma_{\rm off}(\nu)},
\]

rather than only subtracting or dividing by an off-pulse mean: [scint_wilhelm.ipynb:84](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/notebooks/scint_wilhelm.ipynb:84>).

The current pipeline instead:

- Iteratively masks channels using off-window channel means and standard deviations: [core.py:260](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/core.py:260>).
- Optionally masks time bins: [core.py:328](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/core.py:328>).
- Preserves missing data as masks rather than filling them with empirical noise: [core.py:347](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/core.py:347>).
- Flat-fields by dividing each channel by its off-pulse mean, but does not divide by its off-pulse RMS: [freya_scintillation.py:426](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/freya_scintillation.py:426>).
- Optionally subtracts a low-order polynomial baseline: [pipeline.py:238](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/pipeline.py:238>).

The current cleaning is defensible and, in places, more robust. It is not an exact implementation of the recipe that generated the historical measurements.

### 1.3 ACF zero-lag and lag-1 handling differs

The historical behavior is encoded somewhat indirectly. With `zerolag=False`, the old `autocorr` only computes correlations for `i > 1`, storing lag \(i\) at index \(i-1\): [scinttools_old.py:30](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/code/scinttools_old.py:30>), [scinttools_old.py:59](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/code/scinttools_old.py:59>). `acf_scint_plot` then discards the leading unused element, so the fitted ACF starts at physical lag 2: [scinttools_new.py:216](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/code/scinttools_new.py:216>).

The worked Freya notebook repeats that exact sequence explicitly:

- `autocorr(spec_norm)`
- discard the first returned bin
- mirror the remaining positive lags
- fit only the central ±2 MHz interval

See [scint_freya.ipynb:929](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/notebooks/scint_freya.ipynb:929>).

The main current ACF routine instead:

- Calculates physical lags starting at 1: [analysis.py:282](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/analysis.py:282>).
- Inserts a synthetic ACF value of exactly 1 at lag 0, with error \(10^{-9}\): [analysis.py:318](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/analysis.py:318>).
- Excludes lag 0 during the ordinary one-dimensional fit but retains lag 1: [analysis.py:699](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/analysis.py:699>).

`revalidation.py` has the desired option already: `first_lag=2` explicitly drops both zero lag and CHIME lag 1: [revalidation.py:94](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/revalidation.py:94>). That option is not propagated into `calculate_acfs_for_subbands` or the main pipeline.

There is an additional current-path risk: the two-dimensional fitter includes every point satisfying the fit-range bound and does not exclude lag 0: [fitting_2d.py:303](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/fitting_2d.py:303>). Because the main ACF gives the synthetic center an error of \(10^{-9}\), the global weighted fit can be dominated by an invented `ACF(0)=1` constraint. That is unlike both historical notebook fitting and the current one-dimensional fitter.

### 1.4 Original fit windows are measurement-specific

The original notebooks select the on-pulse integration and fit window manually. Freya integrates time bins 725:875 and fits only ±2 MHz, explicitly noting that the window was chosen because it fit the central peak successfully: [scint_freya.ipynb:929](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/notebooks/scint_freya.ipynb:929>).

The generic original helper similarly distinguishes:

- A maximum lag to calculate.
- A smaller `lagrange_for_fit`.
- Symmetric fitting around the omitted center.

See [scinttools_new.py:182](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/code/scinttools_new.py:182>).

The current pipeline has configurable on/off windows and fit ranges, but its normal sub-band path applies one global maximum lag and one global fit range, clipped only by sub-band bandwidth: [analysis.py:524](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/analysis.py:524>), [analysis.py:1494](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/analysis.py:1494>). It does not contain a named compatibility recipe tying each historical burst to its original integration and fit windows.

### 1.5 Sub-band \(\gamma(\nu)\) scaling: available now, but not original-method parity

This capability is not completely absent from the current code.

The originals provide several variants:

- Equal-width or approximately equal-S/N sub-bands: [scinttools_new.py:387](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/code/scinttools_new.py:387>).
- A fixed \(\nu^4\) comparison curve: [scinttools_new.py:466](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/code/scinttools_new.py:466>).
- An unweighted log-log fit of independently measured sub-band \(\gamma\) values in the Wilhelm notebook: [scint_wilhelm.ipynb:240](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/notebooks/scint_wilhelm.ipynb:240>).
- A worked v2 example using 12 equal-S/N sub-bands, a 5 MHz ACF range, and a 2.5 MHz fit range: [scint_chromatica_v2.ipynb:1231](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/notebooks/scint_chromatica_v2.ipynb:1231>).

The current pipeline already:

- Supports uniform or equal-signal sub-band division: [analysis.py:567](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/analysis.py:567>).
- Fits a weighted log-space ODR power law to independently fitted sub-band bandwidths: [analysis.py:1649](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/analysis.py:1649>).
- Provides a stronger joint fit enforcing \(\gamma(\nu)=\gamma_0(\nu/\nu_0)^\alpha\): [fitting_2d.py:236](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/fitting_2d.py:236>).

The missing piece is a selectable reproduction mode using the original sub-band boundaries, lag convention, independent fits, HWHM definition, and unweighted log-log regression. Current ODR and joint-2D results are different estimators and should not be used as proof of historical reproduction.

### 1.6 Modulation index versus frequency and time

The v3 original extracts \(m=\sqrt{\text{Lorentzian amplitude}}\): [scinttools_v3.py:286](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/code/scinttools_v3.py:286>). It provides:

- Sub-band modulation index versus frequency: [scinttools_v3.py:808](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/code/scinttools_v3.py:808>).
- Direct time-chunk modulation \(m=\sigma/\mu\): [scinttools_v3.py:611](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/code/scinttools_v3.py:611>).
- A modulation-versus-time plot paired with mean intensity: [scinttools_v3.py:881](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/code/scinttools_v3.py:881>).

The worked Chromatica v2 notebook actually invokes both frequency and time products: [scint_chromatica_v2.ipynb:1250](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/notebooks/scint_chromatica_v2.ipynb:1250>), [scint_chromatica_v2.ipynb:1265](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/reference_arc/notebooks/scint_chromatica_v2.ipynb:1265>).

The current code has partial equivalents:

- Modulation versus frequency is implemented and plotted from fitted sub-band amplitudes: [plotting.py:300](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/plotting.py:300>).
- A richer ACF-fit-based intra-pulse modulation analysis exists: [analysis.py:1830](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/analysis.py:1830>).
- Its time-evolution plotting also exists: [plotting.py:443](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/plotting.py:443>).

But it is not operational as a replacement for the original workflow:

- The current Freya CHIME config disables it: [freya_chime_hi.yaml:30](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/configs/bursts/freya_chime_hi.yaml:30>).
- The default fit-model string is `"lorentzian_component"`, but the function immediately rejects any name not containing `"1c"`: [analysis.py:1861](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/analysis.py:1861>).
- The extractor then expects parameters such as `l_gamma1` and `l_m1`, whereas the current single-Lorentzian registry uses unnumbered `l_gamma` and `l_m`: [analysis.py:1934](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/analysis.py:1934>).
- `run_analysis.py` generates the overview plot but never invokes `plot_intra_pulse_evolution`: [run_analysis.py:98](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/run_analysis.py:98>).

Finally, the original time-domain statistic is direct \(\sigma/\mu\), while the current implementation fits an ACF in each time slice. Those are distinct quantities and should both be retained if exact historical comparisons are required.

## 2. What the current pipeline does that the originals did not

### 2.1 Explicit artifact and measurement-status gates

The current codebase contains formal CHIME controls absent from the notebooks:

- Required grid regularization, bandpass normalization, and harmonic masking: [chime_artifact_guards.py:43](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/chime_artifact_guards.py:43>).
- A fail-closed CHIME provenance verdict that demotes incomplete runs to `diagnostic_only`: [chime_artifact_guards.py:123](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/chime_artifact_guards.py:123>).
- An off-pulse ACF null test: [chime_artifact_guards.py:177](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/chime_artifact_guards.py:177>).
- A low-lag-excision stability test: [chime_artifact_guards.py:266](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/chime_artifact_guards.py:266>).
- Harmonic-mask sensitivity reported as a systematic rather than a correction: [chime_artifact_guards.py:347](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/chime_artifact_guards.py:347>).
- A combined measurement-status verdict: [chime_artifact_guards.py:385](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/chime_artifact_guards.py:385>).

However, these are currently pure auxiliary functions. `ScintillationAnalysis.run()` does not invoke them; it only exposes the on/off windows for a downstream driver: [pipeline.py:21](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/pipeline.py:21>). Thus, the gates exist in the current codebase but are not enforced by the main pipeline output.

### 2.2 Instrument-specific preprocessing guards

The current pipeline adds two important corrections not present in the original recipe:

- Frequency-grid regularization, which restores physical lag spacing across missing fine channels: [freya_scintillation.py:500](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/freya_scintillation.py:500>).
- Explicit masking of coarse-channel harmonic ACF bins: [analysis.py:664](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/analysis.py:664>).

These address measured artifacts in the newer packaged CHIME products and should not be removed merely for historical compatibility. Reproduction mode should record results both before and after these modern mitigations.

### 2.3 Uncertainty modeling and statistical model selection

The current pipeline adds:

- Per-lag product scatter and finite-scintle uncertainty: [analysis.py:234](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/analysis.py:234>).
- Synthetic radiometer-noise ACF templates and a fitted self-noise component: [analysis.py:509](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/analysis.py:509>).
- BIC-based model selection across sub-bands: [analysis.py:1521](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/analysis.py:1521>).
- A separate BIC plus nested F-test selector for the number of Lorentzian components: [revalidation.py:277](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/revalidation.py:277>).
- Reduced-\(\chi^2\), covariance propagation, and a joint cross-sub-band fit: [fitting_2d.py:394](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/fitting_2d.py:394>).

The originals generally relied on optimizer success, fit reports, and visual inspection, without a machine-readable acceptance status.

### 2.4 Better provenance infrastructure

The current repository documents the producer, upchannelization factors, packaging, recoverability limits, and retired products: [DATA_PROVENANCE.md:96](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/DATA_PROVENANCE.md:96>), [DATA_PROVENANCE.md:155](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/DATA_PROVENANCE.md:155>).

The pipeline also hashes input-path, downsampling, and the complete analysis configuration into cache filenames, preventing stale cached ACFs from being silently reused after a preprocessing change: [pipeline.py:42](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/pipeline.py:42>).

This is useful cache provenance, but final result provenance is incomplete: `run_analysis.py` writes only `final_results`, without the resolved configuration, configuration fingerprint, input-data digest, reference-code version, or artifact-gate verdict: [run_analysis.py:83](</Users/jakobfaber/Developer/scratch/worktrees/flits-scint-rescue/scintillation/scint_analysis/run_analysis.py:83>).

## 3. Ranked implementation deltas

Estimated sizes include implementation and focused tests, but not large binary reference fixtures.

| Rank | Concrete delta | Target files | Estimated size |
|---|---|---|---:|
| **P0-1** | Add an explicit `first_fit_lag`/`acf_first_lag` contract. Use `2` for CHIME upchannelized products and `1` for DSA. Factor or reuse `revalidation._mean_normalized_acf` so the main and reference paths cannot drift. Remove the synthetic lag-0 point from fit inputs; at minimum exclude it from the 2D fit and never give an invented center \(10^{-9}\) uncertainty. | `scintillation/scint_analysis/analysis.py`, `revalidation.py`, `fitting_2d.py`, CHIME burst configs | **100–170 LOC** |
| **P0-2** | Implement a named `canfar_reference` preprocessing mode: fixed LTE mask; exact on/off windows; per-channel off-mean subtraction and off-RMS division; original mean/std RFI thresholds; optional second floor-based RFI pass; deterministic empirical-noise filling. Preserve the modern mask-only path as the default. Emit the cleaned spectrum and masks as inspectable intermediate products. | `core.py`, `pipeline.py`, `freya_scintillation.py`, burst configs | **220–350 LOC** |
| **P0-3** | Add reference-regression tests driven by rescued-recipe behavior. Tests should compare cleaned spectra, retained lag arrays, ACF values, and fitted notebook \(\gamma\) values—not merely pin current outputs. Record whether each expected value comes from an executed notebook output or a recomputation of the rescued source. | `scintillation/scint_analysis/tests/test_reference_arc_parity.py`, small text/NPZ fixtures | **180–300 LOC** plus fixtures |
| **P0-4** | Make the bandwidth definition explicit in results: `gamma_hwhm_mhz`, optional `fwhm_mhz=2*gamma`, and `reported_dnu_definition`. Historical-reproduction outputs must report \(\Delta\nu_d=\gamma\), matching the notebooks. Prevent accidental comparison to v3’s \(2\gamma\) field. | `analysis.py`, `revalidation.py`, `fitting_2d.py`, `run_analysis.py` | **60–100 LOC** |
| **P1-5** | Add a selectable historical sub-band scaling estimator: reproduce the exact equal-S/N boundaries, independently fit each sub-band with the same lag/window rules, and run the notebook’s unweighted log-log \(\gamma\)-versus-\(\nu\) regression. Keep current ODR and joint-2D fits as separate modern estimators; report all three with method names. | `analysis.py`, `plotting.py`, burst configs | **90–150 LOC** |
| **P1-6** | Repair and wire modulation diagnostics. Fix the intra-pulse model key and parameter-name mismatch; save `plot_intra_pulse_evolution`; add the original direct time-chunk \(m=\sigma/\mu\) statistic alongside the ACF-fitted modulation. Continue emitting fitted \(m(\nu)\) per sub-band. | `analysis.py`, `pipeline.py`, `plotting.py`, `run_analysis.py` | **120–190 LOC** |
| **P1-7** | Enforce the existing CHIME gates in the main pipeline. Compute off-pulse ACF slices and low-lag refits, call `finalize_measurement_status`, and embed the verdict, mitigation records, systematic scan, windows, and fit-lag policy in `final_results`. A CHIME JSON result should not be a measurement merely because optimization succeeded. | `pipeline.py`, `chime_artifact_guards.py`, `run_analysis.py` | **140–220 LOC** |
| **P2-8** | Upgrade final-result provenance: include resolved config, config fingerprint, input file SHA-256, git commit, reference recipe/version, channel-grid summary, mask counts, on/off windows, and whether the result is historical-parity or modern-mitigation mode. | `pipeline.py`, `run_analysis.py` | **80–140 LOC** |

## Recommended acceptance sequence

The first reproducibility milestone should be narrow:

1. Run the same cleaned CHIME spectrum through the rescued pair-loop ACF and the current compatibility ACF.
2. Require identical retained lags beginning at lag 2 and numerical ACF agreement within a stated floating-point tolerance.
3. Require the same fit interval and the same HWHM \(\gamma\) within a fit-stability tolerance.
4. Only then compare cleaning modes and end-to-end notebook measurements.
5. Treat modern grid/bandpass/harmonic mitigations as a second, explicitly labeled analysis—not as silent changes to the historical reference result.
6. Promote a CHIME number to `measurement` only after the off-pulse null and low-lag stability gates pass.

No files were modified and no outputs were written during this audit.

