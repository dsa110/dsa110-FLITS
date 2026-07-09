# DSA Lorentzian ACF Bandwidth Fits

Run date: 2026-07-07 PDT.

This directory is the first fresh DSA scintillation-bandwidth pass after the
DSA/CHIME data-products staging cleanup. It deliberately does not read the
legacy `stored_fits` blocks in `scintillation/configs/bursts/*_dsa.yaml` and
does not use any rescued `acf_results.pkl` files. The driver starts from the
staged DSA dynamic-spectrum `.npz` files under:

```text
~/Data/Faber2026/dsa110/scintillation/data/{burst}.npz
```

The run path is:

1. load each checked-in DSA burst config,
2. force fresh data preparation and ACF extraction,
3. evaluate 2, 3, and 4 equal-S/N sub-band splits using fixed viability gates
   rather than inheriting `analysis.acf.num_subbands` from the YAML,
4. stop after ACF extraction for the selected split,
5. fit 1, 2, and 3 Lorentzian components to each sub-band ACF within the
   config's `analysis.fitting.fit_lagrange_mhz` window,
6. select the sub-band component count with the existing BIC plus nested-F
   criterion in `scintillation.scint_analysis.revalidation`,
7. write one multi-panel ACF+fit figure per burst under `results/figures/`.

The sub-band selection policy chooses the largest candidate split for which
every produced sub-band has at least 512 unmasked channels, an 8 MHz fitted lag
window, 30 positive-lag fit samples, and at least one selected component not
carrying a quality flag. If no candidate satisfies all gates, the driver records
and uses the least pathological candidate. The selected count, evaluated
candidates, fixed gates, selected policy, and rejection reasons are written into
each burst JSON under `subband_selection` and summarized in
`DSA_LORENTZIAN_FITS.md`.

The generated tables include `quality_flags` for components that should not be
used as clean bandwidth measurements without manual inspection. In particular,
`dnu_exceeds_fit_window` marks a Lorentzian width larger than the lag span fitted
for that sub-band, and `fractional_dnu_err_gt_1` marks a formally weak width.
The ACF figures show the fitted lag window, ACF points/errors, the selected
total Lorentzian model, the fitted constant term, and individual component
curves. Component annotations include any quality flags.

Diagnostic plots and intermediate caches are disabled for this run, so no
`${FLITS_ROOT}` literal-path plot artifacts or stale ACF caches can affect the
results. Noise descriptors remain enabled because they define the ACF
normalization, but the Monte Carlo noise-template generation is disabled because
this strict Lorentzian-only pass does not fit the template component.

## CHIME artifact-control guards (`--band chime`)

CHIME upchannelized (gen-3) products carry instrumental structure that an ACF
fit can mistake for a real decorrelation scale. The freya experiment
(`docs/rse/specs/experiment-freya-chime-instrumental-origin.md`, arms A/B1/C)
established that the canonical freya CHIME Δν_d ≈ 35 kHz is the product's
noise-correlation scale, not scintillation. This driver promotes that
experiment's one-off arms into standing, fail-closed guards
(`scintillation/scint_analysis/chime_artifact_guards.py`), active for
`telescope: chime` and inert for DSA.

For a CHIME run, each sub-band JSON entry now carries:

- `harmonic_mask` — the coarse-channel comb mask
  (`analysis.fitting.harmonic_mask`, k·0.390625 MHz) is now applied to the
  fit-window ACF **before** the Lorentzian selector (previously the driver
  ignored it — the `--band chime` trap), with `n_bins_removed` / `n_bins_kept`
  recorded.
- `harmonic_mask_systematic` — the fit width with vs without the mask and their
  fractional difference, reported as a **systematic band, not a correction**.
- `off_pulse_null` — refits burst-free noise slices on the *identical* sub-band
  channels; `null_pass=false` when the off-pulse fits reproduce the on-pulse
  scale (the arm-A instrumental signature).
- `low_lag_stability` — refits after excising the first 1–3 channel lag bins;
  `stable=false` when the width collapses (no resolved Lorentzian wing, arm B1).

The burst-level `artifact_control` block and top-level `measurement_status`
combine a **provenance gate** (grid regularization + bandpass normalization +
harmonic mask must all be enabled) with the off-pulse null and low-lag
stability. A CHIME burst is a `measurement` only if all pass; otherwise it is
`diagnostic_only` and the `failed_checks` are named. DSA-band results are never
demoted (no DSA config enables the harmonic mask, so the DSA fit is byte-for-
byte unchanged). See `CHANGES-artifact-controls.md` for the full field list and
provenance.

## Reproduce

From `pipeline/`:

```bash
python analysis/scintillation-dsa-lorentzian-2026-07-07/run_dsa_lorentzian_fits.py
```

Set `FLITS_ROOT` or pass `--flits-root` if the staged data live somewhere other
than `~/Data/Faber2026/dsa110`. Add `--band chime` to run on the CHIME configs;
the artifact-control guards above then apply.
