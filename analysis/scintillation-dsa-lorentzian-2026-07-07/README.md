# DSA Lorentzian ACF Bandwidth Fits

Run date: 2026-07-07 PDT.

## Qualified follow-up measurement

The original survey identified Oran's low DSA sub-band as the strongest
candidate for a defensible bandwidth. A subsequent frozen injection-recovery
analysis now qualifies a Lorentzian HWHM decorrelation bandwidth of
**0.446 MHz**, with a simulation-calibrated 68% interval of
**0.196–0.685 MHz**, at a band center of **1328.24 MHz**. It passes the
independent off-pulse null, fit-window stability, low-lag stability, response
calibration, and calibrated-interval gates. Reproduce it with
`validate_oran_dsa_measurement.py`; the complete evidence and reviewed figure
are in `results/oran_qualified/`.

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
The ACF figures show the fitted lag window, ACF points/errors, and the selected
total Lorentzian model, including its fitted constant term. The validation
context reports whether selected components carry quality flags.

## Visual provenance and canonical figure policy

The explanatory visual grammar comes from the completed Freya
instrumental-origin experiment archived at
`~/Data/Faber2026/dsa110/scintillation-data/exp-instrumental-origin-2026-07-05/`.
That directory is a read-only evidence archive: it contains one-off scripts,
captured output, figures, and scratch products from a historical worktree. It is
not the canonical producer and must not be copied forward as live analysis code.

This tracked driver is the canonical producer. Per-burst figures use the
experiment's publication summary format: separated bandwidth-versus-frequency
panels beside stacked, symmetric-lag sub-band ACFs. Within each sub-band,
positive Lorentzian widths are sorted deterministically: the narrowest is
`gamma_1`, the next broader is `gamma_2`, and so on. The upper-left panel shows
only `gamma_1`; a separate lower-left panel shows broader components and labels
them as excluded from the scaling fit. The `gamma_1` panel fits a free power law
to clean `gamma_1` measurements and overlays the prediction
from the tracked beta-coherent time-frequency joint PBF fit via
`Delta nu_d = C1 / (2 pi tau)`, with the adopted thin-screen convention
`C1 = 1.16` stated in the legend. The free bandwidth law's result record uses
`selection_policy: gamma_1_only` and `included_tracks: [1]`; broader components
are structurally excluded rather than distinguished only by styling. The
PBF uncertainty band is explicitly an approximate propagation
of marginal summaries until joint tau-alpha posterior samples are connected.
The ACF panels use colored data traces,
light-grey uncertainties, and the selected total Lorentzian model in black.
Plotting does not alter measurement eligibility. Manuscript replacement remains
blocked until the upstream producer/ACF/fitting validation records an overall
Phase 0 `PASS`.

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

## One-event two-band layout prototype

`prototype_two_band_event.py` is a reviewable Freya-only layout experiment for
placing CHIME and DSA-110 bandwidth components beside four representative ACF
panels. It is not the canonical measurement producer, a multi-event command,
or an approved manuscript figure. When the CHIME artifact-control result is
`diagnostic_only`, CHIME markers remain hollow and the joint scaling overlay is
explicitly labelled diagnostic rather than being promoted to a measurement.

Run it from the FLITS root with the staged `freya_dsa.yaml` and
`freya_chime.yaml` inputs available under `FLITS_ROOT`:

```bash
NUMBA_DISABLE_JIT=1 python \
  analysis/scintillation-dsa-lorentzian-2026-07-07/prototype_two_band_event.py
```

The script writes temporary fit products under
`/tmp/two-band-event-prototype/` and the rendered PNG/PDF beside
`/tmp/freya-two-band-scintillation-prototype.png`. These outputs are deliberately
untracked. Manuscript replacement remains blocked on the upstream Phase 0 gate
and separate author review.
