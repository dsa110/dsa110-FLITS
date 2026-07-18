# CHIME window-tuning scintillation campaign (2026-07-17)

Objective per-subband scintillation-bandwidth (Δν_d) campaign for the 12 CHIME/FRB ×
DSA-110 co-detections, plus the two-band (CHIME + DSA) scaling-index analysis. This
directory is the reproducible record behind the CHIME scintillation results.

## Estimator decision (reproducible from `inject_discriminate.py`)

The ACF window is the **matched-significance burst core** (`burst_core`), fit with an
unweighted boxcar. This was decided by injection, not asserted:

- On **clean** known-truth ACFs (no spectral difference between burst core and tail) all
  three candidate estimators — core-boxcar, tail-expanded boxcar, matched-weight over the
  tail-expanded window — agree to ~2%.
- When the **scattering tail carries broad spectral structure distinct from the narrow
  core scintle** (the real chromatica/zach case), `burst_core` recovers γ unbiased while
  tail-expanded inflates recovered γ 2.6–3.6× and matched-weight-tail still inflates
  1.3–1.9×.

So the core window costs nothing where the tail is benign and protects the measurement
where it is not. The matched-weight-tail and bare tail-expanded windows are retained as
reported variants so the selection systematic stays visible (`gamma_win_sys`,
`gamma_tail`).

`inject_recover.py` is the companion fitter-validation harness: on known-truth ACFs the
2-component `_lorentz2` model recovers γ unbiased across 0.02–3 MHz, the single-Lorentzian
fails to resolve a narrow scintle under a broad envelope while the 2-component model
recovers it, and the ΔBIC≥6 model-selection gate has a 0/150 false-positive rate on
envelope-only and 0/150 on noise.

## Off-window selection

The off-pulse window feeds per-channel de-scalloping and RFI statistics, so residual burst
power there corrupts every downstream quantity. Two enforced rules in
`scint_analysis/window_optimize.select_windows`:

1. **Pre-burst preference (primary).** The post-burst region is where the scattering tail
   lives; its channels carry frequency-correlated scattered burst power that biases the
   per-channel gain estimate even when a time-domain purity metric passes. The largest
   adequate (≥ `OFF_MIN_BINS`) pre-burst free run is taken outright. For chromatica this is
   the difference between the recovered 4-subband detection (α≈+1.7) and a railed 2-subband
   collapse.
2. **Off-purity gate (fallback).** When no adequate pre-burst run exists, score each
   candidate at the burst scale and take the largest with `off_snr ≤ OFF_SNR_MAX`; this
   guards the oran case, where the offending run rode the rising burst envelope.

A pre-burst run bypasses the `off_snr` gate entirely, which is safe because `off_snr` was
never an RFI guard: RFI is handled separately (pipeline channel mask + `auto_rfi_flag` on
the off-pulse statistics + user bands), and `off_snr` only measures residual *burst*
(time-domain) power in the candidate window. A pre-burst run cannot contain the burst or
its scattering tail, so the quantity `off_snr` scores is structurally absent there; the
per-channel de-scalloping robustness that actually matters (a clean off-pulse mean) is
better served by the pre-burst region than by a post-burst window that passes the gate but
sits in the scattering tail.

## Grid regularization

`analysis.grid_regularization` is **ON** in the γ path: `window_refit._build_spec` enables
it and `freya_scintillation.apply_grid_regularization` applies it, so the fitted grids are
uniform (rel-spread ~1e-11) and absolute γ values are quotable. The "frequency grid
non-uniform" warning originates only from `window_optimize.default_windows`' throwaway
profile build, which intentionally runs grid-reg OFF because that spectrum only seeds the
burst-finder and never feeds γ. The warning is therefore not a caveat on any reported γ.

## Results

- `results/campaign_results.jsonl` — one summary line per burst (config-path rerun,
  grid-reg ON, core-boxcar primary, pre-burst off rule).
- `results/<burst>_campaign.json` — full per-burst record: windows, per-subband fits, and
  the window/estimator variant table (the "2L table").
- `results/two_band_tracks.json` — per-band and forced-joint power-law fits for the triad.

### Pinned CHIME triad (this campaign)

| burst | α (CHIME, 400–800 MHz) | resolved subbands | status |
|---|---|---|---|
| zach | +3.03 ± 0.65 | 3/4 | detection |
| chromatica | +1.72 ± 0.40 | 4/4 | detection (cleanest) |
| freya | — (top subband rails) | 1 physical | non_detection (candidate structure under some analysis choices) |
| hamilton | +8.94 ± 3.16 | 2/4 | diagnostic-only (unphysical) |

## Two-band scaling and the two-screen result

A single power law across the 400→1530 MHz (3.8×) lever assumes one dominant screen in
both bands. The committed one-screen statistic
(`flits.batch.analysis_logic.check_tau_deltanu_consistency`, product τ·Δν_d = C₁/2π,
single-screen range [0.1, 2]) rejects that for the whole triad, recomputed at the pinned
Δν_d values (this campaign's DSA components scaled to 1.4 GHz): chromatica τ·Δν_d = 61.0,
zach 26.7, freya 386.7 — all `different_screens` (the DSA-band Δν_d samples a nearer screen
than the CHIME-band scattering). The forced joint fit agrees: joint reduced-χ² = 7.8
(zach) and 18.0 (chromatica), and chromatica's per-band slopes disagree outright
(CHIME +1.72 vs DSA +6.48).

Per the manuscript's standing rule (physically defensible presentation wins over a forced
fit), the headline is the **per-band α + two-screen decomposition**; the forced joint α is
shown only as the tested-and-rejected single-screen hypothesis with its consistency
statistic. `two_band_joint.py` computes both tracks.
