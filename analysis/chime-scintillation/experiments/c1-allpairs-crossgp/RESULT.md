# C1 all-pairs distinct-time cross-ACF (`c1-allpairs-crossgp`)

**Status: DOCUMENTED-FAIL. Blinded calibration verdict NO-GO; no unblinded
on-pulse fit was produced. This is not a CHIME scintillation-bandwidth
measurement.**

C1 was the owner-selected route
([decision-2026-07-14-figure1-and-chime-c1.md](../../../../../docs/rse/specs/decision-2026-07-14-figure1-and-chime-c1.md),
Faber2026 repo): symmetrized all-pairs cross products over every admissible
distinct-(polarization, time-fold) pair of the retained pol-resolved
upchannelized freya product (627–800 MHz), with leave-one-out off-pulse
template rotation, evaluated by a blinded calibration matrix before any
on-pulse fit.

## Verdict (2026-07-14, frozen config sha256 `444652ec…`)

- **Gated cells (m = 0.15, 0.17 × widths 3/6/10/16 native channels): 0/8 pass.**
  Every gated cell violates the width-bias gate by 4.6–8.3×
  (`median |Δν_d bias|` 0.018–0.070 MHz vs limits of max(0.10·truth,
  0.25·channel)). The w=3 cells additionally violate m-bias (1.2–1.5× limit)
  and 68% coverage (0.95/0.98 vs allowed 0.53–0.83).
- **All m ≤ 0.30 cells fail (16/16)**; the failure decays smoothly with rising
  m (width-bias ratio 9.7 → 1.7 from m=0.10 to m=0.30).
- **All m = 1.00 cells pass (4/4)** — the estimator, fit, and interval
  machinery are sound in the strong-signal regime; the failure is
  regime-specific, not a harness bug.
- **Null campaign FAILS**: max |z| = 4.810 over 24 held-out off-pulse +
  pairing-scramble realizations vs family-wise threshold 4.408
  (α = 0.01, N = 960); one fail-closed fit-level detection (a bound-clear
  control fit with an invalid uncertainty estimate counts as a detection).
- **Aggregate: `go: false`** (24/24 cells present, 128 finite trials each).

## Failure mode

The recovery histograms show low-m fits collapsing bimodally onto the fit
window boundaries (~0.003 and 0.25 MHz) instead of the injected truth — the
same low-modulation collapse B4 documented. At m ≈ 0.15–0.17 (the regime the
real burst occupies per B4 diagnostics) the Lorentzian signal amplitude
(∝ m²) is comparable to the residual structure left after off-pulse template
removal, so the fit has no gradient toward truth. Doubling pair count via
all-pairs products did not buy enough variance reduction to change this.

## Consequence (per the decision doc's stop rule)

No further estimator tuning on the retained product. The successor experiment
must change the information content of the input product itself —
`p1-window-upchan` (windowed re-upchannelization) per
`plan-chime-scint-corrected-products-revival.md` Phase 3. The blinding
boundary held: the only pre-freeze on-pulse fit was the explicitly
diagnostic-only single-template run recorded in `diagnostic/`, and
`frozen_config.json` provenance is independent of it.

## Artifacts

- [Machine verdict](calibration_verdict.json) · [frozen config](frozen_config.json)
- [Per-cell checkpoints + nulls](calibration/) (24 cells × 128 trials, `nulls.json`)
- [Figure manifest](figures.manifest.json) · [independent figure review](figures.review.json)
- [Calibration matrix heatmaps](figures/c1_calibration_matrix.png)
- [Recovery histograms](figures/c1_recovery_histograms.png)
- [Null campaign](figures/c1_nulls.png)
- Harness: [`validate_freya_c1.py`](validate_freya_c1.py) (freeze/calibrate/nulls/aggregate/unblind;
  `unblind` structurally refuses without a GO verdict), driver
  [`run_c1_calibration.py`](run_c1_calibration.py), plots
  [`plot_c1_calibration.py`](plot_c1_calibration.py)

The passing m = 1.00 cells demonstrate a bounded estimator regime; they do not
qualify any real-data fit at the burst's actual modulation.
