# P3′ delay-domain matched Δν_d estimator (`p3-optimal-estimator`)

**Status: `calibrated` → unblinded one-shot (owner-authorized in-session,
2026-07-15). Outcome: highly significant broad spectral structure
(z_max = 40.4), NOT admissible as a Δν_d measurement — the amplitude sits
~11× above the calibrated scintillation ceiling and the width pins at the
scan-grid edge; the measured amplitude matches the burst's intrinsic
spectral envelope (order-unity spectral structure × f_b = 0.05 →
â ≈ 10⁻³). No qualified Δν_d measurement; the scintillation constraint is
censored at the envelope-confusion floor, not the radiometer floor. See
§One-shot unblinding below; interpretation wording awaits owner
ratification.**

P3′ is the owner-sanctioned amended successor to P2
([`experiment-chime-scint-p3-optimal-estimator.md`](../../../../docs/rse/specs/experiment-chime-scint-p3-optimal-estimator.md),
Faber2026, §P3′ amendment): P2's S2 split-ratio construction unchanged (the
on/off ratio cancels the instrumental common mode algebraically, P2 G2), with
the lag-space Lorentzian fit replaced by a delay-domain matched (optimal
quadratic) estimator — global demeaning only (no 64-channel block demeaning),
delay bins `k < 11` excluded (envelope control), null-mean-subtracted
z-scores. Frozen config sha256 `0baf4ea2…`; 19 465 good channels of 23 064
(627–800 MHz).

## Gate 0b (why P3 was amended before it was built)

The originally frozen spec (with block demeaning) failed its predeclared
forecast floor: 1.24σ at 213 kHz vs the required 3 — the demeaning erases
scintles wider than one 390.6 kHz coarse block, which is the entire Gate-0
detectability window (`gate0b_forecast.{py,json,png}`). Diagnostics without
demeaning restored 3.0–4.5σ across 127–352 kHz
(`diagnostic_nodemean*.py/json`, `gate0b_owner_decision.png`); the owner
sanctioned the amendment on that evidence. Gate 0b did exactly what it was
predeclared to do: it stopped a doomed build before a fourth
build-then-fail, and located the sensitivity loss precisely.

## T battery

Unit tests (`scintillation/scint_analysis/tests/test_optimal_dnu.py`): 7/7,
including T1 (bandpass invariance) and T6 (blinding guard). At-scale battery
(`tbattery.json`): T1 max deviation < 1e-12 under the measured 35 kHz-shaped
common-mode gain; T2 unbiasedness at the real noise scale — a **+12 %
amplitude bias** (3.7 SE over 100 realizations) is recorded, consistent with
the ratio's signal-noise coupling, small against the ±2σ pull gate; T3 null
error calibration 0.88 (band [0.8, 1.2]).

## G1″ — injection recovery: PASS

Grid `m ∈ {0.15, 0.17} × Δν_d ∈ {35 control, 77, 127, 213, 352} kHz × 50`,
seeds `1000·cell + realization` (P2 convention), multiplicative injection at
`f_b = 0.05` into real off-pulse frames. Certification: median recovered
width (argmax z over the 25-point scan) within ±30 %, |median pull| ≤ 2,
≥ 90 % converged. Required cells = all `Δν_d ≥ 213 kHz` (the Gate-0 ≥5σ
window):

| cell | median recovered | bias | pull | median max-z | certify |
|---|---|---|---|---|---|
| m=0.15, 213 kHz | 243 kHz | +14 % | 0.49 | 3.07 | ✅ |
| m=0.15, 352 kHz | 353 kHz | +0.3 % | −0.00 | 3.00 | ✅ |
| m=0.17, 213 kHz | 275 kHz | +29 % | 0.17 | 3.56 | ✅ |
| m=0.17, 352 kHz | 353 kHz | +0.3 % | −0.05 | 3.85 | ✅ |
| 35 kHz control (both m) | noise-pinned | — | — | ~1.5 | ❌ (required) |

Reported, not required: m=0.15/127 and m=0.17/77 certify; m=0.17/127 misses
on width bias by a hair (+31.5 % vs 30 %). Median max-z at the detectable
cells (3.0–3.9) matches the Gate 0b forecast — the estimator performs at its
priced sensitivity.

## G2″ — null campaign: PASS

100 calibration nulls (seeds 900000–900099: weights + per-Δν_d mean/σ) and
100 independent evaluation nulls (900100–900199) through the full scan.
Trials-corrected threshold `z_trials = 2.06` (p95 of evaluation max-z);
calibration-half p95 = 2.22 (consistency within the frozen 15 %); σ
calibration on the evaluation half 0.90 (band [0.8, 1.2]).

## One-shot unblinding (owner-authorized, 2026-07-15)

Computed exactly once with the frozen configuration
(`unblind_onpulse.json`, `figures/unblind_onpulse_scan.png`). Facts:

- `z` rises **monotonically** from 13.8 (20 kHz template) to **40.4 at the
  400 kHz grid edge** — no interior peak; the matched filter wants structure
  wider/smoother than every template in the scan.
- `â` is nearly flat: 1.2×10⁻³ → 7.6×10⁻⁴ across the grid. The calibrated
  scintillation model caps `a = (f_b·m)² ≈ 7.2×10⁻⁵` at m = 0.17 — the
  measured amplitude implies m ≈ 0.55–0.70 at every width, far outside the
  calibrated domain (G1″ validated recovery only for m ≤ 0.17,
  Δν_d ≤ 352 kHz).

Interpretation (fail-closed): a ratio spectrum carries `f_b·e(ν)` for a
burst with fractional intrinsic-envelope structure `e(ν)`; order-unity
envelope structure at ≳ MHz scales gives `â ≈ (0.05)² ≈ 2.5×10⁻³·⟨e²⟩` —
the measured 10⁻³ is the natural value for `e ~ O(1)`. The envelope is
time-stable across the burst, so it survives the S2 time-split cross that
kills self-noise, and it is smoother than the k ≥ 11 cut's 12.8 MHz
excision scale was sized to catch. The G2″ nulls (off-pulse only) and the
G1″ injections (scintle-only) could not have flagged it — this is exactly
the confounder limb the record's envelope-control caveat anticipated, at
larger amplitude than the cut could remove.

Under the frozen G3 rules read literally (Δν_d ≥ 77 kHz, z ≥ 5) this would
formally qualify; it is **declined as a Δν_d measurement** because the
amplitude is inconsistent with the calibrated signal model and the width
estimate is censored at the scan boundary — claiming a scintle here would
attribute intrinsic burst structure to propagation. Declining a claim is
the conservative direction of the unblinding rule; no threshold was moved
to create a claim.

Consequence for the scintillation constraint: the on-pulse spectrum carries
a ~10⁻³ envelope foreground at all scan widths, so the achievable upper
limit is **envelope-confusion-limited**, not radiometer-limited: a true
scintle at the calibrated ceiling (7×10⁻⁵) would contribute ≤ 10 % of the
measured `â` and cannot be separated without an envelope model — which
would be a new, separately sanctioned experiment (P4 class), noting that
the one-shot has been spent and any further on-pulse analysis is
post-unblinding by construction.

## Blinding

No statistic was computed on samples 250–350 before the owner-authorized
one-shot above. The structural guard (`BlindingError`, `allow_unblind`) is
unit-tested; the harness `unblind` subcommand refuses without a
`calibrated` selection, an unchanged frozen config, and the explicit
`--unblind-i-know-what-i-am-doing` flag.

## Deliverables & reproducibility

- Estimator: `scintillation/scint_analysis/optimal_dnu.py`; tests
  `tests/test_optimal_dnu.py`.
- Harness: `p3_calibration.py` (`freeze` → `tbattery` → `g1` → `g2` →
  `select`; `unblind` is orchestrator-only).
- Records: `frozen_config.json` (sha `0baf4ea2…`), `scan_assets.npz`,
  `tbattery.json`, `g1_matched.json`, `g2_matched.json`, `selection.json`;
  Gate 0b artifacts as above.
- Figures: `plot_p3.py` → `figures/` (injection recovery, pulls, null
  maxima, template bank + measured variance).
- Env: conda `py312` (numpy 2.4, scipy 1.17, matplotlib 3.10). Data:
  `~/Data/Faber2026/dsa110/upchan_codetections/crossacf-2026-07-14`
  (SHA-256s in `frozen_config.json.inputs_sha256`, matching the pipeline
  `DATA_MANIFEST.yaml`).
