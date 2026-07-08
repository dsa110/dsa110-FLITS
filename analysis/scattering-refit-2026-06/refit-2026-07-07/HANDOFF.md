# HANDOFF — jointmodel_pair bad-fit remediation (2026-07-07)

Context for the next agent taking over the 2D time-frequency (joint CHIME+DSA
multi-component) refits for the manuscript's morphology-audit figures.
**Jakob has pushback/corrections on the first-round fits — get those from him
FIRST before continuing; the "next steps" below are superseded by whatever he
says.**

## 1. Problem

`figures/jointmodel_pair/*_jointmodel_pair.pdf` (11 bursts, appendix
`app:jointmodel-pairs`, built by
`pipeline/analysis/scattering-refit-2026-06/plot_jointmodel_pair.py` from
`beta_campaign_verdicts.json` rows) contained four bad fits, confirmed by
residual-panel audit and matching `KNOWN_MULTIPLICITY_FLAGS` in that script:

- **hamilton** — sharedzeta C1D1 missed the leading CHIME component.
- **casey** — sharedzeta C1D1 missed close double CHIME structure (χ²=3.86).
- **zach** — C1D1 missed the trailing DSA sub-burst complex (+2.1/+2.6 ms).
- **wilhelm** — sharedzeta missed the leading DSA component and leaves a
  coherent bright-pulse residual; this is residual profile structure within the
  beta≈4 exponential/EMG-preferred branch, not evidence against EMG itself.

Marginal (not touched in the original handoff): freya, johndoeII, phineas.
Post-handoff correction: johndoeII's working remediation target is C2D2. The
2026-07-07 scratch pilots produced C2D2 and C2D3 PPC checks with nearly
indistinguishable residuals, so the simpler C2D2 beta-native product was
promoted on 2026-07-07 (`chi2_C/D=1.09/1.23`, `tau_1GHz=2.219 ms`). The old
C2D1 beta product is superseded.

## 2. What exists / was verified

- **All plan-doc "mandatory fixes" are ALREADY LANDED** in
  `pipeline/scattering/scat_analysis/burstfit_joint.py`: `_JointPriorTransformOrdered`
  (smooth dt_min reparametrization, per-band floors), eigenvalue guard
  (`eig_rel_floor=1e-6`) with rank-1 top-eigenpair fallback, proper N(0,s²I)
  gain prior (`proper_gain_prior`/`gain_s2`), frac_culled/max|g| diagnostics.
  `tests/test_gain_marginal_multi_band.py` + `tests/test_joint_prior_ordered.py`:
  14/14 pass (verified in sandbox, py3.10 works despite the >=3.12 pin;
  `pip install -e . --no-deps` + dynesty/emcee/scipy).
  MULTICOMPONENT_PLAN.md §1 "required fixes" is therefore STALE.
- **Template**: `figures/jointmodel_pair/fit_artifacts/run_whitney_c2d2_cwin_chime2resolved.py`
  (per-component t0/zeta windows, `_JointLogLikelihoodGainMulti`, s2=None) —
  Jakob confirmed this is the current approach to build on.
- **Data are local**: `dsa110-FLITS/data/{chime,dsa}/*.npy` are symlinks into
  `/Users/jakobfaber/Data/Faber2026/dsa110/DSA_bursts/`. All 12 bursts, both bands.
- **Fit chain**: run fit → `<b>_joint_fit<suffix>.json` → dump jointmodel npz
  (OLS per-channel gain recovery, `dump_jointmodel.py` logic) →
  `plot_codetection(columns=(data, model, resid))`.

## 3. What was done here (scripts in ./scripts/, artifacts in ./fits, ./figures, ./configs)

- `inspect_profiles.py` — find_peaks on BurstDataset-preprocessed profiles to
  place component windows. **Windows are on the onpulse-cropped time axis and
  depend on `onpulse_pad_factor` — keep pads in sync between fit and dump.**
- `refit_runner.py` — per-burst SPEC (component windows, tau/zeta/ddm priors,
  pads, per-band t_factor override); freya-pattern configs (CHIME dm_init=0
  f16/t24; DSA dm_init=catalog f384/t2); disjoint windows → plain transform,
  overlapping (casey) → Ordered + dt_min.
- `refit_chunk.py` — dynesty checkpoint/resume wrapper (sandbox bash calls are
  killed at 45 s and background procs are reaped; irrelevant on a Mac — run
  `refit_runner.py` directly there).
- `dump_plot.py` — npz + figure + diagnostics (χ² per band, lag-1 of whitened
  freq-summed residual profile). Per-burst `PLOT_PAD_MS={"zach":6,"wilhelm":6}`
  fixes Jakob's report that zach's trailing DSA sub-bursts were cut at the
  right edge (default `crop_bands_to_subburst_window` pad too small).
- Paths in scripts are sandbox-mount form (`/sessions/.../mnt/...`); on a Mac
  set REPO/DATA/RUNS to the real paths.

## 4. Round-1 results (nlive=160, dlogz=0.5, beta~U(3,4), s2=None)

| burst | comps | χ²_C | χ²_D | lag1_C | lag1_D | α | tau_1ghz |
|---|---|---|---|---|---|---|---|
| hamilton | C2D1 | 3.64→1.33 | 1.00 | 0.61→0.01 | 0.34 | 4.02 (β railed) | 0.0221 |
| zach | C1D3 | 1.57 | 1.03 | **0.86** | 0.28 | 4.02 (β railed) | 0.291 |
| wilhelm | C2D2 | 1.25 | **3.05** | 0.74 | −0.10 | 4.02 (β railed) | 0.297 |
| casey | C2D1 | 3.86→1.54 | 0.99 | 0.38 | 0.28 | **5.04 interior** | 0.0058 |

Posterior details in `fits/<b>_joint_fit_*.json`; diagnostics in `fits/<b>_diag.json`.

Notable per-burst posteriors:
- hamilton: t0_C = 11.620/11.952, both interior. Third CHIME peak (12.90 ms,
  10.7σ) unmodeled → residual dipole remains.
- zach: first attempt railed t0_C1 at window floor — **t0 is pulse RISE not
  peak; open windows ≥2.5τ_band before the profile peak** (τ(0.6 GHz)≈2 ms).
  Rerun interior (t0_C1=13.256). delta_dm_C=0.009 (did NOT absorb the CHIME
  residual); trailing red CHIME patch (+2.5–4 ms, 650–780 MHz) coincident with
  the DSA trailing complex → motivated C2D3.
- wilhelm: leading DSA comp found (t0_D1=0.415±0.05, window interior);
  zeta_C2=0.385 near its 0.4 rail, t0_C2 at 5.6 floor — trailing CHIME comp
  poorly constrained. DSA pulse at native 32.8 µs is SMOOTH/resolved (no hidden
  substructure). The remaining χ²_D=3.05 issue is coherent bright-pulse
  profile structure within the beta≈4 exponential/EMG-preferred branch, not
  evidence that EMG is the wrong branch and not a clear extra-component demand.
  Hence the t_factor=1 experiment.
- casey: two CHIME comps resolved at 10.908/11.135 (Δ=0.227 ms > dt_min=0.15),
  zeta_C2=0.0998 touching its 0.10 upper bound (widen if rerun). Only burst
  with β interior (3.32).

## 5. Round 2 — IN FLIGHT when handed off

SPEC in `scripts/refit_runner.py` already updated to:
- zach **C2D3**: + trailing CHIME window (15.0–17.5 ms, axis of pad=0.5 crop). Was
  ~45% converged (iter ~3800, checkpoint in sandbox /tmp — lost across sessions;
  restart is cheap, ~5–10 min on a Mac).
- hamilton **C3D1**: + third window (12.60–13.20). Not started.
- wilhelm **C2D2_tf1** (`t_factor dsa=1`, suffix `_C2D2_cwin_tf1`): not started.

Acceptance rule used: adopt refinement iff χ²/lag1 improve and no new rails;
else keep round 1.

## 6. Known caveats / open issues (be honest about these downstream)

1. **β rails at 4.0** (α→4.02) for hamilton/zach/wilhelm — same as the old
   sharedzeta finals. Multiplicity did NOT un-rail β. Whitney precedent also used
   beta_bounds=(3,4); widening the bound changes the question — coordinate with
   Jakob before touching it.
2. **lnZ with s2=None is NOT a valid Bayes factor across component counts**
   (improper flat gain prior). Selection here = residual whitening + resolved
   components, per whitney precedent. For evidence-based N selection set fixed
   `gain_s2` (proper prior) and hold it across N.
3. Posterior errors at nlive=160 look optimistic (t0 ±0.001 ms); fine for
   figure-quality models, not for citable uncertainties.
4. `_JointPriorTransformOrdered` takes the GROUP window from component 1's
   bounds; per-component windows are ignored for ordered groups.
5. Sandbox quirks (ignore on Mac): mounted files can't be `rm`'d; data symlinks
   need the Data--Faber2026 mount; bash calls hard-capped at 45 s.

## 7. Remaining pipeline steps once fits are accepted

1. Promote accepted PNG/PDF/SVG → `figures/jointmodel_pair/` (Jakob approved
   overwriting) + fit JSON/npz/scripts → `figures/jointmodel_pair/fit_artifacts/`
   (whitney pattern).
2. `beta_campaign_verdicts.json` suffixes for these bursts still point at the
   old fits (`_sharedzeta`, `_C1D1`) — the manuscript-side
   `plot_jointmodel_pair.py` will regenerate the OLD figures unless the rows'
   `suffix`/provenance are updated. Decide with Jakob how verdicts rows get
   refreshed (they carry α/β/χ² used elsewhere, e.g. beta_table.tex).
3. `KNOWN_MULTIPLICITY_FLAGS` in `plot_jointmodel_pair.py` should be updated or
   removed for fixed bursts; zach/wilhelm may need the same `pad_ms` fix there.
4. Captions in `sections/jointmodel_pairs.tex` are generic; no change needed
   unless component counts should be stated.
