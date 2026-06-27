# Research: Survey of apparently-incomplete work in FLITS

**Date:** 2026-06-24
**Scope:** internal codebase
**Codebase state:** commit `3d27970`, branch `feat/figure-vector`
**Related Documents:** `docs/codetection-science-plan.md`, `docs/rse/specs/plan-manuscript-completion.md`, `.agents/deferred-tasks.md`, `analysis/scattering-refit-2026-06/JOINT_FIT_STATE.md`

## Question / Scope

"Identify work that appears to be incomplete." In-scope: any signal of unfinished
work reachable from the working tree at `3d27970` — code stubs / earmarks,
`TODO`/`FIXME`/`TBD` markers, skipped or stub tests, science campaigns with partial
coverage, uncommitted/WIP git lanes, open issues/PRs, and documentation that
describes the code as less finished than it now is. No external prior-art pass.

A finding here means *appears incomplete and worth a decision* — not *is a bug*.
Several items are intentional earmarks or pending science decisions; those are
labelled as such rather than as defects. Detailed remediation design is deferred to
planning.

## Codebase Findings

### 1. Deferred-task ledger — 1 open item (non-blocking)

`.agents/deferred-tasks.md:19` carries one open follow-up, tagged `@decision`:
reconcile the manuscript energies table (`Faber2026 sections/results.tex`
`tab:burst-energies`, abstract "eight … energies") to the regenerated 6-burst
`analysis/burst_energies/burst_energies.{tex,json}`, and add the per-section
exclusion caption. Blocked on a science/naming call (6-vs-8 row set; nickname↔TNS
map under active churn), so it does not block end-of-turn. This is the only ledger
item and it is correctly classified.

### 2. Open GitHub issues and PRs

- **Issue #4** — "Joint multi-component fit: make N=1 evidence commensurable for
  model selection." An implementation already exists *uncommitted* in
  `analysis/scattering-refit-2026-06/joint_ladder/_s2verdict.py` (fixed-s2 cross-N
  Bayes factor: "profiled-s2 lnZ is … NOT comparable across component count N").
  The issue is open while its fix sits untracked (see §5).
- **Issue #5** — "`burstfit_joint`: `dt_min` comment contradicts `max(dts)`; consider
  per-band `dt_min`." Open; not yet addressed in the tracked `burstfit_joint.py`.
- **PRs #47** (`feat/figure-vector`, the current branch), **#49**
  (`figures/vector-clean`), and **#50** (`docs/handoff-figures-2026-06-24`) are an
  active same-day (2026-06-24) figures/docs lane, overlapping in subject. Whether #49
  supersedes #47 and how #50's handoff relates is unresolved — a branch-hygiene
  decision, not code. This lane was moving *during* this survey (HEAD advanced
  `3d27970`→`a25bce0`, a `docs(claude)` CLAUDE.md auto-commit, and #50 opened mid-pass);
  it is a separate-active lane — observed, not touched. The code findings here are
  unaffected (`a25bce0` changed only `CLAUDE.md`).

### 3. Intentional code earmarks (flagged "not implemented")

These are placeholders the authors left deliberately and labelled — incomplete by
intent, not by oversight:

- `scattering/scat_analysis/pipeline/core.py:1004-1025` — four CLI flags whose help
  text ends "(not implemented)": `--anisotropy` enable, anisotropy axial ratio,
  polynomial baseline order to marginalize, and AR(1)/GP residual model. They parse
  but do nothing.
- `simulation/wave_optics.py:142` — "Placeholder for strictly user-requested 1D
  generation"; the 1D path is a stub. `docs/codetection-science-plan.md:25` also
  marks `wave_optics.py` / `multifreq_analysis.py` (Fresnel, "Gpc-infeasible") as
  Incomplete.
- `--auto-components` "placeholder for greedy BIC selection (earmarked)"
  (`docs/README_personal_fork_reference.md:118`).
- `analysis/calculate_burst_energies.py:189` — `raise NotImplementedError("fluxcal
  selected but no fluence_fn supplied for this band")`: a guarded path, not a stub,
  but worth noting as an unbuilt branch.

### 4. Science campaigns with partial coverage

Tracked as remaining work in `docs/codetection-science-plan.md` §C and
`docs/rse/specs/plan-manuscript-completion.md`. The *tooling* for several now
exists; the *coverage* is partial:

- **Scintillation Δν measured 3/12** (casey, freya, wilhelm). The other 9 are blocked
  on hand-tuned scint configs (RFI / manual burst windows), not on code — see §6 for
  why the "config-generation stub" framing is now stale.
- **Two-screen layer present but not wired** — consistency relation, ν-scaling, and
  modulation→size functions live in `scintillation/scint_analysis/analysis.py` but
  are not called by the pipeline (`docs/codetection-science-plan.md:23,49`).
- **NE2025 Galactic-floor integration** not wired
  (`scintillation/.../ne2025/`, plan §C.2).
- **Probabilistic host-DM treatment** — negative host DM diagnosed as expected
  Macquart-mean scatter, not a bug; the deliverable (subtract `p(DM_cosmic|z)`,
  report host DM as a posterior/upper limit) is unbuilt (plan §B, §C.6).
- **`crossmatching/` geometric-delay localization "remains unbuilt"** (plan §C.7);
  association significance + TOA cross-match are done.
- **ACF anomaly re-validation harness** (RFI / self-noise / off-pulse for the 3
  measured Δν) not built (plan §C.4).
- **Isotropic energies 6/12** (the §1 ledger item).

These are science-decision-gated, not autopilot tasks; the plan labels them so.

### 5. Uncommitted / WIP git lanes (in-progress, not landed)

The largest pool of "incomplete" signal is work that exists in the working tree but
is not committed:

- `analysis/scattering-refit-2026-06/joint_ladder/` is **fully untracked** — the
  joint CHIME–DSA scattering-ladder campaign: scripts `_ladder.py`, `_s2verdict.py`
  (issue #4's fix), `_figs.py`, plus ~50 `*_joint_fit*.json` outputs covering **all
  12 bursts** (casey included). This is an active analysis lane, not yet captured in
  git or reconciled with the committed plan.
- `eed6f04 "WIP snapshot (mac): in-progress FLITS work, preserved after
  concurrent-session clobber"` is the most recent commit to touch
  `flits/batch/batch_runner.py` — it is where the scint config-discovery wiring +
  test landed. It is an in-history WIP-labelled commit, **not** the branch tip (the
  tip at survey time was the clean `3d27970`); flagged because a WIP-labelled commit
  carries the scint refactor that the planning docs still call an unbuilt stub (§6).
- Modified-uncommitted: `galaxies/v2_0/sightline_budget.py`,
  `docs/entire-tracing-checkpoints.md`, `analysis/burst_energies/figures.review.json`,
  `.agents/deferred-tasks.md`.
- Stray untracked `.scratch/network_search_excess.py` (scratch lane).

### 6. Documentation that lags the code (confirmed-stale references)

Verified directly against the tree at `3d27970`. Each of these describes the code as
less finished than it is:

- **`batch_runner.py:262,275` scint "config-generation stub"** — RESOLVED. The file
  is now 456 lines with `_run_scintillation_analysis` and `discover_scint_configs`
  (configs are intentionally *discovered* — hand-tuned — not generated), plus
  `flits/batch/tests/test_scint_config_discovery.py` (commit `eed6f04`). There is no
  `# TODO: Add scintillation config generation` anywhere in the file. Still cited as
  an open stub at `docs/codetection-science-plan.md:51` and
  `docs/rse/specs/plan-manuscript-completion.md:71`. (The *campaign* is still 3/12;
  only the "missing code" framing is stale.)
- **`analysis_logic.py:110` "τ(ν) batch placeholder"** — the function at that line,
  `check_tau_deltanu_consistency`, is implemented (computes τ×Δν products, errors,
  thin-screen implied τ). Cited as a placeholder at
  `docs/codetection-science-plan.md:55`.
- **`burstfit_joint.py` "NOT yet committed"** (`JOINT_FIT_STATE.md:60`, and §A row of
  the science plan calling `joint` "not in main flow") — the module is tracked
  (43 KB, `git ls-files` confirms), so the "not yet committed" note is stale.
  *Caveat (verified):* the "11/12 joint fits" claim is **not** stale — the committed
  `analysis/scattering-refit-2026-06/joint_json/` holds the canonical c0/γ joint
  fits for 11 bursts (casey has only single-band multiscale output, no joint c0/γ
  fit). The untracked `joint_ladder/` (12/12 incl. casey) is a *separate, newer
  gain-marginal* ladder campaign, not the same fit type — so "11/12 canonical c0/γ"
  and "12/12 gain-marginal ladder" coexist correctly.
- **`docs/architecture/inventory.md:261-264`** — burst dirs hamilton / phineas /
  whitney / oran listed as "(files TBD)".

### 7. Reference into external code (out of this repo)

`docs/rse/specs/research-chime-singlebeam-flux-units.md:56` cites
`calibration.py:79` `# TODO: In principle, take sensitivity weighted average`. There
is no `calibration.py` in this tree (`fd`/`rg -uu` find none) — it is CHIME baseband
code that runs in the CANFAR image, not an in-repo incompleteness. Noted so it is not
mistaken for a local stub.

### Test-coverage gaps (context, mostly conditional skips)

Most skipped tests are environment-gated, not unfinished: `pytest.importorskip`
(emcee, pygedm) and data-presence skips (`tests/test_flux_cal.py:126,146`,
`tests/test_association.py:176`, `tests/test_chime_singlebeam_toa.py:62`). Two are
hard skips worth a glance: `scattering/scat_analysis/tests/test_priors_physical.py:350`
(`skipif(True, reason="Requires bursts.yaml")`) and the manual-only integration test
`scattering/scripts/test_run_scattering_analysis.py:469-472`.

## Synthesis

The incomplete work splits cleanly into four buckets by *what kind of decision
unblocks it*:

1. **Just needs landing** — the `joint_ladder/` gain-marginal ladder campaign (12/12
   fits + the `_s2verdict.py` cross-N Bayes-factor *diagnostic*). The work exists and
   appears done; it is uncommitted and unreconciled with the committed `joint_json/`.
   *Note (verified):* `_s2verdict.py` is a downstream robustness diagnostic, **not**
   the fix for issue #4 — #4 needs an N=1-routing change in `burstfit_joint.py`
   (~line 673) so 1- and N-component lnZ are commensurable. Landing the lane does not
   close #4.
2. **Needs a science/product decision** — energies 6-vs-8 + naming (§1), probabilistic
   host-DM model, which bursts get deep localization. Correctly parked as `@decision`.
3. **Genuinely unbuilt tooling** — two-screen wiring, NE2025 floor, geometric-delay
   localization, ACF re-validation harness, the `pipeline/core.py` earmark flags.
   Low-to-moderate effort; the science plan §C orders them by leverage.
4. **Documentation drift** — four confirmed-stale references (§6: the two scint-stub
   citations, the `analysis_logic.py:110` placeholder note, the `burstfit_joint.py`
   "not yet committed" note) plus `inventory.md` TBD dirs. These are the cheapest to
   close and the most misleading if left, because they make the codebase read as less
   finished than it is and could send someone to "fix" code that is done. (The "11/12"
   count is *not* drift — see §6 caveat.)

**Gaps / open questions for planning:**
- The `joint_ladder/` lane is ready to commit as analysis artifacts, but does **not**
  close issue #4 (verified: `_s2verdict.py` is a diagnostic; #4 is a `burstfit_joint.py`
  N=1-routing fix + regression test). Plan them as two separate phases.
- Resolve the #47 / #49 / #50 figures-docs lane overlap (supersede or land each).
- The §6 stale references are confirmed; correcting them is a small, isolated
  follow-up (re-point the two scint-stub citations to "code done, 9 configs pending";
  drop the `analysis_logic.py:110` placeholder note; update `burstfit_joint.py`
  committed-status; fill `inventory.md` TBD dirs). Flagged rather than edited here to
  keep this pass a research artifact, not a doc rewrite.

**Light recommendation:** sequence as (4) doc reconciliation → (1) land/verify the
joint lane → (3) tooling by plan §C order, with (2) decisions surfaced to the user as
they gate each science campaign. No detailed design here — that is planning's job.

## References / Sources

Code & artifacts (all at `3d27970`):
- `.agents/deferred-tasks.md:19`
- `scattering/scat_analysis/pipeline/core.py:1004-1025`
- `simulation/wave_optics.py:142`
- `analysis/calculate_burst_energies.py:189`
- `flits/batch/batch_runner.py` (456 lines; `_run_scintillation_analysis`,
  `discover_scint_configs`), `flits/batch/tests/test_scint_config_discovery.py`
- `flits/batch/analysis_logic.py:109` (`check_tau_deltanu_consistency`)
- `scattering/scat_analysis/burstfit_joint.py` (tracked)
- `analysis/scattering-refit-2026-06/joint_ladder/` (untracked: `_ladder.py`,
  `_s2verdict.py`, `_figs.py`, `*_joint_fit*.json` ×~50, 12 bursts)
- `scattering/scat_analysis/tests/test_priors_physical.py:350`,
  `scattering/scripts/test_run_scattering_analysis.py:469-472`
- Git: branch tip `eed6f04` (WIP snapshot); modified `galaxies/v2_0/sightline_budget.py`,
  `docs/entire-tracing-checkpoints.md`, `analysis/burst_energies/figures.review.json`
- GitHub: issues #4, #5; PRs #47, #49, #50 (open during survey; HEAD advanced to
  `a25bce0`, a CLAUDE.md auto-commit, mid-pass)

Docs surveyed (stale references called out in §6):
- `docs/codetection-science-plan.md:23,25,49,51,55`
- `docs/rse/specs/plan-manuscript-completion.md:70-71`
- `analysis/scattering-refit-2026-06/JOINT_FIT_STATE.md:19,56,60`
- `docs/architecture/inventory.md:261-264`
- `docs/rse/specs/research-chime-singlebeam-flux-units.md:56` (external `calibration.py`)
- `docs/README_personal_fork_reference.md:118`
