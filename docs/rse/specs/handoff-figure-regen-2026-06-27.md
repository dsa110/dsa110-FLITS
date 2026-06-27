# Handoff: FINAL figure regen + fig:budget overlay

---
**Date:** 2026-06-27
**Author:** AI Assistant
**Status:** Ready for fresh `/implement` session
**Branch:** `main` @ `69efc4b7` (PR #75 merged)
**Prior handoff:** [handoff-2026-06-26-23-43-pass2-closeout.md](handoff-2026-06-26-23-43-pass2-closeout.md)

---

## Task(s)

Pass 2 docs closeout is complete. The next tranche is **FINAL figure regeneration** (all-exp joint figures → `dsa_figs/` → Faber2026) and **measured-vs-predicted budget overlay** on `fig:budget`.

| Task | Status | Tag | Notes |
|------|--------|-----|-------|
| PR #75 Pass 2 handoff docs | ✅ Merged | — | [PR #75](https://github.com/jakobtfaber/dsa110-FLITS/pull/75) @ `69efc4b7` |
| All-exp joint figure regen | 📋 Blocked | `@decision` | Generation done 2026-06-24; **placement** gated on citable-α lock (now ADR-0005) |
| fig:budget measured τ overlay | 📋 Blocked | `@decision` + `@human` | `make_budget_figure` supports overlay; needs photo-z budget promotion + joint τ wiring |
| Promote photo-z budget → `results/` | 📋 Blocked | `@decision` | `scratch/photoz-fix/` canonical; committed `results/` stale |
| Faber2026 figure copy + Overleaf | 📋 Pending | `@human` | Separate repo lane |
| PR #73 gdrive authority | 🔄 Separate lane | `@separate-lane` | Do not touch |

**Current workflow phase:** Implement (figure campaign) — blocked on `@decision` gates, not `@agent`.

## Deferred-task ledger verdict

`.agents/deferred-tasks.md` has **zero open `@agent` items** for figure work. Relevant open items:

- **Joint figures (#31):** `@decision` — generation complete; commit generators + place in `dsa_figs/` gated on citable-α lock (ADR-0005 now locks roster).
- **Photo-z budget (#37):** `@decision` — promote `scratch/photoz-fix/` → `results/` before regen; Overleaf push `@human`.
- **Campaign count reconciliation (#29):** `@decision` — prose only; after FINAL regen.
- **Stale s² JSON cleanup:** `@human` — optional deletion under safety gate.

**Conclusion:** No `@agent` figure work to start autonomously. Fresh session should `/implement` from this handoff after human clears the `@decision` promotion gate (or explicitly authorizes regen on scratch state).

## Critical references

- `.agents/deferred-tasks.md` — tag inventory (read Open section)
- `analysis/scattering-refit-2026-06/joint_ladder/ALLEXP_PBF_RUN.md` — **Manuscript joint figures — regeneration runbook** (L131+)
- `docs/adr/0005-citable-alpha-roster.md` — Tier A/B α roster (locked 2026-06-26)
- `galaxies/foreground/sightline_budget.py:744` — `make_budget_figure()` already plots measured τ when `tau_obs_ms` populated
- `Faber2026/sections/results.tex:36-38` — prose expects measured overlay on `fig:budget`

## Reproducibility & data state

- **Conda env:** `flits` (Python 3.12)
- **HPCC fits:** `hpcc:/central/scratch/jfaber/flits-runs/data/joint/` — all-exp `*_pbf-exp-exp.{json,npz}`
- **Local pulls:** `analysis/scattering-refit-2026-06/_hpcc_pull/joint/` (7 publishable bursts + s² grids in `joint_ladder/`)
- **Canonical α JSONs:** `analysis/scattering-refit-2026-06/_a1_fits/` + `citable_alpha_roster.json`
- **Photo-z budget (canonical manuscript state):** `scratch/photoz-fix/` — **not** in committed `results/`
- **Committed stale figures:** `analysis/scattering-refit-2026-06/dsa_figs/*.png` — base/mixed-PBF; **do not use for manuscript**

### Seven publishable all-exp bursts (ALLEXP verdict)

freya, casey, chromatica, wilhelm, oran, phineas, whitney — excluded: johndoeII, hamilton, isha, mahi, zach (railed/PBF-unstable).

## Verification state / known-broken

- **Tests:** `main` green post PR #75 merge (Python 3.10/3.12 + review passed before merge)
- **Figure generators:** `_tau_ladder_allexp.py`, `_ppc_montage_allexp.py` exist in refit lane scratchpad — **not committed** on `main` (deferred-tasks #31)
- **Montage assembler:** `joint_ppc_montage` script noted as uncommitted in ALLEXP runbook L151
- **Budget figure:** Right panel shows predicted τ only — measured columns empty because `results/` uses old foreground CSVs and joint τ not wired through `read_measured_tau_ms` for all-exp fits
- **Separate lanes preserved:** PR #73 `docs/gdrive-authority`; Faber2026 untracked `.remember/` — do not merge or sweep

## Learnings

- `make_budget_figure` (L839+) already implements measured-vs-predicted scatter with error bars when `tau_obs_ms` is finite and `tau_obs_quality=PASS`; overlay is a **data/promotion** problem, not a missing plotter.
- All-exp joint figures were rendered 2026-06-24 to HPCC scratch + session scratchpad; committed `dsa_figs/` intentionally left as mixed-PBF baseline.
- χ² on PPC panels requires on-pulse crop ON (`FLITS_ONPULSE_CROP=1`); crop setting not stamped in fit JSONs — regen should stamp into PPC JSON (runbook L160-162).
- `batch_jointmodel.py` + `plot_jointmodel_montage.py` provide Pass-2 joint-model export path using `_a1_fits/` CANON map — alternative to `_figs.py` ladder path.

## Action items — `/implement` steps for fresh session

**Prerequisite (@human):** Authorize photo-z promotion to `results/` OR explicit "regen from `scratch/photoz-fix/` only, no commit yet."

**Worktree (required — NOT gdrive lane):**

```bash
git worktree add ~/Developer/scratch/worktrees/flits-figure-regen -b feat/final-figure-regen origin/main
cd ~/Developer/scratch/worktrees/flits-figure-regen
conda activate flits
```

### Phase 1 — Budget overlay (`fig:budget`)

1. [ ] Promote foreground CSVs: `scratch/photoz-fix/*_galaxies.csv` → `results/` (Casey PSF star already removed per deferred-tasks)
2. [ ] Wire all-exp joint τ into budget build: ensure `build_sightline_budget` / `read_measured_tau_ms` reads from `_a1_fits/` or `citable_alpha_roster.json` (Tier A+B only; withhold FAIL/marginal per ADR-0005)
3. [ ] Regenerate: `python -m galaxies.foreground.sightline_budget` → `results/sightline_dm_scattering_budget.{svg,png,csv,md}`
4. [ ] Figure-review gate: write `results/figures.manifest.json` + visual review → `figures.review.json`
5. [ ] Copy PNG/SVG to `Faber2026/figures/sightline_dm_scattering_budget.*` (Faber2026 lane, `@human` push)

### Phase 2 — All-exp joint figures

1. [ ] Pull remaining all-exp JSONs/npzs from HPCC if missing locally (`_hpcc_pull/joint/` has partial set)
2. [ ] Commit generators on feature branch: `joint_ladder/_tau_ladder_allexp.py`, `joint_ladder/_ppc_montage_allexp.py` (or use committed `_figs.py` + `joint_ppc.py` per runbook)
3. [ ] Regenerate for 7 publishable bursts:
   - `tau_nu_ladder_allexp.{pdf,svg,png}`
   - `joint_ppc_montage_allexp.{pdf,svg,png}`
   - per-burst `*_joint_ppc.png`
4. [ ] Stamp `onpulse_crop` + prep settings into PPC JSON outputs
5. [ ] Replace `analysis/scattering-refit-2026-06/dsa_figs/` (after figure-review gate)
6. [ ] Copy to Faber2026 `figures/` + update `alpha_pbf_systematic` if population α panel needs FINAL values (`plot_alpha_pbf_systematic.py`)

### Phase 3 — Manuscript sync (@human one-way doors)

1. [ ] Faber2026: update `fig:budget` caption (remove "not yet overlaid" clause)
2. [ ] Faber2026: drop abstract/conclusions "pending FINAL regen" qualifiers where figures now land
3. [ ] `make all` on Faber2026; PR + Overleaf push

### Phase 4 — Cleanup (optional)

1. [ ] Reconcile campaign counts in `docs/codetection-science-plan.md` (deferred-tasks #29)
2. [ ] Delete 48 stale mixed-PBF `*_s2-*.json` under deletion-safety gate (`@human`)

**Recommended next skill:** `ai-research-workflows:implementing-plans` — invoke as `/implement docs/rse/specs/handoff-figure-regen-2026-06-27.md` after `@decision` promotion gate cleared.

## Separate lanes — do not touch

| Lane | Branch / location | Action |
|------|-------------------|--------|
| GDrive authority | PR #73 `docs/gdrive-authority` | Preserve |
| Faber2026 agent memory | `Faber2026/.remember/` untracked | Preserve |
| Codetection | `feat/codetection-freya` | Preserve |

## Other notes

- Pass 2 closeout handoff [PR #75](https://github.com/jakobtfaber/dsa110-FLITS/pull/75) merged 2026-06-27; supersedes open items in pass2-closeout handoff "Next tranche" section.
- User authorized one-way merge for docs-only PR #75; figure/manuscript placement remains `@human`/`@decision`.
- If proceeding without photo-z promotion: regen budget from `scratch/photoz-fix/` into a scratch output dir first; do not overwrite committed `results/` without explicit decision.

---

**Handoff created 2026-06-27 — route fresh session via `/implement`**
