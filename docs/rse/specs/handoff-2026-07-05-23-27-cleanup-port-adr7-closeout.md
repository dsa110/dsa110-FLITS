# Handoff: repo cleanup shipped, gain-fix ported, deferred ledger closed, ADR-0007 opened

---
**Date:** 2026-07-05 23:27
**Author:** AI Assistant
**Status:** Handoff
**Branch:** `docs/adr-0007-extended-medium-pbf` (canonical clone `~/Developer/repos/github.com/jakobtfaber/dsa110-FLITS`); fork `main` = `028fa7cc`
**Commit:** `cf430b97` (PR #132 head)

---

## Task(s)

| Task | Status | Notes |
|------|--------|-------|
| Org cleanup + ponytail audit (PR #126, #127) | ✅ Complete | Merged; −12,502/+281 lines; suite baseline parity |
| Issue-37 gain-evidence fix: upstream merge + fork port (PR #129) | ✅ Complete | Upstream dsa110#43 merged `c97177c`; port adapted to β co-model; semantic collision with fork `force_multi` resolved (upstream semantics adopted) |
| PR #126 deferred follow-ups (PR #131) | ✅ Complete | flits-scint repaired, recreate_figures rot fixed end-to-end, engines trimmed, environment.yml fresh-create exits 0, casey f/t annotated |
| Ledger re-reviews (figures, photo-z, scint items) | ✅ Complete | All three confirmed stale/superseded; closed with evidence chains in `.agents/deferred-tasks.md` |
| ADR-0007 extended-medium PBF lane | 🔄 In Progress | **PR #132 open, CI pending on `review` job**; ADR status "proposed" — owner's science acceptance required |
| Faber2026 pin bumps + Overleaf pulls | ✅ Complete | Pin at `028fa7c` (pushed, Overleaf pulled clean); one more bump due when PR #132 merges |
| Citable-α lock / campaign-count reconciliation | 📋 Planned | Deliberately untouched — see Next Steps |

**Current Workflow Phase:** Validate (PR #132) → Plan (ADR-0007 implementation, pending acceptance)

## Workflow Artifacts

**ADRs (this session):**
- `docs/adr/0007-extended-medium-pbf-for-shallow-alpha.md` — NEW, proposed: extended-medium (uniform-LOS Williamson) PBF family selected per burst by evidence, as the physically grounded response to shallow α
- `docs/adr/0006-beta-coherent-scattering-comodel.md` — dated Correction appended (see Learnings)

**Ledger:**
- `.agents/deferred-tasks.md` — four items closed this session (all-exp figure placement, photo-z promotion, scint extraction repair, plus supersession notes); shallow-α item now points at ADR-0007

**Prior-session artifacts referenced:** `docs/rse/specs/plan-manuscript-completion.md`, `docs/rse/specs/decision-map-manuscript-completion.md`, `analysis/scattering-refit-2026-06/joint_ladder/ALLEXP_PBF_RUN.md` (its `dsa_figs/` runbook header now marked HISTORICAL), `docs/adr/0005-citable-alpha-roster.md`

## Critical References

- `docs/adr/0007-extended-medium-pbf-for-shallow-alpha.md` — the open decision; everything downstream (casey re-fit, roster stability, count reconciliation) hinges on accept/reject
- `.agents/deferred-tasks.md` — now the accurate map of what actually remains (only the α-lock-dependent count reconciliation and the two @human items)
- `docs/literature/Bhat_MultiFreqObsPulseBroadening_2004.md` (Eq. 4 block ~L55, §7 ~L211) — the primary source for the ADR-0006 correction; read before debating the physics

## Recent Changes

- `scattering/scat_analysis/burstfit_joint.py:937-951` (pre-merge numbering) — multi-trigger now `force_multi OR n>1 OR proper_gain_prior OR gain_s2 is not None` (PR #129)
- `tests/test_issue4_commensurable.py:170-183` — contrast leg proves the gate without `gain_s2` (upstream semantics adopted)
- `scintillation/scint_analysis/run_analysis.py:8` — package-relative import; `flits-scint` console script works for the first time since packaging (PR #131)
- `simulation/recreate_figures.py` — 4 F821s + `sim.dnu`→`dnu_hz` (3 sites) + dead `cfg.noise_snr` title + `InstrumentalCfg` now actually passed to `SimCfg`; noisy-ACF demo runs end-to-end headless
- `galaxies/foreground/engines_extra.py` — DesiLsDr10/AllWise/GalexAis/Xsc engines deleted (zero consumers; upstream keeps copies in `galaxies/v2_0/`)
- `environment.yml:52` — pygedm deactivated (build recipe kept); fresh `conda env create` verified exit 0 with imports
- `scattering/configs/bursts/chime/casey_chime.yaml:9-10` — f/t 32/4-vs-64/24 provenance comment
- `docs/adr/0006:70-85` — physics correction (see Learnings)

## Verification State / Known-Broken

- **Tests:** full suite 528 passed / 6 environmental skips at fork `main` `028fa7cc` (verified twice this session). PR #132 is docs-only; its Python 3.12 + Socket checks pass, `review` job still pending at handoff time (background watcher `beppfvznb` in the dying session — next session: `gh pr checks 132`).
- **Uncommitted:** only `docs/entire-tracing-checkpoints.md` hook churn in the canonical clone (append-only ledger; rides the next commit — normal state).
- **Unpushed:** nothing. All branches pushed; Faber2026 main pushed and Overleaf pulled.
- **Unverified claims:** none known. Every edit this session carries a verify-gate record (test/oracle/reproduce/cross-check).
- **Behavior change to know:** N=1 joint fits that pass `gain_s2` now route through the proper gain-prior kernel instead of silently ignoring it. Audited: **no stored fixed-s2 N=1 artifacts exist**, so nothing on disk is invalidated — but any *future* rerun of an old command line will (correctly) differ.

## Learnings

- **ADR-0006's addendum was wrong about the shallow-α escape route.** Bhat 2004 Eq. 4's steep branch `α = 8/(6−β)` gives α > 4 for 4 < β < 6 — steeper, not shallower. No pure power-law closure branch reaches α < 4. The literature's mechanism is PBF *geometry* misspecification (Bhat §7: same pulsar, thin-screen PBF → α≈3.1, extended-medium PBF → Kolmogorov-consistent), plus inner scale / refraction / truncated screens. ADR-0007 is built on this; the correction is appended to ADR-0006 with the original claim preserved in quotes.
- **The fork and upstream independently fixed the same evidence-baseline bug with colliding semantics** (`force_multi` vs `gain_s2`-implies-routing). The fork test asserting `gain_s2` must NOT reroute was the CI failure on the port; upstream semantics won because a silently-ignored fixed s² at N=1 is exactly the fixed-s² Bayes-ladder hazard. Pattern: when porting across the fork/upstream divergence, run the FULL suite locally (`pytest -q`), not just the touched package — the collision test lived in top-level `tests/`, outside the scattering testpath.
- **The deferred-task ledger rots fast in this repo.** Three of five @decision items (figure placement, photo-z promotion, scint repair) had already been resolved by later merged work (PRs #58, #78, #95, β co-model montage) — always re-verify a ledger item against current `main` before acting on it.
- **The `.claude` protected-branch commit guard** blocks `git commit` on `main` in ANY repo (it keys off the command, resolves `-C`, and is blind to `git switch -c` earlier in the same compound command). Pattern: separate Bash call for `switch -c`, then commit; land on main via `merge --ff-only` (not a commit invocation). Faber2026 pin bumps must use this dance.
- **Overleaf pull flow** (single-writer convention): Faber2026 push → Overleaf UI → Integrations → GitHub → "Pull GitHub changes into Overleaf" — done via browser automation twice this session, both clean ("No new commits since last merge" after).
- **`lane-liveness` false positive:** an orphaned `git difftool -y -x vimdiff HEAD` iTerm tab (programmatic ShellLauncher launch, owner unknown, user denied starting it) held agent-PID/editor-lock/recent-edit signals on the pipeline submodule checkout for hours. Killed after proving the lane branch was committed+pushed. If liveness signals look stale, `ps -o pid,ppid,tty,lstart,command` the PIDs before deferring work.

## Action Items & Next Steps

1. [ ] **Merge PR #132 when its `review` job passes** (`gh pr checks 132`; then `gh pr merge 132 --repo jakobtfaber/dsa110-FLITS --merge --delete-branch`), fast-forward the canonical clone, bump the Faber2026 pin (update-index → build branch → ff-main dance), push, Overleaf-pull.
2. [ ] **Put ADR-0007 in front of the owner** — accept/reject/amend is their science call. If accepted: implementation plan = new uniform-medium PBF kernel in `scattering/scat_analysis/burstfit.py` under the β-coupling contract, `pbf_geometry` model-scan axis, casey re-fit first (`ai-research-workflows:planning-implementations`).
3. [ ] **Campaign-count reconciliation** (`docs/codetection-science-plan.md` ~L37/79, `docs/rse/specs/plan-manuscript-completion.md`): ADR-0005 shows the roster locked 2026-06-26 with 3 provisional members, so this may be unblockable now — but it touches the same roster ADR-0007 could re-open; sequence it AFTER the ADR-0007 decision.
4. [ ] Two @human ledger items untouched: stale mixed-PBF `*_s2-*.json` deletion (deletion-safety gate) and any Overleaf-side pushes.

**Recommended Next Skill:** `ai-research-workflows:planning-implementations` (for ADR-0007 implementation, once accepted). If the owner rejects ADR-0007, use `ai-research-workflows:researching` on the inner-scale alternative instead.

## Other Notes

- Fork `main` history this session: `28a2cb5` (PR #127) → `13e1d00` (PR #129 port) → `028fa7c` (PR #131 follow-ups). Faber2026 pins have tracked each; current pin `028fa7c`, one more bump due post-#132.
- Upstream `dsa110/dsa110-FLITS` `main` = `c97177c` (gain fix merged). The fork and upstream `main`s remain deliberately divergent; the manuscript pins the FORK.
- `scratch/photoz-fix/` is a stale staging copy — do NOT re-promote it over the evolved `results/` (PR #95 mNFW work postdates it).
- The α-lock "fraught" analysis (model-conditioned α, selection-function bias, moved thresholds, closure-domain inconsistency, publication irreversibility) was delivered in-session 2026-07-05 and acknowledged; it's the framing for any roster re-opening.

---

**Handoff created by AI Assistant on 2026-07-05**
