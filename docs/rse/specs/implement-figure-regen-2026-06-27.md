# Implementation: FINAL figure regen (Phase 1 complete + Phase 2 joint figures)

---
**Date:** 2026-06-27
**Branch:** `feat/final-figure-regen` @ worktree `~/Developer/scratch/worktrees/flits-final-figure-regen`
**Plan:** [handoff-figure-regen-2026-06-27.md](handoff-figure-regen-2026-06-27.md)
**Validation:** [validation-figure-regen-phase1.md](validation-figure-regen-phase1.md)

---

## Phases completed

### Phase 1 — fig:budget τ overlay ✅ (merged PR #77 @ `068f6e79`)

- Code on `main`; fresh validation PASS (25 tests, 9 measured τ rows).
- Figure review PASS for budget PNG (prior session).

### Phase 2 — All-exp joint figures ✅ (this session)

| Step | Status |
|------|--------|
| HPCC npz pull (7 bursts) | ✅ `local_runs/data/joint/*.npz` |
| HPCC npy + configs pull | ✅ ~1.7 GiB staging under `local_runs/` |
| Commit generators | ✅ `_tau_ladder_allexp.py`, `_ppc_montage_allexp.py` (uncommitted on branch) |
| Regen tau ladder | ✅ `dsa_figs/tau_nu_ladder_allexp.{png,svg,pdf}` + canonical copies |
| Regen PPC montage + per-burst | ✅ `joint_ppc_montage_allexp.*`, 7× `*_joint_ppc.png` |
| Stamp onpulse_crop in PPC JSON | ✅ `{burst}_joint_ppc_allexp.json` sidecars in `local_runs/data/joint/` |
| Figure-review gate | ✅ `dsa_figs/figures.manifest.json` + `figures.review.json` |

## Commands (repro)

```bash
cd ~/Developer/scratch/worktrees/flits-final-figure-regen
conda activate flits
export FLITS_REPO=$PWD
export FLITS_RUNS=$PWD/analysis/scattering-refit-2026-06/local_runs
export DSA_FIGS=$PWD/analysis/scattering-refit-2026-06/dsa_figs

python analysis/scattering-refit-2026-06/joint_ladder/_tau_ladder_allexp.py
python analysis/scattering-refit-2026-06/joint_ladder/_ppc_montage_allexp.py
```

Staging: fit JSON from `_hpcc_pull/joint/`; npz/npy/configs rsync from `hpcc:/central/scratch/jfaber/flits-runs/`.

## Key outputs

- `analysis/scattering-refit-2026-06/dsa_figs/tau_nu_ladder.{png,svg,pdf}` (all-exp)
- `analysis/scattering-refit-2026-06/dsa_figs/joint_ppc_montage.{png,svg,pdf}` (all-exp)
- Per-burst PPC for 7 publishable nicknames

## Remaining (@human / Phase 3)

- Faber2026: copy budget + joint figures; update prose; `make all`; PR + Overleaf
- Full photo-z promotion `scratch/photoz-fix/` → `results/` (@decision)
- Open PR for Phase 2 generators + dsa_figs from `feat/final-figure-regen`

## Issues

- whitney DSA PPC panel visually poor (chi2=2.54) — documented in figure review; sightline remains railed exemplar per ADR-0005.
- wilhelm DSA chi2=4.55 elevated — shape still tracks; same as pre-regen mixed-PBF concern.

---

**Next:** `ai-research-workflows:validating-implementations` on Phase 2 when PR opened; Faber2026 lane @human.
