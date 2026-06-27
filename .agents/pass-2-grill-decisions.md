# Pass 2 grill decisions (2026-06-27)

Locked via `/grill-with-docs` for autonomous Phases 1–3.

| # | Topic | Decision |
|---|-------|----------|
| 1 | **Energies roster** | Re-admit **oran + whitney** after HPCC all-exp c₀,γ pull → target **~8 rows** (6 + zach + oran + whitney if both qualify) |
| 2 | **α population stats** | **Qualitative** until all fixed-s² grids adjudicated |
| 3 | **s² campaign** | HPCC accessible — run fixed-s² for **oran, isha, mahi, phineas-DSA** |
| 4 | **c₀,γ authority** | **Pull fresh** from HPCC `/central/scratch/jfaber/flits-runs/data/joint/` before energy re-export |
| 5 | **Git packaging** | **Pipeline PR first**, Faber2026 stacked on submodule pin (ask-matt) |
| 6 | **zach tab:alpha** | **Drop row entirely** — no α quoted in table |
| 7 | **Abstract pending** | Swap gate-regen blocker → **fixed-s² pending for four sightlines** |

## Execution order

1. Pipeline branch `manuscript-pass-2-2026-06`: HPCC pull joint JSONs → gate regen → energy re-export → s² campaign → adjudicate
2. Faber2026 branch `draft/pass-2-closeout`: submodule pin + manuscript tables/prose

## Phase 1 done (2026-06-27)

- Branch `manuscript-pass-2-2026-06` from `manuscript-closeout-2026-06`
- HPCC rsync: 33 all-exp JSONs → `_hpcc_pull/joint/` (scattering-only; **no c0/γ** — verified on cluster)
- Gate regen: **11/11 MARGINAL, 0 L1 FAIL** (already committed on closeout branch)
- Energy re-export: **N=8** {chromatica, hamilton, isha, **oran**, phineas, **whitney**, wilhelm, zach}; c0/γ from mixed-legacy `joint_json/` (alpha-independent per ADR); `--check` PASS
- **Note:** all-exp HPCC pull feeds Phase 2 s²/α; energies correctly stay on mixed-legacy c0/γ until a full-amplitude all-exp fit exists (none on HPCC today)

## Phase 2 launched (2026-06-27)

- Script: `analysis/scattering-refit-2026-06/hpcc/submit_pass2_s2_allexp.sh` (also on HPCC `flits-runs/`)
- **First attempt:** 24 jobs (ids 64618351–64618374) **FAILED** — stale `/home/jfaber/flits/venv` (tree quarantined 2026-06-25)
- **Infra fix (2026-06-26):** symlinks restored (`/central/scratch/jfaber/envs/flits-joint`, `/home/jfaber/flits` → quarantine); `run_joint.sbatch` uses `PATH` not `source activate`
- **Resubmit:** smoke 64618456 (oran C1D1 s2=1) + 23 jobs 64618479–64618501; log `pass2_s2_allexp_submit.20260626T230812.ids`
- Outputs: `data/joint/{burst}_joint_fit_CxDy_s2-{s}_pbf-exp-exp.json`
- Legacy mixed-PBF s² grids already on cluster — **not** used (_s2verdict fail-closed)

## Phase 2 complete (2026-06-26)

- **24/24 COMPLETED** (64618456 + 64618479–64618501); 0 FAILED
- Rsync → `joint_ladder/` (24 new + 6 zach pre-existing)
- **`_s2verdict.py` (all-exp canonical):**
  - **oran** C2D1 vs C1D1: REAL (+184 / +198 / +203)
  - **isha** C2D1 vs C1D1: REAL (+391 / +487 / +545)
  - **mahi** C2D1 vs C1D1: weak (+52 / +52 / +0.7) — extra component not strongly supported
  - **phineas** C3D3 vs C3D2: NOT robust (sign flips)
  - **zach** C2D3 vs C2D2: NOT robust (confirms drop from `tab:alpha`)
