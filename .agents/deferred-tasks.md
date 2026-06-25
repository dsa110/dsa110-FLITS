# Deferred-task ledger

Open follow-ups carried by a session. The deferred-task Stop gate
(`.claude/hooks/deferred-task-gate.sh`) blocks end-of-turn while any **unchecked**
item tagged `@agent` remains — work the agent can do itself. Policy:
[CLAUDE.md → "Deferred tasks gate completion"](../CLAUDE.md).

Tags (exactly one per item):
- `@agent` — the agent can execute/implement it now → **blocks** completion until done.
- `@human` — needs a person or a one-way door (push/publish/PR) → does not block.
- `@decision` — a product/science choice is pending → does not block.
- `@separate-lane` — belongs to another task's git lane → does not block.

To clear an `@agent` item: finish it and change `- [ ]` to `- [x]`. Only retag to a
non-blocking tag if it genuinely cannot be done by the agent now.

## Open

These are the residual *execution* items behind the decision-map answers
([`docs/rse/specs/decision-map-manuscript-completion.md`](../docs/rse/specs/decision-map-manuscript-completion.md),
all 7 tickets resolved 2026-06-24). The science decisions are recorded in
ADR-0002/0003/0004; what remains is data-dependent regeneration, the active
scattering lane, and the separate Faber2026 repo — none agent-completable in a
docs close-out, so none are `@agent`.

- [ ] Fix the **s²-grid reproducibility hazard** (ADR-0003). **Guard landed 2026-06-24** — `_s2verdict.py` is now fail-closed: it adjudicates only the all-exp PBF family by default, excludes the legacy mixed-PBF JSONs (which omit `pbf_C/pbf_D`), and *refuses* to render a verdict when no all-exp fixed-s² grid is present rather than silently using the mixed grid (set `FLITS_S2_PBF=mixed` to inspect the non-canonical legacy grid explicitly). Also fixed the `_s2-<n>_pbf-…` tag-parse crash and added a PBF column to `_ladder.py`; `test_s2verdict.py` self-check passes and encodes the zach mixed-vs-all-exp verdict flip. **Remaining @human:** pull the canonical all-exp fixed-s² grid from HPCC (`/central/scratch/jfaber/flits-runs/.../*_s2-*_pbf-exp-exp.json`) so the default path can actually adjudicate zach C2D3 (today it correctly refuses — the grid is not local), and optionally delete the 48 stale mixed-PBF `*_s2-*.json` under the deletion-safety gate. Guard code is untracked in the active scattering-refit lane; commit it there, not in a docs commit.
- [x] **Faber2026 figure pass 2** (`alpha_pbf_systematic`: recolor zach grey, TNS labels from `burst_metadata`, mixed α from ALLEXP ladder + all-exp from `_a1_fits`) — regen script `plot_alpha_pbf_systematic.py` landed 2026-06-25; exported to `Faber2026/figures/alpha_pbf_systematic.{svg,pdf,png}`.
- [ ] Reconcile the **ADR-contradicting campaign counts** in `docs/codetection-science-plan.md` (line ~37 "11/12 joint scattering fits" vs casey-landed 12/12; line ~79 "Profile-bias α case study: zach 3.32→2.41") and `docs/rse/specs/plan-manuscript-completion.md` ("adversarially trusted α 3/12") against the now-committed ADRs: ADR-0003 supersedes all mixed-PBF α and moves the profile-bias demonstrator off **zach** onto **whitney**; ADR-0004 reclassifies 1.0 ≤ α < 2.0 as MARGINAL. The pure stub-framing references were corrected in-place 2026-06-24; these count/roster lines were deliberately left because their *corrected* values depend on the HPCC zach all-exp grid pull and the citable-α lock (#4). **@decision** — do not invent the corrected counts; reconcile once the citable-α set finalizes.

- [ ] **Joint-fit vector figures** (`tau_nu_ladder`, `joint_ppc_montage`) — vector regen attempted 2026-06-24 against the HPCC joint fits (`/central/scratch/jfaber/flits-runs/data/joint/`). `tau_nu_ladder.{pdf,svg,png}` is **faithful and ready** (npz-only: α johndoeII 1.06 / wilhelm 2.69 / phineas 3.82 / **oran 5.96 ceiling**; staged in this session's scratchpad). `joint_ppc_montage` / per-burst `*_joint_ppc` are **NOT regenerable cleanly**: the joint set has drifted from the Jun-19 PNGs (oran α flipped 1.4→6.0; wilhelm re-fit Jun 22 with poor χ²) and per-band χ² is crop-sensitive in ways **not recorded in the fit JSONs** (oran DSA χ² 5.32 crop-on → 1.14 crop-off; wilhelm gets *worse* crop-off → 29.8). **@decision** — pick the canonical fit set (base vs the active lane's `sharedzeta`/`pbf-exp-exp` refit) and pin per-fit on-pulse-crop provenance before overwriting the committed Jun-19 figures or wiring either into the manuscript; gated on the active scattering lane + citable-α (#4). Not overwriting `dsa_figs/` until adjudicated.
- [ ] **Push the Faber2026 `chime_subband_compare` fold** — committed locally `44db61a` on `draft/fork-b-finish` (budget.tex within-CHIME sub-band block + figure, vector PDF+SVG), tree clean, ahead of origin by 1. **@human** — `git push` to `origin/draft/fork-b-finish` syncs to Overleaf (one-way door).

## Done

- [x] Implement the **L1 sub-Kolmogorov floor** (ADR-0004): `ALPHA_MIN` 1.5→1.0 in `gate_joint_committed.py`, sub-Kolmogorov L3 flag, `test_gate_joint_committed.py` updated, `joint_gate_verdicts.{md,csv}` regenerated (2026-06-25; 11/11 MARGINAL, 0 L1 FAIL).
- [x] **Faber2026 Fork-B consistency pass** (2026-06-25): hybrid authority model in `CONTEXT.md`; abstract/conclusions N=6 energies + qualitative α + explicit pending; unified `tab:alpha` (9 rows incl. whitney); energies trust boundary (6 rows, exclusion caption); freya α=4.356 in `budget.tex`; population stats hedged in `results.tex`.
- [x] Regenerate `analysis/burst_energies/burst_energies.{json,tex}` with the energy trust boundary (#39). **Done 2026-06-24** on jakob-mbp using the local arc replica `~/Developer/dsa110-local-data/DSA_bursts/` (24 cubes; the "arc mount" reachable from here — `DATA_SOURCES.md` local replica), staged under `data/{dsa,chime}/`. Ran `python analysis/calculate_burst_energies.py` in the `flits` env. `--check` PASS; the quality gate refused the 3 FAIL joint fits (johndoeii, oran, whitney). Result: 6-burst energy table = {chromatica, hamilton, isha, phineas, wilhelm, zach} (verified against `burst_energies.json`); the gate excluded the 3 α<1.5 FAIL joint fits (johndoeii, oran, whitney per `joint_gate_verdicts.md`), and casey/mahi are absent (no qualifying spec-z / DSA fluence input). `quality_flag` stamped on every row (all MARGINAL), all calibrated, E_iso 4.6e38–1.1e41 erg. Artifacts now show as `M` in the working tree (tracked); commit/push left to the user per the push gate.
