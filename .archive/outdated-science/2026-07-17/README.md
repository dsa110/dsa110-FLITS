# Outdated science-result quarantine — 2026-07-17

**Do not cite or consume these files as current science.** They are retained
byte-for-byte so their provenance and failure modes can be re-examined before a
later deletion or rehabilitation decision.

Moving a file back to its original path does not restore scientific trust. A
replacement must pass the current evidence gates and receive its own review.

| Original path | Quarantined path | Reason | Re-examination prerequisite |
|---|---|---|---|
| `analysis/beta_campaign/two_screen_consistency.{json,md}` | same path below this directory | Free-alpha joint-fit tau was incorrectly paired with DSA bandwidths | Fixed-index `tau_consistency` refits and current qualified bandwidths |
| `results/joint_fit_summary.md` | same path below this directory | Legacy trust claims were revoked by the July trust reset | Complete post-PL-PBF production campaign and adjudication |
| `galaxies/foreground/data/sightline_attribution_matrix.csv` | same path below this directory | Used retired tau/bandwidth values and stale screen-attribution rules | Regenerate from current evidence ledger and fixed-index screen products |
| `analysis/chime-scintillation/{README.md,INVENTORY.yaml}` | same paths below this directory | Claimed zero qualified CHIME measurements before the finalized campaign | Use `analysis/window-tuning-campaign-2026-07-17/results/` |

Historical generators remain executable for provenance, but their default
destinations now point below `regenerated/` so obsolete products cannot silently
reappear at live paths or overwrite the frozen snapshots above.
