# RFI-fix binning: screen-solution flips (before/after record)

The mask-aware RFI-fix binning does not merely improve fit amplitudes — on at
least one burst it **flips the screen solution to the opposite end of the prior**.
This is a headline methods result: a binning/RFI-handling change can change the
inferred turbulence regime, not just tighten error bars.

## phineas — confirmed flip (2026-07-17)
Source: session-log `[phineas] JOINT` lines (both fits). The pre-bugfix posterior
**samples were overwritten** at 16:45 by the production refit and were never
snapshotted (only casey/chromatica/wilhelm were captured to `_prebugfix/`), so a
full corner overlay is **unavailable**. Percentile-level record below; figure
`figs_multicomp/phineas_flip_beforeafter.png`.

| param | old (buggy binning) | new (RFI-fixed production) |
|---|---|---|
| β | 3.014 (+0.024/−0.010) — **at LOWER prior edge (3.0)** | 3.958 (+0.016/−0.031) — **near UPPER edge (4.0)** |
| α | 5.94 (+0.04/−0.09) — steep | 4.043 (+0.03/−0.02) — near-Kolmogorov |
| τ_1GHz | 0.725 (+0.013/−0.023) ms | 0.452 (+0.0096/−0.0088) ms |
| lnZ | 31718.6 | 30625.7 |

**Reading:** the old fit railed to the steep-index end (β→3, α≈5.9); the RFI-fixed
fit rails to the shallow/Kolmogorov end (β→4, α≈4.0). Both are BETA-AT-PRIOR-EDGE
— phineas rails in both cases, at *opposite* edges. lnZ is **not** comparable
across the two (different binning ⇒ different data ⇒ different likelihood
normalization); it is not evidence for either solution.

## Going forward: snapshot before overwrite (ALL bursts)
The phineas loss is why: any burst can flip, so the pre-refit posterior of every
burst must be preserved, not just the three RFI-fix bursts.

- **Retroactive snapshot taken 2026-07-17 17:15:** `data/joint/_prerefit_snapshot_20260717-1715/`
  (46 files, all current `*_joint_fit_*.json` + `*_joint_samples_*.npz`), captured
  before the still-running production jobs (johndoeII job 62, hamilton job 60,
  wilhelm job 69) could overwrite. In particular `johndoeII_joint_fit_C2D2.json`
  (09:24 version) is now preserved before job 62 overwrites it — watch johndoeII
  for a phineas-style flip when it lands.
- **Helper:** `jobs/snapshot_joint.sh` — timestamped copy of `data/joint` JSON+npz;
  run it before any manual refit (e.g. the zach binning-lever rerun).
