# Citable-α roster — canonical all-exp joint fits, uniform [1.0,6.0] floor

**LOCKED 2026-06-26** — authoritative copy: [ADR-0005](../../docs/adr/0005-citable-alpha-roster.md),
machine-readable: [`citable_alpha_roster.json`](citable_alpha_roster.json).

**Tiers:** this file preserves the legacy all-exp/free-α roster. The
2026-07-07 beta re-lock in `citable_alpha_roster.json` promotes JohnDoeII to
C2D2 for the beta-native morphology/τ product (`beta=3.936`, `alpha=4.07` as a
railed-hi limit, `tau_1GHz=2.219 ms`, `chi2_C/D=1.09/1.23`). The old JohnDoeII
C2D1 numeric row is superseded; do not quote the retired sub-K value as a
current result. A free-α all-exp/fixed-s² C2D2 citation remains a separate gate.

All fits: single-exponential PBF both bands (ADR-0003 canonical), alpha_bounds
[1.0,6.0] (ADR-0004 floor), gain-marginal (shared-zeta or per-component zeta).
Graded through `gate_one` (gate_joint_committed.py) with per-band reduced chi2
from `joint_ppc_multi.py` (matplotlib-free OLS gain recovery; dof=npix-7).

## Reading the gate

FINAL caps at **MARGINAL for the whole roster** by construction, for two reasons
that are NOT fit defects:
- **L3 τ×Δν is unevaluable** — needs a scintillation bandwidth (dnu_d) the joint
  scattering fits don't carry, so L3 caps at MARGINAL even for a perfect Kolmogorov α.
- **Sub-Kolmogorov α is the result.** L3 flags any α off the 3.5–4.5 Kolmogorov
  window as "off-Kolmogorov (inspect)" → MARGINAL. Most sightlines ARE shallow;
  that's the measurement, not a failure.

So the **citable criterion is L1 PASS + L2 acceptable (both bands χ² not FAIL) +
not prior-railed**, with α interior and tightly constrained. FINAL=MARGINAL with a
documented "off-Kolmogorov" reason = a real, well-fit, shallow α.

## Roster (α-sorted)

| burst      | α (med +e/−e)        | χ² C/D     | rail | L1   | L2       | verdict |
|------------|----------------------|------------|------|------|----------|---------|
| freya      | 4.355 +0.037/−0.037  | 1.30/1.03  | no   | PASS | PASS     | **CITABLE** — near-Kolmogorov |
| zach       | 3.319 +0.013/−0.013  | 2.30/1.30  | no   | PASS | MARGINAL | CITABLE (caveat: CHIME χ²=2.30 elevated) |
| chromatica | 3.284 +0.040/−0.040  | 1.14/1.16  | no   | PASS | PASS     | **CITABLE** |
| mahi       | 2.806 +1.660/−1.245  | 1.12/0.86  | YES  | PASS | PASS     | EXCLUDE — α unconstrained (rail, σ≈1.5) |
| oran       | 2.663 +0.161/−0.180  | 1.04/1.16  | no   | PASS | PASS     | **CITABLE** |
| wilhelm    | 2.557 +0.039/−0.039  | 1.14/4.55  | no   | PASS | MARGINAL | CITABLE (caveat) — α robust (per-band-zeta cross-check 2.625), DSA χ²=4.6 is bright-burst inflation, not α-driven |
| casey      | 2.396 +0.014/−0.015  | 1.41/0.99  | no   | PASS | PASS     | **CITABLE** |
| johndoeII  | C2D2 beta promoted; free-α all-exp pending | 1.09/1.23 beta PPC | railed-hi at β=4 | PASS | PASS | beta-native product accepted as α=4 limit; old C2D1/free-α sub-K row superseded |
| isha       | 5.302 +0.579/−3.515  | 1.33/0.93  | YES  | PASS | PASS     | EXCLUDE — DSA non-det, α rails 6.0 (σ−=3.5) |
| phineas    | 3.320 +0.060/−0.064  | 1.02/1.34  | no   | PASS | PASS     | **CITABLE** — C3D3 (3+3 comp), both bands clean; landed 16-core/52-min run |
| hamilton   | non-identifiable     | —          | —    | —    | —        | EXCLUDE from α — CHIME↔DSA component-correspondence ambiguity; per-band: CHIME τ≈0.020 ms (χ²=3.36, single-comp on a multi-comp band), DSA τ→0 (non-detection, upper limit) |

## Summary

- **Citable α (5 + candidates):** freya 4.36, zach 3.32 (caveat), chromatica
  3.28, oran 2.66, casey 2.40 are the legacy free-α roster values. The
  beta-native re-lock supersedes the old JohnDoeII C2D1/free-α sub-K row with a
  C2D2 β=4-limit result.
- **johndoeII** is no longer a current sub-K claim under the beta-native model:
  the promoted C2D2 product rails high at β=4 (`alpha=4.07` summary, quoted as
  an α=4 geometry-conditioned limit). A new all-exp/fixed-s² C2D2 free-α run
  would be needed before making any separate free-α statement.
- **wilhelm** resolved → CITABLE with caveat: α=2.557 (shared-zeta canonical) is
  robust — the per-band-zeta cross-check gives 2.625 with the SAME DSA χ²≈4.6.
  Adversarial fit-validation (independently re-derived) corrected my first read:
  the DSA χ² is NOT S/N inflation but a **coherent residual localized at the
  bright peak** (off-peak pixel RMS≈1.1 white; on-peak RMS≈2.9; freq-summed peak
  residual ±15σ dipole). This should not be read as rejecting the
  exponential/EMG branch; the later β-coherent campaign also drives wilhelm to
  the β≈4 exponential limit. The caveat is narrower: within the preferred
  exponential-tail description, the adopted component model leaves residual
  bright-pulse profile structure. Cite any restored wilhelm scattering statement
  with that honest caveat, NOT "DSA well fit."
- **mahi, isha** excluded: α prior-railed / unconstrained (DSA non-detections).
- **phineas** landed (C3D3, both bands clean, χ² 1.02/1.34): α=3.32. **hamilton**
  excluded from α — per-band τ only (CHIME ≈0.020 ms, DSA non-detection upper limit).

**Tier A (5, cite now):** casey 2.40, wilhelm 2.56 (caveat), chromatica 3.28, zach 3.32
(C1D1 only), freya 4.36.

**Tier B (legacy all-exp/fixed-s²):** oran 2.66 and phineas 3.32 still need
their all-exp s² adjudication. JohnDoeII's beta-native C2D2 morphology/τ product
is promoted, but the old free-α C2D1 sub-K row remains retired rather than
restored.

**Multiplicity exemplar:** whitney α=5.12 (C2D2, prose). **Target manuscript N=8** once
Tier B adjudicates; **N=5** safe today.
