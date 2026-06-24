# Citable-α roster — canonical all-exp joint fits, uniform [1.0,6.0] floor (2026-06-24)

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
| wilhelm    | 2.557 +0.039/−0.039  | 1.14/4.55  | no   | PASS | MARGINAL | SUSPECT — DSA χ²=4.55, shared-zeta law misfits DSA; α a forced compromise |
| casey      | 2.396 +0.014/−0.015  | 1.41/0.99  | no   | PASS | PASS     | **CITABLE** |
| johndoeII  | 1.529 +0.092/−0.088  | 1.04/1.27  | no   | PASS | PASS     | **CITABLE** — sole genuine sub-Kolmogorov, well-fit BOTH bands |
| isha       | 5.302 +0.579/−3.515  | 1.33/0.93  | YES  | PASS | PASS     | EXCLUDE — DSA non-det, α rails 6.0 (σ−=3.5) |
| phineas    | (refitting C3D3)     | —          | —    | —    | —        | PENDING — 64544411 (16 cores, 5h wall; prior C3D3 TIMEOUT at 30 min) |
| hamilton   | non-identifiable     | —          | —    | —    | —        | EXCLUDE from α — CHIME↔DSA component-correspondence ambiguity; per-band τ + DSA limit reported instead |

## Summary

- **Citable α (6):** freya 4.36, zach 3.32 (caveat), chromatica 3.28, oran 2.66,
  casey 2.40, johndoeII 1.53. Predominantly **sub-Kolmogorov**, tight grouping ~2.4–3.3.
- **johndoeII** is the sole genuine sub-K case (1.53) AND well-fit in both bands —
  the strongest shallow-α detection, not a rail artifact.
- **wilhelm** flagged: DSA χ²=4.55 under the shared-zeta law → α=2.56 suspect; needs
  a per-band-zeta cross-check before citing (decision pending).
- **mahi, isha** excluded: α prior-railed / unconstrained (DSA non-detections).
- **phineas** pending refit; **hamilton** excluded from α (single-band per-band τ only).
