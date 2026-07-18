# Component-count ladder audit — how counts were chosen, and the D4 gap

**Date:** 2026-07-17. **Scope:** joint CHIME+DSA scattering mass-refit, 12 bursts.
**Purpose:** the honest, citable answer to "how were the per-burst temporal
component counts (C_n D_m) chosen?" — intended for the paper's methods appendix,
not just internal notes.

## The finding: D4 was never on the table
Across every fit launched in this campaign, the DSA (D-band) component count
histogram is:

| D-count | runs |
|--------:|-----:|
| D1 | 17 |
| D2 | 13 |
| D3 |  6 |
| **D4** | **0** |

`--components-D 4` was **never launched for any burst**. The ladder ceiling was
D3 (zach, phineas) and C4 (hamilton, CHIME side). Consequently no burst could
ever have been *found* to need a fourth DSA component — the hypothesis was never
tested.

## How counts were actually assigned (per burst)
Only two bursts had a genuine evidence ladder that compared neighboring counts:

- **isha** — D1 vs D2 at fixed s2 → D1 selected (D2 lost).
- **whitney β is gain-marginalization-dominated (task #12 classification).** At the
  evidence-confirmed C2D2 count, β sweeps the ENTIRE prior range purely by changing
  the gain-prior variance s2, with τ swinging 27× — the data do not constrain the
  screen index against the gain model:

  | gain treatment | β | τ_1GHz (ms) | lnZ |
  |---|---:|---:|---:|
  | flat / profiled (production convention) | 3.025 (+0.035/−0.018, **floor**) | 0.077 | 20160.07 |
  | s2 = 100 (regularized gain) | 3.429 (**interior**) | 0.037 | 20860.02 |
  | s2 = 10 (tight gain prior) | 3.988 (**ceiling**) | 0.989 | 15320.13 |

  This is a **gain-systematic-dominated** classification, NOT a “β=3 screen”
  measurement. **Valid comparisons:** the two fixed-s2 fits share a proper prior
  normalization, so s2=100 vs s2=10 IS a valid Bayes factor — ΔlnZ = **+5539.9**
  decisively favoring the interior (β=3.43) over the ceiling. The flat/profiled-gain
  lnZ (20160.07) uses a different (improper) gain-prior normalization and is NOT
  comparable to the fixed-s2 values, so no flat-vs-s2 Bayes factor is formed. Bottom
  line: among comparable gain priors the evidence prefers the interior screen; the
  production flat-gain “floor” is a convention-dependent corner of a gain-degenerate
  posterior, not a physical index. hamilton (C4D1, flat-gain β=3.003, τ→0) is being
  probed the same way (s2=10/100 + C5D1/C4D2 neighbors); if its floor also melts under
  a regularized gain, the floor-rail class dissolves into a shared whitney+hamilton
  gain systematic.
- **zach**: the D4 collapse at s2=100 and in the profiled C2D4 (ΔlnZ = −2.3 vs C2D3,
  chi²_D 1.13→1.12, screen params byte-identical, the +2.06 ms member left with a
  +6.6σ residual spike in BOTH D3 and D4 — figure `figs_multicomp/zach_d3_d4_resid.png`)
  is consistent with the resolution limit at 131 µs, not with the member being absent.
  Per the binning lever, the count is settled by a fine-binning refit, not this ladder.
  **Binning-lever diagnosis (2026-07-18):** the 131 us (t4) DSA binning is NOT set by
  the trailing-window cap (WIN_TRAIL_CAP_MS 30->12 is a no-op on the chosen t-factor).
  It is set by common_window=True, which unions DSA with CHIME; CHIME at 400-800 MHz
  scatters ~150x more, so its window runs to the ~44 ms record end and drags DSA to a
  44 ms window, so t_floor forces DSA to t4. Fix #1: common_window=False so DSA is
  independent. BUT (caught by owner review) DSAs own peak-anchored robust window is then
  only ~2.2 ms and TRUNCATES the cluster: the initial pulse sits at the peak and the
  +2.06/+2.52/+3.01 ms members fall OUTSIDE, because the ~1 ms quiet gap between the
  initial and the cluster exceeds WIN_MAX_GAP_MS (1.0). A truncated window breaks the
  count test by construction. Fix #2: a band-aware ENVELOPE window keyed on the full
  >5-sigma component span + margin, applied to the DSA band ONLY; CHIME keeps its
  original tail-following window (binning unchanged at t64/163.8 us, tail not clipped).
  Verified deployed-driver prep: CHIME 163.8 us (unchanged), DSA 32.8 us (t1, 4x finer)
  with a 5.90 ms window spanning [-1.4,+4.5] ms -> all four candidate components in-window.
  STANDING GUARDRAIL (owner): any windowed count test must first prove the window contains
  every candidate component; a peak-anchored window that keys on the brightest pulse only
  is the same failure class as the count shortfall itself. Fine pair (C2D3_fine/C2D4_fine,
  s2=100, _fine-suffixed) running (jobs 133/134); D4 verdict = fine-pair delta-lnZ only (guardrail 1).
- **mode-trap lesson for the methods appendix**: at s2=100, a lower-count fit can get
  trapped in a secondary (steep-β / runaway-τ) mode, inflating the apparent evidence gain
  of the next count. Read a count "win" as real only when the screen parameters are
  continuous across the count step; otherwise it is a mode difference, not a component.
