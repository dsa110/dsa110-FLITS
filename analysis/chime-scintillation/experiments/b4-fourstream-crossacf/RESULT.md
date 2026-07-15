# B4 four-stream cross-ACF

**Status: diagnostic only; documented failure. This is not a CHIME
scintillation-bandwidth measurement.**

B4 uses polarization and early/late time splits plus leave-one-window-out
off-pulse template subtraction. The corrected independent noise null passes,
and `m = 1.0` injections recover at widths of 3, 6, 10, and 16 native channels.
However, recovery fails at `m = 0.15`, `0.20`, and `0.30`. The real burst has
only about `m = 0.15-0.17`, outside the validated envelope. Its diagnostic
Lorentzian width is boundary-dependent and less precise than its fitted width.

## Artifacts

- [Full result narrative](../../../chime-recovery-2026-07-12/results/b4_fourstream/RESULT.md)
- [Machine-readable validation](../../../chime-recovery-2026-07-12/results/b4_fourstream/validation.json)
- [Figure manifest](../../../chime-recovery-2026-07-12/results/b4_fourstream/figures.manifest.json)
- [Independent figure review](../../../chime-recovery-2026-07-12/results/b4_fourstream/figures.review.json)
- [Corrected off-pulse null](../../../chime-recovery-2026-07-12/results/b4_fourstream/figures/freya_b4_offpulse_null.png)
- [Injection recovery](../../../chime-recovery-2026-07-12/results/b4_fourstream/figures/freya_b4_injection_recovery.png)
- [Diagnostic on-pulse ACF](../../../chime-recovery-2026-07-12/results/b4_fourstream/figures/freya_b4_onpulse_acf.png)

The passing high-modulation injections demonstrate a bounded estimator regime;
they do not qualify the lower-modulation real-data fit.
