# B3 high-band polarization cross-ACF

**Status: diagnostic only; documented failure. This is not a CHIME
scintillation-bandwidth measurement.**

B3 cross-correlates the independently upchannelized polarization products.
Input provenance, producer parity, and burst alignment pass. The independent
off-pulse null fails because coherent low-lag structure remains in 11 of 12
control windows. Injection recovery also fails for every tested modulation
index `m = 0.3` cell; only the stronger `m = 1.0` simulations recover within
the fixed bias limits. Because prerequisite gates fail, the validator correctly
suppresses an on-pulse fit.

## Artifacts

- [Machine-readable validation](../../../chime-recovery-2026-07-12/results/b3_crossacf/validation.json)
- [Figure manifest](../../../chime-recovery-2026-07-12/results/b3_crossacf/figures.manifest.json)
- [Independent figure review](../../../chime-recovery-2026-07-12/results/b3_crossacf/figures.review.json)
- [Off-pulse null figure](../../../chime-recovery-2026-07-12/results/b3_crossacf/figures/freya_b3_offpulse_null.png)
- [Injection-recovery figure](../../../chime-recovery-2026-07-12/results/b3_crossacf/figures/freya_b3_injection_recovery.png)

The successful `m = 1.0` injection cells are qualified simulation diagnostics,
not a qualification of the real Freya product.
