# H0 rank-1 correction smoke test

Status: **fail**; `science_status=diagnostic_only`.

The exact two-subband machine-readable gate was reconstructed on 2026-07-14
from the retained rank-1 corrected product and manifest using historical FLITS
commit `8f0479d`. The 513.631 MHz subband failed the off-pulse null (40.919 kHz
on-pulse versus 24.673 kHz off-pulse; ratio 1.658). The 713.439 MHz subband
passed (65.998 versus 29.756 kHz; ratio 2.218). Both subbands passed the low-lag
stability check, but the aggregate correction status is fail.

Canonical evidence:

- [`validation.json`](../../../chime-recovery-2026-07-12/results/h0/validation.json)
- [recovery-loop plan and original adjudication](../../../../docs/rse/specs/plan-chime-recovery-loop.md#h0-reproduction-evidence)

Reproduction requires the historical analysis surface because later FLITS
work removed the CHIME correction-adjudication fields consumed by this runner:

```bash
git worktree add --detach /tmp/flits-h0-reproduce 8f0479d
cd /tmp/flits-h0-reproduce
uv run --frozen python analysis/chime-recovery-2026-07-12/run_freya_gate.py \
  --product ~/Data/Faber2026/dsa110/scintillation-data/freya_chime_coarse_rank1_v1_corrected.npz \
  --manifest ~/Data/Faber2026/dsa110/scintillation-data/freya_chime_coarse_rank1_v1_manifest.json \
  --output /tmp/freya-h0-gate.json --subbands 2
```

This closes H0 as exact reconstructed diagnostics, not a qualified
scintillation measurement.
