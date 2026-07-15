# A1 trigger calibration

## Verdict

The calibration campaign completed successfully: all 68 declared cells are
present (60 null cells with 200 realizations each and 8 power cells with 100
realizations each), every stored value is finite, and the aggregate report
recomputes exactly from the cell checkpoints.

This is **not a qualified CHIME scintillation measurement**.  At the calibrated
1% false-escalation operating point, the envelope threshold is
`Delta ln Z = 59699.69283336272`, and all eight tested two-component power
cells have zero escalation probability.  The calibration therefore identifies
an unusably conservative trigger over the declared grid rather than validating
a burst detection.

## What drove the result

The 1% and 0.5% envelope thresholds are set by
`dnu/dchan = 100`, `num_subbands = 8`, `S/N = 10`; that null cell reaches
`Delta ln Z = 76122.16097938282`.  The 5% threshold is set by
`dnu/dchan = 10`, `num_subbands = 8`, `S/N = 100`.  This strong cell dependence
is visible in the operating curve and explains why a single envelope threshold
eliminates power across the tested alternative grid.

## Provenance

- Source host: `h17` (`lxd110h17`)
- Source report: `/home/ubuntu/a1-trigger-calibration/reports/a1_trigger_calibration.json`
- Source checkpoints: `/home/ubuntu/a1-trigger-calibration/reports/a1_trigger_calibration.cells/`
- Producer Git commit: `e0776116525ff17d4a85d1178e1883ca35dcaa21`
- Producer command: `simulation/scripts/run_a1_trigger_calibration.py --n-real 200 --n-real-power 100 --workers 36 --out reports/a1_trigger_calibration.json`
- Settings: `nlive=500`, `dlogz=0.1`, covariance calibration realizations
  `n_real_cov=150`, HWHM width convention

The selected calibration source files at the producer commit are tree-identical
to their rebased versions at `b899b5a`.  The full report and every checkpoint
are retained because the diagnostic figures are regenerated from the cell
samples, not only from the aggregate report.

## Artifacts

- `a1_trigger_calibration.json`: aggregate machine-readable report
- `a1_trigger_calibration.cells/`: 68 restartable per-cell checkpoints
- `figures/a1_null_dlnz_contactsheet.png`: per-cell null distributions and
  envelope thresholds
- `figures/a1_threshold_vs_rate.png`: false-escalation operating curve
- `figures/a1_power_curves.png`: power at the 1% operating point
- `validation.json`: independently recomputed consistency checks
- `artifact-manifest.json`: repository-relative SHA-256 inventory

Regenerate the figures from the repository root with:

```bash
uv run --frozen python simulation/scripts/plot_a1_trigger_calibration.py \
  --report analysis/chime-scintillation/experiments/trigger-calibration/a1_trigger_calibration.json
```

