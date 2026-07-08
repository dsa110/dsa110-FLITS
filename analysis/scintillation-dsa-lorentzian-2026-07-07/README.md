# DSA Lorentzian ACF Bandwidth Fits

Run date: 2026-07-07 PDT.

This directory is the first fresh DSA scintillation-bandwidth pass after the
DSA/CHIME data-products staging cleanup. It deliberately does not read the
legacy `stored_fits` blocks in `scintillation/configs/bursts/*_dsa.yaml` and
does not use any rescued `acf_results.pkl` files. The driver starts from the
staged DSA dynamic-spectrum `.npz` files under:

```text
~/Data/Faber2026/dsa110/scintillation/data/{burst}.npz
```

The run path is:

1. load each checked-in DSA burst config,
2. force fresh data preparation and ACF extraction,
3. stop after ACF extraction,
4. fit 1, 2, and 3 Lorentzian components to each sub-band ACF within the
   config's `analysis.fitting.fit_lagrange_mhz` window,
5. select the sub-band component count with the existing BIC plus nested-F
   criterion in `scintillation.scint_analysis.revalidation`.

The generated tables include `quality_flags` for components that should not be
used as clean bandwidth measurements without manual inspection. In particular,
`dnu_exceeds_fit_window` marks a Lorentzian width larger than the lag span fitted
for that sub-band, and `fractional_dnu_err_gt_1` marks a formally weak width.

Diagnostic plots and intermediate caches are disabled for this run, so no
`${FLITS_ROOT}` literal-path plot artifacts or stale ACF caches can affect the
results. Noise descriptors remain enabled because they define the ACF
normalization, but the Monte Carlo noise-template generation is disabled because
this strict Lorentzian-only pass does not fit the template component.

## Reproduce

From `pipeline/`:

```bash
python analysis/scintillation-dsa-lorentzian-2026-07-07/run_dsa_lorentzian_fits.py
```

Set `FLITS_ROOT` or pass `--flits-root` if the staged data live somewhere other
than `~/Data/Faber2026/dsa110`.
