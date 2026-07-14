# DM-locked CHIME/DSA joint-fit campaign

This campaign regenerates the morphology-audit fits after adoption of the
validated 12-burst DM-phase v2 catalog. The scientific DM is held fixed rather
than sampled jointly with morphology:

```text
delta_dm_band = DM_adopted - DM_encoded_in_input_product
```

`DM_adopted` is the CHIME-primary value used by the manuscript for both bands.
The fixed residual is applied inside the existing canonical scattering kernel;
it is removed from the nested-sampling volume. DSA `dm_init` is also updated to
the adopted physical DM so the intra-channel-smearing term uses the same value.

The roster preserves the previously selected component counts for the first
pass and includes one expanded variant for each panel previously flagged as
morphologically incomplete. Chromatica receives a C1D1 fit attempt rather than
being assumed to have an acceptable model. Promotion still requires the
repository's Level 1--3 fit gates and visual review of every diagnostic panel.
The review-triggered remediation rows retain Mahi C1D2 and Chromatica C2D1/C2D2
as explicit alternatives after their minimal models failed the residual gate;
Wilhelm C1D3 tests the remaining visible DSA morphology after C1D2 also failed.
Wilhelm C1D4 is the final residual-driven alternative after C1D3 improved the
evidence but retained a DSA reduced chi-square above four.

## HPCC execution

```bash
python analysis/scattering-dm-locked-2026-07-14/prepare_campaign.py \
  --source-configs /central/scratch/jfaber/flits-runs/configs \
  --output /central/scratch/jfaber/flits-dm-locked-20260714 \
  --repo "$PWD"

bash analysis/scattering-dm-locked-2026-07-14/submit_campaign.sh \
  /central/scratch/jfaber/flits-dm-locked-20260714
```

The campaign is complete only when every roster row has a fit result, PPC
metrics, a data/model/residual diagnostic, and a recorded PASS/MARGINAL/FAIL
plus visual-review verdict.

The frozen campaign products live under `results/`. `fit_adjudication.csv`
records the selected morphology variant, fixed adopted DM, numerical PPC
metrics, and the independent visual-residual decision. Only rows marked
`accepted_physical` are eligible for pulse-broadening/scintillation overlays;
`morphology_only` rows remain useful and are shown in the manuscript audit
gallery, but their fitted scattering parameters are not promoted. A numerical
reduced-chi-square pass never overrides coherent structure in the residual
dynamic spectrum.

Each morphology variant writes to `variants/<variant>/`. This isolation is
required when component-count alternatives for the same burst run
concurrently because the fitter's native output names are burst-based.
