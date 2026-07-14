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
