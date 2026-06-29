# CHIME-DSA Holding Tree Drain

Date: 2026-06-29

## Scope

Drained the sparse, non-git holding tree at:

`/Users/jakobfaber/Developer/research-holding/caltech/ovro/dsa110/chime_dsa_codetections`

This tree was not a source for a wholesale merge. FLITS remains the canonical code
home; data remains off-repo under the CHIME-DSA data authority.

## Promoted to FLITS

| holding path | FLITS path | reason |
|---|---|---|
| `frb_cluster_associations.csv` | `notebooks/codetections/data/frb_cluster_associations.csv` | Required by `notebooks/codetections/interveners.ipynb` |
| `frb_halo_associations.csv` | `notebooks/codetections/data/frb_halo_associations.csv` | Required by `notebooks/codetections/interveners.ipynb` |

## Copied to external data authority

| holding path | external path | reason |
|---|---|---|
| `dm/DSA_DM_phase/*.pdf` | `iacobus:~/Research/CHIME_DSA_Codetections/presentations/DSA_DM_phase/` | Presentation-derived DM products; not suitable for git |

`DSA_DM_phase.pptx` and `DSA_DM_phase.key` were already present under
`iacobus:~/Research/CHIME_DSA_Codetections/presentations/`.

## Historical references left in quarantine

These files are superseded by the June 2026 migration docs in FLITS and are kept
only as historical provenance in the quarantined holding tree:

| holding path | classification |
|---|---|
| `AGENTS.md` | historical local guidance |
| `CONSOLIDATION_PLAN.md` | superseded April consolidation plan |
| `PROJECT_INVENTORY.md` | superseded April inventory |
| `GIT_RECONCILIATION_REPORT.md` | superseded fork reconciliation report |
| `codetections_manifest.yaml` | superseded by FLITS `codetections_manifest.yaml` |
| `h23_resilient_transfer.sh` | stale legacy transfer script; see `scripts/migration/h23_to_iacobus.sh` |

## Quarantine-only local state

| holding path | classification |
|---|---|
| `.cursor/hooks/state/continual-learning-index.json` | local agent state |
| `.cursor/hooks/state/continual-learning.json` | local agent state |
| `.cursorignore` | obsolete local ignore file |
| `.vscode/settings.json` | local editor state |
| `halos/.gitignore` | empty satellite stub |
| `halos/AGENTS.md` | empty satellite stub guidance |
| `scattering/flowchart/flowchart.js` | stale layout sketch |
| `synthetic_dynspec.npy` | small binary scratch product; not git material |
| `tmp_convert_ipynb_to_py.py` | empty temporary script |
| `tmp_nb_to_py_skim.py` | temporary notebook conversion helper |

## DM products kept out of git

These local products remain preserved through the external copy and holding-tree
quarantine, not through git:

| holding path | classification |
|---|---|
| `dm/DSA_DM_phase.key` | already external in `presentations/` |
| `dm/DSA_DM_phase.pptx` | already external in `presentations/` |
| `dm/DSA_DM_phase/casey_dsa_botmfref_DM-0.008_pm_0.046_dt32.76800us_768chn_power.pdf` | copied external |
| `dm/DSA_DM_phase/casey_dsa_botmfref_DM-0.008_pm_0.046_dt32.76800us_768chn_waterfall.pdf` | copied external |
| `dm/DSA_DM_phase/chromatica_dsa_botmfref_DM0.239_pm_0.093_dt32.76800us_768chn_power.pdf` | copied external |
| `dm/DSA_DM_phase/chromatica_dsa_botmfref_DM0.239_pm_0.093_dt32.76800us_768chn_waterfall.pdf` | copied external |
| `dm/DSA_DM_phase/freya_dsa_botmfref_DM0.061_pm_0.049_dt32.76800us_768chn_power.pdf` | copied external |
| `dm/DSA_DM_phase/freya_dsa_botmfref_DM0.061_pm_0.049_dt32.76800us_768chn_waterfall.pdf` | copied external |
| `dm/DSA_DM_phase/hamilton_dsa_botmfref_DM-0.028_pm_0.081_dt32.76800us_768chn_power.pdf` | copied external |
| `dm/DSA_DM_phase/hamilton_dsa_botmfref_DM-0.028_pm_0.081_dt32.76800us_768chn_waterfall.pdf` | copied external |
| `dm/DSA_DM_phase/isha_dsa_botmfref_DM-0.180_pm_0.127_dt32.76800us_768chn_power.pdf` | copied external |
| `dm/DSA_DM_phase/isha_dsa_botmfref_DM-0.180_pm_0.127_dt32.76800us_768chn_waterfall.pdf` | copied external |
| `dm/DSA_DM_phase/johndoeII_dsa_botmfref_DM-0.043_pm_0.071_dt32.76800us_768chn_power.pdf` | copied external |
| `dm/DSA_DM_phase/johndoeII_dsa_botmfref_DM-0.043_pm_0.071_dt32.76800us_768chn_waterfall.pdf` | copied external |
| `dm/DSA_DM_phase/mahi_dsa_botmfref_DM-0.009_pm_0.095_dt32.76800us_768chn_power.pdf` | copied external |
| `dm/DSA_DM_phase/mahi_dsa_botmfref_DM-0.009_pm_0.095_dt32.76800us_768chn_waterfall.pdf` | copied external; source is zero bytes |
| `dm/DSA_DM_phase/oran_dsa_botmfref_DM-0.038_pm_0.116_dt32.76800us_768chn_power.pdf` | copied external |
| `dm/DSA_DM_phase/oran_dsa_botmfref_DM-0.038_pm_0.116_dt32.76800us_768chn_waterfall.pdf` | copied external |
| `dm/DSA_DM_phase/phineas_dsa_botmfref_DM-0.025_pm_0.038_dt32.76800us_768chn_power.pdf` | copied external |
| `dm/DSA_DM_phase/phineas_dsa_botmfref_DM-0.025_pm_0.038_dt32.76800us_768chn_waterfall.pdf` | copied external |
| `dm/DSA_DM_phase/whitney_dsa_botmfref_DM0.008_pm_0.028_dt32.76800us_768chn_power.pdf` | copied external |
| `dm/DSA_DM_phase/whitney_dsa_botmfref_DM0.008_pm_0.028_dt32.76800us_768chn_waterfall.pdf` | copied external |
| `dm/DSA_DM_phase/wilhelm_dsa_botmfref_DM-0.012_pm_0.036_dt32.76800us_768chn_power.pdf` | copied external |
| `dm/DSA_DM_phase/wilhelm_dsa_botmfref_DM-0.012_pm_0.036_dt32.76800us_768chn_waterfall.pdf` | copied external |
| `dm/DSA_DM_phase/zach_dsa_botmfref_DM-0.054_pm_0.020_dt32.76800us_768chn_power.pdf` | copied external |
| `dm/DSA_DM_phase/zach_dsa_botmfref_DM-0.054_pm_0.020_dt32.76800us_768chn_waterfall.pdf` | copied external |

## Archive location

After drain verification, the holding tree was archived out of
`research-holding` on 2026-06-29 to:

`~/Developer/scratch/2026-06/archive/chime_dsa_codetections_drained_20260629/`

The consolidated quarantine ledger lives in that archive as
`QUARANTINE_README.md`. `research-holding/.../dsa110/_quarantine/` was removed
after archival.

Restore command:

```bash
mkdir -p ~/Developer/research-holding/caltech/ovro/dsa110/_quarantine
mv ~/Developer/scratch/2026-06/archive/chime_dsa_codetections_drained_20260629 \
  ~/Developer/research-holding/caltech/ovro/dsa110/_quarantine/chime_dsa_codetections_drained_20260629
mv ~/Developer/research-holding/caltech/ovro/dsa110/_quarantine/chime_dsa_codetections_drained_20260629/QUARANTINE_README.md \
  ~/Developer/research-holding/caltech/ovro/dsa110/_quarantine/README.md
```
