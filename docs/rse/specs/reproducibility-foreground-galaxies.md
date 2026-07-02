# Reproducibility: foreground galaxy analysis

This is the repeatable path for the foreground-galaxy and foreground-cluster
analysis used by the manuscript `~/Developer/overleaf/Faber2026`.

## Scope

This workflow covers:

- the actual public-catalog foreground search,
- the validation/adjudication step that turns search candidates into the frozen
  49-object manuscript census,
- the validated intervening-object census,
- the per-sightline dispersion and scattering budget,
- the manuscript budget figure `sightline_dm_scattering_budget`,
- the manuscript foreground figures `galaxies_cgm` and
  `clusters_icm`.

The manuscript does not trust a fresh live search blindly. The frozen 49-object
census in `scratch/codetection/foreground_final.csv` remains the manuscript
source of truth; any new search must be diffed and reconciled against it before
changing the paper.

## Current Provenance

- Code repository: `~/Developer/repos/github.com/jakobtfaber/dsa110-FLITS`
- Code commit at verification: `9964baebd2f17b8afd0c21ce0258804f24850ed3`
- Branch at verification: `verify-claude-review`
- Working tree note: dirty at verification. Pre-existing unrelated dirty state:
  `docs/entire-tracing-checkpoints.md`,
  `analysis/scattering-refit-2026-06/local_runs/runtime`. This run also updated
  the foreground artifacts and fixed the systems-figure generator imports plus
  `matplotlibrc`.
- Runtime: Conda env `flits`
- Python: `3.12.13`
- Key package versions:
  - `numpy 2.4.6`
  - `pandas 2.3.3`
  - `astropy 7.2.0`
  - `scipy 1.17.1`
  - `matplotlib 3.10.8`

## Inputs

Original spreadsheet exports used to seed the 49-object census:

```text
scratch/codetection/source/DSA110_CHIME_Codetection_BurstProperties_Foreground  - Sheet1.csv
scratch/codetection/source/DSA110_CHIME_Codetection_BurstProperties_Foreground  - Sheet2.csv
scratch/codetection/source/DSA110_CHIME_Codetection_BurstProperties_Foreground  - Sheet3.csv
```

Validated foreground census inputs:

```text
ce14b474424efb5ff442c5206020609475ff7b0675aa370cb026a47cc8ff4766  scratch/codetection/foreground_final.csv
38ed01ac7561eddcbd33500e2fabeeb4130c22c4fdca791967415656a4d0cd15  scratch/codetection/foreground.csv
c18fa388cd421d6a90e65b77edceca00afe9c8a9e4cc7feb31a52f468c6e79d7  scratch/codetection/foreground_validated.csv
204fb79727ff71f15269f3d5564215e34d8f027aedbd82719dfda162bdcfb644  scratch/codetection/bursts.csv
```

Grand-figure photo-z-corrected galaxy inputs:

```text
3dffe540b05ad9de1a9be3e8eede4f5c46df9d6253f2032914a8eed05133be77  scratch/photoz-fix/phineas_galaxies.csv
dc72648f43fb52c37bdbc858bf789b0a2e60178a8517e7957a9cc5d63e781407  scratch/photoz-fix/whitney_galaxies.csv
94c9d9575430bf8f75e512ef55878bb723a055bdb50c425029181ff5b98a4edc  scratch/photoz-fix/isha_galaxies.csv
```

Scattering-fit inputs are read by `galaxies.foreground.tau_consistency` from
`analysis/scattering-refit-2026-06/_a1_fits/` when present. Rows without an
available fixed-alpha consistency refit remain marked
`pending - run build_tau_consistency_refits`.

## Run From A Clean Shell

Use the intended Conda env explicitly. This avoids accidental dependence on the
interactive shell or `.envrc`.

```bash
cd ~/Developer/repos/github.com/jakobtfaber/dsa110-FLITS

flits_py() {
  env -i HOME="$HOME" PATH="/opt/anaconda3/bin:/opt/homebrew/bin:/usr/bin:/bin" \
    /opt/anaconda3/bin/conda run -n flits python "$@"
}
```

## Rerun The Foreground Search

The live search is implemented by `galaxies.foreground.search.run_search`. It
runs all 12 configured sightlines from `galaxies.foreground.config.TARGETS`.

Catalog/query engines:

- NED cone search through `NedTapEngine`.
- VizieR GLADE+ (`VII/291/gladep`), DESI Legacy DR8 North photo-z
  (`VII/292/north`), and SDSS DR12 (`V/147/sdss12`).
- All-sky cluster catalogs through `ClusterEngine`: PSZ2, MCXC, MCXC-II.
- Optional DESI DR1 TAP search exists but is disabled by default
  (`ENABLE_EXTRA_ENGINES = False`).

Default search settings:

- galaxy impact limit: `DEFAULT_IMPACT_KPC = 100 kpc`,
- cluster fallback impact limit: `DEFAULT_CLUSTER_IMPACT_KPC = 5000 kpc`,
- foreground redshift buffer: `DEFAULT_Z_EPS = 0.01`,
- minimum credible photo-z floor: `FOREGROUND_PHOTOZ_FLOOR = 0.01`,
- spec-z host/local ambiguity cut: `FOREGROUND_AMBIGUITY_KMS = 500 km/s`,
- max cone radius: `MAX_SEARCH_RADIUS_DEG = 2 deg`.

One-shot run:

```bash
mkdir -p scratch/repro-foreground-search
flits_py - <<'PY'
from galaxies.foreground.search import run_search

run_search(
    output_dir="scratch/repro-foreground-search",
    impact_kpc=100.0,
    build_unified=True,
)
PY
```

Expected outputs:

- `scratch/repro-foreground-search/{nickname}_galaxies.csv` for each sightline
  with at least one retained candidate,
- optional `scratch/repro-foreground-search/{nickname}_unified.csv`,
- `scratch/repro-foreground-search/search_summary.csv`,
- `scratch/repro-foreground-search/survey_coverage.csv` — per sightline × survey
  query log (footprint, raw cone hits, foreground pass count, status).

### Survey coverage maps

Each search records which catalogs were queried and whether the sightline lies
inside each survey's nominal footprint (`galaxies/foreground/survey_coverage.py`):

| Survey | Engine | Nominal footprint (this sample) |
|--------|--------|-----------------------------------|
| NED | `NedTapEngine` | all-sky |
| GLADE+ | `VizierEngine(VII/291/gladep)` | all-sky |
| DESI DR8 North | `VizierEngine(VII/292/north)` | Dec ≥ −20° |
| SDSS DR12 | `VizierEngine(V/147/sdss12)` | SDSS NGC (Dec ≥ 1.26°) |
| Clusters | `ClusterEngine` (PSZ2, MCXC, MCXC-II) | all-sky |

Status per cell: `no_footprint` · `footprint_empty` (in footprint, 0 cone hits) ·
`catalog_hits` · `foreground`.

Regenerate the coverage matrix figure:

```bash
flits_py -m galaxies.foreground.survey_coverage_figures \
  --coverage-csv scratch/repro-foreground-search-hpcc/survey_coverage.csv \
  --out-dir scratch/repro-foreground-coverage-figures
```

For HPCC runs completed before `survey_coverage.csv` existed, backfill from the
Slurm stdout log:

```bash
scp hpcc:~/flits/dsa110-FLITS/logs/foreground_search_64666839.out /tmp/
flits_py -m galaxies.foreground.parse_search_log_coverage \
  --log /tmp/foreground_search_64666839.out \
  --output-dir scratch/repro-foreground-search-hpcc
```

For a more robust long run, use one sightline per process so catalog timeouts do
not lose all progress:

```bash
mkdir -p scratch/repro-foreground-search
flits_py - <<'PY'
from pathlib import Path
import pandas as pd

from galaxies.foreground import search

out = Path("scratch/repro-foreground-search")
out.mkdir(parents=True, exist_ok=True)
done_path = out / "done.txt"
done = set(done_path.read_text().splitlines()) if done_path.exists() else set()
summary_parts = []

for target in search.TARGETS:
    name = target[0]
    if name in done:
        continue
    search.TARGETS = [target]
    search.run_search(output_dir=str(out), impact_kpc=100.0, build_unified=True)
    summary_parts.append(pd.read_csv(out / "search_summary.csv"))
    with done_path.open("a") as fh:
        fh.write(name + "\n")

if summary_parts:
    pd.concat(summary_parts, ignore_index=True).to_csv(out / "_latest_summary_chunk.csv", index=False)
PY
```

Do not copy this live-search output directly into the manuscript. First compare
it with the frozen census:

```bash
flits_py - <<'PY'
from pathlib import Path
import pandas as pd

new_dir = Path("scratch/repro-foreground-search")
old = pd.read_csv("scratch/codetection/foreground_final.csv")

new_rows = []
for path in sorted(new_dir.glob("*_galaxies.csv")):
    if path.name == "search_summary.csv":
        continue
    df = pd.read_csv(path)
    if df.empty:
        continue
    df["nickname"] = path.name.replace("_galaxies.csv", "")
    new_rows.append(df)

new = pd.concat(new_rows, ignore_index=True) if new_rows else pd.DataFrame()
print("frozen census rows:", len(old))
print("fresh search candidate rows:", len(new))
print("frozen verdict counts:")
print(old.final_verdict.value_counts(dropna=False).to_string())
if not new.empty:
    print("fresh search rows by nickname:")
    print(new.nickname.value_counts().sort_index().to_string())
PY
```

Any candidate-list change must be adjudicated through the validation chain below
before it is allowed into `foreground_final.csv`.

## Rebuild The Frozen 49-Object Census

The frozen census was produced from the spreadsheet exports plus independent
catalog validation. Rebuild it with:

```bash
flits_py scratch/codetection/normalize_codetection.py
flits_py scratch/codetection/validate_foreground.py
flits_py scratch/codetection/ps1_strm_resolve.py
flits_py scratch/codetection/ps1_strm_adjudicate.py
flits_py scratch/codetection/merge_final.py
flits_py scratch/codetection/verify_final.py
flits_py scratch/codetection/make_catalog_table.py
flits_py scratch/codetection/verify_catalog_table.py
flits_py scratch/codetection/verify_paper_prose.py
```

What each step does:

- `normalize_codetection.py` parses the foreground spreadsheet exports into
  `bursts.csv` and `foreground.csv`, recomputing separations, impact parameters,
  and internal flags.
- `validate_foreground.py` independently queries LS DR9 photo-z, DESI DR1
  spec-z, NED, and SIMBAD around each spreadsheet object and writes
  `foreground_validated.csv`.
- `ps1_strm_resolve.py` / `ps1_strm_adjudicate.py` resolve the nine WISE/PS1/STRM
  halo rows whose spreadsheet redshifts were not independently sufficient.
- `merge_final.py` merges validation + STRM adjudication into
  `foreground_final.csv` and asserts 49 rows: 29 confirmed, 7 refuted,
  13 inconclusive.
- `make_catalog_table.py` writes `foreground_catalog.csv` and
  `docs-analysis/foreground.md`.
- `verify_*` scripts rederive verdicts and prose/table counts from the data.

After this, rebuild the machine-readable registry and budget artifacts:

```bash
flits_py -m galaxies.foreground.build_artifacts
flits_py -m galaxies.foreground.sightline_budget
flits_py -m galaxies.v2_0.systems_figures \
  --out-dir scratch/repro-foreground-figures
```

Expected command outputs:

- `galaxies.foreground.build_artifacts`
  writes:
  - `galaxies/foreground/data/intervening_census_registry.csv`
  - `galaxies/foreground/data/tau_consistency_catalog.csv`
  - `galaxies/foreground/data/sightline_attribution_matrix.csv`
- `galaxies.foreground.sightline_budget`
  writes:
  - `results/sightline_dm_scattering_budget.csv`
  - `results/sightline_dm_scattering_budget.md`
  - `results/sightline_dm_scattering_budget.png`
- `galaxies.v2_0.systems_figures`
  writes:
  - `galaxies_cgm.{pdf,svg,png}`
  - `clusters_icm.{pdf,svg,png}`

For manuscript production, regenerate and sync in one step:

```bash
bash scripts/manuscript/regenerate_budget_figures.sh
```

(`tools/sync_figures.py --apply` reads `results/figures.manifest.json` and copies
into `~/Developer/overleaf/Faber2026/figures/`.)

When the HPCC search finishes, pull and diff before changing the census:

```bash
bash scripts/manuscript/slot_hpcc_foreground.sh          # rsync + diff
bash scripts/manuscript/slot_hpcc_foreground.sh --apply  # + regen from frozen census
```

Only overwrite manuscript figures after visually checking them.

## Step-by-Step Logic

1. `galaxies.foreground.census_registry` reads the validated scratch products
   and builds the 49-object intervening census. It applies the registry-tier and
   budget-tier gates:
   - `final_verdict == confirmed` is required for registry membership.
   - clusters are budget eligible only when `b/R500 <= 1.0`.
   - confirmed galaxies are budget eligible by default.

2. `galaxies.foreground.build_unified` converts each eligible foreground row into
   physical CGM quantities:
   - best redshift and impact parameter,
   - stellar and halo mass provenance,
   - virial radius and scaled impact,
   - hot FRB/ModifiedNFW baryon column (`galaxies.foreground.scattering_predict.dm_halo_mnfw`,
     mirroring `frb.halos.models.ModifiedNFW`; `alpha=2`, `y0=2`, `c=7.67`, `f_hot=0.75`),
   - cool CGM dispersion term,
   - predicted 1 GHz scattering time and uncertainty bracket.

3. `galaxies.foreground.sightline_budget` combines the foreground prediction with
   the sightline budget:
   - observed burst DM from CHIME/DSA burst config filenames,
   - Milky Way disk DM and tau from NE2001 via `pygedm`,
   - Milky Way halo prior `DM_MW_HALO = 40 pc cm^-3`,
   - Macquart-relation cosmic mean DM for measured host redshifts,
   - summed intervening foreground DM and tau,
   - residual host DM,
   - optional predictive host DM from `galaxies.host` when
     `galaxies/host/data/hosts.yaml` lists `log10_mstar` and/or H-alpha/size
     (FRB `frb/dm/host.py` physics: mNFW halo with `f_hot=0.55`, ISM from
     H-alpha or sSFR).

4. For galaxy-interior intersections, the raw mNFW core extrapolation is not used
   blindly. The physically bounded value re-evaluates the hot column at
   `max(b, 0.1 R_vir)` and rescales the cool term by the same hot-column ratio.
   The raw and capped values are both reported.

5. The `galaxies_cgm` figure is a curated diagnostic, not the full budget.
   It plots the three headline galaxies from `scratch/photoz-fix`:
   - `phineas` / FRB 20230307A,
   - `whitney` / FRB 20220310F,
   - `isha` / FRB 20221113A.

6. The `clusters_icm` figure is also curated. It plots the four innermost
   foreground clusters in the FRB 20230307A field, using the FRB/ModifiedNFW
   hot-baryon column (`alpha=2`, `y0=2`, `f_hot=0.75`) with
   `M200 ~= 1.3 M500`.

## Host-side DM (`galaxies/host/`)

Predictive host columns (optional) mirror `FRBs/FRB` `frb/dm/host.py` without
requiring the `frb` package:

| Module | Role |
|--------|------|
| `galaxies/host/em.py` | H-alpha → EM → DM (Reynolds 1977 / Tendulkar+2017) |
| `galaxies/host/dm_predict.py` | `dm_host_halo`, `dm_host_from_halpha`, `dm_host_from_ssfr` |
| `galaxies/host/data/hosts.yaml` | Per-burst host observables (`log10_mstar`, H-alpha, `reff_arcsec`) |

`sightline_budget` adds `dm_host_halo_pred`, `dm_host_ism_pred`, `dm_host_pred`,
and `dm_host_unattrib = dm_host_capped - dm_host_pred` when metadata is present.
PATH/associate host localization is out of scope (heavy `astropath` deps).

## Verified Output Hashes

After rerunning in Conda env `flits`:

```text
b45d698cde155427b272d0ead4c1a248303ef8c839ddcb84a0393adcdd1ae222  galaxies/foreground/data/intervening_census_registry.csv
3c43fc85187f7a68f898e2af380a8be87e40762100dbf4dc4141205940b8e77f  galaxies/foreground/data/tau_consistency_catalog.csv
36f01d20aa9b28cd8946fefb78c1c32dadc93f4e69aa332b57dc48438709a806  galaxies/foreground/data/sightline_attribution_matrix.csv
4a78d83b71eb5dd5542dca5bb6b2105dea69ac84b3e6402028bd49f25d1b64dc  results/sightline_dm_scattering_budget.csv
f812da9b45fc4ee3346e66d8a0d18506abfb543f9c7ea45fc842b499e7fed195  results/sightline_dm_scattering_budget.md
be86a84c29f4aa3c30ea1537946218f8708257b47e31defd78d5289c26496cc1  results/sightline_dm_scattering_budget.png
f777c55d75a2d9c3eda1c90132f788469fb3ce164c46f37c8cc026c1fe742b17  scratch/repro-foreground-figures/galaxies_cgm.pdf
4d8e9c78bfa1cbdb2ec5384c3179b4432113f4beaf544f663ac45fe167eb58a1  scratch/repro-foreground-figures/clusters_icm.pdf
```

## Verification Checklist

Run these before trusting a regenerated manuscript product:

```bash
cd ~/Developer/repos/github.com/jakobtfaber/dsa110-FLITS

flits_py -m pytest \
  galaxies/foreground/test_census_registry.py \
  galaxies/foreground/test_build_unified.py \
  galaxies/foreground/test_sightline_budget.py \
  galaxies/foreground/test_scattering_predict.py

flits_py -m galaxies.foreground.build_artifacts
flits_py -m galaxies.foreground.sightline_budget
flits_py -m galaxies.v2_0.systems_figures \
  --out-dir scratch/repro-foreground-figures
```

Then visually inspect:

- `results/sightline_dm_scattering_budget.png`
- `scratch/repro-foreground-figures/galaxies_cgm.png`
- `scratch/repro-foreground-figures/clusters_icm.png`

## Known Caveats

- The original catalog discovery is not fully clean-room repeatable from this
  document alone because it depends on live catalog services and the historical
  validation choices captured in `scratch/codetection`. Treat those scratch CSVs
  as the pinned inputs for the manuscript analysis.
- `galaxies/v2_0/systems_figures.py` is the current foreground-systems figure generator.
  It uses current `galaxies.foreground` kernels but still consumes
  `scratch/photoz-fix/*_galaxies.csv` for the three curated galaxy panels.
- The alpha=4 consistency-refit catalog still contains pending rows unless the
  HPC-only refits are run with `galaxies.foreground.run_tau_consistency_refits`.
