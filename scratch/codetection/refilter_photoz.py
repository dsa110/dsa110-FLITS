"""Apply the photo-z floor + error-cap foreground fix to the promoted galaxy CSVs.

The new _foreground_mask is a provable strict subset of the old one (identical for
spec-z rows; stricter for photo-z via the floor and the error cap), so re-filtering
the existing promoted snapshot is equivalent to a full requery — same catalog
snapshot, no live re-query — and rebuilds budget + sensitivity + sky figures from
the corrected set. Writes everything to scratch/photoz-fix for review before promotion.
"""

import os
import sys

REPO = "/Users/jakobfaber/Developer/repos/github.com/jakobtfaber/dsa110-FLITS"
sys.path.insert(0, REPO)
SRC = os.path.join(REPO, "results")
OUT = os.path.join(REPO, "scratch", "photoz-fix")
os.makedirs(OUT, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from galaxies.v2_0.config import (  # noqa: E402
    DEFAULT_CLUSTER_IMPACT_KPC,
    DEFAULT_IMPACT_KPC,
    DEFAULT_Z_EPS,
    TARGETS,
)
from galaxies.v2_0.search import _foreground_mask  # noqa: E402
from galaxies.v2_0.sightline_budget import (  # noqa: E402
    build_all_budgets,
    format_budget_table,
    make_budget_figure,
)

summary = []
for i, (name, ra, dec, z_frb) in enumerate(TARGETS):
    src = os.path.join(SRC, f"{name.lower()}_galaxies.csv")
    n = 0
    if os.path.exists(src):
        df = pd.read_csv(src)
        kept = df[
            _foreground_mask(
                df,
                z_frb=z_frb,
                z_eps=DEFAULT_Z_EPS,
                impact_kpc=DEFAULT_IMPACT_KPC,
                cluster_impact_kpc=DEFAULT_CLUSTER_IMPACT_KPC,
            )
        ]
        n = len(kept)
        if n:
            kept.to_csv(os.path.join(OUT, f"{name.lower()}_galaxies.csv"), index=False)
    summary.append(
        {"name": name, "target_id": i + 1, "ra": ra, "dec": dec, "z_frb": z_frb, "num_galaxies": n}
    )
    print(f"  {name}: {n} foreground", flush=True)

pd.DataFrame(summary).to_csv(os.path.join(OUT, "search_summary.csv"), index=False)

# Budget from the corrected set (observed DM/tau from the tracked results/bursts).
df = build_all_budgets(
    results_dir=OUT,
    configs_dir=os.path.join(REPO, "scattering", "configs", "bursts", "chime"),
    bursts_dir=os.path.join(REPO, "results", "bursts"),
    enrich=False,
)
df.to_csv(os.path.join(OUT, "sightline_dm_scattering_budget.csv"), index=False)
with open(os.path.join(OUT, "sightline_dm_scattering_budget.md"), "w") as fh:
    fh.write("# FRB sightline DM & scattering budgets\n\n")
    fh.write(format_budget_table(df))
    fh.write("\n")
fig = make_budget_figure(df)
fig.savefig(os.path.join(OUT, "sightline_dm_scattering_budget.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"budget rows={len(df)}", flush=True)

# Sensitivity + sky maps from the corrected set.
os.system(
    f'"{sys.executable}" -m galaxies.v2_0.sightline_sensitivity --results-dir "{OUT}" --output-dir "{OUT}"'
)
os.system(
    f'"{sys.executable}" -m galaxies.v2_0.plotting --results-dir "{OUT}" --output-dir "{OUT}"'
)
print("DONE", flush=True)
