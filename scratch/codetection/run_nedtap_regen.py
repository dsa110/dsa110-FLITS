"""Full NED-inclusive regen into scratch (NedTapEngine + ClusterEngine).

Writes fresh galaxy catalogs + DM/scattering budget into scratch/cluster-regen-out
(never the tracked results/), reading observed DM/tau from the tracked
results/bursts so the budget stays comparable to the committed run.
"""

import os
import sys
import time

REPO = "/Users/jakobfaber/Developer/repos/github.com/jakobtfaber/dsa110-FLITS"
sys.path.insert(0, REPO)
OUT = os.path.join(REPO, "scratch", "cluster-regen-out")
os.makedirs(OUT, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402

from galaxies.v2_0.search import run_search  # noqa: E402
from galaxies.v2_0.sightline_budget import (  # noqa: E402
    build_all_budgets,
    format_budget_table,
    make_budget_figure,
)

t0 = time.time()


def stamp(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


stamp(f"run_search -> {OUT}")
run_search(output_dir=OUT)
stamp("run_search done")

configs_dir = os.path.join(REPO, "scattering", "configs", "bursts", "chime")
bursts_dir = os.path.join(REPO, "results", "bursts")
df = build_all_budgets(
    results_dir=OUT, configs_dir=configs_dir, bursts_dir=bursts_dir, enrich=False
)
df.to_csv(os.path.join(OUT, "sightline_dm_scattering_budget.csv"), index=False)
with open(os.path.join(OUT, "sightline_dm_scattering_budget.md"), "w") as fh:
    fh.write("# FRB sightline DM & scattering budgets\n\n")
    fh.write(format_budget_table(df))
    fh.write("\n")
fig = make_budget_figure(df)
fig.savefig(os.path.join(OUT, "sightline_dm_scattering_budget.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
stamp(f"budget done; rows={len(df)}")
