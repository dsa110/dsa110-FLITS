"""Resumable, chunked NED-inclusive regen (works around the 600s call cap).

run_search over all 12 sightlines exceeds 600s because DESI VII/292/north returns
~7e5 rows per cone and run_search computes impact_kpc per-row. This driver runs
one sightline at a time (monkeypatching search.TARGETS), accumulates the summary,
and records progress in done.txt so repeated invocations resume where they left
off. When all sightlines are done it writes search_summary.csv + the budget.

Re-run until it prints ALL DONE.
"""

import os
import sys
import time

REPO = "/Users/jakobfaber/Developer/repos/github.com/jakobtfaber/dsa110-FLITS"
sys.path.insert(0, REPO)
OUT = os.path.join(REPO, "scratch", "cluster-regen-out")
os.makedirs(OUT, exist_ok=True)
DONE = os.path.join(OUT, "done.txt")
ACCUM = os.path.join(OUT, "_accum_summary.csv")
WALL_BUDGET = 540.0

import pandas as pd  # noqa: E402

from galaxies.v2_0 import search as S  # noqa: E402

all_targets = list(S.TARGETS)
done = set()
if os.path.exists(DONE):
    done = {ln.strip() for ln in open(DONE) if ln.strip()}
remaining = [t for t in all_targets if t[0] not in done]
print(f"{len(done)}/{len(all_targets)} done; {len(remaining)} remaining", flush=True)

t0 = time.time()
for tgt in remaining:
    if time.time() - t0 > WALL_BUDGET:
        print(f"[budget] stopping at {time.time() - t0:.0f}s; re-run to continue", flush=True)
        break
    name = tgt[0]
    print(f"=== {name} ===", flush=True)
    S.TARGETS = [tgt]
    S.run_search(output_dir=OUT)
    row = pd.read_csv(os.path.join(OUT, "search_summary.csv"))
    header = not os.path.exists(ACCUM)
    row.to_csv(ACCUM, mode="a", header=header, index=False)
    with open(DONE, "a") as fh:
        fh.write(name + "\n")
    print(f"[done] {name} at {time.time() - t0:.0f}s", flush=True)

done = {ln.strip() for ln in open(DONE) if ln.strip()} if os.path.exists(DONE) else set()
if len(done) >= len(all_targets):
    import matplotlib.pyplot as plt

    from galaxies.v2_0.sightline_budget import (
        build_all_budgets,
        format_budget_table,
        make_budget_figure,
    )

    pd.read_csv(ACCUM).to_csv(os.path.join(OUT, "search_summary.csv"), index=False)
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
    fig.savefig(
        os.path.join(OUT, "sightline_dm_scattering_budget.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig)
    print(f"ALL DONE; budget rows={len(df)}", flush=True)
else:
    print(f"NOT DONE: {len(done)}/{len(all_targets)}", flush=True)
