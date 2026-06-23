#!/usr/bin/env python
"""Build dsa_sefd.csv from the dsa110-rt SEFD dashboard (h23).

No contemporaneous SEFD exists for the 2022-2024 co-detection bursts: dsa110-rt's store is a
2026-02/03 measurement campaign (github.com/dsa110/dsa110-rt, served lxd110h23:5777; raw at
h23:/media/ubuntu/ssd/vikram/sefd/sefd_dashboard/state.json). Each epoch's `full_metrics.median_sefd`
is SEFD = sigma*sqrt(2*dnu*tau*n_pol) from estimate_sefd.py, median over baselines.

We therefore use a single epoch-representative DSA SEFD for every burst: the ROBUST MEDIAN over the
clean epochs (rejecting median_sefd > 15000 Jy, which are flagged/bad solutions), with the robust
fractional scatter (1.4826*MAD/median ~ 0.27) recorded as a documented systematic. The PER-BURST
variation in sigma_S is carried exactly by the beam gain G (analysis/dsa_beam.py + dsa_pointing.csv),
not by the SEFD. See docs/rse/specs/plan-radiometer-flux-cal.md and CALIBRATION_REVIEW.md.

Run from the repo root: python analysis/burst_energies/fetch_dsa_sefd.py
"""

import csv
import json
import statistics
import subprocess
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STATE_JSON = "/media/ubuntu/ssd/vikram/sefd/sefd_dashboard/state.json"
SEFD_BAD_JY = 15000.0  # reject flagged/bad epochs above this
OUT = HERE / "dsa_sefd.csv"


def robust_sefd_from_h23():
    """(median_jy, frac_scatter, n_clean, n_total) over the dashboard epochs on h23."""
    raw = subprocess.run(
        ["ssh", "h23", f"cat {STATE_JSON}"], capture_output=True, text=True, check=True
    ).stdout
    d = json.loads(raw)
    xs = [
        v["full_metrics"]["median_sefd"]
        for v in d.values()
        if v.get("full_metrics", {}).get("median_sefd")
    ]
    clean = [x for x in xs if x < SEFD_BAD_JY]
    med = statistics.median(clean)
    mad = statistics.median([abs(x - med) for x in clean])
    return med, 1.4826 * mad / med, len(clean), len(xs)


def main() -> None:
    med, frac, n_clean, n_total = robust_sefd_from_h23()
    src = f"dsa110-rt dashboard robust median over {n_clean}/{n_total} clean 2026-02/03 epochs (epoch-representative; no contemporaneous SEFD for 2022-2024)"
    bursts = yaml.safe_load((REPO / "configs" / "bursts.yaml").read_text())["bursts"]
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["burst", "mjd", "sefd_jy", "sefd_frac_err", "source"])
        for nick in sorted(bursts):
            w.writerow([nick, bursts[nick]["mjd"], round(med, 1), round(frac, 3), src])
    print(
        f"wrote {OUT}: SEFD={med:.0f} Jy +/- {frac:.0%} for {len(bursts)} bursts ({n_clean}/{n_total} epochs)"
    )


if __name__ == "__main__":
    main()
