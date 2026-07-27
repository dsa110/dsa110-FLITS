"""Diagnose what actually sets zach's ~44 ms window (trail cap did nothing).

Checks, per band: native record span, raw robust_onpulse_bounds (lo,hi) at
several trail caps, common-window vs per-band window, and the resulting DSA
t-factor with common_window=False. Identifies the true binding constraint.
"""
import os, sys, tempfile
import numpy as np

WT = "/home/ubuntu/worktrees/joint-tf-fits"
os.environ.setdefault("FLITS_REPO", WT)
os.environ.setdefault("FLITS_RUNS", "/home/ubuntu/flits-runs")
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, f"{WT}/scattering")
sys.path.insert(0, f"{WT}/analysis/scattering-refit-2026-06")

import joint_tf_prep as jtp

RUNS = os.environ["FLITS_RUNS"]
cC = f"{RUNS}/configs/zach_chime_run.yaml"
cD = f"{RUNS}/configs/zach_dsa_run.yaml"
td = tempfile.mkdtemp(prefix="zachdiag_")

# --- 1. Raw probe of each band (native record + robust window at varying cap) ---
for label, cfg, tag in [("CHIME", cC, "zach_chime"), ("DSA", cD, "zach_dsa")]:
    p = jtp._probe_band(cfg, tag, td, auto=True, snr_target=jtp.SNR_TARGET)
    n = p.native.shape[1]
    dtn = p.dt_native
    lo, hi = p.win
    print(f"[{label}] native: {n} samp x {dtn*1e3:.1f} us = {n*dtn*1e3:.1f} ms record; "
          f"peak@{p.peak}; robust win (lo,hi)=({lo},{hi}) span={(hi-lo)*dtn*1e3:.1f} ms")
    # profile-level robust bounds at several caps (isolate the trailing-cap effect)
    prof = np.nansum(p.native, axis=0)
    for cap in (30.0, 12.0, 4.0, 1.0):
        b = jtp.robust_onpulse_bounds(prof, dtn * 1e3, trail_cap_ms=cap)
        print(f"    trail_cap={cap:5.1f} ms -> bounds {b} span={(b[1]-b[0])*dtn*1e3:.1f} ms")

# --- 2. common-window vs per-band: does DSA binning change without the union? ---
print()
for cw in (True, False):
    (mC, pC), (mD, pD) = jtp.prepare_pair(cC, cD, "zach_cw", td, auto=True, common_window=cw)
    print(f"common_window={cw!s:5}: CHIME {pC.caption().split(';')[0]} win {pC.caption().split('window')[1].split(';')[0].strip()}"
          f" | DSA {pD.caption().split(';')[0]} win {pD.caption().split('window')[1].split(';')[0].strip()}")
