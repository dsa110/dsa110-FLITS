"""Guardrail: does the zach-fine DSA window CONTAIN all four candidate components
(initial + cluster at +2.06/+2.52/+3.01 ms)? A window that truncates the cluster
breaks the D3-vs-D4 test by construction. Prints the DSA burst envelope (first/last
significant sample) vs the chosen window bounds for common_window=False AND True.
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
td = tempfile.mkdtemp(prefix="zachwin_")

pD = jtp._probe_band(cD, "zach_dsa", td, auto=True, snr_target=jtp.SNR_TARGET)
dtn = pD.dt_native  # ms
prof = np.nansum(pD.native, axis=0)
mu = np.median(prof)
mad = np.median(np.abs(prof - mu)) * 1.4826
excess = (prof - mu) / max(mad, 1e-12)
peak = int(np.argmax(excess))
sig_bins = np.where(excess > 5.0)[0]  # >5 sigma samples
print(f"DSA native dt={dtn*1e3:.1f} us, {prof.size} samp, peak@{peak}")
print(f"  >5sigma signal spans bins [{sig_bins.min()}, {sig_bins.max()}] = "
      f"[{(sig_bins.min()-peak)*dtn:+.2f}, {(sig_bins.max()-peak)*dtn:+.2f}] ms rel peak")
# expected component bins from the owner morphology (relative to the initial=peak)
for off in (0.0, 2.06, 2.52, 3.01):
    b = peak + int(round(off / dtn))
    inband = excess[b] if 0 <= b < prof.size else float("nan")
    print(f"  component +{off:.2f} ms -> bin {b}  excess={inband:.1f} sigma")

for cw in (False, True):
    (mC, prepC), (mD, prepD) = jtp.prepare_pair(cC, cD, "zach_win", td, auto=True, common_window=cw)
    # recover the native window bounds this prep used: probe again + downsample map is
    # internal, so report the prep caption + the built window in ms via t_factor*dt
    print(f"common_window={cw!s:5}: DSA {prepD.caption()}")
    # the built model's time axis tells the actual window span
    tD = np.asarray(mD.time, float)
    print(f"    DSA model time-axis span = [{tD.min():.2f}, {tD.max():.2f}] ms, "
          f"width {tD.max()-tD.min():.2f} ms, {tD.size} bins")
