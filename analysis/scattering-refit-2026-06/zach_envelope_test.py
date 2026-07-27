"""Test an envelope on-pulse window (5-sigma component span + margin) for zach-fine,
so the DSA window provably CONTAINS all four candidate components (initial + cluster
at +2.06/+2.52/+3.01 ms) rather than the peak-anchored 2.2 ms window that truncates
the cluster. Verifies coverage + the resulting t-factor, for both bands.
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

_orig = jtp.robust_onpulse_bounds


def envelope_bounds(prof, dt_ms, *, k_hi=jtp.WIN_K_HI, k_lo=jtp.WIN_K_LO,
                    max_gap_ms=jtp.WIN_MAX_GAP_MS, margin_frac=jtp.WIN_MARGIN_FRAC,
                    trail_cap_ms=jtp.WIN_TRAIL_CAP_MS, min_offpulse_frac=jtp.WIN_MIN_OFFPULSE_FRAC):
    """On-pulse window spanning the FULL >k_hi-sigma component envelope + margin, so a
    windowed multi-component count test contains every candidate sub-pulse. Falls back
    to the peak-anchored robust window for faint bursts (no >k_hi sample)."""
    n = prof.size
    mu, sig = jtp._baseline(prof)
    if sig <= 0 or n < 4:
        return 0, n
    excess = prof - mu
    hot = np.where(excess > k_hi * sig)[0]
    if hot.size == 0:
        return _orig(prof, dt_ms, k_hi=k_hi, k_lo=k_lo, max_gap_ms=max_gap_ms,
                     margin_frac=margin_frac, trail_cap_ms=trail_cap_ms,
                     min_offpulse_frac=min_offpulse_frac)
    lo, hi = int(hot.min()), int(hot.max())
    span = hi - lo + 1
    margin = int(round(margin_frac * span))
    return max(0, lo - margin), min(n, hi + margin + 1)


RUNS = os.environ["FLITS_RUNS"]
cC = f"{RUNS}/configs/zach_chime_run.yaml"
cD = f"{RUNS}/configs/zach_dsa_run.yaml"
td = tempfile.mkdtemp(prefix="zachenv_")

# component bins from the DSA native probe
pD = jtp._probe_band(cD, "zach_dsa", td, auto=True, snr_target=jtp.SNR_TARGET)
dtn = pD.dt_native
prof = np.nansum(pD.native, axis=0)
mu, sig = jtp._baseline(prof)
peak = int(np.argmax(prof - mu))
comp_bins = {off: peak + int(round(off / dtn)) for off in (0.0, 2.06, 2.52, 3.01)}
print("component bins:", comp_bins, "peak", peak)
lo, hi = envelope_bounds(prof, dtn)
print(f"ENVELOPE window bins [{lo},{hi}] = [{(lo-peak)*dtn:+.2f},{(hi-peak)*dtn:+.2f}] ms rel peak, "
      f"width {(hi-lo)*dtn:.2f} ms")
covered = {off: (lo <= b < hi) for off, b in comp_bins.items()}
print("  all components covered:", covered, "->", "PASS" if all(covered.values()) else "FAIL")

# BAND-AWARE patch: envelope window for DSA only (bridges the cluster gap); CHIME keeps
# the original tail-following window (guardrail 3: CHIME binning unchanged, tail not clipped).
_orig_probe = jtp._probe_band


def _probe_band_bandaware(cfg, name, outdir, **kw):
    jtp.robust_onpulse_bounds = envelope_bounds if name.endswith("_dsa") else _orig
    try:
        return _orig_probe(cfg, name, outdir, **kw)
    finally:
        jtp.robust_onpulse_bounds = _orig


jtp._probe_band = _probe_band_bandaware
for cw in (False, True):
    (mC, prepC), (mD, prepD) = jtp.prepare_pair(cC, cD, "zach_env", td, auto=True, common_window=cw)
    tD = np.asarray(mD.time, float)
    print(f"common_window={cw!s:5}: CHIME {prepC.caption().split(';')[0]} | DSA {prepD.caption().split(';')[0]}"
          f" ; DSA model-time [{tD.min():.2f},{tD.max():.2f}] ms ({tD.size} bins)")
