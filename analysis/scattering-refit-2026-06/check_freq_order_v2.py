"""Robust freq-ordering check via DISPERSION arrival-time slope (works where
scattering is too weak, e.g. DSA at 1.4 GHz). Higher frequency arrives EARLIER
(delay ∝ ν^-2). Cross-correlate each channel-group profile against the
band-integrated profile to get its time lag, then regress lag vs row index:

  slope > 0  (arrival later at higher row index) => row 0 arrives earliest
             => row 0 is HIGH freq => data DESCENDING
  slope < 0  => row 0 is LOW freq => data ASCENDING

Also reports the scattering-tail metric for comparison.
"""
import sys, numpy as np

npy = sys.argv[1]
raw = np.nan_to_num(np.load(npy).astype(float))
if raw.ndim != 2:
    raw = raw.reshape(raw.shape[0], -1)
nch, nt = raw.shape
ng = 16
edges = np.linspace(0, nch, ng + 1, dtype=int)
gprof = np.array([np.nansum(raw[edges[g]:edges[g+1]], axis=0) for g in range(ng)])
# de-trend each group profile
gprof = gprof - np.median(gprof, axis=1, keepdims=True)
ref = np.nansum(gprof, axis=0)                       # band-integrated profile
ref = ref - ref.mean()
# restrict to on-pulse window (±N bins around global peak) to cut noise
pk = int(np.argmax(ref)); win = max(40, nt // 50)
sl = slice(max(0, pk - win), min(nt, pk + win))
refw = ref[sl]; refw = refw - refw.mean()
lags = []
for g in range(ng):
    p = gprof[g, sl]; p = p - p.mean()
    if np.allclose(p, 0):
        lags.append(np.nan); continue
    cc = np.correlate(p, refw, mode="full")
    lag = np.argmax(cc) - (len(refw) - 1)            # bins; >0 => this group later than ref
    lags.append(lag)
lags = np.array(lags, float)
rows = np.arange(ng)
ok = np.isfinite(lags)
slope = np.polyfit(rows[ok], lags[ok], 1)[0] if ok.sum() > 3 else np.nan

# scattering-tail metric (for comparison)
prof = np.nansum(raw, axis=0); ppk = np.nanargmax(prof); t = np.arange(nt) - ppk
def tail(rows_):
    p = np.nansum(raw[rows_], axis=0); p = p - np.median(p); p[p < 0] = 0
    return float(np.sum(t * p) / (p.sum() + 1e-9))
tlo, thi = tail(slice(0, nch//4)), tail(slice(3*nch//4, nch))

print(f"{npy.split('/')[-1]}  (nch={nch}, nt={nt})")
print(f"  dispersion lag-vs-row slope = {slope:+.3f} bins/group   "
      f"(lags: {np.array2string(lags[ok][:8], precision=0)})")
print(f"  scattering tail: row0={tlo:+.0f}  lastrow={thi:+.0f}")
if np.isfinite(slope) and abs(slope) > 0.05:
    print("  => DESCENDING (row0=high freq)" if slope > 0 else "  => ASCENDING (row0=low freq)")
else:
    print("  => dispersion slope ~0 (well-dedispersed); inconclusive by dispersion")
