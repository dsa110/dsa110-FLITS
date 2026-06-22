"""Determine the TRUE frequency ordering of a CHIME .npy by the scattering-tail
direction. The scattering tail is longest at the LOWEST physical frequency, so
the channel with the largest late-time asymmetry is the low-freq end.

Pipeline assumes data[0] = f_min (ascending) and pairs it with
freq = linspace(f_min, f_max). If the tail is largest at row 0, the data is
ascending (correct). If largest at the last row, the data is DESCENDING and the
ascending-freq assumption is wrong -> tau(nu) fit backwards.
"""
import sys, numpy as np

npy = sys.argv[1]
raw = np.load(npy)
if raw.ndim != 2:
    raw = raw.reshape(raw.shape[0], -1)
nch, nt = raw.shape
print(f"{npy.split('/')[-1]}  shape=(nch={nch}, nt={nt})")

# collapse to 8 channel groups, measure tail asymmetry per group
ng = 8
edges = np.linspace(0, nch, ng + 1, dtype=int)
prof_all = np.nansum(raw, axis=0)
pk = int(np.nanargmax(prof_all))           # global pulse peak (time bin)
t = np.arange(nt) - pk
print(f"global peak bin = {pk}")
print(f"{'grp':>3} {'rows':>12} {'skew(late+)':>11} {'fwhm':>7}")
skews = []
for g in range(ng):
    lo, hi = edges[g], edges[g + 1]
    p = np.nansum(raw[lo:hi], axis=0)
    p = p - np.median(p)
    p[p < 0] = 0
    if p.sum() <= 0:
        skews.append(np.nan); print(f"{g:>3} {f'{lo}:{hi}':>12} {'--':>11}"); continue
    w = p / p.sum()
    mean_t = np.sum(t * w)                  # centroid offset from peak (late tail -> positive)
    var_t = np.sum((t - mean_t) ** 2 * w)
    fwhm = 2.355 * np.sqrt(max(var_t, 1e-9))
    skews.append(mean_t)
    print(f"{g:>3} {f'{lo}:{hi}':>12} {mean_t:>11.2f} {fwhm:>7.1f}")

skews = np.array(skews)
# tail asymmetry should be largest (most positive / widest) at the LOW-freq end
lo_end = np.nanmean(skews[:2])      # rows 0..nch/4
hi_end = np.nanmean(skews[-2:])     # rows 3nch/4..nch
print(f"\nrow-0 end tail-offset = {lo_end:.2f} ;  last-row end = {hi_end:.2f}")
if lo_end > hi_end:
    print(">> tail LONGER at row 0  => row 0 is LOW freq  => data ASCENDING (pipeline assumption OK)")
else:
    print(">> tail LONGER at LAST row => last row is LOW freq => data DESCENDING => "
          "ascending-freq assumption is WRONG (fit pairs tau(nu) backwards)")
