#!/usr/bin/env python
"""Decisive test: is the CHIME broadening frequency-dependent scattering, or
frequency-independent intrinsic (multi-peak) morphology?

For a burst's CHIME band: split into N sub-bands, and per sub-band measure the
on-pulse broadening two model-free ways:
  - w_rms  : sqrt(2nd central moment) of the baseline-subtracted on-pulse profile
  - w_tail : exponential decay time of the trailing edge (scattering-sensitive)
Then fit width(nu) ~ nu^-beta across sub-bands. Scattering => beta ~ alpha (positive,
~1.4 for johndoeII's claim). Intrinsic multi-peak structure => beta ~ 0 or negative.
Also dumps per-sub-band profiles so multi-peak structure is visible.

  python within_chime_test.py <burst> [nsub=4]
"""
import os, sys, json
import numpy as np
REPO = os.environ.get("FLITS_REPO", "/home/jfaber/flits/dsa110-FLITS")
RUNS = os.environ.get("FLITS_RUNS", "/central/scratch/jfaber/flits-runs")
sys.path.insert(0, f"{REPO}/scattering")
import yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scat_analysis.config_utils import load_telescope_block
from scat_analysis.pipeline.io import BurstDataset


def prepare(cfg_path, name, outdir):
    cfg = yaml.safe_load(open(cfg_path))
    tel = load_telescope_block(cfg["telcfg_path"], cfg["telescope"])
    ds = BurstDataset(cfg["path"], outdir, name=name, telescope=tel,
                      f_factor=int(cfg["f_factor"]), t_factor=int(cfg["t_factor"]),
                      outer_trim=float(cfg.get("outer_trim", 0.15)))
    return ds.model


def onpulse_widths(prof, t):
    """Model-free widths of a 1-D profile. Baseline from outer 30%."""
    n = prof.size
    edge = np.r_[prof[:int(0.15*n)], prof[-int(0.15*n):]]
    base = np.median(edge); sig = 1.4826*np.median(np.abs(edge-base)) + 1e-9
    p = prof - base
    on = p > 3*sig
    if on.sum() < 3:
        return np.nan, np.nan, p
    pk = np.argmax(p)
    # rms width over on-pulse
    idx = np.where(on)[0]; w = p[idx]; tc = t[idx]
    mu = np.sum(tc*w)/np.sum(w)
    w_rms = np.sqrt(max(np.sum(w*(tc-mu)**2)/np.sum(w), 0.0))
    # exponential tail: fit log(p) on the trailing edge from peak to where p>3sig
    tail = np.arange(pk, n)
    tail = tail[p[tail] > 3*sig]
    w_tail = np.nan
    if tail.size >= 4:
        yy = np.log(p[tail]); xx = t[tail]
        a = np.polyfit(xx-xx[0], yy, 1)[0]   # slope of log-decay
        if a < 0:
            w_tail = -1.0/a                   # e-folding time (ms)
    return w_rms, w_tail, p


def main():
    b = sys.argv[1]; nsub = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    out = f"{RUNS}/data/joint"
    m = prepare(f"{RUNS}/configs/{b}_chime_run.yaml", f"{b}_chime", out)
    freq, t, data = m.freq, m.time, m.data            # (nch, ntime), GHz ascending
    nch = freq.size
    edges = np.linspace(0, nch, nsub+1).astype(int)
    rows = []
    fig, axes = plt.subplots(nsub, 1, figsize=(7, 2.0*nsub), sharex=True)
    for i in range(nsub):
        sl = slice(edges[i], edges[i+1])
        prof = np.nansum(data[sl], axis=0)
        fc = float(np.mean(freq[sl]))
        w_rms, w_tail, p = onpulse_widths(prof, t)
        rows.append((fc, w_rms, w_tail))
        ax = axes[i]; ax.plot(t, p, "k", lw=0.8)
        ax.set_title(f"{b} CHIME {freq[sl][0]:.3f}-{freq[sl][-1]:.3f} GHz  w_rms={w_rms:.3f} w_tail={w_tail:.3f} ms", fontsize=8)
    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout(); fp = f"{out}/{b}_chime_subband_profiles.png"; fig.savefig(fp, dpi=110)

    rows = np.array(rows)  # fc, w_rms, w_tail
    fc = rows[:,0]
    def slope(w):
        ok = np.isfinite(w) & (w > 0)
        if ok.sum() < 2: return np.nan
        return -np.polyfit(np.log(fc[ok]), np.log(w[ok]), 1)[0]  # width ~ nu^-beta
    b_rms, b_tail = slope(rows[:,1]), slope(rows[:,2])
    print(f"{b}: within-CHIME width-freq scaling beta (width~nu^-beta):  rms={b_rms:.2f}  tail={b_tail:.2f}")
    for fcc, wr, wt in rows:
        print(f"    nu={fcc:.3f} GHz  w_rms={wr:.3f}  w_tail={wt:.3f} ms")
    print(f"    wrote {fp}")
    json.dump({"burst": b, "beta_rms": b_rms, "beta_tail": b_tail,
               "subbands": rows.tolist()}, open(f"{out}/{b}_within_chime.json", "w"), indent=2)


if __name__ == "__main__":
    main()
