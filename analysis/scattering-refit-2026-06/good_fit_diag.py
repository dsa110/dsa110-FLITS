"""Richer single-burst scattering diagnostic than the stock 4-panel figure.

The stock figure shows data / model / model+noise / residual as *separate*
waterfalls, so you cannot directly see whether the model pulse shape matches the
data — and chi2_red≈1 with R^2≈0.05 (faint burst) is visually ambiguous. This
overlays data vs model on shared axes with a sigma-residual strip, checks the
scattering tail per sub-band (the actual scattering signature), and tests
residual whiteness. Run ON HPCC (data + code live there).

Usage: python good_fit_diag.py <config.yaml> <fit_results.json> <out.png>
"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flits.scattering.scat_analysis.config_utils import load_config
from flits.scattering.scat_analysis.pipeline import BurstPipeline, BurstDataset
from flits.scattering.scat_analysis.burstfit import FRBParams

CFG, JSON, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = load_config(CFG)
import pathlib
pipe = BurstPipeline(inpath=cfg.path, outpath=pathlib.Path("/tmp/diag"),
                     name=cfg.path.stem.split("_")[0], dm_init=cfg.dm_init,
                     telescope=cfg.telescope, sampler=cfg.sampler, nproc=1,
                     f_factor=cfg.pipeline.f_factor, t_factor=cfg.pipeline.t_factor,
                     steps=cfg.pipeline.steps, outer_trim=cfg.pipeline.outer_trim)
ds = BurstDataset(inpath=pipe.inpath, outpath=pipe.outpath, name=pipe.name,
                  telescope=cfg.telescope, sampler=cfg.sampler,
                  f_factor=cfg.pipeline.f_factor, t_factor=cfg.pipeline.t_factor,
                  outer_trim=cfg.pipeline.outer_trim)
ds.dm_init = pipe.dm_init; ds.model.dm_init = pipe.dm_init
m = ds.model
data, freq, time, ns, valid = m.data, m.freq, m.time, m.noise_std, m.valid

d = json.load(open(JSON))
bp = d["best_params"]
p = FRBParams(c0=bp["c0"], t0=bp["t0"], gamma=bp["gamma"], zeta=bp["zeta"],
              tau_1ghz=bp["tau_1ghz"], alpha=bp.get("alpha", 4.0), delta_dm=bp["delta_dm"])
model = m(p, d["best_model"])
gof = d["goodness_of_fit"]

V = valid                                    # per-channel validity mask
resid = (data - model)
# frequency-integrated profile (unweighted sum over valid channels) + its noise
prof_d = np.nansum(data[V], axis=0)
prof_m = np.nansum(model[V], axis=0)
prof_sig = np.sqrt(np.nansum(ns[V] ** 2))    # scalar: noise on summed profile per bin
prof_res = (prof_d - prof_m) / prof_sig

fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(3, 4, height_ratios=[2.4, 1, 1.4], hspace=0.35, wspace=0.3)

# (1) integrated profile overlay + residual strip — the key fit-quality view
ax = fig.add_subplot(gs[0, :2])
ax.step(time, prof_d, where="mid", color="0.3", lw=1, label="data")
ax.plot(time, prof_m, color="crimson", lw=2, label=f"model {d['best_model']}")
ax.set_title(f"{pipe.name.upper()}  freq-integrated  "
             f"τ={p.tau_1ghz:.3f} ζ={p.zeta:.3f} ms  "
             f"χ²ᵣ={gof['chi2_reduced']:.3f}  R²={gof['r_squared']:.3f}")
ax.legend(loc="upper right", fontsize=9); ax.set_ylabel("flux (Σ chan)")
axr = fig.add_subplot(gs[1, :2], sharex=ax)
axr.axhspan(-3, 3, color="0.85"); axr.axhspan(-1, 1, color="0.7")
axr.step(time, prof_res, where="mid", color="navy", lw=0.8)
axr.axhline(0, color="k", lw=0.5); axr.set_ylabel("resid (σ)"); axr.set_xlabel("time (ms)")

# (2) data / model / residual waterfalls, shared scale
vmax = np.nanpercentile(data[V], 99)
for j, (arr, ttl, cmap, lo, hi) in enumerate([
        (data, "data", "viridis", 0, vmax),
        (model, "model", "viridis", 0, vmax),
        (resid / ns[:, None], "residual (σ)", "coolwarm", -3, 3)]):
    a = fig.add_subplot(gs[2, j])
    a.imshow(arr, aspect="auto", origin="lower", cmap=cmap, vmin=lo, vmax=hi,
             extent=[time[0], time[-1], freq[0], freq[-1]])
    a.set_title(ttl, fontsize=10); a.set_xlabel("time (ms)")
    if j == 0: a.set_ylabel("freq (GHz)")

# (3) residual whiteness histogram
ax_h = fig.add_subplot(gs[2, 3])
rn = (resid[V] / ns[V, None]).ravel(); rn = rn[np.isfinite(rn)]
ax_h.hist(rn, bins=60, density=True, color="navy", alpha=0.6)
xx = np.linspace(-4, 4, 200)
ax_h.plot(xx, np.exp(-xx**2/2)/np.sqrt(2*np.pi), "crimson", lw=2)
ax_h.set_title(f"resid hist  μ={rn.mean():.2f} σ={rn.std():.2f}", fontsize=10)
ax_h.set_xlabel("residual (σ)")

# (4) sub-band profile overlays in the upper-right block — scattering tail vs freq
nsub = 4
edges = np.linspace(0, V.sum(), nsub + 1).astype(int)
vidx = np.where(V)[0]
ax_sb = fig.add_subplot(gs[0:2, 2:])
off = 0.0
for k in range(nsub):
    chans = vidx[edges[k]:edges[k+1]]
    if chans.size == 0:
        continue
    pd = np.nansum(data[chans], axis=0); pm = np.nansum(model[chans], axis=0)
    sc = np.nanmax(pm) if np.nanmax(pm) > 0 else 1.0
    fctr = freq[chans].mean()
    ax_sb.step(time, pd/sc + off, where="mid", color="0.4", lw=0.8)
    ax_sb.plot(time, pm/sc + off, color="crimson", lw=1.5)
    ax_sb.text(time[0], off + 0.6, f"{fctr:.2f} GHz", fontsize=8, color="navy")
    off += 1.3
ax_sb.set_title("per-sub-band profile (data vs model) — scattering tail vs ν", fontsize=10)
ax_sb.set_xlabel("time (ms)"); ax_sb.set_yticks([])

fig.savefig(OUT, dpi=110, bbox_inches="tight")
print(f"wrote {OUT}  chi2_red={gof['chi2_reduced']:.3f} R2={gof['r_squared']:.3f} "
      f"resid_mu={rn.mean():.3f} resid_sigma={rn.std():.3f}")
