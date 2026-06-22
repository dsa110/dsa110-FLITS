"""Validate the wired plot_fit_quality (the package function the pipeline calls),
using oran's already-computed results json — no full re-fit."""
import sys, json, pathlib
import numpy as np
from flits.scattering.scat_analysis.config_utils import load_config
from flits.scattering.scat_analysis.pipeline import BurstPipeline, BurstDataset
from flits.scattering.scat_analysis.burstfit import FRBParams
from flits.scattering.scat_analysis.visualization import plot_fit_quality   # the wired fn

CFG, JSON, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = load_config(CFG)
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
d = json.load(open(JSON)); bp = d["best_params"]
p = FRBParams(c0=bp["c0"], t0=bp["t0"], gamma=bp["gamma"], zeta=bp["zeta"],
              tau_1ghz=bp["tau_1ghz"], alpha=bp.get("alpha", 4.0), delta_dm=bp["delta_dm"])
model = m(p, d["best_model"])
plot_fit_quality(data=m.data, model=model, freq=m.freq, time=m.time,
                 noise=m.noise_std, valid=m.valid, params=p, results=d,
                 output_path=pathlib.Path(OUT), burst_name=pipe.name, telescope=cfg.telescope)
print(f"OK wrote {OUT}")
