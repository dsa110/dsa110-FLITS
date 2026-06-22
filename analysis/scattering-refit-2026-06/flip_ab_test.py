"""Definitive A/B: fit oran with flip_freq False (current) vs True (data flipped
to ascending to match the freq axis). If flip materially changes logZ/tau/chi2,
the current CHIME fits are corrupted, not just the plots."""
import sys, pathlib, numpy as np
from flits.scattering.scat_analysis.config_utils import load_config
from flits.scattering.scat_analysis.pipeline import BurstPipeline, BurstDataset
from flits.scattering.scat_analysis.burstfit import FRBParams
from flits.scattering.scat_analysis.burstfit_nested import fit_single_model_nested

CFG = sys.argv[1]
cfg = load_config(CFG)

def build(flip):
    ds = BurstDataset(inpath=cfg.path, outpath=pathlib.Path("/tmp/flipab"),
                      name=cfg.path.stem.split("_")[0], telescope=cfg.telescope,
                      sampler=cfg.sampler, f_factor=cfg.pipeline.f_factor,
                      t_factor=cfg.pipeline.t_factor, outer_trim=cfg.pipeline.outer_trim,
                      flip_freq=flip)
    ds.dm_init = cfg.dm_init; ds.model.dm_init = cfg.dm_init
    return ds.model

def tail_dir(m):
    # centroid offset (late tail +) for low vs high row groups
    prof = np.nansum(m.data, axis=0); pk = np.nanargmax(prof); t = np.arange(m.data.shape[1]) - pk
    def off(rows):
        p = np.nansum(m.data[rows], axis=0); p = p - np.median(p); p[p < 0] = 0
        return float(np.sum(t * p) / (p.sum() + 1e-9))
    n = m.data.shape[0]
    return off(slice(0, n // 4)), off(slice(3 * n // 4, n))

def fit_and_chi2(m):
    res = fit_single_model_nested(model=m, init=__import__("flits.scattering.scat_analysis.burstfit_init",
        fromlist=["data_driven_initial_guess"]).data_driven_initial_guess(
        m.data, m.freq, m.time, dm=0.0, verbose=False).params,
        model_key="M3", nlive=400, dlogz=0.5, alpha_fixed=4.0, nproc=8, verbose=False)
    i = res.param_names.index("tau_1ghz"); s = res.samples[:, i]; w = res.weights
    o = np.argsort(s); cs = np.cumsum(w[o]); cs /= cs[-1]
    q = lambda p: float(s[o][np.searchsorted(cs, p)])
    # chi2 at median params
    med = {n: float(np.sum(res.samples[:, j] * res.weights) / res.weights.sum())
           for j, n in enumerate(res.param_names)}
    p = FRBParams(c0=med.get("c0", 1), t0=med.get("t0", 0), gamma=med.get("gamma", 0),
                  zeta=med.get("zeta", 0.5), tau_1ghz=med.get("tau_1ghz", 0.1),
                  alpha=4.0, delta_dm=med.get("delta_dm", 0))
    model = m(p, "M3"); V = m.valid
    r = (m.data[V] - model[V]) / m.noise_std[V, None]
    r = r[np.isfinite(r)]
    chi2 = float(np.sum(r ** 2) / (r.size - len(res.param_names)))
    return res.log_evidence, q(0.16), q(0.50), q(0.84), chi2, float(r.std())

def main():
    for flip in (False, True):
        m = build(flip)
        lo, hi = tail_dir(m)
        lz, t16, t50, t84, chi2, rsig = fit_and_chi2(m)
        print(f"flip={flip!s:5}  tail(row0end={lo:.0f}, lastrow={hi:.0f})  "
              f"logZ={lz:.1f}  tau50={t50:.4f} [{t16:.4f},{t84:.4f}]  chi2={chi2:.3f}  residσ={rsig:.3f}")


if __name__ == "__main__":
    main()
