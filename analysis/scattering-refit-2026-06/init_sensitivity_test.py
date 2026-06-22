"""Init-sensitivity test: does the data-driven initial guess change the nested
posterior vs a trivial coarse guess? If not, the ~930-line estimator is moot.
Same model (M3), same priors (built from init), same nlive/pool/alpha_fixed."""
import pathlib, numpy as np
from flits.scattering.scat_analysis.config_utils import load_config
from flits.scattering.scat_analysis.pipeline import BurstPipeline, BurstDataset
from flits.scattering.scat_analysis.burstfit import FRBParams
from flits.scattering.scat_analysis.burstfit_init import data_driven_initial_guess
from flits.scattering.scat_analysis.burstfit_nested import fit_single_model_nested

CFG = "/Users/jakobfaber/Developer/scratch/2026-06/flits-refit/wilhelm_chime_refit.yaml"


def summ(res):
    i = res.param_names.index("tau_1ghz")
    s = res.samples[:, i]; w = res.weights
    o = np.argsort(s); cs = np.cumsum(w[o]); cs /= cs[-1]
    q = lambda p: float(s[o][np.searchsorted(cs, p)])
    return res.log_evidence, q(0.16), q(0.50), q(0.84)


def main():
    cfg = load_config(CFG)
    out = pathlib.Path("/tmp/init_test"); out.mkdir(exist_ok=True)
    pipe = BurstPipeline(inpath=cfg.path, outpath=out, name=cfg.path.stem.split("_")[0],
                         dm_init=cfg.dm_init, telescope=cfg.telescope, sampler=cfg.sampler,
                         f_factor=cfg.pipeline.f_factor, t_factor=cfg.pipeline.t_factor,
                         steps=cfg.pipeline.steps, nproc=8, fitting_method="nested",
                         outer_trim=cfg.pipeline.outer_trim, nlive=400, dlogz=0.5,
                         nlive_walks=15, alpha_fixed=4.0)
    pipe.dataset = BurstDataset(inpath=pipe.inpath, outpath=pipe.outpath, name=pipe.name,
                                telescope=cfg.telescope, sampler=cfg.sampler,
                                f_factor=cfg.pipeline.f_factor, t_factor=cfg.pipeline.t_factor,
                                outer_trim=cfg.pipeline.outer_trim)
    pipe.dataset.dm_init = pipe.dm_init; pipe.dataset.model.dm_init = pipe.dm_init
    model = pipe.dataset.model

    dd = data_driven_initial_guess(model.data, model.freq, model.time, dm=0.0, verbose=False).params
    prof = np.nansum(model.data, axis=0)
    triv = FRBParams(c0=float(np.nansum(model.data)/max(model.data.shape[0], 1)),
                     t0=float(model.time[np.nanargmax(prof)]),
                     gamma=0.0, zeta=0.5, tau_1ghz=1.0, alpha=4.0, delta_dm=0.0)
    print(f"data-driven init: t0={dd.t0:.2f} c0={dd.c0:.3f} g={dd.gamma:.2f} z={dd.zeta:.3f} tau={dd.tau_1ghz:.4f}")
    print(f"trivial     init: t0={triv.t0:.2f} c0={triv.c0:.3f} g={triv.gamma:.2f} z={triv.zeta:.3f} tau={triv.tau_1ghz:.4f}")

    print(f"\n{'init':<13} {'logZ':>11} {'tau_p16':>9} {'tau_p50':>9} {'tau_p84':>9}")
    for label, init in [("data-driven", dd), ("trivial", triv)]:
        res = fit_single_model_nested(model=model, init=init, model_key="M3", nlive=400,
                                      dlogz=0.5, alpha_fixed=4.0, nproc=8, verbose=False)
        lz, p16, p50, p84 = summ(res)
        print(f"{label:<13} {lz:>11.2f} {p16:>9.4f} {p50:>9.4f} {p84:>9.4f}")


if __name__ == "__main__":
    main()
