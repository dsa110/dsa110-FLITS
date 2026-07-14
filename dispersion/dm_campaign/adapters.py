"""Uniform adapter contract over all five DM estimators (Phase 1).

`ADAPTERS[name].measure(waterfall, freq_ghz, dt_ms, dm_ref, window) -> DMResult`
with waterfall (n_chan, n_time) dedispersed at dm_ref, absolute
`dm = dm_ref + recovered residual` in the physical residual sign (positive =
low frequencies arrive later). Published packages run as released; the two
import-environment shims (mpi4py stub, module-global geometry setup for the
DM-power CLI script) are documented in pipeline/external/README.md.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # both vendored modules import pyplot

import numpy as np

from dispersion.chime_dm import _coarse_dm, measure_dm
from dispersion.dm_power_analysis import measure_dm_power
from dispersion.dmphasev2 import DMPhaseEstimator, dmphase_trial_to_physical_residual_dm

_EXTERNAL = Path(__file__).resolve().parents[2] / "external"
DM_STEP = 0.25
N_BOOT = 100


@dataclass
class DMResult:
    """curve contract: key 'residual_dm' is ALWAYS the physical residual-DM
    axis (positive = low frequencies arrive later, relative to dm_ref); any
    second key is the method's search metric on that axis. Adapters convert
    from their native convention (absolute DM, trial sign) here, so plots and
    downstream consumers never re-derive per-method sign/offset conventions."""

    dm: float | None
    sigma: float | None
    curve: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def _grid(window):
    return np.arange(-window, window + DM_STEP, DM_STEP)


def _measure_arrival_regression(wf, freqs_mhz, dt_s, dm_ref, window):
    res = measure_dm(wf, freqs_mhz, dt_s, dm_ref, dm_window=window, dm_step=DM_STEP)
    return DMResult(res["dm"], res["dm_err"],
                    curve={"residual_dm": np.asarray(res["coarse_curve"]["dm"]) - dm_ref,
                           "snr": np.asarray(res["coarse_curve"]["snr"])},
                    meta=res)


def _measure_dmphase_intree(wf, freqs_mhz, dt_s, dm_ref, window):
    grid = _grid(window)
    est = DMPhaseEstimator(wf.T, freqs_mhz, dt_s, grid, n_boot=N_BOOT, random_state=0)
    dm = dm_ref + dmphase_trial_to_physical_residual_dm(est.dm_best)
    phys = -grid[::-1]  # trial axis -> physical residual (sign flip, re-sorted)
    return DMResult(float(dm), float(est.dm_sigma),
                    curve={"residual_dm": phys, "power": np.asarray(est.dm_curve)[::-1]},
                    meta={"trial_dm_best": float(est.dm_best)})


def _measure_dmpower_intree(wf, freqs_mhz, dt_s, dm_ref, window):
    res = measure_dm_power(wf, freqs_mhz, dt_s, dm_ref, _grid(window),
                           n_boot=N_BOOT, random_state=0)
    return DMResult(res["dm"], res["dm_err"],
                    curve={"residual_dm": np.asarray(res["residual_dm_grid"]),
                           "score": np.asarray(res["score_curve"])},
                    meta=res)


def _measure_dm_phase_published(wf, freqs_mhz, dt_s, dm_ref, window):
    sys.path.insert(0, str(_EXTERNAL / "DM_phase"))
    try:
        import DM_phase
    finally:
        sys.path.pop(0)
    order = np.argsort(freqs_mhz)
    wf, freqs = wf[order], np.asarray(freqs_mhz, float)[order]
    # Two-stage per plan: method-independent coarse centering, then the
    # package's native fine search. Handing get_dm the coarse 0.25 grid
    # directly crashes it when the phase-power peak is narrower than the step
    # (its half-max window collapses to zero points -> empty polyfit).
    ddm_c, *_ = _coarse_dm(wf, freqs, dt_s, float(freqs.max()), window, DM_STEP)
    fine = ddm_c + np.arange(-1.0, 1.0 + 0.02, 0.02)
    # docstring says f_channels is a list, but the code exponentiates it -> ndarray
    dm_res, dm_std = DM_phase.get_dm(np.asarray(wf, float), fine, dt_s, freqs,
                                     ref_freq="top", no_plots=True)
    return DMResult(float(dm_ref + dm_res), float(dm_std),
                    curve={"residual_dm": fine},
                    meta={"residual_dm": float(dm_res), "coarse_ddm": float(ddm_c)})


def _import_dm_power_published():
    if "mpi4py" not in sys.modules:
        try:
            import mpi4py  # noqa: F401  (real install wins if present)
        except ImportError:
            mpi = types.ModuleType("mpi4py.MPI")

            class _Comm:
                def Get_size(self):
                    return 1

                def Get_rank(self):
                    return 0

            mpi.COMM_WORLD = _Comm()
            rc = types.ModuleType("mpi4py.rc")
            rc.threads = False
            stub = types.ModuleType("mpi4py")
            stub.__path__ = []  # mark as package so `import mpi4py.rc` resolves
            stub.rc, stub.MPI = rc, mpi
            sys.modules["mpi4py"] = stub
            sys.modules["mpi4py.rc"] = rc
            sys.modules["mpi4py.MPI"] = mpi
    sys.path.insert(0, str(_EXTERNAL / "DM-power"))
    # The released module runs a required-args argparse at import; feed it a
    # placeholder argv (every consumed global is overwritten per call).
    argv_stash = sys.argv
    sys.argv = ["DM_power.py", "-bw", "200", "-f0", "550", "-nchan", "256",
                "-dt", "0.00016384", "-dm_start", "-5", "-dm_end", "5",
                "-dm_steps", "41", "-trials", "20", "-rescaled", ""]
    try:
        import DM_power
    finally:
        sys.argv = argv_stash
        sys.path.pop(0)
    return DM_power


def _measure_dm_power_published(wf, freqs_mhz, dt_s, dm_ref, window, trials=20):
    import astropy.units as u

    dmp = _import_dm_power_published()
    grid = _grid(window)
    # The released interface is a CLI script reading geometry from module
    # globals; set them exactly as its __main__ does.
    dmp.dt = dt_s * u.s
    dmp.nchan = wf.shape[0]
    dmp.f_arr = np.asarray(freqs_mhz, float) * u.MHz
    dmp.chan_bw = float(np.abs(np.diff(freqs_mhz)).mean()) * u.MHz
    dmp.dm_series = grid
    np.random.seed(0)  # released code bootstraps channels via the global RNG
    with contextlib.redirect_stdout(io.StringIO()):
        _, log_dfr, _, dm_power_log = dmp.get_power(np.asarray(wf, float), grid, trials)
        _, fit_dm = dmp.fit_log_dm_width(log_dfr, trials, dm_power_log)
        opt_dm = dmp.plot_dm_err(fit_dm, log_dfr)
    # sigma: std over bootstrap trials of the per-trial weighted mean -- the
    # single-process analogue of the released multi_tests() spread; weights
    # mirror plot_dm_err (inverse variance across trials per delay-freq bin).
    w = np.nanstd(fit_dm, axis=0) ** -2
    per_trial = np.nansum(fit_dm * w, axis=1) / np.nansum(w)
    sigma = float(np.nanstd(per_trial, ddof=1))
    score = np.nanmean(np.nanmean(dm_power_log, axis=1), axis=1)
    meta = {"residual_dm": float(opt_dm), "trials": trials}
    if not (np.isfinite(opt_dm) and np.isfinite(sigma) and sigma > 0):
        meta["reason"] = "non-finite fit (delay-bin Gaussian fits failed)"
        return DMResult(None, None, curve={"residual_dm": grid, "score": score}, meta=meta)
    return DMResult(float(dm_ref + opt_dm), sigma,
                    curve={"residual_dm": grid, "score": score}, meta=meta)


class _Adapter:
    def __init__(self, name, fn):
        self.name, self._fn = name, fn

    def measure(self, waterfall, freq_ghz, dt_ms, dm_ref, window=5.0):
        # Uniform input cleaning at the contract boundary: real CHIME products
        # carry NaN-masked channels; some estimators handle them internally,
        # the released DM-power does not (SVD fails). Every adapter must see
        # the identical waterfall, so zero-fill once here.
        wf = np.nan_to_num(np.asarray(waterfall, float))
        return self._fn(wf, np.asarray(freq_ghz, float) * 1e3,
                        dt_ms * 1e-3, float(dm_ref), float(window))


ADAPTERS = {
    "arrival_regression": _Adapter("arrival_regression", _measure_arrival_regression),
    "dmphase_variant_intree": _Adapter("dmphase_variant_intree", _measure_dmphase_intree),
    "dmpower_variant_intree": _Adapter("dmpower_variant_intree", _measure_dmpower_intree),
    "dm_phase_published": _Adapter("dm_phase_published", _measure_dm_phase_published),
    "dm_power_published": _Adapter("dm_power_published", _measure_dm_power_published),
}
