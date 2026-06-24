#!/usr/bin/env python
"""Contract check for the joint-fit figure scripts after shared zeta became the
run_joint_fit default (PR: feat/joint-fit-shared-zeta-default).

The canonical <burst>_joint_fit.json now holds the SHARED-zeta(nu) parametrization
(zeta_1ghz, x_zeta; gain-marginal, no zeta_C/zeta_D/c0_C/gamma_C). This check runs
resid_map / joint_ppc / fullband_waterfall main() against BOTH a synthetic shared
json and a synthetic per-band json, with scat_analysis + matplotlib stubbed so no
cluster data is needed, and asserts each reconstructs FRBParams without KeyError:
  - shared  -> zeta is the per-channel array zeta_1ghz*nu^x_zeta, c0=1, gamma=0
  - perband -> zeta is the stored scalar, c0/gamma read from json

Run: python check_joint_json_contract.py   (exits 0 on success, asserts on failure)
"""

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]  # analysis/scattering-refit-2026-06 -> repo root


# --- record every FRBParams built by the scripts -------------------------------
class _Params:
    instances = []

    def __init__(self, **kw):
        self.__dict__.update(kw)
        _Params.instances.append(kw)


class _Model:
    """Minimal stand-in for FRBModel: callable + the attributes the scripts touch."""

    def __init__(self, freq=None, time=None, data=None, dm_init=0.0, **_):
        self.freq = np.asarray(freq) if freq is not None else np.linspace(0.4, 0.8, 16)
        self.time = np.asarray(time) if time is not None else np.linspace(-2, 2, 24)
        self.data = np.zeros((self.freq.size, self.time.size)) if data is None else np.asarray(data)
        self.noise_std = np.ones(self.freq.size)
        self.valid = None
        self.dm_init = dm_init

    def __call__(self, p, key):
        return np.zeros((self.freq.size, self.time.size))

    def gain_spectrum(self, p, key):
        return np.ones(self.freq.size)


class _Dataset:
    def __init__(self, path, outdir, name="", **_):
        f = np.linspace(0.4, 0.8, 16) if "chime" in name else np.linspace(1.3, 1.5, 16)
        self.model = _Model(freq=f)


def _install_stubs():
    """Inject fake scat_analysis.* + matplotlib so the figure scripts import headless."""
    sa = types.ModuleType("scat_analysis")
    cu = types.ModuleType("scat_analysis.config_utils")
    cu.load_telescope_block = lambda *a, **k: {}
    pipe = types.ModuleType("scat_analysis.pipeline")
    io = types.ModuleType("scat_analysis.pipeline.io")
    io.BurstDataset = _Dataset
    bf = types.ModuleType("scat_analysis.burstfit")
    bf.FRBParams = _Params
    bf.FRBModel = _Model
    corner = types.ModuleType("corner")
    corner.corner = lambda *a, **k: None
    dyn = types.ModuleType("dynesty")
    dynu = types.ModuleType("dynesty.utils")
    dynu.resample_equal = lambda s, w: s
    for name, mod in {
        "scat_analysis": sa,
        "scat_analysis.config_utils": cu,
        "scat_analysis.pipeline": pipe,
        "scat_analysis.pipeline.io": io,
        "scat_analysis.burstfit": bf,
        "corner": corner,
        "dynesty": dyn,
        "dynesty.utils": dynu,
    }.items():
        sys.modules[name] = mod
    # headless matplotlib is already Agg in the scripts; nothing else to stub.


def _write_cfgs(runs, b):
    cfgdir = runs / "configs"
    cfgdir.mkdir(parents=True, exist_ok=True)
    for tel in ("chime", "dsa"):
        (cfgdir / f"{b}_{tel}_run.yaml").write_text(
            "telcfg_path: x\ntelescope: t\npath: p\nf_factor: 1\nt_factor: 1\n"
            "outer_trim: 0.15\ndm_init: 0.0\n"
        )


def _percentiles(d):
    # real summaries carry median/lower/upper per param; consumers read alpha/tau bounds
    return {k: {"median": v, "lower": v - 0.1, "upper": v + 0.1} for k, v in d.items()}


# common per-band medians present in BOTH contracts
_COMMON = dict(tau_1ghz=1.5, alpha=4.0, t0_C=0.0, t0_D=0.1, delta_dm_C=0.0, delta_dm_D=0.0)
_SHARED = {**_COMMON, "zeta_1ghz": 0.3, "x_zeta": -1.0}
_PERBAND = {
    **_COMMON,
    "zeta_C": 0.25,
    "zeta_D": 0.18,
    "c0_C": 1.2,
    "c0_D": 0.9,
    "gamma_C": 0.1,
    "gamma_D": -0.1,
}


def _summary(shared):
    pct = _SHARED if shared else _PERBAND
    return {"shared_zeta": shared, "marginalize_gain": False, "percentiles": _percentiles(pct)}


def _run_main(script, b, runs):
    """Import the script fresh and call main() for burst b; return built FRBParams."""
    _Params.instances = []
    os.environ["FLITS_REPO"] = str(REPO)
    os.environ["FLITS_RUNS"] = str(runs)
    spec = importlib.util.spec_from_file_location(f"_chk_{script.stem}", script)
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [str(script), b]
    spec.loader.exec_module(mod)
    mod.main()
    return list(_Params.instances)


def main():
    import tempfile

    _install_stubs()
    scripts = ["resid_map.py", "joint_ppc.py", "fullband_waterfall.py", "validate_gain.py"]
    with tempfile.TemporaryDirectory() as td:
        runs = Path(td)
        joint = runs / "data" / "joint"
        joint.mkdir(parents=True, exist_ok=True)
        b = "wilhelm"
        _write_cfgs(runs, b)
        for name in scripts:
            script = HERE / name
            for shared in (True, False):
                (joint / f"{b}_joint_fit.json").write_text(json.dumps(_summary(shared)))
                params = _run_main(script, b, runs)
                # the two band params are the first two FRBParams the script builds
                assert len(params) >= 2, f"{name}: expected >=2 FRBParams, got {len(params)}"
                pC = params[0]
                z = np.asarray(pC["zeta"])
                if shared:
                    assert z.ndim == 1 and z.size > 1, f"{name} shared: zeta not an array ({z})"
                    exp = _SHARED["zeta_1ghz"] * np.linspace(0.4, 0.8, 16) ** _SHARED["x_zeta"]
                    assert np.allclose(z, exp), f"{name} shared: zeta law mismatch"
                    assert pC["c0"] == 1.0 and pC["gamma"] == 0.0, (
                        f"{name} shared: c0/gamma not unit"
                    )
                else:
                    assert z.shape == () or z.size == 1, f"{name} perband: zeta not scalar ({z})"
                    assert float(z) == _PERBAND["zeta_C"], f"{name} perband: wrong zeta_C"
                    assert pC["c0"] == _PERBAND["c0_C"], f"{name} perband: c0 not from json"
                print(f"  ok  {name:22s} shared={shared}")

        # pure-json top-level scripts: run as subprocesses on a shared-zeta json, expect no crash
        import subprocess

        env = {**os.environ, "FLITS_RUNS": str(runs), "FLITS_REPO": str(REPO)}
        (joint / f"{b}_joint_fit.json").write_text(json.dumps(_summary(True)))
        r = subprocess.run(
            [sys.executable, str(HERE / "verify_joint_fits.py")],
            env=env,
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"verify_joint_fits.py crashed:\n{r.stderr}"
        assert b in r.stdout and "Traceback" not in r.stderr
        print("  ok  verify_joint_fits.py   runs on shared-zeta json")

        (joint / "precrop_backup").mkdir(exist_ok=True)
        (joint / "precrop_backup" / f"{b}_joint_fit.json").write_text(json.dumps(_summary(True)))
        r = subprocess.run(
            [sys.executable, str(HERE / "report_prepost.py"), b],
            env=env,
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"report_prepost.py crashed:\n{r.stderr}"
        print("  ok  report_prepost.py      runs on shared-zeta json (PRE+POST)")

    # plot_joint_posteriors.py: FOCUS must follow the fit's zeta params (import-safe, __main__-guarded)
    spec = importlib.util.spec_from_file_location("_chk_pjp", HERE / "plot_joint_posteriors.py")
    pjp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pjp)
    shared_names = ["tau_1ghz", "alpha", "zeta_1ghz", "x_zeta", "delta_dm_C", "delta_dm_D"]
    perband_names = ["tau_1ghz", "alpha", "zeta_C", "zeta_D", "delta_dm_C", "delta_dm_D"]
    assert pjp.focus_for(shared_names) == [
        "tau_1ghz",
        "alpha",
        "zeta_1ghz",
        "x_zeta",
        "delta_dm_C",
        "delta_dm_D",
    ], "focus_for shared wrong"
    assert pjp.focus_for(perband_names) == perband_names, "focus_for per-band wrong"
    print("  ok  plot_joint_posteriors.py  FOCUS follows the fit's zeta params")

    # build_joint_deck.py writes to a hard-coded path (cannot run here): assert the source guard
    deck = (HERE / "build_joint_deck.py").read_text()
    assert 'd.get("shared_zeta")' in deck, "build_joint_deck.py missing shared_zeta branch"
    print("  ok  build_joint_deck.py       has shared_zeta branch (source-checked)")

    # fullband_aligned.py only changed filename+docstring; assert the contract in source.
    src = (HERE / "fullband_aligned.py").read_text()
    assert "_joint_fit.json" in src and "_joint_fit_sharedzeta.json" not in src, (
        "fullband_aligned.py still reads the retired _sharedzeta filename"
    )
    assert '"zeta_1ghz"' in src and '"x_zeta"' in src, "fullband_aligned.py lost shared-zeta keys"
    print("  ok  fullband_aligned.py    reads canonical _joint_fit.json (shared keys)")
    print("PASS: joint-fit json contract consistent across the figure scripts")


if __name__ == "__main__":
    main()
