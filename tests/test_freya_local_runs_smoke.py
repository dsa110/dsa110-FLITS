"""Band-prep smoke test for the freya local-runs configs (issue #99).

Proves the freya CHIME+DSA run-configs are consumable by the local joint-fit
driver's own prepare() step — config load, telescope block, real .npy load,
bandpass/trim/downsample/crop, FRBModel construction, data-driven init — with
no driver changes. Asserts shapes and ascending-frequency orientation, not
fitted values.

Real data resolves through the pinned manuscript checkout's data symlinks
(external to this repo), so the test skips wherever those are not staged —
same convention as test_flux_cal.py.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_RUNS = REPO_ROOT / "analysis" / "scattering-refit-2026-06" / "local_runs"
DRIVER = LOCAL_RUNS / "run_joint_fit.py"

BANDS = {
    # band -> (config filename, expected n_chan = raw_nchan // f_factor)
    "chime": ("freya_chime_run.yaml", 64),  # 1024 // 16
    "dsa": ("freya_dsa_run.yaml", 16),  # 6144 // 384
}


def _load_driver(monkeypatch):
    # The driver resolves its repo from FLITS_REPO at import time (HPCC default
    # otherwise) and prepends {REPO}/scattering to sys.path for scat_analysis.
    monkeypatch.setenv("FLITS_REPO", str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("run_joint_fit", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.slow
@pytest.mark.parametrize("band", sorted(BANDS))
def test_freya_band_prep_smoke(band, monkeypatch, tmp_path):
    cfg_name, n_chan_expected = BANDS[band]
    cfg_path = LOCAL_RUNS / "configs" / cfg_name
    cfg = yaml.safe_load(cfg_path.read_text())

    data_path = Path(cfg["path"])
    if not data_path.exists():
        pytest.skip(f"{data_path.name} not staged (pinned-checkout data symlink)")
    if not Path(cfg["telcfg_path"]).exists():
        pytest.skip("pinned-checkout telescopes.yaml not staged")

    driver = _load_driver(monkeypatch)
    model, init = driver.prepare(str(cfg_path), f"freya_{band}", str(tmp_path))

    raw_nf, raw_nt = np.load(data_path, mmap_mode="r").shape
    assert raw_nf % int(cfg["f_factor"]) == 0
    assert raw_nf // int(cfg["f_factor"]) == n_chan_expected

    # Shape: channels exact; time bounded by the trimmed+downsampled full
    # window (on-pulse crop, env-default on in the driver, only shrinks it).
    nt_trim = raw_nt - 2 * int(float(cfg["outer_trim"]) * raw_nt)
    nt_full = nt_trim // int(cfg["t_factor"])
    assert model.data.ndim == 2
    assert model.data.shape[0] == n_chan_expected
    assert 0 < model.data.shape[1] <= nt_full
    assert model.data.shape == (model.freq.size, model.time.size)

    # Orientation + band: ascending frequency spanning the telescope block.
    tel = yaml.safe_load(Path(cfg["telcfg_path"]).read_text())[cfg["telescope"]]
    assert np.all(np.diff(model.freq) > 0)
    assert np.isclose(model.freq[0], tel["f_min_GHz"])
    assert np.isclose(model.freq[-1], tel["f_max_GHz"])

    # Prepared band is usable: finite S/N-unit data, per-channel noise vector,
    # and a data-driven init the sampler could start from.
    assert np.isfinite(model.data).all()
    assert model.noise_std.shape == (n_chan_expected,)
    assert np.all(model.noise_std >= 0) and np.any(model.noise_std > 0)
    assert np.isfinite(init.tau_1ghz) and init.tau_1ghz > 0
