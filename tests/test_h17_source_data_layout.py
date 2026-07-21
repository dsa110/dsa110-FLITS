import importlib.util
from pathlib import Path
import sys


SCRIPT = (
    Path(__file__).parents[1]
    / "analysis"
    / "scattering-refit-2026-06"
    / "baseband_recovery"
    / "upchannelize_chime.py"
)
SPEC = importlib.util.spec_from_file_location("upchannelize_chime", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
worker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worker
SPEC.loader.exec_module(worker)


def test_local_h5_path_uses_project_directory():
    assert worker._local_h5_path(
        "zach", "2022/02/07/astro_210456524/singlebeam_210456524.h5"
    ) == Path("/data/Faber2026/data/chime-frb/zach/singlebeam_210456524.h5")
    assert worker._local_h5_path(
        "freya", "2023/03/25/astro_278720455/singlebeam_278720455.h5"
    ) == Path("/data/Faber2026/data/chime-frb/freya/singlebeam_278720455.h5")
