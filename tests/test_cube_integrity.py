import os
import pathlib

import numpy as np
import pytest

from scripts.cube_crosscheck import direct_verdict, robust_snr

DATA = pathlib.Path(os.environ.get("FLITS_DATA", "~/Data/Faber2026")).expanduser()
CUBES = sorted(DATA.rglob("*_32000b_cntr_bpc.npy"))

pytestmark = pytest.mark.skipif(not CUBES, reason="data root absent")


@pytest.mark.parametrize("cube", CUBES, ids=lambda p: p.stem)
def test_cube_intact(cube):
    """A cube is intact if its direct verdict is not `fail` or `missing-cube`.

    `pass` = burst centered, no edge significance. `no-burst-above-5sigma` =
    low-S/N burst not detected above 5σ in the frequency-averaged profile —
    the edge/center checks are vacuously satisfied (no hot bins), so the cube
    is intact, just below the direct-detection threshold. `fail` (edge
    significance or off-center centroid) is the corruption signal.
    """
    arr = np.load(cube, mmap_mode="r")
    prof = np.nanmean(np.asarray(arr, float), axis=0)
    snr = robust_snr(prof)
    verdict, _max_snr, _edge_ok, _centroid = direct_verdict(snr)
    assert verdict in {"pass", "no-burst-above-5sigma"}, (
        f"cube {cube.stem}: verdict={verdict} (expected pass or no-burst-above-5sigma)"
    )
