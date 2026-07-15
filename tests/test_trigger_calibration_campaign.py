"""Campaign-kernel smoke tests for the A1 trigger calibration (Phase 4).

Small-n smoke of the null-cell kernel (full grid runs as an h17 batch via
simulation/scripts/run_a1_trigger_calibration.py) plus the deterministic
threshold-envelope logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "simulation"))
sys.path.insert(0, str(_root / "scintillation"))

from trigger_calibration import null_dlnz_cell, threshold_table  # noqa: E402


@pytest.mark.slow
def test_null_cell_produces_finite_dlnz_sample():
    d = null_dlnz_cell(
        dnu_hwhm_mhz=0.4, snr=25.0, band_width_mhz=6.0,
        channel_width_mhz=0.05, num_subbands=2,
        n_real=3, seed=3, nlive=200, dlogz=1.0, n_real_cov=60,
    )
    assert len(d) == 3
    assert sum(np.isfinite(x) for x in d) >= 2


def test_threshold_table_monotone_and_envelope():
    fake = {
        ("c1",): [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 8.0],
        ("c2",): [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    }
    t = threshold_table(fake, rates=(0.005, 0.01, 0.05))
    assert t[0.005] >= t[0.01] >= t[0.05]
    # envelope: c1's tail dominates
    assert t[0.05] >= float(np.quantile(fake[("c1",)], 0.95)) - 1e-9


def test_threshold_table_rejects_failure_heavy_cell():
    fake = {("bad",): [np.nan, np.nan, np.nan, 1.0]}
    with pytest.raises(ValueError, match="evidence failures"):
        threshold_table(fake, rates=(0.05,))
