"""Phase-1 adapter contract: every estimator behind one uniform interface.

Recovery is asserted on the bright standard case for every (adapter,
geometry) combination; combinations Phase 0 measured as too coarse on a
geometry are xfailed with that finding as the reason (xfail_strict: an
unexpected pass is itself a signal).
"""

import numpy as np
import pytest

from dispersion.dm_campaign.adapters import ADAPTERS
from dispersion.dm_campaign.injection import standard_bright_case


@pytest.mark.parametrize("instrument", ["dsa", "chime"])
@pytest.mark.parametrize("name", list(ADAPTERS))
def test_adapter_recovers_bright_injection(name, instrument):
    wf, freq, dt, truth = standard_bright_case(instrument=instrument, seed=3)
    res = ADAPTERS[name].measure(
        wf, freq_ghz=freq, dt_ms=dt, dm_ref=truth["dm_ref"], window=truth["window"]
    )
    assert res.dm is not None and res.sigma is not None
    assert res.sigma > 0
    # Floor 0.05: quoted sigmas are known miscalibrated (Phase 0, up to ~4.5x
    # under-quoted), so a raw 3-sigma gate false-alarms on calibration noise.
    # 3*0.05 = 0.15 still sits far below the sign-flip signature (2*0.7 = 1.4)
    # and the +-0.5 science requirement -- plumbing errors cannot hide under it.
    assert abs(res.dm - truth["dm_true"]) < 3 * max(res.sigma, 0.05)


@pytest.mark.parametrize("name", list(ADAPTERS))
def test_adapter_returns_search_curve(name):
    wf, freq, dt, truth = standard_bright_case(instrument="chime", seed=5)
    res = ADAPTERS[name].measure(
        wf, freq_ghz=freq, dt_ms=dt, dm_ref=truth["dm_ref"], window=truth["window"]
    )
    assert "residual_dm" in res.curve  # normalized physical-residual axis
    trial = np.asarray(res.curve["residual_dm"])
    assert trial.ndim == 1 and trial.size >= 3
    assert np.all(np.diff(trial) > 0)  # ascending physical axis, any native sign
