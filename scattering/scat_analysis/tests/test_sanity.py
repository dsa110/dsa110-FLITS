from __future__ import annotations

import numpy as np

from FLITS.scattering.scat_analysis.burstfit import FRBParams, FRBModel


def test_frbparams_defaults():
    p = FRBParams(c0=1.0, t0=0.0, gamma=-1.6)
    assert p.alpha == 4.4
    assert p.delta_dm == 0.0


def test_model_call_shapes():
    time = np.linspace(-5.0, 5.0, 256)
    freq = np.linspace(0.4, 0.8, 64)
    data = np.zeros((freq.size, time.size))
    m = FRBModel(time=time, freq=freq, data=data, dm_init=0.0, df_MHz=0.39)
    p = FRBParams(c0=1.0, t0=0.0, gamma=-1.6, zeta=0.1, tau_1ghz=0.2, alpha=4.4, delta_dm=0.0)
    out = m(p, "M3")
    assert out.shape == data.shape

