import sys
import pathlib
import numpy as np

# Ensure package root is on the path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from flits.params import FRBParams
from flits.models import FRBModel, K_DM


def test_frbmodel_peak_time_no_dm():
    t = np.linspace(0, 10, 1001)
    freqs = np.array([1000.0])
    params = FRBParams(dm=0.0, amplitude=1.0, t0=5.0, width=0.5)
    model = FRBModel(params)
    spec = model.simulate(t, freqs)
    peak_time = t[np.argmax(spec[0])]
    assert abs(peak_time - 5.0) < 1e-3


def test_frbmodel_dispersion_delay():
    t = np.linspace(0, 10, 2001)
    freqs = np.array([1000.0, 800.0])
    params = FRBParams(dm=50.0, amplitude=1.0, t0=0.0, width=0.2)
    model = FRBModel(params)
    spec = model.simulate(t, freqs)
    peak_high = t[np.argmax(spec[0])]
    peak_low = t[np.argmax(spec[1])]
    expected_delay = K_DM * params.dm * (1 / freqs[1] ** 2 - 1 / freqs[0] ** 2)
    assert np.isclose(peak_low - peak_high, expected_delay, atol=1e-2)
