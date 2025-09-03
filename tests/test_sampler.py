import sys
import pathlib
import numpy as np

# Ensure package root is on the path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from flits.params import FRBParams
from flits.models import FRBModel
from flits.sampler import FRBFitter, _log_prob_wrapper


def test_fitter_improves_log_prob():
    np.random.seed(0)
    t = np.linspace(-5, 5, 512)
    freqs = np.array([800.0, 400.0])
    true_params = FRBParams(dm=100.0, amplitude=1.0, t0=0.0, width=0.5)
    model = FRBModel(true_params)
    spec = model.simulate(t, freqs)
    noise_std = 0.05
    data = spec + np.random.normal(0, noise_std, size=spec.shape)

    fitter = FRBFitter(t, freqs, data, noise_std)
    initial = np.array([80.0, 0.5])
    initial_lp = _log_prob_wrapper(initial, t, freqs, data, noise_std)
    sampler = fitter.sample(initial, nwalkers=8, nsteps=40)
    final_lp = sampler.get_log_prob(flat=True)[-1]
    assert final_lp > initial_lp
    chain = sampler.get_chain()
    assert chain.shape == (40, 8, 2)
