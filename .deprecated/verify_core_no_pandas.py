import sys
import numpy as np

# Verify core modules that don't depend on pandas
from flits.fitting import VALIDATION_THRESHOLDS as VT
from flits.fitting import diagnostics
from flits.sampler import FRBFitter, _log_prob_wrapper

def test_imports():
    print("Testing core imports...")
    print(f"Loaded VALIDATION_THRESHOLDS: DM_MAX={VT.DM_MAX}")
    print(f"Loaded diagnostics: {diagnostics.__file__}")
    print(f"Loaded FRBFitter from: {sys.modules['flits.sampler'].__file__}")
    print("Core imports successful.")

def test_diagnostics():
    print("\nTesting diagnostics...")
    data = np.random.normal(0, 1, 100)
    model = np.zeros(100)
    # This uses scipy.stats and matplotlib, hopefully they don't hang
    diag = diagnostics.analyze_residuals(data, model)
    print(f"Diagnostics result: {diag.quality_flag}")
    print(diag)

def test_sampler_bounds():
    print("\nTesting sampler bounds...")
    # Test valid params
    params = np.array([100.0, 10.0])
    lp = _log_prob_wrapper(params, np.array([]), np.array([]), np.array([]), 1.0)
    print(f"Valid params log-prob (should be finite if data matches): {lp}")
    # Note: it will return -inf because data is empty so resid calculation fails or model.simulate fails?
    # actually _log_prob_wrapper will try to simulate.
    # checking logic:
    # 1. bounds check (pass)
    # 2. FRBModel(params) (pass)
    # 3. model.simulate(t, freqs) -> will likely fail with empty arrays or return empty
    # let's use dummy t and freqs
    t = np.linspace(0, 10, 100)
    freqs = np.linspace(1000, 1500, 100)
    data = np.zeros((100, 100))
    
    lp = _log_prob_wrapper(params, t, freqs, data, 1.0)
    print(f"Valid params log-prob: {lp}")

    # Test invalid params
    params_bad = np.array([-1.0, 10.0])
    lp_bad = _log_prob_wrapper(params_bad, t, freqs, data, 1.0)
    print(f"Invalid params log-prob (should be -inf): {lp_bad}")
    assert lp_bad == -np.inf

if __name__ == "__main__":
    test_imports()
    test_diagnostics()
    test_sampler_bounds()
    print("\nCore verification tests passed!")
