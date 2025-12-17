import sys
import numpy as np
import pandas as pd
from flits.fitting import VALIDATION_THRESHOLDS as VT
from flits.fitting import diagnostics
from flits.sampler import FRBFitter, _log_prob_wrapper
from flits.batch.analysis_logic import _validate_measurement, check_tau_deltanu_consistency

def test_imports():
    print("Testing imports...")
    print(f"Loaded VALIDATION_THRESHOLDS: DM_MAX={VT.DM_MAX}")
    print(f"Loaded diagnostics: {diagnostics.__file__}")
    print(f"Loaded FRBFitter from: {sys.modules['flits.sampler'].__file__}")
    print("Imports successful.")

def test_diagnostics():
    print("\nTesting diagnostics...")
    data = np.random.normal(0, 1, 100)
    model = np.zeros(100)
    diag = diagnostics.analyze_residuals(data, model)
    print(f"Diagnostics result: {diag.quality_flag}")
    print(diag)

def test_sampler_bounds():
    print("\nTesting sampler bounds...")
    # Test valid params
    lp = _log_prob_wrapper([100, 10], None, None, np.zeros(10), 1.0)
    print(f"Valid params log-prob (should be finite if data matches): {lp}")
    
    # Test invalid params
    lp_bad = _log_prob_wrapper([-1, 10], None, None, np.zeros(10), 1.0)
    print(f"Invalid params log-prob (should be -inf): {lp_bad}")
    assert lp_bad == -np.inf

def test_batch_validation():
    print("\nTesting batch validation...")
    
    # Test helper
    valid, msg = _validate_measurement(10.0, 1.0, "test")
    print(f"Valid measurement (10+/-1): {valid}, {msg}")
    assert valid
    
    valid, msg = _validate_measurement(10.0, 6.0, "test", rel_err_threshold=0.5)
    print(f"Invalid measurement (10+/-6, thresh=0.5): {valid}, {msg}")
    assert not valid

if __name__ == "__main__":
    test_imports()
    test_diagnostics()
    test_sampler_bounds()
    test_batch_validation()
    print("\nAll verification tests passed!")
