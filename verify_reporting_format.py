
from flits.utils.reporting import print_fit_summary

def test_reporting():
    print("Testing Fit Reporting Format...")
    
    # Mock results dictionary
    results = {
        "best_key": "M3",
        "goodness_of_fit": {
            "chi2_reduced": 1.15,
            "r_squared": 0.98,
            "quality_flag": "PASS"
        },
        "param_names": ["tau_1ghz", "t0", "amp", "alpha"],
        "best_params": {
            "tau_1ghz": 5.432,
            "t0": 10.1,
            "amp": 100.0,
            "alpha": 4.4
        },
        "flat_chain": None # Test handling of missing chain
    }
    
    print("\n[TEST 1] Standard Results (No Chain):")
    print_fit_summary(results)
    
    # Test with chain stats
    import numpy as np
    # Mock chain: 100 samples, 4 params.
    # tau=5.4 +/- 0.1
    chain = np.zeros((100, 4))
    chain[:, 0] = np.random.normal(5.432, 0.1, 100)
    chain[:, 1] = np.random.normal(10.1, 0.05, 100)
    chain[:, 2] = np.random.normal(100.0, 5.0, 100)
    chain[:, 3] = np.random.normal(4.4, 0.2, 100)
    
    results["flat_chain"] = chain
    
    print("\n[TEST 2] Results With Chain Statistics:")
    print_fit_summary(results)

if __name__ == "__main__":
    test_reporting()
