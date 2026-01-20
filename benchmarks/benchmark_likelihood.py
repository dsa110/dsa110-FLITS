
import time
import numpy as np
import logging
from scattering.scat_analysis.burstfit import FRBModel, FRBParams

def benchmark_likelihood(n_freq=128, n_time=1024, n_iter=100):
    print(f"Benchmarking Likelihood Evaluation")
    print(f"Dimensions: {n_freq} freq x {n_time} time")
    print(f"Iterations: {n_iter}")
    print("-" * 40)

    # Setup
    t = np.linspace(0, 100, n_time)
    f = np.linspace(1.2, 1.5, n_freq)
    
    # Create dummy data
    true_params = FRBParams(c0=10.0, t0=50.0, gamma=-1.5, zeta=2.0, tau_1ghz=5.0)
    model = FRBModel(t, f, dm_init=50.0)
    
    # Generate data
    clean_signal = model(true_params, "M3")
    noise = np.random.normal(0, 1.0, size=clean_signal.shape)
    data = clean_signal + noise
    
    # Update model with data
    model.data = data
    model.noise_std = np.ones((n_freq,)) * 1.0  # 1D noise
    model.valid = np.ones((n_freq,), dtype=bool) # 1D valid mask
    
    # Benchmark M1 (Gaussian)
    start = time.perf_counter()
    for _ in range(n_iter):
        model.log_likelihood(true_params, "M1")
    end = time.perf_counter()
    dur_m1 = end - start
    rate_m1 = n_iter / dur_m1
    print(f"M1 (Gaussian): {dur_m1:.4f} s total, {dur_m1/n_iter*1000:.4f} ms/call ({rate_m1:.1f} calls/s)")
    
    # Benchmark M3 (Scattering)
    start = time.perf_counter()
    for _ in range(n_iter):
        model.log_likelihood(true_params, "M3")
    end = time.perf_counter()
    dur_m3 = end - start
    rate_m3 = n_iter / dur_m3
    print(f"M3 (Scattering): {dur_m3:.4f} s total, {dur_m3/n_iter*1000:.4f} ms/call ({rate_m3:.1f} calls/s)")

if __name__ == "__main__":
    benchmark_likelihood()
