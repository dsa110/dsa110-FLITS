#!/usr/bin/env python3
"""
Streaming Dispersion Measure (DM) Estimator

A fully streaming, O(1) memory algorithm for estimating DM from dynamic spectra.
Processes data as it arrives without storing the full image.

Algorithm: Weighted linear regression in dispersion space
    t = t0 + K_DM * DM * (ν^-2 - ν_ref^-2)

Author: Generated for FLITS project
"""

import numpy as np
from typing import Optional, Tuple, NamedTuple


# Dispersion constant: delay = K_DM * DM * (ν^-2 - ν_ref^-2)
# Units: seconds when DM is in pc/cm^3 and frequency in MHz
K_DM = 4.148808e3  # s * MHz^2 / (pc cm^-3)


class DMEstimate(NamedTuple):
    """Result from DM estimation."""
    dm: float              # Dispersion measure in pc/cm^3
    t0: float              # Arrival time at reference frequency in seconds
    n_pixels: int          # Number of pixels used in fit
    noise_sigma: float     # Estimated noise level


class StreamingDMEstimator:
    """
    Fully streaming DM estimator using weighted linear regression.
    
    Memory: O(1) - only maintains running sums
    Complexity: O(n) single pass through data
    
    Usage:
        estimator = StreamingDMEstimator(freq_ref=1500.0, noise_sigma=1.0)
        
        # Process data as it streams in
        for freq, times, intensities in data_stream:
            estimator.process_channel(freq, times, intensities)
        
        # Get result
        result = estimator.get_estimate()
        print(f"DM = {result.dm:.2f} pc/cm³")
    
    The algorithm fits: t = t0 + K_DM * DM * (ν^-2 - ν_ref^-2)
    which is linear in the dispersion coordinate x = ν^-2 - ν_ref^-2
    """
    
    def __init__(
        self,
        freq_ref: float,
        noise_sigma: float = 1.0,
        sigma_threshold: float = 3.0,
        adaptive_threshold: bool = False,
    ):
        """
        Initialize the streaming DM estimator.
        
        Args:
            freq_ref: Reference frequency in MHz (typically highest frequency)
            noise_sigma: Noise standard deviation (can be estimated if adaptive_threshold=True)
            sigma_threshold: Detection threshold in units of sigma
            adaptive_threshold: If True, estimate noise online using Welford's algorithm
        """
        self.freq_ref = freq_ref
        self.initial_noise_sigma = noise_sigma
        self.sigma_threshold = sigma_threshold
        self.adaptive_threshold = adaptive_threshold
        
        self.reset()
    
    def reset(self):
        """Reset all accumulators for a new observation."""
        # Running sums for weighted linear regression
        # Fit model: t = intercept + slope * x, where x = ν^-2 - ν_ref^-2
        self.S_X = 0.0    # Σ w * x
        self.S_Y = 0.0    # Σ w * t
        self.S_XX = 0.0   # Σ w * x²
        self.S_XY = 0.0   # Σ w * x * t
        self.W = 0.0      # Σ w (total weight)
        self.n_pixels = 0
        
        # Online noise estimation (Welford's algorithm)
        self.n_samples = 0
        self.welford_mean = 0.0
        self.welford_M2 = 0.0
    
    def _update_noise_estimate(self, value: float):
        """Update running noise estimate using Welford's online algorithm."""
        self.n_samples += 1
        delta = value - self.welford_mean
        self.welford_mean += delta / self.n_samples
        delta2 = value - self.welford_mean
        self.welford_M2 += delta * delta2
    
    @property
    def noise_sigma(self) -> float:
        """Current noise estimate."""
        if self.adaptive_threshold and self.n_samples >= 2:
            return np.sqrt(self.welford_M2 / self.n_samples)
        return self.initial_noise_sigma
    
    @property
    def threshold(self) -> float:
        """Current intensity threshold for pixel selection."""
        return self.sigma_threshold * self.noise_sigma
    
    def process_pixel(self, freq_mhz: float, time_sec: float, intensity: float):
        """
        Process a single pixel. Maximum streaming granularity.
        
        Args:
            freq_mhz: Frequency in MHz
            time_sec: Time in seconds
            intensity: Pixel intensity
        """
        # Update noise estimate
        if self.adaptive_threshold:
            self._update_noise_estimate(intensity)
        
        # Only include pixels above threshold
        if intensity < self.threshold:
            return
        
        # Dispersion coordinate
        x = freq_mhz**-2 - self.freq_ref**-2
        
        # Accumulate weighted sums (intensity-weighted regression)
        self.S_X += intensity * x
        self.S_Y += intensity * time_sec
        self.S_XX += intensity * x * x
        self.S_XY += intensity * x * time_sec
        self.W += intensity
        self.n_pixels += 1
    
    def process_channel(
        self,
        freq_mhz: float,
        time_samples: np.ndarray,
        intensities: np.ndarray,
    ):
        """
        Process one frequency channel. Call as data streams from spectrometer.
        
        Args:
            freq_mhz: Frequency of this channel in MHz
            time_samples: Array of time values in seconds
            intensities: Array of intensity values for this channel
        """
        # Dispersion coordinate for this channel (constant across time samples)
        x = freq_mhz**-2 - self.freq_ref**-2
        
        for t, I in zip(time_samples, intensities):
            # Update noise estimate
            if self.adaptive_threshold:
                self._update_noise_estimate(I)
            
            # Only include bright pixels
            if I < self.threshold:
                continue
            
            self.S_X += I * x
            self.S_Y += I * t
            self.S_XX += I * x * x
            self.S_XY += I * x * t
            self.W += I
            self.n_pixels += 1
    
    def process_spectrum(self, freqs: np.ndarray, times: np.ndarray, image: np.ndarray):
        """
        Process a full dynamic spectrum. Convenience method for batch processing.
        
        Args:
            freqs: Frequency array in MHz (n_chan,)
            times: Time array in seconds (n_time,)
            image: Dynamic spectrum (n_chan, n_time)
        """
        for i, freq in enumerate(freqs):
            self.process_channel(freq, times, image[i, :])
    
    def get_estimate(self) -> Optional[DMEstimate]:
        """
        Get current DM estimate from accumulated statistics.
        
        Can be called at any time to get intermediate estimates.
        
        Returns:
            DMEstimate or None if insufficient data
        """
        if self.W < 1e-10:
            return None
        
        det = self.W * self.S_XX - self.S_X**2
        if abs(det) < 1e-20:
            return None
        
        # Solve weighted linear regression
        slope = (self.W * self.S_XY - self.S_X * self.S_Y) / det
        intercept = (self.S_Y - slope * self.S_X) / self.W
        
        # Convert to physical parameters
        dm = slope / K_DM
        t0 = intercept
        
        return DMEstimate(
            dm=dm,
            t0=t0,
            n_pixels=self.n_pixels,
            noise_sigma=self.noise_sigma,
        )


def dispersion_delay(dm: float, freq_mhz: np.ndarray, freq_ref_mhz: float) -> np.ndarray:
    """
    Compute dispersion delay in seconds.
    
    Args:
        dm: Dispersion measure in pc/cm^3
        freq_mhz: Frequency array in MHz
        freq_ref_mhz: Reference frequency in MHz
    
    Returns:
        Time delay in seconds (positive = arrives later)
    """
    return K_DM * dm * (freq_mhz**-2 - freq_ref_mhz**-2)


def generate_test_spectrum(
    dm: float,
    t0: float,
    width: float,
    amplitude: float,
    freqs: np.ndarray,
    times: np.ndarray,
    noise_level: float = 1.0,
) -> np.ndarray:
    """
    Generate a synthetic dynamic spectrum with a dispersed pulse.
    
    Args:
        dm: Dispersion measure in pc/cm^3
        t0: Arrival time at reference frequency in seconds
        width: Pulse width in seconds
        amplitude: Peak pulse amplitude (S/N in units of noise)
        freqs: Frequency array in MHz
        times: Time array in seconds
        noise_level: Noise standard deviation
    
    Returns:
        Dynamic spectrum array (n_chan, n_time)
    """
    freq_ref = freqs.max()
    delays = dispersion_delay(dm, freqs, freq_ref)
    arrival_times = t0 + delays
    
    t_grid, arr_grid = np.meshgrid(times, arrival_times)
    signal = amplitude * np.exp(-0.5 * ((t_grid - arr_grid) / width) ** 2)
    noise = noise_level * np.random.randn(len(freqs), len(times))
    
    return signal + noise


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    import time as time_module
    
    print("=" * 60)
    print("Streaming DM Estimator - Demo")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Observation parameters
    freq_lo, freq_hi = 1100, 1500  # MHz
    n_chan = 256
    n_time = 1024
    freqs = np.linspace(freq_hi, freq_lo, n_chan)
    times = np.linspace(0, 1.0, n_time)  # 1 second observation
    
    # Pulse parameters
    dm_true = 300.0    # pc/cm^3
    t0_true = 0.05     # 50 ms
    width = 0.003      # 3 ms
    snr = 20           # Signal-to-noise ratio
    
    print(f"\nTrue parameters:")
    print(f"  DM = {dm_true} pc/cm³")
    print(f"  t0 = {t0_true * 1e3:.1f} ms")
    print(f"  S/N = {snr}")
    
    # Generate test data
    print(f"\nGenerating {n_chan} × {n_time} dynamic spectrum...")
    ds = generate_test_spectrum(dm_true, t0_true, width, snr, freqs, times)
    
    # Initialize estimator
    estimator = StreamingDMEstimator(
        freq_ref=freq_hi,
        noise_sigma=1.0,
        sigma_threshold=3.0,
        adaptive_threshold=True,
    )
    
    # Process streaming (channel by channel)
    print("\nStreaming processing (channel by channel):")
    t_start = time_module.time()
    
    for i, freq in enumerate(freqs):
        estimator.process_channel(freq, times, ds[i, :])
        
        # Print intermediate estimates
        if (i + 1) % 64 == 0:
            result = estimator.get_estimate()
            if result:
                print(f"  After {i+1:3d}/{n_chan} channels: "
                      f"DM = {result.dm:6.1f} pc/cm³ "
                      f"(n={result.n_pixels:4d} pixels)")
    
    t_elapsed = time_module.time() - t_start
    
    # Final result
    result = estimator.get_estimate()
    print(f"\n" + "=" * 60)
    print("Final Results:")
    print("=" * 60)
    print(f"  Estimated DM: {result.dm:.2f} pc/cm³")
    print(f"  Estimated t0: {result.t0 * 1e3:.2f} ms")
    print(f"  DM Error: {result.dm - dm_true:.2f} pc/cm³ "
          f"({100 * (result.dm - dm_true) / dm_true:.1f}%)")
    print(f"  Pixels used: {result.n_pixels:,} / {n_chan * n_time:,}")
    print(f"  Noise σ estimate: {result.noise_sigma:.3f}")
    print(f"  Processing time: {t_elapsed * 1e3:.1f} ms")
    print(f"\nMemory: O(1) - only 8 floating point accumulators")
    
    # S/N sweep test
    print("\n" + "=" * 60)
    print("S/N Sweep Test:")
    print("=" * 60)
    print(f"{'S/N':>6} {'Est. DM':>10} {'Error':>10} {'Error %':>10}")
    print("-" * 40)
    
    for snr_test in [3, 5, 10, 20, 50]:
        errors = []
        for trial in range(20):
            ds_test = generate_test_spectrum(dm_true, t0_true, width, snr_test, freqs, times)
            
            est = StreamingDMEstimator(freq_ref=freq_hi, noise_sigma=1.0, sigma_threshold=3.0)
            est.process_spectrum(freqs, times, ds_test)
            result = est.get_estimate()
            
            if result:
                errors.append(result.dm - dm_true)
        
        if errors:
            mean_err = np.mean(errors)
            print(f"{snr_test:>6} {dm_true + mean_err:>10.1f} "
                  f"{mean_err:>+10.1f} {100 * mean_err / dm_true:>+10.1f}%")
    
    print("\nDone!")

