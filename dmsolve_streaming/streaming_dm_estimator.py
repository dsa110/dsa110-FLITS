#!/usr/bin/env python3
"""
Streaming Dispersion Measure (DM) Estimator

A streaming, O(1) memory algorithm for estimating DM from dynamic spectra
via intensity-weighted linear regression in dispersion coordinates.

Algorithm:
    In the coordinate x = ν⁻² - ν_ref⁻², the dispersion relation becomes linear:
        t = t₀ + K_DM · DM · x

    We perform intensity-weighted least squares regression of t on x,
    maintaining only 5 running sums (sufficient statistics).

IMPORTANT LIMITATIONS:
    - This is a CENTROID method, not matched filtering
    - Systematic negative bias at low S/N due to noise contamination
    - Bias scales as: Bias(DM) ≈ -f · DM, where f = noise_pixels / total_pixels
    - Recommended only for S/N > 15-20; use trial dedispersion for low S/N
    - Scattered (asymmetric) pulses will bias the estimate

Appropriate use cases:
    - Fast initial estimate to narrow DM search range
    - Real-time monitoring when S/N is known to be high
    - Memory-constrained environments (FPGA, embedded)

Author: FLITS project
"""

import numpy as np
from typing import Optional, NamedTuple


# Dispersion constant: delay = K_DM * DM * (ν^-2 - ν_ref^-2)
# Units: seconds when DM is in pc/cm^3 and frequency in MHz
K_DM = 4.148808e3  # s * MHz^2 / (pc cm^-3)


class DMEstimate(NamedTuple):
    """Result from DM estimation."""

    dm: float  # Dispersion measure in pc/cm^3
    t0: float  # Arrival time at reference frequency (seconds)
    n_pixels: int  # Number of pixels used in fit
    noise_sigma: float  # Estimated noise level
    signal_fraction: float  # Estimated fraction of pixels that are signal (diagnostic)


class StreamingDMEstimator:
    """
    Streaming DM estimator using intensity-weighted linear regression.

    Mathematical basis:
        The estimator computes the intensity-weighted covariance ratio:
            β̂₁ = Cov_w(x, t) / Var_w(x)

        where x = ν⁻² - ν_ref⁻² is the dispersion coordinate.

        This is equivalent to finding the slope of the intensity-weighted
        centroid locus in (x, t) space.

    Memory: O(1) - only 5 regression accumulators + 3 for noise estimation
    Time: O(N) single pass

    WARNING: This method has systematic bias at low S/N. The bias arises
    because noise pixels above threshold have Cov(x,t) ≈ 0, diluting the
    signal correlation. Expected bias: ~-f·DM where f is noise fraction.

    Usage:
        estimator = StreamingDMEstimator(freq_ref=1500.0, noise_sigma=1.0)

        for freq, times, intensities in data_stream:
            estimator.process_channel(freq, times, intensities)

        result = estimator.get_estimate()
        if result and result.signal_fraction > 0.5:  # Check quality
            print(f"DM = {result.dm:.2f} pc/cm³")
    """

    def __init__(
        self,
        freq_ref: float,
        noise_sigma: float = 1.0,
        sigma_threshold: float = 3.0,
        weight_cap: Optional[float] = None,
        estimate_noise_below_threshold: bool = True,
    ):
        """
        Initialize the streaming DM estimator.

        Args:
            freq_ref: Reference frequency in MHz (typically highest frequency)
            noise_sigma: Known or estimated noise standard deviation
            sigma_threshold: Detection threshold in units of sigma (default: 3)
            weight_cap: Maximum weight per pixel (None = no cap). Helps prevent
                        bright RFI or artifacts from dominating the fit.
            estimate_noise_below_threshold: If True, only update noise estimate
                        from pixels below threshold (avoids pulse contamination)
        """
        self.freq_ref = freq_ref
        self.noise_sigma = noise_sigma
        self.sigma_threshold = sigma_threshold
        self.weight_cap = weight_cap
        self.estimate_noise_below_threshold = estimate_noise_below_threshold

        self.reset()

    def reset(self):
        """Reset all accumulators for a new observation."""
        # Sufficient statistics for weighted linear regression
        # Model: t = β₀ + β₁·x, where x = ν⁻² - ν_ref⁻²
        self.S_X = 0.0  # Σ w·x
        self.S_Y = 0.0  # Σ w·t
        self.S_XX = 0.0  # Σ w·x²
        self.S_XY = 0.0  # Σ w·x·t
        self.W = 0.0  # Σ w
        self.n_pixels = 0
        self.n_total = 0  # Total pixels processed (for signal fraction)

        # Online noise estimation (Welford's algorithm)
        # Only updated for pixels BELOW threshold to avoid pulse contamination
        self.welford_n = 0
        self.welford_mean = 0.0
        self.welford_M2 = 0.0
        self._noise_sigma_estimate = self.noise_sigma

    def _update_noise_estimate(self, value: float):
        """Update running noise estimate using Welford's online algorithm."""
        self.welford_n += 1
        delta = value - self.welford_mean
        self.welford_mean += delta / self.welford_n
        delta2 = value - self.welford_mean
        self.welford_M2 += delta * delta2

        if self.welford_n >= 10:
            self._noise_sigma_estimate = np.sqrt(self.welford_M2 / self.welford_n)

    @property
    def threshold(self) -> float:
        """Current intensity threshold for pixel selection."""
        return self.sigma_threshold * self._noise_sigma_estimate

    def process_pixel(self, freq_mhz: float, time_sec: float, intensity: float):
        """
        Process a single pixel.

        Args:
            freq_mhz: Frequency in MHz
            time_sec: Time in seconds
            intensity: Pixel intensity (background-subtracted recommended)
        """
        self.n_total += 1

        # Update noise estimate from sub-threshold pixels only
        if self.estimate_noise_below_threshold:
            if intensity < self.threshold:
                self._update_noise_estimate(intensity)
        else:
            self._update_noise_estimate(intensity)

        # Only include pixels above threshold in regression
        if intensity < self.threshold:
            return

        # Dispersion coordinate
        x = freq_mhz**-2 - self.freq_ref**-2

        # Weight (with optional cap to prevent outlier domination)
        w = intensity
        if self.weight_cap is not None:
            w = min(w, self.weight_cap)

        # Accumulate sufficient statistics
        self.S_X += w * x
        self.S_Y += w * time_sec
        self.S_XX += w * x * x
        self.S_XY += w * x * time_sec
        self.W += w
        self.n_pixels += 1

    def process_channel(
        self,
        freq_mhz: float,
        time_samples: np.ndarray,
        intensities: np.ndarray,
    ):
        """
        Process one frequency channel.

        Args:
            freq_mhz: Frequency of this channel in MHz
            time_samples: Array of time values in seconds
            intensities: Array of intensity values
        """
        # Dispersion coordinate (constant for this channel)
        x = freq_mhz**-2 - self.freq_ref**-2
        threshold = self.threshold

        for t, I in zip(time_samples, intensities):
            self.n_total += 1

            # Noise estimation from sub-threshold pixels
            if self.estimate_noise_below_threshold:
                if I < threshold:
                    self._update_noise_estimate(I)
            else:
                self._update_noise_estimate(I)

            if I < threshold:
                continue

            # Weight with optional cap
            w = I if self.weight_cap is None else min(I, self.weight_cap)

            self.S_X += w * x
            self.S_Y += w * t
            self.S_XX += w * x * x
            self.S_XY += w * x * t
            self.W += w
            self.n_pixels += 1

    def process_spectrum(self, freqs: np.ndarray, times: np.ndarray, image: np.ndarray):
        """
        Process a full dynamic spectrum (convenience method).

        Args:
            freqs: Frequency array in MHz (n_chan,)
            times: Time array in seconds (n_time,)
            image: Dynamic spectrum (n_chan, n_time)
        """
        for i, freq in enumerate(freqs):
            self.process_channel(freq, times, image[i, :])

    def get_estimate(self) -> Optional[DMEstimate]:
        """
        Compute DM estimate from accumulated statistics.

        The solution is:
            β̂₁ = (W·S_XY - S_X·S_Y) / (W·S_XX - S_X²)
            DM = β̂₁ / K_DM

        Returns:
            DMEstimate with DM, t0, diagnostics, or None if insufficient data
        """
        if self.W < 1e-10 or self.n_pixels < 2:
            return None

        # Determinant of normal equations matrix
        det = self.W * self.S_XX - self.S_X**2
        if abs(det) < 1e-30 * self.W**2:  # Scaled threshold for numerical stability
            return None

        # Weighted least squares solution
        slope = (self.W * self.S_XY - self.S_X * self.S_Y) / det
        intercept = (self.S_Y - slope * self.S_X) / self.W

        # Convert to physical parameters
        dm = slope / K_DM
        t0 = intercept

        # Signal fraction diagnostic (rough estimate)
        # Higher is better; low values indicate noise domination
        signal_fraction = self.n_pixels / max(self.n_total, 1)

        return DMEstimate(
            dm=dm,
            t0=t0,
            n_pixels=self.n_pixels,
            noise_sigma=self._noise_sigma_estimate,
            signal_fraction=signal_fraction,
        )


# =============================================================================
# Utility functions
# =============================================================================


def dispersion_delay(
    dm: float, freq_mhz: np.ndarray, freq_ref_mhz: float
) -> np.ndarray:
    """Compute dispersion delay in seconds."""
    return K_DM * dm * (freq_mhz**-2 - freq_ref_mhz**-2)


def generate_test_spectrum(
    dm: float,
    t0: float,
    width: float,
    snr: float,
    freqs: np.ndarray,
    times: np.ndarray,
    noise_level: float = 1.0,
    scattering_time: float = 0.0,
) -> np.ndarray:
    """
    Generate synthetic dynamic spectrum with dispersed pulse.

    Args:
        dm: Dispersion measure (pc/cm³)
        t0: Arrival time at reference frequency (s)
        width: Intrinsic pulse width (s)
        snr: Peak signal-to-noise ratio
        freqs: Frequency array (MHz)
        times: Time array (s)
        noise_level: Noise standard deviation
        scattering_time: Scattering timescale at ref freq (s), optional

    Returns:
        Dynamic spectrum (n_chan, n_time)
    """
    freq_ref = freqs.max()
    delays = dispersion_delay(dm, freqs, freq_ref)
    arrival_times = t0 + delays

    t_grid, arr_grid = np.meshgrid(times, arrival_times)
    signal = snr * noise_level * np.exp(-0.5 * ((t_grid - arr_grid) / width) ** 2)

    # Optional scattering (asymmetric exponential tail)
    if scattering_time > 0:
        dt = times[1] - times[0]
        for i, freq in enumerate(freqs):
            tau = scattering_time * (freq / freq_ref) ** -4
            if tau > dt:
                kernel_len = min(int(5 * tau / dt), len(times) // 2)
                kernel = np.exp(-np.arange(kernel_len) * dt / tau)
                kernel /= kernel.sum()
                signal[i] = np.convolve(signal[i], kernel, mode="same")

    noise = noise_level * np.random.randn(len(freqs), len(times))
    return signal + noise


# =============================================================================
# Demo with honest performance reporting
# =============================================================================

if __name__ == "__main__":
    import time as time_module

    print("=" * 70)
    print("Streaming DM Estimator - Performance Characterization")
    print("=" * 70)

    np.random.seed(42)

    # Setup - reduced size for fast testing
    freq_lo, freq_hi = 1100, 1500
    n_chan, n_time = 64, 256  # Smaller for speed
    freqs = np.linspace(freq_hi, freq_lo, n_chan)
    times = np.linspace(0, 0.5, n_time)
    dm_true = 300.0
    t0_true = 0.05
    width = 0.003

    print(f"\nTest configuration:")
    print(f"  True DM = {dm_true} pc/cm³")
    print(f"  Channels: {n_chan}, Time samples: {n_time}")

    # S/N sweep with honest reporting
    print("\n" + "-" * 70)
    print("S/N SWEEP: Demonstrating low-S/N bias")
    print("-" * 70)
    print(
        f"{'S/N':>6} {'Mean Est.':>10} {'Bias':>10} {'Bias %':>10} {'Std Dev':>10} {'Sig.Frac':>10}"
    )
    print("-" * 70)

    for snr in [3, 5, 10, 20, 50]:
        estimates = []
        sig_fracs = []

        for trial in range(5):  # Reduced for speed
            ds = generate_test_spectrum(dm_true, t0_true, width, snr, freqs, times)

            est = StreamingDMEstimator(
                freq_ref=freq_hi,
                noise_sigma=1.0,
                sigma_threshold=3.0,
                estimate_noise_below_threshold=True,
            )
            est.process_spectrum(freqs, times, ds)
            result = est.get_estimate()

            if result:
                estimates.append(result.dm)
                sig_fracs.append(result.signal_fraction)

        if estimates:
            mean_dm = np.mean(estimates)
            bias = mean_dm - dm_true
            std = np.std(estimates)
            mean_sf = np.mean(sig_fracs)

            # Flag problematic regimes
            flag = " ⚠️" if abs(bias) > 0.1 * dm_true else ""

            print(
                f"{snr:>6} {mean_dm:>10.1f} {bias:>+10.1f} {100*bias/dm_true:>+10.1f}% "
                f"{std:>10.1f} {mean_sf:>10.3f}{flag}"
            )

    print("-" * 70)
    print("⚠️  = Bias exceeds 10% — use trial dedispersion instead")

    # Scattering test
    print("\n" + "-" * 70)
    print("SCATTERING TEST: Demonstrating asymmetric pulse bias (S/N=20)")
    print("-" * 70)
    print(f"{'τ_scat (ms)':>12} {'Mean Est.':>10} {'Bias':>10} {'Bias %':>10}")
    print("-" * 70)

    for tau_ms in [0, 5, 10, 20]:
        estimates = []
        tau_sec = tau_ms / 1000

        for trial in range(5):  # Reduced for speed
            ds = generate_test_spectrum(
                dm_true, t0_true, width, 20, freqs, times, scattering_time=tau_sec
            )

            est = StreamingDMEstimator(
                freq_ref=freq_hi, noise_sigma=1.0, sigma_threshold=3.0
            )
            est.process_spectrum(freqs, times, ds)
            result = est.get_estimate()

            if result:
                estimates.append(result.dm)

        if estimates:
            mean_dm = np.mean(estimates)
            bias = mean_dm - dm_true
            print(
                f"{tau_ms:>12} {mean_dm:>10.1f} {bias:>+10.1f} {100*bias/dm_true:>+10.1f}%"
            )

    # Speed test
    print("\n" + "-" * 70)
    print("SPEED TEST")
    print("-" * 70)

    ds = generate_test_spectrum(dm_true, t0_true, width, 20, freqs, times)

    t_start = time_module.time()
    n_reps = 10  # Reduced for speed
    for _ in range(n_reps):
        est = StreamingDMEstimator(
            freq_ref=freq_hi, noise_sigma=1.0, sigma_threshold=3.0
        )
        est.process_spectrum(freqs, times, ds)
        _ = est.get_estimate()
    t_elapsed = (time_module.time() - t_start) / n_reps

    print(f"  Time per spectrum: {t_elapsed*1e3:.2f} ms")
    print(f"  Throughput: {n_chan * n_time / t_elapsed / 1e6:.1f} Mpixels/sec")
    print(f"  Memory: O(1) — 8 floating-point accumulators")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        """
This estimator is a FAST CENTROID METHOD, not matched filtering.

✓ Use when:
  - S/N > 15-20 (bias < 5%)
  - Need quick initial estimate for search range
  - Memory constrained (FPGA, real-time)
  - Pulses are roughly symmetric (minimal scattering)

✗ Avoid when:
  - S/N < 10 (systematic underestimation)
  - Precise DM measurement needed
  - Significant scattering expected
  - Detection near threshold

For low S/N or precise work, use trial dedispersion (bowtie method).
"""
    )
