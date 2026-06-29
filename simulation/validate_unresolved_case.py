#!/usr/bin/env python3
# ============================================================================
# validate_unresolved_case.py
#
# This script serves as a validation test for the frb_scintillator module.
# It simulates the "unresolved" two-screen case from Pradeep et al. (2025)
# and verifies that the total modulation index squared (the peak of the ACF)
# approaches the theoretical value of 3, as predicted by Eq. 4.26.
#
# IMPORTANT: m_total^2 -> 3 is the RP -> 0 limit, where the two screens stop
# resolving each other. The peak ACF tracks Pradeep's resolution curve
# (m^2 ~ 3 at RP << 1, ~1.8 at RP ~ 0.2, -> 1 at RP >> 1), so this test must
# use a genuinely unresolved geometry (RP ~ 0.05 here) and average over screen
# realisations -- a single realisation scatters by ~+/-0.3. A residual ~5%
# deficit below 3 is the finite-N / finite-scintle bias of the var/mean^2
# estimator, not a physics error (the cross-term-free separable field gives 3.0).
#
# Usage:
#   python validate_unresolved_case.py
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import sys

# Add the parent directory to the path to find the frb_scintillator module
# This allows the script to be run from within the validation/ directory
sys.path.append('..')
from screen import ScreenCfg
from engine import SimCfg, FRBScintillator
from scintillation.scint_analysis.analysis import interpret_modulation_index

def run_validation(n_trials: int = 12):
    """
    Configures and runs the unresolved two-screen case (RP << 1), averages the
    peak ACF over screen realisations, and checks it against m_total^2 -> 3.
    """

    # Unresolved regime requires RP << 1. Small screens (L_mw=1.75 AU, L_host=10
    # AU) at this geometry give RP ~ 0.05; bw=50 MHz / 8192 chan samples both the
    # broad (MW) and narrow (host) scintles well enough to recover m^2 ~ 3.
    base = dict(
        peak_flux=5 * u.Jy,
        nu0=800 * u.MHz,
        bw=50.0 * u.MHz,
        nchan=8192,
        z_host=0.192,
        D_mw=2.3 * u.kpc,
        D_host_src=2.0 * u.kpc,
        intrinsic_pulse="delta",  # delta pulse avoids self-noise
    )

    print("--- Running Unresolved Regime Validation ---")
    peaks, corr, lags, rp = [], None, None, None
    for i in range(n_trials):
        cfg = SimCfg(
            **base,
            mw=ScreenCfg(N=200, L=1.75 * u.AU, rng_seed=1234 + i),
            host=ScreenCfg(N=200, L=10.0 * u.AU, rng_seed=5678 + i),
        )
        sim = FRBScintillator(cfg)
        rp = sim.resolution_power()
        corr, lags = sim.acf(sim.simulate_time_integrated_spectrum())
        peaks.append(corr[0])

    peaks = np.array(peaks)
    peak_mean, peak_sem = peaks.mean(), peaks.std() / np.sqrt(n_trials)

    # --- Verification ---
    print(f"Resolution Power (RP) = {rp:.3f}  (RP << 1 => unresolved)")
    print("\n" + "=" * 52)
    print("Expected peak ACF (m_total^2) for two unresolved screens: 3.0")
    print(f"Simulated peak ACF over {n_trials} realisations: "
          f"{peak_mean:.3f} +/- {peak_sem:.3f}")

    if abs(peak_mean - 3.0) < 0.5:
        print("✅ SUCCESS: consistent with the analytic two-screen value (m^2 = 3).")
    else:
        print("❌ FAILED: deviates from the unresolved two-screen prediction.")

    # Route the measured m through the canonical Pradeep interpreter (m^2 = 2^N - 1)
    # so the simulator and the pipeline report screen counts identically.
    m_total = float(np.sqrt(max(peak_mean, 0.0)))
    interp = interpret_modulation_index(m_total)
    print(f"m_total = {m_total:.3f} -> regime '{interp['resolution_regime']}', "
          f"N_screens ~ {interp['n_screens_est']:.1f}")
    print("=" * 52 + "\n")

    # --- Visualization (last realisation, representative) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    lags_khz = lags * sim.dnu_hz / 1e3

    ax.plot(lags_khz, corr, 'k-', lw=1.5,
            label=f'Simulated ACF (mean peak = {peak_mean:.3f})')
    ax.axhline(3.0, color='r', ls='--', label='Two unresolved screens (m²=3)')

    ax.set_xlabel("Frequency Lag (kHz)", fontsize=12)
    ax.set_ylabel("Normalized Correlation", fontsize=12)
    ax.set_title("Validation of Unresolved Case ACF", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.set_xlim(left=0, right=lags_khz[len(lags_khz)//10])

    plt.tight_layout()
    plt.savefig("validation_unresolved_case.png", dpi=150)
    # plt.show()

if __name__ == '__main__':
    run_validation()
