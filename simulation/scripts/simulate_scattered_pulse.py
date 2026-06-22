import matplotlib.pyplot as plt
import numpy as np

from scattering.scat_analysis.burstfit import DM_DELAY_MS, FRBModel, FRBParams


def main():
    # 1. Define parameters
    # A moderately dispersed and scattered burst
    dm = 50.0  # pc/cm^3
    width = 2.0  # ms (intrinsic sigma)
    amplitude = 1.0
    t0 = 20.0  # ms (arrival time at infinite frequency)
    tau_1ghz = 5.0  # ms (scattering timescale at 1 GHz)
    tau_alpha = 4.0  # Scattering index (thin screen)

    # 2. Define grid
    # Frequencies: 1200 MHz to 1500 MHz (L-band ish)
    freqs = np.linspace(1200, 1500, 256)

    # Time: 100 to 200 ms (adjusted for dispersion delay)
    t = np.linspace(100, 200, 1024)

    # 3. Simulate with the core kernel. Core references the dispersion delay to
    # f_max; t0 here is the legacy "arrival at infinite frequency", so shift by
    # the delay at the top of the band to keep the same absolute arrival time.
    freqs_ghz = freqs / 1000.0
    df_MHz = abs(freqs[1] - freqs[0])
    p = FRBParams(
        c0=amplitude,
        t0=t0 + DM_DELAY_MS * dm / freqs_ghz.max() ** 2,
        gamma=0.0,
        zeta=width,
        tau_1ghz=tau_1ghz,
        alpha=tau_alpha,
        delta_dm=dm,
    )
    model = FRBModel(time=t, freq=freqs_ghz, dm_init=dm, df_MHz=df_MHz)
    dynspec = model(p, "M3")

    # 4. Calculate time series (frequency-averaged profile)
    time_series = dynspec.mean(axis=0)

    # 5. Plot
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [1, 3]}
    )

    # Top panel: Time Series
    ax1.plot(t, time_series, color="black", lw=1.5)
    ax1.set_ylabel("Intensity (arb)")
    ax1.set_title(
        f"Scattered FRB Simulation\nDM={dm}, Width={width}ms, $\\tau_{{1GHz}}$={tau_1ghz}ms"
    )
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Dynamic Spectrum
    # Use extent to map array indices to physical units
    extent = [t[0], t[-1], freqs[0], freqs[-1]]
    im = ax2.imshow(
        dynspec,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
        interpolation="nearest",
    )
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Frequency (MHz)")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax2, pad=0.02)
    cbar.set_label("Intensity")

    plt.tight_layout()
    output_filename = "scattered_pulse_simulation.png"
    plt.savefig(output_filename, dpi=150)
    print(f"Simulation complete. Plot saved to {output_filename}")


if __name__ == "__main__":
    main()
