import numpy as np
import matplotlib.pyplot as plt
from flits.models import FRBModel
from flits.params import FRBParams


def main():
    # 1. Define parameters
    # A moderately dispersed and scattered burst
    params = FRBParams(
        dm=50.0,  # pc/cm^3
        width=2.0,  # ms (intrinsic width)
        amplitude=1.0,
        t0=20.0,  # ms (arrival time at infinite frequency)
        tau_1ghz=5.0,  # ms (scattering timescale at 1 GHz)
        tau_alpha=4.0,  # Scattering index (thin screen)
    )

    # 2. Define grid
    # Frequencies: 1200 MHz to 1500 MHz (L-band ish)
    freqs = np.linspace(1200, 1500, 256)

    # Time: 100 to 200 ms (adjusted for dispersion delay)
    t = np.linspace(100, 200, 1024)

    # 3. Simulate
    model = FRBModel(params)
    # We simulate with the scattering parameters defined in params
    dynspec = model.simulate(t, freqs)

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
        f"Scattered FRB Simulation\nDM={params.dm}, Width={params.width}ms, $\\tau_{{1GHz}}$={params.tau_1ghz}ms"
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
