import numpy as np
import matplotlib.pyplot as plt
from flits.models import FRBModel, K_DM
from flits.params import FRBParams
from flits.scattering import tau_per_freq


def simulate_and_dedisperse(params, t, freqs):
    """Simulate FRB and dedisperse it."""
    # Simulate with scattering
    model = FRBModel(params)
    dynspec = model.simulate(t, freqs)

    # Dedisperse: shift each frequency channel to align at t0
    delays = K_DM * params.dm / freqs**2
    dynspec_dedispersed = np.zeros_like(dynspec)

    dt = t[1] - t[0]
    for i, delay in enumerate(delays):
        # Compute shift in samples
        shift_samples = int(np.round(delay / dt))

        # Roll to remove dispersion delay
        dynspec_dedispersed[i, :] = np.roll(dynspec[i, :], -shift_samples)

        # Zero out wrapped portion to avoid edge artifacts
        if shift_samples > 0:
            dynspec_dedispersed[i, -shift_samples:] = 0

    return dynspec_dedispersed


def plot_results(t, freqs, time_series, dynspec_dedispersed, params):
    """Plot the dedispersed pulse and dynamic spectrum."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [1, 3]}
    )

    # Top panel: Dedispersed Time Series
    ax1.plot(t, time_series, color="black", lw=1.5)
    ax1.set_ylabel("Intensity (arb)")
    ax1.set_title(
        f"Dedispersed Scattered FRB\nDM={params.dm}, Width={params.width}ms, $\\tau_{{1GHz}}$={params.tau_1ghz}ms, α={params.tau_alpha}"
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(10, 60)  # Focus on pulse region

    # Bottom panel: Dedispersed Dynamic Spectrum
    extent = [t[0], t[-1], freqs[0], freqs[-1]]
    im = ax2.imshow(
        dynspec_dedispersed,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
        interpolation="nearest",
        vmin=0,
    )
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Frequency (MHz)")
    ax2.set_xlim(10, 60)  # Focus on pulse region

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax2, pad=0.02)
    cbar.set_label("Intensity")

    plt.tight_layout()
    output_filename = "dedispersed_scattered_pulse.png"
    plt.savefig(output_filename, dpi=150)
    print(f"Dedispersed scattered pulse plot saved to {output_filename}")


def main():
    # Parameters
    params = FRBParams(
        dm=50.0,  # pc/cm^3
        width=2.0,  # ms (intrinsic width)
        amplitude=1.0,
        t0=20.0,  # ms (arrival time at infinite frequency)
        tau_1ghz=5.0,  # ms (scattering timescale at 1 GHz)
        tau_alpha=4.0,  # Scattering index (thin screen)
    )

    # Frequencies: 1200 MHz to 1500 MHz
    freqs = np.linspace(1200, 1500, 256)

    # Time: must capture the dispersed pulse arrival times!
    # Delay @ 1200 MHz ~ 144ms. t0=20. Arrival ~ 164ms.
    # We simulate 0 to 250ms to be safe.
    t = np.linspace(0, 250, 4096)

    dynspec_dedispersed = simulate_and_dedisperse(params, t, freqs)

    # Calculate dedispersed time series
    time_series = dynspec_dedispersed.mean(axis=0)

    plot_results(t, freqs, time_series, dynspec_dedispersed, params)

    # Print scattering timescales at edges
    tau_low = tau_per_freq(params.tau_1ghz, np.array([freqs[0]]), params.tau_alpha)[0]
    tau_high = tau_per_freq(params.tau_1ghz, np.array([freqs[-1]]), params.tau_alpha)[0]
    print(f"Scattering timescale at {freqs[0]:.0f} MHz: {tau_low:.2f} ms")
    print(f"Scattering timescale at {freqs[-1]:.0f} MHz: {tau_high:.2f} ms")


if __name__ == "__main__":
    main()
