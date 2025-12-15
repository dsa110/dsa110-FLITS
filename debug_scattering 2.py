import numpy as np
import matplotlib.pyplot as plt
from flits.scattering.broaden import scatter_broaden


def test_scattering():
    # Time axis: 0 to 100 ms
    t = np.linspace(0, 100, 1000)
    dt = t[1] - t[0]

    # Signal: Delta-like Gaussian at t=10
    width = 1.0
    signal = np.exp(-0.5 * ((t - 10.0) / width) ** 2)

    # Large scattering time to make it obvious
    tau = 20.0  # ms

    # Apply scattering
    broadened = scatter_broaden(signal, t, tau, causal=True)

    # Plot
    plt.figure()
    plt.plot(t, signal, label="Original (Gaussian)")
    plt.plot(t, broadened, label=f"Broadened (tau={tau}ms)")
    plt.legend()
    plt.title("Scattering Debug")
    plt.savefig("debug_scattering.png")
    print(f"Max original: {signal.max()}")
    print(f"Max broadened: {broadened.max()}")
    print(f"Area original: {np.sum(signal) * dt}")
    print(f"Area broadened: {np.sum(broadened) * dt}")


if __name__ == "__main__":
    test_scattering()
