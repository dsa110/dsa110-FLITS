import numpy as np
import matplotlib.pyplot as plt
from flits.models import FRBModel, K_DM
from flits.params import FRBParams
from flits.scattering import scatter_broaden


def diagnostic():
    # Parameters with stronger scattering to make it obvious
    params = FRBParams(
        dm=0.0,  # Start with no dispersion
        width=1.0,  # Narrower pulse
        amplitude=1.0,
        t0=30.0,
        tau_1ghz=8.0,  # Stronger scattering
        tau_alpha=0.0,  # Frequency-independent first
    )

    # Single frequency
    freq = 1400.0
    freqs = np.array([freq])

    # Time axis
    t = np.linspace(0, 100, 2048)
    dt = t[1] - t[0]

    # 1. Pure Gaussian (no scattering, no dispersion)
    shifted = t - params.t0
    gaussian_only = params.amplitude * np.exp(-0.5 * (shifted / params.width) ** 2)

    # 2. Gaussian + Scattering (manually apply)
    scattered = scatter_broaden(gaussian_only, t, params.tau_1ghz, causal=True)

    # 3. Using FRBModel
    model = FRBModel(params)
    dynspec = model.simulate(t, freqs)
    model_result = dynspec[0, :]

    # Create diagnostic plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Gaussian only
    axes[0].plot(t, gaussian_only, "b-", lw=2, label="Pure Gaussian")
    axes[0].set_ylabel("Intensity")
    axes[0].set_title("Step 1: Pure Gaussian Pulse")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xlim(20, 70)

    # Plot 2: Manual scattering
    axes[1].plot(t, gaussian_only, "b--", alpha=0.5, label="Original Gaussian")
    axes[1].plot(
        t, scattered, "r-", lw=2, label=f"After Scattering (τ={params.tau_1ghz}ms)"
    )
    axes[1].set_ylabel("Intensity")
    axes[1].set_title("Step 2: Manual Scattering Applied")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlim(20, 70)

    # Plot 3: Model result
    axes[2].plot(t, gaussian_only, "b--", alpha=0.5, label="Original Gaussian")
    axes[2].plot(t, model_result, "g-", lw=2, label="FRBModel.simulate()")
    axes[2].set_ylabel("Intensity")
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_title("Step 3: FRBModel Result")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_xlim(20, 70)

    plt.tight_layout()
    plt.savefig("scattering_diagnostic.png", dpi=150)

    # Print statistics
    print(
        f"Gaussian peak: {gaussian_only.max():.4f} at t={t[np.argmax(gaussian_only)]:.2f}ms"
    )
    print(
        f"Manual scattered peak: {scattered.max():.4f} at t={t[np.argmax(scattered)]:.2f}ms"
    )
    print(
        f"Model result peak: {model_result.max():.4f} at t={t[np.argmax(model_result)]:.2f}ms"
    )
    print(f"\nGaussian integral: {np.sum(gaussian_only) * dt:.4f}")
    print(f"Manual scattered integral: {np.sum(scattered) * dt:.4f}")
    print(f"Model result integral: {np.sum(model_result) * dt:.4f}")

    # Check if profiles match
    diff = np.abs(scattered - model_result).max()
    print(f"\nMax difference (manual vs model): {diff:.6f}")
    if diff < 1e-6:
        print("✓ Manual and model results match!")
    else:
        print("✗ Manual and model results differ!")


if __name__ == "__main__":
    diagnostic()
