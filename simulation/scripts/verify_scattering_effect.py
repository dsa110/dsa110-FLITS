import numpy as np
import matplotlib.pyplot as plt
from flits.models import FRBModel
from flits.params import FRBParams
from flits.scattering import scatter_broaden


def verify():
    # Parameters
    params = FRBParams(
        dm=50.0, width=2.0, amplitude=1.0, t0=20.0, tau_1ghz=5.0, tau_alpha=4.0
    )

    # Single low frequency where scattering is strongest
    freq = 1200.0  # MHz
    freqs = np.array([freq])

    # Time
    t = np.linspace(100, 200, 1024)

    # 1. Simulate with scattering DISABLED
    params_no_scat = FRBParams(dm=50.0, width=2.0, amplitude=1.0, t0=20.0, tau_1ghz=0.0)
    model_no_scat = FRBModel(params_no_scat)
    dynspec_no_scat = model_no_scat.simulate(t, freqs)
    profile_no_scat = dynspec_no_scat[0]

    # 2. Simulate with scattering ENABLED
    model_scat = FRBModel(params)
    dynspec_scat = model_scat.simulate(t, freqs)
    profile_scat = dynspec_scat[0]

    # 3. Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(t, profile_no_scat, label="No Scattering", linestyle="--")
    plt.plot(t, profile_scat, label="With Scattering (tau_1ghz=5.0)")
    plt.title(f"Scattering Verification at {freq} MHz")
    plt.xlabel("Time (ms)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("verify_scattering.png")

    print(f"Peak No Scat: {profile_no_scat.max()}")
    print(f"Peak Scat: {profile_scat.max()}")
    print(f"Location No Scat: {t[np.argmax(profile_no_scat)]}")
    print(f"Location Scat: {t[np.argmax(profile_scat)]}")


if __name__ == "__main__":
    verify()
