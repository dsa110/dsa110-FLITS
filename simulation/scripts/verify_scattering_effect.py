import matplotlib.pyplot as plt
import numpy as np

from scattering.scat_analysis.burstfit import DM_DELAY_MS, FRBModel, FRBParams


def verify():
    # Parameters
    dm, width, amplitude, t0, tau_alpha = 50.0, 2.0, 1.0, 20.0, 4.0

    # Single low frequency where scattering is strongest
    freq = 1200.0  # MHz
    freqs = np.array([freq])

    # Time
    t = np.linspace(100, 200, 1024)

    # Core references dispersion to f_max; shift t0 (legacy arrival at infinite
    # frequency) by the band-top delay to preserve the absolute arrival time.
    freqs_ghz = freqs / 1000.0
    t0_core = t0 + DM_DELAY_MS * dm / freqs_ghz.max() ** 2

    def sim(tau_1ghz):
        p = FRBParams(
            c0=amplitude,
            t0=t0_core,
            gamma=0.0,
            zeta=width,
            tau_1ghz=tau_1ghz,
            alpha=tau_alpha,
            delta_dm=dm,
        )
        return FRBModel(time=t, freq=freqs_ghz, dm_init=dm, df_MHz=1.0)(p, "M3")[0]

    # 1. Scattering DISABLED, then 2. ENABLED
    profile_no_scat = sim(0.0)
    profile_scat = sim(5.0)

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
