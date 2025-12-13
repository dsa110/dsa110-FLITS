from dataclasses import dataclass


@dataclass
class FRBParams:
    """Parameters describing a fast radio burst model with optional scattering.

    Attributes:
        dm: Dispersion measure (pc cm^-3)
        amplitude: Pulse amplitude (arbitrary units)
        t0: Reference arrival time (ms)
        width: Intrinsic Gaussian width of the burst (ms)
        tau_1ghz: Scattering timescale at 1 GHz (ms); 0.0 disables scattering
        tau_alpha: Frequency scaling exponent for scattering (τ ∝ ν^−α).
                   Default ≈4.4 for Kolmogorov turbulence.
    """

    dm: float  # Dispersion measure in pc cm^-3
    amplitude: float  # Pulse amplitude in arbitrary units
    t0: float = 0.0  # Reference arrival time in ms
    width: float = 1.0  # Gaussian width of the burst in ms
    tau_1ghz: float = 0.0  # Scattering timescale at 1 GHz (ms); 0 disables
    tau_alpha: float = 4.4  # Frequency scaling exponent (Kolmogorov: ~4.4)


__all__ = ["FRBParams"]
