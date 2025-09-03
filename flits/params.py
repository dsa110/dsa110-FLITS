from dataclasses import dataclass


@dataclass
class FRBParams:
    """Parameters describing a simple fast radio burst model."""

    dm: float  # Dispersion measure in pc cm^-3
    amplitude: float  # Pulse amplitude in arbitrary units
    t0: float = 0.0  # Reference arrival time in ms
    width: float = 1.0  # Gaussian width of the burst in ms


__all__ = ["FRBParams"]
