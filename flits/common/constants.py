"""Shared physical constants for the FLITS project."""

# Dispersion constant in MHz^2 pc^-1 cm^3 s -> seconds
K_DM = 4.148808e3  # MHz^2 pc^-1 cm^3 s

# Dispersion constant in MHz^2 pc^-1 cm^3 ms -> milliseconds
# Use this when computing delay(ms) = K_DM_MS * DM(pc/cm³) / freq(MHz)²
K_DM_MS = 4.148808e6  # MHz^2 pc^-1 cm^3 ms

# Cold-plasma dispersion delay in ms GHz^2 (pc cm^-3)^-1
DM_DELAY_MS = 4.148808  # ms GHz^2 (pc cm^-3)^-1

