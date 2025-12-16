# Simulation Scripts

Utility scripts for scattering simulation, validation, and diagnostics.

## Scripts

### Scattering Simulation

**`simulate_scattered_pulse.py`**

- Creates synthetic scattered pulses using wave optics
- Used for testing pipeline on known inputs

**Usage:**

```bash
python simulation/scripts/simulate_scattered_pulse.py
```

### Plotting & Visualization

**`plot_dedispersed_scattered.py`**

- Generates plots of dedispersed scattered pulses
- Useful for visualizing scattering effects

**Usage:**

```bash
python simulation/scripts/plot_dedispersed_scattered.py
```

### Validation & Verification

**`verify_scattering_effect.py`**

- Verifies scattering implementation correctness
- Outputs: `results/simulation/validation/verify_scattering.png`

**`validate_wave_optics_spectrum.py`** (if exists)

- Validates wave optics spectrum calculations
- Outputs: `results/simulation/validation/validate_wave_optics_spectrum.png`

### Debugging

**`debug_scattering.py`**

- Debug utility for scattering calculations

**`diagnostic_scattering.py`**

- Diagnostics for scattering model components

## Related

- Main simulator: `simulation/` (core module)
- Validation results: `results/simulation/validation/`
- Simulation docs: `docs/architecture/overview.md`
