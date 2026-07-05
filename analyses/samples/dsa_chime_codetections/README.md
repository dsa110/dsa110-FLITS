# DSA-110 + CHIME Co-detections

Multi-burst analyses for the DSA-110 + CHIME co-detection sample.

## Analyses

### Time-of-Arrival Cross-matching

Cross-telescope TOA comparison with corrections for:

- Barycentric delays
- Geometric delays from Earth rotation
- Reference frequency standardization (400 MHz)
- Pulse width (FWHM) measurements

The checked-in notebook artifact was removed; use the maintained
`crossmatching/` modules and distilled fixtures instead.

### 3D Scintillation Mapping

3D visualization of scintillation properties across the co-detection sample.

The checked-in notebook artifact was removed; regenerate exploratory views in
local untracked notebooks when needed.

## Data

- 12 FRBs co-detected by DSA-110 and CHIME
- See `configs/bursts.yaml` for burst properties
- Individual burst analyses in `analyses/bursts/{name}/`
