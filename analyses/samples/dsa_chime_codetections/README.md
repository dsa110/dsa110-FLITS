# DSA-110 + CHIME Co-detections

Multi-burst analyses for the DSA-110 + CHIME co-detection sample.

## Analyses

### Time-of-Arrival Cross-matching

**Notebook:** `toa_crossmatch.ipynb`

Cross-telescope TOA comparison with corrections for:

- Barycentric delays
- Geometric delays from Earth rotation
- Reference frequency standardization (400 MHz)
- Pulse width (FWHM) measurements

### 3D Scintillation Mapping

**Notebook:** `scintillation_3dmap.ipynb`

3D visualization of scintillation properties across the co-detection sample.

## Data

- 12 FRBs co-detected by DSA-110 and CHIME
- See `configs/bursts.yaml` for burst properties
- Individual burst analyses in `analyses/bursts/{name}/`
