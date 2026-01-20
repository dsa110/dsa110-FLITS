I will create a comprehensive documentation file `docs/architecture/system_reference.md` that details the codebase structure, analysis pipeline, and data architecture as requested.

The documentation will cover:
1.  **Codebase Structure**:
    - Detailed directory tree explaining `flits/`, `scattering/`, `configs/`, and `analyses/`.
    - Architecture of the analysis pipeline including `BurstDataset` (loading/preprocessing), `FRBModel` (physics), and `BurstPipeline` (orchestration).

2.  **Data Architecture**:
    - **Raw Data**: 2D Numpy arrays `(nfreq, ntime)` in `.npy` format.
    - **Metadata**: Schema defined in `configs/bursts.yaml` and `chimedsa_burst_specs.csv` (TNS names, coordinates, DM).
    - **Results**: JSON structure for fit results and MCMC chains.

3.  **Special Cases**:
    - Handling of simultaneous detections (DSA-110 + CHIME) via `crossmatching/toa_crossmatch.py`.
    - Naming conventions mapping informal nicknames (e.g., "Casey") to TNS identifiers.

4.  **Implementation Details**:
    - Data preprocessing steps (bandpass correction, trimming, downsampling).
    - File formats and version control practices.

I will then add a reference to this new file in the main `README.md` to ensure it is discoverable.