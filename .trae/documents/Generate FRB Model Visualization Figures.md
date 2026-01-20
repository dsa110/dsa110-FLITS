I will generate the requested model visualization figures.

1.  **Preparation**:
    -   I have verified `SciencePlots` is installed.
    -   I have reviewed `scattering/scat_analysis/burstfit.py` and understand how to instantiate `FRBModel` and `FRBParams`.

2.  **Implementation (`scripts/generate_model_examples.py`)**:
    -   I will create a script that sets up a standard time-frequency grid (DSA-110 like: 1.4 GHz center, +/- 10ms window).
    -   **Single Component Models**:
        -   **M0 (Unresolved)**: `zeta=0`, `tau=0`. Pulse width dominated by DM smearing (I will set a non-zero `delta_dm` or rely on `dm_init` smearing if `delta_dm` is relative). Actually, `FRBModel` uses `dm_init` for the smearing calculation width `_smearing_sigma`, and `delta_dm` for the delay law. To show smearing, I need a non-zero DM.
        -   **M1 (Resolved)**: `zeta=1.0` ms (intrinsic), `tau=0`.
        -   **M2 (Scattered Unresolved)**: `zeta=0`, `tau=2.0` ms, `alpha=4.4` (fixed).
        -   **M3 (Scattered Resolved)**: `zeta=1.0` ms, `tau=2.0` ms, `alpha=3.0` (free/different).
    -   **Multi-Component Model**:
        -   Component 1: **M1** (Early, Resolved).
        -   Component 2: **M3** (Middle, Scattered+Resolved).
        -   Component 3: **M2** (Late, Scattered Unresolved).
    -   **Visualization**:
        -   Use `plt.style.context(['science', 'notebook'])`.
        -   Create a 1x5 subplot layout.
        -   For each panel, plot the dynamic spectrum (waterfall).
        -   Add titles and labels.

3.  **Execution**:
    -   Run the script to produce `model_examples.png`.
    -   Verify the output confirms the expected behaviors (e.g., M0 should look like a thin curve smeared by DM channels, M2 should have a tail, etc.).