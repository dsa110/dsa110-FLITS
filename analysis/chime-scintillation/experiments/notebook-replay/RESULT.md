# Recovered Freya notebook replay and falsification

**Status: falsified. Neither the archived 3.8 MHz fit nor the replayed 190 kHz
fit is a qualified CHIME scintillation measurement.**

The recovered notebook executes against the surviving same-named input, but
that input and helper state differ from the historical run. More decisively,
the inherited `725:875` integration window is off-pulse in the surviving data.
All 24 matched off-pulse windows are non-white and reproduce the same width
family. Moving the estimator to the true burst window changes the fitted number
but does not remove the instrumental covariance.

## Artifacts

- [Replay narrative](../../../../scintillation/scint_analysis/reference_arc/replays/2026-07-14-scint-freya/README.md)
- [Falsification report](../../../../scintillation/scint_analysis/reference_arc/replays/2026-07-14-scint-freya/FALSIFICATION.md)
- [Machine-readable falsification](../../../../scintillation/scint_analysis/reference_arc/replays/2026-07-14-scint-freya/matched-window-falsification.json)
- [Artifact manifest](../../../../scintillation/scint_analysis/reference_arc/replays/2026-07-14-scint-freya/artifacts.manifest.json)
- [Archived fit](../../../../scintillation/scint_analysis/reference_arc/replays/2026-07-14-scint-freya/archived-acf-fit.png)
- [Surviving-input replay](../../../../scintillation/scint_analysis/reference_arc/replays/2026-07-14-scint-freya/replayed-acf-fit.png)
- [Matched-window falsification battery](../../../../scintillation/scint_analysis/reference_arc/replays/2026-07-14-scint-freya/matched-window-falsification.png)

This closes the recovered-notebook route for a physical measurement from this
detected-intensity product.
