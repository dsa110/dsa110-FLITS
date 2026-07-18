# CHIME scintillation artifact inventory

This directory is the canonical index for CHIME-only scintillation validation.
It links to the immutable result bundles where they were originally produced;
it does not duplicate large validation records or raw data.

## Current scientific answer

There is **no qualified CHIME scintillation-bandwidth measurement** in the
indexed results. The H0/A1/H2/H3, B3/B4, and C1 qualification routes failed,
the trigger campaign completed with zero power over its tested alternative
grid, and the recovered-notebook peak was falsified as an off-pulse
instrumental scale. C1 (all-pairs distinct-time cross-ACF, the owner-selected
route after B4) reached a blinded NO-GO — 0/8 gated calibration cells and a
failed null campaign — without any unblinded on-pulse fit; per its stop rule,
further estimator tuning on the retained product is closed, and the next route
must change the input product itself (windowed re-upchannelization). In
particular, this inventory contains no qualified Oran CHIME detection.
DSA-110 results are outside this CHIME-only inventory.

The sanctioned windowed-re-upchannelization successor was executed on Freya
under `p1-window-upchan`. Rectangular, Hann, and Blackman–Harris products at
oversample 2 and 4 all failed the predeclared 10× off-pulse common-mode gate;
the route therefore stopped before any on-pulse fit or new C1 calibration.

## Indexed experiments

| Experiment | Input type | Status | Start here |
| --- | --- | --- | --- |
| H0 rank-1 smoke test | real retained CHIME product | exact reconstructed failure; diagnostic only | [`experiments/h0-smoke/README.md`](experiments/h0-smoke/README.md) |
| A1 additive covariance | real CHIME product plus injections | documented failure; diagnostic only | [`experiments/a1-covariance/README.md`](experiments/a1-covariance/README.md) |
| H2 rank-2 calibration | real CHIME product plus injections | documented failure; diagnostic only | [`experiments/h2-calibration/README.md`](experiments/h2-calibration/README.md) |
| H3 stationary-kernel whitening | real CHIME product plus injections | documented failure; diagnostic only | [`experiments/h3-scallop/README.md`](experiments/h3-scallop/README.md) |
| B3 high-band polarization cross-ACF | real CHIME product plus injections | documented failure; diagnostic only | [`experiments/b3-highband-crossacf/RESULT.md`](experiments/b3-highband-crossacf/RESULT.md) |
| B4 four-stream cross-ACF | real CHIME product plus injections | documented failure; diagnostic only | [`experiments/b4-fourstream-crossacf/RESULT.md`](experiments/b4-fourstream-crossacf/RESULT.md) |
| C1 all-pairs distinct-time cross-ACF | real CHIME product plus injections (blinded) | documented failure; no unblinded fit | [`experiments/c1-allpairs-crossgp/RESULT.md`](experiments/c1-allpairs-crossgp/RESULT.md) |
| P1 windowed re-upchannelization | Freya coherently dedispersed baseband; five shape-compatible variants | documented failure; no on-pulse fit or C1 calibration | [`experiments/p1-window-upchan/RESULT.md`](experiments/p1-window-upchan/RESULT.md) |
| A1 trigger calibration | synthetic null and two-component campaigns | completed but zero power; diagnostic only | [`experiments/trigger-calibration/RESULT.md`](experiments/trigger-calibration/RESULT.md) |
| Recovered notebook replay | real surviving CHIME product | falsified; not a measurement | [`experiments/notebook-replay/RESULT.md`](experiments/notebook-replay/RESULT.md) |

Machine-readable routing and status live in [`INVENTORY.yaml`](INVENTORY.yaml).
External input locations and immutable hashes live in
[`DATA_MANIFEST.yaml`](DATA_MANIFEST.yaml). Figure and replay manifests use
repository-relative paths so they remain valid in any checkout.

## Status vocabulary

- `qualified_measurement`: all predeclared scientific gates pass and a physical
  bandwidth may be reported.
- `qualified_simulation`: a synthetic recovery/control result passed; it does
  not by itself qualify a real-data measurement.
- `diagnostic_only`: useful behavior or a fitted number was obtained, but one
  or more measurement gates failed or were not run.
- `falsified`: a proposed interpretation failed a decisive control.
- `summary_only`: historical evidence exists without a complete reproducible
  artifact bundle.

The experiment summaries state which gates failed. A visually clean figure or
a precise fit is not sufficient to change the status.
