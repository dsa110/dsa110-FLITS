# Plan — Freya CHIME pre-upchannelization voltage calibration (B1)

**Frozen:** 2026-07-13, before injection recovery was examined
**Qualification target:** Freya (`FRB 20230325A`) only
**Starting status:** CHIME `diagnostic_only`; the real on-pulse ACF is forbidden

## Question

Can the exact CHIME voltage-to-waterfall operator preserve a known scintillation-like
frequency scale when a deterministic complex-voltage burst is inserted after coherent
dedispersion but before `_upchannel`, Stokes-I formation, bandpass normalization, and
padded alignment?

B1 is a calibration of the production operator, not another correction hypothesis.
It does not reopen H0--H3 or A1 and it cannot promote the real burst by itself.

## Provenance gate

The live input is Freya's staged 1.2 GB
`singlebeam_278720455.h5` on `lxd110h17`. The producer is
`/data/research/astrophysics/frbs/chime-dsa-codetections/scripts/upchannelize_chime.py`
inside `chimefrb/baseband-analysis@sha256:f510909d...c4c41`
(`baseband_analysis` 1.9.0). Before injections, a fresh raw-HDF5 replay must be
bit-for-bit identical to the canonical `freya_chime_{upchan,freq}.npy` files.

The coherent-dedispersion path is covered by that exact baseline replay. Synthetic
signals are then inserted into the returned complex voltage array immediately before
the producer's `_upchannel(fftsize=128, downfreq=2)` call. This isolates the disputed
upchannelization/ripple and subsequent alignment transfer without modifying the real
burst or claiming that B1 calibrates an arbitrary DM error.

## Frozen injection battery

- retain the real Freya voltage noise and use three burst-free aligned output centers;
- inject only into a protected three-block window at each center;
- use the exact U=64 operator, two CHIME bands (400--627 and 627--800 MHz), and
  nominal Lorentzian correlation widths of 2, 4, 8, and 16 fine channels;
- use two injected signal-to-background power ratios, 1 and 4;
- use one deterministic seeded complex-voltage realization per combination;
- total: `2 bands x 4 widths x 2 levels x 3 centers = 48` trials;
- derive a positive target intensity with a seeded fixed-periodogram stationary field, fit
  that target with the same positive-lag Lorentzian model, and compare recovery to
  the fitted target width rather than assuming the generator is exact;
- form paired injected-minus-baseline spectra after per-channel off-pulse median
  normalization and padded alignment. The pairing is available only because B1 is
  a calibration experiment; it is not an operation permitted on the real burst.

## Frozen success gates

All conditions are required:

1. raw-HDF5 baseline replay hashes match the canonical waterfall and frequency axis;
2. every target fit and all 48 recovered fits are finite;
3. each target realization differs from its nominal width by no more than 10%;
4. every recovered width differs from its fitted target by less than
   `max(10%, 0.25 fine channel)`;
5. for every trial, recovered paired signal power is within 20% of truth at unit
   signal/background and within 10% at ratio 4;
6. manual figure review agrees with the machine verdict.

Any failure closes B1 as `DOCUMENTED-FAIL`. A complete pass establishes only that
the exact voltage-to-waterfall transfer is calibratable. It would authorize a
separate, predeclared response-estimation experiment; it would not authorize an
on-pulse scintillation measurement.

## Final evidence

B1 closed as **DOCUMENTED-FAIL**. The raw-HDF5 replay was bit-for-bit identical
to both canonical arrays, all 48 target and recovered fits were finite, all 48
target realizations passed the generator-width gate, and all 48 injected powers
were recovered within the frozen tolerances. Width recovery failed 0/48.

The failure is structured and independent of injection strength. In the
400--627 MHz band, all 24 recovered widths hit the 0.350 MHz fit ceiling for
truth spanning 0.0110--0.0992 MHz (median fractional bias 9.65; maximum 30.75).
In the 627--800 MHz band, all 24 recovered widths collapsed into
0.0539--0.0681 MHz for truth spanning 0.0114--0.1008 MHz (median fractional
bias 0.785; maximum 4.23). Recovered/injected total power remained
0.9991--1.0014 in the low band and 0.9935--1.0088 in the high band.

The injected power ratios 1 and 4 exceed Freya's measured normalized signal
means of about 0.30 and 0.47 in the low and high bands. Thus B1 did not merely
miss an injection weaker than the real burst. The exact operator conserves
mean signal power, but on real voltage noise the ACF shape is dominated by a
band-dependent signal--noise/instrumental covariance scale rather than the
known injected correlation width. Manual review confirms the two horizontal
recovery loci. No response correction and no real on-pulse fit are authorized.
