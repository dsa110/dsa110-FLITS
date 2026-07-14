# Plan — Freya CHIME folded-scallop voltage calibration (B2)

**Frozen:** 2026-07-13, before any B2 recovery result was examined
**Qualification target:** Freya (`FRB 20230325A`) only
**Starting status:** CHIME `diagnostic_only`; the real on-pulse ACF remains forbidden

## Question

Does a response-consistent correction for the known repeating CHIME PFB
scallop restore scintillation-width transfer through the exact production
upchannelization path?

B1 inserted synthetic fine-channel voltages after the original coarse PFB but
then divided the recovered power by an off-pulse gain that contained the PFB
scallop. The synthetic signal had not been forward-folded through that scalar
gain, so B1 did not test a matched forward/inverse scallop correction.

## Frozen operator and data

- Input HDF5:
  `/data/research/astrophysics/frbs/chime-dsa-codetections/chime_singlebeam/singlebeam_278720455.h5`
- Input SHA-256:
  `676a9033c10926c213603939bee78c44d6d1a011c01e4279b41bccc97127df52`
- Producer baseline: coherent dedispersion, `_upchannel(fftsize=128,
  downfreq=2)`, Stokes I, padded coarse-delay alignment.
- Container digest:
  `chimefrb/baseband-analysis@sha256:f510909d892d0d5224c982c590cbe80967a49a59b79c396ab72bb710105c4c41`
- Canonical waterfall SHA-256:
  `f7ce42985a99eeb70565abb1b8458ec888fa2e65396cef2e2bf9b367941f62b9`
- Canonical frequency SHA-256:
  `b686f99e71ad37a0b9014aae8ea480865b7eda8c94f0eb6c1d82e11d6a800255`

## Scallop model

The scalar response is constrained to the known separable PFB form

\[
G(c,u)=A_c S_u,
\]

where `c` is native coarse channel and `u` is the 64-bin fine-channel
position. `S_u` is obtained by folding the off-pulse spectrum at period 64
and robustly averaging the same fine position over all native channels,
matching the rescued `make_scallop_model` prescription. `A_c` retains one
broad-band gain per native channel. The model is frozen once from the
uninjected baseline and reused for every trial.

The synthetic sky target is forward-folded at the voltage level by
`sqrt(G)`. Recovered Stokes-I signal power is divided by `G`. This tests a
matched scalar forward/inverse response rather than dividing an unfiltered
injection by the noise bandpass.

## Phase-cycled transfer

For an identical real-noise realization `n` and injected voltage `s`, B2
uses the `+s` and `-s` pair:

\[
\frac{|n+s|^2+|n-s|^2}{2}-|n|^2=|s|^2.
\]

This cancels the realization-dependent signal-noise cross term exactly for a
linear voltage operator. A noise-free `s`-only replay is retained as an
independent control.

## Frozen injection grid

- Bands: 400–627 and 627–800 MHz.
- Target HWHM: 2, 4, 8, and 16 fine channels.
- Corrected injected power ratios: 1 and 4 relative to the local baseline.
- Aligned centers: 18, 28, and 38.
- Total: 48 trials per route.
- Target seeds and Lorentzian fit convention match B1.
- Absolute fine-channel IDs remain in the ACF so missing hardware channels
  remain gaps.

The routes are:

1. `matched_noise_free`: forward `sqrt(G)`, no real noise, inverse `G`.
2. `matched_phase_cycled`: forward `sqrt(G)`, real noise, `+s/-s`, inverse `G`.
3. `unmatched_noise_free`: no forward gain and no real noise, then divide by
   `G` — the B1 response mismatch isolated from its cross term.

## Predeclared gates

The B2 scalar-scallop hypothesis passes only if all conditions hold:

1. Baseline replay hashes match the canonical products exactly.
2. The folded gain is finite and positive for every retained fine channel.
3. Split-half folded scallop shapes agree to RMS fractional difference ≤5%.
4. All 144 route/trial fits are finite.
5. Target-generator fits are within 10% of nominal width.
6. `matched_noise_free` recovers width in 48/48 trials within
   `max(10% of truth, 0.25 fine channel)`.
7. `matched_phase_cycled` recovers width under the same 48/48 gate.
8. `matched_phase_cycled` recovers signal power in 48/48 trials within 10%.
9. The matched phase-cycled median absolute fractional width error is at
   least two times smaller than the unmatched route's error.
10. Diagnostic figures pass manual review and agree with the machine record.

Failure of any gate leaves CHIME `diagnostic_only`. Thresholds will not be
changed after results are inspected. The real Freya on-pulse ACF will not be
fit in B2.

## Completed result — 2026-07-13

The exact h17 run completed all 48 grid points and all 144 route fits. Every
predeclared machine gate passed:

- baseline hashes and input provenance matched;
- all 144 fits were finite;
- the folded-scallop split-half RMS fractional difference was 0.01097, below
  the frozen 0.05 limit;
- matched noise-free width recovery passed 48/48 trials;
- matched phase-cycled width recovery passed 48/48 trials;
- matched phase-cycled power recovery passed 48/48 trials;
- the matched phase-cycled median absolute fractional width error was
  `9.63e-09`, versus `2.76` for the unmatched control.

Manual review of the complete figure manifest also passed: the matched routes
lie on the width identity line, matched phase-cycled power lies on unity, and
the unmatched control visibly fails both transfers. B2 therefore qualifies
the scalar folded-scallop forward/inverse calibration itself. Its science
status remains `calibration_only`: B2 did not fit Freya's real on-pulse ACF and
does not by itself establish a CHIME scintillation measurement.
