# Controlled DM-phase algorithm v1

Status: implementation specification. The production estimator is defined here; the
vendored published `DM_phase` package at commit `b7cf5fd61436` is a frozen comparison oracle.

## Inputs and conventions

The input intensity waterfall has shape `(n_frequency, n_time)`. Frequencies are channel
centres in MHz and are sorted into ascending order. Time resolution is in seconds. Trial
values are *physical residual DM* in pc cm^-3 relative to an explicit reference DM:
positive residual DM means that lower frequencies arrive later.

With the top of the band as reference, the residual delay is

```text
delay(nu, dDM) = K_DM dDM (nu^-2 - nu_ref^-2),
K_DM = 1 / 2.41e-4 s MHz^2 pc^-1 cm^3.
```

Dedispersion aligns a positive residual by shifting each low-frequency channel earlier by
`delay / dt` samples. Public results always retain this physical sign; there is no hidden
legacy trial-axis sign conversion.

## Boundary-safe fractional shifting

For each trial, pad both ends of every channel by at least
`ceil(max(abs(delay / dt))) + guard_samples`. Apply the Fourier shift theorem on the padded
axis and crop the original interval. The padding converts the otherwise circular operation
into a zero-filled fractional shift over the retained interval. Samples shifted outside the
record do not re-enter at the opposite edge.

## Phase coherence metric

For each dedispersed trial waterfall `I_nu(t)`:

1. subtract the per-channel median and divide by `1.4826 MAD`;
2. Fourier transform along time, `F_nu(f) = FFT_t[I_nu(t)]`;
3. retain phase only, `P_nu(f) = F_nu(f) / max(|F_nu(f)|, epsilon)`;
4. apply normalized non-negative channel weights and sum,
   `C(f) = sum_nu w_nu P_nu(f)`;
5. form coherent power `Q(f) = |C(f)|^2`;
6. on positive fluctuation frequencies within the frozen cutoff, integrate
   `S(dDM) = sum_f Q(f) f^2`.

The `f^2` weighting matches the structure sensitivity of the published method. Zero-amplitude
bins and masked channels have zero contribution. Uniform amplitude rescaling cannot move the
metric peak.

## Search and uncertainty

A uniform coarse grid first brackets the maximum. A fine grid centred on that cell is then
evaluated independently. A local three-point parabola may interpolate only inside the
neighbouring fine-grid cells and only when its curvature is negative. An edge maximum expands
once when time support permits; a persistent edge maximum is `UNCONSTRAINED`.

Channel bootstrap resamples use a recorded seed. Each resample repeats the phase-only channel
sum on the frozen fine grid. The bootstrap success fraction is the fraction of finite,
interior peaks. Statistical uncertainty is the standard deviation of successful peaks and is
never allowed to be zero. Injection-derived method and resolution terms are added separately
before an adopted total uncertainty is reported.

## Published-oracle relationship and ADRs

The oracle is used for regression evidence, not copied into the production kernel.

- ADR-001: use physical residual-DM sign directly. The released code accepts a trial sign
  whose interpretation depends on how the input was pre-dedispersed.
- ADR-002: replace released `np.roll` integer shifts with padded fractional shifts. This
  prevents wrap-around and removes integer-sample quantization.
- ADR-003: use an explicit, frozen cutoff in Hz for production and test adjacent cutoffs.
  The released automatic cutoff mixes trial-DM and fluctuation-frequency axes and contains
  wrap-boundary smoothing.
- ADR-004: use a bounded local quadratic and bootstrap peaks. The released sixth-order
  polynomial and curvature error can return unstable or zero uncertainties.
- ADR-005: normalize and mask channels explicitly before phase summation. This makes dead-band
  handling and weights part of the result provenance.

Intentional deviations require tests showing their effect. Unexplained disagreement in sign,
peak location, or coherent-power shape blocks science measurements.
