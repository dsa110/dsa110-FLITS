# P1 windowed upchannelization — Freya result

Status: `DOCUMENTED-FAIL`. No variant passed the predeclared mechanism gate;
the unchanged blinded C1 calibration was not run.

The rectangular/oversample-2 variant was first run through the local
shape-compatible path and produced the same Stokes-I SHA-256 as the retained
production product. The pinned Docker equivalence check was bit-for-bit for
the package's private `_upchannel` implementation. The four new candidates
were then generated from coherently dedispersed Freya baseband on h17 using
the same `baseband-analysis` image and retained `2U` output cadence. Each
candidate retained separate detected polarization products and a metadata
hash.

| window | oversample | lag-1 cross-ACF | fitted amplitude | fitted width (kHz) | gate |
| --- | ---: | ---: | ---: | ---: | --- |
| rectangular | 2 | 0.5923 | 0.6499 | 40.77 | FAIL |
| Hann | 2 | 0.5945 | 0.6353 | 39.61 | FAIL |
| Hann | 4 | 0.5867 | 0.6174 | 37.38 | FAIL |
| Blackman–Harris | 2 | 0.6188 | 0.6843 | 41.39 | FAIL |
| Blackman–Harris | 4 | 0.6000 | 0.6354 | 37.94 | FAIL |

The gate required both `|rho_lag1| <= 0.0587` and fitted common-mode
amplitude `<= 0.0586`, corresponding to at least 10× suppression relative to
the retained-product baseline (`rho_lag1=0.587`, amplitude `0.586`). The best
candidate, Hann oversample 4, leaves amplitude `0.6174`, so the upstream
window change does not remove the common-mode response. The result is a
product-level negative result, not a qualified scintillation measurement.

The compact committed validation record is
[`validation.json`](validation.json). Detailed per-product measurement JSONs
and the large NPZ/NPY products remain on h17 under
`/data/research/astrophysics/frbs/chime-dsa-codetections/upchan_codetections/p1-window-upchan/`,
with their SHA-256 values recorded in that validation bundle. No on-pulse fit,
variant selection, or C1 calibration is authorized after this gate failure.
