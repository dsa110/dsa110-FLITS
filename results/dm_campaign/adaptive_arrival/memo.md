# Adaptive arrival-regression results

The same time/frequency/sub-band grid and stability policy was evaluated
on all 24 products. A per-band result is science-grade only when at least
two distinct resolution choices agree within 0.25 pc/cm3 and sigma_DM <= 0.5 pc/cm3, and the canonical reduced-chi2
classifier returns PASS. MARGINAL fits retain their candidate DM for audit
but are excluded from event-level summaries.

The event-level inverse-variance summaries below are not direct cross-band
fits. The stored CHIME and DSA arrays have independent time origins; a fit
that exploits the 0.4--1.5 GHz lever arm remains blocked until the absolute
per-array time origins are restored and verified.

| burst | CHIME | DSA | event support | event DM |
|---|---|---|---|---:|
| casey | marginal-fit: 491.216342 +/- 0.024109 | marginal-fit: 491.189078 +/- 0.006120 | none | -- |
| chromatica | marginal-fit: 272.654973 +/- 0.002766 | marginal-fit: 272.662567 +/- 0.025066 | none | -- |
| freya | science-grade: 912.453524 +/- 0.003831 | science-grade: 912.510747 +/- 0.012035 | two-band-tension | 912.458789 +/- 0.016540 |
| hamilton | marginal-fit: 518.794978 +/- 0.085286 | science-grade: 518.788058 +/- 0.041438 | single-band | 518.788058 +/- 0.041438 |
| isha | marginal-fit: 411.581792 +/- 0.016744 | weak-only | none | -- |
| johndoeII | marginal-fit: 696.514264 +/- 0.006900 | marginal-fit: 696.490656 +/- 0.018566 | none | -- |
| mahi | marginal-fit: 960.098715 +/- 0.024632 | science-grade: 960.158033 +/- 0.042178 | single-band | 960.158033 +/- 0.042178 |
| oran | science-grade: 396.707745 +/- 0.047385 | science-grade: 396.938447 +/- 0.045231 | two-band-tension | 396.828459 +/- 0.115226 |
| phineas | science-grade: 610.481017 +/- 0.008920 | marginal-fit: 610.214258 +/- 0.006211 | single-band | 610.481017 +/- 0.008920 |
| whitney | science-grade: 462.190039 +/- 0.006344 | marginal-fit: 462.191935 +/- 0.015820 | single-band | 462.190039 +/- 0.006344 |
| wilhelm | marginal-fit: 602.370492 +/- 0.011368 | marginal-fit: 602.407621 +/- 0.294603 | none | -- |
| zach | marginal-fit: 262.396482 +/- 0.004337 | marginal-fit: 262.311454 +/- 0.014566 | none | -- |
