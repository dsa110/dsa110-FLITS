# Joint scattering analysis

This site documents how the `dsa110-FLITS` pipeline measures the **scattering
index** $\alpha$ from CHIME+DSA co-detected fast radio bursts, and how it
detects temporal sub-components that, left unmodeled, bias $\alpha$ high.

A single-band scattering fit cannot separate the scattering index from the
scattering time: the pulse broadening
$\tau(\nu) = \tau_{1\,\mathrm{GHz}}\,(\nu/1\,\mathrm{GHz})^{-\alpha}$ enters a
single band only through its value at that band's frequency. A CHIME+DSA
co-detection fits both bands **simultaneously** with a shared
$(\tau_{1\,\mathrm{GHz}}, \alpha)$, and the $\sim$1 GHz lever arm between the
bands measures $\alpha$ directly. But that measurement is only honest if the
*temporal* model in each band is correct — and some bursts hide a second pulse.

## The four pages

- **[The joint likelihood](likelihood.md)** — the per-channel, gain-marginal
  matched-filter evidence: integrating out the per-channel burst amplitude
  (spectrum + scintillation) analytically under a proper finite-variance
  Gaussian prior, the closed-form $\ln Z_f$, the Occam term, and the
  eigenvalue conditioning guard.
- **[Model selection (the grid)](model-selection.md)** — why $\ln Z$ from
  different code paths is not comparable, the `force_multi` matched-normalization
  baseline, and the C1D1/C1D2/C2D1/C2D2 component grid.
- **[Case study: zach & the alpha bias](zach-case-study.md)** — the narrative
  spine: a residual-whiteness gate flags CHIME while a peak-finder sees the
  extra pulse in DSA, the grid resolves the paradox, and $\alpha$ shifts
  $3.32 \to 2.76$ once a hidden CHIME component is modeled.
- **[Adversarial verification](verification.md)** — the reusable protocol that
  tries to *break* a large $\Delta\ln Z$: independent GLS-gain reconstruction,
  an internal control reproducing the gate, pathology guards, and a
  same-metric before/after.

## The headline

For the worked burst (`zach`), adding a hidden second CHIME component is favored
by $\Delta\ln Z \approx +3.6\times10^{3}$ with no likelihood pathology, and the
shared scattering index moves from $\alpha = 3.32 \pm 0.01$ to
$\alpha = 2.76 \pm 0.02$ — about $40\times$ the formal error. Single-component
$\alpha$ is **biased high**. The corrected value is better-conditioned, not
final: even the two-component model does not fully whiten the residual.

!!! note "Provenance"
    These pages were drafted and adversarially verified against the source in
    `scattering/scat_analysis/burstfit_joint.py` and the
    `analysis/scattering-refit-2026-06/` campaign artifacts. Code references are
    given as `burstfit_joint.py:<line>`. Numerical results are from the verified
    `zach` refit run.
