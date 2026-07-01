# Dual-band FRB scattering report

##### [**Undermind**](https://undermind.ai)

---


## Table of Contents

- [Dual-band FRB scattering report](#dual-band-frb-scattering-report)
  - [Ranked open questions](#ranked-open-questions)
  - [What prior work establishes and what remains open](#what-prior-work-establishes-and-what-remains-open)
  - [Differentiation from prior strands of work](#differentiation-from-prior-strands-of-work)
  - [What N=12 can and cannot support](#what-n12-can-and-cannot-support)
  - [Recommended paper structure](#recommended-paper-structure)
    - [Main claim 1](#main-claim-1)
    - [Main claim 2](#main-claim-2)
    - [Main claim 3](#main-claim-3)
    - [Main claim 4](#main-claim-4)
    - [Main claim 5](#main-claim-5)
  - [Recommended narrative arc](#recommended-narrative-arc)
  - [Bottom-line recommendations](#bottom-line-recommendations)
  - [References](#references)

# Dual-band FRB scattering report

A 12-burst CHIME/FRB–DSA-110 co-detection sample is well matched to a **burst-by-burst inference paper**, not a population-cosmology paper. The distinctive asset is the same-burst frequency lever arm from 0.4–0.8 GHz to 1.4 GHz, which can make the scattering index $`\alpha`$ a measured quantity rather than an assumed one, while tying each scattering fit to a redshift-based DM budget and a mapped foreground-galaxy environment. Prior work already supports a default picture in which FRB scattering is usually host- or source-local, while foreground halos can add substantial DM with little scattering \[Cor21, Ock21, Pro19b, Sim20b, Lee23\]. The main open question is whether any bursts in a small but well-characterized dual-band localized sample genuinely force an intervening-halo screen, or whether the sample instead sharpens the host-dominated picture by ruling halo scattering out burst by burst \[Fab24, Sam23, Ock25\].

The literature also makes the main analysis risk clear. High-time-resolution FRB work repeatedly shows that unresolved sub-components can masquerade as scattering tails or bias fitted propagation parameters, especially when a wide frequency lever arm is available \[Day20, Hes18, San24, Fon23, Kum24\]. That makes component-aware dual-band fitting not just a technical preference but one of the central novelty claims available to this dataset.

## Ranked open questions

| Rank | Question | Why this sample is strong | Feasibility with N=12 |
|:---|:---|:---|:---|
| 1 | **Do any bursts require an intervening-halo scattering screen once $`\alpha`$ is measured and DM+scattering are modeled jointly?** | This is the sharpest falsification test enabled by the sample. Localized-FRB work mostly supports host-dominated scattering \[Cor21, Ock21, Ock22c\], while one DSA case shows that an intervening cloudlet can be plausible in an exceptional sightline \[Fab24\]. | **High** for case studies and upper limits; **low** for population incidence. |
| 2 | **What is the burst-by-burst distribution of measured $`\alpha`$ in localized dual-band FRBs?** | Most FRB work still leans on canonical or weakly constrained scattering scalings, while simultaneous wide-lever-arm fitting can expose shallow, variable, or non-Kolmogorov behavior \[Day20, Bha04, Gey16, T25\]. | **High** if joint fits are stable in most bursts. |
| 3 | **Can $`\tau`$ plus $`\Delta\nu`$ plus frequency scaling localize multi-screen systems?** | Two-screen localization exists, but only in a small number of bursts and only when scintillation information is combined with pulse broadening \[Sam23, Nim24, T25\]. | **Moderate** because only 3/12 currently have measured $`\Delta\nu`$. |
| 4 | **Can the sample strengthen the claim that halos often add DM but not scattering?** | Several localized sightlines already point this way \[Pro19b, Sim21b, Sim20b, Sim23, Lee23\], but the literature still lacks many dual-band burst-by-burst tests tying measured $`\alpha`$ to explicit DM decompositions. | **High** for burst-specific constraints; **low** for universal statements. |
| 5 | **How much do unresolved sub-components bias inferred $`\tau`$ and $`\alpha`$?** | Baseband and high-time-resolution work shows that profile complexity is common and can be absorbed into propagation parameters if not modeled \[Day20, San24, Fon23, Kum24\]. | **High** and publishable even if astrophysical conclusions are mixed. |
| 6 | **Can any burst constrain source size or immediate environment through scintillation?** | One burst already did this spectacularly via two scintillation scales \[Nim24\], and theory shows how host-side scintillation could separate compact from extended emission regions \[Kum23\]. | **Low to moderate** unless the $`\Delta\nu`$ and modulation-index measurements are unusually clean. |
| 7 | **Can the sample constrain CGM baryon fractions or halo-profile parameters?** | Large-sample and stacking papers show this needs many more FRBs \[Rav18, Con21, Wu22\]. | **Low**. Not a good main-paper target. |

## What prior work establishes and what remains open

| Topic | Already established | Still open for this sample |
|:---|:---|:---|
| **Default site of scattering** | The working default is that measurable FRB scattering is usually dominated by host-galaxy or source-local media rather than the diffuse IGM or ordinary halo gas \[Cor21, Ock21, Ock22c\]. ASKAP/CRAFT population work also found no clear $`\tau`$–DM trend, which argues against diffuse-path scattering as the main driver \[Qiu20\]. | Whether any dual-band localized bursts in this sample break that default strongly enough to require an intervening halo rather than a host-side screen. |
| **Halo DM versus halo scattering** | Foreground-halo studies of FRBs 180924, 181112, 190608, 20190520B, and other outlier sightlines support a decoupled picture in which halos can contribute appreciable DM without comparable pulse broadening \[Pro19b, Sim21b, Sim20b, Sim23, Lee23\]. Statistical CHIME work points the same way at larger scale \[Con21, Wu22\]. | Whether measured $`\alpha`$, dual-band $`\tau`$, and mapped foreground galaxies can turn that qualitative picture into a burst-by-burst falsification test. |
| **Intervening-halo scattering as a real phenomenon** | Theory allows rare, clumpy, or cloudlet-driven halo scattering \[Ved18, Jow23, Mas25, Ock25\]. The strongest observational precedent is FRB 20221219A, where a crowded DSA sightline made an intervening cloudlet plausible \[Fab24\]. | Whether a sample of 12 contains one or more similarly exceptional cases, and what evidence threshold should count as “required” rather than merely “allowed.” |
| **Measured $`\alpha`$ in FRBs** | FRB scattering-index measurements exist in localized and high-time-resolution studies, but the literature is still sparse and heterogeneous, with many analyses relying on narrow-band data, low time resolution, or assumed canonical scaling \[Day20, Qiu20, Cha21\]. Pulsar literature already shows that non-Kolmogorov or geometry-modified indices are plausible \[Bha04, Gey16, Cor00\]. | A coherent same-instrument-pair sample with joint fits across a $`1`$ GHz lever arm, tied to screen attribution rather than quoted as profile-fit byproducts. |
| **Two-screen localization** | $`\tau`$ alone does not localize the screen. Combined use of pulse broadening, scintillation bandwidth, and modulation behavior can localize or constrain multiple screens \[Sam23, Mai21, Mai22, Nim24, T25\]. | Whether the 3/12 bursts with $`\Delta\nu`$ can provide decisive localization, and whether upper limits for the other 9 can still rule out some screen geometries. |
| **Profile-complexity bias** | Microsecond-resolution CHIME and ASKAP work shows that many FRBs contain blended sub-components, drifting structure, or shoulders that can be misfit as scattering if the model is too simple \[Day20, Hes18, Far18, San24\]. Fitburst-style modeling was built around this exact problem \[Fon23\]. | How often a component-aware re-fit changes $`\alpha`$, $`\tau`$, or the inferred need for an intervening screen in this specific dual-band sample. |

## Differentiation from prior strands of work

| Prior strand | What it did | What this sample can do that is different |
|:---|:---|:---|
| **CRAFT scattering population work** | ASKAP/CRAFT population papers established that scattering exists in a subset of FRBs, that $`\tau`$ does not simply track extragalactic DM, and that host or foreground structures are more plausible than diffuse IGM scattering \[Qiu20\]. High-time-resolution localized ASKAP work also measured morphology and scattering parameters in individual bursts \[Day20\]. | The CHIME/DSA sample is not another population paper. Its leverage comes from **same-burst dual-band joint fitting** of $`\tau`$ and measured $`\alpha`$, plus redshifts and foreground environments. The main product should be screen attribution and model falsification, not another demographic statement. |
| **Cordes/Ocker DM-budget papers** | These papers built the framework in which halos and the IGM can contribute a lot of DM while scattering stays small, implying that host media dominate the observed pulse broadening in most localized FRBs \[Cor21, Ock21\]. The FRB 20190520B work showed how joint DM and scattering arguments can rescue a badly biased DM-only redshift inference \[Ock22c\]. | The new sample can turn that framework into an observational test based on **measured $`\alpha`$** rather than assumed scaling. It can ask whether the same bursts that need extra DM from foreground halos also need halo scattering, or whether the DM and scattering budgets remain cleanly decoupled. |
| **DSA multi-halo precedent** | The strongest current precedent for an intervening-halo scattering claim is the DSA burst FRB 20221219A, where a crowded sightline and unusually large $`\tau`$ made an intervening cloudlet plausible \[Fab24\]. | The CHIME/DSA sample can show whether that case is an exception or the first member of a broader class. The key difference is that a small multi-burst sample can compare exceptional and ordinary sightlines using the same fitting framework. |
| **Foreground-halo DM mapping papers** | FLIMFLAM-style work on individual FRBs showed that some DM outliers are foreground-heavy and others are host-heavy, without treating scattering as a primary observable \[Sim23, Lee23\]. | The new paper can join **foreground mapping** to **propagation modeling** rather than keeping them separate, which is one of the clearest ways to make the report distinct from prior DM-only studies. |
| **Scintillometry and two-screen case studies** | A handful of papers used scintillation scales, annual-velocity arguments, or dual scintillation screens to localize scattering structures \[Sam23, Mai21, Mai22, Nim24, Wu24, T25\]. | In this sample, the goal is narrower and more publishable: use those tools only where the data justify them, and avoid over-selling two-screen claims in bursts without robust $`\Delta\nu`$ or modulation information. |

## What N=12 can and cannot support

| Claim type | N=12 is enough | Needs N(\gt)12 |
|:---|:---|:---|
| **Burst-level measured $`\alpha`$** | Yes, if the fits are stable under sensible component models and instrumental checks. A sample of 12 is already useful because the claim is methodological and comparative rather than demographic \[Day20, Fon23, San24\]. | Large-sample statements about the population distribution of $`\alpha`$, repeaters versus apparent one-offs, or environment-dependent trends need more bursts. |
| **Burst-level host versus halo screen attribution** | Yes, for strong case studies, upper limits, and “required versus allowed” logic built from redshift, DM decomposition, and scattering fits \[Cor21, Ock21, Fab24\]. | Population incidence of intervening-halo scattering and the fraction of FRBs with multi-screen propagation need many more localized bursts. |
| **Testing “halos add DM but not scattering”** | Yes, as a falsification test at the level of individual bursts or a small sample summary. A result such as “none of 12 require halo scattering despite multiple foreground-rich sightlines” would already matter. | Quantitative statements about halo-scattering duty cycle, dependence on halo mass, or universal decoupling need substantially larger and more uniformly selected samples \[Wu22, Con21\]. |
| **Two-screen localization from $`\Delta\nu`$** | Yes, but only for the subset with robust $`\Delta\nu`$, modulation behavior, and enough spectral resolution \[Sam23, Nim24, T25\]. | Any systematic conclusion about two-screen incidence or the typical location of host screens needs more than 3 informative bursts. |
| **CGM baryon or mNFW profile constraints** | No. Even arcsecond-localized FRB studies need of order hundreds of bursts for strong CGM profile inference, and CHIME-like statistical detections need thousands \[Rav18, Wu22\]. | This belongs to future larger surveys or combined-sample work, not the 12-burst paper. |
| **Sub-component bias as a methods result** | Yes. Demonstrating that component-aware dual-band fitting materially shifts $`\alpha`$, $`\tau`$, or screen attribution in a subset of bursts is already a solid result \[Day20, San24, Fon23, Kum24\]. | Estimating the population frequency of such biases needs more bursts and more homogeneous baseband access. |

## Recommended paper structure

### Main claim 1

**Measured $`\alpha`$ changes the screen-attribution problem.**

The paper should lead with the claim that the CHIME/DSA frequency lever arm allows $`\alpha`$ to be measured rather than assumed, and that this materially changes which screen models remain viable. This is the main scientific lever that differentiates the work from narrow-band scattering studies and from DM-budget papers that do not fit scattering jointly \[Day20, Cor21, Ock21, T25\].

| Element | Recommendation |
|:---|:---|
| **Core evidence** | Joint dual-band fits of $`\tau`$ and $`\alpha`$ for all 12 bursts, with model-comparison results against fixed-$`\alpha=4`$ fits. |
| **Key figure** | A 12-row panel showing burst profiles or dynamic spectra in both bands, with posterior $`\tau`$ and $`\alpha`$ summaries beside each burst. |
| **Kill criterion** | If most bursts have such broad posteriors that measured $`\alpha`$ is indistinguishable from the assumed prior or from $`\alpha=4`$, this cannot be a headline claim. It then becomes a methods limitation rather than the lead result. |

### Main claim 2

**Joint DM+scattering modeling can test, burst by burst, whether any sightline requires intervening-halo scattering.**

This should be framed as a falsification exercise, not as a promise to discover many halo screens. The literature baseline is that halos often add DM while host media dominate scattering \[Cor21, Ock21, Sim23, Lee23\]. The clean publishable outcome is either one compelling exception or a stronger version of the default rule.

| Element | Recommendation |
|:---|:---|
| **Core evidence** | For each burst, a table of DM components, host-redshift context, foreground-halo candidates, fitted $`\tau`$, fitted $`\alpha`$, and favored screen class. |
| **Key figure** | A screen-attribution matrix: rows are bursts, columns are Milky Way, host/local, intervening halo, and mixed/two-screen models, with shaded model support. |
| **Kill criterion** | If every burst remains equally compatible with host and intervening interpretations after accounting for uncertainties, the paper should not claim screen localization. It should instead report upper limits and emphasize the stronger methods result on what the data can rule out. |

### Main claim 3

**The sample can directly test the claim that halos add DM without adding much scattering.**

This is best presented as a synthesis claim, not a standalone cosmology claim. The foreground-rich bursts are the critical subset. If they still prefer host-dominated scattering, the paper materially sharpens the literature consensus \[Pro19b, Sim20b, Sim21b, Sim23, Lee23\].

| Element | Recommendation |
|:---|:---|
| **Core evidence** | Comparison between foreground-rich and foreground-poor sightlines, with the same fitting machinery applied to both. |
| **Key figure** | A DM-budget versus scattering-budget diagram showing which bursts need foreground DM and which actually need extra scattering. |
| **Kill criterion** | If the foreground mapping is too incomplete or too uncertain for most bursts, this should be narrowed to a case-study result rather than presented as a sample-wide test. |

### Main claim 4

**Component-aware fitting is necessary because intrinsic profile complexity can bias $`\alpha`$ and $`\tau`$.**

This is the safety rail for the whole paper. High-time-resolution FRB studies make clear that the wrong profile model can create false propagation inferences \[Day20, Hes18, San24, Fon23, Kum24\]. The paper should show this explicitly rather than treating it as a caveat in the discussion.

| Element | Recommendation |
|:---|:---|
| **Core evidence** | A controlled re-fit of the most complex bursts under single-component and multicomponent models, comparing inferred $`\tau`$, $`\alpha`$, and screen attribution. |
| **Key figure** | Before-and-after residual plots or posterior shifts for a few representative complex bursts. |
| **Kill criterion** | If component-aware re-fits do not materially move any key propagation parameters, then this cannot be sold as a major methods contribution. It should remain a robustness check. |

### Main claim 5

**Two-screen localization is a subset result, not the organizing frame of the full paper.**

The literature supports strong two-screen claims only when scintillation information is good enough \[Sam23, Nim24, T25\]. With 3/12 bursts having measured $`\Delta\nu`$, those cases can be strong supporting sections or appendices, but they should not carry the whole paper.

| Element | Recommendation |
|:---|:---|
| **Core evidence** | The subset with measured $`\Delta\nu`$, including $`\tau`$-$`\Delta\nu`$ consistency, modulation behavior, and frequency scaling. |
| **Key figure** | A dedicated two-screen figure showing $`2\pi\tau\Delta\nu`$ consistency and allowed screen-distance regions for the informative bursts. |
| **Kill criterion** | If $`\Delta\nu`$ measurements are too uncertain or the implied geometry depends entirely on untestable screen assumptions, move this material to a secondary section or appendix. |

## Recommended narrative arc

The strongest narrative is not “FRBs probe the CGM” in the broad sense. That story already belongs to larger statistical samples and stacking papers \[Con21, Wu22\]. The strongest narrative is narrower: a dual-band localized sample can turn scattering from a loosely modeled nuisance parameter into a screen-attribution observable. That produces three publishable outcomes even in a small sample.

- **Outcome A:** one or more exceptional bursts that genuinely require an intervening-halo screen, extending the current single-burst DSA precedent \[Fab24\].
- **Outcome B:** no such exceptions, which strengthens the host-dominated interpretation and the DM–scattering decoupling framework \[Cor21, Ock21, Lee23\].
- **Outcome C:** a methods result showing that component-aware dual-band fitting is necessary for credible $`\alpha`$-based inference \[Day20, San24, Fon23\].

Of these, **Outcome B plus Outcome C** is the most likely high-quality paper if the sample behaves like the current literature baseline. **Outcome A** is the higher-risk, higher-reward possibility.

## Bottom-line recommendations

| Priority | Recommendation | Reason |
|:---|:---|:---|
| 1 | **Lead with measured $`\alpha`$ and joint screen attribution.** | This is the real differentiator from both CRAFT population work and DM-only halo mapping \[Qiu20, Sim23\]. |
| 2 | **Frame intervening-halo scattering as an exception test, not an expected discovery rate.** | The literature baseline favors rare halo-scattering outliers rather than common ones \[Ock21, Fab24, Ock25, Mas25\]. |
| 3 | **Make component-aware fitting central, not peripheral.** | Without it, the strongest claims are vulnerable to profile-model bias \[Day20, San24, Fon23, Kum24\]. |
| 4 | **Keep CGM cosmology claims modest.** | N=12 can support burst-specific inference, not robust halo-population or baryon-profile constraints \[Rav18, Wu22\]. |
| 5 | **Treat scintillation-based two-screen localization as a high-value subset analysis.** | It can be decisive in a few bursts, but the current sample is too small and too incomplete in $`\Delta\nu`$ for it to define the whole paper \[Sam23, Nim24, T25\]. |

---

## References

\[Cor21\] J. Cordes, S. Ocker, and S. Chatterjee, “Redshift Estimation and Constraints on Intergalactic and Interstellar Media from Dispersion and Scattering of Fast Radio Bursts,” Aug. 02, 2021. doi: [10.3847/1538-4357/ac6873](https://doi.org/10.3847/1538-4357/ac6873).

\[Ock21\] S. Ocker, J. Cordes, and S. Chatterjee, “Constraining Galaxy Halos from the Dispersion and Scattering of Fast Radio Bursts and Pulsars,” Jan. 12, 2021. doi: [10.3847/1538-4357/abeb6e](https://doi.org/10.3847/1538-4357/abeb6e).

\[Pro19b\] J. Prochaska *et al.*, “The low density and magnetization of a massive galaxy halo exposed by a fast radio burst,” *Science*, vol. 366, pp. 231–234, Sep. 2019, doi: [10.1126/science.aay0073](https://doi.org/10.1126/science.aay0073).

\[Sim20b\] S. Simha *et al.*, “Disentangling the Cosmic Web toward FRB 190608,” *The Astrophysical Journal*, vol. 901, May 2020, doi: [10.3847/1538-4357/abafc3](https://doi.org/10.3847/1538-4357/abafc3).

\[Lee23\] K. Lee *et al.*, “The FRB 20190520B Sight Line Intersects Foreground Galaxy Clusters,” *The Astrophysical Journal Letters*, vol. 954, Jun. 2023, doi: [10.3847/2041-8213/acefb5](https://doi.org/10.3847/2041-8213/acefb5).

\[Fab24\] J. T. Faber *et al.*, “A Heavily Scattered Fast Radio Burst Is Viewed Through Multiple Galaxy Halos,” May 23, 2024.

\[Sam23\] M. Sammons *et al.*, “Two-Screen Scattering in CRAFT FRBs,” May 19, 2023.

\[Ock25\] S. Ocker, M. C. Chen, S. P. Oh, and P. Sharma, “Microphysics of Circumgalactic Turbulence Probed by Fast Radio Bursts and Quasars,” Mar. 04, 2025. doi: [10.3847/1538-4357/ade0bc](https://doi.org/10.3847/1538-4357/ade0bc).

\[Day20\] C. Day *et al.*, “High time resolution and polarization properties of ASKAP-localized fast radio bursts,” May 27, 2020. doi: [10.1093/MNRAS/STAA2138](https://doi.org/10.1093/MNRAS/STAA2138).

\[Hes18\] J. Hessels *et al.*, “FRB 121102 Bursts Show Complex Time–Frequency Structure,” Nov. 26, 2018. doi: [10.3847/2041-8213/ab13ae](https://doi.org/10.3847/2041-8213/ab13ae).

\[San24\] K. R. Sand *et al.*, “Morphology of 137 Fast Radio Bursts Down to Microsecond Timescales from the First CHIME/FRB Baseband Catalog,” *The Astrophysical Journal*, vol. 979, Aug. 2024, doi: [10.3847/1538-4357/ad9b11](https://doi.org/10.3847/1538-4357/ad9b11).

\[Fon23\] E. Fonseca *et al.*, “Modeling the Morphology of Fast Radio Bursts and Radio Pulsars with fitburst,” *The Astrophysical Journal Supplement Series*, vol. 271, Nov. 2023, doi: [10.3847/1538-4365/ad27d6](https://doi.org/10.3847/1538-4365/ad27d6).

\[Kum24\] A. Kumar, F. Rajabi, and M. Houde, “Impact of Propagation Effects on the Spectro-temporal Properties of Fast Radio Bursts,” *The Astrophysical Journal*, vol. 998, Nov. 2024, doi: [10.3847/1538-4357/ae3145](https://doi.org/10.3847/1538-4357/ae3145).

\[Ock22c\] S. K. Ocker *et al.*, “The Large Dispersion and Scattering of FRB 20190520B Are Dominated by the Host Galaxy,” Feb. 27, 2022. doi: [10.3847/1538-4357/ac6504](https://doi.org/10.3847/1538-4357/ac6504).

\[Bha04\] N. Bhat, J. Cordes, F. Camilo, D. Nice, and D. Lorimer, “Multifrequency Observations of Radio Pulse Broadening and Constraints on Interstellar Electron Density Microstructure,” Jan. 07, 2004. doi: [10.1086/382680](https://doi.org/10.1086/382680).

\[Gey16\] M. Geyer and A. Karastergiou, “The frequency dependence of scattering imprints on pulsar observations,” Jul. 18, 2016. doi: [10.1093/mnras/stw1724](https://doi.org/10.1093/mnras/stw1724).

\[T25\] S. T., T. Sprenger, O. Wucknitz, R. A. Main, and L. Spitler, “Scintillometry of fast radio bursts. Resolution effects in two-screen models,” *Astronomy &amp; Astrophysics*, May 2025, doi: [10.1051/0004-6361/202554202](https://doi.org/10.1051/0004-6361/202554202).

\[Nim24\] K. Nimmo *et al.*, “Magnetospheric origin of a fast radio burst constrained using scintillation,” Jun. 16, 2024.

\[Sim21b\] S. Simha *et al.*, “Estimating the Contribution of Foreground Halos to the FRB 180924 Dispersion Measure,” Aug. 23, 2021. doi: [10.3847/1538-4357/ac2000](https://doi.org/10.3847/1538-4357/ac2000).

\[Sim23\] S. Simha *et al.*, “Searching for the Sources of Excess Extragalactic Dispersion of FRBs,” *The Astrophysical Journal*, vol. 954, Mar. 2023, doi: [10.3847/1538-4357/ace324](https://doi.org/10.3847/1538-4357/ace324).

\[Kum23\] P. Kumar, P. Beniamini, O. Gupta, and J. Cordes, “Constraining the FRB mechanism from scintillation in the host galaxy,” *Monthly Notices of the Royal Astronomical Society*, Jul. 2023, doi: [10.1093/mnras/stad3010](https://doi.org/10.1093/mnras/stad3010).

\[Rav18\] V. Ravi, “Measuring the Circumgalactic and Intergalactic Baryon Contents with Fast Radio Bursts,” Apr. 19, 2018. doi: [10.3847/1538-4357/aafb30](https://doi.org/10.3847/1538-4357/aafb30).

\[Con21\] L. Connor and V. Ravi, “The observed impact of galaxy halo gas on fast radio bursts,” Jul. 29, 2021. doi: [10.1038/s41550-022-01719-7](https://doi.org/10.1038/s41550-022-01719-7).

\[Wu22\] X. Wu and M. McQuinn, “A Measurement of Circumgalactic Gas around Nearby Galaxies Using Fast Radio Bursts,” Sep. 09, 2022. doi: [10.3847/1538-4357/acbc7d](https://doi.org/10.3847/1538-4357/acbc7d).

\[Qiu20\] H. Qiu *et al.*, “A population analysis of pulse broadening in ASKAP fast radio bursts,” Jun. 30, 2020. doi: [10.1093/mnras/staa1916](https://doi.org/10.1093/mnras/staa1916).

\[Ved18\] H. Vedantham, H. Vedantham, and E. Phinney, “Radio wave scattering by circumgalactic cool gas clumps,” Nov. 26, 2018. doi: [10.1093/mnras/sty2948](https://doi.org/10.1093/mnras/sty2948).

\[Jow23\] D. Jow, X. Wu, and U.-L. Pen, “Refractive lensing of scintillating FRBs by subparsec cloudlets in the multiphase CGM,” *Proceedings of the National Academy of Sciences of the United States of America*, vol. 121, Sep. 2023, doi: [10.1073/pnas.2406783121](https://doi.org/10.1073/pnas.2406783121).

\[Mas25\] L. Mas-Ribas, M. McQuinn, and J. Prochaska, “Circumgalactic Medium Cloud Sizes from Refractive Fast Radio Burst Scattering,” *The Astrophysical Journal*, vol. 990, Apr. 2025, doi: [10.3847/1538-4357/adf43b](https://doi.org/10.3847/1538-4357/adf43b).

\[Cha21\] P. Chawla *et al.*, “Modeling Fast Radio Burst Dispersion and Scattering Properties in the First CHIME/FRB Catalog,” Jul. 22, 2021. doi: [10.3847/1538-4357/ac49e1](https://doi.org/10.3847/1538-4357/ac49e1).

\[Cor00\] J. Cordes, T. Dept, U. Cornell, and N. Nrl, “Anomalous Radio-Wave Scattering from Interstellar Plasma Structures,” May 24, 2000. doi: [10.1086/319442](https://doi.org/10.1086/319442).

\[Mai21\] R. Main *et al.*, “Scintillation timescale measurement of the highly active FRB20201124A,” Jul. 30, 2021. doi: [10.1093/mnras/stab3218](https://doi.org/10.1093/mnras/stab3218).

\[Mai22\] R. Main *et al.*, “Modelling annual scintillation velocity variations of FRB 20201124A,” Dec. 09, 2022. doi: [10.1093/mnrasl/slad036](https://doi.org/10.1093/mnrasl/slad036).

\[Far18\] W. Farah *et al.*, “FRB microstructure revealed by the real-time detection of FRB170827,” Mar. 15, 2018. doi: [10.1093/mnras/sty1122](https://doi.org/10.1093/mnras/sty1122).

\[Wu24\] Z. Wu *et al.*, “Scintillation Velocity and Arc Observations of FRB 20201124A,” *The Astrophysical Journal Letters*, vol. 969, Jun. 2024, doi: [10.3847/2041-8213/ad5979](https://doi.org/10.3847/2041-8213/ad5979).
