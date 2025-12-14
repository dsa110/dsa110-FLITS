\title{
MULTIFREQUENCY OBSERVATIONS OF RADIO PULSE BROADENING AND CONSTRAINTS ON INTERSTELLAR ELECTRON DENSITY MICROSTRUCTURE
}

\author{
N. D. Ramesh Bhat \\ Massachusetts Institute of Technology, Haystack Observatory, Westford, MA 01886 \\ rbhat@haystack.mit.edu \\ James M. Cordes \\ Astronomy Department and NAIC, Cornell University, Ithaca, NY 14853 \\ cordes@astro.cornell.edu \\ Fernando Camilo \\ Columbia Astrophysics Laboratory, Columbia University, 550 West 120th Street, New York, NY 10027 \\ fernando@astro.columbia.edu \\ David J. Nice \\ Department of Physics, Princeton University, Box 708, Princeton, NJ 08544 \\ dnice@princeton.edu \\ Duncan R. Lorimer \\ University of Manchester, Jodrell Bank Observatory, Macclesfield, Cheshire, SK11 9DL, UK \\ dr1@jb.man.ac.uk
}

Draft version April 23, 2018

\begin{abstract}
We have made observations of 98 low-Galactic-latitude pulsars to measure pulse broadening caused by multipath propagation through the interstellar medium. Data were collected with the \(305-\mathrm{m}\) Arecibo telescope at four radio frequencies between 430 and 2380 MHz . We used a CLEAN-based algorithm to deconvolve interstellar pulse broadening from the measured pulse shapes. We employed two distinct pulse broadening functions (PBFs): \(\mathrm{PBF}_{1}\) is appropriate for a thin screen of scattering material between the Earth and a pulsar, while \(\mathrm{PBF}_{2}\) is appropriate for scattering material uniformly distributed along the line of sight from the Earth to a pulsar. We found that some observations were better fit by \(\mathrm{PBF}_{1}\) and some by \(\mathrm{PBF}_{2}\). Pulse broadening times \(\left(\tau_{d}\right)\) are derived from fits of PBFs to the data, and are compared with the predictions of a smoothed model of the Galactic electron distribution. Several lines of sight show excess broadening, which we model as clumps of high density scattering material. A global analysis of all available data finds that the pulse broadening scales with frequency, \(\nu\), as \(\tau_{d} \propto \nu^{-\alpha}\) where \(\alpha \sim 3.9 \pm 0.2\). This is somewhat shallower than the value \(\alpha=4.4\) expected from a Kolmogorov medium, but could arise if the spectrum of turbulence has an inner cutoff at \(\sim 300-800 \mathrm{~km}\). A few objects follow particularly shallow scaling laws (the mean scaling index \(\langle\alpha\rangle \sim 3.1 \pm 0.1\) and \(\sim 3.8 \pm 0.2\) respectively for the case of \(\mathrm{PBF}_{1}\) and \(\mathrm{PBF}_{2}\) ), which may arise from large scale refraction or from the truncation of scattering screens transverse to the Earth-pulsar line of sight.
\end{abstract}

Subject headings: ISM: structure - methods: data analysis - pulsars: general - radio continuum: general - scattering

\section*{1. INTRODUCTION}

\subsection*{1.1. Overview}

Pulsars make excellent probes of the interstellar medium (ISM). Observed pulse profiles are influenced by dispersion, scattering, and Faraday rotation along the line-ofsight (LOS) from the Earth to the pulsar. Measurements of pulsars in similar directions at different distances can be used to disentangle LOS interstellar effects and to model the ionized content of the ISM (Taylor \& Cordes 1993; Bhat \& Gupta 2002; Cordes \& Lazio 2002a,b).

We have undertaken multifrequency pulse profile observations using the \(305-\mathrm{m}\) Arecibo telescope, concentrating in the swath of the Galactic plane visible from Arecibo, at Galactic longitudes \(30^{\circ} \leq l \leq 75^{\circ}\). The Parkes Multibeam Survey (e.g. Manchester et al. 2001) has discovered hundreds of pulsars at low Galactic latitudes, \(|b|<5^{\circ}\), of which dozens are visible from Arecibo. Many other pulsars
in this region are known from other survey work (e.g. Hulse \& Taylor 1975b). Because the sensitive Multibeam Survey employed a higher frequency, 1400 MHz , than most other surveys, the "multibeam pulsars" tend to be relatively distant and highly scattered, making them particularly useful for ISM studies.

Our most fundamental measurements are the set of pulse shapes at different radio frequencies, from which we estimate the pulse broadening time scales caused by scattering, \(\tau_{d}\), for the pulsars. In addition to providing input data to Galactic electron density models, these measurements can be used to form an empirical relation connecting \(\tau_{d}\) with dispersion measure, which can serve as a useful guide in designing large-scale pulsar surveys and in understanding the observable population of pulsars in the Galaxy (e.g. Bhattacharya et al. 1992; Cordes \& Lazio 2002b).

The paper is organized as follows. Terminology and basic assumptions about the ISM are summarized in § 1.2. Details of observations and data reduction are described in § 2, and our method for deconvolving pulse broadening in § 3. Our results are presented in § 4 and § 5, and in later sections we discuss the implications of pulse-broadening times for the Galactic electron density models (§ 6), as well as the power spectrum of electron density irregularities (§ 7).

\subsection*{1.2. Terminology and Scattering Model}

Several quantities measured by radio observations of a pulsar are integrals of ISM properties along the LOS from the Earth to the pulsar. The dispersion measure, \(\mathrm{DM} \equiv \int_{0}^{D} d s n_{e}(s)\), is the integral of electron density, \(n_{e}\), along the LOS to the pulsar at distance \(D\). Using a Galactic model for electron density, DM is often used to estimate pulsar distances. We use cataloged values of DM in the analysis below to estimate distances. The rotation measure, \(\mathrm{RM} \equiv \int_{0}^{D} n_{e}(s) \mathbf{B} \cdot \mathbf{d s}\), is the LOS integral of magnetic field, \(\mathbf{B}\), weighted by electron density. Analysis of RM measurements from our data will be reported in a future work.

Scattering of pulsar signals depends on fluctuations in the electron density, \(\delta n_{e}\). We assume that the spectral density of these fluctuations follows a power-law model with cutoffs at "inner" and "outer" scales, \(l_{i}\) and \(l_{o}\), which are inversely related to the corresponding wavenumbers, \(\kappa_{i}\) and \(\kappa_{o}\), by \(l_{i}=2 \pi / \kappa_{i}\) and \(l_{o}=2 \pi / \kappa_{o}\). The spectral density is then given by (e.g. Rickett 1977):
\[
P_{n_{e}}(\kappa)= \begin{cases}C_{n}^{2} \kappa^{-\beta}, & \kappa_{o} \leq \kappa \leq \kappa_{i} \\ 0, & \text { elsewhere }\end{cases}
\]

The spectral coefficient \(C_{n}^{2}\) is expressed in units of \(\mathrm{m}^{-20 / 3}\). For Kolmogorov turbulence, the spectral slope is \(\beta=11 / 3\).

Pulse broadening is quantified by a time scale, \(\tau_{d}\), characteristic of a pulse broadening function (PBF) fit to a measured pulse shape. The PBF is the response of the ISM to a delta function. The exact form of the PBF and its scaling with frequency depend on the spatial distribution of scattering material along the LOS and on its wavenumber spectrum (Williamson 1972, 1973; Cordes \& Rickett 1998; Lambert \& Rickett 1999; Cordes \& Lazio 2001; Boldyrev \& Gwinn 2003). Therefore, determination of the PBF forms a useful means for characterizing the underlying scattering geometry and wavenumber spectrum for scattering irregularities. The PBFs used in this work are described in detail in §3.

Measured pulse scattering parameters can be related to the scattering measure, \(\mathrm{SM} \equiv \int_{0}^{D} d s C_{n}^{2}(s)\), which is the LOS integral of \(C_{n}^{2}\). For a Kolmogorov spectrum with a small inner scale (e.g. Rickett 1990; Cordes \& Lazio 1991; Armstrong, Rickett \& Spangler 1995), the pulse broadening, expressed as the mean arrival time of ray bundles (see Cordes \& Rickett 1998), is
\[
\left\langle\tau_{d}\right\rangle \approx 1.1 W_{\tau} \mathrm{SM}^{6 / 5} \nu^{-4.4} D
\]
where \(\nu\) is in \(\mathrm{GHz}, D\) is in \(\mathrm{kpc}, \mathrm{SM}\) is in \(\mathrm{kpc} \mathrm{m}^{-20 / 3}, \tau_{d}\) is in ms , and \(W_{\tau}\) is a geometric factor that depends on the LOS distribution of scattering material.

More generally, for a power-law wavenumber spectrum, the broadening time scale follows a power law,
\[
\tau_{d} \propto \nu^{-\alpha}
\]
where (e.g. Cordes, Pidwerbetsky, \& Lovelace 1986; Romani, Narayan, \& Blandford 1986),
\[
\alpha= \begin{cases}\frac{2 \beta}{(\beta-2)} & \beta<4 \\ \frac{8}{(6-\beta)} & \beta>4\end{cases}
\]

Thus, determination of \(\alpha\) yields information about the wavenumber spectrum. For a Kolmogorov spectrum, \(\beta= 11 / 3\), implying \(\alpha=4.4\), and Eq. 3 reduces to 2. This result holds if the inner scale of the spectrum is too small to influence the measurements (Cordes \& Lazio 2002). As we discuss later, we infer that the inner scale likely does influence some of the scattering measurements.

Finally, the decorrelation bandwidth, \(\nu_{d}\), is related to \(\tau_{d}\) by \(2 \pi \tau_{d} \nu_{d}=C_{1}\), where the constant \(C_{1}\), of order unity, depends on the geometry of the scattering material and the form of the wavenumber spectrum (Cordes \& Rickett 1998).

\section*{2. OBSERVATIONS AND DATA REDUCTION}

The observations were made at the Arecibo Observatory. New data for 81 pulsars were obtained in several observing sessions from 2001 May to 2002 November. For the analysis in this paper we also use the data collected by Lorimer et al. (2002) for 17 pulsars, yielding a total of 98 pulsars. We concentrated on pulsars for which pulse broadening observations had not previously been made. Prominent among these are 38 discovered in the Parkes Multibeam Survey (e.g. Manchester et al. 2001), 30 from the Hulse-Taylor survey, including 17 with new timing solutions (Lorimer et al. 2002), and 30 others (Taylor, Manchester, \& Lyne 1993; Hobbs \& Manchester 2003)

Data acquisition systems used for the observations are summarized in Table 1. Signals were collected separately at four radio frequencies, \(430,1175,1475\) and 2380 MHz . The range of frequencies was chosen to allow detection of pulse broadening over a wide variety of pulsar scattering measures; specific frequencies were chosen according to receiver availability and radio frequency interference environment. The strong dependence of \(\tau_{d}\) on frequency implies that, for most objects, pulse broadening will be measurable at only a subset of the four frequencies. For pulsars with little scattering, pulse broadening is detectable only at the lowest frequency, if at all. By contrast, for pulsars with heavy scattering, broadening may be measurable at high frequencies and may be so large as to render pulsations undetectable at lower frequencies. When pulsations are undetectable at 430 MHz , the cause may also involve a combination of relatively small flux density and large background temperature.

In the absence of any prior knowledge of flux density at the higher frequencies, we adopted fixed integration times for all objects in a first pass of observations. Based on the initial results, one or more re-observations were made during later sessions for those objects and frequencies with low signal-to-noise ratios.

Observations at 430 MHz were made with the Penn State Pulsar Machine (PSPM), an analog filterbank spectrometer providing 128 spectral channels spanning an 8 MHz band in each of two circularly polarized signals. Power measurements in each channel were synchronously averaged in real time at the topocentric pulse period, yielding pulse profiles with time resolution of approximately 1
milliperiod. Dedispersion was done off line, reducing each observation to a single pulse profile.

Observations at 1175,1475 , and 2380 MHz were made with the Wideband Arecibo Pulsar Processor (WAPP), a fast-dump digital correlator (Dowd, Sisk, \& Hagen 2000). Input signals to the WAPP were digitized into three levels and output correlations were accumulated and written to disk as 16 -bit integers. We recorded long time series of auto-correlation functions (ACFs) and cross-correlation functions (CCFs) of the two circularly-polarized polarization channels. The ACFs were used in the analysis of this paper. A polarization analysis that utilizes both the ACFs and CCFs will be reported in a subsequent paper.

In off-line analysis, the ACFs were van Vleck corrected (e.g. Hagen \& Farley 1973), Fourier transformed, dedispersed, and synchronously averaged to form average pulse profiles with, typically, 1 milliperiod resolution. The software tools used to analyze the WAPP and PSPM data are described by Lorimer (2001).

Figure 1 shows the pulse profiles obtained from our multiple frequency data. Profiles for 5 pulsars are not shown due to poor data quality. For the 37 multibeam pulsars shown, these represent the first observations at frequencies other than 1400 MHz , and, in almost all cases, signal-tonoise ratios for the profiles are superior to those obtained in the original multibeam survey data. For nearly all 39 previously known pulsars in the sample shown here, the profiles are the best quality profiles obtained to date.

\section*{3. DECONVOLUTION METHOD}

We used a CLEAN-based method (Bhat et al. 2003) for deconvolving scattering-induced pulse-broadening from the measured pulse shapes. This method does not rely on a priori knowledge of the pulse shape, and it can recover details of the pulse shape on time scales smaller than the width of the PBF. A number of trial PBFs may be used, with varying shapes and broadening times, corresponding to different LOS distributions of scattering material. The "best fit" PBF and broadening time are determined by a set of figures of merit, defined in terms of positivity and symmetry of the final CLEANed pulse, along with the mean and rms of the residual off-pulse regions. Details of the method and tests of its accuracy are given in Bhat et al. (2003).

We used two trial PBFs. The first, \(\mathrm{PBF}_{1}\), is appropriate for a thin slab scattering screen of infinite transverse extent within which density irregularities follow a square-law structure function \({ }^{1}\) (Lambert \& Rickett 2000). The PBF is given by a one-sided exponential (Williamson 1972, 1973),
\[
\operatorname{PBF}_{1}(t)=\tau_{d}^{-1} \exp \left(-t / \tau_{d}\right) U(t)
\]
where \(U(t)\) is the unit step function, \(U(<0)=0, U(\geq 0)=1\). This function has been commonly used in previous pulsar scattering work.

The second broadening function, \(\mathrm{PBF}_{2}\), corresponds to a uniformly distributed medium with a square-law structure function. This PBF has a finite rise time and slower decay,
\[
\operatorname{PBF}_{2}(t)=\left(\pi^{5} \tau_{d}{ }^{3} / 4 t^{5}\right)^{1 / 2} \exp \left(-\pi^{2} \tau_{d} / 4 t\right) U(t)
\]

1
The spatial structure function \(\mathrm{D}_{\mathrm{F}}(\mathrm{s})\) of a quantity \(\mathrm{F}(\mathrm{x})\) is defined as \(\mathrm{D}_{\mathrm{F}}(\mathrm{s})=\left\langle(\mathrm{F}(\mathrm{x}+\mathrm{s})-\mathrm{F}(\mathrm{x}))^{2}\right\rangle\) where s is the spatial separation (lag value).

This PBF is a generic proxy for more realistic distributions of scattering material.

Additional PBFs, not used in our analysis, include those for media with Kolmogorov wavenumber spectra, which can yield non-square-law structure functions (e.g. Lambert \& Rickett 1999), and scattering screens that are truncated in directions transverse to the LOS, as may be the case for filamentary or sheet-like structures, which have PBFs that correspondingly are truncated at large time scales (Cordes \& Lazio 2001).

Note that the pulse broadening time, \(\tau_{d}\), has different meanings for \(\mathrm{PBF}_{1}\) and \(\mathrm{PBF}_{2}\). For \(\mathrm{PBF}_{1}, \tau_{d}\) is both the \(e^{-1}\) point of the distribution and the expectation value of \(t\). For \(\mathrm{PBF}_{2}, \tau_{d}\) is close to the maximum of the distribution, which is at \(\left(\pi^{2} / 10\right) \tau_{d}=0.99 \tau_{d}\), while the expectation value of \(t\) is \(\left(\pi^{2} / 2\right) \tau_{d}=4.93 \tau_{d}\).

For some of the pulsars we obtained an acceptable fit to \(\tau_{d}\) using both \(\mathrm{PBF}_{1}\) and \(\mathrm{PBF}_{2}\), while in others only one of the PBFs provided an acceptable fit. Acceptable fits were those that yielded deconvolved pulse shapes that were positive, semi-definite; we reject cases which yielded unphysical pulse shapes (such as profiles with negative going components). In many cases the pulse broadening is not large enough to be measured, in which case we quote upper limits on \(\tau_{d}\) (see Table 2).

As noted earlier in this section, our method relies on a set of figures of merit for the determination of the best fit PBF for a given choice of the PBF form (see Bhat et al. (2003) for details). Among the different parameters used for this determination, the parameter \(f_{r}\) is a measure of positivity, and can serve as a useful indicator of "goodness" of the CLEAN subtraction. However we emphasize that the absolute value of this parameter may also depend on the degree of scattering, the noise in the data, shape of the intrinsic pulse, etc., and therefore a comparison of the results for different data-sets will not be meaningful. Nonetheless it can still be used for a relative comparison of the results obtained using different PBFs for a given pulse profile. For successfully deconvolved pulses, we expect \(f_{r} \lesssim 1\); larger values imply slightly overCLEANed pulses. Based on this approach, the PBF with a lower value of \(f_{r}\) can be considered to be the better of the two PBFs.

\section*{4. DERIVED INTRINSIC PULSE SHAPES}

Figures 2 and 3 show results from the CLEAN-based deconvolution of our data. In each panel, the best-fit PBF is shown along with the observed pulse shape and the deconvolved (intrinsic) pulse shape. As is evident in the figures, the derived pulse shapes are much narrower and significantly larger in amplitude than the observed ones. In several cases, the deconvolved pulse shapes reveal significant structure which is not easily visible in the measured pulse profiles. For example, PSRs J1913+0832 (Fig. 2) and J1858+0215 (Fig. 3) at 1175 MHz , have derived pulse shapes that are distinct doubles, a property that is almost entirely masked by broadening in the raw profiles. In several other cases (e.g. PSR J1912+0828 at 1175 MHz , Fig. 2; PSR J1927+1852 at 430 MHz , Fig. 3), the measured pulse shapes show faint signatures of a double, which are confirmed and reinforced by the deconvolution process. Data for PSR J1906+0641 at 1175 MHz (Fig. 2) and PSR J1942+1743 at 430 MHz (Fig. 3) show that the
technique yields details of complex, multi-component pulse shapes.

While the deconvolution algorithm usually produces accurate pulsar profiles, some cautions are in order. As we already discussed, for many objects successful deconvolution is possible using both PBF forms. Figures 2 and 3 include several examples of this kind. In some cases, significantly different intrinsic pulse shapes result from deconvolution with the two different PBFs. The data for PSR J1852+0031 show an extreme example of this: deconvolution with \(\mathrm{PBF}_{1}\) yields a pulse shape with three merged components (Fig. 2), while use of \(\mathrm{PBF}_{2}\) yields a distinct double pulse shape (Fig. 3). Further examples of substantial discrepancies include PSR J1858+0215 at 1175 MHz , where use of \(\mathrm{PBF}_{2}\) yields a double pulse, while use of \(\mathrm{PBF}_{1}\) yields a simpler, single-peaked pulse profile, and PSR J1853+0545 at 1175 MHz , where use of \(\mathrm{PBF}_{2}\) yields a much narrower and more featureless profile than \(\mathrm{PBF}_{1}\). However, such cases are exceptions rather than the rule. There are many examples where nearly identical intrinsic pulse shapes result with either of the two PBFs. Data from PSRs J1905+0616 and J1907+0740 at 430 MHz , and J1908+0839 and J1916+1030 at 1175 MHz , belong to this category.

We emphasize that we have used only two extreme examples from the infinite set of possible PBFs, and there may be LOSs for which other PBFs would be more appropriate. An exhaustive analysis using additional PBFs is beyond the scope of this paper, though such an analysis may be valuable in identifying important aspects of the ionized interstellar medium. In the remainder of the paper, we focus on measurements of pulse-broadening times and their implications for models of Galactic free electron density.

\section*{5. PULSE-BROADENING TIMES}

Our estimates of \(\tau_{d}\) are summarized in Table 2. The columns are: (1) pulsar name, (2) reference, (3) pulse period, (4) DM, (5) Galactic longitude, (6) Galactic latitude, (7) observation frequency, (8) estimate of \(\tau_{d}\) using \(\mathrm{PBF}_{1}\), and (9) its figure of merit ( \(f_{r}\) ), (10) estimate of \(\tau_{d}\) using \(\mathrm{PBF}_{2}\), and (11) its figure of merit \(\left(f_{r}\right)\), (12) model estimate of pulse broadening using the TC93 model ( \(\tau_{d, t c 93}\) ), and (13) model estimate of pulse broadening using the NE2001 model ( \(\tau_{d, n e 2001}\) ). The definitions of model estimates are discussed below along with a comparison with measured values of \(\tau_{d}\). We successfully measured \(\tau_{d}\) for 56 of the 98 target objects, (of which 15 have measurements at more than one frequency), and obtained upper limits on \(\tau_{d}\) for 31 objects.

\subsection*{5.1. Scaling of \(\tau_{d}\) with Frequency}

For 15 pulsars, we have measured \(\tau_{d}\) at more than one frequency, typically at 1175 and 1475 MHz , but in one case, PSR J1853+0545, also at 2380 MHz . For 12 of these, estimates of \(\tau_{d}\) were possible using both \(\mathrm{PBF}_{1}\) and \(\mathrm{PBF}_{2}\) (Table 2), and we derive the estimates of the scaling index \(\alpha\) in both cases (columns 4 and 6 in Table 3). We use \(\alpha_{1}\) and \(\alpha_{2}\) to denote the scaling indices for the two PBF cases, \(\mathrm{PBF}_{1}\) and \(\mathrm{PBF}_{2}\), respectively, and the corresponding values for \(\beta\) (obtained by use of Eq. 4) are denoted as \(\beta_{1}\) and \(\beta_{2}\). For PSR J1853+0545, measurements of \(\tau_{d}\) are
available for 3 frequencies, and we estimate \(\alpha\) for all three pairs of frequencies.

Despite the small sample of measurements, we find: (1) most cases show significant departures from the traditional \(\nu^{-4.4}\) scaling expected for \(\tau_{d}\), and (2) the inferred scaling index depends on the type of the PBF adopted for the deconvolution. These have important implications for the interpretations that will ensue in terms of the nature of the wavenumber spectrum, as we discuss below.

\subsection*{5.2. Scaling of \(\tau_{d}\) with DM}

An empirical relation connecting the pulse-broadening time and DM serves as a useful guide in designing largescale pulsar surveys. An ideal pulsar survey will be scattering limited rather than dispersion limited. Most surveys to date have not, in fact, been scattering limited; this is not because they have been poorly designed, but rather because they have been constrained by data throughput and computational limitations. In other words, scattering plays a significant role in determining the maximum distance to which a pulsar can be detected and thus influences the observable population of pulsars. The relation also provides some useful insights into the large-scale distribution of free electrons ( \(n_{e}\) ) and the strength of their density fluctuations ( \(\delta n_{e}\) ) in the Galaxy.

Figure 4 shows a scatter plot of \(\tau_{d}\) and DM. Most of the points at smaller DMs ( \(\lesssim 100 \mathrm{pc} \mathrm{cm}^{-3}\) ) are derived from measurements of decorrelation bandwidth, \(\nu_{d}\), which are converted to scattering times by \(\tau_{d}=C_{1} / 2 \pi \nu_{d}\), assuming \(C_{1}=1\). Direct measurements of pulse-broadening times dominate at larger DMs ( \(\gtrsim 100 \mathrm{pc} \mathrm{cm}^{-3}\) ). Evidently, there is a strong correlation between DM and \(\tau_{d}\) over the 10 orders of magnitude of variation in \(\tau_{d}\). The values of DM cover only 3 orders of magnitude, signifying a strong dependence of pulse-broadening on DM. There is also large scatter of \(\tau_{d}\) about the trend, roughly 2 to 3 orders of magnitude. Some of the scatter results from the fact that we have scaled all measurements to a common frequency of 1 GHz using \(\tau_{d} \propto \nu^{-4.4}\). However, alternative scaling indices will yield an error of no more than about 0.4 in \(\log \tau_{d}\). At lower DMs, some of this scatter may be attributed to refractive scintillation effects which cause fluctuations in the decorrelation bandwidth (e.g. Bhat, Rao, \& Gupta 1999a). Also, some of the scatter may be due to the small numbers of "scintles" contributing to a measurement. At larger DMs, the scatter is primarily caused by strong spatial variations in \(C_{n}^{2}\).

We fit the values of \(\tau_{d}\) and DM shown in Figure 4 using a simple parabolic curve of the form (e.g. Cordes \& Lazio 2002b)
\[
\log \tau_{d} \approx a+b(\log \mathrm{DM})+c(\log \mathrm{DM})^{2}-\alpha \log \nu
\]
where \(\nu\) is the frequency of observation in GHz , and \(\tau_{d}\) is in ms. Previous work has assumed a fixed scaling index, \(\alpha=4.4\), while fitting for the coefficients \(a, b\) and c. In the light of our results discussed in § 5.1, and also other recent work (e.g. Löhmer et al. 2001) that suggest a departure from the traditional \(\tau_{d} \propto \nu^{-4.4}\) behavior, we treat the scaling index \(\alpha\) as an additional parameter in determining the best fit curve. Note that most published measurements of \(\tau_{d}\) were determined by assuming a PBF of the form \(\mathrm{PBF}_{1}\) (and assuming the conventional frequencyextrapolation approach). Hence we use our values of \(\tau_{d}\)
determined by using the same form of PBF (column 8 of Table 2) in order to ensure uniformity of the data used for the fit. Furthermore, to allow an unbiased fit for \(\alpha\), we use measurements in their unscaled form, i.e., direct estimates of \(\tau_{d}\) and \(\nu_{d}\) at the observing frequencies \({ }^{2}\). The data used for our fit, many from prior compilations, include 148 estimates of \(\nu_{d}\) and 223 estimates of \(\tau_{d}\) (of which 64 are our own measurements), thus 371 measurements in total. Note that the upper limits are excluded from the fit, as none of them seem to impose any constraints to the fit. For a subset of these objects, measurements are available at multiple frequencies. The best-fit curve from our analysis is shown as the solid line in Figure 4. Our re-derived coefficients, \(a=-6.46, b=0.154\) and \(c=1.07\), are only slightly different from those of Cordes \& Lazio (2002b), \(a=-6.59\), \(b=0.129\) and \(c=1.02\). Interestingly, the global scaling index derived from our best fit is \(\alpha=3.86 \pm 0.16\), which is significantly less than the canonical value of 4.4 appropriate for a Kolmogorov medium with negligible inner scale.

There are several plausible explanations for departure from the \(\nu^{-4.4}\) scaling behavior for \(\tau_{d}\), such as (i) the presence of a finite wavenumber cutoff associated with an inner scale, (ii) a non-Kolmogorov form for the density spectrum, and (iii) truncation of the scattering medium transverse to the LOS, as addressed by Cordes \& Lazio (2001). Presently available observational data suggest that option (i) may apply, so we investigate the effects of an inner scale on the scaling laws for \(\tau_{d}\). Option (iii) may be relevant for specific LOSs that contain filamentary or sheet-like structures that could mimic truncated screens. In addition, there is yet another effect whereby a weakening of the scaling index (as deduced from measurements of \(\nu_{d}\) and \(\tau_{d}\) ) could result from refraction effects in the ISM. As argued theoretically and demonstrated through observational data, refraction from scales larger than those responsible for diffraction will bias the diffraction bandwidth downward, corresponding to an upward bias on pulsebroadening (e.g. Cordes, Pidwerbetsky, \& Lovelace 1986; Gupta, Rickett, \& Lyne 1994; Bhat et al. 1999b). The refraction effects will be stronger at higher frequencies as one approaches the transition regime between weak and strong scattering, which will tend to weaken the frequency dependence from 4.4 to a lower index. For pulsars at low DMs (say, \(\mathrm{DM} \leqslant 100 \mathrm{pc} \mathrm{cm}^{-3}\) ), this transition is expected near \(\sim 1-3 \mathrm{GHz}\). Our sample contains many low-DM objects with measurements at \(\sim 1-2 \mathrm{GHz}\) where such an effect may be significant.

\subsection*{5.2.1. Effect of finite inner scale on \(\alpha\)}

The presence of a finite inner scale can potentially modify the frequency scaling index as estimated from measurements of \(\nu_{d}\) and \(\tau_{d}\). Cordes \& Lazio (2002b) show that these effects become apparent above a "crossover point" that is a function of distance (or DM) as well as the observing frequency \(\nu\). The crossover point can be defined for commonly used observables such as \(\theta_{d}\) (angular broadening), \(\nu_{d}\) and \(\tau_{d}\). In order to examine our data for any such signatures of an inner scale, we define a "test quantity" in terms of \(\tau_{d}, \nu\), and distance ( \(D\) ) that is directly related to the inner scale, expressed in units of 100 km , 2
Note that many published data, such as those in Taylor et al. (1995), are already pre-scaled to a common frequency of 1 GHz .
\(l_{100}=l_{i} /(100 \mathrm{~km})\). The crossover point \(\tau_{d, \text { cross }}\) is related to the inner scale by (see Eq. A20 of Cordes \& Lazio (2002b)),
\[
\tau_{d, \text { cross }} \approx 5.46 \mathrm{~ms} D\left(\nu_{\mathrm{GHz}} l_{100}\right)^{-2}
\]
where \(D\) and \(\nu\) are in kpc and GHz , respectively. Thus, a useful test quantity for identifying a break point in the frequency scaling is \(\tau_{d, \text { cross }} \nu^{2} / D\). In the analysis that follows, we use a simple linear relation to convert DM measurements to distances, \(D=\mathrm{DM} /\left(1000\left\langle n_{e}\right\rangle\right)\), where \(\left\langle n_{e}\right\rangle=0.03 \mathrm{~cm}^{-3}\) is the mean electron density, and DM is in units of \(\mathrm{pc} \mathrm{cm}^{-3}\). We emphasize that we adopt such a simplistic approach as a preliminary step, and will defer to another paper a more detailed and complete analysis using proper electron density models and the independent pulsar distance estimates.

We split the data-set into two parts, below and above a chosen break point value for this test quantity, and for each case we refit the parabolic curve in Eq. 7 for the best fit \(\alpha\) while keeping the coefficients \(a, b, c\) fixed at their global fit values. We do this exercise for several break point values in the range 0.03-3.3, determining the difference in best fit \(\alpha\) values for the two samples in each case ( \(\delta \alpha=\alpha_{b l}-\alpha_{b h}\), where \(\alpha_{b l}\) and \(\alpha_{b h}\) denote the values of \(\alpha\) for the samples that are below and above the break point). If an inner scale effect is truly relevant, we will expect a significant difference in \(\alpha\) for the two samples (with a larger value for the sample below the break point, i.e., \(\alpha_{b l}>\alpha_{b h}\) ).

Figure 5 shows a plot of \(\delta \alpha\) vs. the test quantity \(\tau_{d, \text { cross }} \nu^{2} / D\), along with a corresponding plot of the best fit \(\chi^{2}\left(\chi^{2}=\chi_{1}^{2}+\chi_{2}^{2}\right.\), where \(\chi_{1}^{2}\) and \(\chi_{2}^{2}\) denote the corresponding values of \(\chi_{i}^{2}\) for the two data-sets). The maximum in \(\delta \alpha\) roughly coincides with the minimum in \(\chi^{2}\), suggesting that the inner scale effect is real. Our analysis shows a sharp minimum for \(\chi^{2}\) at \(\log \left(\tau_{d, \text { cross }} \nu^{2} / D\right) \approx-0.57\) (Fig. 5). Formally, the \(\pm 1 \sigma\) error on the break point value of \(\log \left(\tau_{d, \text { cross }} \nu^{2} / D\right)\) is \(\pm 0.05\). However, the valley in \(\chi^{2}\) is much broader than implied by this error. We take a more realistic range to be -1 to -0.3 in the log, corresponding to an inner scale \(l_{i} \approx 100 \mathrm{~km}\left(5.46 D / \tau_{d} \nu^{2}\right)^{1 / 2} \approx 300\) to 800 km .

The broadness of \(\chi^{2}\) is caused in part by our assumption of a simple proportionality between distance and DM and also by the likely variation of inner scale between locations in the Galaxy. Some theories for density fluctuations in the ISM would associate the inner scale with the proton gyroradius for thermal gas. The gyroradius is \(r_{g} \approx 1658 \mathrm{~km} T_{4}^{1 / 2} B_{\mu \mathrm{G}}^{-1}\) for a temperature \(T=10^{4} T_{4} \mathrm{~K}\) and a magnetic field strength \(B\) expressed in micro gauss. For ionized gas in the warm phase of the ISM, we expect the temperature to vary by a factor of 2 to 4 and the field strength by at least a similar factor. Thus, we would expect the gyroradius to vary by at least a factor of five, which is not inconsistent with the appearance of \(\chi^{2}\) in Figure 5. Given the expected variation of the gyroradius in the ISM, it is perhaps surprising that we see any kind of minimum in \(\chi^{2}\) at all.

Several authors have investigated the effect of an inner scale, and constraints are available from various kinds of observations. For example, Spangler \& Gwinn (1990) derived an inner scale of \(\sim 50-200 \mathrm{~km}\) from an analysis of interferometer visibility measurements from VLBI observations, which are, interestingly, of the order of our estimates
derived from pulse-broadening data. Further, as noted by Moran et al. (1990), observations of NGC 6334B (the object with the largest known scattering disk) at centimeter wavelengths are consistent with an inner scale larger than 35 km . Studies of long-term flux density variations of pulsars at low radio frequencies, however, indicate a much larger inner scale (e.g. \(\sim 10^{2}-10^{4} \mathrm{~km}\) from the work of Gupta, Rickett, \& Coles (1993)). While some discrepancies prevail between the estimates deduced from different observations, it appears that effects due to an inner scale are well supported by a number of observations.

\section*{6. GALACTIC ELECTRON DENSITY MODELS}

Our sample largely comprises high-DM, distant pulsars and hence provides useful data for improving upon electron density models for the inner parts of the Galaxy. We compare our data with predictions from both the TC93 (Taylor \& Cordes 1993) and NE2001 (Cordes \& Lazio 2002a,b) models, which yield values for SM that may be used in Eq. 2. The newer model, NE2001, has made use of only DM values of some of the multibeam pulsars; hence, our measurements of \(\tau_{d}\) allow an independent test of the new model.

Figures 6 and 7 show plots of the measurements of pulsebroadening times against the predictions from the new and old electron density models, respectively. In order to examine more general trends, we also plot all the published measurements (see Taylor et al. (1995) and references therein), after scaling to a common frequency of 1 GHz using \(\tau_{d} \propto \nu^{-4.4}\). A significant number of measurements show reasonable agreement with the model predictions, suggesting that the models depict fairly good representation of the large-scale picture in the Galaxy. However, significant discrepancies are evident in many cases, compared against the predictions from either of the two models. For a majority of the measurements ( \(\sim 75 \%\) ) from our own observations, the discrepancy is significantly lower with the predictions of NE2001 than with those of TC93. In some cases, the agreement with the model prediction shows improvements of the order a factor of two or better. Given that our measurements were not part of the inputs for the new model, this comparison makes an independent test of the new model.

\subsection*{6.1. Clumps of Excess Scattering}

As discussed in § 1.2, the measured broadening time \(\tau_{d}\) is related to the total amount of scattering, usually quantified as the scattering measure, SM (see Eq. 2). For a given scattering geometry (indicated by the corresponding geometric factor \(W_{\tau}\) in the equation), we can invert this equation to derive the scattering measure. We assume \(W_{\tau}=1\) and estimate the effective SM for a uniform medium. The estimated values of SM (SM meas, in the conventional units of \(\mathrm{kpc} \mathrm{m}^{-20 / 3}\) ) are listed in Table 4 (column 7).

Figure 8 shows the distribution of inferred SMs at the locations of pulsars. The spiral arm locations are adopted from the NE2001 model and the pulsar distances are the revised estimates using this new electron density model. A more useful quantity is the departure of the measured quantity ( \(\tau_{d}\) or SM) from the model predictions. In order to examine this in detail, we plot the quantity \(\left|\log \left(\tau_{d} / \tau_{d, n e 2001}\right)\right|\) at the locations of the pulsars (Fig. 9).

Significant departures are seen toward many LOSs. In the case of low-DM pulsars, these may be due largely to measurement errors due to refractive scintillation effects (Bhat, Rao, \& Gupta 1999a). For distant, high-DM pulsars, departures from the model predictions are in general larger in the inter-arm region. Most published data are in good agreement with the model predictions as expected, while several of the new measurements differ significantly from the model expectations.

A closer examination of Figures 6 and 7 reveals that despite the general agreement seen with a large number of measurements, the models still underestimate the total amount of scattering for many LOSs. The underestimates are accounted for easily by relaxing one or more assumptions that underly the calculation of SM and its interpretation, as has been pointed out by Cordes \& Lazio (2002b). In particular, clumps of enhanced scattering are likely due to unmodeled features associated with Hii regions or supernova shocks. Following Cordes \& Lazio (2002b) (see also Chatterjee et al. (2001)), we characterize such "clumps" in terms of the incremental SM and DM due to them. We calculate the increments associated with a clump as
\[
\delta \mathrm{DM}=n_{e, c} \delta s
\]
and
\[
\delta \mathrm{SM}=C_{n, c}^{2} \delta s
\]
where \(n_{e, c}\) is the mean electron density and \(C_{n, c}^{2}\) is a measure of the fluctuating electron density inside a clump and \(\delta s\) is the size of the clump region. The parameter \(C_{n, c}^{2}\) can be expressed in terms of the electron density and the "fluctuation parameter," \(F_{c}\) (see TC93; Cordes \& Lazio (2002a)),
\[
C_{n, c}^{2}=C_{S M} F_{c} n_{e, c}^{2}
\]
where \(C_{S M}\) is a numerical constant that depends on the slope of the wavenumber spectrum, and is defined as \(C_{S M}=\left[3(2 \pi)^{1 / 3}\right]^{-1} K_{u}\) for a Kolmogorov spectrum, where the scale factor \(K_{u}=10.2 \mathrm{~m}^{-20 / 3} \mathrm{~cm}^{6}\) yields SM in the conventional units of \(\mathrm{kpc} \mathrm{m}^{-20 / 3}\). The fluctuation parameter \(F_{c}\) depends on the outer scale ( \(l_{o}\) ), filling factor \((\eta)\), and the fractional rms electron density inside the clump. It is defined as (TC93)
\[
F_{c}=\zeta \epsilon^{2} \eta^{-1} l_{o}^{-2 / 3}
\]
where \(\zeta=\left\langle\overline{n_{e}{ }^{2}}\right\rangle /\left\langle\overline{n_{e}}\right\rangle^{2}\), and \(\epsilon=\left\langle\left(\delta n_{e}\right)^{2}\right\rangle /{\overline{n_{e}}}^{2}\). From equations 9-11, the ratio of increments in SM and DM is given by
\[
\frac{\delta \mathrm{SM}}{\delta \mathrm{DM}}=C_{S M} F_{c} n_{e, c}
\]

The above expression can be re-written as
\[
\delta \mathrm{SM}=C_{S M} \frac{F_{c}(\delta \mathrm{DM})^{2}}{\delta s}
\]

For large distances of a few to several kpc that are relevant for our measurements, it is a fair assumption that the LOS to the pulsar may encounter several such clumps. Assuming a clump thickness of \(\sim 10 \mathrm{pc}\) (typical size of known HıI regions), and a volume number density for clumps, \(n_{c} \sim 1 \mathrm{kpc}^{-3}\), we obtain the values of \(F_{c}(\delta \mathrm{DM})^{2}\) for the subset of measurements in Table 4 that show excess SM (see also

Fig. 10). The constraints derived from our data lie within a broad range of
\[
10^{-5.3}<F_{c}(\delta \mathrm{DM})^{2}<10^{-1.8}
\]
which is consistent with values needed to account for the excess scattering toward the LOS to pulsar B0919+06 derived by Chatterjee et al. (2001). If we assume a fluctuation parameter ( \(F_{c}\) ) of 10 for the clumps, which is consistent with values in TC93 and Cordes \& Lazio (2002a), the required range in \(\delta \mathrm{DM}\) is \(7 \times 10^{-4}<\delta \mathrm{DM}<4 \times 10^{-2} \mathrm{pc} \mathrm{cm}^{-3}\). For the assumed size of 10 pc for the clumps, this implies \(10^{-5} \lesssim n_{e, c} \lesssim 4 \times 10^{-3} \mathrm{~cm}^{-3}\). In reality, both the fluctuation parameter \(F_{c}\), as well as the sizes ( \(\delta s\) ) and number of clumps \(\left(n_{c}\right)\), will vary with the LOS; nonetheless, the inferred values of \(\delta \mathrm{SM}\) are such that the derived constraints on the clumps are well within the range of physical possibilities. Note also that the implied perturbations of DM are rather small, a fact that highlights the situation that relatively small changes in the local mean electron density can translate into large changes in the amount of scattering.

\section*{7. IMPLICATIONS FOR THE ELECTRON DENSITY WAVENUMBER SPECTRUM}

Our measurements of the scaling index \(\alpha\) and the implied power spectral slopes \(\beta\) are summarized in Table 3. In a few cases, the estimates of \(\alpha\) are consistent with the simple, Kolmogorov scaling of \(\alpha=4.4\) (e.g. PSRs J1853+0545, J1913+1145, and J1920+1110). However, in most cases the measured scaling is significantly weaker than even \(\nu^{-4}\) (e.g. PSR J1856+0404). Overall the measurements show a possible departure from the traditional expectation of \(\nu^{-4.4}\) scaling, with a mean scaling index \(\langle\alpha\rangle \approx 3.12 \pm 0.13\) using the results for \(\mathrm{PBF}_{1}\), and \(\approx 3.83 \pm 0.19\) using those for \(\mathrm{PBF}_{2}\), in agreement with other recent work (Löhmer et al. 2001), and also comparable to a global scaling index \(3.86 \pm 0.16\) inferred from our parabolic fit to \(\tau_{d}\) vs DM data.

Figure 11 summarizes the current state of the estimates of \(\alpha\) derived from measurements of decorrelation bandwidths and pulse-broadening times. In addition to the present work which yielded \(\alpha\) for 15 objects, this includes the recent measurements from Löhmer et al. (2001) (for 9 high-DM objects) and those from Cordes, Weisberg, \& Boriakoff (1985) (for 5 objects at low DMs) derived from measurements of \(\nu_{d}\). Barring a few outlier cases, it appears that the scaling index is lower for objects of larger DMs ( \(\gtrsim 200 \mathrm{pc} \mathrm{cm}^{-3}\) ), while it seems consistent with the Kolmogorov expectation for objects at lower DMs (although these are only 5 in number). A similar result is also indicated by our analysis of the DM dependence of pulse broadening times (§ 5.2.1).

We now return our attention to the dependence of the scaling index on the PBF form adopted for the analysis. PSR J1853+0545 is a particularly illustrative example. For this object, based on the results for \(\mathrm{PBF}_{1}\), we estimate a mean scaling index much lower than \(4.4,\langle\alpha\rangle=3.1 \pm 0.2\). However, use of \(\mathrm{PBF}_{2}\) yields scaling indices that agree well with that expected for a \(\beta=11 / 3\) spectrum. Naturally, the two cases may lead to widely different interpretations in terms of the nature of the wavenumber spectrum. Similarly, for PSR J1857+0526, while the estimate of \(\alpha\) deduced from \(\tau_{d}\) values obtained for the case of \(\mathrm{PBF}_{1}\) imply
a power-law index that approaches the Kolmogorov value, even if it is a little low, the results for the case of \(\mathrm{PBF}_{2}\) yield a much lower value.

Given all of this, it is important to attempt to use an approximately correct form for the PBF before attempting any serious interpretation in terms of the nature of the spectrum. A mere departure from the expected \(\nu^{-4.4}\) scaling need not necessarily signify an anomaly for the scattering along that LOS. However, in general it is clearly difficult to know what is the correct form of the PBF for a given LOS. Additional figures of merit such as the derived intrinsic pulse shapes, in particular their dependence with frequency, and the number of CLEAN components (Bhat et al. 2003) may help to resolve ambiguities in some cases, but the general problem remains a difficult one.

\section*{8. SUMMARY AND CONCLUSIONS}

We have used multi-frequency radio data obtained with the Arecibo telescope for a sample of 98 , mostly distant, high-DM, pulsars to measure in particular the pulsebroadening effect due to propagation in the inhomogeneous interstellar medium. For 81 of these objects we obtained data at \(0.4,1.2,1.5\) and 2.4 GHz , while data for the remaining 17, at \(0.4,1.2\) and 1.5 GHz , are from the recent work of Lorimer et al. (2002). We employed a CLEAN-based deconvolution method to measure pulsebroadening times. In this process we tested two possible forms of the pulse-broadening function that characterizes scattering along the LOS. As a by-product, the method also yields estimated shapes of the intrinsic pulse profiles.

The present work has resulted in new measurements of pulse-broadening time for 56 pulsars, and upper limits for 31 pulsars. These data, along with similar measurements from other published work, were compared with the predictions from models for the Galactic free electron density. New measurements allow an independent test of the electron density model recently developed by Cordes \& Lazio (2002a). While a majority of the data is in reasonable agreement with the model predictions, evidence for excess scattering is seen for many LOSs. We consider the possibility whereby the excess scattering can be accounted for by using "clumps," regions of enhanced scattering in the Galaxy. Depending on the distance, a given LOS may contain one or more of such clumps, and we derive useful constraints on their properties.

For a small subset of objects, our data also allow estimation of the frequency scaling indices for the pulsebroadening times, most of which show significant departures from the traditional \(\nu^{-4.4}\) behavior expected for the case of a Kolmogorov power-law form for the spectrum of density irregularities. Our analysis also suggests that the inferred scaling indices depend on the type of PBF adopted for the analysis. We combined our data with those from published work to revise the empirical relation connecting pulse-broadening times and dispersion measures. The inferred frequency scaling index from a globat fit is \(3.9 \pm 0.2\), less than that expected for the case of a Kolmogorov spectrum. Our analysis also suggests the possibility of an inner scale in the range \(\sim 300-800 \mathrm{~km}\) for the spectrum of turbulence. Further, the intrinsic pulse shapes deduced from
our analysis for several of the pulsars are likely to be comparable to the actual pulse shapes, and hence may prove useful for applications such as the study of pulsar emission properties.

We thank Bill Sisk, Jeff Hagen and Andy Dowd for developing the WAPP system at the Arecibo Observatory, which was crucial for providing much of the data analyzed in this paper. This work was supported by NSF grants AST9819931, AST0138263 and AST0206036 to Cornell University, AST0205853 to Columbia University, and AST0206205 to Princeton University. NDRB is supported by an MIT-CfA Postdoctoral Fellowship at Haystack Observatory. DRL is a University Research Fellow funded by the Royal Society. Arecibo Observatory is operated by the National Astronomy and Ionosphere Center, which is operated by Cornell University under cooperative agreement with the National Science Foundation (NSF).

\section*{REFERENCES}

Armstrong, J. W., Rickett, B. J., \& Spangler, S. R. 1995, ApJ, 443, 209
Bhat, N. D. R., Camilo, F., Cordes, J. M., Nice, D. J., Lorimer, D. R., \& Chatterjee, S. 2002, Journal of Astrophysics and Astronomy, 22, 53
Bhat, N. D. R., Cordes, J. M. \& Chatterjee, S. 2003, ApJ, 584, 782
Bhat, N. D. R. \& Gupta, Y. 2002, ApJ, 567, 342
Bhat, N. D. R., Gupta, Y., \& Rao, A. P. 1999, ApJ, 514, 249
Bhat, N. D. R., Rao, A. P., \& Gupta, Y. 1999, ApJS, 121, 483
Bhattacharya, D., Wijers, R. A. M. J., Hartman, J. W., \& Verbunt, F. 1992, A\&A, 254, 198

Boldyrev, S., \& Gwinn, C. R. 2003, ApJ, 584, 791
Chatterjee, S., Cordes, J. M., Lazio, T. J. W., Goss, W. M., Fomalont, E. B., \& Benson, J. M. 2001, ApJ, 550, 287
Cordes, J. M., Pidwerbetsky, A., \& Lovelace, R. V. E. 1986, ApJ, 310, 737
Cordes, J. M. \& Lazio, T. J. 1991, ApJ, 376, 123
Cordes, J. M. \& Lazio, T. J. W. 2001, ApJ, 549, 997
Cordes, J. M. \& Lazio, T. J. W. 2002a, ApJ, submitted (astroph/0207156)
Cordes, J. M. \& Lazio, T. J. W. 2002b, ApJ, submitted (astroph/0301598)
Cordes, J. M. \& Rickett, B. J. 1998, ApJ, 507, 846
Cordes, J. M., Weisberg, J. M., \& Boriakoff, V. 1985, ApJ, 288, 221
Cornwell, T., Braun, R., \& Briggs, D. S. 1999, ASP Conf. Ser. 180: Synthesis Imaging in Radio Astronomy II, 151
Dowd, A., Sisk, W., \& Hagen, J. 2000, ASP Conf. Ser. 202: IAU Colloq. 177: Pulsar Astronomy - 2000 and Beyond, 275
Gupta, Y., Rickett, B. J., \& Coles, W. A. 1993, ApJ, 403, 183
Gupta, Y., Rickett, B. J., \& Lyne, A. G. 1994, MNRAS, 269, 1035
Hagen, J. B. \& Farley, D. 1973, Rad. Sci., 8, 775
Hankins, T. H. \& Rickett, B. J. 1986, ApJ, 311, 684
Hobbs, G. B., \& Manchester, R. N. 2003, ATNF Pulsar Catalog, http://www.atnf.csiro.au/research/pulsar/psrcat
Högbom, J. A. 1974, A\&AS, 15, 417
Hobbs, G. B., et al. 2004, in preparation
Hulse, R. A. \& Taylor, J. H. 1975, ApJ, 201, L55
Johnston, S., Nicastro, L., \& Koribalski, B. 1998, MNRAS, 297, 108
Kramer, M. et al. 2003, MNRAS, 342, 1299
Kuz'min, A. D. \& Izvekova, V. A. 1993, MNRAS, 260, 724
Lambert, H. C. \& Rickett, B. J. 1999, ApJ, 517, 299
Lambert, H. C. \& Rickett, B. J. 2000, ApJ, 531, 883
Löhmer, O., Kramer, M., Mitra, D., Lorimer, D. R., \& Lyne, A. G. 2001, ApJ, 562, L157
Lorimer, D. R. 2001, Arecibo Technical and Operations Memo Series No. 2001-01
Lorimer, D. R., Camilo, F., \& Xilouris, K. M. 2002, AJ, 123, 1750
Manchester, R. N. et al. 2001, MNRAS, 328, 17
Moran, J. M., Greene, B., Rodriguez, L. F., \& Backer, D. C. 1990, ApJ, 348, 147
Morris, D. J. et al. 2002, MNRAS, 335, 275
Ramachandran, R., Mitra, D., Deshpande, A. A., McConnell, D. M., \& Ables, J. G. 1997, MNRAS, 290, 260
Rickett, B. J. 1977, ARA\&A, 15, 479

Rickett, B. J. 1990, ARA\&A, 28, 561
Roberts, D. H., Lehar, J., \& Dreher, J. W. 1987, AJ, 93, 968
Romani, R. W., Narayan, R., \& Blandford, R. 1986, MNRAS, 220, 19
Schwarz, U. J. 1978, A\&A, 65, 345
Spangler, S. R. \& Gwinn, C. R. 1990, ApJ, 353, L29
Taylor, J. H. \& Cordes, J. M. 1993, ApJ, 411, 674
Taylor, J. H., Manchester, R. N., \& Lyne, A. G. 1993, ApJS, 88, 529
Taylor, J. H., Manchester, R. N., \& Lyne, A. G. 1995, catalog available by anonymous ftp at pulsar.princeton.edu
Weisberg, J. M., Pildis, R. A., Cordes, J. M., Spangler, S. R., \& Clifton, T. R. 1990, BAAS, 22, 1244
Williamson, I. P. 1972, MNRAS, 157, 55
Williamson, I. P. 1973, MNRAS, 163, 345
Williamson, I. P. 1974, MNRAS, 166, 499

\section*{APPENDIX}

\section*{APPENDIX A}

\section*{PROFILE DATABASE}

The basic data (i.e. the pulse profiles in Fig. 1) presented in this paper are also available as an electronic data set. The full database is packaged as a gzipped tar file, AOmultifreq_profs.tar.gz (which includes 345 pulse profiles from our observations), and is available over the Internet http://web.haystack.mit.edu/staff/rbhat/aoprofs or can be downloaded via anonymous ftp from the ftp site web.haystack.mit.edu (the directory is pub/rbhat/aoprofs). These profiles are stored as individual files in simple ascii format, which consists of a header line of the basic observing parameters followed by an ascii list of pulse bin number and the intensity value (in arbitrary units) in a two-column format. Each file is given a generic name of the format pulsar.freq.prf, where pulsar is the name of the pulsar and freq is the frequency of observation in MHz . An example header is shown below, along with a description of the various parameters included in the header.
```
# mjdobs mjdsec per np freq refdm nbin siteid scanid source
```

where
```
mjdobs : Date of observation (MJD)
mjdsec : Time of observation (seconds, UTC) with
    respect to mjdobs
per : Pulse period (seconds)
np : Pulse count
freq : Frequency of observation (MHz)
refdm : Dispersion measure (pc cm -3)
nbin : Number of bins in the pulse profile
siteid : Site ID of observations ('3' for Arecibo)
scanid : Scan number of observation
source : Source name
```


\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 1
Data acquisition parameters.}
\begin{tabular}{|l|l|l|l|l|l|}
\hline Frequency (MHz) & Bandwidth (MHz) & Time Resolution & Spectral Channels & Instrument & Integration Time \({ }^{\text {a }}\) per Scan (min) \\
\hline 430 & 8 & \(10^{-3} P\) & 128 & PSPM & 10 \\
\hline 1175 & 100 & \(256 \mu \mathrm{~s}\) & 256 & WAPP & 5 \\
\hline 1475 & 100 & \(256 \mu \mathrm{~s}\) & 256 & WAPP & 5 \\
\hline 2380 & 50 & \(256 \mu \mathrm{~s}\) & 64 & WAPP & 10 \\
\hline
\end{tabular}
\end{table}

\footnotetext{
\({ }^{\text {a }}\) Multiple scans were made for pulsars with low signal-to-noise ratio in the first pass.
}

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 2
New measurements of pulse-broadening times and predictions from the electron density models.}
\begin{tabular}{|l|l|l|l|l|l|l|l|l|l|l|l|l|}
\hline PSR & Ref \({ }^{\mathrm{a}}\) & Period (ms) & DM ( \(\mathrm{pc} \mathrm{cm}^{-3}\) ) & \(\iota \left({ }^{\circ}\right)\) & \(b \left({ }^{\circ}\right)\) & \(\nu\) (MHz) & \(\mathrm{PBF}_{1} \tau_{d}(\mathrm{~ms})\) & \(f_{r}\) & \(\mathrm{PBF}_{2} \tau_{d}(\mathrm{~ms})\) & \(f_{r}\) & \({ }^{\tau} d,{ }_{t c} 93\) (ms) & \({ }^{\tau}\) d,ne2001 (ms) \\
\hline (1) & (2) & (3) & (4) & (5) & (6) & (7) & (8) & (9) & (10) & (11) & (12) & (13) \\
\hline J1848+0826 & 1,400 & 328.64 & 90.8 & 40.1 & 4.6 & 1175 & <0.2 & & <0.2 & & 0.003 & 0.001 \\
\hline J1849+0127 & 2,1400 & 542.11 & 214.4 & 33.9 & 1.2 & 430 & \(78 \pm 21\) & 0.9 & ... \({ }^{b}\) & & 13.4 & 18.42 \\
\hline & & & & & & 1175 & \(6.5 \pm 2.1\) & 1.2 & \(3.3 \pm 1\) & 0.1 & 0.161 & 0.221 \\
\hline J1850+0026 & 2,1400 & 1081.76 & 197.2 & 33.2 & 0.5 & 430 & \(9.6 \pm 2.4\) & 2.5 & \(4.8 \pm 1.1\) & 0.2 & & 18.97 \\
\hline & & & & & & 1175 & <4.2 & & <1.1 & & 0.127 & 0.228 \\
\hline J1851+0118 & 3,1400 & 907.05 & 413.0 & 34.1 & 0.7 & 1175 & <5 & & <3 & & 1.027 & 2.917 \\
\hline J1851+0418 & 1,300 & 284.71 & 112.0 & 36.7 & 2.1 & 1175 & <3.5 & & <1 & & 0.010 & 0.008 \\
\hline J1852+0031 & 1,1400 & 2180.06 & 680.0 & 33.5 & 0.1 & 1175 & \(495 \pm 25\) & 2.5 & \(271 \pm 5.7\) & 7.9 & 4.846 & 411.6 \\
\hline & & & & & & 1475 & \(225 \pm 14\) & 1.6 & \(127 \pm 3.7\) & 1.8 & 1.782 & \\
\hline J1852+0305 & 2,1400 & 1326.06 & 315.6 & 35.7 & 1.3 & 1175 & <14 & & .. \({ }^{b}\) & & 0.448 & 0.728 \\
\hline J1853+0056 & 2,1400 & 275.56 & 182.4 & 33.9 & 0.1 & 1175 & <2 & & <1 & & 0.089 & 0.196 \\
\hline J1853+0505 & 3,1400 & 905.21 & 273.6 & 37.6 & 2.0 & 1175 & \(124 \pm 14\) & 0.5 & .. . \({ }^{b}\) & & 0.243 & 0.193 \\
\hline & & & & & & 1475 & \(54 \pm 3\) & 1.4 & \(23 \pm 4\) & 1.8 & 0.089 & 0.071 \\
\hline J1853+0545 & 5,1400 & 126.39 & 198.7 & 38.2 & 2.3 & 1175 & \(13.6 \pm 2\) & 0.4 & \(8.2 \pm 0.3\) & 2.5 & 0.098 & 0.049 \\
\hline & & & & & & 2380 & \(7.1 \pm 0.9\) & 0.3 & \(0.4 \pm 0.1\) & 0.8 & 0.036 & 0.018 \\
\hline J1855+0422 & 2,1400 & 1677.98 & 438.6 & 37.2 & 1.2 & 1175 & \(27 \pm 3\) & 0.2 & \(16.6 \pm 2.5\) & 0.9 & 1.004 & 1.803 \\
\hline & & & & & & 1475 & <12 & & <4 & & 0.369 & 0.663 \\
\hline J1856+0113 & 1,1400 & 267.46 & 96.7 & 34.5 & -0.5 & 1175 & <1 & & <1 & & 0.006 & 0.007 \\
\hline J1856+0404 & 2,1400 & 420.22 & 345.3 & 37.1 & 0.8 & 1175 & \(9.5 \pm 4\) & 0.3 & \(4.8 \pm 1.6\) & 0.04 & 0.709 & 1.279 \\
\hline & & & & & & 1475 & \(6 \pm 3\) & 0.2 & \(2.8 \pm 1\) & 0.2 & 0.261 & 0.470 \\
\hline J1857+0057 & 1,300 & 356.93 & 83.0 & 34.4 & -0.8 & 430 & <5 & & <1 & & 0.271 & 0.229 \\
\hline & & & & & & & & & & & & \\
\hline J1857+0210 & 2,1400 & 630.94 & 783.2 & 35.5 & -0.3 & 1175 & \(13.4 \pm 3.6\) & 0.06 & \(6.1 \pm 0.65\) & 0.04 & 9.004 & 19.96 \\
\hline J1857+0212 & 1,1400 & 415.80 & 504.0 & 35.5 & -0.2 & 1175 & \(3.8 \pm 0.9\) & 0.1 & \(1.2 \pm 0.3\) & 1.8 & 2.749 & 6.135 \\
\hline & & & & & & 1475 & \(2.2 \pm 0.3\) & 0.8 & \(0.5 \pm 0.4\) & 0.7 & 1.011 & 2.256 \\
\hline J1857+0526 & 5,1400 & 349.92 & 468.3 & 38.4 & 1.2 & 1175 & \(14.5 \pm 1.7\) & 0.4 & \(6 \pm 1\) & 0.6 & 0.975 & 1.700 \\
\hline & & & & & & 1475 & \(6.2 \pm 1.3\) & 0.4
0.04 & \(3 \pm 1\) & 0.03 & & 0.625 \\
\hline J1857+0809 & 3,1400 & 502.96 & 284.2 & 40.8 & 2.5 & 1175 & <3.5 & & \(<2{ }_{c}\) & & 0.195 & 0.061 \\
\hline J1857+0943 & 1,400 & 5.36 & 13.3 & 42.2 & 3.2 & 430 & ⋯ & & & & 0.001 & 0.001 \\
\hline J1858+0215 & 2,1400 & 745.77 & 702.0 & 35.7 & -0.4 & 1175 & \(18.4 \pm 4.7\) & 3.3 & \(22.5 \pm 6.5\) & 0.05 & 6.267 & 13.77 \\
\hline J1858+0241 & 3,1400 & 4693.60 & 341.7 & 36.1 & -0.2 & 1175 & <22 & & <16 & & 0.758 & 1.702 \\
\hline J1900+0227 & 2,1400 & 374.24 & 201.1 & 36.1 & -0.8 & 1175 & <4 & & <2 & & 0.116 & 0.178 \\
\hline J1901+0156 & 1,400 & 288.22 & 102.1 & 35.7 & -1.2 & 430
1175 & \(3.5 \pm 1\) & 0.2 & \(0.7 \pm 0.3\) & 0.2 & 0.619 & 0.164 \\
\hline & & & & & & & <2 & & <1 & & & 0.002 \\
\hline J1901+0331 & 1,400 & 655.48 & 401.2 & 37.2 & -0.5 & 430 & & 0.8 & \(44 \pm 4.2\) & 1.5 & 107.9 & 76.16 \\
\hline & & & & & & & \(60 \pm 3\)
\(<3\) 3 & & <1 & & 1.295 & \\
\hline J1901+0355 & 3,1400 & 554.80 & 546.2 & 37.5 & -0.3 & 1175 & <4 & & ... \({ }^{b}\) & & 2.97 & 6.459 \\
\hline J1901+0413 & 2,1400 & 2662.88 & 367.0 & 37.8 & -0.2 & 430 & <558 & & .. . \({ }^{b}\) & & 79.5 & 161.7 \\
\hline & & & & & & 1175 & <13 & & <3 & & 0.954 & 1.940 \\
\hline J1901+0716 & 1,1400 & 644.02 & 252.8 & 40.5 & 1.2 & 430 1175 & \(10.1 \pm 2.4\) <2.7 & 0.07 & \(8.5 \pm 3.4\) & 0.04 & 17.12 & 23.05 \\
\hline J1901+1306 & 1,400 & 1830.72 & 75.0 & 45.7 & 3.9 & 1175 & \(\ldots{ }^{d}\) & & ... \({ }^{d}\) & & 0.001 & \(\ll 1\) \\
\hline J1902+0556 & 1,400 & 746.60 & 179.7 & 39.4 & 0.4 & 430 & \(12 \pm 1.1\) & 1.1 & \(5.7 \pm 1.4\) & 0.3 & 4.674 & 10.05 \\
\hline & & & & & & 1175 & <4.2 & & <1.6 & & 0.056 & 0.121 \\
\hline J1902+0723 & 1,400 & 487.82 & 105.0 & 40.7 & 1.0 & 430 & <8 & & <3 & & 0.455 & 0.473 \\
\hline J1903+0135 & 1,400 & 729.33 & 246.4 & 35.7 & -1.8 & 430 & \(11.4 \pm 0.9\) & 3.1 & \(5.2 \pm 0.6\) & 3.5 & 16.83 & 13.73 \\
\hline J1903+0601 & 3,1400 & 374.11 & 398.0 & 39.7 & 0.2 & 1175 & \(2.7 \pm 0.5\) & 0.03 & \(1 \pm 0.4\) & 0.01 & 1.100 & 2.093 \\
\hline & & & & & & 1475 & <1.7 & & <1 & & 0.405 & 0.77 \\
\hline J1904+0004 & 1,400 & 139.53 & 233.7 & 34.4 & -2.8 & 430 & \(3.1 \pm 1\) & 0.7 & \(4 \pm 1\) & 0.8 & 10.78 & 4.66 \\
\hline & & & & & & 1175 & <1.4 & & <0.7 & & 0.129 & 0.56 \\
\hline J1904+0412 & 2,1400 & 71.09 & 185.9 & 38.1 & -0.9 & 430 & ... \({ }^{d}\) & & ... \(d\) & & 5.963 & 10.24 \\
\hline J1904+0800 & 5,1400 & 263.37 & 438.3 & 41.5 & 0.9 & 430 & ... \({ }^{e}\) & & ... \({ }^{e}\) & & 61.02 & 118.7 \\
\hline & & & & & & 1175 & \(3 \pm 0.3\) & 0.1 & \(1.2 \pm 0.12\) & 0.4 & 0.723 & 1.390 \\
\hline & & & & & & 1475 & <2 & & <1 & & 0.266 & 0.511 \\
\hline J1904+1011 & 1,400 & 1856.64 & 135.0 & 43.4 & 1.9 & 430 & .. \({ }^{b}\) & & <4.4 & & 1.015 & 0.937 \\
\hline J1905+0400 & 3,1400 & 3.78 & 25.8 & 38.0 & -1.2 & 430 & ... \({ }^{d}\) & & ... \({ }^{d}\) & & 0.005 & 0.001 \\
\hline J1905+0616 & 2,1400 & 989.64 & 262.7 & 40.1 & -0.2 & 1175 & \(13.5 \pm 3.1\) & 0.04 & \(7.8 \pm 2.1\) & 0.2 & 0.244 & 0.514 \\
\hline & & & & & & 1475 & <5.8 & & <1.5 & & 0.09 & 0.189 \\
\hline J1905+0709 & 1,1400 & 648.05 & 269.0 & 40.8 & 0.3 & 430 & ... \({ }^{b}\) & & \(41 \pm 10\) & 0.1 & 20.95 & 43.77 \\
\hline & & & & & & 1175 & \(7 \pm 4\) & 0.03 & \(3.2 \pm 1.6\) & 0.02 & 0.251 & 0.525 \\
\hline J1906+0641 & 1,1400 & 267.29 & 473.0 & 40.5 & -0.2 & 1175 & \(4.4 \pm 1.1\) & 0.2 & \(2.4 \pm 0.4\) & 0.7 & 1.347 & 111.1 \\
\hline & & & & & & 1475 & \(2.6 \pm 0.7\) & 0.05 & \(1.1 \pm 0.4\) & 0.12 & 0.495 & 40.85 \\
\hline J1906+0912 & 2,1400 & 775.29 & 260.5 & 42.8 & 1.0 & 1175 & <8 & & <4 & & 0.189 & 0.278 \\
\hline J1907+0534 & 2,1400 & 1138.31 & 526.7 & 39.7 & -0.9 & 1175 & <14 & & <7 & & 1.351 & 2.674 \\
\hline J1907+0740 & 2,1400 & 574.65 & 329.3 & 41.5 & 0.1 & 430 & \(10.1 \pm 3.8\) & 0.03 & \(6.6 \pm 1.6\) & 0.03 & 41.23 & 85.56 \\
\hline & & & & & & 1175 & <2.3 & & <1 & & 0.495 & 1.026 \\
\hline J1907+0918 & 1,1400 & 226.12 & 358.0 & 43.0 & 0.8 & 1175 & <2.4 & & <1.2 & & 0.471 & 0.766 \\
\hline J1907+1247 & 1,400 & 827.11 & 257.0 & 46.1 & 2.4 & 430 & .. \({ }^{b}\) & & <1 & & 7.631 & 3.25 \\
\hline J1908+0457 & 1,400 & 846.83 & 360.0 & 39.2 & -1.4 & 430 & . . \({ }^{b}\) & & \(18 \pm 7\) & 0.02 & 42.15 & 57.1 \\
\hline & & & & & & 1175 & <3.6 & & <1.8 & & 0.506 & 0.69 \\
\hline J1908+0500 & 1,400 & 291.03 & 201.4 & 39.3 & -1.4 & 1175 & \(4 \pm 0.7\) & 0.74 & \(2.1 \pm 0.25\) & 0.15 & 7.703 & 0.12 \\
\hline J1908+0734 & 1,400 & 212.34 & 11.1 & 41.6 & -0.2 & 1175 & ... \({ }^{c}\) & & .\(^{c}\) & & \(\ll 1\) & \(\ll 1\) \\
\hline J1908+0839 & 2,1400 & 185.38 & 516.6 & 42.5 & 0.3 & 1175 & \(5.6 \pm 1.3\) & 0.4 & \(3.5 \pm 0.8\) & 0.2 & 1.131 & 2.595 \\
\hline & & & & & & 1475 & \(2.6 \pm 0.6\) & 0.55 & \(1.4 \pm 0.2\) & 0.3 & 0.416 & 0.954 \\
\hline J1908+0909 & 2,1400 & 336.53 & 464.5 & 43.0 & 0.5 & 1175 & \(4.9 \pm 0.9\) & 1.14 & \(2.4 \pm 0.5\) & 1 & 0.757 & 1.655 \\
\hline & & & & & & 1475 & <3 & & . \({ }^{b}\) & & 0.279 & 0.609 \\
\hline J1908+0916 & 1,400 & 830.31 & 250.0 & 43.1 & 0.6 & 430 & .. . \({ }^{b}\) & & \(12.7 \pm 2.1\) & 0.04 & 12.87 & 25.13 \\
\hline J1909+0007 & 1,400 & 1016.96 & 112.9 & 35.0 & -3.9 & 430 & <2 & & <1 & & 1.006 & 0.195 \\
\hline J1909+0254 & 1,400 & 989.85 & 172.1 & 37.5 & -2.6 & 430 & <2.7 & & <1.1 & & 5.03 & 1.995 \\
\hline J1909+0616 & 2,1400 & 251.98 & 348.6 & 40.5 & -1.0 & 1175 & ... \({ }^{d}\) & & ... \({ }^{d}\) & & 46.15 & 69.79 \\
\hline J1909+0912 & 2,1400 & 222.93 & 421.4 & 43.1 & 0.3 & 1175 & \(5.1 \pm 1.2\) & 0.04 & \(2.2 \pm 0.5\) & 0.03 & 0.675 & 1.681 \\
\hline J1909+1102 & 1,400 & 283.65 & 148.4 & 44.7 & 1.2 & 1475 & \(1.5 \pm 0.1\) & 2.4 & \(0.33 \pm 0.03\) & 5.2 & 0.248 & 1.076 \\
\hline
\end{tabular}
\end{table}

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 2-Continued}
\begin{tabular}{|l|l|l|l|l|l|l|l|l|l|l|l|l|}
\hline PSR & Ref \({ }^{\text {a }}\) & Period (ms) & DM ( \(\mathrm{pc} \mathrm{cm} \mathrm{cm}^{-3}\) ) & \(l \left({ }^{\circ}\right)\) & \(b \left({ }^{\circ}\right)\) & & & & & & & \\
\hline & & & & & & \(\nu\) (MHz) & \(\mathrm{PBF}_{1} \tau_{d}(\mathrm{~ms})\) & \(f_{r}\) & \(\mathrm{PBF}_{2} \tau_{d}(\mathrm{~ms})\) & \(f_{r}\) & \({ }^{\tau} d, t c 93\) (ms) & \({ }^{\tau}\) d,ne2001 (ms) \\
\hline (1) & (2) & (3) & (4) & (5) & (6) & (7) & (8) & (9) & (10) & (11) & (12) & (13) \\
\hline \multirow[b]{2}{*}{J1909+1450} & \multirow[b]{2}{*}{1,400} & \multirow[b]{2}{*}{996.04} & \multirow[b]{2}{*}{119.5} & \multirow[b]{2}{*}{48.1} & \multirow[b]{2}{*}{2.9} & 1175 & \(<1\) & & ... \({ }^{b}\) & \multirow{2}{*}{} & 0.016 & 0.013 \\
\hline & & & & & & 1175 & \(\ldots d\) & & \(\ldots d\) & & 0.003 & 0.012 \\
\hline J1910+0358 & 1,300 & 2330.30 & 78.8 & 38.6 & -2.3 & 430 & < 4.7 & & <5.7 & \multirow{3}{*}{0.1} & 0.14 & 0.114 \\
\hline \multirow[t]{2}{*}{J1910+0534} & 2,1400 & 452.83 & 484.0 & 40.0 & -1.6 & 1175 & \(12.5 \pm 3.8\) & 0.7 & \(5.4 \pm 2\) & & 0.676 & 0.773 \\
\hline & & & & & & 1475 & <6.2 & & <2.2 & & 0.249 & 0.284 \\
\hline J1910+0714 & 1,400 & 2712.51 & 124.1 & 41.5 & -0.8 & 430 & \(9.3 \pm 2.6\) & 0.01 & <1.1 & & 0.887 & 1.628 \\
\hline \multirow[t]{2}{*}{J1910+1231} & 1,400 & 1441.81 & 274.4 & 46.2 & 1.6 & 430 & \(24.6 \pm 6.4\) & 0.06 & \(14.7 \pm 1.8\) & 0.2 & 10.59 & 8.278 \\
\hline & & & & & & 1175 & <3 & & <1.5 & & 0.127 & 0.099 \\
\hline \multirow[t]{2}{*}{J1912+1036} & 1,400 & 409.38 & 147.0 & 44.7 & 0.3 & 430 & \(4.2 \pm 2.7\) & 0.04 & \(2.9 \pm 1.5\) & 0.03 & 1.317 & 1.933 \\
\hline & & & & & & 1175 & <3 & & <2 & & 0.013 & 0.023 \\
\hline J1913+0446 & 5,1400 & 1616.09 & 109.1 & 39.7 & -2.6 & 1175 & <2 & & <1.4 & & 0.008 & 0.005 \\
\hline \multirow[t]{2}{*}{\(\mathrm{J} 1913+0832^{\mathrm{m}}\)} & 2,1400 & 134.40 & 359.5 & 43.0 & -0.9 & 1175 & \(7.7 \pm 1.7\) & 0.7 & \(5.1 \pm 1.4\) & 0.1 & 0.466 & 0.734 \\
\hline & & & & & & 1475 & \(2.3 \pm 1\) & 0.03 & . . . \(b\) & & 0.171 & 0.27 \\
\hline \multirow[t]{2}{*}{J1913+0832 \({ }^{\text {i }}\)} & & & & & & 1175 & \(6.6 \pm 1.8\) & \multirow[t]{2}{*}{0.12} & \(3 \pm 1.4\) & 0.2 & 0.509 & 0.905 \\
\hline & & & & & & 1475 & <2 & & <1 & & 0.187 & 0.333 \\
\hline J1913+0936 & 1,400 & 1242.06 & 157.0 & 43.9 & -0.4 & 430 & <1.3 & & <0.4 & & 1.874 & 35.38 \\
\hline \multirow[t]{2}{*}{J1913+1000} & 3,1400 & 837.21 & 419.4 & 44.3 & -0.2 & 1175 & \(11.1 \pm 4\) & 0.03 & \(5 \pm 3\) & 0.03 & 0.556 & 16.94 \\
\hline & & & & & & 1475 & <6 & & <3 & & 0.204 & 6.228 \\
\hline \multirow[t]{2}{*}{J1913+1011} & \multirow[t]{2}{*}{2,1400} & 35.91 & \multirow[t]{2}{*}{178.9} & \multirow[t]{2}{*}{44.4} & -0.1 & 430 & \(2 \pm 1\) & 0.3 & \(0.9 \pm 0.7\) & 0.4 & 2.986 & 107.9 \\
\hline & & & & & & 1175 & <0.4 & & <0.2 & & 0.036 & 1.294 \\
\hline \multirow[t]{2}{*}{J1913+1145} & \multirow[t]{2}{*}{2,1400} & \multirow[t]{2}{*}{306.05} & \multirow[t]{2}{*}{630.7} & \multirow[t]{2}{*}{45.8} & \multirow[t]{2}{*}{0.6} & 1175 & \(9.2 \pm 1\) & 0.3 & \(5.6 \pm 1.3\) & 0.1 & 1.568 & 1.204 \\
\hline & & & & & & 1475 & \(4.3 \pm 0.5\) & 0.2 & \(2.2 \pm 0.9\) & 0.01 & 0.577 & 0.443 \\
\hline \multirow[t]{2}{*}{J1913+1400} & \multirow[t]{2}{*}{1,400} & \multirow[t]{2}{*}{521.50} & \multirow[t]{2}{*}{144.4} & \multirow[t]{2}{*}{47.8} & \multirow[t]{2}{*}{1.7} & 430 & \(3.3 \pm 0.6\) & 1.7 & <0.53 & & 0.706 & 2.126 \\
\hline & & & & & & 1175 & <2 & & <1 & & 0.009 & 0.026 \\
\hline J1914+1122 & 1,400 & 600.96 & 80.0 & 45.6 & 0.2 & 1175 & \(\ldots d\) & & \(\ldots d\) & & 0.001 & 0.003 \\
\hline J1915+0839 & 3,1400 & 342.77 & 369.1 & 43.4 & -1.2 & 1175 & <6 & & <2 & & 0.396 & 0.46 \\
\hline J1915+0738 & 1,400 & 1542.73 & 39.0 & 42.4 & -1.7 & 430 & ... \({ }^{c}\) & & ... \({ }^{c}\) & & 0.012 & 0.002 \\
\hline \multirow[t]{2}{*}{J1915+1009} & 1,400 & 404.56 & 246.1 & 44.6 & -0.6 & 430 & \(15.4 \pm 1.3\) & 6.9 & \(11.1 \pm 1\) & 0.9 & 10.20 & 20.27 \\
\hline & & & & & & 1175 & <1.1 & & <0.4 & & 0.122 & 0.243 \\
\hline J1915+1606 & 1,400 & 59.06 & 168.8 & 49.9 & 2.2 & 1175 & \(0.33 \pm 0.1\) & 0.4 & <0.07 & & 0.007 & 0.021 \\
\hline \multirow[t]{2}{*}{J1916+0844} & 3,1400 & 440.03 & 339.0 & 43.6 & -1.1 & 1175 & \(7.7 \pm 1\) & 0.1 & \(4 \pm 0.6\) & 1.4 & 0.357 & 0.428 \\
\hline & & & & & & 1475 & \(3.6 \pm 0.9\) & 0.1 & \(1.1 \pm 0.4\) & 0.3 & 0.131 & 0.158 \\
\hline J1916+0951 & 1,400 & 270.26 & 61.4 & 44.5 & -0.9 & 430 & \(1 \pm 0.4\) & 0.6 & <0.6 & & 0.037 & 0.025 \\
\hline \multirow[t]{2}{*}{J1916+1030} & 1,400 & 628.92 & 387.0 & 45.1 & -0.6 & 1175 & \(9.2 \pm 2.9\) & 0.02 & \(4.8 \pm 1.5\) & 0.02 & 0.4 & 0.714 \\
\hline & & & & & & 1475 & <4 & & <2 & & 0.147 & 0.262 \\
\hline J1916+1312 & 1,400 & 281.86 & 236.9 & 47.5 & 0.7 & 430 & \(2.8 \pm 0.1\) & 5.7 & \(0.9 \pm 0.4\) & 2.4 & 5.958 & 12.93 \\
\hline & & & & & & 1175 & <1 & & <0.4 & & 0.071 & 0.155 \\
\hline J1917+1353 & 1,300 & 194.62 & 94.5 & 48.2 & 0.8 & 1175 & \(1.2 \pm 0.4\) & 0.3 & \(0.4 \pm 0.1\) & 0.4 & 0.001 & 0.011 \\
\hline J1918+1444 & 1,400 & 1181.08 & 30.0 & 49.0 & 0.9 & 430 & ... \({ }^{c}\) & & ... \({ }^{c}\) & & 0.006 & 0.001 \\
\hline J1918+1541 & 1,400 & 370.86 & 13.0 & 49.9 & 1.4 & 1175 & \(\ldots{ }^{c}\) & & \(\ldots{ }^{c}\) & & \(\ll 1\) & \(\ll 1\) \\
\hline \multirow[t]{2}{*}{J1920+1110} & \multirow[t]{2}{*}{2,1400} & 509.85 & 181.1 & 46.1 & -1.2 & 1175 & \(14 \pm 3.3\) & 0.02 & \(7.4 \pm 1.4\) & 0.01 & 0.032 & 0.06 \\
\hline & & & & & & 1475 & \(6.3 \pm 2.1\) & 0.4 & \(2.9 \pm 1.7\) & 0.5 & 0.012 & 0.022 \\
\hline \multirow[t]{2}{*}{J1921+1419} & \multirow[t]{2}{*}{1,400} & 618.14 & 91.9 & 49.0 & 0.1 & 1175 & \(6 \pm 4\) & 0.4 & \(3.2 \pm 1.9\) & 0.9 & 0.002 & 0.01 \\
\hline & & & & & & 1475 & \(4 \pm 3\) & 0.3 & \(2.9 \pm 2.2\) & 0.05 & 0.001 & 0.004 \\
\hline J1921+2003 & 4,400 & 760.70 & 101.0 & 54.1 & 2.8 & 430 & \(\ldots{ }^{b}\) & & <1.2 & & 0.096 & 0.07 \\
\hline J1923+1706 & 4,400 & 547.23 & 142.5 & 51.7 & 1.0 & 430 & ... \({ }^{b}\) & & <1 & & 0.406 & 1.23 \\
\hline J1926+1434 & 1,400 & 1324.99 & 205.0 & 49.8 & -0.8 & 430 & \(8.4 \pm 1.3\) & 4.2 & <7.4 & & 1.934 & 5.43 \\
\hline J1926+1928 & 4,400 & 1346.05 & 445.0 & 54.1 & 1.5 & 430 & \(\ldots{ }^{b}\) & & \(34.5 \pm 6.8\) & 0.4 & 11.46 & 3.46 \\
\hline \multirow[t]{2}{*}{J1927+1852} & 4,400 & 482.79 & 254.0 & 53.7 & 1.0 & 430 & \(\ldots{ }^{b}\) & & \(5.8 \pm 1.3\) & 0.5 & 2.754 & 1.97 \\
\hline & & & & & & 1175 & ... \({ }^{b}\) & & <1 & & 0.033 & 0.024 \\
\hline J1927+1856 & 4,400 & 298.34 & 90.0 & 53.8 & 1.0 & 430 & ... \({ }^{b}\) & & <0.2 & & 0.09 & 0.09 \\
\hline J1929+1844 & 4,400 & 1220.38 & 109.0 & 53.8 & 0.5 & 1175 & ... \({ }^{b}\) & & <1.1 & & 0.002 & 0.003 \\
\hline \multirow[t]{2}{*}{J1930+1316} & 1,400 & 760.05 & 207.3 & 49.1 & -2.3 & 430 & ... \({ }^{b}\) & & \(4.1 \pm 2.4\) & 0.1 & 1.526 & 1.89 \\
\hline & & & & & & 1175 & ... \({ }^{b}\) & & <1.2 & & 0.018 & 0.023 \\
\hline J1931+1536 & 4,400 & 314.37 & 140.0 & 51.3 & -1.4 & 430 & ... \({ }^{b}\) & & \(0.9 \pm 0.2\) & 0.01 & 0.33 & 0.87 \\
\hline J1933+1304 & 4,400 & 928.38 & 177.9 & 49.3 & -3.1 & 430 & ... \({ }^{b}\) & & <0.25 & & 0.660 & 0.334 \\
\hline J1935+1745 & 4,400 & 654.44 & 214.6 & 53.6 & -1.2 & 430 & \(15.9 \pm 1.5\) & 1.2 & \(9.2 \pm 3.3\) & 0.02 & 0.751 & 2.72 \\
\hline & & & & & & 1175 & <1.1 & & <1.2 & & 0.008 & 0.033 \\
\hline J1942+1743 & 4,400 & 696.30 & 190.0 & 54.4 & -2.7 & 430 & ... & & \(5 \pm 1.5\) & 0.5 & 0.403 & 0.29 \\
\hline J1944+1755 & 4,400 & 1996.78 & 175.0 & 54.8 & -3.0 & 430 & ... \({ }^{b}\) & & <5 & & 0.308 & 0.215 \\
\hline J1945+1834 & 4,400 & 1068.80 & 215.0 & 55.5 & -2.9 & 430 & ... \({ }^{b}\) & & \(3.3 \pm 1.3\) & 0.2 & 0.7 & 0.524 \\
\hline J2027+2146 & 4,400 & 398.20 & 96.8 & 63.5 & -9.5 & 430 & <0.4 & & <0.4 & & 0.091 & 0.041 \\
\hline
\end{tabular}
\end{table}

\footnotetext{
\({ }^{\mathrm{a}}\) Reference(s) to pulsar parameters, followed by the frequency band (in MHz ) of the survey that discovered the pulsar. (1) ATNF pulsar catalog, available at http://www.atnf.csiro.au/research/pulsar/psrcat; (2) Morris et al. (2002); (3) Hobbs et al. (2004), in preparation; (4) Lorimer et al. (2002); (5) Kramer et al. (2003). Note that pulsars in (2), (3), and (5) (Parkes multibeam survey discoveries), as well as in (4), are also available in (1).
\({ }^{\mathrm{b}}\) The PBF yields unphysical residuals from the deconvolution for any realistic value of \(\tau_{d}\) (see text).
\({ }^{\mathrm{c}}\) Signal-to-noise ratio is too small to allow a meaningful fit to the PBF.
\({ }^{\mathrm{d}} \tau_{d}\) is negligibly small.
\({ }^{\mathrm{e}}\) The pulsar is not detected at this frequency.
\({ }^{m},^{i}\) Main pulse (m) and inter pulse (i) of the pulsar (see also Fig. 1).
}

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 3
Frequency scaling indices from measurements of pulse-broadening times.}
\begin{tabular}{|l|l|l|l|l|l|l|}
\hline \multirow{2}{*}{PSR} & \multirow[b]{2}{*}{\(\nu_{1}\) (MHz)} & \multirow[b]{2}{*}{\(\nu_{2}\) (MHz)} & \multicolumn{2}{|c|}{\(\mathrm{PBF}_{1}\) deconvolution} & \multicolumn{2}{|c|}{\(\mathrm{PBF}_{2}\) deconvolution} \\
\hline & & & \(\alpha_{1}\) & \(\beta_{1}\) & \(\alpha_{2}\) & \(\beta_{2}\) \\
\hline (1) & (2) & (3) & (4) & (5) & (6) & (7) \\
\hline J1849+0127 & 430 & 1175 & \(2.5 \pm 0.1\) & . . \({ }^{a}\) & ⋯ & ⋯ \\
\hline J1852+0031 & 1175 & 1475 & \(3.5 \pm 0.1\) & \(4.7 \pm 0.4\) & \(3.3 \pm 0.1\) & \(5 \pm 0.2\) \\
\hline \(\mathrm{J} 1853+0505\) & 1175 & 1475 & \(3.7 \pm 0.2\) & \(4.4 \pm 0.5\) & ⋯ & ⋯ \\
\hline J1853+0545 & 1175 & 1475 & \(2.8 \pm 0.3\) & \(6.8 \pm 1.5\) & \(4.4 \pm 0.3\) & \(3.7 \pm 0.5\) \\
\hline & 1475 & 2380 & \(3.2 \pm 0.1\) & \(5.3 \pm 0.5\) & \(4.2 \pm 0.2\) & \(3.8 \pm 0.3\) \\
\hline & 1175 & 2380 & \(3.1 \pm 0.1\) & \(5.7 \pm 0.3\) & \(4.3 \pm 0.1\) & \(3.8 \pm 0.1\) \\
\hline J1856+0404 & 1175 & 1475 & \(2.0 \pm 0.8\) & ... \({ }^{a}\) & \(2.4 \pm 0.7\) & .. \({ }^{a}\) \\
\hline \(\mathrm{J} 1857+0212\) & 1175 & 1475 & \(2.4 \pm 0.4\) & ... \({ }^{a}\) & \(3.9 \pm 1.4\) & \(4.1 \pm 3.2\) \\
\hline \(\mathrm{J} 1857+0526\) & 1175 & 1475 & \(3.7 \pm 0.4\) & \(4.3 \pm 1\) & \(3 \pm 0.6\) & \(6.1 \pm 2.6\) \\
\hline J1858+0215 & 1175 & 1475 & \(3.2 \pm 0.4\) & \(5.4 \pm 1.6\) & \(2.9 \pm 0.8\) & \(6.6 \pm 3.8\) \\
\hline J1906+0641 & 1175 & 1475 & \(2.3 \pm 0.5\) & . . \({ }^{a}\) & \(3.4 \pm 0.6\) & \(4.8 \pm 2\) \\
\hline J1908+0839 & 1175 & 1475 & \(3.4 \pm 0.5\) & \(4.9 \pm 1.7\) & \(4.0 \pm 0.4\) & \(4 \pm 1\) \\
\hline \(\mathrm{J} 1913+0832\) & 1175 & 1475 & \(5.3 \pm 0.8\) & \(3.2 \pm 1.1\) & ⋯ & ⋯ \\
\hline J1913+1145 & 1175 & 1475 & \(3.4 \pm 0.3\) & \(5 \pm 0.8\) & \(4.1 \pm 0.8\) & \(3.9 \pm 1.6\) \\
\hline J1916+0844 & 1175 & 1475 & \(3.3 \pm 0.4\) & \(5 \pm 1.5\) & \(5.7 \pm 0.6\) & \(3.1 \pm 0.7\) \\
\hline J1920+1110 & 1175 & 1475 & \(3.5 \pm 0.7\) & \(4.7 \pm 1.9\) & \(4.1 \pm 1\) & \(3.9 \pm 2.1\) \\
\hline J1921+1419 & 1175 & 1475 & \(1.8 \pm 1.2\) & . . \({ }^{a}\) & \(0.4 \pm 0.4\) & ... \({ }^{a}\) \\
\hline
\end{tabular}
\end{table}

\footnotetext{
\({ }^{\mathrm{a}}\) Implied values for \(\beta\) are unphysically large.
}

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 4
Estimates of scattering measures and constraints on the properties of clumps.}
\begin{tabular}{|l|l|l|l|l|l|l|l|l|l|}
\hline PSR (1) & DM ( \(\mathrm{pc} \mathrm{cm}^{-3}\) ) & D (kpc) & \(l\) (deg) & \(b\) (deg) & Freq. (MHz) & \(\mathrm{SM}_{\text {meas }}\) ( \(\mathrm{kpc} \mathrm{m}^{-20 / 3}\) ) & \(\mathrm{SM}_{n e 2001}\) ( \(\mathrm{kpc} \mathrm{m}^{-20 / 3}\) ) & \(\delta\) SM ( \(\mathrm{kpc} \mathrm{m}^{-20 / 3}\) ) & \(F_{c}(\delta \mathrm{DM})^{2}\) (log) (10) \\
\hline J1849+0127 & 214.4 & 5.49 & 33.95 & 1.20 & 430 & 0.21 & 0.11 & 0.093 & -4.04 \\
\hline J1853+0546 & 197.2 & 5.37 & 38.24 & 2.28 & 1175 & 3.58 & 0.033 & 3.54 & -2.45 \\
\hline & & & & & 1475 & 4.79 & & 4.76 & -2.32 \\
\hline & & & & & 2380 & 7.67 & & 7.64 & -2.11 \\
\hline J1855+0422 & 438.6 & 8.39 & 37.22 & 1.20 & 1175 & 4.36 & 0.46 & 3.90 & -2.60 \\
\hline J1856+0404 & 345.3 & 6.85 & 37.07 & 0.84 & 1175 & 2.15 & 0.41 & 1.74 & -2.86 \\
\hline J1857+0526 & 468.3 & 8.92 & 38.40 & 1.24 & 1175 & 2.47 & 0.41 & 2.05 & -2.90 \\
\hline & & & & & 1475 & 2.33 & & 1.92 & -2.93 \\
\hline J1858+0215 & 702.0 & 10.02 & 35.68 & -0.43 & 1175 & 4.98 & 2.14 & 2.84 & -2.81 \\
\hline J1901+0413 & 367.0 & 6.81 & 37.77 & -0.20 & 430 & 4.32 & 0.58 & 3.75 & -2.53 \\
\hline J1903+0609 & 398.0 & 7.21 & 39.72 & 0.24 & 1175 & 0.73 & 0.59 & 0.14 & -3.98 \\
\hline J1904+0802 & 438.3 & 8.67 & 41.51 & 0.89 & 1175 & 0.68 & 0.36 & 0.32 & -3.69 \\
\hline J1905+0616 & 262.7 & 5.76 & 40.05 & -0.15 & 1175 & 3.35 & 0.22 & 3.13 & -2.53 \\
\hline J1908+0839 & 516.6 & 9.35 & 42.51 & 0.29 & 1175 & 1.07 & 0.57 & 0.51 & -3.53 \\
\hline J1908+0909 & 464.5 & 8.96 & 42.96 & 0.52 & 1175 & 0.99 & 0.40 & 0.60 & -3.44 \\
\hline J1909+0912 & 421.4 & 8.27 & 43.11 & 0.32 & 1175 & 1.11 & 0.44 & 0.67 & -3.36 \\
\hline J1910+0534 & 484.0 & 10.40 & 40.00 & -1.57 & 1175 & 1.92 & 0.19 & 1.73 & -3.04 \\
\hline J1912+0828 & 359.5 & 7.74 & 42.81 & -0.67 & 1175 & 0.80 & 0.28 & 0.53 & -3.43 \\
\hline \(\mathrm{J} 1913+0832\) & 359.5 & 7.93 & 42.98 & -0.86 & 1175 & 1.60 & 0.23 & 1.38 & -3.03 \\
\hline J1913+1145 & 630.7 & 14.56 & 45.83 & 0.63 & 1175 & 1.12 & 0.21 & 0.92 & -3.47 \\
\hline J1920+1110 & 181.1 & 5.60 & 46.11 & -1.16 & 1175 & 3.53 & 0.038 & 3.49 & -2.47 \\
\hline J1852+0031 & 680.0 & 7.91 & 33.46 & 0.11 & 1175 & 51.65 & 44.29 & 7.36 & -2.30 \\
\hline & & & & & 1475 & 61.52 & & 17.23 & -1.93 \\
\hline J1902+0556 & 179.7 & 4.70 & 39.41 & 0.36 & 430 & 0.09 & 0.078 & 0.013 & -4.84 \\
\hline J1909+1102 & 148.4 & 4.17 & 44.74 & 1.17 & 430 & 0.017 & 0.013 & 0.0038 & -5.31 \\
\hline J1910+0714 & 124.1 & 4.05 & 41.48 & -0.80 & 430 & 0.082 & 0.019 & 0.063 & -4.07 \\
\hline J1910+1231 & 274.4 & 7.73 & 46.17 & 1.63 & 430 & 0.11 & 0.044 & 0.065 & -4.34 \\
\hline J1913+1400 & 144.4 & 5.12 & 47.82 & 1.67 & 430 & 0.029 & 0.020 & 0.0089 & -5.03 \\
\hline J1915+1606 & 168.8 & 5.91 & 49.91 & 2.22 & 1175 & 0.15 & 0.015 & 0.13 & -3.91 \\
\hline J1916+1030 & 387.0 & 8.59 & 45.06 & -0.60 & 1175 & 1.74 & 0.21 & 1.54 & -3.01 \\
\hline J1926+1434 & 205.0 & 6.44 & 49.80 & -0.84 & 430 & 0.052 & 0.036 & 0.016 & -4.88 \\
\hline J1853+0505 & 273.6 & 6.49 & 37.64 & 1.97 & 1175 & 19.20 & 0.088 & 19.11 & -1.80 \\
\hline & & & & & 1475 & 22.17 & & 22.08 & -1.73 \\
\hline J1915+0856 & 339.0 & 7.96 & 43.56 & -1.11 & 1175 & 1.60 & 0.14 & 1.46 & -3.00 \\
\hline & & & & & 1475 & 1.93 & & 1.78 & -2.92 \\
\hline J1935+1745 & 214.6 & 6.93 & 53.63 & -1.21 & 430 & 0.082 & 0.019 & 0.06 & -4.30 \\
\hline
\end{tabular}
\end{table}
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=439&width=347&top_left_y=320&top_left_x=283)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=203&width=340&top_left_y=322&top_left_x=698)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=203&width=343&top_left_y=322&top_left_x=1108)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=203&width=340&top_left_y=322&top_left_x=1521)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=201&width=338&top_left_y=545&top_left_x=700)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=201&width=339&top_left_y=545&top_left_x=1112)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=201&width=338&top_left_y=545&top_left_x=1521)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=196&width=334&top_left_y=769&top_left_x=704)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=199&width=337&top_left_y=766&top_left_x=1112)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=203&width=340&top_left_y=986&top_left_x=1521)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=205&width=336&top_left_y=1207&top_left_x=702)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=203&width=341&top_left_y=986&top_left_x=1108)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=199&width=341&top_left_y=1209&top_left_x=1518)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=201&width=341&top_left_y=1430&top_left_x=287)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=199&width=338&top_left_y=1430&top_left_x=700)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=199&width=334&top_left_y=1209&top_left_x=1112)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=198&width=338&top_left_y=1652&top_left_x=700)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=199&width=336&top_left_y=1430&top_left_x=1110)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=201&width=338&top_left_y=1430&top_left_x=1521)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=205&width=340&top_left_y=1869&top_left_x=698)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=205&width=336&top_left_y=1645&top_left_x=1110)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=198&width=338&top_left_y=1652&top_left_x=1521)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=201&width=336&top_left_y=2092&top_left_x=702)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=200&width=338&top_left_y=1869&top_left_x=1108)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=200&width=338&top_left_y=1869&top_left_x=1521)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=196&width=334&top_left_y=2316&top_left_x=704)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=201&width=334&top_left_y=2092&top_left_x=1112)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=201&width=338&top_left_y=2092&top_left_x=1521)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=196&width=334&top_left_y=2316&top_left_x=1112)

\begin{tabular}{|l|l|}
\hline P1855+0422 \(\mathrm{P}=1678 \mathrm{~ms}\) & 2380 MHZ DM = 438.6 \\
\hline ![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-15.jpg?height=62\&width=172\&top_left_y=2432\&top_left_x=1521) & \\
\hline
\end{tabular}

\begin{tabular}{|c|r|}
\hline\(P 1901+0156\) \\
\(P=288 \mathrm{~ms}\) \\
& \\
& \\
\hline
\end{tabular}

\(\left.\begin{array}{r}01901+0331 \\
0=655 \mathrm{~ms}\end{array}\right\}\)\begin{tabular}{r}
430 MHZ \\
\(\mathrm{DM}=401.2\)
\end{tabular}
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-17.jpg?height=201&width=338&top_left_y=322&top_left_x=700)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-17.jpg?height=201&width=338&top_left_y=545&top_left_x=700)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-17.jpg?height=201&width=339&top_left_y=322&top_left_x=1110)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-17.jpg?height=201&width=340&top_left_y=322&top_left_x=1521)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-17.jpg?height=201&width=339&top_left_y=545&top_left_x=1112)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-17.jpg?height=199&width=339&top_left_y=766&top_left_x=1112)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-17.jpg?height=201&width=340&top_left_y=1207&top_left_x=1521)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-17.jpg?height=201&width=338&top_left_y=988&top_left_x=1108)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-17.jpg?height=201&width=343&top_left_y=1207&top_left_x=1108)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-17.jpg?height=203&width=342&top_left_y=764&top_left_x=1521)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-17.jpg?height=198&width=338&top_left_y=986&top_left_x=1521)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-17.jpg?height=200&width=341&top_left_y=1650&top_left_x=287)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-17.jpg?height=199&width=338&top_left_y=766&top_left_x=700)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-17.jpg?height=206&width=340&top_left_y=983&top_left_x=698)

\begin{tabular}{|c|r|}
\hline \(01902+0556\) & \begin{tabular}{r}
430 MHF \\
\(\mathrm{P}=747 \mathrm{~ms}\)
\end{tabular} \\
\(\mathrm{DM}=179.7\)
\end{tabular}
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-17.jpg?height=199&width=341&top_left_y=1207&top_left_x=287)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-17.jpg?height=203&width=338&top_left_y=1205&top_left_x=700)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-17.jpg?height=199&width=338&top_left_y=1428&top_left_x=700)

\begin{tabular}{|l|r|}
\hline \(01902+0556\) \\
\(P=747 \mathrm{~ms}\) \\
\hline
\end{tabular}
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-17.jpg?height=200&width=341&top_left_y=1650&top_left_x=1108)

\(01903+0135\)
\(\mathbf{P = 7 2 9} \mathrm{~ms}\)\(\quad\)\begin{tabular}{r}
430 MHZ \\
\(\mathrm{DM}=246.4\)
\end{tabular}
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-17.jpg?height=199&width=339&top_left_y=2313&top_left_x=287)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-17.jpg?height=196&width=338&top_left_y=1654&top_left_x=700)

\begin{tabular}{|l|r|}
\hline \(\mathbf{P 1 9 0 3 + 0 1 3 5}\) \\
\(\mathbf{P = 7 2 9} \mathbf{~ m s}\) \\
\hline
\end{tabular}

\(01903+0135\)
\(P=729 \mathrm{~ms}\)\(\quad\)\begin{tabular}{r}
1475 MHz \\
DM \(=246.4\)
\end{tabular}

\begin{tabular}{|l|l|}
\hline 1903+0601 \(\mathrm{P}=374 \mathrm{~ms}\) & 1475 MHZ DM \(=398.0\) \\
\hline
\end{tabular}

\(01904+0004\)
\(P=140 \mathrm{~ms}\)

\begin{tabular}{|c|c|}
\hline \(\mathbf{P 1 9 0 3 + 0 1 3 5}\) \\
\(\mathbf{P = 7 2 9 ~ m s}\)
\end{tabular}\(|\)\begin{tabular}{r}
2380 MHz \\
\(\mathrm{DM}=246.4\)
\end{tabular}

\begin{tabular}{|l|r|}
\hline\(P 1903+0601\) & \begin{tabular}{r}
2380 MHH \\
\(P=374 \mathrm{~ms}\)
\end{tabular} \\
& \(D M=398.0\) \\
& \\
& \\
\hline & \\
\hline
\end{tabular}

\begin{tabular}{|l|l|}
\hline & 2380 MHz DM = 233.7 \\
\hline 01904+0004 \(\mathrm{P}=140 \mathrm{~ms}\) & \\
\hline
\end{tabular}

\begin{tabular}{|l|l|l|l|}
\hline \multirow{2}{*}{} & ![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-18.jpg?height=199\&width=342\&top_left_y=324\&top_left_x=697) & ![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-18.jpg?height=199\&width=337\&top_left_y=324\&top_left_x=1111) & ![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-18.jpg?height=199\&width=337\&top_left_y=324\&top_left_x=1521) \\
\hline & ![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-18.jpg?height=196\&width=338\&top_left_y=548\&top_left_x=700) & ![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-18.jpg?height=200\&width=337\&top_left_y=544\&top_left_x=1111) & ![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-18.jpg?height=198\&width=337\&top_left_y=546\&top_left_x=1521) \\
\hline ![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-18.jpg?height=199\&width=339\&top_left_y=765\&top_left_x=288) & ![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-18.jpg?height=197\&width=338\&top_left_y=767\&top_left_x=700) & ![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-18.jpg?height=197\&width=337\&top_left_y=767\&top_left_x=1111) & \\
\hline \(\mathbf{P 1 9 0 5}+0616\)
\(\mathbf{P}=990 \mathrm{~ms}\) 430 MHz
\(\mathrm{DM}=262.7\) & ![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-18.jpg?height=197\&width=341\&top_left_y=988\&top_left_x=697) & ![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-18.jpg?height=197\&width=337\&top_left_y=988\&top_left_x=1111) & \\
\hline \begin{tabular}{l}
01905+0709 \(\mathrm{P}=648 \mathrm{~ms}\) \\
430 MHz DM =269.0
\end{tabular} & \begin{tabular}{l}
01905+0709 \\
1175 MHz \\
\(\mathrm{DM}=269.0\)
\end{tabular} & \begin{tabular}{l}
01905+0709 \\
\(1475 \mathrm{MH}_{2}^{2}\) \\
DM \(=269.0\)
\end{tabular} & ![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-18.jpg?height=201\&width=341\&top_left_y=1210\&top_left_x=1521) \\
\hline & \begin{tabular}{l}
\(01906+0641\) \\
1175 MHz \(\mathrm{P}=267 \mathrm{~ms}\) \\
DM \(=473.0\)
\end{tabular} & \begin{tabular}{l}
\(01906+0641\) \\
1475 MHz \\
DM \(=473.0\)
\end{tabular} & ![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-18.jpg?height=196\&width=337\&top_left_y=1431\&top_left_x=1521) \\
\hline & \begin{tabular}{l}
P1906+0912 \\
\(1175 \mathrm{MH}_{2}^{2} \mathrm{DM}=260.5\)
\end{tabular} & \begin{tabular}{l}
P1906+0912 \\
\(1475 \mathrm{MH}_{2}^{2}\) \\
\(\mathrm{DM}=260.5\)
\end{tabular} & ![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-18.jpg?height=201\&width=341\&top_left_y=1651\&top_left_x=1521) \\
\hline & ![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-18.jpg?height=194\&width=338\&top_left_y=1876\&top_left_x=700) & ![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-18.jpg?height=196\&width=337\&top_left_y=1874\&top_left_x=1111) & ![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-18.jpg?height=194\&width=337\&top_left_y=1876\&top_left_x=1521) \\
\hline \begin{tabular}{l}
1907+0740 \\
430 MHz \(\mathrm{P}=575 \mathrm{~ms}\) \\
\(\mathrm{DM}=329.3\)
\end{tabular} & \begin{tabular}{|l|r|}
\hline \(1907+0740\) & \begin{tabular}{r}
1175 MHz \\
\(\mathrm{P}=575 \mathrm{~ms}\)
\end{tabular} \\
& \(\mathrm{DM}=329.3\) \\
\hline
\end{tabular} & \begin{tabular}{|l|r|}
\hline \(1907+0740\) & 1475 MHz \\
& \\
& \\
& \\
& \\
& \\
&
\end{tabular} & ![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-18.jpg?height=194\&width=337\&top_left_y=2096\&top_left_x=1521) \\
\hline \begin{tabular}{l}
01907+0918 \\
430 MHz \\
\(\mathrm{P}=226 \mathrm{~ms}\) \\
DM = 358.0
\end{tabular} & \begin{tabular}{l}
01907+0918 \\
1175 MHZ \\
DM \(=357.9\)
\end{tabular} & ![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-18.jpg?height=196\&width=341\&top_left_y=2317\&top_left_x=1107) & \begin{tabular}{|l|r|}
\hline \(01907+0918\) & 2380 MHz \\
\hline\(P=226 \mathrm{~ms}\) & \(D M=358.0\) \\
& \\
& \\
\hline
\end{tabular} \\
\hline
\end{tabular}
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=424&width=345&top_left_y=322&top_left_x=285)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=203&width=342&top_left_y=322&top_left_x=698)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=203&width=343&top_left_y=322&top_left_x=1108)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=205&width=345&top_left_y=322&top_left_x=1518)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=201&width=334&top_left_y=545&top_left_x=704)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=201&width=341&top_left_y=545&top_left_x=1110)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=201&width=340&top_left_y=545&top_left_x=1521)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=201&width=338&top_left_y=766&top_left_x=700)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=203&width=341&top_left_y=764&top_left_x=1110)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=201&width=338&top_left_y=766&top_left_x=1521)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=201&width=334&top_left_y=988&top_left_x=704)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=201&width=343&top_left_y=988&top_left_x=1108)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=201&width=338&top_left_y=988&top_left_x=1521)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=199&width=338&top_left_y=1209&top_left_x=700)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=199&width=336&top_left_y=1209&top_left_x=1110)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=201&width=338&top_left_y=1207&top_left_x=1521)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=199&width=341&top_left_y=1430&top_left_x=287)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=199&width=338&top_left_y=1430&top_left_x=700)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=199&width=336&top_left_y=1430&top_left_x=1110)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=205&width=341&top_left_y=1645&top_left_x=287)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=198&width=338&top_left_y=1652&top_left_x=700)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=198&width=341&top_left_y=1652&top_left_x=1108)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=200&width=339&top_left_y=1869&top_left_x=287)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=196&width=338&top_left_y=1873&top_left_x=700)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=200&width=339&top_left_y=1869&top_left_x=1110)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=203&width=338&top_left_y=1869&top_left_x=1521)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=201&width=339&top_left_y=2092&top_left_x=287)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=201&width=338&top_left_y=2092&top_left_x=700)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=201&width=338&top_left_y=2092&top_left_x=1108)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=196&width=338&top_left_y=2316&top_left_x=700)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=196&width=334&top_left_y=2316&top_left_x=1112)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-19.jpg?height=196&width=338&top_left_y=2316&top_left_x=1521)

\begin{tabular}{r|r|}
\hline \\
\hline
\end{tabular}
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-20.jpg?height=201&width=343&top_left_y=545&top_left_x=287)

\begin{tabular}{|l|r|}
\hline \(11909+1102\) & \begin{tabular}{r}
1175 MHH \\
\(\mathrm{P}=284 \mathrm{~ms}\) \\
DM
\end{tabular} \\
\hline
\end{tabular}

\begin{tabular}{r|r|r}
\(81910+0358\) & \begin{tabular}{r}
1175 \\
\(\mathrm{P}=2330 \mathrm{~ms}\) \\
DM
\end{tabular} \\
\hline
\end{tabular}

\begin{tabular}{|c|c|}
\hline \(01909+1102\) & \begin{tabular}{r}
1475 \\
\(P=284 \mathrm{~ms}\) \\
DM
\end{tabular} \\
\hline
\end{tabular}
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-20.jpg?height=201&width=341&top_left_y=545&top_left_x=1108)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-20.jpg?height=203&width=343&top_left_y=764&top_left_x=1108)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-20.jpg?height=206&width=343&top_left_y=983&top_left_x=1108)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-20.jpg?height=201&width=343&top_left_y=1207&top_left_x=1108)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-20.jpg?height=205&width=343&top_left_y=1426&top_left_x=1108)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-20.jpg?height=207&width=343&top_left_y=1645&top_left_x=1108)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-20.jpg?height=205&width=343&top_left_y=1869&top_left_x=1108)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-20.jpg?height=203&width=343&top_left_y=2092&top_left_x=1108)

\begin{tabular}{|l|l|}
\hline 51913+1145 \(\mathrm{P}=306 \mathrm{~ms}\) & ![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-20.jpg?height=37\&width=157\&top_left_y=2458\&top_left_x=1290) \\
\hline
\end{tabular}
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-20.jpg?height=203&width=345&top_left_y=543&top_left_x=1516)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-20.jpg?height=205&width=345&top_left_y=764&top_left_x=1516)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-20.jpg?height=201&width=343&top_left_y=1207&top_left_x=1518)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-20.jpg?height=203&width=343&top_left_y=1428&top_left_x=1518)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-20.jpg?height=202&width=343&top_left_y=1650&top_left_x=1518)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-20.jpg?height=205&width=345&top_left_y=1869&top_left_x=1516)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-20.jpg?height=203&width=345&top_left_y=2092&top_left_x=1518)

\begin{tabular}{|l|l|}
\hline P1913+1145 \(\mathbf{P = 3 0 6 ~ m s}\) & ![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-20.jpg?height=57\&width=166\&top_left_y=2440\&top_left_x=1693) \\
\hline
\end{tabular}

\begin{tabular}{r|r|}
\hline \(01913+1400\) \\
\(0=521 \mathrm{~ms}\)
\end{tabular}\(|\)\begin{tabular}{r}
430 MHz \\
\(\mathrm{DM}=144.4\)
\end{tabular}

\(01913+1400\)
\(P=521 \mathrm{~ms}\)\(|\)\begin{tabular}{r}
1175 MHZ \\
DM \(=144.4\)
\end{tabular}

\begin{tabular}{|c|c|}
\hline \(01913+1400\) & \begin{tabular}{r}
1475 MHz \\
\(\mathrm{P}=521 \mathrm{~ms}\) \\
\(\mathrm{DM}=144.4\)
\end{tabular} \\
\hline
\end{tabular}
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=201&width=340&top_left_y=322&top_left_x=1521)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=201&width=334&top_left_y=545&top_left_x=704)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=201&width=337&top_left_y=545&top_left_x=1112)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=201&width=338&top_left_y=545&top_left_x=1521)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=199&width=341&top_left_y=766&top_left_x=287)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=196&width=338&top_left_y=769&top_left_x=700)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=199&width=337&top_left_y=766&top_left_x=1112)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=201&width=338&top_left_y=988&top_left_x=1521)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=199&width=341&top_left_y=1209&top_left_x=287)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=201&width=338&top_left_y=988&top_left_x=700)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=201&width=339&top_left_y=988&top_left_x=1110)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=199&width=338&top_left_y=1209&top_left_x=700)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=197&width=334&top_left_y=1209&top_left_x=1112)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=199&width=338&top_left_y=1209&top_left_x=1521)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=197&width=339&top_left_y=1430&top_left_x=287)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=197&width=338&top_left_y=1430&top_left_x=700)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=197&width=336&top_left_y=1430&top_left_x=1110)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=199&width=338&top_left_y=1430&top_left_x=1521)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=196&width=334&top_left_y=1654&top_left_x=704)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=198&width=336&top_left_y=1652&top_left_x=1110)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=203&width=341&top_left_y=1869&top_left_x=287)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=200&width=338&top_left_y=1869&top_left_x=700)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=200&width=339&top_left_y=1869&top_left_x=1110)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=200&width=338&top_left_y=1869&top_left_x=1521)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=201&width=338&top_left_y=2092&top_left_x=700)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=201&width=338&top_left_y=2092&top_left_x=1108)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=196&width=339&top_left_y=2316&top_left_x=287)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=198&width=338&top_left_y=2316&top_left_x=700)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=196&width=334&top_left_y=2316&top_left_x=1112)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-21.jpg?height=196&width=338&top_left_y=2316&top_left_x=1521)

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-22.jpg?height=1312&width=1580&top_left_y=678&top_left_x=283}
\captionsetup{labelformat=empty}
\caption{Fig. 1.-Integrated pulse profiles of pulsars from Arecibo observations at \(430,1175,1475\) and 2380 MHz . Data at 430 MHz were taken with the PSPM, and those at higher frequencies were taken with the WAPP. All profiles are plotted with a pulse phase resolution of 2 milli-periods, where consistent with the data acquisition time resolution (Table 1). The highest point in the profile is placed at phase 0.5 . The pulsar ID, period and the dispersion measure are indicated at the top of each panel, along with the center frequency of observation. Objects with labels (top left) starting with ' P ' refer to new discoveries from the Parkes multibeam survey, and those with ' J ' are previously known pulsars.}
\end{figure}
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-23.jpg?height=308&width=487&top_left_y=322&top_left_x=283)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-23.jpg?height=310&width=500&top_left_y=322&top_left_x=820)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-23.jpg?height=308&width=484&top_left_y=322&top_left_x=1377)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-23.jpg?height=304&width=474&top_left_y=640&top_left_x=296)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-23.jpg?height=304&width=488&top_left_y=640&top_left_x=829)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-23.jpg?height=306&width=489&top_left_y=638&top_left_x=1370)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-23.jpg?height=304&width=493&top_left_y=953&top_left_x=283)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-23.jpg?height=311&width=491&top_left_y=951&top_left_x=829)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-23.jpg?height=313&width=484&top_left_y=949&top_left_x=1375)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-23.jpg?height=306&width=484&top_left_y=1267&top_left_x=292)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-23.jpg?height=306&width=487&top_left_y=1267&top_left_x=833)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-23.jpg?height=308&width=482&top_left_y=1263&top_left_x=1377)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-23.jpg?height=306&width=493&top_left_y=1581&top_left_x=283)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-23.jpg?height=306&width=491&top_left_y=1581&top_left_x=829)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-23.jpg?height=309&width=489&top_left_y=1576&top_left_x=1370)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-23.jpg?height=301&width=487&top_left_y=1897&top_left_x=283)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-23.jpg?height=306&width=493&top_left_y=1897&top_left_x=827)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-23.jpg?height=309&width=495&top_left_y=1894&top_left_x=1370)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-23.jpg?height=308&width=495&top_left_y=2208&top_left_x=281)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-23.jpg?height=306&width=493&top_left_y=2210&top_left_x=827)
![](https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-23.jpg?height=304&width=489&top_left_y=2212&top_left_x=1374)

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-24.jpg?height=639&width=1597&top_left_y=1031&top_left_x=275}
\captionsetup{labelformat=empty}
\caption{FIG. 2.- Examples of the intrinsic pulse shapes (light solid [red] curves with highest peaks) and the best fit PBFs (solid [blue] curves rising from zero at left of each panel) obtained by application of the CLEAN method; the PBF is assumed to be a simple one-sided exponential (PBF \({ }_{1}\), appropriate to a thin slab scattering geometry). The amplitudes of both the PBFs and the measured profiles (heavy solid [green] curves) are normalized to unity, and the areas under the intrinsic and measured pulse profiles are identical.}
\end{figure}

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-25.jpg?height=2195&width=1595&top_left_y=283&top_left_x=275}
\captionsetup{labelformat=empty}
\caption{Fig. 3.- Similar to Figure 2, except that the pulse broadening function employed by CLEAN has a more rounded shape ( \(\mathrm{PBF}_{2}\), due to a uniform scattering medium between the pulsar and the Earth).}
\end{figure}

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-26.jpg?height=1196&width=1588&top_left_y=734&top_left_x=277}
\captionsetup{labelformat=empty}
\caption{Fig. 4.- Measurements of pulse-broadening times plotted against dispersion measures. The new measurements are shown as filled circles. The open circles with crosses ( \(\mathrm{DM} \leq 200 \mathrm{pc} \mathrm{cm}^{-3}\) ) are derived from the measurements of decorrelation bandwidths, while the open circles are published \(\tau_{d}\) measurements. The solid curve represents the best fit model for the empirical relation between \(\tau_{d}\) and DM, the frequencyindependent coefficients for which are only slightly different from those obtained by Cordes \& Lazio (2002b) based on the published data alone (see § 5.2 for details).}
\end{figure}

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-27.jpg?height=1337&width=798&top_left_y=659&top_left_x=674}
\captionsetup{labelformat=empty}
\caption{Fig. 5.- Analysis of the frequency dependence of pulse broadening that takes into account an inner scale for the wavenumber spectrum of electron-density irregularities. (Top panel) Plot of \(\delta \alpha\), the difference in exponent in the relation \(\tau_{d} \propto \nu^{-\alpha}\) above and below a break point defined by the composite quantity \(\tau_{d, \text { cross }} \nu^{2} / D\). We calculate the best fit values of \(\alpha\) for data points above and below the break point and calculate \(\delta \alpha\) as a function of \(\tau_{d, \text { cross }} \nu^{2} / D\). The units of \(\tau_{d, \text { cross }} \nu^{2} / D\) are ( \(\mathrm{ms} \mathrm{GHz}^{2} \mathrm{kpc}^{-1}\) ). (Bottom panel) \(\chi^{2}\) for the fit as a function of \(\tau_{d, \text { cross }} \nu^{2} / D\), defined here as the sum of the squares of data-model (see text).}
\end{figure}

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-28.jpg?height=1195&width=1593&top_left_y=739&top_left_x=277}
\captionsetup{labelformat=empty}
\caption{FIG. 6.- Measurements of pulse-broadening times plotted against the predictions from the new electron density model NE2001 (Cordes \& Lazio 2002a). The filled circles are the published measurements. The new measurements from our observations are shown as open circles. All measurements are scaled to a common frequency of 1 GHz using \(\tau_{d} \propto \nu^{-4.4}\). The dashed line is of unity slope. As evident from the figure, a significant number of both the published and new measurements are well above the dashed line, which implies that the model tends to underestimate the degree of scattering toward many lines of sight.}
\end{figure}

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-29.jpg?height=1198&width=1593&top_left_y=751&top_left_x=277}
\captionsetup{labelformat=empty}
\caption{Fig. 7.- Measurements of pulse-broadening times plotted against the predictions from the electron density model TC93 (Taylor \& Cordes 1993). The filled circles are the published measurements, and the new measurements from our observations are shown as open circles. All measurements are scaled to a common frequency of 1 GHz using \(\tau_{d} \propto \nu^{-4.4}\). The dashed line is of unity slope. As for Figure 6, the model tends to underestimate the degree of scattering toward many lines of sight.}
\end{figure}

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-30.jpg?height=1619&width=1590&top_left_y=554&top_left_x=277}
\captionsetup{labelformat=empty}
\caption{Fig. 8.- Estimates of scattering measure (SM) derived from all pulse-broadening data available. The size of the symbol is proportional to \(\log\) (SM). Pulsar positions are projected onto the Galactic plane; filled circles represent the published data, and the new measurements from our observations are shown as open circles. The spiral arm locations are adopted from the NE2001 model of Cordes \& Lazio (2002a,b).}
\end{figure}

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-31.jpg?height=1621&width=1593&top_left_y=554&top_left_x=277}
\captionsetup{labelformat=empty}
\caption{Fig. 9.- Similar to the plot in Figure 8, except that the quantity plotted is the departure of the measured pulse-broadening time \(\left(\tau_{d}\right)\) from the prediction of the NE2001 model ( \(\tau_{d, n e 2001}\) ); the size of the symbol is proportional to the absolute value of \(\log \left(\tau_{d} / \tau_{d, n e 2001}\right)\). As for Figure 8, the filled circles represent the published data, while the open circles are the new measurements.}
\end{figure}

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-32.jpg?height=989&width=1331&top_left_y=339&top_left_x=412}
\captionsetup{labelformat=empty}
\caption{Fig. 10.- Estimates of \(F_{c}(\delta \mathrm{DM})^{2}\) for the clumps of enhanced scattering, derived from the excess scattering measures (Table 4), are plotted against the respective pulsar dispersion measures. The results are for a clump size of \(\sim 10 \mathrm{pc}\) and a volume number density \(\sim 1 \mathrm{kpc}^{-3}\) for the clumps. For a fluctuation parameter of \(F_{c}=10\), these results imply excess DM within the range \(7 \times 10^{-4}-4 \times 10^{-2} \mathrm{pc} \mathrm{cm} \mathrm{cm}^{-3}\).}
\end{figure}

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_11_26_36f97742d760670a7da6g-32.jpg?height=630&width=1286&top_left_y=1723&top_left_x=429}
\captionsetup{labelformat=empty}
\caption{Fig. 11.- Measurements of frequency scaling index \(\left(\alpha_{1}\right)\) against the respective DMs. The results for low-DM objects (Cordes, Weisberg, \& Boriakoff 1985) are derived from measurements of decorrelation bandwidths. For PSR J1852 +0031 , the only object common between our sample and that of Löhmer et al. (2001), estimates of \(\alpha\) are consistent within measurement errors. The dashed line corresponds to the Kolmogorov scaling index.}
\end{figure}