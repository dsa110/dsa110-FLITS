# Case study: zach and the alpha bias

This is the narrative spine of the multi-component CHIME+DSA joint analysis. It follows a single co-detected FRB, nicknamed **zach**, from the moment a residual-whiteness gate flagged it through to a headline result: the scattering index $\alpha$ for zach shifts from $3.32$ to $2.76$ once a hidden second CHIME component is modeled. zach is the cleanest worked example of a bias that, left unmodeled, inflates $\alpha$ high across the single-component sample.

Everything below is grounded in the build-state and plan documents and the deterministic peak counter for this refit campaign (see **Source** at the bottom). The numbers are verified results from the matched-normalization joint fits; do not rescale or reinterpret them.

## The setup: why zach was a fit you could not trust

zach is a CHIME (0.4–0.8 GHz) + DSA-110 (1.311–1.499 GHz) co-detection. Both bands were originally fit single-component with the M3 scattering model, and both came back **FAIL/marginal** — the build-state notes zach as "both bands FAIL/marginal (CHIME under-dedispersed)" with single-band $\tau_{1\text{GHz}}$ rails of CHIME $0.262$ ms vs DSA $0.44$ ms (JOINT_FIT_STATE.md:14).

The joint fit exists precisely because of this kind of cross-band tension. Single-band fits fix $\alpha=4$ and then disagree on $\tau_{1\text{GHz}}$ for the *same sightline*; the joint fit instead **shares** $\tau_{1\text{GHz}}$ and $\alpha$ across telescopes and lets the $\sim 1$ GHz lever arm between the bands measure $\alpha$ directly (JOINT_FIT_STATE.md:3–6). DSA $\tau_{1\text{GHz}}$ systematically exceeding CHIME is the signature that the true $\alpha$ is shallower than the canonical Kolmogorov value of 4.

But a joint fit only measures $\alpha$ honestly if the *temporal* model in each band is correct. If a band hides an unmodeled second pulse, the fitter absorbs that extra structure into the scattering tail and the intrinsic width — and $\alpha$ pays the price. That is the entire story of zach.

### The gate that flagged it

The mandatory residual-whiteness check (part of the fit-validation contract) flagged zach on the CHIME side:

$$\chi^2_{\text{red}} = 2.3, \qquad \rho_{\text{lag-1}} = +0.82$$

A lag-1 autocorrelation of the band-integrated residual near $+0.8$ means the residual is **not white** — successive time samples are strongly correlated, the classic fingerprint of a coherent structure the model failed to capture. zach entered the multi-component program with exactly this signature: "CHIME lag1=+0.82; $\alpha\sim3.3$; DSA lag1=0.80 borderline — add DSA secondary if persists" (MULTICOMPONENT_PLAN.md:104).

## The apparent paradox: the gate says CHIME, find_peaks says DSA

To remove the subjective element from "how many sub-bursts are there," the campaign uses a deterministic peak counter: `scipy.signal.find_peaks` on the band-integrated on-pulse profile, with a prominence threshold set in units of robust MAD noise (`peak_count.py:39-55`). The prominence threshold is $4\sigma$ and peaks must be separated by at least 2 samples (`peak_count.py:24-26`):

```python
PROM_SIGMA = 4.0     # peak prominence threshold in noise sigma
MIN_SEP = 2          # min samples between peaks
SMOOTH = 1.0         # light Gaussian smoothing (samples)
```

The noise scale is the robust MAD estimate, and prominence is reported per peak in $\sigma$ so marginal second components are explicit rather than hidden behind a yes/no (`peak_count.py:8-9`, `peak_count.py:44`, `peak_count.py:49-50`):

$$\sigma_{\text{noise}} = 1.4826 \cdot \operatorname{median}\bigl(|p - \operatorname{median}(p)|\bigr)$$

For zach, the deterministic count returns:

| Band  | $N_{\text{peaks}}$ | Peak locations (prominence)                         |
|-------|--------------------|-----------------------------------------------------|
| CHIME | 1                  | 13.517 ms (62.7$\sigma$)                            |
| DSA   | 2                  | 9.634 ms (47.5$\sigma$), 12.124 ms (15.4$\sigma$)  |

This is the paradox. The whiteness gate is screaming about **CHIME**, where `find_peaks` sees only **one** clean, very bright peak. Meanwhile `find_peaks` resolves **two** peaks in **DSA**, the band the gate was relatively quiet about. If you trusted the peak counter alone, you would add a component to the wrong band.

The resolution: `find_peaks` can only see a second component if it is *separated enough to produce a distinct local maximum* in the integrated profile. A **blended companion** — a second pulse buried under the wings of the first, within roughly a scattering-kernel width — produces no separate peak. It produces exactly what the gate measures: a correlated, non-white residual. The gate is sensitive to structure the peak counter is blind to. zach's CHIME residual is the textbook case of a hidden, blended second component.

## The matched-normalization grid: the evidence

The decisive test is to refit zach with one and two temporal components in each band, **all on the same multi-component gain-marginal joint likelihood**, so the evidences $\ln Z$ are directly comparable. Notation: `C` = number of CHIME temporal components, `D` = number of DSA components.

| model | $\alpha$              | $\tau_{\text{ms}}$ | $\ln Z$   | $\Delta\ln Z$ vs C1D1 | what it adds                          |
|-------|-----------------------|--------------------|-----------|-----------------------|----------------------------------------|
| C1D1  | $3.319 \pm 0.013$     | 0.4162             | 32965.1   | 0                     | baseline (`force_multi` at $N=1$)      |
| C1D2  | $3.332 \pm 0.013$     | 0.4132             | 33060.3   | +95.2                 | 2nd component in DSA                    |
| C2D1  | $2.755 \pm 0.015$     | 0.5577             | 36577.1   | +3612.0               | 2nd component in CHIME                  |
| C2D2  | $2.762 \pm 0.015$     | 0.5551             | 37733.7   | +4768.7               | 2nd in both                            |

Read the $\Delta\ln Z$ column against the Jeffreys threshold (the campaign's adopted bar is $\Delta\ln Z > 5$ for "strong," MULTICOMPONENT_PLAN.md:125):

- **A second DSA component (C1D2) is favored by +95 nats.** Real, decisive — this is the resolved 12.1 ms peak `find_peaks` already saw.
- **A second CHIME component (C2D1) is favored by +3612 nats** — roughly **38×** the DSA improvement. This is the blended companion the gate flagged and `find_peaks` could not resolve. It is by far the dominant missing piece of zach's model.
- **C2D2 (both) is best at +4768.7**, and is the model the headline $\alpha$ is read from.

The CHIME improvement dwarfing the DSA one ($+3612$ vs $+95$) is the quantitative form of the paradox's resolution: the band that produced no second peak is the band that desperately needed a second component.

### Where the components sit

Posterior-median placements (ms) for the two-component CHIME model:

- **C2D1 CHIME:** $t_{0,C1} = 13.29$ ms (narrow, $\zeta = 0.093$), $t_{0,C2} = 13.63$ ms (broad, $\zeta = 0.413$) — **separation 0.34 ms**.
- **C2D1 DSA:** single component at $t_{0,D1} = 9.529$ ms ($\zeta = 0.042$).
- **C2D2** additionally resolves DSA $t_{0,D2} = 11.94$ ms ($\zeta = 0.31$), matching the 12.1 ms `find_peaks` detection.

The second CHIME component lands **0.34 ms** from the first. That is the companion: close enough that the two pulses merge into a single 62.7$\sigma$ local maximum in the integrated profile, far enough that the joint likelihood resolves it overwhelmingly.

## Is the new CHIME component real, or a merge artifact?

The multi-component likelihood has a known, adversarially-reproduced pathology: on pure noise it *rewards* a spurious second component by up to $+324$ nats as the two component times $t_{0,2}\to t_{0,1}$ merge, the Occam penalty flips the wrong sign, and the gain amplitude blows up to $\max|g|\sim 4673$ (MULTICOMPONENT_PLAN.md:1, MULTICOMPONENT_PLAN.md:30–36). A second component favored by thousands of nats is exactly what that singularity would also produce, so the +3612 must be defended against it.

Two guards establish that zach's CHIME component is physical, not a merge artifact:

**1. The minimum-separation prior keeps zach out of the singularity.** The ordered prior enforces $t_{0,i+1}-t_{0,i}\ge dt_{\min}$ (MULTICOMPONENT_PLAN.md:38–39, MULTICOMPONENT_PLAN.md:72–79). For zach the auto floor is

$$dt_{\min} = 0.197~\text{ms} = 3 \times \max(\text{median } dt),$$

set from the per-band median time resolutions (CHIME $dt = 0.0614$ ms, DSA $dt = 0.0655$ ms). The CHIME components sit at a separation of $0.34$ ms $= 1.7 \times dt_{\min}$ — **comfortably above the floor, not pinned to it.** The fitter chose that separation; it was not forced there. (Contrast the DSA secondary in C1D2: separation $0.207$ ms $= 1.1\times dt_{\min}$, sitting essentially *at* the floor, where the model is structurally indifferent — see the caveat below.)

**2. Adversarial reconstruction shows the residual genuinely whitens with no likelihood pathology.** A separate verifier (`verify_zach_c2.py`) reconstructs the per-channel GLS gain MAP, forms the band-integrated residual, and measures the same lag-1 metric the gate uses. Its internal control reproduces the gate: CHIME C1D1 gives $\rho_{\text{lag-1}}=+0.836$, $\chi^2_{\text{red}}=2.30$ — matching the gate's original $+0.82 / 2.3$ flag, confirming the verifier is measuring the same thing.

| reconstruction | $\rho_{\text{lag-1}}$ | $\chi^2_{\text{red}}$ | frac culled | ill-cond ch | note |
|---|---|---|---|---|---|
| CHIME C1D1 | $+0.836$ | 2.30 | 0 | 0/16 | **internal control** — reproduces the gate's $+0.82$ |
| CHIME C2D1 | $+0.707$ | 1.57 | 0 | 0/16 | $t_0$ sep $0.340$ ms $=1.7\times dt_{\min}$ (not pinned) |
| DSA C1D1 | $+0.795$ | 1.30 | — | — | |
| DSA C1D2 | $+0.783$ | 1.27 | — | — | 2nd-comp sep $0.207$ ms $=1.1\times dt_{\min}$ (pinned, does nothing) |
| DSA C2D2 | $+0.461$ | 1.09 | — | — | 2nd-comp sep $2.408$ ms $=12.2\times dt_{\min}$ (= the `find_peaks` 12.1 ms peak) |

Adding the CHIME component drops the CHIME residual correlation from $+0.836$ to $+0.707$ and $\chi^2_{\text{red}}$ from $2.30$ to $1.57$, with **`frac_culled = 0` and 0/16 ill-conditioned channels** — none of the eigenvalue guard's conditioning fallback fired. The merge singularity is characterized by exploding gains and ill-conditioning; zach exhibits neither. The improvement is a real reduction in residual structure, not a degeneracy being exploited.

## Cross-band coupling: the DSA secondary only resolves once CHIME pins $\tau$ and $\alpha$

The grid also exposes a coupling that is easy to miss. The DSA second component at 12.1 ms is a clean $15.4\sigma$ `find_peaks` detection — yet in the verifier table it only becomes a well-separated, residual-whitening component in **C2D2**, where its separation is $2.408$ ms $= 12.2\times dt_{\min}$ and the DSA residual drops to $\rho_{\text{lag-1}}=+0.461$, $\chi^2_{\text{red}}=1.09$.

In **C1D2** — DSA-secondary-on but CHIME still single-component — that same DSA component sits pinned at the floor ($0.207$ ms $=1.1\times dt_{\min}$) and "does nothing": the DSA residual barely moves ($+0.795 \to +0.783$). The reason is the shared parameters. $\tau_{1\text{GHz}}$ and $\alpha$ are common to both bands. While CHIME is mis-modeled by a single blended pulse, it drags the *shared* $\tau$ and $\alpha$ to absorb that error ($\alpha$ pinned near 3.32, $\tau\approx0.413$ ms). Those wrong shared values then mis-shape the DSA scattering kernel, leaving no room for the DSA model to place a genuine second component at its true 2.4 ms separation — the DSA secondary collapses to the prior floor instead.

Only once the CHIME side is correctly two-component (C2D1 $\to$ C2D2) do the shared $\tau$ and $\alpha$ relax to their honest values ($\alpha\approx2.76$, $\tau\approx0.555$ ms), and the DSA secondary then snaps out to its real 2.4 ms location and whitens the DSA residual. **The DSA companion is only resolvable once CHIME stops corrupting the shared parameters.** This is the joint fit's coupling working as designed — and a warning that per-band model errors propagate across telescopes through the shared scattering law.

## The headline: $\alpha$ bias and what it means for the sample

The shared scattering index moves with the model:

$$\alpha: \quad 3.32 \;\longrightarrow\; 2.76 \qquad (\Delta\alpha \approx 0.56)$$

That shift is about **40× the formal posterior error** ($\pm0.013$–$0.015$). It is not statistical scatter; it is a systematic bias that the single-component model carried silently. The direction is unambiguous and physically important: the unmodeled CHIME companion's flux, blended into the first pulse's tail, *steepens* the apparent frequency dependence of the scattering, so

> **single-component $\alpha$ is biased HIGH.**

The bias is favored by $+3612$ nats with **no likelihood pathology** — `frac_culled = 0`, conditioning clean, the second component sitting at $1.7\times dt_{\min}$ rather than pinned. zach is therefore the campaign's clean demonstration that the "second pulse drags $\alpha$" effect is real on actual data, not just a synthetic prediction.

The implication for the broader sample follows directly. The single-component exclusion set in this campaign is **non-random**: it preferentially contains the multi-peak, brighter, more-scattered bursts — exactly the tail the analysis cares about (MULTICOMPONENT_PLAN.md:128). zach proves that at least one of those flagged bursts had its $\alpha$ pushed high by a hidden component the deterministic peak counter could not see. Any population-level statement about $\alpha$ built on single-component fits — including claims of Kolmogorov ($\alpha\approx4$) or sub-Kolmogorov values — inherits a high bias of order a few tenths wherever a blended companion went unmodeled. The whiteness gate, not `find_peaks`, is the instrument that catches these.

## The honest caveat: better-conditioned, not final

zach is a clean result, but it is not a closed one. Even the best model, **C2D2**, does not produce fully white residuals:

$$\rho_{\text{lag-1}}^{\text{CHIME}} \approx 0.71, \qquad \rho_{\text{lag-1}}^{\text{DSA}} \approx 0.46.$$

A sharp **CHIME peak-edge feature survives** the two-component model. Its origin is genuinely ambiguous among three live possibilities:

1. an **intrinsic fast rise** of the burst that the smooth scattering kernel cannot reproduce;
2. a **scattering-kernel-shape limit** — the assumed thin/extended-screen kernel is the wrong functional form near the leading edge;
3. a **possible third component** even closer to the main pulse than the resolved companion.

Because that residual structure remains, the correct statement is that

> $\alpha = 2.76$ is **better-conditioned, not final.**

The campaign's own discipline guards against the obvious failure mode: tuning $N$ upward until every residual whitens would produce a spuriously over-fitted result. The bar for adopting another component stays at $\Delta\ln Z > 5$ **plus** a resolved component separated beyond its own width **plus** positive gain power — never relaxed to chase lag-1 alone (MULTICOMPONENT_PLAN.md:125, MULTICOMPONENT_PLAN.md:130). zach has cleared that bar for two CHIME components and resolved its DSA companion; whether a third feature is a physical pulse or an intrinsic/kernel-shape limit is the open question that keeps $\alpha = 2.76$ a provisional, not a final, number.

## The likelihood that makes this comparison valid

The entire grid is interpretable only because all four models use the **same gain-marginal joint likelihood**, so the evidences live on a common scale. Per band, per channel $f$, with on-pulse data $d_t$ and component templates $K_{i,t}$, the likelihood forms the $N\times N$ template Gram matrix and the projection vector (MULTICOMPONENT_PLAN.md:15):

$$M_{ij} = \sum_t K_{i,t}\,K_{j,t}, \qquad b_i = \sum_t d_t\,K_{i,t}, \qquad S_{dd} = \sum_t d_t^2,$$

and analytically integrates the per-component gains $g\sim\mathcal{N}(0, s^2 I_N)$ to give the per-channel log-evidence:

$$\ln Z_f = -\tfrac{1}{2}\!\left[\frac{S_{dd}}{\sigma^2} - \frac{b^{\mathsf T}\!\left(M + \tfrac{\sigma^2}{s^2}I\right)^{-1}\! b}{\sigma^2}\right] - \tfrac{1}{2}\,T\ln(2\pi\sigma^2) - \tfrac{1}{2}\ln\det\!\left(I_N + \tfrac{s^2}{\sigma^2}M\right).$$

Three features make the model-selection trustworthy here:

- **The Occam term grows with $N$.** The $-\tfrac{1}{2}\ln\det(I_N + \tfrac{s^2}{\sigma^2}M)$ factor penalizes extra components, so a $\Delta\ln Z$ of $+3612$ is paid *after* the complexity cost — it is not free flexibility.
- **A finite-variance gain prior makes $\ln Z$ a valid Bayes factor.** The gains are integrated against a proper $\mathcal{N}(0, s^2 I_N)$ prior ($s^2$ the gain-prior variance, profiled by 1-D ML when not fixed), which is precisely the fix required for cross-$N$ evidence comparisons to be legitimate (MULTICOMPONENT_PLAN.md:40, MULTICOMPONENT_PLAN.md:161).
- **An eigenvalue conditioning guard culls degenerate channels** (min/max eigenvalue of $M_f < 10^{-6}$) using a rank-1 top-eigenpair fallback — crucially **not** a gain-zeroing — which is why the verifier can report `frac_culled` and `ill-cond ch` per band and confirm none fired for zach.

The shared $(\tau_{1\text{GHz}}, \alpha)$ between the CHIME and DSA blocks (with per-telescope $t_0$, $\zeta$, $\delta_{\text{DM}}$) is what turns this from two independent fits into a single $\sim1$ GHz lever on $\alpha$ — and what makes zach's cross-band coupling, and its $\alpha$ bias, a property of one shared scattering law rather than two unrelated bands.

---

**Source.** This page documents:

- `analysis/scattering-refit-2026-06/JOINT_FIT_STATE.md` — joint-fit build state, per-burst $\tau_{1\text{GHz}}$ rails, zach's FAIL/marginal single-band status, shared-$(\tau,\alpha)$ rationale, and resolutions.
- `analysis/scattering-refit-2026-06/MULTICOMPONENT_PLAN.md` — multi-component gain-marginal likelihood, the merge-singularity adversarial verdict and minimum-separation prior, the $\Delta\ln Z>5$ component-selection gate, and zach's per-burst prediction (`MULTICOMPONENT_PLAN.md:104`).
- `analysis/scattering-refit-2026-06/peak_count.py` — the deterministic `find_peaks` sub-burst counter (`PROM_SIGMA = 4.0`, robust MAD noise) that resolves 2 DSA / 1 CHIME peaks for zach.

Numerical results (the matched grid, component placements, $dt_{\min}$, and the `verify_zach_c2.py` reconstruction table) are verified joint-fit outputs from this refit campaign.
