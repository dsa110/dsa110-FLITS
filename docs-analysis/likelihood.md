# The multi-component gain-marginal joint likelihood

This page documents the core likelihood used for the CHIME+DSA joint scattering model-selection grid: `_gain_marginal_multi_band` and its picklable wrapper `_JointLogLikelihoodGainMulti`. Both live in `scattering/scat_analysis/burstfit_joint.py`. They evaluate the marginal Bayesian evidence of an $N$-component-per-band scattering model after analytically integrating out a per-channel, per-component gain under a *proper* finite-variance Gaussian prior. This is what makes the per-band $\ln Z$ values comparable across model sizes (C1D1, C1D2, C2D1, C2D2) and lets a second temporal component be accepted or rejected on evidence rather than on $\chi^2$ alone.

## What the likelihood computes

For each band (CHIME or DSA), the model places $N$ temporal scattering kernels per frequency channel. The data in channel $f$ over time samples $t$ is $d_t$, with per-channel noise standard deviation $\sigma_f$ (variance $\sigma_f^2$). The $i$-th component kernel is $K_{i,t}$, built by evaluating the canonical `FRBModel` at unit amplitude (`c0=1.0`, `gamma=0.0`) so the kernel carries only *shape*, not amplitude (burstfit_joint.py:206-211).

Each component is multiplied by a free per-channel gain $g_i$. The forward model in a channel is $\hat d_t = \sum_i g_i K_{i,t}$. The key move: the gain vector $g = (g_1,\dots,g_N)$ is a nuisance parameter that absorbs the *burst spectrum* (how bright the source is at this frequency) **and** the *scintillation* (the multiplicative frequency structure imprinted by the ISM). It is integrated out analytically rather than sampled, because it is conditionally linear-Gaussian given the shape parameters.

### The per-channel matched-filter normal equations

The sufficient statistics are the standard matched-filter / least-squares normal equations, formed per channel (burstfit_joint.py:217-219):

$$
M_{ij} = \sum_t K_{i,t}\,K_{j,t}\quad(N\times N),\qquad
b_i = \sum_t d_t\,K_{i,t}\quad(N),\qquad
S_{dd} = \sum_t d_t^2 .
$$

$M$ is the kernel Gram matrix (symmetric positive-semidefinite), $b$ is the matched filter of the data against each kernel, and $S_{dd}$ is the total data power. In code these are batched over all channels at once with `np.einsum` (shapes $M$: `(F, N, N)`, $b$: `(F, N)`, $S_{dd}$: `(F,)`).

### Why the gain is integrated out

A flat per-channel amplitude is exactly the right object to marginalize: it is the one degree of freedom per channel that (a) is degenerate with the unknown source spectrum and scintillation, and (b) enters the model linearly, so the marginal integral is a Gaussian and has a closed form. Sampling it instead would add $N \times F$ nuisance dimensions (hundreds), wreck sampler efficiency, and leave the 2D time–frequency residual un-whitened. Marginalizing it analytically collapses all of that into the matrix algebra above, keeps the sampled vector low-dimensional, and whitens the residual so $\chi^2$ becomes a valid goodness-of-fit gate (see the design note at burstfit_joint.py:88-90).

## The proper finite-variance gain prior

The gains carry a zero-mean Gaussian prior with finite, isotropic variance:

$$
g \sim \mathcal{N}\!\left(0,\; s^2\, I_N\right),
$$

where $s^2$ is the gain-prior variance hyperparameter (docstring at burstfit_joint.py:159-185). The finiteness is essential. An improper flat prior ($s^2 \to \infty$) gives an Occam term $-\tfrac{1}{2}\ln\det M$ that behaves like $+N\ln s^2$ and **rewards** spurious merged components; the proper prior fixes exactly this pathology (burstfit_joint.py:176-181).

### The closed-form per-channel evidence

With a Gaussian likelihood in $d$ and a Gaussian prior in $g$, the per-channel marginal evidence integrates in closed form (docstring at burstfit_joint.py:169-171):

$$
\boxed{\;
\ln Z_f = -\tfrac{1}{2}\!\left[\frac{S_{dd}}{\sigma^2}
- \frac{b^{\mathsf T}\!\left(M + \tfrac{\sigma^2}{s^2} I\right)^{-1} b}{\sigma^2}\right]
- \tfrac{1}{2}\,T\ln(2\pi\sigma^2)
- \tfrac{1}{2}\ln\det\!\left(I_N + \tfrac{s^2}{\sigma^2} M\right)
\;}
$$

with all of $\sigma^2, S_{dd}, b, M$ taken in channel $f$, and $T$ the number of time samples (full-data normalization). The total band evidence is $\ln Z = \sum_f \ln Z_f$.

Three pieces, matching the implementation in `_lnZ_at` (burstfit_joint.py:246-290):

- **The data term** $-\tfrac{1}{2} S_{dd}/\sigma^2$: the no-signal data power.
- **The matched-filter gain (Wiener) term** $+\tfrac{1}{2}\,b^{\mathsf T}(M + \tfrac{\sigma^2}{s^2}I)^{-1} b/\sigma^2$. The ridge $\tfrac{\sigma^2}{s^2}I$ added to $M$ is the regularizer from the finite prior; the MAP gain is $g = (M + \tfrac{\sigma^2}{s^2}I)^{-1} b$, solved directly (burstfit_joint.py:255-258), and `quad` $= b^{\mathsf T} A^{-1} b$ is the realized matched-filter gain (burstfit_joint.py:258).
- **The full-data Gaussian normalization** $-\tfrac{1}{2} T \ln(2\pi\sigma^2)$.

A documented correction lives in the docstring: the quadratic divisor is $\sigma^2$, **not** $\sigma^4$. This was verified against the brute-force Gaussian evidence $d^{\mathsf T}\Sigma_d^{-1} d$ with $\Sigma_d = \sigma^2 I_T + s^2 K K^{\mathsf T}$ via the Woodbury identity; the spec's $\sigma^4$ was a transcription slip (burstfit_joint.py:173-176).

## The Occam term and why it grows with $N$

The third piece,

$$
-\tfrac{1}{2}\ln\det\!\left(I_N + \tfrac{s^2}{\sigma^2} M\right)
= -\tfrac{1}{2}\sum_{k=1}^{N}\ln\!\left(1 + \tfrac{s^2}{\sigma^2}\lambda_k\right),
$$

is the Occam factor, computed from the eigenvalues $\lambda_k$ of $M$ as `log1p((s2/var) * eigvals)` summed over components (burstfit_joint.py:261-264). Each additional component contributes another positive $\ln(1 + \tfrac{s^2}{\sigma^2}\lambda_k)$ term, so the penalty **grows with $N$** and **with $s^2$**. This is the correct, finite penalty that the improper flat version got wrong: the flat case had $-\tfrac{1}{2}\ln\det M$, which diverges as $+N\ln s^2$ as $s^2\to\infty$ and thereby rewards adding spurious (especially merged) components instead of penalizing them (burstfit_joint.py:176-181). With the proper prior, an extra component must buy back its Occam cost in fit improvement before it raises $\ln Z$ — which is precisely why, in the verified grid, the data add a second CHIME component (+3612 nats) but the spurious second DSA component is worth only +95.

As a consistency check, the docstring records that in the $s^2 \to \infty$ limit $\ln Z_f$ reduces to the flat F-statistic profile $-\tfrac{1}{2}\chi^2_{\min,f} - \tfrac{1}{2}\ln\det M_f + \tfrac{1}{2}N\ln s^2$, where the last term is a divergent parameter-independent constant that cancels in any $\ln Z$ *difference* (burstfit_joint.py:181-182).

## The eigenvalue conditioning guard

When two components nearly merge — common in the DSA "damage" band where kernels become collinear — $M_f$ becomes singular and the full-$N$ solve explodes $|g|$. A per-channel eigenvalue guard intercepts this (burstfit_joint.py:221-243). $M$ is eigendecomposed with `eigh` (keeping eigenvectors, not just eigenvalues), and a channel is classified by its conditioning ratio:

$$
\frac{\lambda_{\min}(M_f)}{\lambda_{\max}(M_f)} \;\ge\; \texttt{eig\_rel\_floor}\;(=10^{-6}) .
$$

Three routes follow:

- **Well-conditioned (`ok`)** — ratio above the floor: the full closed-form $\ln Z_f$ above (burstfit_joint.py:251-271).
- **Culled-but-supported (`cull`)** — there is real signal ($\lambda_{\max} > 10^{-30}$) but the kernels are collinear: fall back to a **rank-1 proper-prior evidence** on the top eigenpair. The effective single kernel has squared norm $\lambda_{\max}$, the projected data is $b\cdot v_{\text{top}}$, and the scalar MAP gain is $g = b_{\text{proj}} / (\lambda_{\max} + \sigma^2/s^2)$, with its own rank-1 Occam term $\ln(1 + \tfrac{s^2}{\sigma^2}\lambda_{\max})$ (burstfit_joint.py:272-289).
- **Unsupported** — $\lambda_{\max} \approx 0$, no signal: the gain$=0$ baseline $-\tfrac{1}{2}S_{dd}/\sigma^2 - \tfrac{1}{2}T\ln(2\pi\sigma^2)$ (burstfit_joint.py:248-249).

### Why the rank-1 fallback (merge protection)

The critical design point: a culled-but-supported channel **must not** fall back to the gain$=0$ baseline. At large fixed $s^2$, the gain$=0$ baseline sits *above* the proper $N=1$ evidence (which carries the divergent $+\tfrac{1}{2}\ln(s^2/\sigma^2)$ Occam per channel), so culling-to-baseline would *reward* a degenerate merge by roughly $+\tfrac{1}{2}F\ln(s^2/\sigma^2)$ — e.g. +676 nats at $s^2 = 10^8$ — reintroducing the very bug the proper prior fixes (burstfit_joint.py:236-242). The rank-1 fallback is continuous with the proper $N=1$ model, so a near-merge is an Occam *penalty*, not a reward. In the verified grid this guard fires zero times where it matters: `frac_culled=0` and `ill-cond ch=0/16` for both the C1D1 control and the favored C2D1 model — the +3612 nat preference is not a conditioning artifact.

### Diagnostics and the two denominators

`_lnZ_at` returns `(lnZ, g_all)`; the wrapper assembles a diagnostics dict (burstfit_joint.py:312-323): `frac_culled` $= \text{mean}(\sim\!\texttt{ok})$, `max_abs_g` per component, the used `s2`, and `n_supported`. Note the deliberate denominator mismatch (burstfit_joint.py:314-317): `n_supported` counts only well-conditioned full-rank-$N$ channels (`ok`), while `frac_culled` also counts rank-1-fallback channels as culled, so in general `n_supported` $\ne (1 - \texttt{frac\_culled})\,F$.

## The $s^2$ hyperparameter

`s2` is the gain-prior variance. If a float is passed it is fixed; if `s2 is None` it is profiled by a 1-D maximum-likelihood search over $\ln s^2$, shared per band (burstfit_joint.py:292-310). The search range is anchored on the data scale: $\hat a = b/\mathrm{diag}(M)$ is the matched-filter gain, and `scale` $= \mathrm{var}(\hat a)$ over well-conditioned channels sets the center of an $\pm 18$-wide bracket in $\ln s^2$, minimized with `scipy.optimize.minimize_scalar` (bounded, `xatol=1e-3`). Because the profiled $s^2$ is shared per band per call, the Occam penalty is calibrated to the actual gain dispersion rather than an arbitrary constant.

## How the two bands share parameters

The wrapper `_JointLogLikelihoodGainMulti` (burstfit_joint.py:632-685) combines the two bands. It is picklable (holds two `FRBModel`s, the component counts `n_C`/`n_D`, and the `s2` policy) so `dynesty.pool` can ship it to fork-workers. The sampled vector is `JOINT_PARAM_NAMES_GAIN_MULTI(n_C, n_D)` (burstfit_joint.py:140-149):

$$
[\,\tau_{1\,\mathrm{GHz}},\ \alpha,\ \underbrace{t0_{C1}, \zeta_{C1}, \dots, t0_{C n_C}, \zeta_{C n_C}, \delta\!\mathrm{DM}_C}_{\text{CHIME}},\ \underbrace{t0_{D1}, \zeta_{D1}, \dots, t0_{D n_D}, \zeta_{D n_D}, \delta\!\mathrm{DM}_D}_{\text{DSA}}\,]
$$

- **Shared across both bands and all components:** $\tau_{1\,\mathrm{GHz}}$ (scattering normalization at 1 GHz) and $\alpha$ (scattering index). One sightline, one scattering law $\tau(\nu) = \tau_{1\,\mathrm{GHz}}\,\nu^{-\alpha}$. The ~1 GHz lever arm between CHIME (~0.6 GHz) and DSA (~1.4 GHz) is what breaks the $\tau$–$\alpha$ degeneracy a single band cannot resolve (module docstring, burstfit_joint.py:7-16).
- **Per-band:** $\delta\!\mathrm{DM}_C$ and $\delta\!\mathrm{DM}_D$ — one cold-plasma dispersion column per telescope.
- **Per-component, per-band:** each $(t0_i, \zeta_i)$ pair — the arrival time and intrinsic width of each pulse.

In `__call__` (burstfit_joint.py:673-685), `theta[0:2]` are the shared $(\tau, \alpha)$; `_band_params` (burstfit_joint.py:661-671) unpacks each band's `delta_dm` (positioned right after that band's $(t0,\zeta)$ block) and builds one `FRBParams` per component with `c0=1.0`, `gamma=0.0` (amplitude marginalized, not fit). Each band is evaluated by `_gain_marginal_multi_band` with `["M3"] * n_comp`; independent noise makes the joint log-likelihood additive:

$$
\ln Z_{\text{joint}} = \ln Z_C + \ln Z_D ,
$$

returned as $-10^{100}$ if non-finite (burstfit_joint.py:682-685).

The ordered prior transform `_JointPriorTransformOrdered` (burstfit_joint.py:466-509) sorts each band's $t0$ group ascending (collapsing the $N!$ label-swap modes to one) and enforces $t0_{i+1} - t0_i \ge \texttt{dt\_min}$ by re-mapping the unit cube onto the feasible simplex, so every cube point is valid (no rejected volume). When a band is too narrow for $N$ separated components, the group collapses to a point and the resulting degenerate kernels are culled by the eigenvalue guard — so a forced merge is Occam-penalized, not rewarded, closing the loop with the conditioning guard above.

## Source

This page documents:
- `scattering/scat_analysis/burstfit_joint.py` — `_gain_marginal_multi_band` (burstfit_joint.py:152-324), `_JointLogLikelihoodGainMulti` (burstfit_joint.py:632-685), `JOINT_PARAM_NAMES_GAIN_MULTI` (burstfit_joint.py:140-149), `_joint_prior_spec_gain_multi` (burstfit_joint.py:415-441), and `_JointPriorTransformOrdered` (burstfit_joint.py:466-509).
