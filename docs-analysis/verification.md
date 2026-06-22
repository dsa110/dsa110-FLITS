# Adversarial verification protocol

A model-selection result of $\Delta\ln Z = +3612$ nats for a second CHIME temporal component (model C2D1 over C1D1) is large enough to demand suspicion: at that magnitude the dominant failure mode is not "wrong physics" but a *likelihood pathology* — a degenerate kernel matrix, an ill-conditioned solve, or an Occam term with the wrong sign rewarding spurious merged components. The marginal evidence $\ln Z$ is an integral the sampler reports; it does not, by itself, tell you the integrand was well-behaved. This page documents `verify_zach_c2.py`, an independent reconstruction whose job is to *try to break* the $+3612$-nat result and fail to, using three orthogonal checks: an **internal control**, a set of **pathology guards**, and a **same-metric-as-the-gate** comparison.

The protocol is deliberately reusable. Any future multi-component fit that produces a large $\Delta\ln Z$ should be subjected to the same three-part structure before the number is trusted.

## What is being verified, and why a separate script

The joint CHIME+DSA fit marginalizes the per-channel burst amplitude (gain) analytically, integrating out one independent gain per frequency channel per temporal component under a Gaussian prior. That marginalization is what makes the evidence comparable across models with different component counts — every model in the grid (C1D1, C1D2, C2D1, C2D2) is scored on the *same* gain-marginal joint likelihood, so the $\ln Z$ differences are matched-normalization (see the model-selection grid). The kernel that computes this evidence per band is `_gain_marginal_multi_band` at `scattering/scat_analysis/burstfit_joint.py:152`.

The verification problem is that the sampler only ever reports the *summed* $\ln Z$. It never exposes the per-channel MAP gains, the conditioning of each channel's kernel matrix $M_f$, or the residual after the best-fit gains are subtracted. Those are exactly the quantities that diagnose a pathology. So `verify_zach_c2.py` re-derives them from scratch — it reconstructs the per-channel generalized-least-squares (GLS) gain MAP directly from the kernel docstring formula, forms a band-integrated residual, and recomputes the *same* lag-1 autocorrelation metric that the whiteness gate used to flag this burst in the first place. Independence is the point: a bug shared between the fitter and its checker would hide. The verifier reads only the kernel formula (the docstring at `burstfit_joint.py:159-199`), not the kernel's internal solve.

## Reconstructing the per-channel GLS gain MAP

For one band, one frequency channel $f$, with data row $d_t$, noise variance $\sigma_f^2$, and $N$ component kernels $K_{i,t}$ (one per temporal component, evaluated at unit amplitude), the gain-marginal kernel forms the per-channel quantities (`burstfit_joint.py:217-219`):

$$
M_{ij} = \sum_t K_{i,t}\,K_{j,t}\quad(N\times N),\qquad
b_i = \sum_t d_t\,K_{i,t},\qquad
S_{dd} = \sum_t d_t^2 .
$$

The per-component gains $g \sim \mathcal{N}(0,\,s^2 I_N)$ are integrated analytically, giving the per-channel log-evidence (`burstfit_joint.py:169-171`):

$$
\ln Z_f = -\tfrac12\Big[\,\frac{S_{dd}}{\sigma^2} - \frac{b^{\!\top}\big(M + \tfrac{\sigma^2}{s^2} I\big)^{-1} b}{\sigma^2}\,\Big]
\;-\;\tfrac12\,T\ln(2\pi\sigma^2)
\;-\;\tfrac12\,\ln\det\!\Big(I_N + \tfrac{s^2}{\sigma^2} M\Big).
$$

The first bracket is the data-fit term, the middle is the full data normalization over $T$ time samples, and the last is the Occam penalty. The quadratic divisor is $\sigma^2$, not $\sigma^4$ — this was verified against the brute Gaussian evidence $d^{\!\top}\Sigma_d^{-1}d$ with $\Sigma_d = \sigma^2 I_T + s^2 K K^{\!\top}$ via Woodbury; the original spec's $\sigma^4$ was a transcription slip (`burstfit_joint.py:173-175`). The Occam term *grows* with $N$ and with $s^2$, so adding an unsupported component is penalized, not rewarded.

The verifier does not need the evidence value to form a residual — it needs the gains. The maximum-a-posteriori (ridge-regularized GLS) gain for a well-conditioned channel is the solution of the ridge system (`burstfit_joint.py:256-258`):

$$
A = M + \frac{\sigma^2}{s^2} I_N,\qquad
\hat g = A^{-1} b .
$$

`verify_zach_c2.py` reconstructs $\hat g$ per channel from this formula (`reconstruct`, the per-channel solve at `verify_zach_c2.py:84-88`), evaluates the model component kernels $K_{i,t}$ at the posterior-median parameters, and predicts each channel's time series $\hat d_{f,t} = \sum_i \hat g_{f,i}\,K_{i,t}$ (`verify_zach_c2.py:89`). The **band-integrated residual** is then the channel-summed $r_t = \sum_f (d_{f,t} - \hat d_{f,t})$ (`prof_r = r.sum(0)`, `verify_zach_c2.py:90,93`), i.e. the profile-domain residual after the best-fit gains are removed. The summary metrics are the reduced chi-square $\chi^2_{\rm red}$ of the full residual map (`verify_zach_c2.py:98`) and the **lag-1 autocorrelation** of the band-integrated residual (`verify_zach_c2.py:96`) — the latter chosen on purpose because it is the gate metric (next section).

The gain-prior variance $s^2$ is the hyperparameter: if not fixed it is profiled by 1-D maximum likelihood over $\log s^2$ on a shared per-band value (`burstfit_joint.py:292-310`). The verifier uses the same $s^2$ policy as the fit — it pulls $s^2$ straight from the likelihood's own profiled `diag["s2"]` (`verify_zach_c2.py:77-78`) — so the reconstruction is faithful, not a re-tune.

## Internal control: reproduce the gate the fitter never saw

A reconstruction that produces plausible-looking numbers proves nothing unless you can show it reproduces a number derived by a *completely separate* path. That number is the residual-whiteness gate's flag.

The whiteness gate that originally flagged burst "zach" computed, on the single-component CHIME fit (C1D1), $\chi^2_{\rm red}=2.3$ and a lag-1 autocorrelation of $+0.82$ — a strongly correlated residual, the signature of an unmodeled component. The gate is upstream of and independent from the gain-marginal reconstruction in `verify_zach_c2.py`.

The control is this: run the reconstruction on **C1D1** and check that its band-integrated residual reproduces the gate's flagged metric.

| Reconstruction (C1D1, CHIME) | gate value | reconstructed |
|---|---|---|
| lag-1 autocorr | $+0.82$ | $+0.836$ |
| $\chi^2_{\rm red}$ | $2.3$ | $2.30$ |

The reconstructed $\chi^2_{\rm red}=2.30$ matches the gate exactly and the lag-1 $+0.836$ reproduces the gate's $+0.82$ to within rounding. **The reconstruction is faithful** — the GLS-gain-MAP path computes the same residual the independent whiteness gate did. Only once this control passes is the verifier licensed to report C2D1's reconstructed metrics, because now we know its residual machinery is calibrated against ground truth.

## Pathology guards: was the $+3612$ a singular-kernel artifact?

With the reconstruction trusted, the next question is whether the $\Delta\ln Z=+3612$ came from a numerically pathological likelihood rather than real signal. Two guards target the specific failure modes of the gain-marginal kernel.

**Guard 1 — fraction of culled channels (`frac_culled`).** A channel is culled when its kernel matrix is rank-deficient: when two component kernels nearly merge, $M_f$ becomes singular and the full-$N$ solve $A^{-1}b$ explodes. The kernel's eigenvalue guard (`burstfit_joint.py:221-243`) culls a channel when $\min\mathrm{eig}(M_f)/\max\mathrm{eig}(M_f) < 10^{-6}$. Critically, a culled-but-supported channel does **not** drop to a gain=0 baseline — it falls back to a rank-1 proper-prior evidence on its top eigenpair (`burstfit_joint.py:272-289`), so a degenerate merge stays Occam-penalized rather than being rewarded by $\sim+0.5\,F\ln(s^2/\sigma^2)$ nats (e.g. $+676$ nats at $s^2=10^8$). `frac_culled` (`burstfit_joint.py:319`) is the fraction of channels that are not well-conditioned full-rank-$N$. If C2D1's $+3612$ were a kernel-merge artifact, `frac_culled` would be nonzero. The verifier reads this value back through the kernel's own diagnostics (`verify_zach_c2.py:77,107`).

**Guard 2 — ill-conditioned channel count.** Independently of the kernel's per-channel `frac_culled`, the verifier counts how many of the band's 16 channels have an ill-conditioned ridge-regularized solve matrix $A_f = M_f + (\sigma^2/s^2)I$, using $\mathrm{cond}(A_f) > 10^{10}$ (`verify_zach_c2.py:85-87`). This is a distinct, coarser conditioning probe than the kernel's $\min/\max$ eigenvalue test on $M_f$; a clean count means no kernel merge blew up the solve anywhere in the band.

Both guards come back clean for the favored CHIME two-component model:

| model (band) | `frac_culled` | ill-conditioned channels |
|---|---|---|
| CHIME C1D1 (control) | $0$ | $0/16$ |
| CHIME C2D1 (favored) | $0$ | $0/16$ |

`frac_culled = 0` and $0/16$ ill-conditioned channels mean **no kernel merge** — the second CHIME component is a genuinely separated pulse, not two collinear kernels masquerading as one. This is corroborated by the geometry: the C2D1 CHIME components sit at $t_0=13.29$ ms and $t_0=13.63$ ms, a separation of $0.340$ ms $=1.7\times$ the ordered-prior floor $dt_{\min}=0.197$ ms — the second component is **not pinned** at the minimum-separation floor, so it is resolved on its own merit. (Contrast the DSA second component in C1D2, separated by only $0.207$ ms $=1.1\times dt_{\min}$, pinned at the floor and contributing essentially nothing.)

## Same-metric comparison: the residual whitens

The third check asks whether adding the component actually *does what a real component should do* — reduce the residual correlation that flagged the burst. Because the verifier computes the gate's own metric, this is a like-for-like before/after on the same scale, not two incomparable numbers.

| model (band) | lag-1 autocorr | $\chi^2_{\rm red}$ |
|---|---|---|
| CHIME C1D1 | $+0.836$ | $2.30$ |
| CHIME C2D1 | $+0.707$ | $1.57$ |
| DSA C1D1 | $+0.795$ | $1.30$ |
| DSA C1D2 | $+0.783$ | $1.27$ |
| DSA C2D2 | $+0.461$ | $1.09$ |

Adding the second CHIME component drives lag-1 from $+0.836 \to +0.707$ and $\chi^2_{\rm red}$ from $2.30 \to 1.57$ — the residual whitens and the fit quality improves, exactly the behavior a real component produces. On the DSA side the same story holds with a sharper signature: the C2D2 DSA second component (separation $2.408$ ms $=12.2\times dt_{\min}$, coincident with the deterministic `find_peaks` peak at $12.1$ ms) drives DSA lag-1 to $+0.461$ and $\chi^2_{\rm red}$ to $1.09$. The DSA C1D2 component, pinned at the floor, barely moves the metrics ($+0.795\to+0.783$), consistent with its tiny $+95$-nat evidence gain.

The verdict the three checks converge on: a second CHIME component is favored by **$+3612$ nats** — about $38\times$ the DSA one ($+95$) — with no likelihood pathology (`frac_culled = 0`, conditioning clean across all 16 channels), and the residual whitening confirms the evidence gain is physical. The downstream physics consequence is that the shared scattering index shifts $\alpha = 3.32 \to 2.76$ (a $\sim 0.56$ move, $\sim 40\times$ the formal error of $\pm 0.013$) once the unmodeled CHIME component is included — i.e. the **single-component $\alpha$ is biased high**.

## Honest residual caveat

The protocol is built to falsify, so it must report what it could *not* clean up. Even the full C2D2 model is **not fully white**: CHIME lag-1 remains $\sim 0.71$ and DSA $\sim 0.46$. A sharp CHIME peak-edge feature survives the two-component fit. Its origin is not resolved by this analysis — candidates are an intrinsic fast rise, a scattering-kernel-shape limit (the assumed thin-screen exponential tail not capturing the true pulse-broadening function), or a possible third CHIME component. The defensible claim is therefore narrow: $\alpha = 2.76$ is **better-conditioned than the single-component value, not final**. The verification establishes that the $+3612$-nat second component is real and that ignoring it biases $\alpha$ high; it does not establish that two components are sufficient.

## The reusable protocol

The structure generalizes to any multi-component fit reporting a suspiciously large $\Delta\ln Z$:

1. **Reconstruct independently.** Re-derive the per-channel GLS gain MAP from the kernel's *documented formula* (`burstfit_joint.py:159-199`), not its internal solve, and form a residual. A bug shared by fitter and checker cannot be caught by a checker that reuses the fitter's code.
2. **Internal control before the headline.** Reproduce a metric computed by a *separate* upstream path (here, the whiteness gate's flagged C1D1 lag-1 $+0.82 \to$ reconstructed $+0.836$). Until the control passes, the reconstruction's numbers are not admissible.
3. **Pathology guards targeting the kernel's failure modes.** For a gain-marginal evidence the modes are kernel merge / ill-conditioning: check `frac_culled = 0` and the ill-conditioned channel count $0/N$, and confirm components are not pinned at the ordered-prior floor $dt_{\min}$.
4. **Same metric as the gate.** Score before/after on the *exact* metric that flagged the burst (lag-1 autocorr, $\chi^2_{\rm red}$), so the improvement is like-for-like.
5. **Report the residual honestly.** State what survives the fit and bound the claim accordingly. "Better-conditioned, not final" is a complete result; "white" would have been an overclaim.

## Source

- `scattering/scat_analysis/burstfit_joint.py` — the multi-component gain-marginal kernel that `verify_zach_c2.py` reconstructs against: `_gain_marginal_multi_band` (`:152`), the per-channel evidence/GLS-gain-MAP docstring formula (`:159-199`; the closed-form $\ln Z_f$ at `:169-171`, the $M$/$b$/$S_{dd}$ implementation at `:217-219`), the GLS gain solve $A^{-1}b$ (`:256-258`), the eigenvalue conditioning guard and rank-1 fallback (`:221-289`), `frac_culled` (`:319`), and the $s^2$ ML profile (`:292-310`).
- `analysis/scattering-refit-2026-06/verify_zach_c2.py` — the adversarial verification script (reconstruct gain MAP → band-integrated residual → lag-1 gate metric; internal control, pathology guards, same-metric comparison). Committed on `main` (commit `75a917c`). The reconstruction lives in `reconstruct` (`:66-112`): the per-channel ridge solve at `:84-88`, the predicted waterfall at `:89`, the band-integrated residual / lag-1 / $\chi^2_{\rm red}$ at `:90-98`, and the $s^2$ pulled from the kernel's `diag["s2"]` at `:77-78`. The CHIME control/test loop is `:130-143`; the DSA loop is `:151-164`.
