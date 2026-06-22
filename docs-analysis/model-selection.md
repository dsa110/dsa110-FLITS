# Matched-normalization model selection (the component grid)

## Why raw `lnZ` from different code paths is not comparable

`fit_joint_scattering` (`scattering/scat_analysis/burstfit_joint.py:712`) can build the joint likelihood through **four different code paths**, selected by its keyword flags:

| Path | Flag | Likelihood class | Param vector |
|---|---|---|---|
| Full M3, gains sampled | (default) | `_JointLogLikelihood` (`burstfit_joint.py:512`) | `JOINT_PARAM_NAMES` (12) |
| Gain-marginal, flat gain prior | `marginalize_gain=True` | `_JointLogLikelihoodGain` (`burstfit_joint.py:547`) | `JOINT_PARAM_NAMES_GAIN` (8) |
| Gain-marginal + scintillation GP | `marginalize_gain_gp=True` | `_JointLogLikelihoodGainGP` (`burstfit_joint.py:585`) | `JOINT_PARAM_NAMES_GAIN_GP` (10) |
| **Multi-component, proper finite gain prior** | `force_multi=True` or `components_C>1` or `components_D>1` | `_JointLogLikelihoodGainMulti` (`burstfit_joint.py:632`) | `JOINT_PARAM_NAMES_GAIN_MULTI` (`burstfit_joint.py:140`) |

These paths integrate out the per-channel gains under **different priors** and therefore carry **different additive normalization constants** in $\ln Z$. The flat-gain path uses an *improper* (infinite-variance) gain prior, whose normalization includes a divergent term: as $s^2 \to \infty$ the per-channel evidence picks up a divergent, parameter-independent $+\tfrac{1}{2}N\ln s^2$ constant (documented in the `_gain_marginal_multi_band` docstring, `burstfit_joint.py:180-181`). The multi-component path instead uses a **proper** $g \sim \mathcal{N}(0, s^2 I_N)$ prior with a finite Occam term (`burstfit_joint.py:165-198`):

$$
\ln Z_f = -\tfrac{1}{2}\Big[ \tfrac{S_{dd}}{\sigma^2} - \tfrac{1}{\sigma^2}\, b^\top \big(M + \tfrac{\sigma^2}{s^2} I\big)^{-1} b \Big]
- \tfrac{1}{2}\,T \ln(2\pi\sigma^2)
- \tfrac{1}{2}\ln\det\!\Big( I_N + \tfrac{s^2}{\sigma^2} M \Big),
$$

with the per-channel sufficient statistics

$$
M_{ij} = \sum_t K_{i,t} K_{j,t}, \qquad b_i = \sum_t d_t K_{i,t}, \qquad S_{dd} = \sum_t d_t^2 .
$$

Two consequences follow directly:

1. The **full-data normalization** $-\tfrac{1}{2}T\ln(2\pi\sigma^2)$ is retained in this path (`burstfit_joint.py:170`; code at `:248`, `:267`, `:286`), so it is a *real* evidence, not a profile statistic with a dropped constant.
2. The **Occam term grows with $N$ and with $s^2$** (`burstfit_joint.py:178`). This is the load-bearing difference: the improper flat version had $-\tfrac{1}{2}\ln\det M \sim +N\ln s^2$ as $s^2 \to \infty$, which *rewarded* spurious merged components (`burstfit_joint.py:178-181`). Comparing an $N=2$ flat-path evidence against an $N=1$ flat-path evidence would be polluted by this divergence.

**Therefore model selection requires a matched baseline.** Every cell of the grid — including the $N=1$ "single component" case — must be evaluated through the **same** `_JointLogLikelihoodGainMulti` machinery with the **same** proper $s^2$ prior, so that all four $\ln Z$ share one normalization and the differences $\Delta\ln Z$ are meaningful. Mixing a default-path or flat-gain $\ln Z$ into the grid would compare apples to oranges.

## The `force_multi` baseline: running multi even at $N=1$

The control flow that picks the multi-component path is at `burstfit_joint.py:759`:

```python
multi = bool(force_multi) or int(components_C) > 1 or int(components_D) > 1
```

The `components_C>1 or components_D>1` half of that gate alone would **not** route the symmetric single-component case (`components_C == components_D == 1`) into the multi path — it would fall through to the 8-vector `marginalize_gain` branch (`burstfit_joint.py:786-789`), whose flat-gain normalization is *not* comparable to the multi path. The content of the $N=1$ multi vector is byte-for-byte identical to the 8-vector ordering (`burstfit_joint.py:136-138`: names differ only by the `_C1`/`_D1` suffix), but the **likelihood normalization is not** — one uses the proper finite-variance prior, the other the flat improper one.

The **`force_multi`** flag (a real `bool` keyword of `fit_joint_scattering`, default `False`, `burstfit_joint.py:731`) closes exactly this gap: by being the first term of the `multi` gate (`burstfit_joint.py:759`), it forces the fit down the `_JointLogLikelihoodGainMulti` path **even at $N=1$ in both bands**, producing the **C1D1 baseline** on the same gain-marginal likelihood as every richer model in the grid. That is what makes the C1D1 row a legitimate zero point — its $\ln Z$ is computed by `_gain_marginal_multi_band` with the identical proper prior used for C2D2, so $\Delta\ln Z$ subtraction is exact. The CLI driver exposes it as `--force-multi` (`analysis/scattering-refit-2026-06/run_joint_fit.py:104-110`), with help text "run the multi-component likelihood even at C1D1, so its lnZ is normalization-matched to C2/D2 runs (model-selection baseline)".

## The C1D1 / C1D2 / C2D1 / C2D2 grid

The grid varies the number of **temporal components per band**:

- $C$ = number of CHIME temporal components,
- $D$ = number of DSA temporal components.

`C{n}D{m}` denotes `components_C = n, components_D = m`. The four-cell grid spans the single $\to$ paired component choice in each band independently, all evaluated on the matched multi-component gain-marginal likelihood. Read $\Delta\ln Z$ as the log-Bayes-factor of that cell **relative to C1D1**: a positive $\Delta\ln Z$ favors the richer model; a value $\gtrsim 5$ is decisive on the Jeffreys scale, and the values here are orders of magnitude beyond that.

### Grid table (FRB "zach", CHIME+DSA co-detection)

All rows on the multi-component gain-marginal joint likelihood, so the $\ln Z$ are directly comparable.

| model | $\alpha$ | $\tau$ (ms) | $\ln Z$ | $\Delta\ln Z$ vs C1D1 | meaning |
|---|---|---|---|---|---|
| **C1D1** | $3.319 \pm 0.013$ | 0.4162 | 32965.1 | 0 | baseline (`force_multi` at $N=1$) |
| **C1D2** | $3.332 \pm 0.013$ | 0.4132 | 33060.3 | +95.2 | 2nd component in DSA |
| **C2D1** | $2.755 \pm 0.015$ | 0.5577 | 36577.1 | +3612.0 | 2nd component in CHIME |
| **C2D2** | $2.762 \pm 0.015$ | 0.5551 | 37733.7 | +4768.7 | 2nd in both |

**How to read it.** The dominant signal is in CHIME: adding a second CHIME component buys $+3612$ nats (C2D1), roughly $38\times$ the $+95$ nats from a second DSA component (C1D2). Adding the second component in *both* bands (C2D2) is favored over C1D1 by $+4768.7$ nats and over C2D1 by a further $+1156.6$ nats. Crucially, the shared scattering index moves with the CHIME component: $\alpha$ shifts from $3.32$ (single CHIME component) to $2.76$ (paired), a change of $\sim 0.56$ — about $40\times$ the formal $\pm 0.013$ error. The single-component fit's $\alpha$ is therefore **biased high** by an unmodeled CHIME component; once it is included, the recovered slope is sub-Kolmogorov ($\alpha \approx 2.76$ vs the Kolmogorov $\alpha = 4$).

This is corroborated by deterministic peak detection (scipy `find_peaks`, prominence $\geq 4\sigma$ of MAD noise): CHIME shows 1 peak at 13.517 ms (62.7$\sigma$) while DSA shows 2 peaks at 9.634 ms (47.5$\sigma$) and 12.124 ms (15.4$\sigma$). The DSA 12.1 ms peak matches the C2D2 second DSA component placement (see below); the favored *second CHIME* component is a sub-peak structure (a sharp peak-edge feature) that a single peak-finder does not resolve but the evidence strongly prefers.

### Why there is no likelihood pathology

The C2D1/C2D2 wins are not artifacts of a degenerate likelihood. The adversarial reconstruction (`analysis/scattering-refit-2026-06/verify_zach_c2.py`: per-channel GLS gain MAP $\to$ band-integrated residual $\to$ lag-1 autocorrelation, the same metric as the whiteness gate) shows clean conditioning throughout:

| band/model | lag-1 ACF | $\chi^2_\mathrm{red}$ | `frac_culled` | ill-cond ch | note |
|---|---|---|---|---|---|
| CHIME C1D1 | +0.836 | 2.30 | 0 | 0/16 | internal control: reproduces the gate's +0.82 |
| CHIME C2D1 | +0.707 | 1.57 | 0 | 0/16 | $t_0$ sep 0.340 ms $= 1.7\times$ `dt_min` (not pinned) |
| DSA C1D1 | +0.795 | 1.30 | — | — | |
| DSA C1D2 | +0.783 | 1.27 | — | — | 2nd-comp sep 0.207 ms $= 1.1\times$ `dt_min` (pinned at floor) |
| DSA C2D2 | +0.461 | 1.09 | — | — | 2nd-comp sep 2.408 ms $= 12.2\times$ `dt_min` (= the find_peaks 12.1 ms peak) |

`frac_culled = 0` and `0/16` ill-conditioned channels mean the eigenvalue guard (`burstfit_joint.py:221-243`) never fired: no channel had near-degenerate kernels, so the second CHIME component is genuinely resolved, not a merged pair rescued by the rank-1 fallback. The CHIME C1D1 reconstruction reproducing the gate's +0.82 (here +0.836) is the internal control that validates the whole verification chain.

**Honest caveat.** Even C2D2 is not fully white (CHIME lag-1 $\approx 0.71$, DSA $\approx 0.46$). A sharp CHIME peak-edge feature survives — an intrinsic fast rise, a scattering-kernel-shape limit, or a possible third component. So $\alpha = 2.76$ is **better-conditioned, not final**.

## The ordered + min-separation $t_0$ prior transform

The multi-component path replaces the plain cube-to-parameter map with `_JointPriorTransformOrdered` (`burstfit_joint.py:466`), constructed at `burstfit_joint.py:781`. It solves two problems at once for each band's group of component arrival times $t_0$.

### 1. Label-swap degeneracy

With $N$ identical-form components per band, the posterior has $N!$ identical modes (any relabeling of components is the same physical model). Sorting each band's $t_0$ group ascending collapses these to one mode (`burstfit_joint.py:469-470`, `:501`).

### 2. Min-separation, enforced by remapping — not rejection

A naive ordering still lets two components sit arbitrarily close and merge into a degenerate (singular $M_f$) kernel pair. The transform instead **remaps the $t_0$ unit-cube group onto the feasible simplex** $\{t_{0,1} \le \cdots \le t_{0,N},\ \text{gaps} \ge \mathtt{dt\_min}\}$, so **every cube point lands in the feasible region** — no rejected volume, no $-\infty$ returned from the transform (`burstfit_joint.py:471-478`). This totality is required because dynesty's prior transform must be a total map of the unit cube.

The mechanics (`burstfit_joint.py:491-509`): for a band's $t_0$ group with bounds $[\mathrm{lo}, \mathrm{hi}]$, reserve $(N-1)\,\mathtt{dt\_min}$ of separation, giving usable width

$$
\text{usable} = \mathrm{hi} - \mathrm{lo} - (N-1)\,\mathtt{dt\_min}.
$$

Sort the $N$ unit coordinates $\tilde u = \mathrm{sort}(u_\text{grp})$ and place

$$
t_{0,i} = \mathrm{lo} + \tilde u_i\,\text{usable} + (i-1)\,\mathtt{dt\_min}, \qquad i = 0,\dots,N-1,
$$

which is exactly `pts = lo + uu * usable + np.arange(n) * self.dt_min` (`burstfit_joint.py:504`). The cumulative $(i-1)\,\mathtt{dt\_min}$ shift guarantees the gaps. If the band is too narrow ($\text{usable} \le 0$), the group **collapses to a single point** (`burstfit_joint.py:505-507`); the resulting degenerate kernels are then culled by the eigenvalue guard and the merge is **Occam-penalized, not rewarded** (`burstfit_joint.py:474-478`). This is the same fix described above, now enforced geometrically: a model that wants two components where the data supports one pays for it through the Occam term rather than sneaking in free evidence.

### Reading the separations in the grid

The verification table reports each second component's separation as a multiple of `dt_min`:

- **DSA C1D2**: 0.207 ms $= 1.1\times$ `dt_min` — **pinned at the floor**. The second DSA component sits right at the minimum separation, i.e. the transform's `dt_min` constraint is binding; it "does nothing" physically, consistent with the small $+95$ nat gain.
- **CHIME C2D1**: 0.340 ms $= 1.7\times$ `dt_min` — **not pinned**. The two CHIME components ($t_{0,C1}=13.29$ ms, $\zeta=0.093$ narrow; $t_{0,C2}=13.63$ ms, $\zeta=0.413$ broad) are genuinely separated above the floor — a real resolved sub-structure, consistent with the $+3612$ nat gain.
- **DSA C2D2**: 2.408 ms $= 12.2\times$ `dt_min` — far from the floor, and coincident with the deterministic `find_peaks` 12.1 ms DSA peak. C2D2 also places $t_{0,D2}=11.94$ ms ($\zeta=0.31$) on top of $t_{0,D1}=9.529$ ms ($\zeta=0.042$).

A separation pinned at $\sim 1\times$ `dt_min` is the signature of a component the data does not need; a separation at several $\times$ `dt_min` (or matching an independent peak-finder) is a genuinely resolved component.

### Auto `dt_min` = 3× median sample spacing

When the caller does not pass `dt_min`, `fit_joint_scattering` derives it from the band time grids (`burstfit_joint.py:771-776`):

```python
if dt_min is None:
    dts = []
    for m in (model_C, model_D):
        t = np.asarray(m.time, dtype=float)
        dts.append(float(np.median(np.abs(np.diff(t)))) * 3.0)
    dt_min = max(dts)
```

i.e. $\mathtt{dt\_min} = 3 \times \max_\text{bands}\big(\mathrm{median}\,\Delta t\big)$. The factor 3 keeps the minimum separation at least a few time samples — at the kernel's resolution limit (`burstfit_joint.py:480-481`, `:769-770`). The **binding constraint is the coarser (larger-$\Delta t$) band**, hence `max(...)` over bands. For zach:

$$
\mathtt{dt\_min} = 3 \times \max(0.0614,\ 0.0655)\ \text{ms} = 3 \times 0.0655 = 0.197\ \text{ms},
$$

with CHIME $\Delta t = 0.0614$ ms and DSA $\Delta t = 0.0655$ ms (median sample spacings). This single `dt_min` is applied to **both** bands' $t_0$ groups (the same transform instance, `burstfit_joint.py:781`), which is why the verification separations above are all quoted against the one $0.197$ ms floor.

## Reproduction checklist

1. Evaluate **all four** grid cells through `_JointLogLikelihoodGainMulti` (the multi path), including C1D1 at $N=1$ both bands — do **not** let the symmetric case fall through to the 8-vector `marginalize_gain` branch, or its $\ln Z$ will not be comparable. This is the role of `force_multi` (CLI `--force-multi`).
2. Keep the $s^2$ gain-prior policy identical across cells (either all ML-profiled per call, `gain_s2=None`, or all fixed to the same float — `gain_s2`, `burstfit_joint.py:729`; passed into `_JointLogLikelihoodGainMulti(..., s2=gain_s2)` at `:767`).
3. Use the auto `dt_min` ($= 3\times$ median $\Delta t$, $0.197$ ms here) so the ordered transform's floor is data-derived and identical across cells.
4. Report $\Delta\ln Z$ relative to C1D1, and cross-check each surviving component against (a) the eigenvalue-guard diagnostics (`frac_culled`, ill-conditioned channel count) and (b) the residual lag-1 ACF, treating a separation pinned at $\sim 1\times$ `dt_min` as a non-detection.

---

**Source.** This page documents `scattering/scat_analysis/burstfit_joint.py` (`fit_joint_scattering` and its `multi` gate at `:759`, the `force_multi` keyword at `:731`; `_JointLogLikelihoodGainMulti` `:632`; `_gain_marginal_multi_band` likelihood/Occam form `:152`–`:324`; `JOINT_PARAM_NAMES_GAIN_MULTI` `:140`; `_JointPriorTransformOrdered` `:466` and the auto-`dt_min` block `:771`–`:776`), the CLI driver `analysis/scattering-refit-2026-06/run_joint_fit.py` (`--force-multi` `:104`–`:110`, `fit_joint_scattering` call `:133`–`:147`), and `analysis/scattering-refit-2026-06/MULTICOMPONENT_PLAN.md`. The grid numbers, peak counts, and `analysis/scattering-refit-2026-06/verify_zach_c2.py` reconstruction values are from the verified FRB "zach" refit run. All three analysis files exist in this checkout (`main` branch); `force_multi` is an implemented keyword, not a caller-side concept absent from the code.
