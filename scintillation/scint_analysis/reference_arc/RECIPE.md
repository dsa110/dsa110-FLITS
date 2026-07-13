# RECIPE.md — reconstructed CANFAR-era CHIME/DSA scintillation analysis sequence

This document reconstructs, as an executable recipe, the collaborator-era
scintillation analysis captured verbatim in this directory (see `ORIGIN.md` for
provenance). Every claim cites the source file and line, or the notebook and
cell index (`notebook:cellN`). The originals are read-only evidence; nothing
here has been edited. Where the code is undetermined, contradictory, or
magic-numbered, it is flagged in §7 rather than guessed.

The methodology lineage is Kenzi Nimmo's spectral-ACF scintillation sequence
(`nimmo2025`), with the two-screen / redshift-aware extensions of Pradeep et al.
(`pradeep2025`). `code/analysis-Copy1.py:874-891` cites both explicitly.


## 1. Module lineage — which file is which, and what actually calls it

The three `scinttools_*.py` files are a refactor chain, and the notebooks import
them under *different* names than the captured filenames. Establishing the
mapping is a prerequisite to reading the recipe, because the same notebook line
means different things depending on which module is bound.

- **`scinttools_old.py`** — the earliest module: `bandpasscorr`, `shift`,
  `autocorr`, and two Lorentzians `lorentz(x,gamma1,m1,c)` /
  `doublelorentz(...)` (`scinttools_old.py:104-108`). Its `lorentz` carries an
  explicit additive offset `c`.
- **`scinttools_new.py`** — adds the real CHIME machinery: `upchannel` /
  `upchannel_nopol` (`scinttools_new.py:11,84`), `make_scallop_model`
  (`:156`), `acf_scint_plot` (`:182`), `acf_per_subband` (`:387`),
  `scint_freq_relation` (`:476`), `emission_size`/`res` (`:526,511`), plus
  `lmfit`-style residual functions (`lorentz_withc_min` `:325`). Its bare
  `lorentz(x,gamma1,m1)` has **no** offset (`:322`), but `lorentz_w_c`
  (`:316`) does. The function set is an exact match for the arc module the
  notebooks import as **`scinttools_v2` / `scint_funcs`** (see below).
- **`scinttools_v3.py`** — a documented rewrite (`scinttools_v3.py:1-9`):
  `calculate_acf` (`:45`), `fit_acf_model` (`:190`), `extract_scint_params`
  (`:286`), `analyze_spectrum` (`:372`), `analyze_subbands` (`:470`),
  `analyze_modulation_over_time` (`:611`), and the plotters. This is the module
  pasted **inline** into the `*_v2` notebooks.

Notebook → module bindings (from the import cells):

| Notebook | Import statement | Effective module |
|---|---|---|
| `scint_freya.ipynb:cell0` | `from scinttools import *` | old-style `autocorr`/`lorentz` (offset form) → **`scinttools_old`** semantics |
| `scint_chromatica.ipynb:cell0` | `from scinttools import *` (+ `from scint...`) | old-style |
| `scint_hamilton.ipynb` | `import scinttools ... as sct` | `sct.autocorr`/`sct.lorentz` |
| `scint_wilhelm.ipynb:cell1` | **no import** — defines its own `compute_acf`/`fit_acf`/`measure_scintillation` inline | self-contained third implementation |
| `scint_freya-Copy1.ipynb:cell0,15` | `import scinttools_v2 as sct`; then `from scint_funcs import upchannel, make_scallop_model, acf_scint_plot, acf_per_subband, scint_freq_relation, ...` | **`scinttools_new`** (arc name `scint_funcs`/`scinttools_v2`) |
| `scint_chromatica_v2.ipynb:cell1` | full **`scinttools_v3`** source pasted inline; also `import scinttools_new as sct`-equivalent (`sct.autocorr`, `sct.lorentz`) in `:cell16-17` | **`scinttools_v3`** + old-style cross-check |
| `scint_wilhelm_v2.ipynb:cell1` | full **`scinttools_v3`** pasted inline | **`scinttools_v3`** |
| `arc_home/scint_{freya,chromatica}_trash.ipynb` | `sct.autocorr`/`sct.lorentz` | old/new-style |

**`analysis-Copy1.py` is not imported by any captured notebook.** Its header
(`analysis-Copy1.py:1-6`) names it `scint_analysis/scint_analysis/analysis.py`
— it is the *later library refactor* (config-driven, BIC model selection, ODR
power-law, self-noise Gaussian, finite-scintle errors) that supersedes the
notebook workflow. It is treated below as the most-evolved *intended* recipe,
distinct from what the notebooks actually ran.


## 2. Two instrument data paths (they share the ACF backbone)

The notebooks operate on two different pre-processed products, with different
channelisations. This matters because `f_res` (the MHz-per-lag conversion) is
set differently in each.

- **DSA-110 path** — `hamilton`, `chromatica`, `chromatica_v2`, `wilhelm`,
  `wilhelm_v2`, `chromatica_trash`. Data are pre-made upchannelized polcal
  filterbanks loaded from `.npy`
  (`scint_hamilton.ipynb:cell1`,
  `/media/ubuntu/ssd/jfaber/.../polcal_fils/<name>_14000_16500.npy`). Resolution
  is hard-coded `tres = 32.768e-3 ms`, `fres = 0.03051757812 MHz`
  (= 30.51757812 kHz), band `1311.25–1498.75 MHz`
  (`scint_hamilton.ipynb:cell2`). These are intensity dynamic spectra; no
  upchannelization is done in-notebook.
- **CHIME path** — `freya`, `freya-Copy1`, `chromatica` (CHIME pkl variant).
  `scint_freya.ipynb:cell1` loads
  `CHIME_pkl/freya_278720455_fullstokes_interp.pkl` (a full-Stokes,
  interpolated intensity cube; `data['I']`, `data['delta_t (ms)']`,
  `data['delta_f (MHz)']`). `freya-Copy1` additionally starts from **raw
  baseband voltages** and upchannelizes them in-notebook (§3).

Both paths then funnel into the same `autocorr → Lorentzian` core.


## 3. End-to-end step sequence

The fullest, most CHIME-faithful realisation is `scint_freya-Copy1.ipynb`
(baseband → upchannelize → scallop-correct → normalize → mask → per-time-sample
ACF → per-subband ACF). The DSA notebooks are a reduced version of the same
pipeline that skips upchannelization and scallop correction. Steps below give
the fullest form first, then the reduction.

### 3a. Upchannelization (CHIME baseband only)

`upchannel(wfall, freq_id, fftsize, downfreq)` (`scinttools_new.py:84-154`;
identical core in `baseband_analysis_core.py:1507-1573`) performs the CHIME
per-channel PFB inversion: for each native 0.390625-MHz channel it takes an
`fftsize`-sample voltage block, `fftshift(fft(...))`, and averages every
`downfreq` output bins. The **upchannelization factor** is
`upchan = fftsize // downfreq` and the resulting fine-channel width is
`0.390625 MHz / upchan`.

- The library default is `fftsize=32, downfreq=2` → factor **16** → fine width
  **≈ 0.02441 MHz** (`scinttools_new.py:84`,
  `baseband_analysis_core.py:1497,1502-1503` — note the core wrapper hard-codes
  `fftsize=32, downfreq=2`, ignoring its own arguments).
- `freya-Copy1` runs it twice: `fftsize=16, downfreq=1` → factor **16**
  (`scint_freya-Copy1.ipynb:cell20`), and a fine `fftsize=512, downfreq=1` →
  factor **512** for the high-resolution spectrum
  (`scint_freya-Copy1.ipynb:cell42`). The notebook's own fine-channel width is
  `f_res = 0.39101 / (fftsize // downfreq)` (`scint_freya-Copy1.ipynb:cell25`),
  i.e. it uses **0.39101 MHz** as the native channel width, not 0.390625.
- The CHIME band edges are `FREQ_TOP=800.1953125 MHz`,
  `FREQ_BOTTOM=400.1953125 MHz`, mapped linearly over `upchan*1024` fine
  channels (`scinttools_new.py:58-62,131-135`). Time is **not** averaged
  (`downtime=1`, "no averaging over complex numbers"; `:39,112`).

`downsample_data(data, f_factor, t_factor)` (`burstfittools.py:542`) is the
plain rebinner used for display and for the morphology fit — it clips to the
nearest multiple and block-averages in frequency then time. It is *not* part of
the scintillation-resolution chain (that would destroy the fine channels);
e.g. `freya` downsamples by `f_factor=190, t_factor=48`
(`scint_freya.ipynb:cell3`) only for the waterfall/`FRBModel` morphology fit.

### 3b. Scalloping / bandpass correction (CHIME baseband only)

The CHIME PFB leaves a periodic intra-channel "scallop" ripple. It is modeled
from the **off-burst** data and divided out:

`make_scallop_model(off_data, fftsize, downfreq)`
(`scinttools_new.py:156-179`): (i) form the off-burst noise power
`|V|²` and average over time to a 1-D spectrum; (ii) z-score it and flag bins
with `|z| > 3` (`:171`) — these spike indices `inds` are returned for later
masking; (iii) reshape the masked spectrum into `(nchan, upchan)` and take the
per-fine-channel mean across native channels → one `upchan`-long scallop shape
`model_scallop`; (iv) tile it back to full length (`:176`). The model is the
*repeating* PFB shape common to all native channels.

Application (`scint_freya-Copy1.ipynb:cell23`): each time column of the burst
intensity is divided by `model_ds` (`I_corrected[:,t] = I[:,t] / model`), then
each frequency channel is normalized to per-channel S/N (§3c).

`bandpasscorr` in `scinttools_old.py:4-15` is a simpler per-row z-score; its own
docstring says it is "not needed for scintillation analysis."

### 3c. Normalization to per-channel S/N

Per frequency channel, subtract the off-burst mean and divide by the off-burst
RMS over an off-pulse time window (`scint_freya-Copy1.ipynb:cell23`, off window
bins `0:200`):
`I_corr[f,:] = (I_corr[f,:] − mean(I_off[f])) / std(I_off[f])`. The
frequency-summed profile is likewise re-zeroed and divided by its off-pulse
std to yield a S/N profile (`:cell23`). `wilhelm`'s inline `normalize_spectrum`
(`scint_wilhelm.ipynb:cell1`) does the same `(on−off_mean)/off_rms` per channel.

The DSA notebooks skip the scallop step but keep this normalization implicitly
by working with the already-calibrated `.npy` intensity and passing an
`offspec_mean` into the ACF (§3f).

### 3d. RFI / masking (in order)

1. **PFB-spike flags** from `make_scallop_model` (`|z|>3`,
   `scinttools_new.py:171`) — applied as `newinds` in
   `scint_freya-Copy1.ipynb:cell52`.
2. **Manual band excision**: specific corrupted frequency ranges are zeroed by
   hand — in `freya-Copy1` a ±4096-fine-channel window around each of
   501.76, 504.88, 492.38, 643.75, 602.35 MHz
   (`scint_freya-Copy1.ipynb:cell52`), "determined by making ACFs of the
   off-burst data."
3. **Zero → masked-array conversion**: after zeroing, `np.ma.masked_where(x==0,
   x)` turns flagged bins into a mask the ACF respects
   (`scint_freya-Copy1.ipynb:cell52`; `scinttools_new.py:210`;
   `analysis-Copy1.py` works throughout on `np.ma.MaskedArray`). NaNs are
   likewise treated as masked (`np.nan_to_num`/`isnan` throughout).

The CHIME baseband library layer `get_snr`
(`baseband_analysis_analysis.py:72`) is the upstream cleaning that produced the
CHIME pkls: it normalizes power, refines DM over `DM ± DM_range/2` in
`DM_step=0.01` steps (`:89,132-135`), removes RFI channels
(`get_RFI_channels`, thresholds `thres_mean=5, thres_std=3`, `:86-87`), and
cuts empty-band edges (`spectrum_lim`, `spectrum_thresh=2`, `:80-81`). This is
provenance context for the pkls, not a step the scintillation notebooks re-run.

### 3e. On-pulse / off-pulse selection

Uniformly by **hand-picked time-bin windows**, never an automated matched
filter:

- DSA notebooks: on-pulse = a symmetric window of `±outer_bound` bins about the
  profile centre, summed over time; off = the leading bins. `hamilton`
  `outer_bound=30` (`scint_hamilton.ipynb:cell11`); `chromatica_v2`/`_trash`
  `outer_bound=5`, off = bins `0:150` (`scint_chromatica_v2.ipynb:cell16`,
  `scint_chromatica_trash.ipynb:cell12`). The burst is first roll-centred by
  `argmax` of the timeseries (`scint_hamilton.ipynb:cell2`).
- `freya.ipynb:cell12-13`: on-pulse = time bins `725:875` (hand-read from the
  profile), off implicit.
- `freya-Copy1`: burst window `beginburst=250, endburst=400`
  (`:cell22`), off window `0:200` (`:cell23`); the per-time-sample loop scans
  bins `250:400` (`:cell25`).
- `scinttools_v3` `analyze_spectrum`/`analyze_subbands` take an explicit
  `time_range=(burst_start_bin, burst_end_bin)` — e.g. `(100,120)` for
  `chromatica_v2` (`:cell4`), `(95,130)` for `wilhelm_v2` (`:cell4`) — and
  average over it; off = `[:burst_start]` (`scint_chromatica_v2.ipynb:cell4`).
- `analysis-Copy1.py` derives the off window from `burst_lims`: either symmetric
  (`use_symmetric_noise_window`) or `[0 : burst_lims[0] − off_burst_buffer]`
  with `off_burst_buffer=100` (`analysis-Copy1.py:319-326`).

### 3f. ACF computation

There are **four** distinct ACF implementations. All normalize so ACF(0) ≈ the
squared modulation index m², all mean-subtract, and all handle masks pairwise.

**(i) `scinttools_old`/`_new` `autocorr`** (`scinttools_old.py:30-68`,
`scinttools_new.py:272-310`) — the notebook workhorse (`sct.autocorr`).
- Mean-subtracts the unmasked spectrum (`x[v!=0] -= nanmean`, `:53/:295`).
- Normalization denominator `denom = xmean²`, or `(xmean − offspec_mean)²` if
  an off-burst mean is supplied (`:47-50/:289-292`). This is what makes ACF(0)
  read as m².
- Each lag: `Σ shift(x,0)·shift(x,i)·mask / (Σmask · denom)` using a
  triple-length `shift` buffer to carry negative lags (`shift`, `:17-27/:260`).
- **Zero-lag noise spike**: `zerolag=False` (the default everywhere) skips
  lags 0 and 1 (`if i>1`, `:61/:303`), writing `ACF[i-1]`, so the noise spike at
  lag 0 is dropped and the array is offset by one bin.
- `maxlag=None` computes all lags; otherwise `maxlag_bin = int(maxlag/f_res)`
  (`scinttools_new.py:220`).
- The driver then builds a symmetric ACF by mirroring
  (`acf = concat(acf[::-1], acf)`; lags `× f_res`; e.g.
  `scint_freya.ipynb:cell13`, `scinttools_new.py:226-229`).

**(ii) `scinttools_v3.calculate_acf`** (`scinttools_v3.py:45-187`) — the
rewrite. Pairwise `work_spec[idx1]·work_spec[idx2]` over `idx1=[0..n-lag)`,
`idx2=[lag..n)` with a per-lag valid-pair count (`:130-141`), divided by the
count (`:148`). Normalization is by `effective_mean² = (mean − offspec_mean)²`
(`:155-161`), with fallbacks to variance if that is ~0. Lag 0 is kept and equals
Var/⟨I⟩² (`:122`). `max_lag_bins = ceil(max_lag_mhz / freq_res)`
(`analyze_spectrum`, `:422`).

**(iii) `analysis-Copy1.calculate_acf`** (`analysis-Copy1.py:130-238`) — adds
per-lag **errors**. ACF value = mean of valid products / `denom`
(`denom=(mean_on−off_mean)²`, `:170,186`); **statistical error** = standard
error of the mean of the products (`:190-192`); **finite-scintle error** =
`|ACF| · N_scintles^{-1/2}` where `N_scintles = total_BW / Δν_DC` and Δν_DC is
the interpolated HWHM of the ACF (`:208-225`); the two are combined in
quadrature (`:236`). The symmetric array inserts an explicit **1.0 at zero lag**
(`:229`) with a `1e-9` error floor (`:232`). Default `max_lag_bins =
n_unmasked//4` (`:163`).

**(iv) `wilhelm` inline `compute_acf`** (`scint_wilhelm.ipynb:cell1`) — a
self-contained pairwise ACF with denominator `mask.sum()·(mean_spec − 1.0)`
(note the literal `−1.0`, assuming an off-pulse baseline normalized to unity).
Its `measure_scintillation` (`:cell1`) also does the equal-S/N sub-banding.

**Sub-band splitting schemes** (all split the *burst-integrated spectrum* in
frequency):
- **Equal-frequency**: contiguous blocks of `nchan // num_subbands`
  (`scinttools_new.py:404-411` with `snsubband=False`;
  `scinttools_v3.py:512-517`; `analysis-Copy1.py:347-349`).
- **Equal-S/N** (accumulate channels until each sub-band holds ≈`total/N` of the
  integrated flux): `scinttools_new.py:412-424` (`snsubband=True`),
  `scinttools_v3.py:519-560` (`divide_method='equal_snr'`, cumulative-signal
  `searchsorted`), `analysis-Copy1.py:350-359`.
- Sub-band counts actually used: `freya-Copy1` `num_subbands=8`, equal-S/N,
  `maxlag=1000` (`scint_freya-Copy1.ipynb:cell54`); `chromatica_v2` `12`,
  equal-S/N, `max_lag_mhz=5` (`:cell4`); `wilhelm_v2` `6`, equal-S/N,
  `max_lag_mhz=5` (`:cell4`); the `analysis-Copy1` default is `8`
  (`:307`).

### 3g. Fit model — Lorentzian form, HWHM/FWHM convention

The scintillation ACF is fit with a Lorentzian (sum) plus a constant offset.
The peak amplitude maps to m² and the width parameter maps to the scintillation
bandwidth. **The HWHM/FWHM convention differs between generations — this is the
single most important consistency hazard.**

- **Single Lorentzian**: `m² / (1 + (Δν/γ)²) + c`
  (`scinttools_old.py:107`, `scinttools_new.py:316`,
  `analysis-Copy1.py:18`, `scinttools_v3.py:27-35`). Here **`γ` is the HWHM**,
  and `m` (the fitted amplitude's square root) is the modulation index of that
  component.
- **Scintillation bandwidth Δν_d = γ (HWHM) in the notebooks.** Every notebook
  reports `result.params['gamma1']` directly as "Δν" — e.g.
  `scint_freya.ipynb:cell13` (`Δν = gamma1 ± stderr`, kHz),
  `scint_hamilton.ipynb:cell11`, `scint_chromatica_v2.ipynb:cell16`. No factor
  of 2 is applied. `freya-Copy1` likewise reads `gamma1` from
  `lorentz_withc_min` (`:cell25`).
- **`scinttools_v3` breaks this convention**: it fits HWHM `gamma` but *reports*
  **FWHM = 2·gamma** as the scintillation bandwidth (`scinttools_v3.py:19-24`
  config note; `extract_scint_params` `fwhm_mhz = 2.0*gam1`, `:313`), and
  `mod_index = sqrt(amplitude)` (`:317`). So the `*_v2` notebooks' subband
  "fwhm_mhz" is 2× the notebooks' "gamma". They are the same physical width only
  after that factor.
- **Multi-component / mixed models** (used for candidate two-screen fits):
  double and triple Lorentzian (`scinttools_new.py:313,319`,
  `analysis-Copy1.py:21,26`), Gaussian components (`analysis-Copy1.py:32-44`),
  Gaussian+Lorentzian (`:46`), and a physical two-unresolved-screen model
  `lor1+lor2+lor1·lor2+c` (`:51-55`). For Gaussian components the width is
  converted to a HWHM-equivalent by `σ·√(2 ln 2)` before reporting
  (`analysis-Copy1.py:758,1008`).
- **Fit range / weighting / zero-lag handling in the fit**:
  - Notebooks fit only the central `±lag_range_for_fit` window: `2000 kHz`
    (`freya`, `hamilton`, `wilhelm_v2`), `4000 kHz`
    (`chromatica_v2:cell16`), `3 MHz` (`freya-Copy1:cell25`). "Somewhat
    arbitrary; 2000 works well to fit the central peak" is the verbatim comment
    (`scint_hamilton.ipynb:cell11`).
  - `scinttools_new.acf_scint_plot` fits with `lmfit` `Model(lorentz_w_c)`,
    initial `gamma=0.001, m=1, c=0` (`scinttools_new.py:242`).
  - `analysis-Copy1._fit_acf_models` (`:521-648`) fits **all** candidate models
    with **error-weighted** `lmfit` (`weights = 1/max(err,1e-9)`, `:544`),
    **excludes the zero-lag point** (`fit_mask = |lag|≤range & lag≠0`, `:536`),
    and constrains multi-component widths to be ordered
    (`gamma2 = gamma1·factor`, `factor>1.01`, `:574-575`).
  - Model choice: `analysis-Copy1._select_overall_best_model` sums **BIC**
    across sub-bands and picks the lowest (`:650-692`), unless a `force_model`
    is set in config (`:717-724`). The notebooks instead fix a single-Lorentzian
    by hand.

### 3h. Frequency scaling index γ (a.k.a. α/β) — Δν_d(ν) power law

The scaling of scintillation bandwidth with frequency, `Δν_d(ν) ∝ ν^α`, is
extracted from the per-sub-band `(ν_center, Δν_d)` points. **How this is done is
the least settled part of the recipe** and differs sharply across generations:

- **`scinttools_new`**: `acf_per_subband` collects `sub_scint = |gamma|` vs
  `sub_cent` per sub-band and *overplots a fixed `ν⁴` reference*
  `sub_scint[-1]·(freqs/sub_cent[-1])**4` (`scinttools_new.py:468-469`) — it
  does **not fit** the index. A fit function `scint_freq_relation(v,c,n)=c·vⁿ`
  and its residual exist (`:476-484`) but are not called on real sub-band data
  in any captured notebook.
- **`scinttools_v3`**: `plot_subband_summary` similarly overlays a hard-coded
  `∝ ν^4.0` line anchored at the highest-frequency point
  (`scinttools_v3.py:851-864`); no index is fit.
- **Notebooks (`chromatica_v2:cell23`, `chromatica_trash:cell19`)**: fit
  `power_law(x)=a·x^b+c` with `lmfit`/`curve_fit`, `b` initialised to 3 — but on
  **hard-coded** arrays `xdata=[0.45,0.6,0.75,1.4]` GHz, `ydata=[0.4,0.5,0.75,
  1.1]` (`scint_chromatica_v2.ipynb:cell23`). These are placeholder numbers, not
  wired to the measured sub-bands (see §7).
- **`analysis-Copy1.py` (the intended method)**: fits the index properly in
  **log-space** with orthogonal-distance regression to handle x- and y-errors:
  `log10(Δν) = α·log10(ν) + log10(c)`, `ODR(..., beta0=[4.0,0.0])`
  (`analysis-Copy1.py:844-873`). Sub-band errors are `√(fit_err² +
  finite_scintle_err²)`, propagated into log-space (`:839,847`). Components are
  sorted narrowest-first before the fit (`:814`). The fitted α is then
  interpreted against theory: **α≈4** = unresolved point source, **α≈3** =
  resolved emission region, **α≈1** = two fully-resolving screens (Pradeep/Nimmo;
  `analysis-Copy1.py:880-891`), accepted only within 3σ.


## 4. Modulation index m — vs time and vs frequency

Two independent definitions coexist:

- **From the ACF amplitude (vs frequency, per sub-band)**: `m = m1` (the fitted
  Lorentzian amplitude's root) — `scinttools_v3.extract_scint_params`
  `mod_index = √amplitude` (`scinttools_v3.py:317`);
  `analysis-Copy1` carries `mod` per component (`:773,817`);
  `freya-Copy1:cell25` records `mods.append(result_acf.params['m1'])` and an
  **alternative** `m = √(max(ACF_fit) − c)` (peak-minus-offset;
  `scint_freya-Copy1.ipynb:cell25`) with propagated error.
- **From the raw statistics (vs time)**: `m = std/mean` of the
  frequency-averaged intensity in a time chunk —
  `scinttools_v3.analyze_modulation_over_time` (`scinttools_v3.py:611-735`,
  `mod_index = chunk_std/chunk_mean`, `:731`). Chunking is sliding-window:
  `time_chunk_size_bins` with `time_overlap_bins` overlap — `chromatica_v2`
  `(3,2)`, `wilhelm_v2` `(3,2)` real / `(10,2)` synthetic
  (`scint_chromatica_v2.ipynb:cell4`, `scint_wilhelm_v2.ipynb:cell4-5`).
  `freya-Copy1:cell25` is the manual equivalent: it walks time bins `250:400`
  (step 2), and for every bin above a **S/N threshold of 64**
  (`if prof_sn[i-1] > 64`) computes a per-time-sample ACF and fits m(t) and
  Δν(t).
- **`analysis-Copy1.analyze_intra_pulse_scintillation`** (`:915-1033`) is the
  library version of m(t)/Δν(t): it slices the on-pulse into
  `intra_pulse_time_bins` (default 10, `:938`) chunks, ACF-fits a *1-component*
  model per chunk (it refuses multi-component here, `:944-946`), and records
  `(time_s, bw, mod)` per slice.

The physical modulation-index → emission-size link is
`σ = √((1/m² − 1)/4)`, `emission_size = σ · phys_res`
(`scinttools_new.py:526-535`), with lens resolution `res(...)` from the
scattering time (`:511-524`) — used downstream, not in the ACF fit itself.


## 5. Per-burst deviation table

Per-burst tweaks encode the undocumented judgment calls; they are enumerated
here rather than averaged away.

| Notebook | Instrument | ACF impl. | On-pulse window | fit lag range | Sub-bands | γ(ν) | m computed? | Notable deviation |
|---|---|---|---|---|---|---|---|---|
| `freya.ipynb` | CHIME pkl | `old.autocorr` | time bins `725:875` (`cell13`) | 2000 kHz | none | no | no | Adds an **FFT structure-function** cross-check `D(Δν)=2[R0−R]`, half-power/e-folding Δν (`cell14-15`); `f_res=30.51757812 kHz` hard-coded (a DSA number reused on CHIME data — see §7) |
| `freya-Copy1.ipynb` | CHIME baseband | `new` (`scint_funcs`) | burst `250:400`, off `0:200` | 3 MHz | 8, equal-S/N (`cell54`) | `scint_freq_relation` available, ^4 overplot | **yes**, m(t) and Δν(t) per time sample | The only notebook doing real **upchannelization (16 & 512) + scallop model + manual band excision**; ACF correlated errors `var_f=1+2·Σacf²`; S/N gate 64 |
| `chromatica.ipynb` | CHIME pkl | `old.autocorr` | central ±`outer_bound` | 2000 kHz | none in main | structure-function path | no | Mirrors `freya.ipynb` structure (ACF + structure function) |
| `chromatica_v2.ipynb` | DSA `.npy` | `v3` inline + `sct` cross-check | `(100,120)` (`cell4`); central ±5 (`cell16`) | 2500 kHz fit / 4000 kHz cross-check | 12, equal-S/N | `power_law a·xᵇ+c`, **hard-coded xy** (`cell23`) | yes (v3 subband m and m(t)) | Two parallel measurements: v3 machinery **and** an old-style single-band `sct.autocorr` fit; also a manual 1536-channel frequency-chunk scan (`cell17`) |
| `wilhelm.ipynb` | DSA `.npy` | **own inline** `compute_acf`/`fit_acf` | `on_idx`/`off_idx` args | curve_fit auto | 8, equal-S/N (`measure_scintillation`) | not fit | via amplitudes | Entirely self-contained (no `scinttools`); `fit_acf` defaults to **3 Lorentzians** (`n_screens=2` ⇒ 1 broad + 2 narrow); ACF denom uses `mean_spec−1.0` |
| `wilhelm_v2.ipynb` | DSA `.npy` | `v3` inline | `(95,130)` real (`cell4`); `(50,150)` synthetic (`cell5`) | 2500 kHz real / 1500 kHz synth | 6 equal-S/N real; 8 equal-freq synth | ^4 overplot only | yes (std/mean, chunk 3/2) | Contains a full **synthetic-data validation** cell (`cell5`: injects `β=4` scint, `α=−2.5` spectrum, 1 screen) alongside the real burst |
| `hamilton.ipynb` | DSA `.npy` | `sct.autocorr` | central ±30 | 2000 kHz | none | no | no | Simplest single-band Δν; `outer_bound=30` (widest on-pulse window); loads `chromatica_240203aacl` despite the filename |
| `chromatica_trash.ipynb` | DSA `.npy` | `sct.autocorr` | central ±5, off `0:150` | 4000 kHz | none in main | `power_law` hard-coded (`cell19`) | no | arc-home earlier draft of `chromatica_v2`; passes explicit `offspec_mean` into `autocorr` |
| `freya_trash.ipynb` | CHIME pkl | `sct.autocorr` | central, off passed | (as `cell12`) | none | no | no | arc-home earlier draft of the `freya` single-band path |
| `casey.ipynb` | — | — | — | — | — | — | — | **Empty stub** (0 cells); kept for completeness only |


## 6. What `analysis-Copy1.py` adds over the notebooks (intended end-state)

`analysis-Copy1.py` is the refactor the notebooks were converging toward and is
the best statement of intent even though no captured notebook imports it:
per-lag ACF errors (statistical + finite-scintle, `:190-225`); error-weighted,
zero-lag-excluded, width-ordered `lmfit` fits of 8 candidate models
(`:521-648`); BIC-based model selection across sub-bands (`:650-692`);
log-space ODR power-law with a physical α interpretation (`:844-891`); a fixed
Gaussian self-noise term whose width σ_self is computed from the 16–84% pulse
energy interval per Pradeep Eq. 7 (`:74-92,57-64`); and Monte-Carlo synthetic
radiometer-noise subtraction from the ACF (`:95-124,392-400`). Porting the
notebook recipe *forward* means adopting these; porting it *faithfully* means
reproducing the notebook's HWHM (γ) convention and hand-picked windows.


## 7. Ambiguities, magic numbers, and contradictions (flagged, not resolved)

1. **HWHM vs FWHM scintillation-bandwidth convention is inconsistent.** The
   notebooks report Δν_d = γ (HWHM) directly (§3g), but `scinttools_v3` reports
   FWHM = 2γ (`scinttools_v3.py:313`). Any table mixing `*_v2` sub-band widths
   with the older single-band γ values is off by a factor of 2 unless
   reconciled. Downstream comparisons must state which convention each number
   uses.
2. **The frequency scaling index γ/α is never actually fit to measured data in
   the captured notebooks.** `chromatica_v2:cell23` and
   `chromatica_trash:cell19` fit `a·xᵇ+c` to **hard-coded** `xdata=[0.45,0.6,
   0.75,1.4]`, `ydata=[0.4,0.5,0.75,1.1]` — placeholder values, initial `b=3`.
   `scinttools_new`/`_v3` only *overplot* a fixed `ν⁴`. A real per-burst α comes
   only from `analysis-Copy1.py`'s ODR (`beta0=[4,0]`), which no notebook runs.
   The published α values must be traced to whichever script actually consumed
   the sub-band measurements — it is not these notebooks.
3. **`f_res` provenance for CHIME is suspect.** `freya.ipynb:cell13` hard-codes
   `f_res = 30.51757812 kHz` — the **DSA-110** fine-channel width
   (`scint_hamilton.ipynb:cell2`) — while working on a CHIME pkl. Whether the
   CHIME `*_fullstokes_interp.pkl` was genuinely interpolated onto a
   30.518-kHz grid, or this is a copy-paste of the DSA number, is
   undetermined; the pkl's own `data['delta_f (MHz)']` is read at `cell2` but
   then ignored at `cell13`. Its imshow `extent` also shows `1300–1500 MHz`
   (DSA band) on CHIME data — likely a stale plotting constant. **Do not trust
   the CHIME Δν absolute scale from `freya.ipynb` without re-deriving the grid.**
4. **Native CHIME channel width is written two ways.** `upchannel` uses
   0.390625 MHz implicitly (`400/1024`; band 800.1953125→400.1953125,
   `scinttools_new.py:131-135`) but `freya-Copy1:cell25` computes
   `f_res = 0.39101/(fftsize//downfreq)` — 0.39101 vs 0.390625 is a ~0.1%
   discrepancy of unknown origin (rounding, or a different edge convention).
5. **The captured `scinttools_new.py` may not be byte-identical to the arc
   module the notebook binds.** `chromatica_v2:cell16` and `hamilton:cell11`
   call `sct.lorentz(x, gamma1, m1, c)` **with an offset**, but the captured
   `scinttools_new.lorentz` (`:322`) takes no `c`. So the arc `scinttools_v2`
   the notebooks actually imported had an offset-bearing `lorentz` (matching
   `scinttools_old`), which the captured file does not. The exact fit function
   used at run time is therefore not fully pinned by this capture.
6. **The zero-lag `i>1` skip in `autocorr` silently shifts the ACF by one bin.**
   `ACF[i-1] = ...` under `zerolag=False` (`scinttools_old.py:63`) means lag
   index 1 is dropped and every subsequent bin is written one position early,
   which the mirror-and-concatenate driver then treats as symmetric. Whether the
   half-bin offset this introduces was intended is undocumented.
7. **`fit_acf` component count in `wilhelm.ipynb`.** `n_screens=2` yields
   **three** Lorentzians (one broad seed + two narrow, `scint_wilhelm.ipynb:cell1`),
   not two — the naming implies screens, the code adds an extra broad component.
   Which physical picture (2 vs 3 screens) the fit encodes is ambiguous.
8. **`max_lag`/`maxlag` units are overloaded.** `acf_scint_plot` treats `maxlag`
   as MHz (`scinttools_new.py:220`), but `acf_per_subband` is called with
   `maxlag=1000` (`freya-Copy1:cell54`) and `maxlag=4` (`cell25`) — the former
   cannot be MHz for a ~few-hundred-MHz band, so it is presumably bins there.
   The same keyword means different units in different calls.
9. **`analyze_modulation_over_time` `m=std/mean` vs ACF-amplitude `m`.** The two
   modulation-index definitions (§4) are not guaranteed to agree (the first
   includes radiometer noise, the second is a fitted, noise-subtracted
   amplitude). Notebooks report whichever is convenient per burst without
   reconciling them.
10. **Off-burst window choice is per-burst and unjustified.** Off windows are
    `0:200` (`freya-Copy1`), `0:150` (`chromatica_v2`, `chromatica_trash`),
    or "leading bins" (DSA path) with no stated rule tying the window length to
    the burst width; `analysis-Copy1.py` offers a symmetric-window option but the
    notebooks do not use it.
