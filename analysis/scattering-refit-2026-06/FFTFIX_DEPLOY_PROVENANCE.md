# burstfit.py next_fast_len FFT-length fix — mid-campaign deploy provenance

**Deployed:** 2026-07-17T16:39:31-07:00 (functional fix live); comment reworded
2026-07-17T16:44:07-07:00 (numerically identical, comment-only).
**File:** `scattering/scat_analysis/burstfit.py` (worktree `joint-tf-fits`).
**Backup:** `scattering/scat_analysis/burstfit.py.bak-fftfix-20260717-163931`.

## The change (2 lines + explanatory comment)
`gaussian_powerlaw_convolution` (the β<3.98 power-law-PBF kernel):

```python
from scipy.fft import next_fast_len as _next_fast_len   # + numpy fallback
...
L = _next_fast_len(2 * T)      # was:  L = 2 * T
```

`next_fast_len` snaps the transform length up to the next pocketfft-fast value
(composite of primes ≤ 11). Exact because L ≥ 2*T still captures the full linear
convolution; the extra zero-padded tail is discarded by the existing `[:T]`
slice. No other kernel touched; `analytic_gaussian_exp_convolution` (β≥3.98,
erfcx) is unchanged.

## Why (the diagnosis)
Prime/near-prime `2*T` forces pocketfft off its O(N log N) path into an O(N^2)
fallback — e.g. freya T=257 (prime) → 2*T=514=2·257. Profiling a production fit
showed the convolution/FFT is the real hot path (92% convolution, 78% in
rfft/irfft), NOT erfcx as an earlier microbench-based note had claimed (that
note is superseded). End-to-end loglike speedup from this one line: **3.16×**.

## Equivalence (validation bar was max|ΔlogL| < 1e-9)
- Kernel, this deploy, smoke in `flits-a1-312`: max|new − old| = **1.11e-15**
  over T ∈ {256,257,300,511,514} (0.0 when 2*T already smooth; ~1e-15 float
  noise on the prime cases). Old path reproduced by monkeypatching
  `_next_fast_len` → identity.
- One-sample joint logL, both branches finite: FFT branch (β=3.62) and analytic
  branch (β=3.99).
- Prior-session end-to-end: max|ΔlogL| = **4.0e-11** on 1000 stored freya
  posterior samples (burstfit_fast.py A/B, same change) — well under the 1e-9 bar.
- Full-stack import (flits + scat_analysis) OK; `py_compile` OK pre- and
  post-swap. Atomic same-filesystem rename so no backfilling job could import a
  half-written file.

## Job provenance (pre-fix vs post-fix)
Running jobs imported burstfit.py at their own start; only NEW starts pick up the
fix. Cutoff = swap at 16:39:31. Because the change is numerically exact, mixing
pre- and post-fix results in the same campaign/TOA table is valid.

**PRE-FIX** (imported old `L = 2*T`; all started ≤ 16:32:44):
- Running at deploy: 58 phineas, 59 zach (D3 baseline), 60 hamilton,
  62 johndoeII, 64 oran, 65 isha, 67 casey, 68 chromatica.
- Already completed pre-deploy (RC=0): freya, mahi, whitney.

**POST-FIX** (import `next_fast_len` on start; all PENDING at deploy):
- 69 wilhelm (production EMG mass-refit backfill).
- 70 jtfra_casey, 71 jtfra_wilhelm (relaxed-α A/B).
- 87–101 pli_* (PL-PBF injection recover/embias grid).
- zach C2D4 (queued behind job 59 — see MULTICOMPONENT).

The two biggest beneficiaries (zach C2D4: 8-CPU, D-band with prime-ish T; and the
relaxed-α pair) both run entirely on the fast path.
