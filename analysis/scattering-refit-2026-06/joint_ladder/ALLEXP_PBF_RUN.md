# All-exponential PBF ladder rerun — HPCC campaign (2026-06-24)

PBF = pulse-broadening function; lnZ = log Bayesian evidence.

## Purpose

Every joint CHIME–DSA fit in the original ladder used the `run_joint_fit.py`
default of a **mixed** PBF — `--pbf-C powerlaw --pbf-D exp` (CHIME a Kolmogorov
power-law tail, DSA an exponential). That default is physically incoherent: a
single burst on a single sightline sees one scattering medium, so the PBF
*functional form* is fixed by the screen geometry and only its timescale τ scales
with frequency. A power-law tail at CHIME and an exponential at DSA cannot both
describe the same screen. The per-band preference that motivated the default
(wilhelm, ΔlnZ ≈ +4.0 for per-band over all-exponential) is the model using PBF
shape freedom to absorb band-specific *profile* structure — overfitting, not
medium physics.

This run re-fits the whole ladder under the **physical all-exponential PBF**
(`--pbf-C exp --pbf-D exp`), changing only the PBF and holding each burst's
best-model configuration, α-prior, and gain treatment fixed, so that
α_exp − α_mixed isolates the PBF systematic per burst.

## Design and provenance

- **Driver:** dynamic workflow `allexp-ladder` (run `wf_4b71052d-9b7`; 17 agents,
  ~33 min wall) — Submit → Drain (poll `squeue`) → per-burst Verify → Synthesize.
- **Compute:** Caltech HPCC, `partition=expansion`, via
  `/central/scratch/jfaber/flits-runs/run_joint.sbatch` (8 cores, 16 GB,
  30 min default).
- **Per-burst command** (model flags from each burst's ladder best-model;
  α-bounds [1.5, 6.0] throughout):

  ```
  sbatch run_joint.sbatch <burst> 600 <model-flags> \
         --alpha-lo 1.5 --alpha-hi 6.0 --pbf-C exp --pbf-D exp
  ```

  sharedzeta bursts (`--shared-zeta`): freya, casey, chromatica, wilhelm,
  hamilton. Component bursts (`--marginalize-gain --components-C X
  --components-D Y`): mahi (C1D1 `--force-multi`), phineas (C3D3), oran (C2D1),
  isha (C2D1), zach (C2D3), johndoeII (C2D1). whitney was done separately on the
  local machine (C2D2, re-prepped) and is folded in for completeness.
- **Output naming:** the PBF-tracking edit (this session) records
  `pbf_C/pbf_D/beta_C/beta_D` in each JSON and auto-suffixes non-default PBF runs,
  so all outputs land as `<burst>_joint_fit_<tag>_pbf-exp-exp.json` with no
  clobber of the mixed-default files. Verdicts computed by
  `/central/scratch/jfaber/flits-runs/_allexp_verify.py` and independently
  re-checked from the JSONs (10/10 α match).

## Results

α_mixed is the original mixed-PBF ladder value; α_exp is this run. "railed/wide"
flags a posterior whose median sits within 3σ of a prior bound (1.5 or 6.0) —
either a hard pin or a wide tail that reaches the bound.

| burst | α_mixed | α_exp (all-exp) | Δα | railed/wide | status |
|---|---|---|---|---|---|
| whitney | 5.21 | 5.12 | −0.09 | no | done (local, C2D2, re-prep) |
| isha | 5.39 | 5.48 +0.42/−1.98 | +0.09 | wide → upper | done |
| freya | 4.36 | 4.356 ± 0.04 | −0.004 | no | done |
| phineas | 3.33 | 3.426 ± 0.05 | +0.096 | no | done (job 64538743) — PBF-robust |
| chromatica | 3.28 | 3.286 ± ~0.04 | +0.006 | no | done |
| mahi | 3.80 | 3.17 +1.47/−1.18 | **−0.63** | wide → floor | done |
| oran | 2.69 | 2.662 ± 0.16 | −0.028 | no | done |
| wilhelm | 2.56 | 2.558 ± ~0.04 | −0.002 | no | done |
| zach | 2.41 | 4.59 ± 0.04 | **+2.18** | no (un-railed) | done (job 64538745, COMPLETED 01:44) — **PBF-confounded; see note** |
| casey | 2.40 | 2.396 ± ~0.04 | −0.004 | no | done |
| johndoeII | 1.58 | 1.573 (≈floor) | −0.007 | railed (floor) | done |
| hamilton | 1.50 | 1.504 (≈floor) | +0.004 | railed (floor) | done |

(lnZ is not tabulated here: the sharedzeta and component likelihoods sit on
different normalizations, so lnZ is comparable only exp-vs-mixed *within* a burst,
not across the column.)

## Interpretation

1. **The PBF systematic is small for the well-constrained bursts.** The six
   un-railed bursts — freya, casey, chromatica, wilhelm, oran, and whitney — move
   by |Δα| ≤ 0.09 between the mixed and physical PBF. Their published α are
   *confirmed, not overturned*; the all-exp values agree and become canonical for
   physical consistency.
2. **The per-band PBF was immaterial to the science.** wilhelm — the burst whose
   +4.0 evidence justified the mixed default — has α 2.56 → 2.558 under all-exp.
   The per-band PBF should be dropped: it neither changes a result nor reflects a
   physical screen.
3. **mahi is a new concern.** Under all-exp its α falls 3.80 → 3.17 and the
   posterior widens to +1.47/−1.18, reaching the prior floor. mahi's α is
   PBF-sensitive and poorly constrained once the unphysical CHIME power-law tail
   is removed; it needs a re-prep / model re-check (as whitney did), not a simple
   value swap.
4. **isha, johndoeII, hamilton are unconstrained under either PBF.** isha rails
   the upper bound with a wide lower tail; johndoeII and hamilton are hard-pinned
   at the 1.5 floor. No PBF choice rescues them.

## Outstanding

- **zach (C2D3) COMPLETED** (job 64538745, 2026-06-24 01:44): all-exp α = **4.59 ±
  0.04** (un-railed) vs mixed-PBF **2.41** — **Δα = +2.18, the largest shift in the
  campaign.** zach is the marquee profile-bias case; the mixed-PBF C2D3 value (2.41)
  that the manuscript cites as "hidden component biases α low" is **PBF-confounded**.
  Under the physical all-exp PBF the multi-component α is 4.59, i.e. ABOVE the
  single-component mixed value (3.32) — the claimed bias direction does not survive.
  **Settled (job 64539614):** the single-component all-exp α = **3.319** ≈ the
  single-component mixed 3.32, so under one physical PBF modeling components *raises*
  α (3.319 → 4.59), the reverse of the mixed-PBF artifact (3.32 → 2.41). **Still
  required before any publication claim:** run the fixed-s2 grid to confirm C2D3 is
  still favored under all-exp (the component count rested on profiled-only lnZ).
  Treat zach like mahi/whitney — needs re-prep,
  not a value swap. Result JSON on `hpcc:/central/scratch/jfaber/flits-runs/data/joint/zach_joint_fit_C2D3_pbf-exp-exp.json` (pull to repo).
- **phineas (C3D3) COMPLETED** (job 64538743): all-exp α = **3.426 ± 0.05**
  (un-railed) vs mixed 3.33 — Δα +0.096, **PBF-robust**. Joins the trustworthy
  group; no follow-up needed.
- **zach single-component all-exp COMPLETED** (job **64539614**, C1D1
  `--force-multi`, all-exp): α = **3.319 ± 0.013** (un-railed), essentially
  identical to the single-component *mixed* value 3.32 (Δα = −0.001). This
  settles the direction: **the PBF is immaterial for zach's single-component
  fit** (3.32 either PBF), but the *multiplicity correction* flips sign with the
  PBF — under mixed it lowers α (3.32 → 2.41), under the physical all-exp it
  *raises* α (3.319 → 4.59). The marquee "hidden component biases α high, true
  value lower" thesis is therefore a **mixed-PBF artifact**: under the physical
  PBF the bias direction reverses. zach cannot be cited for profile bias in
  either direction until the fixed-s2 grid confirms C2D3 is still favored under
  all-exp (the component count rested on profiled-only lnZ). JSONs pulled to
  `allexp_json/zach_joint_fit_{C1D1,C2D3}_pbf-exp-exp.json`.
- All 12 bursts now have a completed all-exp fit. The all-exp α for the seven
  well-constrained bursts (freya, casey, chromatica, wilhelm, oran, whitney,
  phineas) become the canonical ladder values, and the manuscript's PBF section
  (`fig:pbf_evidence`, §jointfit "the two bands require different PBFs") is revised
  to drop the per-band PBF as unphysical-and-immaterial.

## Manuscript joint figures — regeneration runbook

Three manuscript figures live in `analysis/scattering-refit-2026-06/dsa_figs/`:
`tau_nu_ladder.png`, `joint_ppc_montage.png`, and the per-burst `*_joint_ppc.png`.
**The committed PNGs are base/mixed-PBF (CHIME powerlaw / DSA exp) and are
SUPERSEDED by ADR-0003 — they must be regenerated from the all-exp family before
manuscript use.** This is the runbook; the underlying α science is the rest of
this doc + `LADDER_SUMMARY.md`. Do not overwrite `dsa_figs/` until the
`@decision` gates in `.agents/deferred-tasks.md` (#32/#33) clear.

**Canonical input set.** All-exp single-exp PBF (`--pbf-C exp --pbf-D exp`), each
burst at its own best-model — the family is **heterogeneous**, not uniform (e.g.
johndoeII C2D1, oran C2D1, wilhelm sharedzeta, phineas C3D3). The per-burst chosen
model is the `chosen` map in `joint_ladder/_figs.py`. Fits + samples:
`<burst>_joint_fit_<tag>_pbf-exp-exp.{json,npz}` on
`hpcc:/central/scratch/jfaber/flits-runs/data/joint/`; locally under
`joint_ladder/allexp_json/` (currently only zach pulled — pull the rest first).

**Scripts.** `_figs.py` (tau_nu_ladder, reads `*_joint_fit*.json`) ·
`joint_ppc.py <burst>` (per-burst PPC, writes `<burst>_joint_ppc.{png,json}`) ·
`plot_joint_posteriors.py`. The `joint_ppc_montage` assembler is **not committed**
(staged in the #33 scratchpad) — commit it on the feature branch when adjudicated.

**χ² convention + the crop reproducibility gap (read before regenerating).**
- Each PPC panel's per-band reduced χ² is computed **at the fit medians on the
  fit's on-pulse-crop window**. The joint fits ran with crop **ON**
  (`FLITS_ONPULSE_CROP` default `"1"` in `run_joint_fit.py:47` and `joint_ppc.py:45`;
  **no override exists on HPCC** — verified by grep). Reproduce χ² with the same
  crop or the numbers move (e.g. oran DSA 5.32 crop-ON → 1.14 crop-OFF).
- **The crop/prep setting is NOT recorded in the fit JSONs** — a known gap. The
  regen pass **should stamp `onpulse_crop` + prep settings into the PPC JSON** so
  χ² is reproducible without re-deriving the flag.
- **Display vs χ² window:** panels should ideally *display* the full uncropped
  0–60 ms profile for inspection while χ² stays on the crop window. A crop-OFF
  re-prepare for display only was prototyped, but `outer_trim`/downsampling also
  narrows the window — the crop flag alone does **not** give a full-window view;
  the prep's trim must also be handled.

**α values vs citable status.** The α *values* are renderable now from the all-exp
`.npz`. Their **quoted/citable status is gated** on the ADR-0004 `ALPHA_MIN`
1.5→1.0 verdict regen (`.agents/deferred-tasks.md` #4, `@human`): which all-exp α
are MARGINAL vs quotable is not yet locked. Render values; do not stamp "citable"
until that gate clears.

## Verdict

The physical all-exp PBF **confirms seven well-constrained bursts** (freya, casey,
chromatica, wilhelm, oran, whitney, phineas; |Δα| ≤ 0.1) — these are publishable.
**Five block the ladder:** **zach** (Δα +2.18, the marquee profile-bias number was
PBF-confounded — single-comp all-exp α = 3.319 ≈ single-comp mixed 3.32, so the
multiplicity correction *reverses* sign with the PBF and the "biases high" thesis
does not survive; needs the all-exp fixed-s2 component-count check before any
claim), **mahi** (Δα −0.63,
wide/floor), **isha** (wide/upper), and **johndoeII**, **hamilton** (hard-railed at
1.5). The dominant lesson: the PBF systematic is negligible for clean,
well-constrained sightlines but can dominate α (zach, mahi) for the marginal ones,
so it must be physical before any α is quoted.
