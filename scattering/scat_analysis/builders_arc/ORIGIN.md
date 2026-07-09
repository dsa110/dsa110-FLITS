# Scattering-cube builder provenance (P2.1 verdict: UNVERIFIED_BUILDER)

Captured 2026-07-06 during the trust-reset P2.1 builder hunt
(plan-trust-reset-revalidation.md). Files here are verbatim copies from
CANFAR VOSpace; do not edit.

## What was found

| File | Origin | sha256 |
|---|---|---|
| `get_stokes.ipynb` | `arc:home/jfaber/baseband_morphologies/chime_dsa_codetections/get_stokes/get_stokes.ipynb` | `26bfa2a2c52457c37aed6a3e29a6d9294f24ade3c7dc8ec2dcbca1c20dbd536a` |
| `utils.py` | same dir | `6dbcf5554ff58c830023bd3055d2fdad60d5b3b784306d1e394491723460abcf` |

`get_stokes.ipynb` is the CHIME-side Stokes-I production path: BBData →
`coherent_dedisp(frb_bbdata, dm, time_shift=False, write=True)` →
`incoherent_dedisp` → channel/time masking → power. **The dedispersion call
uses the safe convention** (`time_shift=False`; cf. the wrap footgun
documented at `crossmatching/chime_singlebeam.py:111`) — the P2.1 acceptance
question, answered for this stage.

## What was NOT found — hence UNVERIFIED_BUILDER

The final step that writes the manifest cubes
(`<nick>_<tel>_I_<dm>_{32000|2500}b_cntr_bpc.npy` — centered window,
bandpass-corrected) was located on none of the live hosts. Searched
2026-07-06, all read-only:

- **h17**: `find`/`grep` over `/data/research/astrophysics/frbs/
  chime-dsa-codetections`, `/data/jfaber`, `/data/ubuntu/
  chime-dsa-codetections` — zero `.py` hits for `cntr_bpc|_cntr|32000b|2500b`;
  the only cube-named file is a stray
  `archive/arc_trash_2026-06/stokes_cubes_npy/phineas_dsa_I_610_274_5121b_…`
  (different 5121b window, not a manifest cube).
- **h23** (retired): quarantine + residual tree — no hits.
- **iacobus**: whole `Research/CHIME_DSA_Codetections` — `cntr_bpc` appears
  only in three *consumer* notebooks (`scat_analysis.ipynb` loads
  chromatica; `chromatica_v0.ipynb`/`wilhelm-Copy1.ipynb` load DSA cubes).
- **arc**: `get_stokes/` (this capture) + `arc_trash_2026-06/code/*.py`
  (analysis tools, no cube writer).

Partial chain reconstructed from the archived notebooks
(`arc_trash_2026-06/notebooks/`, mirrored h17 + iacobus):
`<id>_fullstokes.pkl` → `Codetections_Analysis.ipynb` (`np.save('I_<id>')`)
→ `Codetection_Waterfalls.ipynb` (DM-grid `I_<id>_<nick>_<DM>.npy` variants)
→ **[missing: center/window + bandpass-correct + rename to *_cntr_bpc]** →
manifest cubes (arc node dates: 2025-05-19).

## Consequences (plan P2.1 fallback)

- `data-manifest.csv` rows carry `builder=UNVERIFIED_BUILDER`.
- Trust rung (i) for the cubes can only be satisfied by the direct-data
  checks (P2.3 edge/centering/cross-lineage tests) plus lane-B
  regeneration — not by builder audit.
- Byte-identity local↔arc was verified separately (P2.2, 2026-07-06):
  see the manifest status column.

## P2.2 arc byte cross-check procedure (one-shot, performed 2026-07-06)

The `ARC_BYTE_MATCH` status in `data-manifest.csv` was produced by a one-shot
manual procedure, not a committed script:

1. For each of the 24 manifest rows, download the arc VOSpace node to a
   temporary directory using `vcp` (CANFAR VOSpace client):
   `vcp arc:<node_path> <tmp_dir>/<filename>`
2. Compute sha256 of the downloaded file: `sha256sum <tmp_dir>/<filename>`.
3. Compare against the manifest's `sha256` column (written by
   `scripts/fill_data_manifest.py` in P0.1).
4. If all 24 match, upgrade the manifest's `status` column from
   `HASHED_LOCAL` to `ARC_BYTE_MATCH`.

Result (2026-07-06): 24/24 match, 0 mismatch, 0 download failures. Every
cross-check hash equals the manifest sha256. VOSpace exposes no MD5 node
property (checked: creator/date/ispublic/length only), so byte download was
the only content-level comparison available; sizes had already matched 24/24.

The `ARC_BYTE_MATCH` state is pinned by `tests/test_data_manifest.py`. To
re-run the cross-check, repeat the procedure above; no committed script wraps
it because `vcp` is a one-shot manual client and the result is already
byte-pinned in the manifest.
