#!/usr/bin/env python
"""On-pulse residual check for the joint CHIME+DSA fits — the component-count backstop.

Owner accept gate (2026-07-17 amendment): after each accepted fit, any *contiguous*
residual structure above ~5 sigma in the on-pulse window of EITHER band means a
temporal component was ignored -> escalate that band's component count and refit.

Metric (per band, from a `<burst>_jointmodel<tag>.npz` dump):
  1. Residual grid  r[f,t] = (data - model) / noise   over valid channels.
  2. Band-integrated residual S/N profile  P[t] = sum_f r[f,t] / sqrt(F_valid).
     Under a correct model + independent Gaussian noise this is ~N(0,1) per bin,
     so a real ignored sub-pulse shows as a contiguous positive run in P.
  3. On-pulse window = time bins where the band-integrated MODEL exceeds 1% of its
     peak (the residual test is restricted here, per "on-pulse window").
  4. Escalate iff the longest contiguous run of P > RESID_SIGMA within the on-pulse
     window is >= MIN_CONTIG bins. A genuine ignored sub-pulse is a *localized*,
     coherent, >5 sigma positive excess spanning several bins — exactly a contiguous
     run above threshold. A broad envelope / scattering-tail mismatch instead spreads
     its power thin (each bin < 5 sigma) and inflates only the wide boxcar
     matched-filter; that is a model-adequacy issue, NOT a missing component (adding
     components would not fix it), so `matched_resid_snr` is REPORTED as a diagnostic
     but does NOT drive escalation. Single-bin spikes (MIN_CONTIG guard) are RFI.

`resid_prof_max` (signed peak of P in the on-pulse window) is the number reported
per burst per band in the campaign table.

  python residual_check.py <jointmodel_npz> [<jointmodel_npz> ...]   # prints + writes _resid.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RESID_SIGMA = 5.0        # contiguous run threshold (sigma), owner spec ~5
MIN_CONTIG = 2           # min contiguous bins above threshold to count (reject 1-bin RFI)
ONPULSE_FRAC = 0.01      # on-pulse = model band-profile above this fraction of its peak
POS_DOMINANCE = 1.5      # emission excess must exceed POS_DOMINANCE x |negative dip| to escalate
# A shape_mismatch driven by a NARROW (few-bin matched-filter) +/- dipole is the
# signature of a sub-bin timing/width mismatch -- often just under-resolution of a
# bright narrow feature. The owner rule (2026-07-17): a single-bin matched dipole
# in a band => drop that band's time binning one level and refit, UNLESS the band
# is already at its native product resolution, in which case no finer bin exists
# and the dipole is a genuine pulse-shape (PBF model-family) limit -> shape-model
# review. A BROAD dipole (wide matched width) is a diffuse envelope/tail mismatch
# that finer binning would not fix -> shape-model review regardless.
NARROW_DIPOLE_MAXW = 2   # matched-filter width (bins) at/below which a dipole is "narrow"
# Native product time-sample floors (ms) = telescopes.yaml dt_ms_raw. A band whose
# fit bin width is within ~1.5x of this is already at native -> cannot bin finer.
NATIVE_DT_MS = {"CHIME": 2.56e-3, "DSA": 32.768e-3}
NATIVE_DT_TOL = 1.5


def _runs_above(mask: np.ndarray) -> list[tuple[int, int]]:
    """[(start, end)] half-open index runs where boolean mask is True."""
    m = np.asarray(mask, bool)
    idx = np.flatnonzero(np.diff(np.r_[0, m.view(np.int8), 0]))
    return list(zip(idx[::2], idx[1::2]))


def _matched_max(prof: np.ndarray, max_width: int | None = None) -> tuple[float, int, int]:
    """(snr, width, center) of the maximum boxcar matched-filter response on a
    unit-variance profile: sum(w bins)/sqrt(w) scanned over log-spaced widths."""
    n = prof.size
    if n < 2:
        return (float(prof.max()) if n else 0.0, 1, 0)
    if max_width is None:
        max_width = max(2, n)
    widths, w = [], 1
    while w <= max_width:
        widths.append(w)
        w *= 2
    best = (-np.inf, 1, 0)
    c = np.cumsum(np.r_[0.0, prof])
    for w in widths:
        if w > n:
            break
        s = (c[w:] - c[:-w]) / np.sqrt(w)
        i = int(np.argmax(s))
        if s[i] > best[0]:
            best = (float(s[i]), w, i + w // 2)
    return best


def _dipole_recommendation(shape_mismatch, matched_w, at_native):
    """Owner rule: what to do about a shape_mismatch dipole.

    Narrow single-bin dipole + a finer bin available -> "refine_time" (the
    under-resolution fix). Narrow dipole already at native -> "shape_model_review"
    (no finer bin; it is a PBF pulse-shape limit). Broad dipole -> "shape_model_review"
    regardless. ``at_native=None`` (band resolution unknown) keeps the branch open.
    Returns None when there is no shape_mismatch to act on."""
    if not shape_mismatch:
        return None
    narrow = matched_w <= NARROW_DIPOLE_MAXW
    if not narrow:
        return "shape_model_review"      # diffuse envelope/tail mismatch
    if at_native is True:
        return "shape_model_review"      # narrow dipole but no finer bin -> shape model
    if at_native is False:
        return "refine_time"             # narrow dipole, finer bin exists -> drop one level
    return "refine_time_unless_native"   # narrow dipole, native-ness not resolved here


def band_residual(data, model, noise, valid, *, band="", at_native=None):
    """Per-band residual metrics from a dumped data/model/noise/valid grid.

    ``at_native`` (bool | None): whether this band is already at its native
    product time resolution, used to branch the shape_mismatch recommendation
    (see :func:`_dipole_recommendation`)."""
    data = np.asarray(data, float)
    model = np.asarray(model, float)
    sig = np.clip(np.asarray(noise, float).reshape(-1), 1e-9, None)
    v = np.asarray(valid).reshape(-1).astype(bool)
    if v.sum() == 0:
        v = np.ones(data.shape[0], bool)
    resid = (data - model) / sig[:, None]
    rv = resid[v]
    rv = np.where(np.isfinite(rv), rv, 0.0)
    fv = max(int(v.sum()), 1)
    prof = rv.sum(axis=0) / np.sqrt(fv)                 # ~N(0,1) per bin under null
    mband = model[v].sum(axis=0)                        # band-integrated model
    onpulse = mband > ONPULSE_FRAC * float(mband.max() if mband.size else 0.0)
    if not onpulse.any():
        onpulse = np.ones_like(prof, bool)
    op_idx = np.flatnonzero(onpulse)
    lo, hi = int(op_idx[0]), int(op_idx[-1]) + 1
    op = prof[lo:hi]
    # signed peak (positive = unmodeled emission excess = the ignored-component signature)
    imax = int(np.argmax(op))
    resid_prof_max = float(op[imax])
    resid_prof_min = float(op.min())
    # longest contiguous run above +RESID_SIGMA within the on-pulse window
    runs = _runs_above(op > RESID_SIGMA)
    longest = max((e - s for s, e in runs), default=0)
    matched_snr, matched_w, matched_c = _matched_max(op)
    # A contiguous >5 sigma run is either (a) an ignored sub-pulse -> a POSITIVE emission
    # excess (resid_prof_max dominates), which escalation fixes; or (b) a near-symmetric
    # +/- dipole on a bright narrow feature -> a sub-bin timing/width model-shape mismatch,
    # which more components would NOT fix (they over-fit). Escalate only on (a); flag (b)
    # as shape_mismatch for visual vetting (the owner vets by eye). matched_resid_snr is a
    # reported diagnostic of diffuse (envelope/tail) mismatch and never drives escalation.
    contig = longest >= MIN_CONTIG
    pos_dominated = resid_prof_max > POS_DOMINANCE * abs(resid_prof_min)
    escalate = bool(contig and pos_dominated)
    shape_mismatch = bool(contig and not pos_dominated and abs(resid_prof_min) > RESID_SIGMA)
    recommend = _dipole_recommendation(shape_mismatch, int(matched_w), at_native)
    return dict(
        band=band,
        resid_prof_max=resid_prof_max,
        resid_prof_min=resid_prof_min,
        resid_prof_max_abs=float(max(abs(resid_prof_max), abs(resid_prof_min))),
        matched_resid_snr=float(matched_snr),
        matched_width=int(matched_w),
        n_contig_5sig=int(longest),
        onpulse_bins=int(hi - lo),
        escalate=escalate,
        shape_mismatch=shape_mismatch,
        at_native=(None if at_native is None else bool(at_native)),
        recommend=recommend,
    )


def _at_native(z, key: str, band: str) -> bool | None:
    """Is this band's fit bin width within NATIVE_DT_TOL of the native product
    floor (telescopes.yaml dt_ms_raw)? Derived from the dumped time axis (ms)."""
    t = np.asarray(z[key], float) if key in z.files else None
    floor = NATIVE_DT_MS.get(band)
    if t is None or t.size < 2 or floor is None:
        return None
    dt = float(np.median(np.diff(t)))
    return bool(dt <= NATIVE_DT_TOL * floor)


def check_dump(npz_path) -> dict:
    z = np.load(npz_path, allow_pickle=True)
    out = {"npz": str(npz_path), "burst": str(z["burst"]) if "burst" in z else "?",
           "nC": int(z["nC"]) if "nC" in z else 1, "nD": int(z["nD"]) if "nD" in z else 1}
    out["C"] = band_residual(z["dataC"], z["modelC"], z["noiseC"], z["validC"],
                             band="CHIME", at_native=_at_native(z, "timeC", "CHIME"))
    out["D"] = band_residual(z["dataD"], z["modelD"], z["noiseD"], z["validD"],
                             band="DSA", at_native=_at_native(z, "timeD", "DSA"))
    out["escalate_C"] = out["C"]["escalate"]
    out["escalate_D"] = out["D"]["escalate"]
    out["escalate"] = out["escalate_C"] or out["escalate_D"]
    out["shape_mismatch"] = out["C"]["shape_mismatch"] or out["D"]["shape_mismatch"]
    return out


def _fmt(r):
    flag = " ESCALATE" if r["escalate"] else (" SHAPE-MISMATCH" if r["shape_mismatch"] else "")
    rec = r.get("recommend")
    nat = {True: " @native", False: " sub-native", None: ""}[r.get("at_native")]
    rec_s = f" -> {rec}{nat}" if rec else ""
    return (f"{r['band']:5s} resid_max={r['resid_prof_max']:+6.2f}s "
            f"(min {r['resid_prof_min']:+.2f}) matched={r['matched_resid_snr']:.2f}s "
            f"w={r['matched_width']} contig5s={r['n_contig_5sig']}bin{flag}{rec_s}")


def main(argv):
    for p in argv:
        try:
            r = check_dump(p)
        except Exception as e:
            print(f"{p}: FAILED {type(e).__name__}: {e}")
            continue
        print(f"\n{r['burst']}  C{r['nC']}D{r['nD']}  [{Path(p).name}]")
        print("   " + _fmt(r["C"]))
        print("   " + _fmt(r["D"]))
        jp = Path(p).with_name(Path(p).stem + "_resid.json")
        json.dump(r, open(jp, "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
