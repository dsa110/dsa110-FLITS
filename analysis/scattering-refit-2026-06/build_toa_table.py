#!/usr/bin/env python
"""Joint-fit relative-coordinate campaign table: per burst, per instrument.

Columns: chosen component counts, crop-relative centroid +/- err, alpha, beta, tau_1ghz,
delta_dm per band, resolution factors (f/t) + achieved peak S/N, residual max per
band, and OLD-vs-NEW where an OLD fit exists.

MODEL-COORDINATE REFERENCE (uniform, stated): the FLUENCE-WEIGHTED CENTROID of the
component arrival coordinates per band, weights = OLS-recovered per-component spectral fluence at
the posterior median (from the jointmodel dump), error = posterior spread of the
centroid (component t0's varied over the equal-weight posterior, weights fixed).
These coordinates are relative to each independently cropped model time axis; they
are not absolute or cross-telescope TOAs. Reduces exactly to t0 +/- err for a single-component band. Component-count changes
vs OLD, and bursts where CHIME and DSA resolve DIFFERENT counts (matched reference
most delicate), are flagged explicitly.

  FLITS_RUNS=~/Developer/scratch/flits-local-runs \
  CAMPAIGN_LOGS=<...>/campaign_A1 python build_toa_table.py
"""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

import numpy as np
from dynesty.utils import resample_equal

RUNS = Path(os.environ.get("FLITS_RUNS", os.path.expanduser("~/Developer/scratch/flits-local-runs")))
JOINT = RUNS / "data/joint"
OLD = Path(os.environ.get("OLD_FITS", "/private/tmp/claude-501/-Users-jakobfaber-Developer-repos-github-com-jakobtfaber-Faber2026/a573656f-9ea8-4ebd-aab6-58332c63c659/scratchpad/fits_OLD_campaign"))
LOGS = Path(os.environ.get("CAMPAIGN_LOGS", "/private/tmp/claude-501/-Users-jakobfaber-Developer-repos-github-com-jakobtfaber-Faber2026/a573656f-9ea8-4ebd-aab6-58332c63c659/scratchpad/campaign_A1"))
HERE = Path(__file__).resolve().parent

# burst -> (new_tag, old_tag). Chosen counts = joint_ladder/_figs.py `chosen` map,
# beta-native re-fit. whitney corrected base->C2D2 per owner. Component-count deltas
# vs OLD (hamilton sharedzeta->C4D1, zach C1D1->C2D3) are surfaced as flags below.
BURSTS = [
    ("freya", "", "_sharedzeta"),
    ("casey", "", "_sharedzeta"),
    ("chromatica", "", "_sharedzeta"),
    ("wilhelm", "", "_sharedzeta"),
    ("hamilton", "_C4D1", "_sharedzeta"),
    ("mahi", "_C1D1", "_C1D1"),
    ("oran", "_C2D1", "_C2D1"),
    ("isha", "_C2D1", "_C2D1"),
    ("whitney_fine", "_C2D2", "_C2D2"),
    ("johndoeII", "_C2D2", "_C2D2"),
    ("phineas", "_C3D3", "_C3D3"),
    ("zach", "_C2D3", "_C1D1"),
]


def load(fp: Path):
    return json.load(open(fp)) if fp.exists() else None


def pget(d, k):
    if not d:
        return None
    p = d.get("percentiles", {}).get(k)
    if p:
        return p["median"], p.get("err_minus", 0.0), p.get("err_plus", 0.0)
    if k in d and isinstance(d[k], dict) and "median" in d[k]:
        return d[k]["median"], d[k].get("err_minus", 0.0), d[k].get("err_plus", 0.0)
    return None


def t0_cols(names, band):
    """Column indices for a band's t0 params (t0_C or t0_C1..t0_C{n}), in order."""
    idx = [(n, i) for i, n in enumerate(names) if re.fullmatch(rf"t0_{band}\d*", n)]
    idx.sort(key=lambda x: (len(x[0]), x[0]))  # t0_C before t0_C1..; numeric order otherwise
    return [i for _, i in idx], [n for n, _ in idx]


def centroid_toa(samples_npz: Path, fluence, band):
    """Fluence-weighted centroid TOA (median, err_minus, err_plus) + per-component
    t0 medians, from the equal-weight posterior. Weights fixed at median fluence."""
    if not samples_npz.exists():
        return None, []
    z = np.load(samples_npz, allow_pickle=True)
    names = list(z["param_names"])
    cols, cnames = t0_cols(names, band)
    if not cols:
        return None, []
    eq = resample_equal(z["samples"], z["weights"])   # (E, P) equal-weight
    t0s = eq[:, cols]                                   # (E, k)
    w = np.ones(len(cols), float)
    if fluence is not None and len(fluence) == len(cols):
        w = np.clip(np.asarray(fluence, float), 0.0, None)
        if w.sum() <= 0:
            w = np.ones(len(cols), float)
    w = w / w.sum()
    cen = t0s @ w                                       # (E,)
    med, lo, hi = np.percentile(cen, [50, 16, 84])
    comp_meds = [(cnames[j], float(np.median(t0s[:, j])), float(w[j])) for j in range(len(cols))]
    return (float(med), float(med - lo), float(hi - med)), comp_meds


def caption_from_log(burst):
    fp = LOGS / f"{burst}.log"
    if not fp.exists():
        return {}
    txt = fp.read_text()
    out = {}
    for band in ("CHIME", "DSA"):
        m = re.search(rf"AUTO-TF {band}\s*:\s*(.*)", txt)
        if m:
            c = m.group(1).strip()
            g = dict(caption=c)
            for key, pat in (("ff", r"f(\d+)/"), ("tf", r"/t(\d+)\)"),
                             ("win_ms", r"window ([\d.]+) ms"), ("snr", r"peak S/N (\d+)/px"),
                             ("nch", r"(\d+) ch"), ("dt_us", r"dt=([\d.]+) us")):
                mm = re.search(pat, c)
                if mm:
                    g[key] = mm.group(1)
            out[band] = g
    return out


def resid_from_json(burst, tag):
    fp = JOINT / f"{burst}_jointmodel{tag}_resid.json"
    if not fp.exists():
        return None
    return json.load(open(fp))


def dump_arrays(burst, tag):
    npz = JOINT / f"{burst}_jointmodel{tag}.npz"
    if not npz.exists():
        return None
    z = np.load(npz, allow_pickle=True)
    return dict(
        chi2C=float(z["chi2C"]) if "chi2C" in z else None,
        chi2D=float(z["chi2D"]) if "chi2D" in z else None,
        fluenceC=z["fluenceC"] if "fluenceC" in z else None,
        fluenceD=z["fluenceD"] if "fluenceD" in z else None,
        nC=int(z["nC"]) if "nC" in z else 1,
        nD=int(z["nD"]) if "nD" in z else 1,
    )


def fmt(v, u=1e3):
    return "--" if v is None else f"{v[0]:+.4f} (+{v[2]:.4f}/-{v[1]:.4f})"


def main():
    rows = []
    print("=" * 120)
    print("JOINT-FIT CAMPAIGN (amended): NEW = beta-native [3,4], AUTO S/N-driven prep, evidence-selected counts")
    print("RELATIVE MODEL COORDINATE = fluence-weighted centroid per independently cropped band")
    print("=" * 120)
    for burst, ntag, otag in BURSTS:
        new = load(JOINT / f"{burst}_joint_fit{ntag}.json")
        old = load(OLD / f"{burst}_joint_fit{otag}.json")
        cap = caption_from_log(burst)
        da = dump_arrays(burst, ntag)
        rj = resid_from_json(burst, ntag)
        nC = (new or {}).get("components_C", 1)
        nD = (new or {}).get("components_D", 1)
        missing = "" if new else "  [NEW FIT MISSING]"
        # TOA centroids (NEW)
        samp = JOINT / f"{burst}_joint_samples{ntag}.npz"
        toaC, compC = centroid_toa(samp, (da or {}).get("fluenceC"), "C")
        toaD, compD = centroid_toa(samp, (da or {}).get("fluenceD"), "D")

        flags = []
        if nC != nD:
            flags.append(f"CHIME/DSA resolve DIFFERENT counts (C{nC} vs D{nD}) — matched-ref delicate")
        oc = (old or {}).get("components_C", 1)
        od = (old or {}).get("components_D", 1)
        if old and (oc != nC or od != nD):
            flags.append(
                f"component count CHANGED vs OLD (C{oc}D{od}->C{nC}D{nD}) — "
                "relative centroid may shift materially"
            )
        for band, r in (("CHIME", (rj or {}).get("C")), ("DSA", (rj or {}).get("D"))):
            if r and r.get("escalate"):
                flags.append(f"{band} residual ESCALATE (resid_max {r['resid_prof_max']:+.1f}s, "
                             f"{r['n_contig_5sig']}bin) — ignored component; refit at higher count")
            elif r and r.get("shape_mismatch"):
                flags.append(f"{band} residual SHAPE-MISMATCH (+/-{r['resid_prof_max_abs']:.1f}s dipole) "
                             f"— bright-pulse shape/resolution, NOT a missing component")

        print(f"\n----- {burst}  NEW C{nC}D{nD}{missing} -----")
        for band, g in (("CHIME", cap.get("CHIME")), ("DSA", cap.get("DSA"))):
            if g:
                print(f"   {band:5s} resolution: f{g.get('ff','?')}/t{g.get('tf','?')}  "
                      f"{g.get('nch','?')}ch  dt={g.get('dt_us','?')}us  win={g.get('win_ms','?')}ms  "
                      f"peakS/N={g.get('snr','?')}/px")
        print(f"   chi2_red  CHIME={(da or {}).get('chi2C')}  DSA={(da or {}).get('chi2D')}")
        if rj:
            rc, rd = rj.get("C", {}), rj.get("D", {})
            print(f"   residual_max  CHIME={rc.get('resid_prof_max'):+.2f}s (contig5s={rc.get('n_contig_5sig')})  "
                  f"DSA={rd.get('resid_prof_max'):+.2f}s (contig5s={rd.get('n_contig_5sig')})")
        for key, lab in [("tau_1ghz", "tau_1GHz(ms)"), ("alpha", "alpha"), ("beta", "beta"),
                         ("delta_dm_C", "dDM_C(pc/cc)"), ("delta_dm_D", "dDM_D(pc/cc)")]:
            print(f"   {lab:14s} OLD {fmt(pget(old,key)):32s}  NEW {fmt(pget(new,key))}")
        print(f"   RELATIVE_MODEL_CHIME (centroid, ms)  NEW {fmt(toaC)}")
        for nm, m, w in compC:
            print(f"       component {nm:7s} t0={m:+.4f} ms  (fluence weight {w:.2f})")
        print(f"   RELATIVE_MODEL_DSA   (centroid, ms)  NEW {fmt(toaD)}")
        for nm, m, w in compD:
            print(f"       component {nm:7s} t0={m:+.4f} ms  (fluence weight {w:.2f})")
        if flags:
            for fl in flags:
                print(f"   >> FLAG: {fl}")

        chime_snr = (cap.get("CHIME") or {}).get("snr")
        dsa_snr = (cap.get("DSA") or {}).get("snr")
        rcC = (rj or {}).get("C", {}) or {}
        rcD = (rj or {}).get("D", {}) or {}
        tau_n = pget(new, "tau_1ghz")
        al_n = pget(new, "alpha")
        rows.append(dict(
            burst=burst, comp=f"C{nC}D{nD}",
            chime_ff_tf=f"f{(cap.get('CHIME') or {}).get('ff','?')}/t{(cap.get('CHIME') or {}).get('tf','?')}",
            dsa_ff_tf=f"f{(cap.get('DSA') or {}).get('ff','?')}/t{(cap.get('DSA') or {}).get('tf','?')}",
            chime_peaksnr=chime_snr, dsa_peaksnr=dsa_snr,
            chi2_C=(da or {}).get("chi2C"), chi2_D=(da or {}).get("chi2D"),
            tau_ms=(tau_n[0] if tau_n else None),
            alpha=(al_n[0] if al_n else None),
            relative_model_C_ms=(toaC[0] if toaC else None),
            relative_model_C_err_ms=((toaC[1] + toaC[2]) / 2 if toaC else None),
            relative_model_D_ms=(toaD[0] if toaD else None),
            relative_model_D_err_ms=((toaD[1] + toaD[2]) / 2 if toaD else None),
            resid_max_C=rcC.get("resid_prof_max"), resid_max_D=rcD.get("resid_prof_max"),
            escalate_C=rcC.get("escalate"), escalate_D=rcD.get("escalate"),
            flags="; ".join(flags),
        ))
    csv_fp = HERE / "joint_tf_relative_model_coordinates.csv"
    with open(csv_fp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_fp}")


if __name__ == "__main__":
    main()
