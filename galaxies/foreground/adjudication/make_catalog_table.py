"""Build the publication foreground catalog (all 49 objects, including inconclusive)
in three forms from the verified pipeline outputs:
  - foreground_catalog.csv                  (full machine-readable table)
  - ../docs-analysis/foreground.md          (MkDocs page: summary + table + provenance)
Single source of truth: foreground_final.csv, enriched with impact parameter (foreground.csv),
redshift errors / morphology (foreground_validated.csv), and TNS names (bursts.csv).
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

fin = pd.read_csv(os.path.join(HERE, "foreground_final.csv"))
fgr = pd.read_csv(os.path.join(HERE, "foreground.csv"))
val = pd.read_csv(os.path.join(HERE, "foreground_validated.csv"))
bur = pd.read_csv(os.path.join(HERE, "bursts.csv"))
for d in (fin, fgr, val):
    d["obj"] = d["obj"].astype(str)
tns = dict(zip(bur.nickname, bur.tns))
mjd = dict(zip(bur.nickname, bur.mjd))

fgr_i = fgr.set_index(["nickname", "type", "obj"])
val_i = val.set_index(["nickname", "type", "obj"])

SURVEY = {"WISE, PS1, STRM": "WISE/PS1/STRM", "Legacy DR8 (Zhou et al. 2021)": "Legacy/Zhou21"}


def survey_short(s):
    s = str(s)
    if "Wen" in s or "DESI, WISE" in s:
        return "DESI/WISE (Wen+)"
    return SURVEY.get(s, s)


def short_reason(verdict, reason, src):
    r = str(reason)
    if verdict == "confirmed":
        return f"{src} < host" if src else "z < host"
    if verdict == "refuted":
        return f"{src} > host" if src else "z > host"
    if "1sigma" in r or "1 sigma" in r:
        return "within 1σ of host"
    if "host-z-unknown" in r:
        return "host z unknown"
    if "not-galaxy" in r:
        return "not a galaxy (STRM UNSURE)"
    if "extrapolated" in r:
        return "photo-z extrapolated"
    return r


def best_redshift(r):
    """Return (z, zerr, source_label, classification)."""
    key = (r.nickname, r.type, r.obj)
    v = val_i.loc[key] if key in val_i.index else None
    # STRM halos: PS1-STRM is authoritative
    if pd.notna(r.strm_class):
        cls = str(r.strm_class)
        if pd.notna(r.strm_zphot):
            return r.strm_zphot, r.strm_zphoterr, "PS1-STRM phot", cls
        return np.nan, np.nan, "PS1-STRM", cls
    # clusters
    if r.type == "cluster":
        src = r.best_z_source
        if src == "desi_specz":
            return r.best_z, (v.desi_zerr if v is not None else np.nan), "DESI spec", "cluster"
        if src == "ned_z":
            return r.best_z, np.nan, "NED", "cluster"
        return r.best_z, np.nan, "phot", "cluster"
    # non-STRM halos
    src = r.best_z_source
    cls = "galaxy"
    if v is not None and pd.notna(v.lsdr9_type):
        cls = {"PSF": "point src"}.get(str(v.lsdr9_type), str(v.lsdr9_type))
    if src == "desi_specz":
        return r.best_z, (v.desi_zerr if v is not None else np.nan), "DESI spec", cls
    if src == "lsdr9_zspec":
        return r.best_z, 0.0, "LS DR9 spec", cls
    if src == "lsdr9_zphot":
        return r.best_z, (v.lsdr9_zphot_std if v is not None else np.nan), "LS/Zhou phot", cls
    if src == "ned_z":
        return r.best_z, np.nan, "NED", cls
    return np.nan, np.nan, "none", cls


rows = []
for _, r in fin.iterrows():
    key = (r.nickname, r.type, r.obj)
    fg = fgr_i.loc[key] if key in fgr_i.index else None
    z, zerr, zsrc, cls = best_redshift(r)
    rows.append(
        dict(
            burst=r.nickname,
            tns=tns.get(r.nickname, ""),
            mjd=mjd.get(r.nickname, np.nan),
            host_z=r.host_z_spec,
            type=r.type,
            obj_id=r.obj,
            survey=survey_short(r.survey),
            ra_deg=r.ra_deg,
            dec_deg=r.dec_deg,
            impact_kpc=(
                round(float(fg.impact_kpc_listed), 1)
                if fg is not None and pd.notna(fg.impact_kpc_listed)
                else np.nan
            ),
            b_over_r500=(
                round(float(fg.b_over_r500), 2)
                if fg is not None and pd.notna(fg.b_over_r500)
                else np.nan
            ),
            redshift=(round(z, 4) if pd.notna(z) else np.nan),
            redshift_err=(round(zerr, 4) if pd.notna(zerr) else np.nan),
            redshift_source=zsrc,
            classification=cls,
            verdict=r.final_verdict,
            reason=short_reason(r.final_verdict, r.final_reason, zsrc),
            sheet_zphot=r.sheet_zphot,
        )
    )
cat = pd.DataFrame(rows)
# order by burst MJD, then type (cluster after halo), then impact
cat["_mjd"] = cat.burst.map(mjd)
cat = cat.sort_values(["_mjd", "type", "impact_kpc"]).drop(columns="_mjd").reset_index(drop=True)
cat.to_csv(os.path.join(HERE, "foreground_catalog.csv"), index=False)


def zfmt(z, e, md=True):
    if pd.isna(z):
        return "—" if md else "\\nodata"
    if pd.isna(e) or e < 5e-4:  # spec-z error is negligible -> show z alone
        return f"{z:.3f}"
    return f"{z:.3f} ± {e:.3f}" if md else f"${z:.3f}\\pm{e:.3f}$"


# ---------- Markdown (MkDocs) ----------
nconf = (cat.verdict == "confirmed").sum()
nref = (cat.verdict == "refuted").sum()
ninc = (cat.verdict == "inconclusive").sum()
md = []
md.append("# Intervening foreground catalog\n")
md.append(
    "Foreground halos and galaxy clusters along the sightlines to the 12 CHIME/DSA "
    "co-detected FRBs, with each candidate **independently validated against public "
    "catalogs** (DESI Legacy Survey DR9 / Zhou+2021 photo-z, DESI DR1 spec-z, NED, "
    "PS1-STRM). Every object — confirmed, refuted, and inconclusive — is listed.\n"
)
md.append('!!! note "Verdict summary"\n')
md.append(
    f"    **{len(cat)}** candidate intervening objects across 12 FRBs: "
    f"**{nconf} confirmed** foreground · **{nref} refuted** (background) · "
    f"**{ninc} inconclusive**. All 49 exist in ≥1 public catalog.\n"
)
md.append(
    "**Verdict definitions** — *confirmed*: best catalog redshift (spec-z, or "
    "photo-z ± error) lies below the FRB host redshift; *refuted*: redshift at/above "
    "the host (background); *inconclusive*: redshift straddles the host within 1σ, the "
    "host has no spec-z, or no trustworthy redshift exists (PS1-STRM `UNSURE` / "
    "extrapolated photo-z).\n"
)
md.append('!!! warning "Caveats carried in this table"\n')
md.append(
    "    - **14 of 15 clusters lie outside their own $R_{500}$** (`b/R500`>1): real "
    "foreground systems, but the sightline does not pierce them.\n"
)
md.append(
    "    - The original spreadsheet `z_phot` column is **unreliable** — for the "
    "PS1-STRM halos it is decoupled from the actual catalog value (e.g. zach 0.013→0.469). "
    "Trust the `redshift` column here, not the sheet.\n"
)
cols = [
    "burst",
    "tns",
    "type",
    "obj_id",
    "survey",
    "impact_kpc",
    "b_over_r500",
    "redshift_disp",
    "redshift_source",
    "classification",
    "verdict",
    "reason",
]
hdr = [
    "Burst",
    "TNS",
    "Type",
    "Obj ID",
    "Survey",
    "$b$ (kpc)",
    "$b/R_{500}$",
    "$z$",
    "$z$ source",
    "Class",
    "Verdict",
    "Note",
]
md.append("\n| " + " | ".join(hdr) + " |")
md.append("|" + "|".join(["---"] * len(hdr)) + "|")
for _, r in cat.iterrows():
    cell = {
        "burst": r.burst,
        "tns": r.tns,
        "type": r.type,
        "obj_id": r.obj_id,
        "survey": r.survey,
        "impact_kpc": ("" if pd.isna(r.impact_kpc) else f"{r.impact_kpc:g}"),
        "b_over_r500": ("" if pd.isna(r.b_over_r500) else f"{r.b_over_r500:g}"),
        "redshift_disp": zfmt(r.redshift, r.redshift_err, md=True),
        "redshift_source": r.redshift_source,
        "classification": r.classification,
        "verdict": r.verdict,
        "reason": r.reason,
    }
    md.append("| " + " | ".join(str(cell[c]) for c in cols) + " |")
md.append('\n!!! info "Provenance"\n')
md.append(
    "    Generated by `scratch/codetection/make_catalog_table.py` from the verified "
    "pipeline (`normalize_codetection.py` → `validate_foreground.py` → "
    "`ps1_strm_adjudicate.py` → `merge_final.py`). Source spreadsheet: "
    "`DSA110_CHIME_Codetection_BurstProperties_Foreground`. Cosmology: Planck18.\n"
)
with open(os.path.join(REPO, "docs-analysis", "foreground.md"), "w") as fh:
    fh.write("\n".join(md) + "\n")

print(
    f"wrote foreground_catalog.csv ({len(cat)} rows), docs-analysis/foreground.md"
)
print(f"verdicts: confirmed={nconf} refuted={nref} inconclusive={ninc}")
print(f"md table data rows: {sum(1 for x in md if x.startswith('| ') and '---' not in x)}")
