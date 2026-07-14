"""Final adjudication of the 9 'WISE,PS1,STRM' halos using the actual PS1-STRM catalog.

Source rows (strm_catalog_rows.csv) were extracted by streaming the PS1-STRM HLSP
declination strip p69-p77 and grepping the 9 objIDs. PS1-STRM (Beck+2020) gives, per
objID: class (GALAXY/STAR/QSO/UNSURE @ prob threshold 0.7), prob_Galaxy, z_phot,
z_photErr (calibrated), z_phot0, and extrapolation_Photoz (1 => photo-z outside the
training coverage, unreliable). This is the redshift + error the spreadsheet omitted.

Foreground rule vs FRB host spec-z, only when STRM trusts the source as a galaxy:
  class != GALAXY                      -> inconclusive-not-galaxy (STRM)
  extrapolation_Photoz == 1            -> inconclusive-extrapolated-photoz
  host z unknown                       -> inconclusive-host-z-unknown
  z_phot + z_photErr < host            -> confirmed foreground
  z_phot - z_photErr > host            -> refuted (background)
  else (within 1 sigma)                -> inconclusive-borderline
Also records sheet_zphot vs strm_zphot to expose the spreadsheet's STRM photo-z error.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
strm = pd.read_csv(os.path.join(HERE, "strm_catalog_rows.csv"))
res = pd.read_csv(
    os.path.join(HERE, "ps1_strm_resolution.csv")
)  # has nickname, host_z, sheet_zphot
res["obj"] = res.obj.astype("int64")
strm["objID"] = strm.objID.astype("int64")
m = res.merge(strm, left_on="obj", right_on="objID", how="left")


def adjudicate(r):
    if r["class"] != "GALAXY":
        return f"inconclusive-not-galaxy (STRM class={r['class']}, prob_gal={r.prob_Galaxy:.2f})"
    if r.extrapolation_Photoz == 1:
        return "inconclusive-extrapolated-photoz (STRM photo-z outside training coverage)"
    host = r.host_z_spec
    z, dz = r.z_phot, r.z_photErr
    if not np.isfinite(host):
        return "inconclusive-host-z-unknown"
    if z + dz < host:
        return "confirmed"
    if z - dz > host:
        return "refuted"
    return "inconclusive-borderline (within 1 sigma of host)"


rows = []
for _, r in m.iterrows():
    has_z = r["class"] == "GALAXY" and r.z_phot > -90
    verdict = adjudicate(r)
    zmis = abs(r.sheet_zphot - r.z_phot) if has_z and np.isfinite(r.sheet_zphot) else np.nan
    rows.append(
        dict(
            nickname=r.nickname,
            obj=r.obj,
            host_z_spec=r.host_z_spec,
            sheet_zphot=r.sheet_zphot,
            strm_class=r["class"],
            strm_prob_galaxy=round(r.prob_Galaxy, 3),
            strm_zphot=(r.z_phot if has_z else np.nan),
            strm_zphoterr=(r.z_photErr if has_z else np.nan),
            strm_extrapolated=(
                int(r.extrapolation_Photoz) if r.extrapolation_Photoz > -90 else np.nan
            ),
            sheet_vs_strm_zphot_diff=(round(zmis, 3) if np.isfinite(zmis) else np.nan),
            strm_verdict=verdict,
        )
    )
out = pd.DataFrame(rows)
out.to_csv(os.path.join(HERE, "ps1_strm_resolution.csv"), index=False)

print(
    out[
        [
            "nickname",
            "obj",
            "host_z_spec",
            "sheet_zphot",
            "strm_class",
            "strm_zphot",
            "strm_zphoterr",
            "strm_extrapolated",
            "sheet_vs_strm_zphot_diff",
            "strm_verdict",
        ]
    ].to_string(index=False)
)
print("\n=== STRM-based verdicts ===")
print(out.strm_verdict.apply(lambda s: s.split(" ")[0]).value_counts().to_string())
big = out[out.sheet_vs_strm_zphot_diff > 0.05]
print(
    f"\nsheet z_phot disagrees with PS1-STRM by >0.05 for {len(big)}/"
    f"{out.sheet_vs_strm_zphot_diff.notna().sum()} galaxies with a STRM photo-z:"
)
print(
    big[["nickname", "obj", "sheet_zphot", "strm_zphot", "sheet_vs_strm_zphot_diff"]].to_string(
        index=False
    )
)
print(f"\nSTRM classifies {(out.strm_class != 'GALAXY').sum()}/9 as non-galaxy (UNSURE).")
