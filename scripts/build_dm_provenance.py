#!/usr/bin/env python
"""Build the per-burst, per-telescope DM provenance table (V6 / Phase 6 P6.2).

Joins the CHIME-side DM measurements in ``crossmatching/chime_side_inputs.json``
with the DSA-side reference DMs in ``configs/bursts.yaml``, documents the
dedispersion method and producing artifact for each side, and quantifies the
CHIME-DSA agreement per burst (``delta_dm`` and ``delta_dm_sigma``). Emits
``crossmatching/dm_provenance.csv``.

Trust context (CONTEXT.md, Wave 3): DM_obs is revoked until this per-telescope
provenance exists. This table does NOT by itself re-certify DM_obs -- it exposes
exactly what is and is not documented so the owner can decide. Two structural
findings are baked into the data and surfaced here rather than smoothed over:

  1. The DSA-side dm_err is a uniform 0.1 pc/cm^3 placeholder floor
     (configs/bursts.yaml), not a measured uncertainty. delta_dm_sigma uses the
     same sigma_eff = max(quadrature errors, 1.0 pc/cm^3) convention as
     crossmatching/association.py::dm_agreement -- the 1 pc/cm^3 physical floor
     that keeps a sub-pc offset from reading as many-sigma. So delta_dm_sigma
     matches the committed association_report.json ``n_sigma`` (up to sign), and
     a small value means "consistent within the physical DM scale". See ``note``.
  2. Four bursts (whitney, oran, johndoeII, mahi) are CHIME-unconstrained
     (0-2 sub-bands above S/N 4); their CHIME DM is absent and the agreement is
     undefined -- documented-null (the pipeline ran and reported non-detection),
     not UNDOCUMENTED.

Run from a FLITS checkout at the Faber2026 pin:
    conda run -n flits python scripts/build_dm_provenance.py
"""
from __future__ import annotations

import csv
import json
import pathlib
from math import hypot

ROOT = pathlib.Path(__file__).parents[1]
CHIME_INPUTS = ROOT / "crossmatching/chime_side_inputs.json"
BURSTS_YAML = ROOT / "configs/bursts.yaml"
OUT = ROOT / "crossmatching/dm_provenance.csv"

# Physical DM tolerance floor on the combined sigma, matching
# crossmatching/association.py::dm_agreement (dm_floor). The CHIME arrival
# regression returns a statistical sigma far below the ~1 pc/cm^3 scale at which
# a DM difference is physically meaningful, so without the floor a sub-pc offset
# reads as many-sigma. sigma_eff = max(quadrature errors, DM_FLOOR).
DM_FLOOR = 1.0

# DSA-side DM is a frozen catalog reference remeasured by neither this pipeline
# nor CHIME (DM audit, dm-provenance-audit-2026-07-07.md): bursts.yaml carries
# the value with a uniform dm_err=0.1 placeholder. Document both facts.
DM_DSA_METHOD = "catalog value (frozen DSA-110 reference DM)"
DM_DSA_SOURCE = (
    "configs/bursts.yaml [dm]; dm_err=0.1 is a uniform placeholder floor, "
    "not a measured per-burst uncertainty"
)
# CHIME extraction artifacts (chime_dm_final.json, grid NPZ) live off-repo on
# h17 and are not pinned (DM audit gap #2); the pinned repo has only the summary
# JSON + audit note. Cite that chain honestly.
CHIME_SOURCE = (
    "crossmatching/chime_side_inputs.json (audit: .agents/audit-chime-side-dm.md); "
    "extraction artifacts chime_dm_final.json + grid NPZ off-repo on "
    "h17:/data/... (not pinned)"
)

FIELDS = [
    "nickname",
    "dm_dsa", "dm_dsa_err", "dm_dsa_method", "dm_dsa_source",
    "dm_chime", "dm_chime_err", "dm_chime_method", "dm_chime_source",
    "delta_dm", "delta_dm_sigma",
    "dm_confidence", "note",
]


def _load_bursts_yaml() -> dict:
    import yaml

    raw = yaml.safe_load(BURSTS_YAML.open())
    return raw["bursts"]


def build_rows() -> list[dict]:
    chime = json.load(CHIME_INPUTS.open())
    dsa = _load_bursts_yaml()

    rows: list[dict] = []
    for rec in chime:  # preserves the sample (chronological-by-MJD) order
        nick = rec["name"]
        # bursts.yaml keys are lowercased (johndoeii); chime inputs use
        # johndoeII. Normalise the join key.
        dsa_rec = dsa[nick.lower()]

        dm_dsa = dsa_rec["dm"]
        dm_dsa_err = dsa_rec["dm_err"]
        dm_chime = rec.get("dm_chime")
        dm_chime_err = rec.get("dm_chime_err")
        method = rec.get("method") or "UNDOCUMENTED"

        if dm_chime is not None and dm_chime_err is not None:
            delta = dm_chime - dm_dsa  # signed: CHIME - DSA (negative = CHIME below)
            # sigma_eff floors the quadrature error at the 1 pc/cm^3 physical
            # scale (association.py convention); |delta_dm_sigma| == report n_sigma.
            sigma_eff = max(hypot(dm_chime_err, dm_dsa_err), DM_FLOOR)
            delta_s = f"{delta:+.4f}"
            sigma_s = f"{delta / sigma_eff:+.3f}"
            note = ("sigma_eff = max(quadrature, 1.0 floor); DSA err is a 0.1 "
                    "placeholder, so the floor governs; consistent within the "
                    "physical DM scale")
        else:
            # CHIME-unconstrained: documented non-detection, agreement undefined.
            delta_s = ""
            sigma_s = ""
            note = "CHIME DM unconstrained (agreement undefined): " + (
                rec.get("dm_status") or "no dm_status recorded"
            )

        rows.append({
            "nickname": nick,
            "dm_dsa": f"{dm_dsa:.4f}",
            "dm_dsa_err": f"{dm_dsa_err:.4f}",
            "dm_dsa_method": DM_DSA_METHOD,
            "dm_dsa_source": DM_DSA_SOURCE,
            "dm_chime": "" if dm_chime is None else f"{dm_chime:.4f}",
            "dm_chime_err": "" if dm_chime_err is None else f"{dm_chime_err:.4f}",
            "dm_chime_method": method,
            "dm_chime_source": CHIME_SOURCE,
            "delta_dm": delta_s,
            "delta_dm_sigma": sigma_s,
            "dm_confidence": rec.get("dm_confidence", ""),
            "note": note,
        })
    return rows


def main() -> None:
    rows = build_rows()
    with OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    n_constrained = sum(1 for r in rows if r["delta_dm"])
    print(f"wrote {OUT.relative_to(ROOT)} ({len(rows)} rows; "
          f"{n_constrained} with CHIME-DSA agreement, "
          f"{len(rows) - n_constrained} CHIME-unconstrained)")


if __name__ == "__main__":
    main()
