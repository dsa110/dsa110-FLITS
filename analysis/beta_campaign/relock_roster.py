#!/usr/bin/env python
"""Re-lock the citable roster from the beta-campaign verdicts (Phase 6/9).

Rewrites analysis/scattering-refit-2026-06/citable_alpha_roster.json (the path
every consumer -- budget tau ingestion, tau_consistency, figures -- already
reads) with beta-era entries, and copies each member's campaign fit + PPC JSON
into analysis/beta_campaign/fits/ (committed) with a per-entry fit_json
override so find_citable_joint_json resolves them without the legacy
_a1_fits naming.

Membership (plan-beta-coherent-thin-screen-campaign):
  - gate_one final != FAIL (chromatica-class L2 catastrophes are out);
  - quoted values become beta with derived alpha; railed-hi bursts quote
    alpha = 4 as a geometry-conditioned limit (exponential-consistent,
    ADR-0007 re-open candidates), not a measurement.
Tiering: PASS/MARGINAL with both bands' chi2 < 2 -> tier A; other MARGINAL ->
tier B (caveat carried from the gate reason). whitney_fine stays the
multiplicity exemplar. zach keeps its Pass-2 tab:beta exclusion flag
(fixed-s2 multiplicity not robust; CONTEXT.md 2026-06-27) while remaining a
budget-tau member, matching the pre-campaign split.

  conda run -n flits python analysis/beta_campaign/relock_roster.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROSTER = REPO / "analysis" / "scattering-refit-2026-06" / "citable_alpha_roster.json"
FITS_DIR = HERE / "fits"

CHI2_TIER_A_MAX = 2.0


def _entry(r: dict, runs_root: Path) -> dict:
    burst, suffix = r["burst"], r["suffix"]
    fit_rel = f"analysis/beta_campaign/fits/{burst}_joint_fit{suffix}.json"
    railed = r["rail_class"] == "railed-hi"
    e = {
        "nickname": burst.removesuffix("_fine"),
        "tns": None,  # filled below (import kept local for HPCC-safety)
        "model": suffix.lstrip("_"),
        "beta": round(r["beta"], 4),
        "beta_err": [round(x, 4) for x in r["beta_err"]],
        "rail_class": r["rail_class"],
        "tau_1ghz": r["tau"],
        "chi2_chime": r["chi2_chime"],
        "chi2_dsa": r["chi2_dsa"],
        "gate_final": r["final"],
        "fit_json": fit_rel,
    }
    if railed:
        e["alpha_limit"] = 4.0
        e["alpha_note"] = (
            "posterior railed at beta=4 (square-law/exponential member): alpha = 4 "
            "quotable only as a geometry-conditioned limit; ADR-0007 re-open candidate"
        )
    elif r["rail_class"] == "interior":
        e["alpha"] = round(r["alpha"], 4)
    else:
        # railed-lo (beta=3 prior edge, alpha=6 L1 ceiling) or mass at both
        # bounds: not a measurement and not a physical boundary member --
        # neither alpha nor a limit is quotable.
        e["alpha_unconstrained"] = True
        e["alpha_note"] = r["rail_detail"]["detail"]
        e["excluded_from_tab_beta"] = "beta unconstrained (" + r["rail_class"] + ")"
    if burst == "zach":
        e["excluded_from_tab_beta"] = (
            "Pass-2 fixed-s2 multiplicity (C2D3 vs C2D2) not robust "
            "(CONTEXT.md 2026-06-27); budget-tau member only"
        )
    return e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    verdicts = json.loads((HERE / "beta_campaign_verdicts.json").read_text())
    if verdicts["missing"]:
        raise SystemExit(f"refusing to re-lock with missing fits: {verdicts['missing']}")
    runs_root = Path(verdicts["runs_root"])

    import sys
    from datetime import date

    sys.path.insert(0, str(REPO))
    from scattering.scat_analysis.burst_metadata import load_tns_name

    tier_a, tier_b, excluded, exemplar = [], [], [], None
    for r in verdicts["rows"]:
        if r["final"] == "FAIL":
            excluded.append({"nickname": r["burst"], "reason": r["reason"]})
            continue
        e = _entry(r, runs_root)
        e["tns"] = load_tns_name(e["nickname"]) or e["nickname"]
        cc, cd = r["chi2_chime"], r["chi2_dsa"]
        clean = all(x is not None and x < CHI2_TIER_A_MAX for x in (cc, cd))
        citable_rail = r["rail_class"] in ("interior", "railed-hi")
        if r["burst"] == "whitney_fine":
            e["note"] = "multiplicity exemplar (C2D2 after local re-prep)"
            exemplar = e
        elif clean and citable_rail:
            tier_a.append(e)
        else:
            e["caveat"] = r["reason"]
            tier_b.append(e)
        if not args.dry_run:
            FITS_DIR.mkdir(exist_ok=True)
            for kind in ("fit", "ppc_multi"):
                src = runs_root / "data/joint" / f"{r['burst']}_joint_{kind}{r['suffix']}.json"
                if src.exists():
                    shutil.copy2(src, FITS_DIR / src.name)

    roster = {
        "locked_utc": str(date.today()),
        "adr": "docs/adr/0005-citable-alpha-roster.md "
        "(beta re-lock under ADR-0006; docs/rse/specs/plan-beta-coherent-thin-screen-campaign.md)",
        "membership_rule": [
            "beta co-model joint fit (ADR-0006), thin-screen pass 1",
            "gate_one final != FAIL",
            "interior posteriors quote beta and derived alpha",
            "beta=4-railed posteriors quote alpha = 4 as a geometry-conditioned limit "
            "(ADR-0004 rail rule; ADR-0007 re-open candidates)",
        ],
        "tier_a_fully_adjudicated": tier_a,
        "tier_b_provisional_pending_s2": tier_b,
        "multiplicity_exemplar": exemplar,
        "excluded_gate_fail": excluded,
    }
    out = json.dumps(roster, indent=2)
    if args.dry_run:
        print(out)
        return 0
    ROSTER.write_text(out + "\n")
    print(
        f"[relock] wrote {ROSTER}: {len(tier_a)} tier A, {len(tier_b)} tier B, "
        f"exemplar={'yes' if exemplar else 'no'}, {len(excluded)} gate-FAIL excluded; "
        f"fits copied to {FITS_DIR}"
    )
    return 0


if __name__ == "__main__":
    main()
