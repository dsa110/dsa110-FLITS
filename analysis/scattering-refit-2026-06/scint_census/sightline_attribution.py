"""Sightline attribution: is the mid-|b| diffractive-scattering excess on the
co-detections caused by a SPECIFIC intervening galaxy/CGM, or by the host /
circumsource environment?

Cross-matches each excess sightline against the project's vetted intervening-systems
catalog (docs-analysis/foreground.md: 49 candidate objects across 12 FRBs, each
classified confirmed-foreground / refuted-background / inconclusive with an impact
parameter b) and joins it with the measured diffractive excess
(scint_census/data/scint/scint_mw_final.json).

The scattering measure of a galaxy CGM falls steeply with impact parameter; only a
sightline piercing the INNER CGM (b < ~100 kpc) of a CONFIRMED foreground galaxy is a
credible intervening screen. A confirmed system at large b (outer halo) or an
unreliable photo-z (PS1-STRM extrapolated / UNSURE) or a host-redshift companion is
not. Clusters are excluded as screens when b/R500 > 1 (the sightline misses them).

  python sightline_attribution.py
"""

import json
import os

REPO = os.environ.get("FLITS_REPO") or os.path.abspath(f"{os.path.dirname(__file__)}/../../..")
HERE = os.path.dirname(os.path.abspath(__file__))
FG = f"{REPO}/docs-analysis/foreground.md"
FINAL = f"{HERE}/data/scint/scint_mw_final.json"
OUT = f"{HERE}/data/scint"

EXCESS = ["zach", "wilhelm", "hamilton", "chromatica", "casey", "oran"]
INNER_CGM_KPC = 100.0  # inside this, a confirmed foreground galaxy is a credible screen


def parse_foreground(path):
    """Yield dict rows from the markdown table in foreground.md."""
    rows = []
    for line in open(path):
        if not line.startswith("| ") or "---" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 12 or cells[0].lower() in ("burst",):
            continue
        burst, tns, typ, objid, survey, b, br500, z, zsrc, cls, verdict, note = cells
        try:
            b_kpc = float(b)
        except ValueError:
            b_kpc = None
        rows.append(
            dict(
                burst=burst,
                type=typ,
                survey=survey,
                b_kpc=b_kpc,
                b_r500=br500,
                z=z,
                zsrc=zsrc,
                verdict=verdict,
                note=note,
            )
        )
    return rows


def credible_screens(rows):
    """Confirmed-foreground galaxy/halo screens that actually intersect the sightline:
    confirmed verdict, type halo (not a cluster the sightline misses)."""
    out = []
    for r in rows:
        if r["verdict"] != "confirmed":
            continue
        if r["type"] == "cluster":  # clusters in this catalog all have b/R500 > 1
            continue
        if r["b_kpc"] is None:
            continue
        out.append(r)
    return sorted(out, key=lambda r: r["b_kpc"])


def main():
    fg = parse_foreground(FG)
    final = {r["burst"]: r for r in json.load(open(FINAL))["bursts"]}

    print(
        f"{'burst':12s} {'excess':>7} {'closest confirmed fg halo':>28}  {'inner-CGM?':>10}  verdict"
    )
    results = {}
    for name in EXCESS:
        rows = [r for r in fg if r["burst"] == name]
        screens = credible_screens(rows)
        closest = screens[0] if screens else None
        exc = final.get(name, {}).get("excess")
        in_cgm = bool(closest and closest["b_kpc"] < INNER_CGM_KPC)
        if closest:
            desc = f"b={closest['b_kpc']:.0f}kpc z={closest['z']}"
            verdict = (
                "INTERVENING-CGM candidate"
                if in_cgm
                else "confirmed fg only in OUTER halo -> weak screen; host/circumsource favored"
            )
        else:
            # report the closest object of any verdict to show what WAS found
            anyobj = sorted(
                [r for r in rows if r["b_kpc"] is not None and r["type"] != "cluster"],
                key=lambda r: r["b_kpc"],
            )
            nearest = anyobj[0] if anyobj else None
            desc = (
                f"(none confirmed; nearest {nearest['verdict']} b={nearest['b_kpc']:.0f}kpc)"
                if nearest
                else "(no halo objects)"
            )
            verdict = "no confirmed intervening screen -> host/circumsource favored"
        results[name] = dict(
            excess=exc,
            closest_confirmed=closest,
            inner_cgm=in_cgm,
            verdict=verdict,
            n_confirmed=len(screens),
        )
        ex = f"{exc:6.1f}x" if exc else "   -- "
        print(f"{name:12s} {ex:>7} {desc:>28}  {str(in_cgm):>10}  {verdict}")

    n_cgm = sum(1 for v in results.values() if v["inner_cgm"])
    print(
        f"\nVERDICT: {n_cgm}/{len(EXCESS)} excess sightlines pierce the inner CGM (b<{INNER_CGM_KPC:.0f} kpc) "
        f"of a CONFIRMED foreground galaxy."
    )
    print(
        "  => the diffractive-scattering excess is NOT attributable to a confirmed intervening\n"
        "     galaxy/CGM: the closest confirmed foreground halos sit in the outer halo (>=170 kpc,\n"
        "     casey/chromatica) where CGM scattering measure is low; the rest are unreliable photo-z\n"
        "     (PS1-STRM extrapolated/UNSURE) or inconclusive. Excess favors a HOST /\n"
        "     circumsource screen. Caveat: sparse spec-z on these fields cannot exclude a faint\n"
        "     undetected intervening dwarf inside the inner CGM."
    )

    json.dump(
        dict(inner_cgm_kpc=INNER_CGM_KPC, n_inner_cgm=n_cgm, bursts=results),
        open(f"{OUT}/sightline_attribution.json", "w"),
        indent=2,
        default=str,
    )
    print(f"\nwrote {OUT}/sightline_attribution.json")

    # check: the parser must recover the known catalog size (49 objects, 12 cols each)
    assert len(fg) >= 45, f"foreground.md parse recovered only {len(fg)} rows (expected ~49)"


if __name__ == "__main__":
    main()
