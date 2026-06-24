"""Fixed-s2 cross-N Bayes factor: is the extra component real?

Profiled-s2 lnZ is an empirical-Bayes (profile) Z and is NOT comparable across
component count N. Only the fixed-s2 ladder gives a clean cross-N Bayes factor.
A component is statistically real only if ΔlnZ(N+1 vs N) is consistently
positive (>~5) across ALL s2 values; a sign flip with the prior scale means the
extra component is prior-driven, not data-driven.

ADR-0003: lnZ is also NOT comparable across PBF families. A mixed-PBF fit
(pbf_C=powerlaw, pbf_D=exp — the legacy default, written before pbf_C/pbf_D were
recorded, so those keys are absent) and an all-exponential fit (pbf_C=pbf_D="exp")
are physically incoherent to compare. Mixing them flips the zach C2D3 verdict from
the canonical "rejected" to a spurious "REAL". So this adjudicator is fail-closed:
by default it uses ONLY all-exp records and refuses to render a verdict from the
legacy mixed-PBF grid. Set FLITS_S2_PBF=mixed to inspect the (non-canonical) legacy
grid explicitly.
"""

import glob
import json
import os
import re
from collections import defaultdict

ALLEXP = ("exp", "exp")
MIXED_LEGACY = ("powerlaw", "exp")  # run_joint_fit's pre-all-exp default; written w/o pbf_* keys
S2VALS = (1, 10, 100)


def pbf_family(d):
    """PBF family (pbf_C, pbf_D) of a fit record. Legacy mixed-PBF files predate the
    pbf_C/pbf_D fields and omit them; that default was powerlaw (CHIME) + exp (DSA)."""
    pc, pd = d.get("pbf_C"), d.get("pbf_D")
    if pc is None and pd is None:
        return MIXED_LEGACY
    return (pc, pd)


def parse_tag(tag):
    """(base 'CxDy', s2) from a fixed-s2 tag, tolerant of a trailing _pbf-... suffix
    (e.g. 'C2D3_s2-1_pbf-exp-exp'). Returns (None, None) if not a fixed-s2 tag.

    The old `tag.split('_s2-'); int(s2)` crashed on the _pbf-* suffix the all-exp
    grids carry — exactly the grids this tool must read once they land.
    """
    m = re.search(r"(C\d+D\d+).*?_s2-(\d+)", tag)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def _cd(base):
    m = re.match(r"C(\d+)D(\d+)", base)
    return (int(m.group(1)), int(m.group(2)))


def adjudicate(grids, s2vals=S2VALS):
    """grids: {base 'CxDy': {s2: lnZ}}, all from ONE PBF family. Yields
    (a, b, deltas, verdict) for every config pair differing by exactly one component."""
    bases = sorted(grids)
    for a in bases:
        for b in bases:
            ca, da = _cd(a)
            cb, db = _cd(b)
            if (cb == ca + 1 and db == da) or (db == da + 1 and cb == ca):
                deltas = [
                    grids[b][s] - grids[a][s] for s in s2vals if s in grids[a] and s in grids[b]
                ]
                if not deltas:
                    continue
                signs = {"+" if x > 0 else "-" for x in deltas}
                verdict = (
                    "REAL (consistent +)"
                    if all(x > 5 for x in deltas)
                    else ("NOT robust (sign flips)" if len(signs) > 1 else "weak/consistent-neg")
                )
                yield a, b, deltas, verdict


def load_records(directory, allowed_pbf):
    """{(burst, tag): lnZ} for fits in allowed_pbf only; count of off-family files dropped."""
    recs, excluded = {}, 0
    for f in glob.glob(os.path.join(directory, "*_joint_fit*.json")):
        d = json.load(open(f))
        m = re.match(r"(.+?)_joint_fit_(.*)\.json", os.path.basename(f))
        if not m:
            continue
        if pbf_family(d) != allowed_pbf:
            excluded += 1
            continue
        recs[(m.group(1), m.group(2))] = d.get("log_evidence")
    return recs, excluded


def report(directory, allowed_pbf):
    label = (
        "all-exp (canonical)"
        if allowed_pbf == ALLEXP
        else f"{allowed_pbf} (NON-CANONICAL — ADR-0003)"
    )
    recs, excluded = load_records(directory, allowed_pbf)
    print(f"# PBF family: {allowed_pbf} [{label}]; excluded {excluded} off-family JSON(s)")
    any_grid = False
    for b in sorted({bb for (bb, _) in recs}):
        grids = defaultdict(dict)
        for (bb, t), lnz in recs.items():
            if bb != b:
                continue
            base, s2 = parse_tag(t)
            if base is not None:
                grids[base][s2] = lnz
        if not grids:
            continue
        any_grid = True
        print(f"\n===== {b} : fixed-s2 cross-N test ({label}) =====")
        for base in sorted(grids):
            line = f"  {base:6}"
            for s in S2VALS:
                v = grids[base].get(s)
                line += f"  s2={s:<3} lnZ={v:.1f}" if v is not None else f"  s2={s:<3} --      "
            print(line)
        for a, bb, deltas, verdict in adjudicate(grids):
            ds = " / ".join(f"{x:+.1f}" for x in deltas)
            print(f"    Δ({bb} vs {a}) across s2: {ds}   -> {verdict}")
    if not any_grid:
        print(
            "\n!! No fixed-s2 grid in the canonical all-exp PBF family. Refusing to adjudicate"
            "\n   from the legacy mixed-PBF grid (ADR-0003 — the families are not comparable)."
            "\n   Pull the all-exp fixed-s2 grid, or set FLITS_S2_PBF=mixed to inspect the"
            "\n   non-canonical legacy grid explicitly."
        )


if __name__ == "__main__":
    family = MIXED_LEGACY if os.environ.get("FLITS_S2_PBF") == "mixed" else ALLEXP
    report(os.path.dirname(__file__), family)
