"""Self-check for the ADR-0003 fail-closed PBF guard in _s2verdict.py.

Run: `python test_s2verdict.py` (no pytest/fixtures needed). Encodes the actual
zach hazard — the mixed-PBF and all-exp fixed-s2 grids give OPPOSITE cross-N
verdicts, so the family the tool reads decides the science.
"""

from _s2verdict import ALLEXP, MIXED_LEGACY, adjudicate, load_records, parse_tag, pbf_family


def test_parse_tag_tolerates_pbf_suffix():
    # the bug: old split('_s2-'); int() crashed on the _pbf-* suffix the all-exp grids carry
    assert parse_tag("C2D3_s2-1_pbf-exp-exp") == ("C2D3", 1)
    assert parse_tag("C2D3_s2-100") == ("C2D3", 100)
    assert parse_tag("sharedzeta") == (None, None)


def test_pbf_family_legacy_vs_allexp():
    assert pbf_family({"pbf_C": "exp", "pbf_D": "exp"}) == ALLEXP
    assert pbf_family({}) == MIXED_LEGACY  # legacy files omit the keys -> mixed


def test_families_give_opposite_verdicts():
    # zach C2D3-vs-C2D2 ΔlnZ across s2=1/10/100 (from ALLEXP_PBF_RUN.md & local mixed JSONs)
    def verdict(deltas):
        grids = {"C2D2": {1: 0, 10: 0, 100: 0}, "C2D3": dict(zip((1, 10, 100), deltas))}
        return next(v for a, b, d, v in adjudicate(grids) if b == "C2D3")

    assert "REAL" in verdict([2216.5, 1112.9, 315.9])  # mixed-PBF: spurious "real"
    assert "NOT robust" in verdict([1443.4, -758.7, -0.4])  # all-exp: correctly rejected


def test_fail_closed_separates_pbf_families():
    # A joint_ladder dir can hold BOTH the canonical all-exp fixed-s2 grid
    # (pbf-exp-exp, pbf_C=pbf_D="exp") and the legacy mixed _s2- grid (no pbf_* keys);
    # the two families must never co-mingle in one adjudication. Build a synthetic dir
    # so this is self-contained — the real all-exp grid is HPCC-pulled/untracked and the
    # legacy files are slated for deletion, so neither is present on a clean checkout.
    import json
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "zach_joint_fit_C2D3_s2-1_pbf-exp-exp.json"), "w") as fh:
            json.dump({"pbf_C": "exp", "pbf_D": "exp", "log_evidence": -1.0}, fh)
        with open(os.path.join(d, "zach_joint_fit_C2D3_s2-1.json"), "w") as fh:
            json.dump({"log_evidence": -2.0}, fh)  # omits pbf_* -> MIXED_LEGACY
        allexp, ex_allexp = load_records(d, ALLEXP)
        mixed, _ = load_records(d, MIXED_LEGACY)
        # all-exp pass: includes the all-exp fixed-s2 grid, excludes every legacy mixed _s2-
        assert any(t.endswith("pbf-exp-exp") and "_s2-" in t for (_, t) in allexp), (
            "all-exp pass should include the all-exp fixed-s2 grid"
        )
        assert not any("_s2-" in t and not t.endswith("pbf-exp-exp") for (_, t) in allexp), (
            "legacy mixed _s2- files must not leak into the all-exp pass"
        )
        assert ex_allexp > 0, "all-exp pass should exclude the legacy mixed JSONs"
        # mixed pass: includes the legacy _s2- grid, excludes the all-exp files
        assert any("_s2-" in t and not t.endswith("pbf-exp-exp") for (_, t) in mixed), (
            "mixed pass should include the legacy mixed _s2- grid"
        )
        assert not any(t.endswith("pbf-exp-exp") for (_, t) in mixed), (
            "all-exp files must not leak into the mixed pass"
        )


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all self-checks passed")
