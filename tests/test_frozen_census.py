import hashlib
import pathlib
import re

ROOT = pathlib.Path(__file__).parents[1]
DOC = ROOT / "docs/rse/specs/reproducibility-foreground-galaxies.md"
FROZEN = ROOT / "galaxies/foreground/data/frozen_census"


def pinned():
    pat = re.compile(
        r"^([a-f0-9]{64})\s+scratch/codetection/(\S+\.csv)\s*$",
        re.MULTILINE,
    )
    out = dict()
    for sha, name in pat.findall(DOC.read_text()):
        out[name] = sha
    assert len(out) == 6, out
    return out


def test_frozen_census_files_match_pinned_hashes():
    for name, sha in pinned().items():
        p = FROZEN / name
        assert p.exists(), f"missing {p}"
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        assert got == sha, f"{name}: {got} != pinned {sha}"
